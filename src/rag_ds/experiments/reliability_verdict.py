"""文档级可靠性实验：来源可信度确有差异时，可靠性折扣有没有用。

此前所有数据集的文档可靠性都是 1.0，D-S 的文档级折扣是空转的。这里在一个
**只有两类倾向标签**（``supported`` / ``refuted``）、文档来源可信度各不相同的
数据集上，回答两个问题：

1. **可靠性有没有用**：同一个方法，把文档可靠性从「全为 1」换成「按来源可信度
   赋值」，倾向判断是否变好；
2. **D-S 的折扣是否优于简单加权**：同一份可靠性，D-S（折扣后做 Dempster 融合）
   与加权平均（把可靠性当权重）谁更好。

不调任何阈值
------------
倾向只看支持与反驳哪边大：D-S 用融合后的 ``m_support - m_refute``，加权平均
用 ``加权平均 p_support - 加权平均 p_refute``，多数票用 ``(支持票 - 反驳票) / 票数``。
没有需要在数据上选的参数，因此不需要交叉验证，直接在全体 claim 上评估。

近乎完全冲突
------------
证据互相矛盾得很厉害时（一篇几乎 100% 支持、另一篇几乎 100% 反驳），Dempster
归一化的分母 ``1 - K`` 逼近 0。``K`` 恰好等于 1 时 pipeline 给出「完全冲突」，
质量未定义；``1 - K`` 只是极小（约 1e-12 到 1e-6）时，归一化后的质量已经没有
几位有效数字，``ds`` 子包会直接报错而不是返回靠不住的数。两种情况这里都记为
``undetermined``（margin 取 0，算判错），并用 ``ds_undefined`` 标出来 ——
这是 Dempster 规则在高冲突下的固有弱点，不能悄悄绕过去。

指标
----
类别可能很不均衡，所以同时报告：两类的 Macro-F1（平局或完全冲突记为
``undetermined``，算判错），以及把 margin 当连续分数时区分 ``supported`` 与
``refuted`` 的 AUROC（与阈值无关，也不受类别比例影响）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.baselines.stat_gate import ConflictStatistic, stat_gate_scores
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.reliability_sensitivity import apply_document_reliability
from rag_ds.integrity import PipelineError
from rag_ds.metrics.classification import UNDETERMINED_LABEL
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import EvidenceState, NonEmptyStr, RAGSample, RelationPrediction

__all__ = [
    "VERDICT_LABELS",
    "VERDICT_METHODS",
    "VerdictScore",
    "reliability_from_credibility",
    "verdict_scores",
]

#: 这类实验里参与 Macro-F1 的两类。
VERDICT_LABELS: tuple[str, ...] = (
    EvidenceState.SUPPORTED.value,
    EvidenceState.REFUTED.value,
)
VERDICT_METHODS: tuple[str, ...] = ("ds", "weighted_average", "majority_vote")

#: 判为平局的 margin 上限，与二维门控的默认平局容差一致。
_TIE_TOLERANCE = 1e-6


class VerdictScore(BaseModel):
    """某个方法对一条 claim 的倾向分数与判定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: NonEmptyStr
    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    gold_label: NonEmptyStr
    #: 支持分减反驳分；正数偏向支持。完全冲突（质量未定义）时记 0。
    margin: float = Field(ge=-1.0, le=1.0)
    predicted_label: NonEmptyStr
    #: 仅 ``ds``：Dempster 融合在这条 claim 上完全冲突或数值上不可靠。
    ds_undefined: bool = False


def _label(margin: float) -> str:
    if margin > _TIE_TOLERANCE:
        return EvidenceState.SUPPORTED.value
    if margin < -_TIE_TOLERANCE:
        return EvidenceState.REFUTED.value
    return UNDETERMINED_LABEL


def reliability_from_credibility(
    credibility_by_doc: Mapping[str, str | None],
    mapping: Mapping[str, float],
    unrated: float,
) -> dict[str, float]:
    """把每篇文档的可信度分级换成可靠性；没有评级的取 ``unrated``。"""
    return {
        doc_id: (mapping[credibility] if credibility is not None else unrated)
        for doc_id, credibility in credibility_by_doc.items()
    }


def verdict_scores(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    reliability: Mapping[str, float] | float,
) -> list[VerdictScore]:
    """在给定的文档可靠性下，算出三个方法对每条 claim 的倾向。

    Args:
        samples: 样本，``gold_state`` 须为 ``supported`` 或 ``refuted``。
        predictions: 关系预测（单个评估器）。
        reliability: 统一取值，或 ``{doc_id: reliability}``。

    Returns:
        每个 (方法, claim) 一条，方法顺序见 :data:`VERDICT_METHODS`。

    Raises:
        ValueError: 样本的金标准不是两类倾向之一。
        PipelineError: 输入不满足完整性要求。
    """
    bad = [
        s.sample_id
        for s in samples
        if s.gold_state is None or s.gold_state.value not in VERDICT_LABELS
    ]
    if bad:
        raise ValueError(f"以下样本的金标准不是 supported / refuted：{sorted(bad)[:5]}")

    weighted = apply_document_reliability(samples, reliability)
    gold = {
        (s.sample_id, claim.claim_id): s.gold_state.value  # type: ignore[union-attr]
        for s in weighted
        for claim in s.claims
    }
    scores: list[VerdictScore] = []

    def _add(
        method: str, key: tuple[str, str], margin: float, ds_undefined: bool = False
    ) -> None:
        scores.append(
            VerdictScore(
                method=method,
                sample_id=key[0],
                claim_id=key[1],
                gold_label=gold[key],
                margin=margin,
                predicted_label=_label(margin),
                ds_undefined=ds_undefined,
            )
        )

    # D-S：文档可靠性折扣 → Dempster 融合。门控阈值不影响 m_support / m_refute。
    # 逐条样本跑：某一条近乎完全冲突而数值报错时，只影响它自己。
    by_sample: dict[str, list[RelationPrediction]] = {}
    for prediction in predictions:
        by_sample.setdefault(prediction.sample_id, []).append(prediction)
    for sample in weighted:
        try:
            results = run_pipeline(
                [sample], by_sample.get(sample.sample_id, []), DiagnosticThresholds()
            )
        except PipelineError:
            raise
        except ValueError:
            for claim in sample.claims:
                _add("ds", (sample.sample_id, claim.claim_id), 0.0, ds_undefined=True)
            continue
        for result in results:
            margin = result.diagnostic.support_refute_margin
            _add(
                "ds",
                (result.sample_id, result.claim_id),
                0.0 if margin is None else margin,
                ds_undefined=margin is None,
            )

    # 加权平均：同一份可靠性作权重。平均分与冲突统计量的选择无关。
    for item in stat_gate_scores(weighted, predictions, ConflictStatistic.MAX_PRODUCT):
        _add(
            "weighted_average",
            (item.sample_id, item.claim_id),
            item.score_support - item.score_refute,
        )

    # 多数票：一篇文档一票，不看可靠性。
    votes: dict[tuple[str, str], list[int]] = {key: [] for key in gold}
    for prediction in predictions:
        top = max(prediction.p_support, prediction.p_refute, prediction.p_unknown)
        if prediction.p_support == top and prediction.p_refute != top:
            vote = 1
        elif prediction.p_refute == top and prediction.p_support != top:
            vote = -1
        else:
            vote = 0  # 不确定最高，或支持与反驳并列
        votes[(prediction.sample_id, prediction.claim_id)].append(vote)
    for key, ballots in votes.items():
        _add("majority_vote", key, sum(ballots) / len(ballots) if ballots else 0.0)

    order = {method: index for index, method in enumerate(VERDICT_METHODS)}
    return sorted(scores, key=lambda s: (order[s.method], s.sample_id, s.claim_id))
