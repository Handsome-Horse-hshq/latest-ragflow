"""可靠性折扣的敏感性分析。

为什么需要这个模块
------------------
``no_reliability`` 消融在 CLIMATE-FEVER v1 上是**恒等变换**：数据集里 2000 个
``reliability`` 全是 1.0，"去掉折扣"等于什么都没做，Δ 必然为 0。那个 0 说明的
是消融没跑起来，**不能**读成「可靠性折扣没有作用」。

可靠性折扣是 D-S 链路的核心机制之一，不能就这么留成一片空白。本模块用两组
敏感性分析把它实际跑起来：

S1 —— 均匀可靠性扫描（不含任何 oracle）
    把所有文档可靠性统一设为 r，扫 r。这检验的是**机制本身的性质**：
    折扣只把质量从确定焦元移向 Theta，因此 r 下降时融合后的 ``m_theta``
    应当单调上升、``K_doc`` 单调下降。它不依赖任何标注，属于方法性质，
    可以直接报告。

S2 —— 标注一致度作为可靠性（**oracle 上界**）
    CLIMATE-FEVER 为每段证据记录了标注者投票与其熵值。一致度高的证据给高
    可靠性::

        r = 1 - H / ln(3)

    这用到了人工标注，因此**只能作为上界报告**，回答「若存在一个完美的
    可靠性估计器，折扣最多能带来多少」，**不能**当作模型性能。

两组都是敏感性分析，不是主结果。输出里的 ``is_oracle`` 字段把 S2 标了出来。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.metrics.classification import ClassificationReport, classification_report
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import ContextChunk, RAGSample, RelationPrediction
from rag_ds.tuning.threshold_search import predicted_label

__all__ = [
    "MAX_VOTE_ENTROPY",
    "ReliabilitySensitivityPoint",
    "apply_document_reliability",
    "oracle_reliability_from_provenance",
    "run_reliability_sensitivity",
    "uniform_reliability_sweep",
]

#: 三个标签上的最大投票熵，用作归一化常数。
MAX_VOTE_ENTROPY: float = math.log(3.0)


class ReliabilitySensitivityPoint(BaseModel):
    """一个可靠性设定下的结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 该设定的名称，例如 ``uniform_0.60`` 或 ``oracle_vote_agreement``。
    setting: str
    #: 实际施加的平均文档可靠性。
    mean_reliability: float = Field(ge=0.0, le=1.0)
    report: ClassificationReport
    #: 融合后 m_theta 的均值；完全冲突的 claim 不计入。
    mean_m_theta: float = Field(ge=0.0, le=1.0)
    #: 融合后 K_doc 的均值。
    mean_k_doc: float = Field(ge=0.0, le=1.0)
    #: 相对 ``r = 1.0``（即原始输入）的 Macro-F1 变化量。
    macro_f1_delta: float = Field(ge=-1.0, le=1.0)
    #: 该设定是否用到了人工标注；为 ``True`` 时只能作为上界报告。
    is_oracle: bool = False


def apply_document_reliability(
    samples: Sequence[RAGSample], reliability: Mapping[str, float] | float
) -> list[RAGSample]:
    """重建样本，替换每篇文档的 ``reliability``。

    Args:
        samples: 原始样本。
        reliability: 统一取值，或 ``{doc_id: reliability}`` 映射。映射里缺失的
            ``doc_id`` 保留原值 —— 不会被悄悄当成 1.0。

    Returns:
        新的样本列表；原对象不被修改（``RAGSample`` 是 frozen 的）。
    """
    def _value(chunk: ContextChunk) -> float:
        if isinstance(reliability, Mapping):
            return reliability.get(chunk.doc_id, chunk.reliability)
        return float(reliability)

    return [
        RAGSample(
            sample_id=s.sample_id,
            question=s.question,
            answer=s.answer,
            reference_answer=s.reference_answer,
            claims=list(s.claims),
            contexts=[
                ContextChunk(
                    doc_id=c.doc_id,
                    text=c.text,
                    retrieval_score=c.retrieval_score,
                    reliability=_value(c),
                )
                for c in s.contexts
            ],
            gold_state=s.gold_state,
        )
        for s in samples
    ]


def oracle_reliability_from_provenance(
    provenance: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """由标注投票熵算出每篇文档的可靠性（**oracle 信号**）。

    ``r = 1 - H / ln(3)``，其中 H 是 CLIMATE-FEVER 记录的投票熵。熵为 0
    （标注者完全一致）时 r = 1；熵越高 r 越低，截断到 [0, 1]。

    .. warning::
        这用到了人工标注，只能作为**上界**报告，不能当作模型性能。

    Args:
        provenance: ``*_provenance.jsonl`` 的记录。

    Returns:
        ``{doc_id: reliability}``。

    Raises:
        ValueError: 记录缺少 ``contexts``，或某条缺少 ``doc_id`` / ``entropy``。
    """
    out: dict[str, float] = {}
    for record in provenance:
        contexts = record.get("contexts")
        if not isinstance(contexts, list):
            raise ValueError(
                f"谱系记录 {record.get('sample_id')!r} 缺少 contexts 数组"
            )
        for chunk in contexts:
            doc_id = chunk.get("doc_id")
            entropy = chunk.get("entropy")
            if doc_id is None or entropy is None:
                raise ValueError(
                    f"谱系记录 {record.get('sample_id')!r} 的某条 context "
                    "缺少 doc_id 或 entropy"
                )
            value = 1.0 - float(entropy) / MAX_VOTE_ENTROPY
            out[str(doc_id)] = min(max(value, 0.0), 1.0)
    return out


def _evaluate(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    thresholds: DiagnosticThresholds,
    setting: str,
    is_oracle: bool,
) -> tuple[ReliabilitySensitivityPoint, float]:
    """跑一次完整链路并汇总。返回 (结果点, macro_f1)。"""
    results = run_pipeline(samples, predictions, thresholds)
    truth = [r.gold_state.value for r in results]  # type: ignore[union-attr]
    predicted = [predicted_label(r.diagnostic) for r in results]
    report = classification_report(truth, predicted, method=setting)

    thetas = [
        r.diagnostic.m_theta for r in results if r.diagnostic.m_theta is not None
    ]
    k_docs = [r.diagnostic.k_doc for r in results]
    reliabilities = [c.reliability for s in samples for c in s.contexts]

    point = ReliabilitySensitivityPoint(
        setting=setting,
        mean_reliability=sum(reliabilities) / len(reliabilities),
        report=report,
        mean_m_theta=sum(thetas) / len(thetas) if thetas else 0.0,
        mean_k_doc=sum(k_docs) / len(k_docs) if k_docs else 0.0,
        macro_f1_delta=0.0,
        is_oracle=is_oracle,
    )
    return point, report.macro_f1


def uniform_reliability_sweep(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    thresholds: DiagnosticThresholds,
    levels: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
) -> list[ReliabilitySensitivityPoint]:
    """S1：把所有文档可靠性统一设为 r，逐点评估。

    Args:
        samples: 样本，每条须带 ``gold_state``。
        predictions: 关系预测。
        thresholds: 门控阈值（应为验证集上选出的那一组）。
        levels: 要扫的 r，须含 1.0 作为基准。

    Returns:
        每个 r 一条结果，顺序与 ``levels`` 一致。

    Raises:
        ValueError: ``levels`` 为空、取值越界，或不含基准 1.0。
    """
    if not levels:
        raise ValueError("levels 不能为空")
    if any(not 0.0 <= level <= 1.0 for level in levels):
        raise ValueError(f"levels 的取值必须位于 [0, 1]：{tuple(levels)}")
    if 1.0 not in tuple(levels):
        raise ValueError("levels 必须包含 1.0 作为 Δ 的基准")

    points: list[ReliabilitySensitivityPoint] = []
    baseline: float | None = None
    for level in levels:
        adjusted = apply_document_reliability(samples, level)
        point, macro_f1 = _evaluate(
            adjusted, predictions, thresholds, f"uniform_{level:.2f}", False
        )
        if level == 1.0:
            baseline = macro_f1
        points.append(point)

    assert baseline is not None
    return [
        p.model_copy(update={"macro_f1_delta": p.report.macro_f1 - baseline})
        for p in points
    ]


def run_reliability_sensitivity(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    thresholds: DiagnosticThresholds,
    *,
    levels: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    oracle_reliability: Mapping[str, float] | None = None,
) -> list[ReliabilitySensitivityPoint]:
    """跑完 S1，并在给出 oracle 可靠性时追加 S2。

    Args:
        samples: 样本。
        predictions: 关系预测。
        thresholds: 门控阈值。
        levels: S1 扫描的可靠性取值。
        oracle_reliability: ``{doc_id: reliability}``；``None`` 表示跳过 S2。

    Returns:
        S1 的全部结果点，后面跟着 S2（若有），S2 的 ``is_oracle`` 为 ``True``。
    """
    points = uniform_reliability_sweep(samples, predictions, thresholds, levels)
    if oracle_reliability is None:
        return points

    baseline = next(p for p in points if p.setting == "uniform_1.00").report.macro_f1
    adjusted = apply_document_reliability(samples, oracle_reliability)
    point, macro_f1 = _evaluate(
        adjusted, predictions, thresholds, "oracle_vote_agreement", True
    )
    points.append(point.model_copy(update={"macro_f1_delta": macro_f1 - baseline}))
    return points
