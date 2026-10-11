"""检测实验的公平对照：拿 D-S 的 m_theta / K_doc 与不用 D-S 的简单统计量比。

为什么要重做
------------
原实验 2/3 里 baseline 的冲突分数是 ``1 - |score_support - score_refute|``。
当所有文档都说「不确定」时，两个平均分都接近 0，这个分数反而接近 1 ——
它把「证据不足」当成了「冲突」，在 v2 上 AUROC 只有 0.46，低于随机。拿
``K_doc`` 去赢这样的对手说明不了什么。

这里把几个同样简单、但定义合理的统计量**事先全部列出、全部报告**，不从中
挑选。所有分数都与阈值无关，因此直接在全体 claim 上算，不需要交叉验证。

可靠性的处理与 D-S 一致：平均类统计量按 文档可靠性 × 评估器可靠性 加权；
取最大/最小/连乘的统计量先按同一权重给概率打折（支持、反驳乘以权重，
不确定取 ``1 - 权重 × (1 - p_unknown)``），与 D-S 的可靠性折扣对应。

冲突（正类 ``conflicting``，参照 ``K_doc``）：

==========================  ==================================================
``one_minus_abs_margin``    ``1 - |mean_s - mean_r|``，原实验用的代理，保留作对照
``min_mean``                ``min(mean_s, mean_r)``，conflict_aware 规则所用的量
``mean_product``            ``mean_s × mean_r``
``max_product``             最强支持 × 最强反驳（Statistic Gate 的一个纵轴）
``min_max``                 ``min(最强支持, 最强反驳)``
``sr_std``                  各条关系 ``s - r`` 的加权标准差（Statistic Gate 的另一个纵轴）
==========================  ==================================================

证据不足（正类 ``insufficient``，参照 ``m_theta``）：

==========================  ==================================================
``mean_unknown``            加权平均 ``p_unknown``，即 baseline 的 ``score_unknown``
``min_unknown``             各条关系里最小的 ``p_unknown``：所有文档都说不清才高
``unknown_product``         ``p_unknown`` 连乘：不做归一化的「全体都不知道」
==========================  ==================================================
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.baselines._shared import contexts_by_doc_id
from rag_ds.baselines.runner import _claim_predictions
from rag_ds.baselines.stat_gate import ConflictStatistic, stat_gate_scores
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.integrity import check_samples, index_predictions
from rag_ds.metrics.detection import (
    AurocComparison,
    DetectionReport,
    detection_report,
    paired_auroc_bootstrap,
)
from rag_ds.metrics.significance import DEFAULT_RESAMPLES, DEFAULT_SEED
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import EvidenceState, NonEmptyStr, RAGSample, RelationPrediction

__all__ = [
    "CONFLICT_PROXIES",
    "INSUFFICIENCY_PROXIES",
    "ClaimDetectionScores",
    "DetectionComparison",
    "compare_detection",
    "detection_scores",
]

CONFLICT_PROXIES: tuple[str, ...] = (
    "one_minus_abs_margin",
    "min_mean",
    "mean_product",
    "max_product",
    "min_max",
    "sr_std",
)
INSUFFICIENCY_PROXIES: tuple[str, ...] = (
    "mean_unknown",
    "min_unknown",
    "unknown_product",
)


class ClaimDetectionScores(BaseModel):
    """一条 claim 的 D-S 检测分数与全部对照统计量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    gold_label: NonEmptyStr
    #: 完全冲突时质量未定义，与 :mod:`rag_ds.experiments.comparison` 一致记为 0。
    m_theta: float = Field(ge=0.0, le=1.0)
    k_doc: float = Field(ge=0.0, le=1.0)
    conflict: dict[str, float]
    insufficiency: dict[str, float]


def detection_scores(
    samples: Sequence[RAGSample], predictions: Sequence[RelationPrediction]
) -> list[ClaimDetectionScores]:
    """对每条 claim 算出 D-S 的两个检测分数与全部对照统计量。

    Args:
        samples: 样本，每条须带 ``gold_state``。
        predictions: 关系预测。

    Returns:
        每条 claim 一条，按「样本 → claim」顺序。

    Raises:
        ValueError: 存在缺少 ``gold_state`` 的样本。
        PipelineError: 输入不满足完整性要求。
    """
    missing = [s.sample_id for s in samples if s.gold_state is None]
    if missing:
        raise ValueError(
            f"以下样本缺少 gold_state，无法参与检测实验：{sorted(missing)[:5]}"
        )

    # m_theta 与 K_doc 在门控之前就已确定，阈值取任何值都不影响它们。
    ds = {
        (r.sample_id, r.claim_id): r.diagnostic
        for r in run_pipeline(samples, predictions, DiagnosticThresholds())
    }
    by_statistic = {
        statistic: {
            (s.sample_id, s.claim_id): s
            for s in stat_gate_scores(samples, predictions, statistic)
        }
        for statistic in ConflictStatistic
    }

    check_samples(list(samples))
    table = index_predictions(list(samples), list(predictions))
    rows: list[ClaimDetectionScores] = []
    for sample in samples:
        contexts = contexts_by_doc_id(sample)
        for claim in sample.claims:
            key = (sample.sample_id, claim.claim_id)
            collected = _claim_predictions(table, sample, claim.claim_id)
            weights = [
                contexts[p.doc_id].reliability * p.evaluator_reliability
                for p in collected
            ]
            support = [w * p.p_support for w, p in zip(weights, collected)]
            refute = [w * p.p_refute for w, p in zip(weights, collected)]
            unknown = [1.0 - w * (1.0 - p.p_unknown) for w, p in zip(weights, collected)]

            std = by_statistic[ConflictStatistic.SUPPORT_REFUTE_STD][key]
            product = by_statistic[ConflictStatistic.MAX_PRODUCT][key]
            mean_s, mean_r = std.score_support, std.score_refute
            diagnostic = ds[key]
            rows.append(
                ClaimDetectionScores(
                    sample_id=sample.sample_id,
                    claim_id=claim.claim_id,
                    gold_label=sample.gold_state.value,  # type: ignore[union-attr]
                    m_theta=diagnostic.m_theta or 0.0,
                    k_doc=diagnostic.k_doc,
                    conflict={
                        "one_minus_abs_margin": 1.0 - abs(mean_s - mean_r),
                        "min_mean": min(mean_s, mean_r),
                        "mean_product": mean_s * mean_r,
                        "max_product": product.conflict,
                        "min_max": min(max(support, default=0.0), max(refute, default=0.0)),
                        "sr_std": std.conflict,
                    },
                    insufficiency={
                        "mean_unknown": std.score_unknown,
                        "min_unknown": min(unknown, default=1.0),
                        "unknown_product": math.prod(unknown),
                    },
                )
            )
    return rows


class DetectionComparison(BaseModel):
    """一个检测任务上参照分数与全部对照的结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_label: str
    reference: str
    #: 参照在前，然后按对照统计量的声明顺序。
    reports: tuple[DetectionReport, ...]
    comparisons: tuple[AurocComparison, ...]


def compare_detection(
    rows: Sequence[ClaimDetectionScores],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, DetectionComparison]:
    """冲突与证据不足两个检测任务，各自拿 D-S 分数与全部对照做配对比较。

    Returns:
        ``{"conflict": ..., "insufficiency": ...}``。
    """
    tasks = {
        "conflict": (
            EvidenceState.CONFLICTING.value,
            "k_doc",
            [r.k_doc for r in rows],
            {name: [r.conflict[name] for r in rows] for name in CONFLICT_PROXIES},
        ),
        "insufficiency": (
            EvidenceState.INSUFFICIENT.value,
            "m_theta",
            [r.m_theta for r in rows],
            {
                name: [r.insufficiency[name] for r in rows]
                for name in INSUFFICIENCY_PROXIES
            },
        ),
    }
    results: dict[str, DetectionComparison] = {}
    for task, (positive, reference, reference_scores, rivals) in tasks.items():
        is_positive = [r.gold_label == positive for r in rows]
        reports = [
            detection_report(scores, is_positive, name, positive)
            for name, scores in [(reference, reference_scores), *rivals.items()]
        ]
        results[task] = DetectionComparison(
            positive_label=positive,
            reference=reference,
            reports=tuple(reports),
            comparisons=tuple(
                paired_auroc_bootstrap(
                    is_positive,
                    reference_scores,
                    rivals,
                    reference,
                    positive,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            ),
        )
    return results
