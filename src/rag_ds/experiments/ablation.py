"""消融实验：逐项去掉方法组件，观察指标变化。

每个变体去掉什么，都在 :class:`AblationVariant` 的文档里写死，避免论文里
出现「消融了什么」说不清的情况。
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.metrics.classification import ClassificationReport, classification_report
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import ContextChunk, RAGSample, RelationPrediction
from rag_ds.tuning.threshold_search import predicted_label

__all__ = [
    "CLASSIFICATION_ABLATION_VARIANTS",
    "AblationVariant",
    "AblationResult",
    "run_ablation",
]


class AblationVariant(str, Enum):
    """消融变体。"""

    #: 完整方法，作为对照基准。
    FULL = "full"
    #: 去掉可靠性折扣：所有文档与评估器可靠性强制为 1.0。
    NO_RELIABILITY = "no_reliability"
    #: 去掉 m_theta 门控：theta 阈值抬到 1.0，几乎不再判「证据不足」。
    NO_THETA_GATE = "no_theta_gate"
    #: 去掉 K_doc 门控：文档冲突阈值抬到 1.0，几乎不再判「文档冲突」。
    NO_DOC_CONFLICT_GATE = "no_doc_conflict_gate"
    #: 去掉 K_eval 警报：评估器冲突阈值抬到 1.0。
    NO_EVAL_CONFLICT_ALERT = "no_eval_conflict_alert"
    #: 同时去掉两个门控维度，退化为只看支持/反驳方向。
    NO_TWO_DIMENSIONAL_GATE = "no_two_dimensional_gate"


# K_eval 只改变额外警报，不改变 primary_state；把 NO_EVAL_CONFLICT_ALERT
# 放入以分类 Macro-F1 为指标的默认消融会得到结构上恒为 0 的伪结论。
CLASSIFICATION_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant.FULL,
    AblationVariant.NO_RELIABILITY,
    AblationVariant.NO_THETA_GATE,
    AblationVariant.NO_DOC_CONFLICT_GATE,
    AblationVariant.NO_TWO_DIMENSIONAL_GATE,
)


class AblationResult(BaseModel):
    """单个消融变体的评估结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: AblationVariant
    report: ClassificationReport
    thresholds: DiagnosticThresholds
    #: 相对 ``full`` 变体的 Macro-F1 变化量（``full`` 自身为 0）。
    macro_f1_delta: float = Field(ge=-1.0, le=1.0)


def _strip_reliability(
    samples: Sequence[RAGSample], predictions: Sequence[RelationPrediction]
) -> tuple[list[RAGSample], list[RelationPrediction]]:
    """把所有可靠性置为 1.0，得到「无折扣」版本的输入。"""
    stripped_samples = [
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
                    reliability=1.0,
                )
                for c in s.contexts
            ],
            gold_state=s.gold_state,
        )
        for s in samples
    ]
    stripped_predictions = [
        p.model_copy(update={"evaluator_reliability": 1.0}) for p in predictions
    ]
    return stripped_samples, stripped_predictions


def _variant_thresholds(
    variant: AblationVariant, base: DiagnosticThresholds
) -> DiagnosticThresholds:
    """给出该变体使用的阈值。"""
    if variant is AblationVariant.NO_THETA_GATE:
        return base.model_copy(update={"theta_threshold": 1.0})
    if variant is AblationVariant.NO_DOC_CONFLICT_GATE:
        return base.model_copy(update={"document_conflict_threshold": 1.0})
    if variant is AblationVariant.NO_TWO_DIMENSIONAL_GATE:
        return base.model_copy(
            update={"theta_threshold": 1.0, "document_conflict_threshold": 1.0}
        )
    return base


def run_ablation(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    base_thresholds: DiagnosticThresholds,
    variants: Sequence[AblationVariant] | None = None,
) -> list[AblationResult]:
    """依次运行各消融变体并计算分类指标。

    Args:
        samples: 数据集样本，每条须带 ``gold_state``。
        predictions: 关系评估器输出。
        base_thresholds: 完整方法使用的阈值（应为验证集上选出的那一组）。
        variants: 要跑的变体；``None`` 表示全部。

    Returns:
        每个变体一条 :class:`AblationResult`，``full`` 排在最前。

    Raises:
        ValueError: 存在缺少 ``gold_state`` 的样本。
    """
    missing = [s.sample_id for s in samples if s.gold_state is None]
    if missing:
        raise ValueError(
            f"以下样本缺少 gold_state，无法参与消融实验：{sorted(missing)[:5]}"
        )

    ordered = (
        list(variants)
        if variants is not None
        else list(CLASSIFICATION_ABLATION_VARIANTS)
    )
    if AblationVariant.NO_EVAL_CONFLICT_ALERT in ordered:
        raise ValueError(
            "no_eval_conflict_alert 不能用四分类 Macro-F1 做消融：K_eval 只改变 "
            "evaluator_disagreement 警报，不改变 primary_state。请在具备独立的"
            "评估器分歧金标准后，单独报告该警报的 AUROC/AUPRC/F1"
        )
    if AblationVariant.FULL in ordered:
        ordered = [AblationVariant.FULL] + [
            v for v in ordered if v is not AblationVariant.FULL
        ]

    results: list[AblationResult] = []
    full_macro_f1: float | None = None
    for variant in ordered:
        if variant is AblationVariant.NO_RELIABILITY:
            variant_samples, variant_predictions = _strip_reliability(
                samples, predictions
            )
        else:
            variant_samples, variant_predictions = list(samples), list(predictions)

        thresholds = _variant_thresholds(variant, base_thresholds)
        claim_results = run_pipeline(
            variant_samples, variant_predictions, thresholds
        )
        report = classification_report(
            [r.gold_state.value for r in claim_results],  # type: ignore[union-attr]
            [predicted_label(r.diagnostic) for r in claim_results],
            method=variant.value,
        )
        if variant is AblationVariant.FULL:
            full_macro_f1 = report.macro_f1
        delta = 0.0 if full_macro_f1 is None else report.macro_f1 - full_macro_f1
        results.append(
            AblationResult(
                variant=variant,
                report=report,
                thresholds=thresholds,
                macro_f1_delta=delta,
            )
        )
    return results
