"""嵌套交叉验证下的消融：每个变体在每折内层**各自重新搜阈值**。

为什么不能沿用 :mod:`rag_ds.experiments.ablation`
------------------------------------------------
原消融在单次留出划分上做：先用完整方法在验证集上选好阈值，再把某个阈值抬到
1.0 看测试集掉多少。这有两个问题：

1. v1 测试集只有 80 条，Δ 没有置信区间，离散版与连续版甚至给出相反结论
   （去掉 theta 门控一个 +0.031、一个 −0.069）；
2. 去掉一个门控后，**剩下那个门控的最优阈值本该跟着变**。沿用完整方法的
   阈值等于没给消融变体调参的机会，会高估每个组件的贡献。

这里的做法与主实验的嵌套交叉验证完全一致：同一分折、同一内层网格构造，
每个变体只是网格不同 —— 被去掉的那一维固定为 1.0（门控基本不再触发，与原
消融的定义相同），另一维照常在内层观测值分位数上搜。汇总后每条 claim 在
每个变体下恰好一条预测，可以直接做配对自助检验。

``no_reliability`` 不在这里：数据集里全部可靠性都是 1.0，去折扣是恒等变换
（见 :attr:`~rag_ds.experiments.ablation.AblationResult.is_vacuous`）。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.ablation import AblationVariant
from rag_ds.experiments.comparison import MethodPrediction
from rag_ds.experiments.cross_validation import _subset, make_stratified_folds
from rag_ds.experiments.selective_confidence import ds_confidence
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import RAGSample, RelationPrediction
from rag_ds.tuning.threshold_search import (
    SplitName,
    ThresholdGrid,
    predicted_label,
    rediagnose,
    search_thresholds,
)

__all__ = [
    "CV_ABLATION_VARIANTS",
    "AblationFoldThresholds",
    "NestedAblationResult",
    "run_nested_ablation",
]

#: 以四分类 Macro-F1 为指标、在本数据上不空转的变体，``full`` 在最前。
CV_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant.FULL,
    AblationVariant.NO_THETA_GATE,
    AblationVariant.NO_DOC_CONFLICT_GATE,
    AblationVariant.NO_TWO_DIMENSIONAL_GATE,
)

#: 「关掉」一维门控时该维固定的阈值，与原消融相同。
_GATE_OFF = 1.0


class AblationFoldThresholds(BaseModel):
    """某个外折里各变体在内层选出的阈值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold: int = Field(ge=0)
    test_size: int = Field(ge=1)
    inner_size: int = Field(ge=1)
    thresholds: dict[str, DiagnosticThresholds]
    inner_macro_f1: dict[str, float]


class NestedAblationResult(BaseModel):
    """一次嵌套交叉验证消融的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_folds: int = Field(ge=2)
    seed: int
    #: 汇总后的逐条预测；``method`` 字段是变体名，每个变体每条 claim 一条。
    predictions: tuple[MethodPrediction, ...]
    folds: tuple[AblationFoldThresholds, ...]
    variants: tuple[str, ...]


def _variant_grid(variant: AblationVariant, observed: ThresholdGrid) -> ThresholdGrid:
    """被去掉的那一维固定为 1.0，另一维沿用内层观测网格。"""
    theta = observed.theta_values
    doc = observed.document_conflict_values
    if variant in (
        AblationVariant.NO_THETA_GATE,
        AblationVariant.NO_TWO_DIMENSIONAL_GATE,
    ):
        theta = (_GATE_OFF,)
    if variant in (
        AblationVariant.NO_DOC_CONFLICT_GATE,
        AblationVariant.NO_TWO_DIMENSIONAL_GATE,
    ):
        doc = (_GATE_OFF,)
    return observed.model_copy(
        update={"theta_values": theta, "document_conflict_values": doc}
    )


def run_nested_ablation(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    *,
    n_folds: int = 5,
    seed: int = 42,
    grid_steps: int = 9,
    evaluator_conflict_threshold: float = 0.4,
    variants: Sequence[AblationVariant] = CV_ABLATION_VARIANTS,
) -> NestedAblationResult:
    """在与主实验相同的嵌套交叉验证里跑门控消融。

    Args:
        samples: 全部样本，每条须带 ``gold_state``。
        predictions: 全部关系预测。
        n_folds: 外折数，须与主实验一致才能与其结果对照。
        seed: 分折随机种子，同上。
        grid_steps: 每个轴的候选点数，同上。
        evaluator_conflict_threshold: 固定的 K_eval 告警阈值。
        variants: 要跑的变体；只接受 :data:`CV_ABLATION_VARIANTS` 中的取值。

    Returns:
        :class:`NestedAblationResult`。

    Raises:
        ValueError: 变体不受支持、分折失败，或某折内层没有可用的 m_theta。
    """
    unsupported = [v.value for v in variants if v not in CV_ABLATION_VARIANTS]
    if unsupported:
        raise ValueError(
            f"嵌套消融不支持这些变体：{unsupported}；"
            "no_reliability 在全部可靠性为 1.0 的数据上是恒等变换，"
            "no_eval_conflict_alert 不改变四分类标签"
        )
    ordered = [v for v in CV_ABLATION_VARIANTS if v in variants]

    folds = make_stratified_folds(samples, n_folds, seed)
    base = DiagnosticThresholds(
        evaluator_conflict_threshold=evaluator_conflict_threshold
    )

    pooled: list[MethodPrediction] = []
    records: list[AblationFoldThresholds] = []
    for fold_index, test_indices in enumerate(folds):
        inner_indices = [
            i for position, fold in enumerate(folds) if position != fold_index
            for i in fold
        ]
        inner_samples, inner_predictions = _subset(samples, predictions, inner_indices)
        test_samples, test_predictions = _subset(samples, predictions, test_indices)

        # 阈值不影响门控之前的任何 D-S 计算：两边各跑一次 pipeline，
        # 之后每个变体、每个网格点都只重做门控。
        inner_results = run_pipeline(inner_samples, inner_predictions, base)
        test_results = run_pipeline(test_samples, test_predictions, base)
        observed_theta = [
            r.diagnostic.m_theta
            for r in inner_results
            if r.diagnostic.m_theta is not None
        ]
        if not observed_theta:
            raise ValueError(
                f"第 {fold_index} 折的内层数据没有可用的 m_theta，无法搜索阈值"
            )
        observed = ThresholdGrid.from_observed(
            observed_theta,
            [r.diagnostic.k_doc for r in inner_results],
            steps=grid_steps,
            evaluator_conflict_threshold=evaluator_conflict_threshold,
        )

        chosen: dict[str, DiagnosticThresholds] = {}
        inner_f1: dict[str, float] = {}
        for variant in ordered:
            search = search_thresholds(
                inner_results, SplitName.VALIDATION, _variant_grid(variant, observed)
            )
            chosen[variant.value] = search.best.thresholds
            inner_f1[variant.value] = search.best.macro_f1
            for result in test_results:
                diagnostic = rediagnose(result, search.best.thresholds)
                label = predicted_label(diagnostic)
                pooled.append(
                    MethodPrediction(
                        method=variant.value,
                        sample_id=result.sample_id,
                        claim_id=result.claim_id,
                        predicted_label=label,
                        gold_label=result.gold_state.value,  # type: ignore[union-attr]
                        insufficiency_score=diagnostic.m_theta or 0.0,
                        conflict_score=diagnostic.k_doc,
                        confidence=ds_confidence(
                            label,
                            diagnostic.m_support,
                            diagnostic.m_refute,
                            diagnostic.m_theta,
                            diagnostic.k_doc,
                        ),
                    )
                )

        records.append(
            AblationFoldThresholds(
                fold=fold_index,
                test_size=len(test_samples),
                inner_size=len(inner_samples),
                thresholds=chosen,
                inner_macro_f1=inner_f1,
            )
        )

    return NestedAblationResult(
        n_folds=n_folds,
        seed=seed,
        predictions=tuple(pooled),
        folds=tuple(records),
        variants=tuple(v.value for v in ordered),
    )
