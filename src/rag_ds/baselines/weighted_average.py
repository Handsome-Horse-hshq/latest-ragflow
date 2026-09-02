"""Weighted Average baseline。

对该 claim 的**全部评估器 × 全部文档**的关系概率做一次加权平均::

    weight = context.reliability * prediction.evaluator_reliability

    score_support = sum(weight * p_support) / sum(weight)
    score_refute  = sum(weight * p_refute)  / sum(weight)
    score_unknown = sum(weight * p_unknown) / sum(weight)

这里的两级可靠性是**一次性相乘**的，与 D-S 侧刻意分两级施加不同 ——
本模块正是要复现「朴素做法」的行为，不做任何 D-S 组合、不计算冲突量、
不先把概率转成 BPA。

``retrieval_score`` 与 ``gold_state`` 都不参与计算。
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_ds.baselines._shared import contexts_by_doc_id, no_evidence_prediction
from rag_ds.baselines.decision import decide_baseline_state
from rag_ds.baselines.models import (
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.schemas import Claim, RAGSample, RelationPrediction

__all__ = ["predict_weighted_average"]


def predict_weighted_average(
    sample: RAGSample,
    claim: Claim,
    predictions: Iterable[RelationPrediction],
    thresholds: BaselineThresholds,
) -> BaselinePrediction:
    """按 文档可靠性 x 评估器可靠性 加权平均三个概率。

    Args:
        sample: claim 所属样本，用于按 ``doc_id`` 找到对应文档。
        claim: 待判定的断言。
        predictions: 该 claim 的关系预测（可来自多个评估器）。
        thresholds: 判定阈值。

    Returns:
        :class:`BaselinePrediction`；总权重为零时返回
        ``(0, 0, 1)`` 与 ``no_evidence``。

    Raises:
        KeyError: 某条预测的 ``doc_id`` 不在 ``sample.contexts`` 中。

    Note:
        不修改输入对象。
    """
    ordered = list(predictions)
    contexts = contexts_by_doc_id(sample)

    total_weight = 0.0
    weighted_support = 0.0
    weighted_refute = 0.0
    weighted_unknown = 0.0
    for prediction in ordered:
        weight = (
            contexts[prediction.doc_id].reliability * prediction.evaluator_reliability
        )
        total_weight += weight
        weighted_support += weight * prediction.p_support
        weighted_refute += weight * prediction.p_refute
        weighted_unknown += weight * prediction.p_unknown

    if total_weight <= 0.0:
        return no_evidence_prediction(
            sample, claim, BaselineMethod.WEIGHTED_AVERAGE, len(ordered)
        )

    score_support = weighted_support / total_weight
    score_refute = weighted_refute / total_weight
    score_unknown = weighted_unknown / total_weight
    predicted_state, reason = decide_baseline_state(
        score_support, score_refute, score_unknown, thresholds
    )

    return BaselinePrediction(
        sample_id=sample.sample_id,
        claim_id=claim.claim_id,
        method=BaselineMethod.WEIGHTED_AVERAGE,
        evaluator=None,
        score_support=score_support,
        score_refute=score_refute,
        score_unknown=score_unknown,
        predicted_state=predicted_state,
        reason=reason,
        input_count=len(ordered),
        gold_state=sample.gold_state,  # 只带走，不参与计算
    )
