"""Single Evaluator baseline。

只使用**一个指定评估器**的判断，按文档可靠性加权平均::

    score_support = sum(context.reliability * p_support)
                    / sum(context.reliability)

其余两个分数同理。

``evaluator_reliability`` **不参与**：本方法只有一个评估器，对它整体打折
会同比例缩放三个分数，除了把所有值压向 0 之外不改变任何相对关系，还会让
分数失去归一性。这与 D-S 侧「评估器可靠性在文档融合之后只施加一次」的
处理是一致的取舍 —— 都不在每条文档上重复施加。

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
    MissingBaselineEvaluatorError,
)
from rag_ds.schemas import Claim, RAGSample, RelationPrediction

__all__ = ["predict_single_evaluator"]


def predict_single_evaluator(
    sample: RAGSample,
    claim: Claim,
    predictions: Iterable[RelationPrediction],
    evaluator: str,
    thresholds: BaselineThresholds,
) -> BaselinePrediction:
    """只用指定评估器的预测做文档可靠性加权平均。

    Args:
        sample: claim 所属样本。
        claim: 待判定的断言。
        predictions: 该 claim 的全部关系预测；非指定评估器的会被过滤掉。
        evaluator: 要使用的评估器名称。
        thresholds: 判定阈值。

    Returns:
        :class:`BaselinePrediction`，``evaluator`` 字段记录所用评估器。

    Raises:
        MissingBaselineEvaluatorError: 该 claim 下没有指定评估器的预测。

    Note:
        不使用其他评估器，不修改输入对象。
    """
    selected = [p for p in predictions if p.evaluator == evaluator]
    if not selected:
        raise MissingBaselineEvaluatorError(
            f"single-evaluator baseline 指定的评估器 {evaluator!r} "
            f"在 sample_id={sample.sample_id!r}, claim_id={claim.claim_id!r} "
            "下没有任何关系预测；不会改用其他评估器代替"
        )

    contexts = contexts_by_doc_id(sample)
    total_weight = 0.0
    weighted_support = 0.0
    weighted_refute = 0.0
    weighted_unknown = 0.0
    for prediction in selected:
        weight = contexts[prediction.doc_id].reliability
        total_weight += weight
        weighted_support += weight * prediction.p_support
        weighted_refute += weight * prediction.p_refute
        weighted_unknown += weight * prediction.p_unknown

    if total_weight <= 0.0:
        return no_evidence_prediction(
            sample,
            claim,
            BaselineMethod.SINGLE_EVALUATOR,
            len(selected),
            evaluator=evaluator,
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
        method=BaselineMethod.SINGLE_EVALUATOR,
        evaluator=evaluator,
        score_support=score_support,
        score_refute=score_refute,
        score_unknown=score_unknown,
        predicted_state=predicted_state,
        reason=reason,
        input_count=len(selected),
        gold_state=sample.gold_state,  # 只带走，不参与计算
    )
