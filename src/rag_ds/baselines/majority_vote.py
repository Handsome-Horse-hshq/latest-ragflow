"""Majority Vote baseline。

每条关系预测投出**一票**，票重相同::

    p_support 最大  -> support 票
    p_refute  最大  -> refute  票
    p_unknown 最大  -> unknown 票
    该条内部平局    -> unknown 票

    score_x = x 票数 / 总票数

**完全不使用可靠性**：文档可靠性、评估器可靠性、``retrieval_score`` 一律
不参与，一篇可疑文档与一篇权威文档同样是一票。这正是本 baseline 要暴露的
弱点之一。

也不计算任何冲突量：两票针锋相对时只会得到 0.5 / 0.5，随后被统一判定规则
压成 ``insufficient`` / ``score_tie``，看不出「有人说是、有人说否」。
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_ds.baselines._shared import no_evidence_prediction
from rag_ds.baselines.decision import decide_baseline_state
from rag_ds.baselines.models import (
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.schemas import Claim, RAGSample, RelationPrediction

__all__ = ["cast_vote", "predict_majority_vote"]

#: 投票用的三个槽位。
_SUPPORT, _REFUTE, _UNKNOWN = "support", "refute", "unknown"


def cast_vote(prediction: RelationPrediction) -> str:
    """把一条关系预测转成一票。

    单条预测内部出现并列最大值时投 ``unknown`` —— 该条本身就分不清方向，
    不应替它选一边。

    Args:
        prediction: 关系预测。

    Returns:
        ``"support"`` / ``"refute"`` / ``"unknown"`` 之一。
    """
    scores = (
        (prediction.p_support, _SUPPORT),
        (prediction.p_refute, _REFUTE),
        (prediction.p_unknown, _UNKNOWN),
    )
    highest = max(score for score, _ in scores)
    winners = [slot for score, slot in scores if score == highest]
    return winners[0] if len(winners) == 1 else _UNKNOWN


def predict_majority_vote(
    sample: RAGSample,
    claim: Claim,
    predictions: Iterable[RelationPrediction],
    thresholds: BaselineThresholds,
) -> BaselinePrediction:
    """每条关系预测一票，少数服从多数。

    Args:
        sample: claim 所属样本。
        claim: 待判定的断言。
        predictions: 该 claim 的关系预测（可来自多个评估器）。
        thresholds: 判定阈值。

    Returns:
        :class:`BaselinePrediction`；没有任何票时返回 ``(0, 0, 1)`` 与
        ``no_evidence``。

    Note:
        不使用任何可靠性，不修改输入对象。
    """
    ordered = list(predictions)
    if not ordered:
        return no_evidence_prediction(
            sample, claim, BaselineMethod.MAJORITY_VOTE, 0
        )

    votes = [cast_vote(prediction) for prediction in ordered]
    total = len(votes)
    score_support = votes.count(_SUPPORT) / total
    score_refute = votes.count(_REFUTE) / total
    score_unknown = votes.count(_UNKNOWN) / total

    predicted_state, reason = decide_baseline_state(
        score_support, score_refute, score_unknown, thresholds
    )

    return BaselinePrediction(
        sample_id=sample.sample_id,
        claim_id=claim.claim_id,
        method=BaselineMethod.MAJORITY_VOTE,
        evaluator=None,
        score_support=score_support,
        score_refute=score_refute,
        score_unknown=score_unknown,
        predicted_state=predicted_state,
        reason=reason,
        input_count=total,
        gold_state=sample.gold_state,  # 只带走，不参与计算
    )
