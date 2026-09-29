"""Conflict-Aware Weighted Average baseline：能输出 conflicting 的公平对照。

为什么需要它
------------
前三个 baseline 在结构上无法输出 ``conflicting``，而 D-S 可以输出四类，
于是「D-S 四分类占优」总有一部分功劳可以算在输出空间差异上。本方法在
**与 weighted_average 完全相同的加权分数**上加一条显式冲突规则，让
baseline 也拥有四类输出空间 —— 剩下的差距才能归因于 D-S 融合机制本身。

规则
----
分数计算与 :mod:`~rag_ds.baselines.weighted_average` 逐字相同（权重为
文档可靠性 × 评估器可靠性）。判定顺序::

    1. min(s, r) >= conflict_threshold 且 |s - r| <= conflict_margin
         -> conflicting, conflict_detected
    2. 其余情况 -> 走 decide_baseline_state 的标准级联

**冲突检查排在最前**，与 D-S 侧「混合区域（高无知 + 高冲突）映射为
CONFLICTING」的取舍一致：当两派证据都足够强时，「它们互相矛盾」是比
「整体信心不足」更具体、更有行动价值的信息。阈值检查若排在前面，
针锋相对的证据（平均后两个方向分数都不到 decision_threshold）会永远
被压成 below_threshold，规则形同虚设。

``conflict_margin`` 限制两个方向必须**势均力敌**：0.90 / 0.35 这种一边倒
不是冲突，是支持。``conflict_threshold`` 限制两个方向都必须**真的有分量**：
0.05 / 0.05 的平局是无知，不是冲突。

本方法不做任何 D-S 组合、不计算冲突量 K —— 它看到的仍然只是三个平均后的
分数，这是它与 D-S 的本质区别，也是对照的意义所在。

``retrieval_score`` 与 ``gold_state`` 都不参与计算。
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_ds.baselines._shared import contexts_by_doc_id, no_evidence_prediction
from rag_ds.baselines.decision import decide_baseline_state
from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.schemas import Claim, EvidenceState, RAGSample, RelationPrediction

__all__ = ["decide_conflict_aware_state", "predict_conflict_aware"]

#: margin 比较允许的浮点噪声：0.45 - 0.35 在二进制下是 0.10000000000000003，
#: 没有它「差距恰好等于 margin」这个边界会随机偏向一侧。
_MARGIN_FLOAT_SLACK = 1e-12


def decide_conflict_aware_state(
    score_support: float,
    score_refute: float,
    score_unknown: float,
    thresholds: BaselineThresholds,
) -> tuple[EvidenceState, BaselineDecisionReason]:
    """在已算好的三个分数上应用「冲突优先」判定。

    与 :func:`predict_conflict_aware` 内部使用的规则完全一致，单独暴露是
    为了让阈值搜索可以对同一组分数反复试不同阈值，而不必重复加权计算。
    """
    if (
        min(score_support, score_refute) >= thresholds.conflict_threshold
        and abs(score_support - score_refute)
        <= thresholds.conflict_margin + _MARGIN_FLOAT_SLACK
    ):
        return EvidenceState.CONFLICTING, BaselineDecisionReason.CONFLICT_DETECTED
    return decide_baseline_state(
        score_support, score_refute, score_unknown, thresholds
    )


def predict_conflict_aware(
    sample: RAGSample,
    claim: Claim,
    predictions: Iterable[RelationPrediction],
    thresholds: BaselineThresholds,
) -> BaselinePrediction:
    """加权平均 + 显式冲突规则，唯一能输出 conflicting 的 baseline。

    Args:
        sample: claim 所属样本，用于按 ``doc_id`` 找到对应文档。
        claim: 待判定的断言。
        predictions: 该 claim 的关系预测（可来自多个评估器）。
        thresholds: 判定阈值，含 ``conflict_threshold`` 与
            ``conflict_margin``。

    Returns:
        :class:`BaselinePrediction`；总权重为零时返回 ``(0, 0, 1)`` 与
        ``no_evidence``（没有证据不是冲突）。

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
            sample, claim, BaselineMethod.CONFLICT_AWARE, len(ordered)
        )

    score_support = weighted_support / total_weight
    score_refute = weighted_refute / total_weight
    score_unknown = weighted_unknown / total_weight

    predicted_state, reason = decide_conflict_aware_state(
        score_support, score_refute, score_unknown, thresholds
    )

    return BaselinePrediction(
        sample_id=sample.sample_id,
        claim_id=claim.claim_id,
        method=BaselineMethod.CONFLICT_AWARE,
        evaluator=None,
        score_support=score_support,
        score_refute=score_refute,
        score_unknown=score_unknown,
        predicted_state=predicted_state,
        reason=reason,
        input_count=len(ordered),
        gold_state=sample.gold_state,  # 只带走，不参与计算
    )
