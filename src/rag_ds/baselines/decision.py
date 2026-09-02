"""三个 baseline 共用的判定规则。

给定三个归一化分数，按固定顺序判定::

    1. 校验三个分数之和为 1
    2. 最高分 < decision_threshold        -> insufficient, below_threshold
    3. 最高分之间差距 <= tie_tolerance    -> insufficient, score_tie
    4. unknown 最高                        -> insufficient, unknown_highest
    5. support 最高                        -> supported,    decided
    6. refute 最高                         -> refuted,      decided

顺序是有意的：**阈值检查排在平局检查之前**。两条针锋相对的文档加权平均后
常常同时满足「分数接近」与「都不够高」，此时报告 ``below_threshold``
（整体信心不足）比报告 ``score_tie`` 更贴近实际发生的事。

平局一律判为 ``insufficient``：不随机打破，不默认偏向 ``supported``，
也永远不会输出 ``conflicting`` —— 后者不在 baseline 的输出空间内。
"""

from __future__ import annotations

from rag_ds.baselines.models import (
    BASELINE_SCORE_SUM_TOLERANCE,
    BaselineDecisionReason,
    BaselineThresholds,
)
from rag_ds.schemas import EvidenceState

__all__ = ["decide_baseline_state"]


def decide_baseline_state(
    score_support: float,
    score_refute: float,
    score_unknown: float,
    thresholds: BaselineThresholds,
) -> tuple[EvidenceState, BaselineDecisionReason]:
    """把三个分数判定成状态与原因。

    Args:
        score_support: 支持分数。
        score_refute: 反驳分数。
        score_unknown: 不确定分数。
        thresholds: 判定阈值。

    Returns:
        ``(状态, 原因)``；状态只可能是 ``SUPPORTED`` / ``REFUTED`` /
        ``INSUFFICIENT``。

    Raises:
        ValueError: 三个分数之和不为 1。

    Note:
        纯函数，不读取 ``gold_state``，不使用随机数。
    """
    total = score_support + score_refute + score_unknown
    if abs(total - 1.0) > BASELINE_SCORE_SUM_TOLERANCE:
        raise ValueError(
            "三个分数之和必须等于 1，"
            f"当前为 {total!r}（允许误差 {BASELINE_SCORE_SUM_TOLERANCE}）"
        )

    scored = (
        (score_support, EvidenceState.SUPPORTED),
        (score_refute, EvidenceState.REFUTED),
        (score_unknown, EvidenceState.INSUFFICIENT),
    )
    highest = max(score for score, _ in scored)

    if highest < thresholds.decision_threshold:
        return EvidenceState.INSUFFICIENT, BaselineDecisionReason.BELOW_THRESHOLD

    contenders = [
        state for score, state in scored if highest - score <= thresholds.tie_tolerance
    ]
    if len(contenders) > 1:
        return EvidenceState.INSUFFICIENT, BaselineDecisionReason.SCORE_TIE

    winner = contenders[0]
    if winner is EvidenceState.INSUFFICIENT:
        return EvidenceState.INSUFFICIENT, BaselineDecisionReason.UNKNOWN_HIGHEST
    return winner, BaselineDecisionReason.DECIDED
