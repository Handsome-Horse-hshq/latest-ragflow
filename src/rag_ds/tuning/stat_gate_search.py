"""在验证集上为 Statistic Gate 搜索两维门控阈值。

与 D-S 的 :func:`~rag_ds.tuning.threshold_search.search_thresholds` 用**同一套
协议**：网格由 :meth:`ThresholdGrid.from_observed` 按验证集上实际观测到的两个
坐标分位数构造，目标是 Macro-F1，并列时取阈值字典序最小的一组。两边唯一的
差别是坐标轴上的分数 —— 这正是这个对照组要隔离的变量。

与其他搜索一样，只允许在验证集上搜。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_ds.baselines.stat_gate import StatGateScores, decide_stat_gate
from rag_ds.metrics.classification import UNDETERMINED_LABEL, classification_report
from rag_ds.schemas import EvidenceState
from rag_ds.tuning.threshold_search import (
    SplitName,
    ThresholdCandidate,
    ThresholdGrid,
    ThresholdSearchResult,
)

__all__ = ["search_stat_gate_thresholds", "stat_gate_label"]


def stat_gate_label(state: EvidenceState | None) -> str:
    """把主状态转成指标用的标签；``None`` 记为 ``undetermined``，与 D-S 一致。"""
    return UNDETERMINED_LABEL if state is None else state.value


def search_stat_gate_thresholds(
    scores: Sequence[StatGateScores],
    gold: Mapping[tuple[str, str], str],
    split: SplitName,
    *,
    steps: int = 9,
    evaluator_conflict_threshold: float = 0.4,
    tie_tolerance: float = 1e-6,
) -> ThresholdSearchResult:
    """在验证集上网格搜索使 Macro-F1 最大的 Statistic Gate 阈值。

    Args:
        scores: 验证集上每条 claim 的 :class:`StatGateScores`。
        gold: ``(sample_id, claim_id) -> 金标准标签``。
        split: 数据划分名称；**必须**为 :attr:`SplitName.VALIDATION`。
        steps: 每个轴的候选点数（按观测值分位数构造，与 D-S 相同）。
        evaluator_conflict_threshold: 只为填满阈值模型，本方法不使用。
        tie_tolerance: 平局容差，不参与搜索。

    Returns:
        :class:`ThresholdSearchResult`；``theta_threshold`` 作用于横轴
        （证据不足统计量），``document_conflict_threshold`` 作用于纵轴
        （冲突统计量）。

    Raises:
        ValueError: ``split`` 不是验证集、``scores`` 为空，或某条缺少金标准。
    """
    if split is not SplitName.VALIDATION:
        raise ValueError(
            f"阈值只能在验证集上搜索，收到 split={split.value!r}；"
            "用测试集选阈值再用测试集报告结果没有意义"
        )
    if not scores:
        raise ValueError("阈值搜索收到空的验证集")
    missing = [s.claim_id for s in scores if (s.sample_id, s.claim_id) not in gold]
    if missing:
        raise ValueError(
            f"以下 claim 缺少金标准，无法用于阈值搜索：{sorted(missing)[:5]}"
        )

    y_true = [gold[(s.sample_id, s.claim_id)] for s in scores]
    grid = ThresholdGrid.from_observed(
        [s.insufficiency for s in scores],
        [s.conflict for s in scores],
        steps=steps,
        evaluator_conflict_threshold=evaluator_conflict_threshold,
        tie_tolerance=tie_tolerance,
    )

    candidates: list[ThresholdCandidate] = []
    for thresholds in grid.candidates():
        y_pred = [stat_gate_label(decide_stat_gate(s, thresholds)) for s in scores]
        report = classification_report(y_true, y_pred, method="stat_gate")
        candidates.append(
            ThresholdCandidate(
                thresholds=thresholds,
                macro_f1=report.macro_f1,
                accuracy=report.accuracy,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda c: (
            -c.macro_f1,
            c.thresholds.theta_threshold,
            c.thresholds.document_conflict_threshold,
        ),
    )
    return ThresholdSearchResult(
        split=split,
        best=ordered[0],
        candidates=tuple(ordered),
        claim_count=len(scores),
    )
