"""第十一阶段 baseline 统一判定规则的测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_ds.baselines.decision import decide_baseline_state
from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.schemas import EvidenceState

THRESHOLDS = BaselineThresholds()


# --------------------------------------------------------------------------
# 阈值模型
# --------------------------------------------------------------------------


def test_default_thresholds_are_debug_values() -> None:
    """默认阈值为调试值 0.5 / 1e-6。"""
    assert THRESHOLDS.decision_threshold == pytest.approx(0.5)
    assert THRESHOLDS.tie_tolerance == pytest.approx(1e-6)


@pytest.mark.parametrize("field", ["decision_threshold", "tie_tolerance"])
@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_thresholds_out_of_range_are_rejected(field: str, bad_value: float) -> None:
    """阈值超出 [0, 1] 时被拒绝。"""
    with pytest.raises(ValidationError, match=field):
        BaselineThresholds(**{field: bad_value})


def test_thresholds_reject_unknown_field_and_are_immutable() -> None:
    """阈值模型禁止未定义字段且不可变。"""
    with pytest.raises(ValidationError, match="tau"):
        BaselineThresholds(tau=0.5)

    with pytest.raises(ValidationError):
        THRESHOLDS.decision_threshold = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------
# 1-5. 五条判定规则
# --------------------------------------------------------------------------


def test_below_threshold_gives_insufficient() -> None:
    """最高分低于 decision_threshold -> insufficient / below_threshold。"""
    state, reason = decide_baseline_state(0.4, 0.35, 0.25, THRESHOLDS)

    assert state is EvidenceState.INSUFFICIENT
    assert reason is BaselineDecisionReason.BELOW_THRESHOLD


def test_score_tie_gives_insufficient() -> None:
    """最高分平局 -> insufficient / score_tie。"""
    state, reason = decide_baseline_state(0.5, 0.5, 0.0, THRESHOLDS)

    assert state is EvidenceState.INSUFFICIENT
    assert reason is BaselineDecisionReason.SCORE_TIE


def test_near_tie_within_tolerance_is_a_tie() -> None:
    """差距在 tie_tolerance 内也算平局。"""
    state, reason = decide_baseline_state(0.5 + 5e-7, 0.5 - 5e-7, 0.0, THRESHOLDS)

    assert state is EvidenceState.INSUFFICIENT
    assert reason is BaselineDecisionReason.SCORE_TIE


def test_unknown_highest_gives_insufficient() -> None:
    """unknown 最高 -> insufficient / unknown_highest。"""
    state, reason = decide_baseline_state(0.2, 0.1, 0.7, THRESHOLDS)

    assert state is EvidenceState.INSUFFICIENT
    assert reason is BaselineDecisionReason.UNKNOWN_HIGHEST


def test_support_highest_gives_supported() -> None:
    """support 最高且达到阈值 -> supported / decided。"""
    state, reason = decide_baseline_state(0.7, 0.2, 0.1, THRESHOLDS)

    assert state is EvidenceState.SUPPORTED
    assert reason is BaselineDecisionReason.DECIDED


def test_refute_highest_gives_refuted() -> None:
    """refute 最高且达到阈值 -> refuted / decided。"""
    state, reason = decide_baseline_state(0.2, 0.7, 0.1, THRESHOLDS)

    assert state is EvidenceState.REFUTED
    assert reason is BaselineDecisionReason.DECIDED


def test_score_exactly_at_threshold_is_not_below() -> None:
    """恰好等于阈值时不算「低于阈值」。"""
    state, reason = decide_baseline_state(0.5, 0.3, 0.2, THRESHOLDS)

    assert state is EvidenceState.SUPPORTED
    assert reason is BaselineDecisionReason.DECIDED


# --------------------------------------------------------------------------
# 判定顺序
# --------------------------------------------------------------------------


def test_threshold_check_precedes_tie_check() -> None:
    """既平局又低于阈值时，报告 below_threshold（阈值检查在前）。"""
    state, reason = decide_baseline_state(0.475, 0.475, 0.05, THRESHOLDS)

    assert state is EvidenceState.INSUFFICIENT
    assert reason is BaselineDecisionReason.BELOW_THRESHOLD


def test_same_scores_become_a_tie_under_a_lower_threshold() -> None:
    """把阈值调到 0.4，同一组分数就变成 score_tie。"""
    state, reason = decide_baseline_state(
        0.475, 0.475, 0.05, BaselineThresholds(decision_threshold=0.4)
    )

    assert state is EvidenceState.INSUFFICIENT
    assert reason is BaselineDecisionReason.SCORE_TIE


def test_tie_is_never_broken_towards_supported() -> None:
    """平局不会被默认判成 supported，正反顺序结果一致。"""
    forward = decide_baseline_state(0.5, 0.5, 0.0, THRESHOLDS)
    backward = decide_baseline_state(0.5, 0.5, 0.0, THRESHOLDS)

    assert forward == backward
    assert forward[0] is not EvidenceState.SUPPORTED


def test_decision_is_deterministic() -> None:
    """相同输入多次判定结果完全一致。"""
    runs = [decide_baseline_state(0.6, 0.3, 0.1, THRESHOLDS) for _ in range(5)]

    assert all(run == runs[0] for run in runs)


# --------------------------------------------------------------------------
# 6. 永不输出 conflicting
# --------------------------------------------------------------------------


def test_decision_never_returns_conflicting() -> None:
    """遍历大量分数组合，永远不会返回 conflicting。"""
    step = 0.05
    checked = 0
    value = 0.0
    while value <= 1.0 + 1e-9:
        other = 0.0
        while other <= 1.0 - value + 1e-9:
            state, _ = decide_baseline_state(
                value, other, 1.0 - value - other, THRESHOLDS
            )
            assert state is not EvidenceState.CONFLICTING
            assert state in {
                EvidenceState.SUPPORTED,
                EvidenceState.REFUTED,
                EvidenceState.INSUFFICIENT,
            }
            checked += 1
            other += step
        value += step

    assert checked > 200


def test_prediction_model_rejects_conflicting() -> None:
    """BaselinePrediction 在结构上禁止 conflicting。"""
    with pytest.raises(ValidationError, match="不允许输出"):
        BaselinePrediction(
            sample_id="s1",
            claim_id="c1",
            method=BaselineMethod.WEIGHTED_AVERAGE,
            evaluator=None,
            score_support=0.5,
            score_refute=0.4,
            score_unknown=0.1,
            predicted_state=EvidenceState.CONFLICTING,
            reason=BaselineDecisionReason.DECIDED,
            input_count=2,
        )


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------


def test_scores_must_sum_to_one() -> None:
    """三个分数之和不为 1 时报错。"""
    with pytest.raises(ValueError, match="必须等于 1"):
        decide_baseline_state(0.5, 0.3, 0.1, THRESHOLDS)


def test_prediction_model_rejects_bad_sum() -> None:
    """结果模型同样要求分数归一。"""
    with pytest.raises(ValidationError, match="必须等于 1"):
        BaselinePrediction(
            sample_id="s1",
            claim_id="c1",
            method=BaselineMethod.MAJORITY_VOTE,
            evaluator=None,
            score_support=0.5,
            score_refute=0.3,
            score_unknown=0.1,
            predicted_state=EvidenceState.SUPPORTED,
            reason=BaselineDecisionReason.DECIDED,
            input_count=1,
        )


def test_evaluator_field_matches_method() -> None:
    """只有 single_evaluator 方法可以（且必须）带 evaluator。"""
    with pytest.raises(ValidationError, match="只有 single_evaluator"):
        BaselinePrediction(
            sample_id="s1",
            claim_id="c1",
            method=BaselineMethod.MAJORITY_VOTE,
            evaluator="mock_a",
            score_support=0.6,
            score_refute=0.3,
            score_unknown=0.1,
            predicted_state=EvidenceState.SUPPORTED,
            reason=BaselineDecisionReason.DECIDED,
            input_count=1,
        )

    with pytest.raises(ValidationError, match="只有 single_evaluator"):
        BaselinePrediction(
            sample_id="s1",
            claim_id="c1",
            method=BaselineMethod.SINGLE_EVALUATOR,
            evaluator=None,
            score_support=0.6,
            score_refute=0.3,
            score_unknown=0.1,
            predicted_state=EvidenceState.SUPPORTED,
            reason=BaselineDecisionReason.DECIDED,
            input_count=1,
        )
