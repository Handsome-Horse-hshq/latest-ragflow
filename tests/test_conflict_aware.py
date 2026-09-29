"""Conflict-Aware baseline 的测试。

核心论点：它与 weighted_average 共用同一组加权分数，唯一差别是判定阶段多
一条显式冲突规则 —— 因此它是「输出空间公平」的对照，而不是第二个 D-S。
"""

from __future__ import annotations

import pytest

from rag_ds.baselines.conflict_aware import (
    decide_conflict_aware_state,
    predict_conflict_aware,
)
from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.baselines.weighted_average import predict_weighted_average
from rag_ds.schemas import (
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

THRESHOLDS = BaselineThresholds(
    decision_threshold=0.5,
    tie_tolerance=1e-6,
    conflict_threshold=0.3,
    conflict_margin=0.1,
)
CLAIM = Claim(claim_id="c1", text="断言。")


def _sample(
    docs: list[tuple[str, float]],
    gold_state: EvidenceState | None = None,
) -> RAGSample:
    """构造样本；``docs`` 为 ``(doc_id, reliability)``。"""
    return RAGSample(
        sample_id="s1",
        question="问题？",
        answer="答案。",
        claims=[CLAIM],
        contexts=[
            ContextChunk(doc_id=doc_id, text=f"文档 {doc_id}。", reliability=reliability)
            for doc_id, reliability in docs
        ],
        gold_state=gold_state,
    )


def _prediction(
    doc_id: str,
    probabilities: tuple[float, float, float],
    evaluator: str = "mock_a",
) -> RelationPrediction:
    """构造一条关系预测。"""
    return RelationPrediction(
        sample_id="s1",
        claim_id="c1",
        doc_id=doc_id,
        evaluator=evaluator,
        p_support=probabilities[0],
        p_refute=probabilities[1],
        p_unknown=probabilities[2],
        evaluator_reliability=1.0,
    )


# --------------------------------------------------------------------------
# 分数与 weighted_average 完全一致
# --------------------------------------------------------------------------


def test_scores_are_identical_to_weighted_average() -> None:
    """三个分数与 weighted_average 逐位相同，差异只在判定阶段。"""
    sample = _sample([("d1", 0.9), ("d2", 0.4)])
    predictions = [
        _prediction("d1", (0.8, 0.1, 0.1)),
        _prediction("d2", (0.2, 0.7, 0.1)),
    ]

    ca = predict_conflict_aware(sample, CLAIM, predictions, THRESHOLDS)
    wa = predict_weighted_average(sample, CLAIM, predictions, THRESHOLDS)

    assert ca.score_support == pytest.approx(wa.score_support)
    assert ca.score_refute == pytest.approx(wa.score_refute)
    assert ca.score_unknown == pytest.approx(wa.score_unknown)
    assert ca.method is BaselineMethod.CONFLICT_AWARE
    assert ca.evaluator is None


# --------------------------------------------------------------------------
# 冲突规则
# --------------------------------------------------------------------------


def test_balanced_strong_opposition_gives_conflicting() -> None:
    """支持与反驳都强且接近 → conflicting / conflict_detected。"""
    sample = _sample(
        [("d1", 1.0), ("d2", 1.0)], gold_state=EvidenceState.CONFLICTING
    )
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05)),
        _prediction("d2", (0.05, 0.9, 0.05)),
    ]

    result = predict_conflict_aware(sample, CLAIM, predictions, THRESHOLDS)

    # 平均后约 0.475 / 0.475 / 0.05：两方向都 >= 0.3 且差距 <= 0.1。
    assert result.score_support == pytest.approx(0.475)
    assert result.predicted_state is EvidenceState.CONFLICTING
    assert result.reason is BaselineDecisionReason.CONFLICT_DETECTED


def test_one_sided_evidence_is_not_conflict() -> None:
    """一边倒（0.9 vs 0.05 量级差距超过 margin）不是冲突。"""
    sample = _sample([("d1", 1.0)])
    predictions = [_prediction("d1", (0.9, 0.05, 0.05))]

    result = predict_conflict_aware(sample, CLAIM, predictions, THRESHOLDS)

    assert result.predicted_state is EvidenceState.SUPPORTED
    assert result.reason is BaselineDecisionReason.DECIDED


def test_weak_both_sides_is_not_conflict() -> None:
    """两方向都低于 conflict_threshold：是无知，不是冲突。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0)])
    predictions = [
        _prediction("d1", (0.2, 0.2, 0.6)),
        _prediction("d2", (0.2, 0.2, 0.6)),
    ]

    result = predict_conflict_aware(sample, CLAIM, predictions, THRESHOLDS)

    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is not BaselineDecisionReason.CONFLICT_DETECTED


def test_conflict_check_precedes_below_threshold() -> None:
    """两方向都不到 decision_threshold 但势均力敌时仍判冲突。

    这是规则顺序的意义：0.45 / 0.45 若先走 below_threshold 会被压成
    insufficient，「有冲突」这个信息就丢了。
    """
    state, reason = decide_conflict_aware_state(0.45, 0.45, 0.10, THRESHOLDS)

    assert state is EvidenceState.CONFLICTING
    assert reason is BaselineDecisionReason.CONFLICT_DETECTED


def test_conflict_threshold_boundary_is_inclusive() -> None:
    """min(s, r) 恰好等于 conflict_threshold 时触发（>= 语义）。"""
    state, _ = decide_conflict_aware_state(0.30, 0.30, 0.40, THRESHOLDS)

    assert state is EvidenceState.CONFLICTING


def test_conflict_margin_boundary_is_inclusive() -> None:
    """差距恰好等于 conflict_margin 时触发（<= 语义）。"""
    state, _ = decide_conflict_aware_state(0.45, 0.35, 0.20, THRESHOLDS)

    assert state is EvidenceState.CONFLICTING

    state, _ = decide_conflict_aware_state(0.46, 0.35, 0.19, THRESHOLDS)
    assert state is not EvidenceState.CONFLICTING


# --------------------------------------------------------------------------
# 边界与不变量
# --------------------------------------------------------------------------


def test_no_predictions_gives_no_evidence_not_conflict() -> None:
    """没有任何证据时是 no_evidence：「没有证据」不是「证据冲突」。"""
    result = predict_conflict_aware(_sample([]), CLAIM, [], THRESHOLDS)

    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is BaselineDecisionReason.NO_EVIDENCE
    assert result.input_count == 0


def test_zero_weight_gives_no_evidence() -> None:
    """权重和为零时同样降级为 no_evidence。"""
    sample = _sample([("d1", 0.0)])

    result = predict_conflict_aware(
        sample, CLAIM, [_prediction("d1", (0.8, 0.1, 0.1))], THRESHOLDS
    )

    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is BaselineDecisionReason.NO_EVIDENCE


@pytest.mark.parametrize(
    "gold_state",
    [None, EvidenceState.SUPPORTED, EvidenceState.CONFLICTING],
)
def test_gold_state_does_not_affect_the_result(
    gold_state: EvidenceState | None,
) -> None:
    """gold_state 只被带走，不改变判定。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0)], gold_state=gold_state)
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05)),
        _prediction("d2", (0.05, 0.9, 0.05)),
    ]

    result = predict_conflict_aware(sample, CLAIM, predictions, THRESHOLDS)

    assert result.predicted_state is EvidenceState.CONFLICTING
    assert result.gold_state is gold_state


def test_inputs_are_not_modified() -> None:
    """不修改传入对象。"""
    sample = _sample([("d1", 1.0)])
    predictions = [_prediction("d1", (0.8, 0.1, 0.1))]
    before = (sample.model_dump(), [p.model_dump() for p in predictions])

    predict_conflict_aware(sample, CLAIM, predictions, THRESHOLDS)

    assert (sample.model_dump(), [p.model_dump() for p in predictions]) == before


# --------------------------------------------------------------------------
# 输出空间约束
# --------------------------------------------------------------------------


def test_conflict_aware_may_output_conflicting() -> None:
    """模型层允许 conflict_aware 输出 conflicting。"""
    prediction = BaselinePrediction(
        sample_id="s1",
        claim_id="c1",
        method=BaselineMethod.CONFLICT_AWARE,
        score_support=0.45,
        score_refute=0.45,
        score_unknown=0.10,
        predicted_state=EvidenceState.CONFLICTING,
        reason=BaselineDecisionReason.CONFLICT_DETECTED,
        input_count=2,
    )

    assert prediction.predicted_state is EvidenceState.CONFLICTING


@pytest.mark.parametrize(
    "method",
    [
        BaselineMethod.WEIGHTED_AVERAGE,
        BaselineMethod.MAJORITY_VOTE,
        BaselineMethod.SINGLE_EVALUATOR,
    ],
)
def test_other_methods_still_cannot_output_conflicting(
    method: BaselineMethod,
) -> None:
    """其余三个方法在模型层仍然禁止 conflicting。"""
    with pytest.raises(ValueError, match="conflicting"):
        BaselinePrediction(
            sample_id="s1",
            claim_id="c1",
            method=method,
            evaluator="mock_a" if method is BaselineMethod.SINGLE_EVALUATOR else None,
            score_support=0.45,
            score_refute=0.45,
            score_unknown=0.10,
            predicted_state=EvidenceState.CONFLICTING,
            reason=BaselineDecisionReason.DECIDED,
            input_count=2,
        )
