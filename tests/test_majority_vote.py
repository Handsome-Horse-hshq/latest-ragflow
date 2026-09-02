"""第十一阶段 Majority Vote baseline 的测试。"""

from __future__ import annotations

import pytest

from rag_ds.baselines.majority_vote import cast_vote, predict_majority_vote
from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
    BaselineThresholds,
)
from rag_ds.schemas import (
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

THRESHOLDS = BaselineThresholds()
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
    evaluator_reliability: float = 1.0,
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
        evaluator_reliability=evaluator_reliability,
    )


# --------------------------------------------------------------------------
# 10-11. 投票规则
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ((0.8, 0.1, 0.1), "support"),
        ((0.1, 0.8, 0.1), "refute"),
        ((0.1, 0.1, 0.8), "unknown"),
        ((0.5, 0.5, 0.0), "unknown"),  # 内部平局
        ((0.5, 0.0, 0.5), "unknown"),
        ((0.0, 0.5, 0.5), "unknown"),
        ((1 / 3, 1 / 3, 1 / 3), "unknown"),
    ],
)
def test_cast_vote(probabilities: tuple[float, float, float], expected: str) -> None:
    """单条预测投票：最大者得票，内部平局投 unknown。"""
    assert cast_vote(_prediction("d1", probabilities)) == expected


def test_vote_counting() -> None:
    """票数比例计算正确。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0), ("d3", 1.0), ("d4", 1.0)])
    predictions = [
        _prediction("d1", (0.8, 0.1, 0.1)),
        _prediction("d2", (0.7, 0.2, 0.1)),
        _prediction("d3", (0.1, 0.8, 0.1)),
        _prediction("d4", (0.1, 0.1, 0.8)),
    ]

    result = predict_majority_vote(sample, CLAIM, predictions, THRESHOLDS)

    assert result.score_support == pytest.approx(0.5)
    assert result.score_refute == pytest.approx(0.25)
    assert result.score_unknown == pytest.approx(0.25)
    assert result.predicted_state is EvidenceState.SUPPORTED
    assert result.method is BaselineMethod.MAJORITY_VOTE
    assert result.evaluator is None
    assert result.input_count == 4


def test_internal_tie_counts_as_unknown() -> None:
    """单条预测内部平局计为 unknown 票。"""
    sample = _sample([("d1", 1.0)])

    result = predict_majority_vote(
        sample, CLAIM, [_prediction("d1", (0.5, 0.5, 0.0))], THRESHOLDS
    )

    assert result.score_unknown == pytest.approx(1.0)
    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is BaselineDecisionReason.UNKNOWN_HIGHEST


# --------------------------------------------------------------------------
# 12. 不使用任何可靠性
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reliability", [0.0, 0.1, 0.5, 1.0])
def test_document_reliability_is_ignored(reliability: float) -> None:
    """改变文档可靠性完全不影响投票结果。"""
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05)),
        _prediction("d2", (0.8, 0.1, 0.1)),
    ]
    baseline = predict_majority_vote(
        _sample([("d1", 1.0), ("d2", 1.0)]), CLAIM, predictions, THRESHOLDS
    )
    variant = predict_majority_vote(
        _sample([("d1", reliability), ("d2", reliability)]),
        CLAIM,
        predictions,
        THRESHOLDS,
    )

    assert variant.score_support == pytest.approx(baseline.score_support)
    assert variant.predicted_state is baseline.predicted_state


@pytest.mark.parametrize("evaluator_reliability", [0.0, 0.3, 1.0])
def test_evaluator_reliability_is_ignored(evaluator_reliability: float) -> None:
    """改变评估器可靠性也不影响投票结果。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0)])
    baseline = predict_majority_vote(
        sample,
        CLAIM,
        [_prediction("d1", (0.9, 0.05, 0.05)), _prediction("d2", (0.8, 0.1, 0.1))],
        THRESHOLDS,
    )
    variant = predict_majority_vote(
        sample,
        CLAIM,
        [
            _prediction("d1", (0.9, 0.05, 0.05)),
            _prediction(
                "d2", (0.8, 0.1, 0.1), evaluator_reliability=evaluator_reliability
            ),
        ],
        THRESHOLDS,
    )

    assert variant.score_support == pytest.approx(baseline.score_support)


def test_weighted_average_and_majority_vote_can_disagree() -> None:
    """一票一权与可靠性加权可以给出不同结论 —— 证明投票确实没用可靠性。"""
    from rag_ds.baselines.weighted_average import predict_weighted_average

    sample = _sample([("d1", 1.0), ("d2", 0.01), ("d3", 0.01)])
    predictions = [
        _prediction("d1", (0.95, 0.025, 0.025)),
        _prediction("d2", (0.025, 0.95, 0.025)),
        _prediction("d3", (0.025, 0.95, 0.025)),
    ]

    votes = predict_majority_vote(sample, CLAIM, predictions, THRESHOLDS)
    weighted = predict_weighted_average(sample, CLAIM, predictions, THRESHOLDS)

    assert votes.predicted_state is EvidenceState.REFUTED  # 2 : 1 反驳票胜
    assert weighted.predicted_state is EvidenceState.SUPPORTED  # 高可靠文档占压倒权重


# --------------------------------------------------------------------------
# 13. 投票平局与无票
# --------------------------------------------------------------------------


def test_vote_tie_gives_insufficient() -> None:
    """投票平局输出 insufficient / score_tie。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0)])
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05)),
        _prediction("d2", (0.05, 0.9, 0.05)),
    ]

    result = predict_majority_vote(sample, CLAIM, predictions, THRESHOLDS)

    assert result.score_support == pytest.approx(0.5)
    assert result.score_refute == pytest.approx(0.5)
    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is BaselineDecisionReason.SCORE_TIE
    assert result.predicted_state is not EvidenceState.CONFLICTING


def test_no_votes_gives_no_evidence() -> None:
    """没有任何预测时输出 no_evidence。"""
    result = predict_majority_vote(_sample([]), CLAIM, [], THRESHOLDS)

    assert result.score_unknown == pytest.approx(1.0)
    assert result.reason is BaselineDecisionReason.NO_EVIDENCE
    assert result.input_count == 0


@pytest.mark.parametrize(
    "gold_state",
    [None, EvidenceState.SUPPORTED, EvidenceState.CONFLICTING],
)
def test_gold_state_does_not_affect_the_result(
    gold_state: EvidenceState | None,
) -> None:
    """gold_state 只被带走，不影响任何分数。"""
    result = predict_majority_vote(
        _sample([("d1", 1.0)], gold_state=gold_state),
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1))],
        THRESHOLDS,
    )

    assert result.score_support == pytest.approx(1.0)
    assert result.gold_state is gold_state


def test_inputs_are_not_modified() -> None:
    """不修改传入对象。"""
    sample = _sample([("d1", 0.9)])
    predictions = [_prediction("d1", (0.8, 0.1, 0.1))]
    before = (sample.model_dump(), [p.model_dump() for p in predictions])

    predict_majority_vote(sample, CLAIM, predictions, THRESHOLDS)

    assert (sample.model_dump(), [p.model_dump() for p in predictions]) == before
