"""第十一阶段 Single Evaluator baseline 的测试。"""

from __future__ import annotations

import pytest

from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
    BaselineThresholds,
    MissingBaselineEvaluatorError,
)
from rag_ds.baselines.single_evaluator import predict_single_evaluator
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
# 14-15. 只用指定评估器 / 文档加权平均
# --------------------------------------------------------------------------


def test_only_the_named_evaluator_is_used() -> None:
    """另一个评估器的预测被完全忽略。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0)])
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05), evaluator="mock_a"),
        _prediction("d2", (0.05, 0.9, 0.05), evaluator="mock_b"),
    ]

    result = predict_single_evaluator(sample, CLAIM, predictions, "mock_a", THRESHOLDS)

    assert result.score_support == pytest.approx(0.9)
    assert result.score_refute == pytest.approx(0.05)
    assert result.predicted_state is EvidenceState.SUPPORTED
    assert result.method is BaselineMethod.SINGLE_EVALUATOR
    assert result.evaluator == "mock_a"
    assert result.input_count == 1  # 只数了 mock_a 的那一条


def test_switching_evaluator_switches_the_conclusion() -> None:
    """换一个评估器就得到相反结论。"""
    sample = _sample([("d1", 1.0), ("d2", 1.0)])
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05), evaluator="mock_a"),
        _prediction("d2", (0.05, 0.9, 0.05), evaluator="mock_b"),
    ]

    a = predict_single_evaluator(sample, CLAIM, predictions, "mock_a", THRESHOLDS)
    b = predict_single_evaluator(sample, CLAIM, predictions, "mock_b", THRESHOLDS)

    assert a.predicted_state is EvidenceState.SUPPORTED
    assert b.predicted_state is EvidenceState.REFUTED


def test_document_weighted_average_formula() -> None:
    """按文档可靠性加权平均，手工复算。"""
    sample = _sample([("d1", 0.9), ("d2", 0.4)])
    predictions = [
        _prediction("d1", (0.8, 0.1, 0.1)),
        _prediction("d2", (0.2, 0.7, 0.1)),
    ]

    result = predict_single_evaluator(sample, CLAIM, predictions, "mock_a", THRESHOLDS)

    total = 0.9 + 0.4
    assert result.score_support == pytest.approx((0.9 * 0.8 + 0.4 * 0.2) / total)
    assert result.score_refute == pytest.approx((0.9 * 0.1 + 0.4 * 0.7) / total)
    assert result.score_unknown == pytest.approx((0.9 * 0.1 + 0.4 * 0.1) / total)
    assert result.input_count == 2


@pytest.mark.parametrize("evaluator_reliability", [0.0, 0.2, 0.5, 1.0])
def test_evaluator_reliability_is_not_applied_per_document(
    evaluator_reliability: float,
) -> None:
    """evaluator_reliability 不在每条文档上重复施加，结果与它无关。"""
    sample = _sample([("d1", 0.9), ("d2", 0.4)])
    predictions = [
        _prediction("d1", (0.8, 0.1, 0.1), evaluator_reliability=evaluator_reliability),
        _prediction("d2", (0.2, 0.7, 0.1), evaluator_reliability=evaluator_reliability),
    ]
    reference = predict_single_evaluator(
        sample,
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1)), _prediction("d2", (0.2, 0.7, 0.1))],
        "mock_a",
        THRESHOLDS,
    )

    result = predict_single_evaluator(sample, CLAIM, predictions, "mock_a", THRESHOLDS)

    assert result.score_support == pytest.approx(reference.score_support)
    assert result.score_refute == pytest.approx(reference.score_refute)
    assert result.score_unknown == pytest.approx(reference.score_unknown)


# --------------------------------------------------------------------------
# 16. 评估器不存在
# --------------------------------------------------------------------------


def test_missing_evaluator_raises() -> None:
    """指定评估器不存在时报出专门的错误，不改用其他评估器。"""
    sample = _sample([("d1", 1.0)])
    predictions = [_prediction("d1", (0.8, 0.1, 0.1), evaluator="mock_a")]

    with pytest.raises(MissingBaselineEvaluatorError, match="mock_ghost"):
        predict_single_evaluator(sample, CLAIM, predictions, "mock_ghost", THRESHOLDS)


def test_missing_evaluator_error_reports_ids() -> None:
    """错误信息包含 sample_id 与 claim_id。"""
    sample = _sample([("d1", 1.0)])

    with pytest.raises(MissingBaselineEvaluatorError) as excinfo:
        predict_single_evaluator(
            sample, CLAIM, [_prediction("d1", (0.8, 0.1, 0.1))], "ghost", THRESHOLDS
        )

    message = str(excinfo.value)
    assert "s1" in message
    assert "c1" in message


def test_no_predictions_at_all_also_raises() -> None:
    """完全没有预测时同样报错，而不是返回 no_evidence。"""
    with pytest.raises(MissingBaselineEvaluatorError):
        predict_single_evaluator(_sample([]), CLAIM, [], "mock_a", THRESHOLDS)


# --------------------------------------------------------------------------
# 零权重与无关字段
# --------------------------------------------------------------------------


def test_zero_document_weight_gives_no_evidence() -> None:
    """文档权重之和为零时输出 (0, 0, 1) 与 no_evidence。"""
    sample = _sample([("d1", 0.0)])

    result = predict_single_evaluator(
        sample, CLAIM, [_prediction("d1", (0.8, 0.1, 0.1))], "mock_a", THRESHOLDS
    )

    assert result.score_unknown == pytest.approx(1.0)
    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is BaselineDecisionReason.NO_EVIDENCE
    assert result.evaluator == "mock_a"


@pytest.mark.parametrize(
    "gold_state",
    [None, EvidenceState.REFUTED, EvidenceState.CONFLICTING],
)
def test_gold_state_does_not_affect_the_result(
    gold_state: EvidenceState | None,
) -> None:
    """gold_state 只被带走。"""
    result = predict_single_evaluator(
        _sample([("d1", 0.9)], gold_state=gold_state),
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1))],
        "mock_a",
        THRESHOLDS,
    )

    assert result.score_support == pytest.approx(0.8)
    assert result.gold_state is gold_state


def test_opposing_documents_are_compressed() -> None:
    """同一评估器给出的相反文档同样被压成 insufficient。"""
    sample = _sample(
        [("d1", 1.0), ("d2", 1.0)], gold_state=EvidenceState.CONFLICTING
    )
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05)),
        _prediction("d2", (0.05, 0.9, 0.05)),
    ]

    result = predict_single_evaluator(sample, CLAIM, predictions, "mock_a", THRESHOLDS)

    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.predicted_state is not EvidenceState.CONFLICTING


def test_inputs_are_not_modified() -> None:
    """不修改传入对象。"""
    sample = _sample([("d1", 0.9)])
    predictions = [_prediction("d1", (0.8, 0.1, 0.1))]
    before = (sample.model_dump(), [p.model_dump() for p in predictions])

    predict_single_evaluator(sample, CLAIM, predictions, "mock_a", THRESHOLDS)

    assert (sample.model_dump(), [p.model_dump() for p in predictions]) == before
