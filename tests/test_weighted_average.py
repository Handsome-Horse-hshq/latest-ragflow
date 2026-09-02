"""第十一阶段 Weighted Average baseline 的测试。"""

from __future__ import annotations

import pytest

from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
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

THRESHOLDS = BaselineThresholds()
CLAIM = Claim(claim_id="c1", text="断言。")


def _sample(
    docs: list[tuple[str, float, float | None]],
    gold_state: EvidenceState | None = None,
) -> RAGSample:
    """构造样本；``docs`` 为 ``(doc_id, reliability, retrieval_score)``。"""
    return RAGSample(
        sample_id="s1",
        question="问题？",
        answer="答案。",
        claims=[CLAIM],
        contexts=[
            ContextChunk(
                doc_id=doc_id,
                text=f"文档 {doc_id}。",
                reliability=reliability,
                retrieval_score=retrieval_score,
            )
            for doc_id, reliability, retrieval_score in docs
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
# 7-8. 加权平均公式
# --------------------------------------------------------------------------


def test_single_prediction_reproduces_the_probabilities() -> None:
    """只有一条预测时，分数就是原始概率（权重被约掉）。"""
    sample = _sample([("d1", 0.7, None)])

    result = predict_weighted_average(
        sample, CLAIM, [_prediction("d1", (0.8, 0.1, 0.1))], THRESHOLDS
    )

    assert result.score_support == pytest.approx(0.8)
    assert result.score_refute == pytest.approx(0.1)
    assert result.score_unknown == pytest.approx(0.1)
    assert result.method is BaselineMethod.WEIGHTED_AVERAGE
    assert result.evaluator is None
    assert result.input_count == 1


def test_weighted_average_formula() -> None:
    """手工复算加权平均。"""
    sample = _sample([("d1", 0.9, None), ("d2", 0.5, None)])
    predictions = [
        _prediction("d1", (0.8, 0.1, 0.1)),
        _prediction("d2", (0.2, 0.7, 0.1)),
    ]

    result = predict_weighted_average(sample, CLAIM, predictions, THRESHOLDS)

    total = 0.9 + 0.5
    assert result.score_support == pytest.approx((0.9 * 0.8 + 0.5 * 0.2) / total)
    assert result.score_refute == pytest.approx((0.9 * 0.1 + 0.5 * 0.7) / total)
    assert result.score_unknown == pytest.approx((0.9 * 0.1 + 0.5 * 0.1) / total)


def test_weight_uses_document_times_evaluator_reliability() -> None:
    """权重是 文档可靠性 × 评估器可靠性。"""
    sample = _sample([("d1", 0.8, None), ("d2", 0.8, None)])
    predictions = [
        _prediction("d1", (1.0, 0.0, 0.0), evaluator="mock_a", evaluator_reliability=1.0),
        _prediction("d2", (0.0, 1.0, 0.0), evaluator="mock_b", evaluator_reliability=0.25),
    ]

    result = predict_weighted_average(sample, CLAIM, predictions, THRESHOLDS)

    weight_a, weight_b = 0.8 * 1.0, 0.8 * 0.25
    assert result.score_support == pytest.approx(weight_a / (weight_a + weight_b))
    assert result.score_refute == pytest.approx(weight_b / (weight_a + weight_b))


def test_lower_evaluator_reliability_shifts_the_result() -> None:
    """降低某评估器的可靠性会让结果向另一方倾斜。"""
    sample = _sample([("d1", 1.0, None), ("d2", 1.0, None)])
    equal = predict_weighted_average(
        sample,
        CLAIM,
        [
            _prediction("d1", (0.9, 0.05, 0.05), evaluator="mock_a"),
            _prediction("d2", (0.05, 0.9, 0.05), evaluator="mock_b"),
        ],
        THRESHOLDS,
    )
    skewed = predict_weighted_average(
        sample,
        CLAIM,
        [
            _prediction("d1", (0.9, 0.05, 0.05), evaluator="mock_a"),
            _prediction(
                "d2", (0.05, 0.9, 0.05), evaluator="mock_b", evaluator_reliability=0.1
            ),
        ],
        THRESHOLDS,
    )

    assert skewed.score_support > equal.score_support
    assert skewed.score_refute < equal.score_refute


# --------------------------------------------------------------------------
# 9. 零权重
# --------------------------------------------------------------------------


def test_zero_total_weight_gives_no_evidence() -> None:
    """总权重为零时输出 (0, 0, 1) 与 no_evidence。"""
    sample = _sample([("d1", 0.0, None)])

    result = predict_weighted_average(
        sample, CLAIM, [_prediction("d1", (0.8, 0.1, 0.1))], THRESHOLDS
    )

    assert result.score_support == pytest.approx(0.0)
    assert result.score_refute == pytest.approx(0.0)
    assert result.score_unknown == pytest.approx(1.0)
    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.reason is BaselineDecisionReason.NO_EVIDENCE


def test_no_predictions_gives_no_evidence() -> None:
    """没有任何预测时同样是 no_evidence。"""
    result = predict_weighted_average(_sample([]), CLAIM, [], THRESHOLDS)

    assert result.reason is BaselineDecisionReason.NO_EVIDENCE
    assert result.input_count == 0


def test_zero_evaluator_reliability_also_gives_no_evidence() -> None:
    """评估器可靠性为零同样使总权重归零。"""
    sample = _sample([("d1", 0.9, None)])

    result = predict_weighted_average(
        sample,
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1), evaluator_reliability=0.0)],
        THRESHOLDS,
    )

    assert result.reason is BaselineDecisionReason.NO_EVIDENCE


# --------------------------------------------------------------------------
# 17-20. 无关字段与冲突压缩
# --------------------------------------------------------------------------


@pytest.mark.parametrize("retrieval_score", [None, 0.0, 0.5, 1.0])
def test_retrieval_score_does_not_affect_the_result(
    retrieval_score: float | None,
) -> None:
    """retrieval_score 不参与计算。"""
    baseline = predict_weighted_average(
        _sample([("d1", 0.9, None)]),
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1))],
        THRESHOLDS,
    )
    variant = predict_weighted_average(
        _sample([("d1", 0.9, retrieval_score)]),
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1))],
        THRESHOLDS,
    )

    assert variant.score_support == pytest.approx(baseline.score_support)
    assert variant.predicted_state is baseline.predicted_state


@pytest.mark.parametrize(
    "gold_state",
    [None, EvidenceState.SUPPORTED, EvidenceState.REFUTED, EvidenceState.CONFLICTING],
)
def test_gold_state_does_not_affect_the_result(
    gold_state: EvidenceState | None,
) -> None:
    """改变 gold_state 不改变任何分数或结论，只改变被带走的标注。"""
    result = predict_weighted_average(
        _sample([("d1", 0.9, None)], gold_state=gold_state),
        CLAIM,
        [_prediction("d1", (0.8, 0.1, 0.1))],
        THRESHOLDS,
    )

    assert result.score_support == pytest.approx(0.8)
    assert result.predicted_state is EvidenceState.SUPPORTED
    assert result.gold_state is gold_state


def test_opposing_documents_are_compressed_not_flagged_as_conflict() -> None:
    """两条针锋相对的文档被压成 insufficient，而不是 conflicting。

    这正是 baseline 的固有局限：D-S 侧会用 K_doc 明确标出文档冲突。
    """
    sample = _sample(
        [("d1", 1.0, None), ("d2", 1.0, None)], gold_state=EvidenceState.CONFLICTING
    )
    predictions = [
        _prediction("d1", (0.9, 0.05, 0.05)),
        _prediction("d2", (0.05, 0.9, 0.05)),
    ]

    result = predict_weighted_average(sample, CLAIM, predictions, THRESHOLDS)

    assert result.score_support == pytest.approx(0.475)
    assert result.score_refute == pytest.approx(0.475)
    assert result.score_unknown == pytest.approx(0.05)
    assert result.predicted_state is EvidenceState.INSUFFICIENT
    assert result.predicted_state is not EvidenceState.CONFLICTING
    assert result.reason is BaselineDecisionReason.BELOW_THRESHOLD
    assert result.gold_state is EvidenceState.CONFLICTING  # 标注仍是冲突


def test_inputs_are_not_modified() -> None:
    """不修改传入的样本与预测。"""
    sample = _sample([("d1", 0.9, 0.5)])
    predictions = [_prediction("d1", (0.8, 0.1, 0.1))]
    before = (sample.model_dump(), [p.model_dump() for p in predictions])

    predict_weighted_average(sample, CLAIM, predictions, THRESHOLDS)

    assert (sample.model_dump(), [p.model_dump() for p in predictions]) == before
