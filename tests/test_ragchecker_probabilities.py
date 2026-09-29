"""连续概率摄取路径测试。

离散标签会把诊断空间压成有限格点：每篇文档只有 3 个标签、共 n 篇文档，而
Dempster 组合与 ``K_doc = 1 - prod(1 - K_i)`` 都与顺序无关，于是结果只取决于
标签计数，最多 ``C(n+2, 2)`` 种。连续概率没有这个问题 —— 这组测试钉住的就是
「同样的输出，连续路径能分出更多不同取值」这件事。
"""

from __future__ import annotations

import pytest

from rag_ds.ragchecker_ingest import (
    NLI_LABEL_ORDER,
    RAGCheckerOutputError,
    distributions_to_predictions,
    parse_checking_output_probabilities,
    parse_checking_outputs,
)
from rag_ds.relation_evaluation.ragchecker_adapter import DEFAULT_LABEL_MAPPING
from rag_ds.schemas import Claim, ContextChunk, RAGSample


def _sample(sample_id: str = "s1", doc_count: int = 2) -> RAGSample:
    return RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=tuple(
            ContextChunk(doc_id=f"{sample_id}-d{i}", text=f"文档 {i}。")
            for i in range(1, doc_count + 1)
        ),
    )


def _payload(
    sample: RAGSample,
    labels: list[list[str]],
    probabilities: list[list[list[float]]] | None,
) -> dict:
    record = {
        "query_id": sample.sample_id,
        "query": sample.question,
        "gt_answer": "[PLACEHOLDER]",
        "response": sample.answer,
        "retrieved_context": [
            {"doc_id": c.doc_id, "text": c.text} for c in sample.contexts
        ],
        "retrieved2response": labels,
    }
    if probabilities is not None:
        record["retrieved2response_probabilities"] = probabilities
    return {"results": [record]}


class TestLabelOrderMapping:
    """NLI 的类别顺序与本项目三元组顺序不同，中间两项必须换位。"""

    def test_label_order_matches_refchecker(self) -> None:
        assert NLI_LABEL_ORDER == ("Entailment", "Neutral", "Contradiction")

    def test_neutral_becomes_unknown_not_refute(self) -> None:
        sample = _sample(doc_count=1)
        # Entailment=0.1, Neutral=0.7, Contradiction=0.2
        payload = _payload(sample, [["Neutral"]], [[[0.1, 0.7, 0.2]]])

        predictions = parse_checking_output_probabilities(
            payload, [sample], evaluator="nli"
        )

        assert predictions[0].p_support == pytest.approx(0.1)
        assert predictions[0].p_unknown == pytest.approx(0.7)
        assert predictions[0].p_refute == pytest.approx(0.2)

    def test_predictions_carry_evaluator_and_reliability(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(sample, [["Entailment"]], [[[0.8, 0.1, 0.1]]])

        predictions = parse_checking_output_probabilities(
            payload, [sample], evaluator="nli", evaluator_reliability=0.7
        )

        assert predictions[0].evaluator == "nli"
        assert predictions[0].evaluator_reliability == 0.7
        assert predictions[0].claim_id == "s1-c1"
        assert predictions[0].doc_id == "s1-d1"


class TestGranularity:
    """连续概率能分出的取值远多于离散标签。"""

    def test_same_labels_can_carry_different_probabilities(self) -> None:
        """两条组合标签都是 Neutral，但置信度不同 —— 离散路径看不出区别。"""
        sample = _sample(doc_count=2)
        payload = _payload(
            sample,
            [["Neutral", "Neutral"]],
            [[[0.05, 0.90, 0.05], [0.40, 0.45, 0.15]]],
        )

        continuous = parse_checking_output_probabilities(
            payload, [sample], evaluator="nli"
        )
        discrete = distributions_to_predictions(
            parse_checking_outputs(payload, [sample]),
            DEFAULT_LABEL_MAPPING,
            evaluator="nli",
        )

        assert continuous[0].p_support != continuous[1].p_support
        # 离散路径下两条完全相同 —— 这正是格点塌缩的来源。
        assert discrete[0].p_support == discrete[1].p_support


class TestMultipleSubClaims:
    def test_multiple_subclaims_are_averaged_per_document(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(
            sample,
            [["Entailment"], ["Contradiction"]],
            [[[0.8, 0.1, 0.1]], [[0.2, 0.2, 0.6]]],
        )

        predictions = parse_checking_output_probabilities(
            payload, [sample], evaluator="nli"
        )

        assert predictions[0].p_support == pytest.approx(0.5)
        assert predictions[0].p_unknown == pytest.approx(0.15)
        assert predictions[0].p_refute == pytest.approx(0.35)


class TestFailLoudly:
    def test_missing_probability_field_points_at_the_flag(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(sample, [["Neutral"]], None)

        with pytest.raises(RAGCheckerOutputError, match="--emit-probabilities"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_unnormalised_triple_is_rejected(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(sample, [["Neutral"]], [[[0.1, 0.2, 0.3]]])

        with pytest.raises(RAGCheckerOutputError, match="不归一"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_out_of_range_value_is_rejected(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(sample, [["Neutral"]], [[[-0.1, 0.9, 0.2]]])

        with pytest.raises(RAGCheckerOutputError, match="超出"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_wrong_row_length_is_rejected(self) -> None:
        sample = _sample(doc_count=2)
        payload = _payload(
            sample, [["Neutral", "Neutral"]], [[[0.1, 0.8, 0.1]]]
        )

        with pytest.raises(RAGCheckerOutputError, match="长度不是"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_non_triple_entry_is_rejected(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(sample, [["Neutral"]], [[[0.5, 0.5]]])

        with pytest.raises(RAGCheckerOutputError, match="不是三元组"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_subclaim_count_must_match_the_label_matrix(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(
            sample,
            [["Entailment"], ["Neutral"]],
            [[[0.8, 0.1, 0.1]]],
        )

        with pytest.raises(RAGCheckerOutputError, match="必须同源"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_alignment_checks_are_shared_with_the_label_path(self) -> None:
        """doc_id 对不上时，连续路径同样要报错。"""
        sample = _sample(doc_count=2)
        payload = _payload(
            sample,
            [["Neutral", "Neutral"]],
            [[[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]],
        )
        payload["results"][0]["retrieved_context"][1]["doc_id"] = "别的文档"

        with pytest.raises(RAGCheckerOutputError, match="doc_id 对不上"):
            parse_checking_output_probabilities(payload, [sample], evaluator="nli")

    def test_empty_evaluator_name_is_rejected(self) -> None:
        sample = _sample(doc_count=1)
        payload = _payload(sample, [["Neutral"]], [[[0.1, 0.8, 0.1]]])

        with pytest.raises(ValueError, match="evaluator 名称不能为空"):
            parse_checking_output_probabilities(payload, [sample], evaluator="  ")
