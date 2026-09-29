"""RAGChecker 输出摄取测试：形状契约、claim 对齐与失败模式。"""

from __future__ import annotations

import pytest

from rag_ds.ragchecker_ingest import (
    ClaimLabelDistribution,
    RAGCheckerOutputError,
    distribution_to_probabilities,
    distributions_to_predictions,
    parse_checking_outputs,
)
from rag_ds.relation_evaluation.ragchecker_adapter import (
    DEFAULT_LABEL_MAPPING,
    LabelProbabilityMapping,
    RAGCheckerLabel,
)
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


def _payload(sample: RAGSample, matrix: list[list[str]]) -> dict:
    return {
        "results": [
            {
                "query_id": sample.sample_id,
                "query": sample.question,
                "gt_answer": "[PLACEHOLDER]",
                "response": sample.answer,
                "retrieved_context": [
                    {"doc_id": chunk.doc_id, "text": chunk.text}
                    for chunk in sample.contexts
                ],
                "retrieved2response": matrix,
            }
        ]
    }


class TestParseShape:
    """形状与对齐契约。"""

    def test_single_claim_maps_one_to_one(self) -> None:
        sample = _sample()
        payload = _payload(sample, [["Entailment", "Neutral"]])

        distributions = parse_checking_outputs(payload, [sample])

        assert [d.doc_id for d in distributions] == ["s1-d1", "s1-d2"]
        assert all(d.claim_id == "s1-c1" for d in distributions)
        assert distributions[0].entailment == 1
        assert distributions[1].neutral == 1

    def test_labels_are_case_insensitive(self) -> None:
        sample = _sample()
        payload = _payload(sample, [["entailment", "CONTRADICTION"]])

        distributions = parse_checking_outputs(payload, [sample])

        assert distributions[0].majority_label is RAGCheckerLabel.ENTAILMENT
        assert distributions[1].majority_label is RAGCheckerLabel.CONTRADICTION

    def test_multiple_subclaims_are_tallied_per_document(self) -> None:
        """外层是 claim、内层是文档；两维不能调换。"""
        sample = _sample()
        payload = _payload(
            sample,
            [["Entailment", "Neutral"], ["Entailment", "Contradiction"]],
        )

        distributions = parse_checking_outputs(payload, [sample])

        first, second = distributions
        assert (first.entailment, first.neutral, first.contradiction) == (2, 0, 0)
        assert (second.entailment, second.neutral, second.contradiction) == (0, 1, 1)
        assert second.majority_label is None  # 并列

    def test_row_length_must_match_document_count(self) -> None:
        sample = _sample(doc_count=2)
        payload = _payload(sample, [["Entailment"]])

        with pytest.raises(RAGCheckerOutputError, match="形状必须是"):
            parse_checking_outputs(payload, [sample])


class TestFailLoudly:
    """缺失与错位一律报错，绝不补全。"""

    def test_missing_sample_is_rejected(self) -> None:
        present, absent = _sample("s1"), _sample("s2")
        payload = _payload(present, [["Entailment", "Neutral"]])

        with pytest.raises(RAGCheckerOutputError, match="不会被补成 neutral"):
            parse_checking_outputs(payload, [present, absent])

    def test_extra_query_id_is_rejected(self) -> None:
        sample, stranger = _sample("s1"), _sample("s9")
        payload = _payload(sample, [["Entailment", "Neutral"]])
        payload["results"].extend(
            _payload(stranger, [["Neutral", "Neutral"]])["results"]
        )

        with pytest.raises(RAGCheckerOutputError, match="不属于本 split"):
            parse_checking_outputs(payload, [sample])

    def test_doc_id_mismatch_is_rejected(self) -> None:
        sample = _sample()
        payload = _payload(sample, [["Entailment", "Neutral"]])
        payload["results"][0]["retrieved_context"][1]["doc_id"] = "别的文档"

        with pytest.raises(RAGCheckerOutputError, match="doc_id 对不上"):
            parse_checking_outputs(payload, [sample])

    def test_empty_matrix_is_not_treated_as_neutral(self) -> None:
        sample = _sample()
        payload = _payload(sample, [])

        with pytest.raises(RAGCheckerOutputError, match="不会被当作 neutral"):
            parse_checking_outputs(payload, [sample])

    def test_missing_retrieved2response_points_at_the_metric(self) -> None:
        sample = _sample()
        payload = _payload(sample, [["Entailment", "Neutral"]])
        payload["results"][0]["retrieved2response"] = None

        with pytest.raises(RAGCheckerOutputError, match="faithfulness"):
            parse_checking_outputs(payload, [sample])

    def test_unknown_label_is_rejected(self) -> None:
        sample = _sample()
        payload = _payload(sample, [["Supported", "Neutral"]])

        with pytest.raises(RAGCheckerOutputError, match="未知标签"):
            parse_checking_outputs(payload, [sample])

    def test_strict_mode_rejects_multiple_subclaims(self) -> None:
        sample = _sample()
        payload = _payload(
            sample, [["Entailment", "Neutral"], ["Neutral", "Neutral"]]
        )

        with pytest.raises(RAGCheckerOutputError, match="严格一一对齐"):
            parse_checking_outputs(payload, [sample], strict_single_claim=True)

    def test_multi_claim_sample_is_rejected(self) -> None:
        sample = RAGSample(
            sample_id="s1",
            question="问题？",
            answer="答案。",
            claims=(
                Claim(claim_id="s1-c1", text="断言一。"),
                Claim(claim_id="s1-c2", text="断言二。"),
            ),
            contexts=(ContextChunk(doc_id="s1-d1", text="文档。"),),
        )
        payload = _payload(sample, [["Entailment"]])

        with pytest.raises(RAGCheckerOutputError, match="恰好一条 claim"):
            parse_checking_outputs(payload, [sample])


class TestProbabilities:
    """标签分布到三元概率的换算。"""

    def test_one_hot_equals_table_lookup(self) -> None:
        """单条子 claim 时，凸组合必须精确退化为查表。"""
        for label in RAGCheckerLabel:
            counts = {label.value: 1}
            distribution = ClaimLabelDistribution(
                sample_id="s", claim_id="c", doc_id="d", **counts
            )

            assert distribution_to_probabilities(
                distribution, DEFAULT_LABEL_MAPPING
            ) == pytest.approx(DEFAULT_LABEL_MAPPING.probabilities(label))

    def test_mixture_is_a_convex_combination(self) -> None:
        distribution = ClaimLabelDistribution(
            sample_id="s", claim_id="c", doc_id="d", entailment=1, contradiction=1
        )

        p_support, p_refute, p_unknown = distribution_to_probabilities(
            distribution, DEFAULT_LABEL_MAPPING
        )

        assert p_support == pytest.approx(0.475)
        assert p_refute == pytest.approx(0.475)
        assert p_unknown + p_support + p_refute == pytest.approx(1.0)

    def test_predictions_carry_evaluator_and_reliability(self) -> None:
        sample = _sample()
        payload = _payload(sample, [["Entailment", "Contradiction"]])
        distributions = parse_checking_outputs(payload, [sample])

        predictions = distributions_to_predictions(
            distributions,
            LabelProbabilityMapping(),
            evaluator="ragchecker",
            evaluator_reliability=0.8,
        )

        assert [p.evaluator for p in predictions] == ["ragchecker", "ragchecker"]
        assert all(p.evaluator_reliability == 0.8 for p in predictions)
        assert predictions[0].p_support == pytest.approx(0.90)
        assert predictions[1].p_refute == pytest.approx(0.90)

    def test_empty_evaluator_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="evaluator 名称不能为空"):
            distributions_to_predictions([], LabelProbabilityMapping(), evaluator="  ")
