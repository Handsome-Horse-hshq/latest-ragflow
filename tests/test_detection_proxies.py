"""检测实验公平对照的测试：统计量的定义与配对 AUROC 自助检验。"""

from __future__ import annotations

import math

import pytest

from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.detection_proxies import (
    CONFLICT_PROXIES,
    INSUFFICIENCY_PROXIES,
    compare_detection,
    detection_scores,
)
from rag_ds.metrics.detection import paired_auroc_bootstrap
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction

EVALUATOR = "e1"


def _sample(sample_id: str, gold: EvidenceState) -> RAGSample:
    return RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=(
            ContextChunk(doc_id=f"{sample_id}-d1", text="文档一。"),
            ContextChunk(doc_id=f"{sample_id}-d2", text="文档二。"),
        ),
        gold_state=gold,
    )


def _predictions(sample_id: str, first, second) -> list[RelationPrediction]:
    return [
        RelationPrediction(
            sample_id=sample_id,
            claim_id=f"{sample_id}-c1",
            doc_id=f"{sample_id}-d{index}",
            evaluator=EVALUATOR,
            p_support=triple[0],
            p_refute=triple[1],
            p_unknown=triple[2],
        )
        for index, triple in enumerate((first, second), start=1)
    ]


def _dataset():
    triples = {
        EvidenceState.SUPPORTED: ((0.6, 0.1, 0.3), (0.5, 0.1, 0.4)),
        EvidenceState.REFUTED: ((0.1, 0.6, 0.3), (0.1, 0.5, 0.4)),
        EvidenceState.INSUFFICIENT: ((0.1, 0.1, 0.8), (0.05, 0.05, 0.9)),
        EvidenceState.CONFLICTING: ((0.7, 0.2, 0.1), (0.2, 0.7, 0.1)),
    }
    samples, predictions = [], []
    for index, (gold, (first, second)) in enumerate(triples.items()):
        sample_id = f"s{index}"
        samples.append(_sample(sample_id, gold))
        predictions.extend(_predictions(sample_id, first, second))
    return samples, predictions


class TestDetectionScores:
    def test_proxy_definitions(self) -> None:
        samples, predictions = _dataset()

        rows = {r.gold_label: r for r in detection_scores(samples, predictions)}
        conflicting = rows["conflicting"]
        insufficient = rows["insufficient"]

        assert conflicting.conflict["one_minus_abs_margin"] == pytest.approx(1.0)
        assert conflicting.conflict["min_mean"] == pytest.approx(0.45)
        assert conflicting.conflict["mean_product"] == pytest.approx(0.45 * 0.45)
        assert conflicting.conflict["max_product"] == pytest.approx(0.49)
        assert conflicting.conflict["min_max"] == pytest.approx(0.7)
        assert conflicting.conflict["sr_std"] == pytest.approx(0.5)
        assert insufficient.insufficiency["mean_unknown"] == pytest.approx(0.85)
        assert insufficient.insufficiency["min_unknown"] == pytest.approx(0.8)
        assert insufficient.insufficiency["unknown_product"] == pytest.approx(0.72)
        assert set(conflicting.conflict) == set(CONFLICT_PROXIES)
        assert set(insufficient.insufficiency) == set(INSUFFICIENCY_PROXIES)

    def test_ds_scores_match_the_pipeline(self) -> None:
        samples, predictions = _dataset()

        rows = detection_scores(samples, predictions)
        pipeline = run_pipeline(samples, predictions, DiagnosticThresholds())

        for row, result in zip(rows, pipeline):
            assert row.k_doc == result.diagnostic.k_doc
            assert row.m_theta == result.diagnostic.m_theta

    def test_old_proxy_mistakes_ignorance_for_conflict(self) -> None:
        """全部文档都说不清时，旧代理 1-|s-r| 也接近 1 —— 这是换掉它的原因。"""
        samples, predictions = _dataset()

        rows = {r.gold_label: r for r in detection_scores(samples, predictions)}

        assert rows["insufficient"].conflict["one_minus_abs_margin"] == pytest.approx(1.0)
        assert rows["insufficient"].conflict["sr_std"] == pytest.approx(0.0)

    def test_compare_detection_covers_both_tasks(self) -> None:
        samples, predictions = _dataset()

        result = compare_detection(
            detection_scores(samples, predictions), n_resamples=50, seed=1
        )

        assert set(result) == {"conflict", "insufficiency"}
        assert [r.score_name for r in result["conflict"].reports] == [
            "k_doc",
            *CONFLICT_PROXIES,
        ]
        assert [c.rival for c in result["insufficiency"].comparisons] == list(
            INSUFFICIENCY_PROXIES
        )


class TestPairedAurocBootstrap:
    LABELS = [True, False] * 20
    SCORES = [0.9 - 0.01 * i if i % 2 == 0 else 0.1 + 0.01 * i for i in range(40)]

    def test_identical_scores_give_zero_delta(self) -> None:
        (comparison,) = paired_auroc_bootstrap(
            self.LABELS, self.SCORES, {"same": self.SCORES}, "ref", "pos",
            n_resamples=200, seed=3,
        )

        assert comparison.observed_delta == 0.0
        assert (comparison.ci_low, comparison.ci_high) == (0.0, 0.0)
        assert comparison.p_value == 1.0
        assert not comparison.is_significant

    def test_reversed_scores_are_significantly_worse(self) -> None:
        reversed_scores = [1.0 - s for s in self.SCORES]

        (comparison,) = paired_auroc_bootstrap(
            self.LABELS, self.SCORES, {"reversed": reversed_scores}, "ref", "pos",
            n_resamples=200, seed=3,
        )

        assert comparison.auroc_reference == pytest.approx(1.0)
        assert comparison.auroc_rival == pytest.approx(0.0)
        assert comparison.is_significant
        assert comparison.ci_low > 0.0

    def test_single_class_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="正负两类"):
            paired_auroc_bootstrap([True] * 4, [0.1] * 4, {"x": [0.2] * 4}, "ref", "pos")

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="长度"):
            paired_auroc_bootstrap(
                [True, False], [0.1, 0.2], {"x": [0.2]}, "ref", "pos"
            )


def test_unknown_product_uses_discounted_unknown() -> None:
    """文档可靠性 0.5 时不确定被折扣抬高：1 - 0.5 × (1 - 0.8) = 0.9。"""
    sample = RAGSample(
        sample_id="s",
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id="s-c1", text="答案。"),),
        contexts=(
            ContextChunk(doc_id="s-d1", text="文档一。", reliability=0.5),
            ContextChunk(doc_id="s-d2", text="文档二。"),
        ),
        gold_state=EvidenceState.INSUFFICIENT,
    )
    predictions = _predictions("s", (0.1, 0.1, 0.8), (0.1, 0.1, 0.8))

    (row,) = detection_scores([sample], predictions)

    assert row.insufficiency["unknown_product"] == pytest.approx(0.9 * 0.8)
    assert row.insufficiency["min_unknown"] == pytest.approx(0.8)
    assert math.isclose(row.insufficiency["mean_unknown"], 0.8)
