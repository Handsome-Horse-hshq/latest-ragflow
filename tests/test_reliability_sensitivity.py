"""可靠性敏感性分析测试。

``no_reliability`` 消融在本数据集上是恒等变换（所有 reliability 都是 1.0），
因此可靠性折扣这条链路只能靠敏感性分析来实际跑起来。这组测试钉住两件事：
折扣的**理论性质**（质量只从确定焦元流向 Theta）在整条链路上依然成立，
以及 oracle 设定必须被标记出来。
"""

from __future__ import annotations

import math

import pytest

from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.reliability_sensitivity import (
    MAX_VOTE_ENTROPY,
    apply_document_reliability,
    oracle_reliability_from_provenance,
    run_reliability_sensitivity,
    uniform_reliability_sweep,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction


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


def _prediction(sample_id: str, doc: str, triple: tuple[float, float, float]):
    return RelationPrediction(
        sample_id=sample_id,
        claim_id=f"{sample_id}-c1",
        doc_id=f"{sample_id}-{doc}",
        evaluator="e1",
        p_support=triple[0],
        p_refute=triple[1],
        p_unknown=triple[2],
    )


def _dataset():
    samples = [
        _sample("s1", EvidenceState.SUPPORTED),
        _sample("s2", EvidenceState.REFUTED),
        _sample("s3", EvidenceState.CONFLICTING),
    ]
    predictions = [
        _prediction("s1", "d1", (0.7, 0.2, 0.1)),
        _prediction("s1", "d2", (0.6, 0.2, 0.2)),
        _prediction("s2", "d1", (0.2, 0.7, 0.1)),
        _prediction("s2", "d2", (0.2, 0.6, 0.2)),
        _prediction("s3", "d1", (0.7, 0.2, 0.1)),
        _prediction("s3", "d2", (0.2, 0.7, 0.1)),
    ]
    return samples, predictions


THRESHOLDS = DiagnosticThresholds(
    theta_threshold=0.5, document_conflict_threshold=0.3
)


class TestApplyReliability:
    def test_uniform_value_replaces_every_document(self) -> None:
        samples, _ = _dataset()

        adjusted = apply_document_reliability(samples, 0.4)

        assert {c.reliability for s in adjusted for c in s.contexts} == {0.4}

    def test_mapping_only_touches_listed_documents(self) -> None:
        """映射里没有的 doc_id 保留原值，不会被悄悄当成 1.0。"""
        samples, _ = _dataset()
        samples = apply_document_reliability(samples, 0.5)

        adjusted = apply_document_reliability(samples, {"s1-d1": 0.2})

        by_id = {c.doc_id: c.reliability for s in adjusted for c in s.contexts}
        assert by_id["s1-d1"] == pytest.approx(0.2)
        assert by_id["s1-d2"] == pytest.approx(0.5)

    def test_original_samples_are_not_mutated(self) -> None:
        samples, _ = _dataset()

        apply_document_reliability(samples, 0.1)

        assert all(c.reliability == 1.0 for s in samples for c in s.contexts)


class TestUniformSweep:
    def test_m_theta_rises_monotonically_as_reliability_falls(self) -> None:
        """折扣只把质量从确定焦元移向 Theta，因此这条单调性必须成立。"""
        samples, predictions = _dataset()
        levels = (1.0, 0.8, 0.6, 0.4, 0.2)

        points = uniform_reliability_sweep(samples, predictions, THRESHOLDS, levels)

        thetas = [p.mean_m_theta for p in points]
        assert thetas == sorted(thetas)
        assert thetas[0] < thetas[-1]

    def test_k_doc_falls_monotonically_as_reliability_falls(self) -> None:
        samples, predictions = _dataset()
        levels = (1.0, 0.8, 0.6, 0.4, 0.2)

        points = uniform_reliability_sweep(samples, predictions, THRESHOLDS, levels)

        k_docs = [p.mean_k_doc for p in points]
        assert k_docs == sorted(k_docs, reverse=True)

    def test_baseline_point_has_zero_delta(self) -> None:
        samples, predictions = _dataset()

        points = uniform_reliability_sweep(
            samples, predictions, THRESHOLDS, (1.0, 0.5)
        )

        baseline = next(p for p in points if p.setting == "uniform_1.00")
        assert baseline.macro_f1_delta == pytest.approx(0.0)
        assert baseline.mean_reliability == pytest.approx(1.0)

    def test_no_point_is_flagged_as_oracle(self) -> None:
        samples, predictions = _dataset()

        points = uniform_reliability_sweep(samples, predictions, THRESHOLDS, (1.0, 0.5))

        assert not any(p.is_oracle for p in points)

    def test_levels_must_contain_the_baseline(self) -> None:
        samples, predictions = _dataset()

        with pytest.raises(ValueError, match="必须包含 1.0"):
            uniform_reliability_sweep(samples, predictions, THRESHOLDS, (0.8, 0.5))

    def test_empty_levels_are_rejected(self) -> None:
        samples, predictions = _dataset()

        with pytest.raises(ValueError, match="不能为空"):
            uniform_reliability_sweep(samples, predictions, THRESHOLDS, ())

    def test_out_of_range_levels_are_rejected(self) -> None:
        samples, predictions = _dataset()

        with pytest.raises(ValueError, match="必须位于"):
            uniform_reliability_sweep(samples, predictions, THRESHOLDS, (1.0, 1.5))


class TestOracleReliability:
    def test_zero_entropy_means_full_reliability(self) -> None:
        provenance = [
            {"sample_id": "s1", "contexts": [{"doc_id": "s1-d1", "entropy": 0.0}]}
        ]

        assert oracle_reliability_from_provenance(provenance)["s1-d1"] == pytest.approx(1.0)

    def test_max_entropy_means_zero_reliability(self) -> None:
        provenance = [
            {
                "sample_id": "s1",
                "contexts": [{"doc_id": "s1-d1", "entropy": MAX_VOTE_ENTROPY}],
            }
        ]

        assert oracle_reliability_from_provenance(provenance)["s1-d1"] == pytest.approx(0.0)

    def test_entropy_above_the_constant_is_clipped_not_negative(self) -> None:
        provenance = [
            {"sample_id": "s1", "contexts": [{"doc_id": "s1-d1", "entropy": 99.0}]}
        ]

        assert oracle_reliability_from_provenance(provenance)["s1-d1"] == 0.0

    def test_half_entropy_is_half_reliability(self) -> None:
        provenance = [
            {
                "sample_id": "s1",
                "contexts": [{"doc_id": "s1-d1", "entropy": math.log(3.0) / 2}],
            }
        ]

        assert oracle_reliability_from_provenance(provenance)["s1-d1"] == pytest.approx(0.5)

    def test_missing_entropy_is_rejected(self) -> None:
        provenance = [{"sample_id": "s1", "contexts": [{"doc_id": "s1-d1"}]}]

        with pytest.raises(ValueError, match="缺少 doc_id 或 entropy"):
            oracle_reliability_from_provenance(provenance)

    def test_missing_contexts_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="缺少 contexts"):
            oracle_reliability_from_provenance([{"sample_id": "s1"}])


class TestCombinedRun:
    def test_oracle_point_is_appended_and_flagged(self) -> None:
        samples, predictions = _dataset()
        oracle = {"s1-d1": 0.9, "s1-d2": 0.2, "s2-d1": 0.5, "s2-d2": 0.5,
                  "s3-d1": 0.8, "s3-d2": 0.3}

        points = run_reliability_sensitivity(
            samples, predictions, THRESHOLDS,
            levels=(1.0, 0.5), oracle_reliability=oracle,
        )

        assert points[-1].setting == "oracle_vote_agreement"
        assert points[-1].is_oracle is True
        assert sum(p.is_oracle for p in points) == 1

    def test_without_oracle_only_the_sweep_is_returned(self) -> None:
        samples, predictions = _dataset()

        points = run_reliability_sensitivity(
            samples, predictions, THRESHOLDS, levels=(1.0, 0.5)
        )

        assert [p.setting for p in points] == ["uniform_1.00", "uniform_0.50"]
