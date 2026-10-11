"""Statistic Gate 的「只数证据篇数」对照。

它不是一个冲突统计量，而是一把尺子：证据篇数不固定的数据集上，量出「完全不看
文档内容、只数有几篇」能拿到多少分。
"""

from __future__ import annotations

import pytest

from rag_ds.baselines.stat_gate import (
    ALL_STAT_GATE_VARIANTS,
    STAT_GATE_CONTROLS,
    STAT_GATE_VARIANTS,
    ConflictStatistic,
    decide_stat_gate,
    stat_gate_scores,
)
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.cross_validation import run_nested_cross_validation
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction


def _claim(sample_id: str, triples, gold=EvidenceState.SUPPORTED, evaluators=("e1",)):
    sample = RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=tuple(
            ContextChunk(doc_id=f"{sample_id}-d{n}", text=f"文档 {n}。")
            for n in range(1, len(triples) + 1)
        ),
        gold_state=gold,
    )
    predictions = [
        RelationPrediction(
            sample_id=sample_id, claim_id=f"{sample_id}-c1", doc_id=f"{sample_id}-d{n}",
            evaluator=evaluator, p_support=s, p_refute=r, p_unknown=u,
        )
        for evaluator in evaluators
        for n, (s, r, u) in enumerate(triples, start=1)
    ]
    return sample, predictions


class TestRegistry:
    def test_controls_are_kept_apart_from_the_four_real_variants(self) -> None:
        assert list(STAT_GATE_CONTROLS) == ["doc_count", "doc_count_uprod"]
        assert not set(STAT_GATE_CONTROLS) & set(STAT_GATE_VARIANTS)
        assert list(ALL_STAT_GATE_VARIANTS) == [*STAT_GATE_VARIANTS, *STAT_GATE_CONTROLS]
        assert all(
            v.conflict is not ConflictStatistic.DOCUMENT_COUNT
            for v in STAT_GATE_VARIANTS.values()
        )
        assert STAT_GATE_CONTROLS["doc_count"].method_name == "stat_gate_doc_count"


class TestScore:
    @pytest.mark.parametrize(("documents", "expected"), [(1, 0.0), (2, 0.5), (4, 0.75)])
    def test_only_the_number_of_documents_matters(self, documents: int, expected: float) -> None:
        agreeing, a_predictions = _claim("a", [(0.9, 0.0, 0.1)] * documents)
        opposing, o_predictions = _claim(
            "o", [(0.9, 0.0, 0.1), (0.0, 0.9, 0.1), (0.9, 0.0, 0.1), (0.0, 0.9, 0.1)][:documents]
        )

        (a,) = stat_gate_scores([agreeing], a_predictions, STAT_GATE_CONTROLS["doc_count"])
        (o,) = stat_gate_scores([opposing], o_predictions, STAT_GATE_CONTROLS["doc_count"])

        # 文档互相矛盾与否，对这个对照没有任何影响。
        assert a.conflict == o.conflict == pytest.approx(expected)

    def test_a_document_judged_by_two_evaluators_counts_once(self) -> None:
        sample, predictions = _claim("s", [(0.9, 0.0, 0.1)] * 2, evaluators=("e1", "e2"))

        (scores,) = stat_gate_scores([sample], predictions, STAT_GATE_CONTROLS["doc_count"])

        assert scores.input_count == 4
        assert scores.conflict == pytest.approx(0.5)

    def test_gate_calls_any_claim_with_enough_documents_conflicting(self) -> None:
        sample, predictions = _claim("s", [(0.9, 0.0, 0.1)] * 4)
        (scores,) = stat_gate_scores([sample], predictions, STAT_GATE_CONTROLS["doc_count"])
        thresholds = DiagnosticThresholds(theta_threshold=0.9, document_conflict_threshold=0.7)

        # 四篇完全一致的支持文档也会被判成「冲突」：这正说明它只是一把尺子。
        assert decide_stat_gate(scores, thresholds) is EvidenceState.CONFLICTING


def test_control_runs_inside_the_nested_cross_validation() -> None:
    """冲突类 claim 的证据篇数更多、内容却毫无矛盾时，只数篇数就能把它们找出来。"""
    samples, predictions = [], []
    plan = (
        (EvidenceState.CONFLICTING, [(0.5, 0.1, 0.4)] * 4),
        (EvidenceState.SUPPORTED, [(0.7, 0.1, 0.2)] * 2),
        (EvidenceState.REFUTED, [(0.1, 0.7, 0.2)] * 2),
        (EvidenceState.INSUFFICIENT, [(0.05, 0.05, 0.9)] * 2),
    )
    for index in range(10):
        for offset, (gold, triples) in enumerate(plan):
            sample, preds = _claim(f"s{index * 4 + offset:03d}", triples, gold)
            samples.append(sample)
            predictions.extend(preds)

    result = run_nested_cross_validation(
        samples, predictions, "e1", n_folds=5, seed=42,
        stat_gate_statistics=[STAT_GATE_CONTROLS["doc_count"]],
    )
    by_claim = {
        (r.sample_id, r.claim_id): r
        for r in result.predictions
        if r.method == "stat_gate_doc_count"
    }

    assert len(by_claim) == len(samples)
    conflicting = [r for r in by_claim.values() if r.gold_label == "conflicting"]
    assert all(r.predicted_label == "conflicting" for r in conflicting)
