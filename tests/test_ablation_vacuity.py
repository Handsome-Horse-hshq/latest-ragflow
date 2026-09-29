"""消融空转检测测试。

CLIMATE-FEVER v1 里所有 ``reliability`` 都是 1.0，``no_reliability`` 因此退化
成恒等变换，Δ 必然为 0。这个 0 说明的是**消融没跑起来**，不是「可靠性折扣没有
作用」—— 两者在 CSV 里长得一模一样，所以必须有一个字段把它们区分开。
"""

from __future__ import annotations

import pytest

from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.ablation import AblationVariant, run_ablation
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction

VARIANTS = (AblationVariant.FULL, AblationVariant.NO_RELIABILITY)


def _sample(sample_id: str, gold: EvidenceState, reliability: float) -> RAGSample:
    return RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=(
            ContextChunk(doc_id=f"{sample_id}-d1", text="文档一。", reliability=reliability),
            ContextChunk(doc_id=f"{sample_id}-d2", text="文档二。", reliability=reliability),
        ),
        gold_state=gold,
    )


def _prediction(
    sample_id: str, doc: str, triple: tuple[float, float, float], reliability: float
) -> RelationPrediction:
    return RelationPrediction(
        sample_id=sample_id,
        claim_id=f"{sample_id}-c1",
        doc_id=f"{sample_id}-{doc}",
        evaluator="e1",
        p_support=triple[0],
        p_refute=triple[1],
        p_unknown=triple[2],
        evaluator_reliability=reliability,
    )


def _dataset(reliability: float):
    samples = [
        _sample("s1", EvidenceState.SUPPORTED, reliability),
        _sample("s2", EvidenceState.REFUTED, reliability),
    ]
    predictions = [
        _prediction("s1", "d1", (0.8, 0.1, 0.1), reliability),
        _prediction("s1", "d2", (0.7, 0.2, 0.1), reliability),
        _prediction("s2", "d1", (0.1, 0.8, 0.1), reliability),
        _prediction("s2", "d2", (0.2, 0.7, 0.1), reliability),
    ]
    return samples, predictions


class TestVacuityFlag:
    def test_all_reliabilities_one_makes_the_variant_vacuous(self) -> None:
        """这正是 climate_fever_v1 的情形。"""
        samples, predictions = _dataset(1.0)

        results = run_ablation(
            samples, predictions, DiagnosticThresholds(), variants=VARIANTS
        )

        stripped = next(r for r in results if r.variant is AblationVariant.NO_RELIABILITY)
        assert stripped.is_vacuous is True
        assert stripped.macro_f1_delta == pytest.approx(0.0)

    def test_varying_reliability_is_not_vacuous(self) -> None:
        samples, predictions = _dataset(0.6)

        results = run_ablation(
            samples, predictions, DiagnosticThresholds(), variants=VARIANTS
        )

        stripped = next(r for r in results if r.variant is AblationVariant.NO_RELIABILITY)
        assert stripped.is_vacuous is False

    def test_full_variant_is_never_flagged(self) -> None:
        samples, predictions = _dataset(1.0)

        results = run_ablation(
            samples, predictions, DiagnosticThresholds(), variants=VARIANTS
        )

        full = next(r for r in results if r.variant is AblationVariant.FULL)
        assert full.is_vacuous is False

    def test_live_dataset_reliabilities_are_all_one(self) -> None:
        """钉住这个事实：一旦数据集引入可靠性差异，这个测试会提醒更新论文表述。"""
        from pathlib import Path

        from rag_ds.data_io import load_samples

        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "climate_fever_v1"
            / "test.jsonl"
        )
        if not path.is_file():
            pytest.skip("需要已构建的 climate_fever_v1 数据集")

        reliabilities = {
            chunk.reliability for sample in load_samples(path) for chunk in sample.contexts
        }

        assert reliabilities == {1.0}
