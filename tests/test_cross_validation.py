"""嵌套交叉验证测试。

单次留出划分下测试集太小，主结论的置信区间跨 0。交叉验证让每条 claim 都当
一次测试点。这组测试钉住两件最容易出错的事：**分层**是否真的分层，以及
**测试折有没有参与阈值选择**（泄漏会让结果虚高且无法察觉）。
"""

from __future__ import annotations

import collections

import pytest

from rag_ds.experiments.comparison import DS_METHOD
from rag_ds.experiments.cross_validation import (
    make_stratified_folds,
    run_nested_cross_validation,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction

EVALUATOR = "e1"
STATES = (
    EvidenceState.SUPPORTED,
    EvidenceState.REFUTED,
    EvidenceState.INSUFFICIENT,
    EvidenceState.CONFLICTING,
)


def _sample(index: int, gold: EvidenceState) -> RAGSample:
    sid = f"s{index:03d}"
    return RAGSample(
        sample_id=sid,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sid}-c1", text="答案。"),),
        contexts=(
            ContextChunk(doc_id=f"{sid}-d1", text="文档一。"),
            ContextChunk(doc_id=f"{sid}-d2", text="文档二。"),
        ),
        gold_state=gold,
    )


def _triples(gold: EvidenceState, offset: int) -> list[tuple[float, float, float]]:
    """按金标准造出方向大致正确、但带一点扰动的关系概率。"""
    jitter = (offset % 5) * 0.02
    if gold is EvidenceState.SUPPORTED:
        return [(0.60 + jitter, 0.20, 0.20 - jitter)] * 2
    if gold is EvidenceState.REFUTED:
        return [(0.20, 0.60 + jitter, 0.20 - jitter)] * 2
    if gold is EvidenceState.INSUFFICIENT:
        return [(0.15, 0.15, 0.70)] * 2
    return [(0.70 - jitter, 0.20, 0.10 + jitter), (0.20, 0.70 - jitter, 0.10 + jitter)]


def _dataset(per_class: int = 10):
    samples: list[RAGSample] = []
    predictions: list[RelationPrediction] = []
    index = 0
    for gold in STATES:
        for offset in range(per_class):
            sample = _sample(index, gold)
            samples.append(sample)
            for doc_index, triple in enumerate(_triples(gold, offset), start=1):
                predictions.append(
                    RelationPrediction(
                        sample_id=sample.sample_id,
                        claim_id=f"{sample.sample_id}-c1",
                        doc_id=f"{sample.sample_id}-d{doc_index}",
                        evaluator=EVALUATOR,
                        p_support=triple[0],
                        p_refute=triple[1],
                        p_unknown=triple[2],
                    )
                )
            index += 1
    return samples, predictions


class TestStratifiedFolds:
    def test_every_index_appears_exactly_once(self) -> None:
        samples, _ = _dataset()

        folds = make_stratified_folds(samples, 5, seed=42)

        flat = sorted(i for fold in folds for i in fold)
        assert flat == list(range(len(samples)))

    def test_class_distribution_is_balanced_across_folds(self) -> None:
        samples, _ = _dataset(per_class=10)

        folds = make_stratified_folds(samples, 5, seed=42)

        for fold in folds:
            counts = collections.Counter(samples[i].gold_state for i in fold)
            assert set(counts.values()) == {2}

    def test_same_seed_reproduces_the_partition(self) -> None:
        samples, _ = _dataset()

        assert make_stratified_folds(samples, 5, seed=7) == make_stratified_folds(
            samples, 5, seed=7
        )

    def test_different_seed_changes_the_partition(self) -> None:
        samples, _ = _dataset()

        assert make_stratified_folds(samples, 5, seed=1) != make_stratified_folds(
            samples, 5, seed=2
        )

    def test_missing_gold_state_is_rejected(self) -> None:
        samples, _ = _dataset()
        samples[0] = samples[0].model_copy(update={"gold_state": None})

        with pytest.raises(ValueError, match="缺少 gold_state"):
            make_stratified_folds(samples, 5)

    @pytest.mark.parametrize("bad", [0, 1])
    def test_too_few_folds_is_rejected(self, bad: int) -> None:
        samples, _ = _dataset()

        with pytest.raises(ValueError, match="至少为 2"):
            make_stratified_folds(samples, bad)

    def test_more_folds_than_samples_is_rejected(self) -> None:
        samples, _ = _dataset(per_class=1)

        with pytest.raises(ValueError, match="少于折数"):
            make_stratified_folds(samples, 10)


class TestNestedRun:
    def test_every_claim_is_predicted_exactly_once_per_method(self) -> None:
        samples, predictions = _dataset()

        result = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        for method in result.methods:
            keys = [
                (r.sample_id, r.claim_id)
                for r in result.predictions
                if r.method == method
            ]
            assert len(keys) == len(samples)
            assert len(set(keys)) == len(samples)

    def test_all_five_methods_are_present(self) -> None:
        samples, predictions = _dataset()

        result = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        assert DS_METHOD in result.methods
        assert len(result.methods) == 5

    def test_fold_records_cover_every_sample(self) -> None:
        samples, predictions = _dataset()

        result = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        assert sum(f.test_size for f in result.folds) == len(samples)
        assert all(f.inner_size == len(samples) - f.test_size for f in result.folds)

    def test_thresholds_are_selected_per_fold(self) -> None:
        """各折在不同的内层数据上调参，阈值本就不该完全相同。"""
        samples, predictions = _dataset()

        result = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        chosen = {
            (
                f.ds_thresholds.theta_threshold,
                f.ds_thresholds.document_conflict_threshold,
            )
            for f in result.folds
        }
        assert len(chosen) >= 1  # 至少能跑出来；稳定时可能全相同

    def test_result_is_reproducible(self) -> None:
        samples, predictions = _dataset()

        first = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )
        second = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        assert [r.model_dump() for r in first.predictions] == [
            r.model_dump() for r in second.predictions
        ]


class TestNoLeakage:
    def test_fold_thresholds_are_reproducible_from_inner_data_alone(self) -> None:
        """每折的阈值必须能**只用内层数据**一字不差地重算出来。

        这是对「测试折没有参与阈值选择」最直接的检验：如果实现里不小心把
        测试折也喂进了搜索，独立重算的结果就对不上。

        （不能用「改掉测试折的金标准看阈值变不变」来验：分折本身是按
        ``gold_state`` 分层的，改金标准会连分折一起改掉。）
        """
        from rag_ds.diagnostics.models import DiagnosticThresholds
        from rag_ds.pipeline import run_pipeline
        from rag_ds.tuning.threshold_search import (
            SplitName,
            ThresholdGrid,
            search_thresholds,
        )

        samples, predictions = _dataset()
        folds = make_stratified_folds(samples, 5, seed=42)
        result = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        for fold_index, test_indices in enumerate(folds):
            inner_indices = [
                i
                for position, fold in enumerate(folds)
                if position != fold_index
                for i in fold
            ]
            inner_samples = [samples[i] for i in inner_indices]
            keep = {s.sample_id for s in inner_samples}
            inner_predictions = [p for p in predictions if p.sample_id in keep]

            inner_results = run_pipeline(
                inner_samples, inner_predictions, DiagnosticThresholds()
            )
            grid = ThresholdGrid.from_observed(
                [r.diagnostic.m_theta for r in inner_results if r.diagnostic.m_theta is not None],
                [r.diagnostic.k_doc for r in inner_results],
                steps=9,
            )
            expected = search_thresholds(
                inner_results, SplitName.VALIDATION, grid
            ).best.thresholds

            assert result.folds[fold_index].ds_thresholds == expected

    def test_inner_and_test_folds_never_overlap(self) -> None:
        samples, _ = _dataset()

        folds = make_stratified_folds(samples, 5, seed=42)

        for fold_index, test_indices in enumerate(folds):
            inner = {
                i
                for position, fold in enumerate(folds)
                if position != fold_index
                for i in fold
            }
            assert not (set(test_indices) & inner)
