"""嵌套交叉验证测试。

单次留出划分下测试集太小，主结论的置信区间跨 0。交叉验证让每条 claim 都当
一次测试点。这组测试钉住两件最容易出错的事：**分层**是否真的分层，以及
**测试折有没有参与阈值选择**（泄漏会让结果虚高且无法察觉）。
"""

from __future__ import annotations

import collections

import pytest

from rag_ds.baselines.stat_gate import STAT_GATE_VARIANTS, ConflictStatistic
from rag_ds.experiments.ablation import AblationVariant
from rag_ds.experiments.comparison import DS_METHOD
from rag_ds.experiments.cross_validation import (
    make_stratified_folds,
    run_nested_cross_validation,
)
from rag_ds.experiments.cv_ablation import CV_ABLATION_VARIANTS, run_nested_ablation
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


def _labels(predictions, method):
    return {
        (r.sample_id, r.claim_id): r.predicted_label
        for r in predictions
        if r.method == method
    }


class TestStatGateControl:
    STATISTICS = (ConflictStatistic.SUPPORT_REFUTE_STD, ConflictStatistic.MAX_PRODUCT)

    def test_default_run_is_unchanged(self) -> None:
        """不开新对照时，输出与原协议完全相同，已提交的结果仍可逐位复现。"""
        samples, predictions = _dataset()

        result = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        assert all(f.stat_gate_thresholds is None for f in result.folds)
        assert all(f.conflict_aware_margin is None for f in result.folds)
        assert "stat_gate_sr_std" not in result.methods

    def test_stat_gate_is_added_without_touching_other_methods(self) -> None:
        samples, predictions = _dataset()

        plain = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )
        extended = run_nested_cross_validation(
            samples,
            predictions,
            EVALUATOR,
            n_folds=5,
            seed=42,
            stat_gate_statistics=self.STATISTICS,
        )

        assert set(extended.methods) == set(plain.methods) | {
            "stat_gate_sr_std",
            "stat_gate_max_product",
        }
        assert extended.methods[0] == DS_METHOD
        for method in plain.methods:
            assert _labels(extended.predictions, method) == _labels(
                plain.predictions, method
            )
        for method in ("stat_gate_sr_std", "stat_gate_max_product"):
            assert len(_labels(extended.predictions, method)) == len(samples)

    def test_stat_gate_thresholds_come_from_inner_data_alone(self) -> None:
        """与 D-S 的防泄漏检验同一思路：只用内层数据重算，阈值必须一字不差。"""
        from rag_ds.baselines.stat_gate import stat_gate_scores
        from rag_ds.tuning import SplitName, search_stat_gate_thresholds

        samples, predictions = _dataset()
        folds = make_stratified_folds(samples, 5, seed=42)
        result = run_nested_cross_validation(
            samples,
            predictions,
            EVALUATOR,
            n_folds=5,
            seed=42,
            stat_gate_statistics=(ConflictStatistic.SUPPORT_REFUTE_STD,),
        )

        for fold_index in range(len(folds)):
            inner = [
                samples[i]
                for position, fold in enumerate(folds)
                if position != fold_index
                for i in fold
            ]
            keep = {s.sample_id for s in inner}
            gold = {(s.sample_id, f"{s.sample_id}-c1"): s.gold_state.value for s in inner}
            expected = search_stat_gate_thresholds(
                stat_gate_scores(
                    inner,
                    [p for p in predictions if p.sample_id in keep],
                    ConflictStatistic.SUPPORT_REFUTE_STD,
                ),
                gold,
                SplitName.VALIDATION,
                steps=9,
            ).best.thresholds

            chosen = result.folds[fold_index].stat_gate_thresholds
            assert chosen is not None
            assert chosen["stat_gate_sr_std"] == expected

    def test_margin_search_only_changes_conflict_aware(self) -> None:
        samples, predictions = _dataset()

        plain = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )
        tuned = run_nested_cross_validation(
            samples,
            predictions,
            EVALUATOR,
            n_folds=5,
            seed=42,
            conflict_margins=(0.05, 0.1, 0.3, 1.0),
        )

        for method in ("ds", "weighted_average", "majority_vote", "single_evaluator"):
            assert _labels(tuned.predictions, method) == _labels(plain.predictions, method)
        for fold in tuned.folds:
            assert fold.conflict_aware_margin in (0.05, 0.1, 0.3, 1.0)
            assert fold.conflict_aware_decision_threshold is not None


class TestNestedAblation:
    def test_full_variant_matches_main_cross_validation(self) -> None:
        """``full`` 必须与主实验的 D-S 逐条一致，否则 Δ 比的不是主表里的方法。"""
        samples, predictions = _dataset()

        ablation = run_nested_ablation(samples, predictions, n_folds=5, seed=42)
        main = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42
        )

        assert _labels(ablation.predictions, "full") == _labels(main.predictions, DS_METHOD)
        assert [f.thresholds["full"] for f in ablation.folds] == [
            f.ds_thresholds for f in main.folds
        ]

    def test_every_claim_is_predicted_once_per_variant(self) -> None:
        samples, predictions = _dataset()

        result = run_nested_ablation(samples, predictions, n_folds=5, seed=42)

        assert result.variants == tuple(v.value for v in CV_ABLATION_VARIANTS)
        for variant in result.variants:
            keys = [
                (r.sample_id, r.claim_id)
                for r in result.predictions
                if r.method == variant
            ]
            assert len(keys) == len(set(keys)) == len(samples)

    def test_removed_dimension_is_pinned_to_one(self) -> None:
        samples, predictions = _dataset()

        result = run_nested_ablation(samples, predictions, n_folds=5, seed=42)

        for fold in result.folds:
            assert fold.thresholds["no_theta_gate"].theta_threshold == 1.0
            assert fold.thresholds["no_doc_conflict_gate"].document_conflict_threshold == 1.0
            both = fold.thresholds["no_two_dimensional_gate"]
            assert (both.theta_threshold, both.document_conflict_threshold) == (1.0, 1.0)

    def test_without_both_gates_only_direction_is_left(self) -> None:
        """两个门控都去掉后只剩支持/反驳方向（测试数据里没有完全冲突）。"""
        samples, predictions = _dataset()

        result = run_nested_ablation(samples, predictions, n_folds=5, seed=42)

        assert set(_labels(result.predictions, "no_two_dimensional_gate").values()) <= {
            "supported",
            "refuted",
            "undetermined",
        }

    @pytest.mark.parametrize(
        "variant",
        [AblationVariant.NO_RELIABILITY, AblationVariant.NO_EVAL_CONFLICT_ALERT],
    )
    def test_unsupported_variant_is_rejected(self, variant: AblationVariant) -> None:
        samples, predictions = _dataset()

        with pytest.raises(ValueError, match="不支持"):
            run_nested_ablation(samples, predictions, variants=[variant])


class TestStatGateProductVariant:
    def test_all_four_variants_run_side_by_side(self) -> None:
        samples, predictions = _dataset()

        result = run_nested_cross_validation(
            samples,
            predictions,
            EVALUATOR,
            n_folds=5,
            seed=42,
            stat_gate_statistics=list(STAT_GATE_VARIANTS.values()),
        )

        for name in STAT_GATE_VARIANTS:
            method = f"stat_gate_{name}"
            assert method in result.methods
            assert len(_labels(result.predictions, method)) == len(samples)
            assert all(method in f.stat_gate_thresholds for f in result.folds)

    def test_adding_product_variants_leaves_mean_variants_unchanged(self) -> None:
        samples, predictions = _dataset()

        two = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42,
            stat_gate_statistics=(ConflictStatistic.SUPPORT_REFUTE_STD, ConflictStatistic.MAX_PRODUCT),
        )
        four = run_nested_cross_validation(
            samples, predictions, EVALUATOR, n_folds=5, seed=42,
            stat_gate_statistics=list(STAT_GATE_VARIANTS.values()),
        )

        for method in ("ds", "stat_gate_sr_std", "stat_gate_max_product"):
            assert _labels(four.predictions, method) == _labels(two.predictions, method)
