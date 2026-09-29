"""baseline 阈值搜索测试：验证集约束、平台取中位数与退化识别。

D-S 的门控阈值在验证集上搜过，baseline 若仍用调试默认值 0.5，两边就不在同一
条件下比较。这个模块存在的意义就是把两边的调参机会拉平。
"""

from __future__ import annotations

import pytest

from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction
from rag_ds.tuning import SplitName, search_baseline_thresholds
from rag_ds.tuning.baseline_threshold_search import BaselineThresholdSearchResult

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


def _prediction(sample_id: str, doc: str, triple: tuple[float, float, float]):
    return RelationPrediction(
        sample_id=sample_id,
        claim_id=f"{sample_id}-c1",
        doc_id=f"{sample_id}-{doc}",
        evaluator=EVALUATOR,
        p_support=triple[0],
        p_refute=triple[1],
        p_unknown=triple[2],
    )


def _dataset():
    """四条样本覆盖 supported / refuted / insufficient，分数刻意偏软。"""
    samples = [
        _sample("s1", EvidenceState.SUPPORTED),
        _sample("s2", EvidenceState.REFUTED),
        _sample("s3", EvidenceState.INSUFFICIENT),
        _sample("s4", EvidenceState.SUPPORTED),
    ]
    predictions = [
        _prediction("s1", "d1", (0.45, 0.25, 0.30)),
        _prediction("s1", "d2", (0.40, 0.30, 0.30)),
        _prediction("s2", "d1", (0.25, 0.45, 0.30)),
        _prediction("s2", "d2", (0.30, 0.40, 0.30)),
        _prediction("s3", "d1", (0.20, 0.20, 0.60)),
        _prediction("s3", "d2", (0.25, 0.25, 0.50)),
        _prediction("s4", "d1", (0.48, 0.22, 0.30)),
        _prediction("s4", "d2", (0.44, 0.26, 0.30)),
    ]
    return samples, predictions


class TestValidationOnly:
    @pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.TEST])
    def test_non_validation_split_is_rejected(self, split: SplitName) -> None:
        samples, predictions = _dataset()

        with pytest.raises(ValueError, match="只能在验证集上搜索"):
            search_baseline_thresholds(samples, predictions, split, EVALUATOR)

    def test_result_model_also_refuses_non_validation(self) -> None:
        samples, predictions = _dataset()
        result = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR
        )

        with pytest.raises(ValueError, match="只能在验证集上搜索"):
            BaselineThresholdSearchResult(
                split=SplitName.TEST,
                best=result.best,
                candidates=result.candidates,
                claim_count=result.claim_count,
            )


class TestSearch:
    def test_all_four_methods_are_searched(self) -> None:
        samples, predictions = _dataset()

        result = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR
        )

        assert set(result.best) == {
            "weighted_average",
            "majority_vote",
            "single_evaluator",
            "conflict_aware",
        }
        assert result.claim_count == 4
        # conflict_aware 的最优候选带 conflict_threshold，其余方法不带。
        assert result.best["conflict_aware"].conflict_threshold is not None
        assert result.best["weighted_average"].conflict_threshold is None

    def test_default_threshold_of_half_is_degenerate_for_averaging_methods(self) -> None:
        """分数偏软时 0.5 把所有 claim 判成 insufficient —— 这正是要避免的情形。

        ``majority_vote`` 不受影响：它的分数是投票比例，两票一致就是 1.0。
        三个方法的分数刻度不同，这正是必须**按方法分别搜索**阈值的原因。
        """
        samples, predictions = _dataset()

        result = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR,
            decision_thresholds=(0.5,),
        )

        for method in ("weighted_average", "single_evaluator"):
            candidate = result.best[method]
            assert candidate.is_degenerate
            assert set(candidate.label_counts) == {"insufficient"}
        assert not result.best["majority_vote"].is_degenerate

    def test_tuning_beats_the_debug_default(self) -> None:
        samples, predictions = _dataset()

        tuned = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR,
            decision_thresholds=tuple(round(0.05 * i, 2) for i in range(1, 19)),
        )
        untuned = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR,
            decision_thresholds=(0.5,),
        )

        for method, candidate in tuned.best.items():
            assert candidate.macro_f1 >= untuned.best[method].macro_f1

    def test_plateau_median_is_chosen(self) -> None:
        """整段阈值同分时取中位数，而不是脆弱的边缘值。"""
        samples, predictions = _dataset()
        grid = tuple(round(0.05 * i, 2) for i in range(1, 8))  # 0.05 ~ 0.35

        result = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR,
            decision_thresholds=grid,
        )

        for method, best in result.best.items():
            plateau = sorted(
                c.decision_threshold
                for c in result.candidates
                if c.method.value == method and c.macro_f1 == best.macro_f1
            )
            assert best.decision_threshold == plateau[(len(plateau) - 1) // 2]

    def test_candidates_cover_every_threshold_and_method(self) -> None:
        samples, predictions = _dataset()
        grid = (0.2, 0.3, 0.4)
        conflict_grid = (0.1, 0.2)

        result = search_baseline_thresholds(
            samples, predictions, SplitName.VALIDATION, EVALUATOR,
            decision_thresholds=grid,
            conflict_thresholds=conflict_grid,
        )

        # 三个朴素方法各 len(grid) 条；conflict_aware 是 len(grid) × len(conflict_grid) 条。
        assert len(result.candidates) == len(grid) * 3 + len(grid) * len(conflict_grid)
        ca = [c for c in result.candidates if c.method.value == "conflict_aware"]
        assert {c.conflict_threshold for c in ca} == set(conflict_grid)


class TestGuards:
    def test_empty_candidate_grid_is_rejected(self) -> None:
        samples, predictions = _dataset()

        with pytest.raises(ValueError, match="不能为空"):
            search_baseline_thresholds(
                samples, predictions, SplitName.VALIDATION, EVALUATOR,
                decision_thresholds=(),
            )

    def test_missing_gold_state_is_rejected(self) -> None:
        samples, predictions = _dataset()
        samples[0] = samples[0].model_copy(update={"gold_state": None})

        with pytest.raises(ValueError, match="缺少 gold_state"):
            search_baseline_thresholds(
                samples, predictions, SplitName.VALIDATION, EVALUATOR
            )
