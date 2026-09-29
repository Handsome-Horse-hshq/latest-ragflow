"""配对自助重采样测试。

测试集只有 80 条 claim 时，0.08 的 Macro-F1 差距其 95% 区间是跨 0 的。
只报点估计等于把「可能只是噪声」说成「方法更好」，所以这一层必须有。
"""

from __future__ import annotations

import pytest

from rag_ds.metrics import GOLD_LABELS, bootstrap_interval, paired_bootstrap


def _repeat(pattern: list[str], times: int) -> list[str]:
    return pattern * times


class TestPairedBootstrap:
    def test_identical_predictions_give_zero_delta(self) -> None:
        truth = _repeat(["supported", "refuted", "insufficient", "conflicting"], 10)

        result = paired_bootstrap(truth, truth, truth, "a", "b", n_resamples=200)

        assert result.observed_delta == pytest.approx(0.0)
        assert result.ci_low == pytest.approx(0.0)
        assert result.ci_high == pytest.approx(0.0)
        assert result.is_significant is False

    def test_large_consistent_gap_is_significant(self) -> None:
        truth = _repeat(["supported", "refuted", "insufficient", "conflicting"], 25)
        perfect = list(truth)
        useless = ["supported"] * len(truth)

        result = paired_bootstrap(
            truth, perfect, useless, "perfect", "useless", n_resamples=500
        )

        assert result.observed_delta > 0.5
        assert result.is_significant is True
        assert result.ci_low > 0.0
        assert result.p_value < 0.05

    def test_tiny_gap_on_small_sample_is_not_significant(self) -> None:
        """这正是当前实验的处境：差距看着不小，但样本撑不住。"""
        truth = _repeat(["supported", "refuted", "insufficient", "conflicting"], 5)
        a = list(truth)
        b = list(truth)
        b[0] = "refuted"  # 只错一条

        result = paired_bootstrap(truth, a, b, "a", "b", n_resamples=800)

        assert result.observed_delta > 0.0
        assert result.ci_low <= 0.0
        assert result.is_significant is False

    def test_delta_is_oriented_a_minus_b(self) -> None:
        truth = _repeat(["supported", "refuted"], 10)
        good, bad = list(truth), ["supported"] * len(truth)

        forward = paired_bootstrap(truth, good, bad, "good", "bad", n_resamples=200)
        backward = paired_bootstrap(truth, bad, good, "bad", "good", n_resamples=200)

        assert forward.observed_delta == pytest.approx(-backward.observed_delta)

    def test_same_seed_reproduces_exactly(self) -> None:
        truth = _repeat(["supported", "refuted", "insufficient"], 8)
        a = list(truth)
        b = ["refuted"] * len(truth)

        first = paired_bootstrap(truth, a, b, "a", "b", n_resamples=300, seed=7)
        second = paired_bootstrap(truth, a, b, "a", "b", n_resamples=300, seed=7)

        assert first.model_dump() == second.model_dump()

    def test_different_seeds_give_close_but_distinct_intervals(self) -> None:
        """两个方法各自散着出错时，重采样分布足够连续，换种子会挪动分位点。"""
        truth = _repeat(["supported", "refuted", "insufficient", "conflicting"], 15)
        a = list(truth)
        b = list(truth)
        for i in range(0, len(truth), 3):
            a[i] = "insufficient"
        for i in range(1, len(truth), 4):
            b[i] = "conflicting"

        first = paired_bootstrap(truth, a, b, "a", "b", n_resamples=400, seed=1)
        second = paired_bootstrap(truth, a, b, "a", "b", n_resamples=400, seed=2)

        assert first.observed_delta == pytest.approx(second.observed_delta)
        assert first.ci_low != second.ci_low or first.ci_high != second.ci_high
        assert abs(first.ci_low - second.ci_low) < 0.1

    def test_accuracy_metric_is_supported(self) -> None:
        truth = _repeat(["supported", "refuted"], 20)
        a, b = list(truth), ["supported"] * len(truth)

        result = paired_bootstrap(
            truth, a, b, "a", "b", metric="accuracy", n_resamples=200
        )

        assert result.metric == "accuracy"
        assert result.observed_a == pytest.approx(1.0)
        assert result.observed_b == pytest.approx(0.5)


class TestFixedLabelSet:
    """重采样必须固定 Macro-F1 的标签集，否则估计有偏。"""

    def test_label_set_defaults_to_the_four_gold_classes(self) -> None:
        truth = _repeat(["supported", "refuted"], 10)

        result = paired_bootstrap(truth, truth, truth, "a", "b", n_resamples=100)

        # 金标准里只出现两类，但平均仍按四类固定标签集进行。
        assert result.observed_a == pytest.approx(
            2.0 / len(GOLD_LABELS)
        )

    def test_explicit_label_subset_changes_the_denominator(self) -> None:
        truth = _repeat(["supported", "refuted"], 10)

        result = paired_bootstrap(
            truth, truth, truth, "a", "b",
            labels=("supported", "refuted"), n_resamples=100,
        )

        assert result.observed_a == pytest.approx(1.0)


class TestInterval:
    def test_interval_brackets_the_observed_value(self) -> None:
        truth = _repeat(["supported", "refuted", "insufficient", "conflicting"], 15)
        predicted = list(truth)
        predicted[0] = "refuted"
        predicted[5] = "supported"

        result = bootstrap_interval(truth, predicted, "ds", n_resamples=500)

        assert result.ci_low <= result.observed <= result.ci_high
        assert result.sample_count == len(truth)

    def test_perfect_prediction_has_a_degenerate_interval(self) -> None:
        truth = _repeat(["supported", "refuted", "insufficient", "conflicting"], 10)

        result = bootstrap_interval(truth, truth, "ds", n_resamples=200)

        assert result.observed == pytest.approx(1.0)
        assert result.ci_low == pytest.approx(1.0)


class TestGuards:
    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="长度与金标准不同"):
            paired_bootstrap(["supported"], ["supported", "refuted"], ["supported"], "a", "b")

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="空序列"):
            paired_bootstrap([], [], [], "a", "b")
