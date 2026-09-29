"""按观测值构造阈值网格的测试。

固定网格 ``(0.3, ..., 0.7)`` 隐含「m_theta 会落在 0.3 以上」这个假设。换成
校准后的软概率、或文档数变多时，Dempster 组合会把 m_theta 压得很小，整条
theta 轴就全部落在观测范围之外 —— 门控一次都不触发，而搜索结果看上去毫无
异常。:meth:`ThresholdGrid.from_observed` 就是为这种情况准备的。
"""

from __future__ import annotations

import pytest

from rag_ds.tuning import ThresholdGrid


class TestFromObserved:
    """网格必须落在观测范围之内。"""

    def test_candidates_stay_within_the_observed_range(self) -> None:
        theta = [0.001, 0.01, 0.02, 0.05, 0.08, 0.12, 0.14]
        k_doc = [0.41, 0.48, 0.52, 0.58, 0.67, 0.74, 0.83]

        grid = ThresholdGrid.from_observed(theta, k_doc)

        assert all(min(theta) <= v <= max(theta) for v in grid.theta_values)
        assert all(min(k_doc) <= v <= max(k_doc) for v in grid.document_conflict_values)

    def test_fixed_grid_would_miss_a_small_theta_range(self) -> None:
        """这正是彩排里暴露的退化：固定网格全部高于观测最大值。"""
        theta = [0.001, 0.01, 0.03, 0.07, 0.14]

        assert all(v > max(theta) for v in ThresholdGrid().theta_values)

        grid = ThresholdGrid.from_observed(theta, [0.4, 0.6, 0.8])

        assert any(v <= max(theta) for v in grid.theta_values)

    def test_steps_controls_candidate_count(self) -> None:
        values = [i / 100 for i in range(100)]

        grid = ThresholdGrid.from_observed(values, values, steps=3)

        assert len(grid.theta_values) == 3
        assert len(grid) == 9

    def test_duplicates_are_collapsed(self) -> None:
        grid = ThresholdGrid.from_observed([0.2] * 10, [0.5] * 10)

        assert grid.theta_values == (0.2,)
        assert grid.document_conflict_values == (0.5,)

    def test_candidates_are_ordered_and_usable(self) -> None:
        grid = ThresholdGrid.from_observed([0.05, 0.1, 0.2], [0.4, 0.5, 0.6], steps=2)

        candidates = list(grid.candidates())

        assert len(candidates) == len(grid)
        assert all(0.0 <= c.theta_threshold <= 1.0 for c in candidates)

    def test_evaluator_threshold_and_tie_tolerance_pass_through(self) -> None:
        grid = ThresholdGrid.from_observed(
            [0.1], [0.5], evaluator_conflict_threshold=0.25, tie_tolerance=1e-9
        )

        assert grid.evaluator_conflict_threshold == 0.25
        assert grid.tie_tolerance == 1e-9

    @pytest.mark.parametrize("bad_steps", [0, -1])
    def test_non_positive_steps_are_rejected(self, bad_steps: int) -> None:
        with pytest.raises(ValueError, match="steps 必须是正整数"):
            ThresholdGrid.from_observed([0.1], [0.5], steps=bad_steps)

    def test_empty_observations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="观测值序列不能为空"):
            ThresholdGrid.from_observed([], [0.5])

    def test_default_grid_is_unchanged(self) -> None:
        """既有 oracle 结果必须保持可复现，默认网格不能动。"""
        grid = ThresholdGrid()

        assert grid.theta_values == (0.3, 0.4, 0.5, 0.6, 0.7)
        assert grid.document_conflict_values == (0.2, 0.3, 0.4, 0.5, 0.6)
