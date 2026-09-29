"""选择性回答置信度测试。

原来的风险分数 ``max(m_theta, K_doc)`` 对 D-S 系统性不公：那两个量是四类里
**两类的证据本身**，不是不确定性。这组测试钉住修正后的口径 —— 置信度必须取
**所预测那一类**自己的证据。
"""

from __future__ import annotations

import pytest

from rag_ds.experiments.selective_confidence import (
    baseline_confidence,
    ds_confidence,
)


class TestDsConfidence:
    """D-S 按预测类别取该类自己的质量。"""

    def test_supported_uses_support_mass(self) -> None:
        assert ds_confidence("supported", 0.7, 0.2, 0.1, 0.3) == pytest.approx(0.7)

    def test_refuted_uses_refute_mass(self) -> None:
        assert ds_confidence("refuted", 0.2, 0.7, 0.1, 0.3) == pytest.approx(0.7)

    def test_insufficient_uses_theta_mass(self) -> None:
        """m_theta 高 = 有把握判「证据不足」，是置信不是风险。"""
        assert ds_confidence("insufficient", 0.1, 0.1, 0.8, 0.2) == pytest.approx(0.8)

    def test_conflicting_uses_k_doc(self) -> None:
        """K_doc 高 = 有把握判「冲突」，同理。"""
        assert ds_confidence("conflicting", 0.4, 0.4, 0.2, 0.9) == pytest.approx(0.9)

    def test_undetermined_has_zero_confidence(self) -> None:
        """没有认领任何一类，应当最先被弃答。"""
        assert ds_confidence("undetermined", 0.4, 0.4, 0.2, 0.1) == 0.0

    def test_total_conflict_keeps_k_doc_for_conflicting(self) -> None:
        """完全冲突时三个质量为 None，但 conflicting 仍可用 K_doc 表达置信。"""
        assert ds_confidence("conflicting", None, None, None, 1.0) == pytest.approx(1.0)

    def test_total_conflict_gives_zero_for_other_classes(self) -> None:
        assert ds_confidence("supported", None, None, None, 1.0) == 0.0

    def test_unknown_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知的预测标签"):
            ds_confidence("something_else", 0.5, 0.3, 0.2, 0.1)


class TestBaselineConfidence:
    """baseline 侧口径对称：同样取所预测那一类的分数。"""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [("supported", 0.6), ("refuted", 0.3), ("insufficient", 0.1)],
    )
    def test_each_class_uses_its_own_score(self, label: str, expected: float) -> None:
        assert baseline_confidence(label, 0.6, 0.3, 0.1) == pytest.approx(expected)

    def test_conflicting_uses_the_weaker_of_the_two_directions(self) -> None:
        """conflict_aware 的冲突规则就是对 min(s, r) 设阈值。"""
        assert baseline_confidence("conflicting", 0.4, 0.35, 0.25) == pytest.approx(0.35)

    def test_undetermined_has_zero_confidence(self) -> None:
        assert baseline_confidence("undetermined", 0.4, 0.35, 0.25) == 0.0

    def test_unknown_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知的预测标签"):
            baseline_confidence("nope", 0.4, 0.35, 0.25)


class TestContrastWithOldDefinition:
    """修正口径与旧口径在关键场景上必须给出相反的排序。"""

    def test_confident_insufficient_is_low_risk_now(self) -> None:
        """m_theta=0.95 的 insufficient 预测：旧口径算高风险，新口径算高置信。"""
        m_theta, k_doc = 0.95, 0.02
        old_risk = max(m_theta, k_doc)
        new_risk = 1.0 - ds_confidence("insufficient", 0.02, 0.01, m_theta, k_doc)

        assert old_risk == pytest.approx(0.95)   # 旧：最先被弃答
        assert new_risk == pytest.approx(0.05)   # 新：最后才被弃答

    def test_confident_conflicting_is_low_risk_now(self) -> None:
        m_theta, k_doc = 0.05, 0.90
        old_risk = max(m_theta, k_doc)
        new_risk = 1.0 - ds_confidence("conflicting", 0.45, 0.45, m_theta, k_doc)

        assert old_risk == pytest.approx(0.90)
        assert new_risk == pytest.approx(0.10)

    def test_weak_supported_is_high_risk_under_both(self) -> None:
        """真正没把握的预测在两套口径下都该是高风险。"""
        m_support, m_theta, k_doc = 0.12, 0.10, 0.08
        new_risk = 1.0 - ds_confidence("supported", m_support, 0.10, m_theta, k_doc)

        assert new_risk > 0.85
