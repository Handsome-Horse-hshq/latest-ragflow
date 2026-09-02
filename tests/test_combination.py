"""第六阶段两条 BPA 的 Dempster 组合与单次冲突量 K 的测试。

规格算例：left=(0.8, 0.1, 0.1) 与 right=(0.1, 0.8, 0.1)，
K = 0.65，未归一化 (0.17, 0.17, 0.01)，分母 0.35，
归一化后约 (0.4857142857, 0.4857142857, 0.0285714286)。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rag_ds.ds.combination import (
    TOTAL_CONFLICT_EPSILON,
    CombinedMass,
    PairwiseCombinationResult,
    TotalConflictError,
    combine_two_masses,
)
from rag_ds.ds.mass import MassFunction

#: 用于遍历的合法 BPA 三元组。
MASS_TRIPLES = [
    (0.8, 0.1, 0.1),
    (0.1, 0.8, 0.1),
    (0.0, 0.0, 1.0),
    (0.5, 0.2, 0.3),
    (0.2, 0.5, 0.3),
    (0.9, 0.05, 0.05),
    (0.05, 0.9, 0.05),
    (1 / 3, 1 / 3, 1 / 3),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
]


def _combined(m_support: float, m_refute: float, m_theta: float) -> CombinedMass:
    """构造一个 CombinedMass。"""
    return CombinedMass(m_support=m_support, m_refute=m_refute, m_theta=m_theta)


def _mass(
    m_support: float,
    m_refute: float,
    m_theta: float,
    reliability_applied: float = 1.0,
    doc_id: str = "d1",
) -> MassFunction:
    """构造一个带 ID 的 MassFunction。"""
    return MassFunction(
        sample_id="s1",
        claim_id="c1",
        doc_id=doc_id,
        evaluator="mock_evaluator",
        m_support=m_support,
        m_refute=m_refute,
        m_theta=m_theta,
        reliability_applied=reliability_applied,
    )


def _almost_total_conflict(delta: float) -> tuple[CombinedMass, CombinedMass]:
    """构造一对几乎完全冲突的证据，``1 - K`` 约为 ``2 * delta``。"""
    return (
        _combined(1.0 - delta, 0.0, delta),
        _combined(0.0, 1.0 - delta, delta),
    )


# --------------------------------------------------------------------------
# 1-2. CombinedMass 模型
# --------------------------------------------------------------------------


def test_combined_mass_accepts_valid_masses() -> None:
    """合法质量可以构造 CombinedMass。"""
    mass = _combined(0.5, 0.2, 0.3)

    assert mass.m_support == pytest.approx(0.5)
    assert mass.m_refute == pytest.approx(0.2)
    assert mass.m_theta == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("m_support", "m_refute", "m_theta"),
    [(0.5, 0.2, 0.1), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)],
)
def test_combined_mass_rejects_bad_sum(
    m_support: float, m_refute: float, m_theta: float
) -> None:
    """三个质量之和不为 1 时被拒绝。"""
    with pytest.raises(ValidationError, match="必须等于 1"):
        _combined(m_support, m_refute, m_theta)


@pytest.mark.parametrize("field", ["m_support", "m_refute", "m_theta"])
@pytest.mark.parametrize("bad_value", [-0.1, 1.5])
def test_combined_mass_rejects_out_of_range(field: str, bad_value: float) -> None:
    """质量超出 [0, 1] 时被拒绝。"""
    payload: dict[str, Any] = {"m_support": 0.5, "m_refute": 0.2, "m_theta": 0.3}
    payload[field] = bad_value

    with pytest.raises(ValidationError, match=field):
        CombinedMass.model_validate(payload)


def test_combined_mass_rejects_unknown_field() -> None:
    """未定义字段被拒绝 —— 特别是伪造的 ID。"""
    with pytest.raises(ValidationError, match="doc_id"):
        CombinedMass.model_validate(
            {"m_support": 0.5, "m_refute": 0.2, "m_theta": 0.3, "doc_id": "doc1+doc2"}
        )


def test_combined_mass_is_immutable() -> None:
    """CombinedMass 不可变。"""
    mass = _combined(0.5, 0.2, 0.3)

    with pytest.raises(ValidationError):
        mass.m_support = 0.0  # type: ignore[misc]


def test_combined_mass_carries_no_identifiers() -> None:
    """CombinedMass 只有三个质量字段，不携带任何业务 ID。"""
    assert set(CombinedMass.model_fields) == {"m_support", "m_refute", "m_theta"}


# --------------------------------------------------------------------------
# 3-5. K、分母与规格算例
# --------------------------------------------------------------------------


def test_result_preserves_conflict_and_denominator() -> None:
    """PairwiseCombinationResult 保存归一化前的 K 与 1-K。"""
    result = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    assert isinstance(result, PairwiseCombinationResult)
    assert result.conflict == pytest.approx(0.65)
    assert result.normalization_denominator == pytest.approx(0.35)
    assert result.normalization_denominator == pytest.approx(1.0 - result.conflict)


def test_worked_numeric_example() -> None:
    """规格给出的算例。"""
    result = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    assert result.conflict == pytest.approx(0.65)
    assert result.normalization_denominator == pytest.approx(0.35)
    assert result.mass.m_support == pytest.approx(0.4857142857, abs=1e-9)
    assert result.mass.m_refute == pytest.approx(0.4857142857, abs=1e-9)
    assert result.mass.m_theta == pytest.approx(0.0285714286, abs=1e-9)


def test_worked_example_unnormalised_intermediates() -> None:
    """算例的未归一化中间量：0.17 / 0.17 / 0.01。"""
    result = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))
    denominator = result.normalization_denominator

    assert result.mass.m_support * denominator == pytest.approx(0.17)
    assert result.mass.m_refute * denominator == pytest.approx(0.17)
    assert result.mass.m_theta * denominator == pytest.approx(0.01)


def test_result_model_rejects_inconsistent_denominator() -> None:
    """手工构造时，分母与 1-K 不一致会被拒绝。"""
    with pytest.raises(ValidationError, match="必须等于 1 - conflict"):
        PairwiseCombinationResult(
            mass=_combined(0.5, 0.2, 0.3),
            conflict=0.65,
            normalization_denominator=0.9,
        )


# --------------------------------------------------------------------------
# 6-7. 结果不变量
# --------------------------------------------------------------------------


def test_combined_masses_always_sum_to_one_and_stay_in_range() -> None:
    """遍历所有可组合的三元组，质量恒在 [0, 1] 且和为 1。"""
    checked = 0
    for left_triple in MASS_TRIPLES:
        for right_triple in MASS_TRIPLES:
            result = combine_two_masses(
                _combined(*left_triple), _combined(*right_triple)
            )
            mass = result.mass

            for value in (mass.m_support, mass.m_refute, mass.m_theta):
                assert 0.0 <= value <= 1.0
            assert mass.m_support + mass.m_refute + mass.m_theta == pytest.approx(1.0)
            assert 0.0 <= result.conflict <= 1.0
            assert 0.0 <= result.normalization_denominator <= 1.0
            checked += 1

    assert checked == len(MASS_TRIPLES) ** 2


def test_combination_matches_manual_formula() -> None:
    """与手写公式逐点比对。"""
    for left_triple in MASS_TRIPLES:
        for right_triple in MASS_TRIPLES:
            left, right = _combined(*left_triple), _combined(*right_triple)
            conflict = (
                left.m_support * right.m_refute + left.m_refute * right.m_support
            )
            denominator = 1.0 - conflict
            support_raw = (
                left.m_support * right.m_support
                + left.m_support * right.m_theta
                + left.m_theta * right.m_support
            )
            refute_raw = (
                left.m_refute * right.m_refute
                + left.m_refute * right.m_theta
                + left.m_theta * right.m_refute
            )
            theta_raw = left.m_theta * right.m_theta

            result = combine_two_masses(left, right)

            assert result.conflict == pytest.approx(conflict)
            assert result.mass.m_support == pytest.approx(support_raw / denominator)
            assert result.mass.m_refute == pytest.approx(refute_raw / denominator)
            assert result.mass.m_theta == pytest.approx(theta_raw / denominator)


# --------------------------------------------------------------------------
# 8-9. 同向证据互相加强
# --------------------------------------------------------------------------


def test_two_supporting_evidences_increase_support() -> None:
    """两条支持证据融合后支持质量提高。"""
    single = _combined(0.6, 0.1, 0.3)

    result = combine_two_masses(single, single)

    assert result.mass.m_support > single.m_support
    assert result.mass.m_theta < single.m_theta


def test_two_refuting_evidences_increase_refute() -> None:
    """两条反驳证据融合后反驳质量提高。"""
    single = _combined(0.1, 0.6, 0.3)

    result = combine_two_masses(single, single)

    assert result.mass.m_refute > single.m_refute
    assert result.mass.m_theta < single.m_theta


# --------------------------------------------------------------------------
# 10-11. 单位元与交换性
# --------------------------------------------------------------------------


@pytest.mark.parametrize("triple", MASS_TRIPLES)
def test_total_ignorance_is_the_identity_element(
    triple: tuple[float, float, float]
) -> None:
    """完全无知 (0, 0, 1) 是组合的单位元。"""
    vacuous = _combined(0.0, 0.0, 1.0)
    other = _combined(*triple)

    for result in (
        combine_two_masses(vacuous, other),
        combine_two_masses(other, vacuous),
    ):
        assert result.conflict == pytest.approx(0.0)
        assert result.normalization_denominator == pytest.approx(1.0)
        assert result.mass.m_support == pytest.approx(other.m_support)
        assert result.mass.m_refute == pytest.approx(other.m_refute)
        assert result.mass.m_theta == pytest.approx(other.m_theta)


def test_combination_is_commutative() -> None:
    """combine(a, b) 与 combine(b, a) 结果相同。"""
    for left_triple in MASS_TRIPLES:
        for right_triple in MASS_TRIPLES:
            left, right = _combined(*left_triple), _combined(*right_triple)

            forward = combine_two_masses(left, right)
            backward = combine_two_masses(right, left)

            assert forward.conflict == pytest.approx(backward.conflict)
            assert forward.mass.m_support == pytest.approx(backward.mass.m_support)
            assert forward.mass.m_refute == pytest.approx(backward.mass.m_refute)
            assert forward.mass.m_theta == pytest.approx(backward.mass.m_theta)


def test_input_order_does_not_change_conflict() -> None:
    """K 的定义本身对称，交换输入不改变 K。"""
    left, right = _combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1)

    assert combine_two_masses(left, right).conflict == pytest.approx(
        combine_two_masses(right, left).conflict
    )


# --------------------------------------------------------------------------
# 12-16. 完全冲突
# --------------------------------------------------------------------------


def test_full_support_versus_full_refute_gives_conflict_one() -> None:
    """一条完全支持与一条完全反驳，K = 1。"""
    left, right = _combined(1.0, 0.0, 0.0), _combined(0.0, 1.0, 0.0)

    conflict = left.m_support * right.m_refute + left.m_refute * right.m_support

    assert conflict == pytest.approx(1.0)


def test_total_conflict_raises() -> None:
    """K = 1 时抛出 TotalConflictError，而不是返回任何数值。"""
    with pytest.raises(TotalConflictError):
        combine_two_masses(_combined(1.0, 0.0, 0.0), _combined(0.0, 1.0, 0.0))


def test_total_conflict_error_reports_k_and_denominator() -> None:
    """异常同时给出 K、1-K，并说明标准规则在此没有定义。"""
    with pytest.raises(TotalConflictError) as excinfo:
        combine_two_masses(_combined(1.0, 0.0, 0.0), _combined(0.0, 1.0, 0.0))

    error = excinfo.value
    assert error.conflict == pytest.approx(1.0)
    assert error.denominator == pytest.approx(0.0)

    message = str(error)
    assert "K=1.0" in message
    assert "1-K=0.0" in message
    assert "没有定义" in message


@pytest.mark.parametrize("delta", [0.0, 1e-16, 1e-15, 1e-13])
def test_near_total_conflict_below_threshold_raises(delta: float) -> None:
    """1-K 不大于阈值时抛出异常，不会用 epsilon 强行归一化。"""
    left, right = _almost_total_conflict(delta)

    with pytest.raises(TotalConflictError) as excinfo:
        combine_two_masses(left, right)

    assert excinfo.value.denominator <= TOTAL_CONFLICT_EPSILON


def test_high_conflict_above_threshold_still_computes() -> None:
    """高冲突但分母远大于阈值时仍能正常计算。"""
    left, right = _almost_total_conflict(5e-4)
    result = combine_two_masses(left, right)

    assert result.normalization_denominator > TOTAL_CONFLICT_EPSILON
    assert result.conflict > 0.999
    assert result.mass.m_support + result.mass.m_refute + result.mass.m_theta == (
        pytest.approx(1.0)
    )


def test_conflict_close_to_one_but_computable() -> None:
    """K = 0.99 这类高冲突场景正常返回结果。"""
    result = combine_two_masses(_combined(0.99, 0.0, 0.01), _combined(0.0, 1.0, 0.0))

    assert result.conflict == pytest.approx(0.99)
    assert result.normalization_denominator == pytest.approx(0.01)
    assert result.mass.m_refute == pytest.approx(1.0)


def test_catastrophic_cancellation_band_fails_loudly() -> None:
    """分母落在 1e-12 与约 2e-11 之间时，宁可报错也不返回不可信数值。

    此时 ``1 - K`` 由两个接近 1 的数相减得到，有效位数已所剩无几，
    归一化后三个质量之和偏离 1 超过容差，被 CombinedMass 挡下。
    """
    left, right = _almost_total_conflict(1e-12)

    with pytest.raises(ValidationError, match="必须等于 1"):
        combine_two_masses(left, right)


# --------------------------------------------------------------------------
# 17-23. 输入类型、不可变性与无关字段
# --------------------------------------------------------------------------


def test_inputs_are_not_modified() -> None:
    """combine_two_masses 不修改 left 与 right。"""
    left = _mass(0.8, 0.1, 0.1, reliability_applied=0.7)
    right = _mass(0.1, 0.8, 0.1, reliability_applied=0.3, doc_id="d2")
    before_left, before_right = left.model_dump(), right.model_dump()

    combine_two_masses(left, right)

    assert left.model_dump() == before_left
    assert right.model_dump() == before_right


def test_accepts_two_mass_functions() -> None:
    """两个 MassFunction 可以直接组合。"""
    result = combine_two_masses(_mass(0.8, 0.1, 0.1), _mass(0.1, 0.8, 0.1))

    assert result.conflict == pytest.approx(0.65)
    assert isinstance(result.mass, CombinedMass)


def test_accepts_two_combined_masses() -> None:
    """两个 CombinedMass 可以直接组合。"""
    result = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    assert result.conflict == pytest.approx(0.65)


def test_accepts_mixed_input_types() -> None:
    """MassFunction 与 CombinedMass 可以混合组合，且与同类型结果一致。"""
    mixed = combine_two_masses(_mass(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))
    reference = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    assert mixed.mass == reference.mass
    assert mixed.conflict == pytest.approx(reference.conflict)


@pytest.mark.parametrize("reliability_applied", [0.0, 0.3, 0.7, 1.0])
def test_reliability_applied_does_not_enter_the_formula(
    reliability_applied: float,
) -> None:
    """reliability_applied 不参与组合公式 —— 折扣是上一阶段的事。"""
    left = _mass(0.8, 0.1, 0.1, reliability_applied=reliability_applied)
    right = _mass(0.1, 0.8, 0.1, reliability_applied=reliability_applied)
    reference = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    result = combine_two_masses(left, right)

    assert result.mass == reference.mass
    assert result.conflict == pytest.approx(reference.conflict)


def test_identifiers_do_not_enter_the_formula() -> None:
    """ID 与 evaluator 名称不影响结果。"""
    left = MassFunction(
        sample_id="whatever",
        claim_id="another",
        doc_id="dX",
        evaluator="some_evaluator",
        m_support=0.8,
        m_refute=0.1,
        m_theta=0.1,
    )
    right = _mass(0.1, 0.8, 0.1, doc_id="dY")
    reference = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    assert combine_two_masses(left, right).mass == reference.mass


def test_result_is_immutable() -> None:
    """PairwiseCombinationResult 不可变，K 不会被事后改写。"""
    result = combine_two_masses(_combined(0.8, 0.1, 0.1), _combined(0.1, 0.8, 0.1))

    with pytest.raises(ValidationError):
        result.conflict = 0.0  # type: ignore[misc]
