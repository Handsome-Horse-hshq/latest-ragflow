"""第五阶段 BPA 数据模型与基础映射的测试。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rag_ds.ds.mass import MASS_SUM_TOLERANCE, MassFunction, mass_from_prediction
from rag_ds.schemas import RelationPrediction


def _prediction(
    p_support: float = 0.8,
    p_refute: float = 0.1,
    p_unknown: float = 0.1,
    evaluator_reliability: float = 1.0,
) -> RelationPrediction:
    """构造一条关系预测。"""
    return RelationPrediction(
        sample_id="s1",
        claim_id="c1",
        doc_id="d1",
        evaluator="mock_evaluator",
        p_support=p_support,
        p_refute=p_refute,
        p_unknown=p_unknown,
        evaluator_reliability=evaluator_reliability,
    )


def _mass_payload(**overrides: Any) -> dict[str, Any]:
    """返回一份合法的 MassFunction 输入。"""
    payload: dict[str, Any] = {
        "sample_id": "s1",
        "claim_id": "c1",
        "doc_id": "d1",
        "evaluator": "mock_evaluator",
        "m_support": 0.6,
        "m_refute": 0.1,
        "m_theta": 0.3,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# 1-5. MassFunction 模型约束
# --------------------------------------------------------------------------


def test_valid_mass_function_can_be_created() -> None:
    """合法输入可以构造 MassFunction。"""
    mass = MassFunction.model_validate(_mass_payload())

    assert mass.m_support == pytest.approx(0.6)
    assert mass.m_refute == pytest.approx(0.1)
    assert mass.m_theta == pytest.approx(0.3)
    assert mass.reliability_applied == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("m_support", "m_refute", "m_theta"),
    [
        (0.6, 0.1, 0.1),  # 和为 0.8
        (0.6, 0.3, 0.3),  # 和为 1.2
        (0.0, 0.0, 0.0),  # 和为 0
    ],
)
def test_masses_must_sum_to_one(
    m_support: float, m_refute: float, m_theta: float
) -> None:
    """三个焦元质量之和不为 1 时被拒绝。"""
    payload = _mass_payload(m_support=m_support, m_refute=m_refute, m_theta=m_theta)

    with pytest.raises(ValidationError, match="必须等于 1"):
        MassFunction.model_validate(payload)


def test_sum_tolerance_is_accepted() -> None:
    """容差以内的浮点误差可以接受。"""
    mass = MassFunction.model_validate(
        _mass_payload(m_support=1 / 3, m_refute=1 / 3, m_theta=1 / 3)
    )

    total = mass.m_support + mass.m_refute + mass.m_theta
    assert abs(total - 1.0) <= MASS_SUM_TOLERANCE


def test_error_above_tolerance_is_rejected() -> None:
    """超出容差的偏差仍被拒绝。"""
    payload = _mass_payload(m_support=0.6, m_refute=0.1, m_theta=0.3 + 1e-4)

    with pytest.raises(ValidationError, match="必须等于 1"):
        MassFunction.model_validate(payload)


@pytest.mark.parametrize("field", ["m_support", "m_refute", "m_theta"])
def test_negative_mass_is_rejected(field: str) -> None:
    """质量小于 0 时被拒绝。"""
    payload = _mass_payload(**{field: -0.1})

    with pytest.raises(ValidationError, match=field):
        MassFunction.model_validate(payload)


@pytest.mark.parametrize("field", ["m_support", "m_refute", "m_theta"])
def test_mass_above_one_is_rejected(field: str) -> None:
    """质量大于 1 时被拒绝。"""
    payload = _mass_payload(**{field: 1.5})

    with pytest.raises(ValidationError, match=field):
        MassFunction.model_validate(payload)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_reliability_applied_out_of_range_is_rejected(bad_value: float) -> None:
    """reliability_applied 超出 [0, 1] 时被拒绝。"""
    payload = _mass_payload(reliability_applied=bad_value)

    with pytest.raises(ValidationError, match="reliability_applied"):
        MassFunction.model_validate(payload)


def test_unknown_field_is_rejected() -> None:
    """未定义字段被拒绝。"""
    payload = _mass_payload(m_conflict=0.0)

    with pytest.raises(ValidationError, match="m_conflict"):
        MassFunction.model_validate(payload)


@pytest.mark.parametrize("field", ["sample_id", "claim_id", "doc_id", "evaluator"])
def test_empty_identifier_is_rejected(field: str) -> None:
    """四个标识字段不接受空串或纯空白。"""
    payload = _mass_payload(**{field: "   "})

    with pytest.raises(ValidationError, match=field):
        MassFunction.model_validate(payload)


def test_mass_function_is_immutable() -> None:
    """MassFunction 是不可变对象，无法就地修改。"""
    mass = MassFunction.model_validate(_mass_payload())

    with pytest.raises(ValidationError):
        mass.m_support = 0.0  # type: ignore[misc]

    assert mass.m_support == pytest.approx(0.6)


# --------------------------------------------------------------------------
# 6-10. mass_from_prediction
# --------------------------------------------------------------------------


def test_mass_from_prediction_copies_the_three_probabilities() -> None:
    """三个概率被原样搬运到三个焦元。"""
    mass = mass_from_prediction(_prediction(0.8, 0.1, 0.1))

    assert mass.m_support == pytest.approx(0.8)
    assert mass.m_refute == pytest.approx(0.1)
    assert mass.m_theta == pytest.approx(0.1)


def test_theta_base_value_equals_p_unknown() -> None:
    """m_theta 的基础值就是 p_unknown。"""
    for p_unknown in (0.0, 0.05, 0.3, 0.9, 1.0):
        remainder = (1.0 - p_unknown) / 2
        prediction = _prediction(remainder, remainder, p_unknown)

        assert mass_from_prediction(prediction).m_theta == pytest.approx(p_unknown)


def test_identifiers_are_preserved() -> None:
    """四个标识字段被完整保留。"""
    prediction = _prediction()

    mass = mass_from_prediction(prediction)

    assert mass.sample_id == prediction.sample_id
    assert mass.claim_id == prediction.claim_id
    assert mass.doc_id == prediction.doc_id
    assert mass.evaluator == prediction.evaluator


@pytest.mark.parametrize("evaluator_reliability", [0.0, 0.5, 0.8, 1.0])
def test_mass_from_prediction_applies_no_reliability(
    evaluator_reliability: float,
) -> None:
    """基础映射不应用任何可靠性，evaluator_reliability 被刻意忽略。"""
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=evaluator_reliability)

    mass = mass_from_prediction(prediction)

    assert mass.m_support == pytest.approx(0.8)
    assert mass.m_refute == pytest.approx(0.1)
    assert mass.m_theta == pytest.approx(0.1)
    assert mass.reliability_applied == pytest.approx(1.0)


def test_input_prediction_is_not_modified() -> None:
    """传入的 RelationPrediction 不会被修改。"""
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=0.6)
    before = prediction.model_dump()

    mass_from_prediction(prediction)

    assert prediction.model_dump() == before


def test_resulting_masses_always_sum_to_one() -> None:
    """任意合法概率三元组映射出的 BPA 都归一。"""
    triples = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.8, 0.1, 0.1),
        (0.05, 0.9, 0.05),
        (0.5, 0.5, 0.0),
        (1 / 3, 1 / 3, 1 / 3),
    ]

    for p_support, p_refute, p_unknown in triples:
        mass = mass_from_prediction(_prediction(p_support, p_refute, p_unknown))
        total = mass.m_support + mass.m_refute + mass.m_theta

        assert total == pytest.approx(1.0)
