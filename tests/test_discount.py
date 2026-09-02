"""第五阶段可靠性折扣的测试。

第八阶段修正后，文档级折扣只应用 context.reliability：
p=(0.8, 0.1, 0.1) 与文档可靠性 0.9 得到 (0.72, 0.09, 0.19)。
评估器可靠性 0.8 在文档融合之后由 discount_combined_mass 只施加一次，
最终得到 (0.576, 0.072, 0.352)。
"""

from __future__ import annotations

import pytest

from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.discount import (
    discount_combined_mass,
    discount_mass,
    discounted_mass_from_prediction,
    document_discounted_mass_from_prediction,
    effective_reliability,
)
from rag_ds.ds.mass import MassFunction, mass_from_prediction
from rag_ds.schemas import (
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

#: 覆盖端点与中间值的可靠性取值，全部为确定值，不使用随机数。
RELIABILITIES = [0.0, 0.1, 0.25, 0.5, 0.72, 0.9, 1.0]

#: 合法的三元概率组合。
PROBABILITY_TRIPLES = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.8, 0.1, 0.1),
    (0.05, 0.9, 0.05),
    (0.5, 0.5, 0.0),
    (1 / 3, 1 / 3, 1 / 3),
    (0.9, 0.05, 0.05),
]


def _prediction(
    p_support: float = 0.8,
    p_refute: float = 0.1,
    p_unknown: float = 0.1,
    evaluator_reliability: float = 1.0,
    doc_id: str = "d1",
) -> RelationPrediction:
    """构造一条关系预测。"""
    return RelationPrediction(
        sample_id="s1",
        claim_id="c1",
        doc_id=doc_id,
        evaluator="mock_evaluator",
        p_support=p_support,
        p_refute=p_refute,
        p_unknown=p_unknown,
        evaluator_reliability=evaluator_reliability,
    )


def _context(
    reliability: float = 1.0,
    retrieval_score: float | None = None,
    doc_id: str = "d1",
) -> ContextChunk:
    """构造一段检索文档。"""
    return ContextChunk(
        doc_id=doc_id,
        text="文档正文。",
        retrieval_score=retrieval_score,
        reliability=reliability,
    )


def _mass(
    m_support: float = 0.6, m_refute: float = 0.1, m_theta: float = 0.3
) -> MassFunction:
    """构造一个基础 BPA。"""
    return MassFunction(
        sample_id="s1",
        claim_id="c1",
        doc_id="d1",
        evaluator="mock_evaluator",
        m_support=m_support,
        m_refute=m_refute,
        m_theta=m_theta,
    )


# --------------------------------------------------------------------------
# 1-3. 折扣端点与中间值
# --------------------------------------------------------------------------


def test_reliability_one_leaves_masses_unchanged() -> None:
    """reliability=1 时质量不变。"""
    mass = _mass()

    result = discount_mass(mass, 1.0)

    assert result.m_support == pytest.approx(mass.m_support)
    assert result.m_refute == pytest.approx(mass.m_refute)
    assert result.m_theta == pytest.approx(mass.m_theta)
    assert result.reliability_applied == pytest.approx(1.0)


def test_reliability_zero_moves_everything_to_theta() -> None:
    """reliability=0 时全部质量进入 Theta，退化为完全无知。"""
    result = discount_mass(_mass(), 0.0)

    assert result.m_support == pytest.approx(0.0)
    assert result.m_refute == pytest.approx(0.0)
    assert result.m_theta == pytest.approx(1.0)
    assert result.reliability_applied == pytest.approx(0.0)


def test_reliability_half_is_computed_correctly() -> None:
    """reliability=0.5 时确定质量减半，其余回流到 Theta。"""
    result = discount_mass(_mass(0.6, 0.1, 0.3), 0.5)

    assert result.m_support == pytest.approx(0.3)
    assert result.m_refute == pytest.approx(0.05)
    assert result.m_theta == pytest.approx(0.65)
    assert result.reliability_applied == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 4-6. 单调性：质量只从确定焦元流向 Theta
# --------------------------------------------------------------------------


def test_lower_reliability_never_increases_support_or_refute() -> None:
    """可靠性下降会降低 m_support 与 m_refute。"""
    mass = _mass(0.6, 0.3, 0.1)
    results = [discount_mass(mass, r) for r in sorted(RELIABILITIES)]

    supports = [r.m_support for r in results]
    refutes = [r.m_refute for r in results]

    assert supports == sorted(supports)
    assert refutes == sorted(refutes)
    assert supports[0] < supports[-1]
    assert refutes[0] < refutes[-1]


def test_lower_reliability_never_decreases_theta() -> None:
    """可靠性下降不会降低 m_theta —— 质量只会流向无知，不会流出。"""
    mass = _mass(0.6, 0.3, 0.1)
    thetas = [discount_mass(mass, r).m_theta for r in sorted(RELIABILITIES, reverse=True)]

    assert thetas == sorted(thetas)
    assert thetas[0] < thetas[-1]


def test_discount_never_moves_mass_out_of_theta() -> None:
    """折扣后的 m_theta 永远不小于折扣前。"""
    for triple in PROBABILITY_TRIPLES:
        mass = mass_from_prediction(_prediction(*triple))
        for reliability in RELIABILITIES:
            discounted = discount_mass(mass, reliability)

            assert discounted.m_theta >= mass.m_theta - 1e-12


# --------------------------------------------------------------------------
# 7. 连续折扣等价于乘积折扣
# --------------------------------------------------------------------------


def test_successive_discounts_equal_product_discount() -> None:
    """先折 0.8 再折 0.5，等价于一次折 0.4。"""
    mass = _mass(0.6, 0.1, 0.3)

    two_steps = discount_mass(discount_mass(mass, 0.8), 0.5)
    one_step = discount_mass(mass, 0.4)

    assert two_steps.m_support == pytest.approx(one_step.m_support)
    assert two_steps.m_refute == pytest.approx(one_step.m_refute)
    assert two_steps.m_theta == pytest.approx(one_step.m_theta)
    assert two_steps.reliability_applied == pytest.approx(0.4)


@pytest.mark.parametrize("first", [0.25, 0.5, 0.9])
@pytest.mark.parametrize("second", [0.1, 0.72, 1.0])
def test_discount_is_multiplicative_over_many_pairs(
    first: float, second: float
) -> None:
    """任意两次折扣都等价于其乘积的一次折扣。"""
    mass = _mass(0.5, 0.2, 0.3)

    two_steps = discount_mass(discount_mass(mass, first), second)
    one_step = discount_mass(mass, first * second)

    assert two_steps.m_support == pytest.approx(one_step.m_support)
    assert two_steps.m_refute == pytest.approx(one_step.m_refute)
    assert two_steps.m_theta == pytest.approx(one_step.m_theta)
    assert two_steps.reliability_applied == pytest.approx(first * second)


# --------------------------------------------------------------------------
# 8-10. 参数校验与不可变性
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [-0.1, -1.0])
def test_negative_reliability_is_rejected(bad_value: float) -> None:
    """reliability 小于 0 时被拒绝。"""
    with pytest.raises(ValueError, match="reliability 必须位于"):
        discount_mass(_mass(), bad_value)


@pytest.mark.parametrize("bad_value", [1.1, 2.0])
def test_reliability_above_one_is_rejected(bad_value: float) -> None:
    """reliability 大于 1 时被拒绝。"""
    with pytest.raises(ValueError, match="reliability 必须位于"):
        discount_mass(_mass(), bad_value)


def test_nan_reliability_is_rejected() -> None:
    """NaN 同样被拒绝，而不是悄悄传播。"""
    with pytest.raises(ValueError, match="reliability 必须位于"):
        discount_mass(_mass(), float("nan"))


def test_discount_does_not_modify_the_input() -> None:
    """discount_mass 不修改传入对象，返回的是新实例。"""
    mass = _mass(0.6, 0.1, 0.3)
    before = mass.model_dump()

    result = discount_mass(mass, 0.5)

    assert mass.model_dump() == before
    assert result is not mass


# --------------------------------------------------------------------------
# 11-12. 两个可靠性的组合与数值示例
# --------------------------------------------------------------------------


def test_document_stage_applies_only_document_reliability() -> None:
    """文档级折扣只应用 context.reliability，完全不看 evaluator_reliability。"""
    context = _context(reliability=0.9)

    results = [
        document_discounted_mass_from_prediction(
            _prediction(evaluator_reliability=evaluator_reliability), context
        )
        for evaluator_reliability in (0.0, 0.3, 0.8, 1.0)
    ]

    for result in results:
        assert result.reliability_applied == pytest.approx(0.9)
        assert result == results[0]


def test_document_level_worked_numeric_example() -> None:
    """文档级算例：0.8/0.1/0.1，文档可靠性 0.9 得到 0.72/0.09/0.19。"""
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=0.8)
    context = _context(reliability=0.9)

    result = document_discounted_mass_from_prediction(prediction, context)

    assert result.m_support == pytest.approx(0.72)
    assert result.m_refute == pytest.approx(0.09)
    assert result.m_theta == pytest.approx(0.19)
    assert result.reliability_applied == pytest.approx(0.9)
    assert result.m_support + result.m_refute + result.m_theta == pytest.approx(1.0)


def test_full_chain_document_then_evaluator_discount() -> None:
    """完整两级顺序：文档折扣 0.9 得 (0.72, 0.09, 0.19)，再评估器折扣 0.8。"""
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=0.8)
    context = _context(reliability=0.9)

    document_mass = document_discounted_mass_from_prediction(prediction, context)
    assert document_mass.m_support == pytest.approx(0.72)
    assert document_mass.m_refute == pytest.approx(0.09)
    assert document_mass.m_theta == pytest.approx(0.19)

    evaluator_mass = discount_combined_mass(
        CombinedMass(
            m_support=document_mass.m_support,
            m_refute=document_mass.m_refute,
            m_theta=document_mass.m_theta,
        ),
        prediction.evaluator_reliability,
    )

    assert evaluator_mass.m_support == pytest.approx(0.576)
    assert evaluator_mass.m_refute == pytest.approx(0.072)
    assert evaluator_mass.m_theta == pytest.approx(0.352)


# --------------------------------------------------------------------------
# 13-15. 配错文档、retrieval_score 与 gold_state
# --------------------------------------------------------------------------


def test_mismatched_doc_id_is_rejected() -> None:
    """prediction 与 context 的 doc_id 不一致时报错，信息含两个 ID。"""
    prediction = _prediction(doc_id="d1")
    context = _context(doc_id="d2")

    with pytest.raises(ValueError) as excinfo:
        document_discounted_mass_from_prediction(prediction, context)

    message = str(excinfo.value)
    assert "d1" in message
    assert "d2" in message


@pytest.mark.parametrize("retrieval_score", [None, 0.0, 0.3, 1.0])
def test_retrieval_score_does_not_affect_the_bpa(
    retrieval_score: float | None,
) -> None:
    """retrieval_score 不参与可靠性计算，改变它不会改变 BPA。"""
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=0.8)

    baseline = document_discounted_mass_from_prediction(
        prediction, _context(reliability=0.9, retrieval_score=None)
    )
    variant = document_discounted_mass_from_prediction(
        prediction, _context(reliability=0.9, retrieval_score=retrieval_score)
    )

    assert variant == baseline


def test_retrieval_score_is_not_used_as_reliability() -> None:
    """高 retrieval_score 不能替代低 reliability。"""
    prediction = _prediction(0.8, 0.1, 0.1)

    result = document_discounted_mass_from_prediction(
        prediction, _context(reliability=0.5, retrieval_score=1.0)
    )

    # 若误用 retrieval_score=1.0 当可靠性，m_support 会是 0.8。
    assert result.m_support == pytest.approx(0.4)
    assert result.reliability_applied == pytest.approx(0.5)


@pytest.mark.parametrize(
    "gold_state",
    [None, EvidenceState.SUPPORTED, EvidenceState.REFUTED, EvidenceState.CONFLICTING],
)
def test_gold_state_does_not_affect_the_bpa(gold_state: EvidenceState | None) -> None:
    """改变样本标签不会改变 BPA —— 折扣链路上根本读不到 gold_state。"""
    context = _context(reliability=0.9)
    sample = RAGSample(
        sample_id="s1",
        question="问题？",
        answer="答案。",
        claims=[Claim(claim_id="c1", text="断言。")],
        contexts=[context],
        gold_state=gold_state,
    )
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=0.8)

    result = document_discounted_mass_from_prediction(prediction, sample.contexts[0])

    assert result.m_support == pytest.approx(0.72)
    assert result.m_refute == pytest.approx(0.09)
    assert result.m_theta == pytest.approx(0.19)


# --------------------------------------------------------------------------
# 16-18. 不变量
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reliability", RELIABILITIES)
def test_total_ignorance_stays_total_under_any_discount(reliability: float) -> None:
    """p_unknown=1 时，任何可靠性下 m_theta 都是 1。"""
    mass = mass_from_prediction(_prediction(0.0, 0.0, 1.0))

    result = discount_mass(mass, reliability)

    assert result.m_theta == pytest.approx(1.0)
    assert result.m_support == pytest.approx(0.0)
    assert result.m_refute == pytest.approx(0.0)


def test_masses_stay_in_unit_interval_and_sum_to_one() -> None:
    """遍历概率与可靠性网格，质量始终位于 [0, 1] 且和为 1。"""
    checked = 0
    for triple in PROBABILITY_TRIPLES:
        for doc_reliability in RELIABILITIES:
            for evaluator_reliability in RELIABILITIES:
                prediction = _prediction(
                    *triple, evaluator_reliability=evaluator_reliability
                )
                result = document_discounted_mass_from_prediction(
                    prediction, _context(reliability=doc_reliability)
                )

                for value in (result.m_support, result.m_refute, result.m_theta):
                    assert 0.0 <= value <= 1.0
                total = result.m_support + result.m_refute + result.m_theta
                assert total == pytest.approx(1.0)
                assert result.reliability_applied == pytest.approx(doc_reliability)
                checked += 1

    assert checked == len(PROBABILITY_TRIPLES) * len(RELIABILITIES) ** 2


def test_discounted_mass_matches_manual_formula() -> None:
    """与手写公式逐点比对，确认实现没有走样。"""
    for triple in PROBABILITY_TRIPLES:
        p_support, p_refute, _ = triple
        for doc_reliability in RELIABILITIES:
            for evaluator_reliability in RELIABILITIES:
                effective = doc_reliability
                result = document_discounted_mass_from_prediction(
                    _prediction(*triple, evaluator_reliability=evaluator_reliability),
                    _context(reliability=doc_reliability),
                )

                assert result.m_support == pytest.approx(effective * p_support)
                assert result.m_refute == pytest.approx(effective * p_refute)
                assert result.m_theta == pytest.approx(
                    1.0 - effective * p_support - effective * p_refute
                )


# --------------------------------------------------------------------------
# 第八阶段：评估器级折扣 discount_combined_mass
# --------------------------------------------------------------------------


def _combined(
    m_support: float = 0.6, m_refute: float = 0.1, m_theta: float = 0.3
) -> CombinedMass:
    """构造一个融合后的 BPA。"""
    return CombinedMass(m_support=m_support, m_refute=m_refute, m_theta=m_theta)


def test_combined_discount_reliability_one_is_identity() -> None:
    """reliability=1 时融合质量不变。"""
    mass = _combined()

    result = discount_combined_mass(mass, 1.0)

    assert result.m_support == pytest.approx(mass.m_support)
    assert result.m_refute == pytest.approx(mass.m_refute)
    assert result.m_theta == pytest.approx(mass.m_theta)


def test_combined_discount_reliability_zero_gives_full_ignorance() -> None:
    """reliability=0 时质量全部进入 Theta。"""
    result = discount_combined_mass(_combined(), 0.0)

    assert result.m_support == pytest.approx(0.0)
    assert result.m_refute == pytest.approx(0.0)
    assert result.m_theta == pytest.approx(1.0)


def test_combined_discount_half() -> None:
    """reliability=0.5 时确定质量减半。"""
    result = discount_combined_mass(_combined(0.6, 0.1, 0.3), 0.5)

    assert result.m_support == pytest.approx(0.3)
    assert result.m_refute == pytest.approx(0.05)
    assert result.m_theta == pytest.approx(0.65)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, float("nan")])
def test_combined_discount_rejects_invalid_reliability(bad_value: float) -> None:
    """reliability 超出 [0, 1] 或为 NaN 时被拒绝。"""
    with pytest.raises(ValueError, match="reliability 必须位于"):
        discount_combined_mass(_combined(), bad_value)


def test_combined_discount_does_not_modify_input() -> None:
    """discount_combined_mass 不修改传入对象。"""
    mass = _combined(0.6, 0.1, 0.3)
    before = mass.model_dump()

    result = discount_combined_mass(mass, 0.5)

    assert mass.model_dump() == before
    assert result is not mass


@pytest.mark.parametrize("reliability", RELIABILITIES)
@pytest.mark.parametrize("triple", [(0.6, 0.1, 0.3), (0.1, 0.6, 0.3), (0.0, 0.0, 1.0)])
def test_both_discount_functions_share_the_same_maths(
    triple: tuple[float, float, float], reliability: float
) -> None:
    """discount_mass 与 discount_combined_mass 数值完全一致。"""
    with_ids = discount_mass(_mass(*triple), reliability)
    combined = discount_combined_mass(_combined(*triple), reliability)

    assert combined.m_support == pytest.approx(with_ids.m_support)
    assert combined.m_refute == pytest.approx(with_ids.m_refute)
    assert combined.m_theta == pytest.approx(with_ids.m_theta)


# --------------------------------------------------------------------------
# 第八阶段：旧接口已废弃
# --------------------------------------------------------------------------


def test_old_function_name_warns_and_no_longer_applies_evaluator_reliability() -> None:
    """旧函数发出 DeprecationWarning，且不再重复应用评估器可靠性。"""
    prediction = _prediction(0.8, 0.1, 0.1, evaluator_reliability=0.8)
    context = _context(reliability=0.9)

    with pytest.warns(DeprecationWarning, match="document_discounted_mass_from_prediction"):
        legacy = discounted_mass_from_prediction(prediction, context)

    assert legacy == document_discounted_mass_from_prediction(prediction, context)
    assert legacy.m_support == pytest.approx(0.72)  # 而不是旧行为的 0.576
    assert legacy.reliability_applied == pytest.approx(0.9)


def test_effective_reliability_is_deprecated() -> None:
    """effective_reliability 已废弃：两级可靠性不应在同一处相乘。"""
    with pytest.warns(DeprecationWarning, match="重复计入"):
        value = effective_reliability(
            _prediction(evaluator_reliability=0.8), _context(reliability=0.9)
        )

    assert value == pytest.approx(0.72)
