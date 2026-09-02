"""第八阶段多评估器融合、K_eval 与加权 K_doc 的测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.discount import document_discounted_mass_from_prediction
from rag_ds.ds.document_aggregation import (
    DocumentAggregationResult,
    EmptyEvidenceError,
    aggregate_document_masses,
)
from rag_ds.ds.evaluator_aggregation import (
    EvaluatorAggregationResult,
    EvaluatorEvidence,
    UndefinedDocumentMassError,
    aggregate_evaluators,
)
from rag_ds.ds.mass import MassFunction

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_PATH = _DATA_DIR / "demo.jsonl"
MOCK_PATH = _DATA_DIR / "mock_relations.jsonl"


def _mass(
    doc_id: str,
    m_support: float,
    m_refute: float,
    m_theta: float,
    evaluator: str = "eval_a",
    sample_id: str = "s1",
    claim_id: str = "c1",
) -> MassFunction:
    """构造一条已完成文档可靠性折扣的 BPA。"""
    return MassFunction(
        sample_id=sample_id,
        claim_id=claim_id,
        doc_id=doc_id,
        evaluator=evaluator,
        m_support=m_support,
        m_refute=m_refute,
        m_theta=m_theta,
    )


def _evidence(
    evaluator: str,
    triples: list[tuple[str, float, float, float]],
    evaluator_reliability: float = 1.0,
    sample_id: str = "s1",
    claim_id: str = "c1",
) -> EvaluatorEvidence:
    """由若干文档 BPA 先做文档聚合，再包成一条评估器证据。"""
    document_result = aggregate_document_masses(
        [
            _mass(doc_id, s, r, t, evaluator=evaluator, sample_id=sample_id, claim_id=claim_id)
            for doc_id, s, r, t in triples
        ]
    )
    return EvaluatorEvidence(
        document_result=document_result,
        evaluator_reliability=evaluator_reliability,
    )


def _simple_evidence(
    evaluator: str,
    m_support: float,
    m_refute: float,
    m_theta: float,
    evaluator_reliability: float = 1.0,
    sample_id: str = "s1",
    claim_id: str = "c1",
) -> EvaluatorEvidence:
    """单文档的评估器证据，K_doc 必为 0。"""
    return _evidence(
        evaluator,
        [(f"{evaluator}-d1", m_support, m_refute, m_theta)],
        evaluator_reliability=evaluator_reliability,
        sample_id=sample_id,
        claim_id=claim_id,
    )


def _expected_k(result: EvaluatorAggregationResult) -> float:
    """按定义 1 - ∏(1-K_i) 手工复算 K_eval。"""
    product = 1.0
    for step in result.steps:
        product *= 1.0 - step.conflict
    return 1.0 - product


# --------------------------------------------------------------------------
# 1-5. 空输入与单评估器
# --------------------------------------------------------------------------


def test_empty_input_is_rejected() -> None:
    """空输入被拒绝，不返回全无知 BPA。"""
    with pytest.raises(EmptyEvidenceError):
        aggregate_evaluators([])


def test_single_evaluator_k_eval_is_zero() -> None:
    """单评估器时 K_eval 为 0 且没有融合步骤。"""
    result = aggregate_evaluators([_simple_evidence("eval_a", 0.6, 0.1, 0.3)])

    assert result.k_eval == pytest.approx(0.0)
    assert result.steps == ()
    assert result.evaluators == ("eval_a",)
    assert result.is_total_conflict is False


def test_single_evaluator_applies_reliability_exactly_once() -> None:
    """单评估器时质量恰好被折扣一次。"""
    result = aggregate_evaluators(
        [_simple_evidence("eval_a", 0.8, 0.1, 0.1, evaluator_reliability=0.5)]
    )

    assert result.mass is not None
    assert result.mass.m_support == pytest.approx(0.4)
    assert result.mass.m_refute == pytest.approx(0.05)
    assert result.mass.m_theta == pytest.approx(0.55)


def test_reliability_one_leaves_mass_unchanged() -> None:
    """evaluator_reliability=1 时质量不变。"""
    result = aggregate_evaluators(
        [_simple_evidence("eval_a", 0.6, 0.1, 0.3, evaluator_reliability=1.0)]
    )

    assert result.mass is not None
    assert result.mass.m_support == pytest.approx(0.6)
    assert result.mass.m_refute == pytest.approx(0.1)
    assert result.mass.m_theta == pytest.approx(0.3)


def test_reliability_zero_moves_everything_to_theta() -> None:
    """evaluator_reliability=0 时质量全部进入 Theta。"""
    result = aggregate_evaluators(
        [_simple_evidence("eval_a", 0.8, 0.1, 0.1, evaluator_reliability=0.0)]
    )

    assert result.mass is not None
    assert result.mass.m_support == pytest.approx(0.0)
    assert result.mass.m_refute == pytest.approx(0.0)
    assert result.mass.m_theta == pytest.approx(1.0)
    assert result.k_eval == pytest.approx(0.0)
    assert result.k_doc_weighted == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 6-12. 多评估器融合与 K_eval
# --------------------------------------------------------------------------


def test_agreeing_evaluators_give_low_k_eval() -> None:
    """两个意见一致的评估器产生较低 K_eval。"""
    result = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.8, 0.1, 0.1),
            _simple_evidence("eval_b", 0.8, 0.1, 0.1),
        ]
    )

    assert result.k_eval == pytest.approx(0.16)
    assert result.mass is not None
    assert result.mass.m_support > 0.8


def test_opposing_evaluators_give_high_k_eval() -> None:
    """一个支持、一个反驳的评估器产生较高 K_eval。"""
    agreeing = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.8, 0.1, 0.1),
            _simple_evidence("eval_b", 0.8, 0.1, 0.1),
        ]
    )
    opposing = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.8, 0.1, 0.1),
            _simple_evidence("eval_b", 0.1, 0.8, 0.1),
        ]
    )

    assert opposing.k_eval == pytest.approx(0.65)
    assert opposing.k_eval > agreeing.k_eval


def test_three_evaluators_are_folded_in_order() -> None:
    """三个评估器可以依次融合。"""
    result = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.7, 0.1, 0.2),
            _simple_evidence("eval_b", 0.2, 0.6, 0.2),
            _simple_evidence("eval_c", 0.5, 0.2, 0.3),
        ]
    )

    assert result.evaluators == ("eval_a", "eval_b", "eval_c")
    assert len(result.steps) == 2
    assert result.mass is not None


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_step_count_is_evaluator_count_minus_one(count: int) -> None:
    """steps 数量等于评估器数量减 1。"""
    evidences = [
        _simple_evidence(f"eval_{i}", 0.6, 0.2, 0.2) for i in range(count)
    ]

    assert len(aggregate_evaluators(evidences).steps) == count - 1


def test_steps_record_incoming_and_accumulated_evaluators() -> None:
    """每一步记录正确的 incoming_evaluator 与此前的累计评估器。"""
    result = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.7, 0.1, 0.2),
            _simple_evidence("eval_b", 0.2, 0.6, 0.2),
            _simple_evidence("eval_c", 0.5, 0.2, 0.3),
        ]
    )

    first, second = result.steps
    assert first.step_index == 1
    assert first.accumulated_evaluators == ("eval_a",)
    assert first.incoming_evaluator == "eval_b"
    assert second.step_index == 2
    assert second.accumulated_evaluators == ("eval_a", "eval_b")
    assert second.incoming_evaluator == "eval_c"


def test_k_eval_matches_the_product_formula() -> None:
    """K_eval 等于 1 - ∏(1-K_i)，且不是各步的平均值。"""
    result = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.8, 0.1, 0.1),
            _simple_evidence("eval_b", 0.1, 0.8, 0.1),
            _simple_evidence("eval_c", 0.1, 0.8, 0.1),
        ]
    )

    conflicts = [step.conflict for step in result.steps]
    average = sum(conflicts) / len(conflicts)

    assert result.k_eval == pytest.approx(_expected_k(result))
    assert result.k_eval != pytest.approx(average)


def test_k_eval_in_range_and_masses_sum_to_one() -> None:
    """遍历多组输入，K_eval 在 [0, 1]，最终质量之和为 1。"""
    triples = [
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.0, 0.0, 1.0),
        (0.5, 0.2, 0.3),
        (0.34, 0.33, 0.33),
    ]
    for left in triples:
        for right in triples:
            result = aggregate_evaluators(
                [
                    _simple_evidence("eval_a", *left),
                    _simple_evidence("eval_b", *right),
                ]
            )

            assert 0.0 <= result.k_eval <= 1.0
            assert 0.0 <= result.k_doc_weighted <= 1.0
            assert result.mass is not None
            total = result.mass.m_support + result.mass.m_refute + result.mass.m_theta
            assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 13-17. 输入校验
# --------------------------------------------------------------------------


def test_mixed_sample_id_is_rejected() -> None:
    """混合 sample_id 时被拒绝。"""
    with pytest.raises(ValueError, match="sample_id"):
        aggregate_evaluators(
            [
                _simple_evidence("eval_a", 0.6, 0.2, 0.2),
                _simple_evidence("eval_b", 0.6, 0.2, 0.2, sample_id="s2"),
            ]
        )


def test_mixed_claim_id_is_rejected() -> None:
    """混合 claim_id 时被拒绝。"""
    with pytest.raises(ValueError, match="claim_id"):
        aggregate_evaluators(
            [
                _simple_evidence("eval_a", 0.6, 0.2, 0.2),
                _simple_evidence("eval_b", 0.6, 0.2, 0.2, claim_id="c2"),
            ]
        )


def test_duplicate_evaluator_name_is_rejected() -> None:
    """评估器名称重复时被拒绝。"""
    with pytest.raises(ValueError, match="evaluator"):
        aggregate_evaluators(
            [
                _simple_evidence("eval_a", 0.6, 0.2, 0.2),
                _simple_evidence("eval_a", 0.5, 0.3, 0.2),
            ]
        )


def _document_conflicted_evidence() -> EvaluatorEvidence:
    """文档级完全冲突（mass=None）的评估器证据。"""
    document_result = aggregate_document_masses(
        [
            _mass("d1", 1.0, 0.0, 0.0, evaluator="eval_b"),
            _mass("d2", 0.0, 1.0, 0.0, evaluator="eval_b"),
        ]
    )
    assert document_result.mass is None
    return EvaluatorEvidence(document_result=document_result, evaluator_reliability=0.9)


def test_undefined_document_mass_is_rejected() -> None:
    """document_result.mass=None 时抛出 UndefinedDocumentMassError。"""
    with pytest.raises(UndefinedDocumentMassError):
        aggregate_evaluators(
            [_simple_evidence("eval_a", 0.6, 0.2, 0.2), _document_conflicted_evidence()]
        )


def test_undefined_document_mass_error_reports_context() -> None:
    """异常包含 sample、claim、evaluator 与该评估器的 K_doc。"""
    with pytest.raises(UndefinedDocumentMassError) as excinfo:
        aggregate_evaluators([_document_conflicted_evidence()])

    error = excinfo.value
    assert error.sample_id == "s1"
    assert error.claim_id == "c1"
    assert error.evaluator == "eval_b"
    assert error.k_doc == pytest.approx(1.0)

    message = str(error)
    for token in ("s1", "c1", "eval_b", "K_doc"):
        assert token in message
    assert "无法计算 K_eval" in message


# --------------------------------------------------------------------------
# 18-20. 评估器级完全冲突
# --------------------------------------------------------------------------


def _totally_conflicting_evaluators() -> list[EvaluatorEvidence]:
    """一个完全支持、一个完全反驳，外加一个永远轮不到的评估器。"""
    return [
        _simple_evidence("eval_a", 1.0, 0.0, 0.0),
        _simple_evidence("eval_b", 0.0, 1.0, 0.0),
        _simple_evidence("eval_c", 0.5, 0.2, 0.3),
    ]


def test_total_conflict_result_fields() -> None:
    """评估器完全冲突时 mass=None、K_eval=1、is_total_conflict=True。"""
    result = aggregate_evaluators(_totally_conflicting_evaluators())

    assert result.mass is None
    assert result.k_eval == pytest.approx(1.0)
    assert result.is_total_conflict is True


def test_total_conflict_stops_further_evaluators() -> None:
    """完全冲突后不再融合剩余评估器，但诊断信息仍覆盖全部输入。"""
    result = aggregate_evaluators(_totally_conflicting_evaluators())

    assert len(result.steps) == 1
    assert result.steps[0].is_total_conflict is True
    assert result.steps[0].conflict == pytest.approx(1.0)
    assert result.steps[0].normalization_denominator == pytest.approx(0.0)
    assert result.steps[0].result_mass is None
    assert result.steps[0].incoming_evaluator == "eval_b"
    assert result.evaluators == ("eval_a", "eval_b", "eval_c")
    assert len(result.evaluator_diagnostics) == 3


def test_total_conflict_is_not_faked_as_full_ignorance() -> None:
    """完全冲突不会被伪造成 m_theta=1。"""
    assert aggregate_evaluators(_totally_conflicting_evaluators()).mass is None


# --------------------------------------------------------------------------
# 21-23. 顺序
# --------------------------------------------------------------------------


def test_input_order_is_preserved() -> None:
    """evaluators 保留输入顺序。"""
    result = aggregate_evaluators(
        [
            _simple_evidence("eval_z", 0.7, 0.1, 0.2),
            _simple_evidence("eval_a", 0.6, 0.2, 0.2),
        ]
    )

    assert result.evaluators == ("eval_z", "eval_a")


def test_reordering_gives_approximately_the_same_result() -> None:
    """改变顺序后最终质量与 K_eval 近似相同。"""
    evidences = [
        _simple_evidence("eval_a", 0.7, 0.1, 0.2, evaluator_reliability=0.9),
        _simple_evidence("eval_b", 0.2, 0.6, 0.2, evaluator_reliability=0.8),
        _simple_evidence("eval_c", 0.5, 0.2, 0.3, evaluator_reliability=1.0),
    ]
    forward = aggregate_evaluators(evidences)
    backward = aggregate_evaluators(list(reversed(evidences)))

    assert forward.mass is not None
    assert backward.mass is not None
    assert backward.mass.m_support == pytest.approx(forward.mass.m_support)
    assert backward.mass.m_refute == pytest.approx(forward.mass.m_refute)
    assert backward.mass.m_theta == pytest.approx(forward.mass.m_theta)
    assert backward.k_eval == pytest.approx(forward.k_eval)
    assert backward.k_doc_weighted == pytest.approx(forward.k_doc_weighted)


# --------------------------------------------------------------------------
# 24-27. K_doc 加权汇总与诊断
# --------------------------------------------------------------------------


def test_weighted_k_doc_is_computed_correctly() -> None:
    """k_doc_weighted = Σ(r_e × K_doc,e) / Σr_e。"""
    conflicted = _evidence(
        "eval_a",
        [("a-d1", 0.8, 0.1, 0.1), ("a-d2", 0.1, 0.8, 0.1)],
        evaluator_reliability=0.5,
    )
    clean = _simple_evidence("eval_b", 0.6, 0.2, 0.2, evaluator_reliability=1.0)

    result = aggregate_evaluators([conflicted, clean])

    k_doc_a = conflicted.document_result.k_doc
    expected = (0.5 * k_doc_a + 1.0 * 0.0) / (0.5 + 1.0)

    assert k_doc_a == pytest.approx(0.65)
    assert result.k_doc_weighted == pytest.approx(expected)


def test_weighted_k_doc_is_zero_when_all_reliabilities_are_zero() -> None:
    """所有评估器可靠性为零时 k_doc_weighted=0，且最终质量为完全无知。"""
    conflicted = _evidence(
        "eval_a",
        [("a-d1", 0.8, 0.1, 0.1), ("a-d2", 0.1, 0.8, 0.1)],
        evaluator_reliability=0.0,
    )
    other = _evidence(
        "eval_b",
        [("b-d1", 0.8, 0.1, 0.1), ("b-d2", 0.1, 0.8, 0.1)],
        evaluator_reliability=0.0,
    )

    result = aggregate_evaluators([conflicted, other])

    assert result.k_doc_weighted == pytest.approx(0.0)
    assert result.mass is not None
    assert result.mass.m_theta == pytest.approx(1.0)
    assert result.k_eval == pytest.approx(0.0)


def test_original_k_doc_is_preserved_per_evaluator() -> None:
    """每个评估器原始的 K_doc 都保留在诊断里，而不是只留加权平均。"""
    conflicted = _evidence(
        "eval_a",
        [("a-d1", 0.8, 0.1, 0.1), ("a-d2", 0.1, 0.8, 0.1)],
        evaluator_reliability=0.5,
    )
    clean = _simple_evidence("eval_b", 0.6, 0.2, 0.2)

    result = aggregate_evaluators([conflicted, clean])

    by_name = {d.evaluator: d for d in result.evaluator_diagnostics}
    assert by_name["eval_a"].k_doc == pytest.approx(0.65)
    assert by_name["eval_b"].k_doc == pytest.approx(0.0)
    assert by_name["eval_a"].document_ids == ("a-d1", "a-d2")
    assert result.k_doc_weighted != pytest.approx(by_name["eval_a"].k_doc)


def test_diagnostics_keep_mass_before_and_after_discount() -> None:
    """诊断同时保留评估器折扣前后的质量。"""
    result = aggregate_evaluators(
        [_simple_evidence("eval_a", 0.8, 0.1, 0.1, evaluator_reliability=0.5)]
    )

    diagnostic = result.evaluator_diagnostics[0]
    assert diagnostic.mass_before_evaluator_discount.m_support == pytest.approx(0.8)
    assert diagnostic.mass_after_evaluator_discount.m_support == pytest.approx(0.4)
    assert diagnostic.evaluator_reliability == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 28-32. 不修改输入 / 可靠性只用一次 / 无关字段
# --------------------------------------------------------------------------


def test_inputs_are_not_modified() -> None:
    """聚合不修改传入的证据对象。"""
    evidences = [
        _simple_evidence("eval_a", 0.7, 0.1, 0.2, evaluator_reliability=0.9),
        _simple_evidence("eval_b", 0.2, 0.6, 0.2, evaluator_reliability=0.8),
    ]
    before = [e.model_dump() for e in evidences]

    aggregate_evaluators(evidences)

    assert [e.model_dump() for e in evidences] == before


def test_evaluator_reliability_is_applied_exactly_once() -> None:
    """单评估器折扣一次的结果，等价于对文档级质量手工折扣一次。"""
    evidence = _simple_evidence("eval_a", 0.8, 0.1, 0.1, evaluator_reliability=0.6)

    result = aggregate_evaluators([evidence])

    assert evidence.document_result.mass is not None
    assert result.mass is not None
    assert result.mass.m_support == pytest.approx(0.6 * 0.8)
    # 折两次会得到 0.36 * 0.8 = 0.288，这里必须不是那个值。
    assert result.mass.m_support != pytest.approx(0.6 * 0.6 * 0.8)


@pytest.mark.parametrize("document_count", [1, 2, 3, 5])
def test_document_count_does_not_change_evaluator_discount(
    document_count: int,
) -> None:
    """文档数量增加不会让评估器可靠性被重复折扣。

    每条文档都用同一个 BPA，文档融合会让确定质量变高，但
    ``reliability_applied`` 式的评估器折扣始终只作用一次：
    折扣后与折扣前的 m_support 之比恒等于 evaluator_reliability。
    """
    reliability = 0.7
    evidence = _evidence(
        "eval_a",
        [(f"d{i}", 0.6, 0.1, 0.3) for i in range(document_count)],
        evaluator_reliability=reliability,
    )

    result = aggregate_evaluators([evidence])

    diagnostic = result.evaluator_diagnostics[0]
    before = diagnostic.mass_before_evaluator_discount
    after = diagnostic.mass_after_evaluator_discount

    assert after.m_support == pytest.approx(reliability * before.m_support)
    assert after.m_refute == pytest.approx(reliability * before.m_refute)


def test_retrieval_score_and_gold_state_never_reach_this_layer() -> None:
    """本层的输入模型里根本没有 retrieval_score 或 gold_state 字段。"""
    fields = (
        set(EvaluatorEvidence.model_fields)
        | set(DocumentAggregationResult.model_fields)
        | set(EvaluatorAggregationResult.model_fields)
    )

    assert "retrieval_score" not in fields
    assert "gold_state" not in fields


# --------------------------------------------------------------------------
# 结果模型自身的约束
# --------------------------------------------------------------------------


def test_result_is_immutable() -> None:
    """EvaluatorAggregationResult 不可变。"""
    result = aggregate_evaluators([_simple_evidence("eval_a", 0.6, 0.2, 0.2)])

    with pytest.raises(ValidationError):
        result.k_eval = 1.0  # type: ignore[misc]


def test_result_model_rejects_inconsistent_total_conflict() -> None:
    """手工构造时，完全冲突与 mass / k_eval 必须一致。"""
    with pytest.raises(ValidationError, match="完全冲突时 mass 必须为 None"):
        EvaluatorAggregationResult(
            sample_id="s1",
            claim_id="c1",
            evaluators=("eval_a",),
            mass=CombinedMass(m_support=0.5, m_refute=0.2, m_theta=0.3),
            k_eval=1.0,
            k_doc_weighted=0.0,
            evaluator_diagnostics=(),
            steps=(),
            is_total_conflict=True,
        )

    with pytest.raises(ValidationError, match="mass 不能为 None"):
        EvaluatorAggregationResult(
            sample_id="s1",
            claim_id="c1",
            evaluators=("eval_a",),
            mass=None,
            k_eval=0.0,
            k_doc_weighted=0.0,
            evaluator_diagnostics=(),
            steps=(),
            is_total_conflict=False,
        )


def test_result_model_rejects_duplicate_evaluators() -> None:
    """结果模型拒绝重复的评估器名称。"""
    with pytest.raises(ValidationError, match="不允许重复"):
        EvaluatorAggregationResult(
            sample_id="s1",
            claim_id="c1",
            evaluators=("eval_a", "eval_a"),
            mass=CombinedMass(m_support=0.5, m_refute=0.2, m_theta=0.3),
            k_eval=0.0,
            k_doc_weighted=0.0,
            evaluator_diagnostics=(),
            steps=(),
            is_total_conflict=False,
        )


# --------------------------------------------------------------------------
# 三个指标互不等价
# --------------------------------------------------------------------------


def test_k_doc_k_eval_and_theta_are_independent() -> None:
    """K_doc 高、K_eval 高、m_theta 高可以互相独立出现。"""
    # 两个评估器各自内部有文档冲突，但彼此意见一致 -> K_doc 高、K_eval 低。
    internally_conflicted = aggregate_evaluators(
        [
            _evidence("eval_a", [("a1", 0.8, 0.1, 0.1), ("a2", 0.1, 0.8, 0.1)]),
            _evidence("eval_b", [("b1", 0.8, 0.1, 0.1), ("b2", 0.1, 0.8, 0.1)]),
        ]
    )
    assert internally_conflicted.k_doc_weighted == pytest.approx(0.65)
    assert internally_conflicted.k_eval < internally_conflicted.k_doc_weighted

    # 两个评估器各自内部无冲突，但彼此对立 -> K_doc 为 0、K_eval 高。
    mutually_opposed = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.8, 0.1, 0.1),
            _simple_evidence("eval_b", 0.1, 0.8, 0.1),
        ]
    )
    assert mutually_opposed.k_doc_weighted == pytest.approx(0.0)
    assert mutually_opposed.k_eval == pytest.approx(0.65)

    # 证据都很弱 -> m_theta 高，两个 K 都为 0。
    weak = aggregate_evaluators(
        [
            _simple_evidence("eval_a", 0.05, 0.0, 0.95),
            _simple_evidence("eval_b", 0.05, 0.0, 0.95),
        ]
    )
    assert weak.mass is not None
    assert weak.mass.m_theta > 0.9
    assert weak.k_doc_weighted == pytest.approx(0.0)
    assert weak.k_eval == pytest.approx(0.0)


# --------------------------------------------------------------------------
# demo 数据的端到端串联（仅测试内手工串联，不实现正式 pipeline）
# --------------------------------------------------------------------------


def test_demo_end_to_end_chain() -> None:
    """走完整链路：文档折扣 -> 文档聚合 -> 评估器折扣 -> 评估器聚合。"""
    samples = {s.sample_id: s for s in load_samples(DEMO_PATH)}
    predictions = load_relation_predictions(MOCK_PATH)
    sample = samples["demo-004"]
    contexts = {chunk.doc_id: chunk for chunk in sample.contexts}

    document_masses = [
        document_discounted_mass_from_prediction(p, contexts[p.doc_id])
        for p in predictions
        if p.sample_id == "demo-004"
    ]
    document_result = aggregate_document_masses(document_masses)

    result = aggregate_evaluators(
        [
            EvaluatorEvidence(
                document_result=document_result, evaluator_reliability=0.8
            )
        ]
    )

    assert result.evaluators == ("mock_evaluator",)
    assert result.k_eval == pytest.approx(0.0)  # 只有一个评估器
    assert result.k_doc_weighted == pytest.approx(document_result.k_doc)
    assert result.k_doc_weighted > 0.4  # conflicting 样例内部文档冲突明显
    assert result.mass is not None
    total = result.mass.m_support + result.mass.m_refute + result.mass.m_theta
    assert total == pytest.approx(1.0)
