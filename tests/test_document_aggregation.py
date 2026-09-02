"""第七阶段多文档融合与 K_doc 的测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.discount import document_discounted_mass_from_prediction
from rag_ds.ds.document_aggregation import (
    DocumentAggregationResult,
    DocumentCombinationStep,
    EmptyEvidenceError,
    aggregate_document_masses,
)
from rag_ds.ds.mass import MassFunction
from rag_ds.schemas import RAGSample, RelationPrediction

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_PATH = _DATA_DIR / "demo.jsonl"
MOCK_PATH = _DATA_DIR / "mock_relations.jsonl"


def _mass(
    doc_id: str,
    m_support: float,
    m_refute: float,
    m_theta: float,
    sample_id: str = "s1",
    claim_id: str = "c1",
    evaluator: str = "mock_evaluator",
    reliability_applied: float = 1.0,
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
        reliability_applied=reliability_applied,
    )


def _expected_k_doc(result: DocumentAggregationResult) -> float:
    """按定义 1 - ∏(1-K_i) 手工复算 K_doc。"""
    product = 1.0
    for step in result.steps:
        product *= 1.0 - step.conflict
    return 1.0 - product


# --------------------------------------------------------------------------
# 1-4. 空输入与单文档
# --------------------------------------------------------------------------


def test_empty_input_is_rejected() -> None:
    """空输入抛出 EmptyEvidenceError，不返回全无知 BPA。"""
    with pytest.raises(EmptyEvidenceError):
        aggregate_document_masses([])


def test_single_document_returns_the_original_masses() -> None:
    """单文档时三个质量原样复制为 CombinedMass。"""
    single = _mass("d1", 0.6, 0.1, 0.3)

    result = aggregate_document_masses([single])

    assert isinstance(result.mass, CombinedMass)
    assert result.mass.m_support == pytest.approx(0.6)
    assert result.mass.m_refute == pytest.approx(0.1)
    assert result.mass.m_theta == pytest.approx(0.3)
    assert result.document_ids == ("d1",)
    assert result.sample_id == "s1"
    assert result.claim_id == "c1"
    assert result.evaluator == "mock_evaluator"
    assert result.is_total_conflict is False


def test_single_document_k_doc_is_zero() -> None:
    """单文档没有融合步骤，K_doc 为 0。"""
    assert aggregate_document_masses([_mass("d1", 0.6, 0.1, 0.3)]).k_doc == (
        pytest.approx(0.0)
    )


def test_single_document_has_no_steps() -> None:
    """单文档时 steps 为空元组。"""
    assert aggregate_document_masses([_mass("d1", 0.6, 0.1, 0.3)]).steps == ()


# --------------------------------------------------------------------------
# 5-9. 融合行为
# --------------------------------------------------------------------------


def test_two_supporting_documents_increase_support() -> None:
    """两条支持文档融合后支持质量提高。"""
    result = aggregate_document_masses(
        [_mass("d1", 0.6, 0.1, 0.3), _mass("d2", 0.6, 0.1, 0.3)]
    )

    assert result.mass is not None
    assert result.mass.m_support > 0.6
    assert result.mass.m_theta < 0.3


def test_two_refuting_documents_increase_refute() -> None:
    """两条反驳文档融合后反驳质量提高。"""
    result = aggregate_document_masses(
        [_mass("d1", 0.1, 0.6, 0.3), _mass("d2", 0.1, 0.6, 0.3)]
    )

    assert result.mass is not None
    assert result.mass.m_refute > 0.6
    assert result.mass.m_theta < 0.3


def test_opposing_documents_produce_high_k_doc() -> None:
    """一条支持、一条反驳的文档产生较高 K_doc。"""
    agreeing = aggregate_document_masses(
        [_mass("d1", 0.8, 0.1, 0.1), _mass("d2", 0.8, 0.1, 0.1)]
    )
    opposing = aggregate_document_masses(
        [_mass("d1", 0.8, 0.1, 0.1), _mass("d2", 0.1, 0.8, 0.1)]
    )

    assert opposing.k_doc == pytest.approx(0.65)
    assert opposing.k_doc > agreeing.k_doc


def test_vacuous_document_does_not_add_conflict() -> None:
    """完全无知的文档不会增加 K_doc，也不改变累计 BPA。"""
    baseline = aggregate_document_masses([_mass("d1", 0.8, 0.1, 0.1)])
    with_vacuous = aggregate_document_masses(
        [_mass("d1", 0.8, 0.1, 0.1), _mass("d2", 0.0, 0.0, 1.0)]
    )

    assert with_vacuous.k_doc == pytest.approx(0.0)
    assert baseline.mass is not None
    assert with_vacuous.mass is not None
    assert with_vacuous.mass.m_support == pytest.approx(baseline.mass.m_support)
    assert with_vacuous.mass.m_theta == pytest.approx(baseline.mass.m_theta)


def test_three_documents_are_folded_in_order() -> None:
    """三条文档可以依次融合。"""
    result = aggregate_document_masses(
        [
            _mass("d1", 0.7, 0.1, 0.2),
            _mass("d2", 0.6, 0.2, 0.2),
            _mass("d3", 0.5, 0.1, 0.4),
        ]
    )

    assert result.mass is not None
    assert len(result.steps) == 2
    assert result.document_ids == ("d1", "d2", "d3")


# --------------------------------------------------------------------------
# 10-15. 步骤记录与 K_doc
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_step_count_is_document_count_minus_one(count: int) -> None:
    """steps 数量等于文档数量减 1。"""
    masses = [_mass(f"d{i}", 0.7, 0.1, 0.2) for i in range(count)]

    assert len(aggregate_document_masses(masses).steps) == count - 1


def test_steps_record_incoming_and_accumulated_ids() -> None:
    """每一步记录正确的 incoming_doc_id 与此前的累计文档 ID。"""
    result = aggregate_document_masses(
        [
            _mass("d1", 0.7, 0.1, 0.2),
            _mass("d2", 0.6, 0.2, 0.2),
            _mass("d3", 0.5, 0.1, 0.4),
        ]
    )

    first, second = result.steps
    assert first.step_index == 1
    assert first.accumulated_doc_ids == ("d1",)
    assert first.incoming_doc_id == "d2"
    assert second.step_index == 2
    assert second.accumulated_doc_ids == ("d1", "d2")
    assert second.incoming_doc_id == "d3"


def test_steps_record_conflict_and_denominator() -> None:
    """每一步保存 K_i 与 1-K_i，且二者一致。"""
    result = aggregate_document_masses(
        [_mass("d1", 0.8, 0.1, 0.1), _mass("d2", 0.1, 0.8, 0.1)]
    )

    step = result.steps[0]
    assert step.conflict == pytest.approx(0.65)
    assert step.normalization_denominator == pytest.approx(0.35)
    assert step.result_mass is not None
    assert step.is_total_conflict is False


def test_k_doc_matches_the_product_formula() -> None:
    """K_doc 等于 1 - ∏(1-K_i)。"""
    for masses in (
        [_mass("d1", 0.8, 0.1, 0.1), _mass("d2", 0.1, 0.8, 0.1)],
        [
            _mass("d1", 0.7, 0.2, 0.1),
            _mass("d2", 0.2, 0.7, 0.1),
            _mass("d3", 0.5, 0.3, 0.2),
        ],
        [_mass(f"d{i}", 0.6, 0.3, 0.1) for i in range(4)],
    ):
        result = aggregate_document_masses(masses)

        assert result.k_doc == pytest.approx(_expected_k_doc(result))


def test_k_doc_matches_the_spec_worked_example() -> None:
    """规格示例：K1=0.2、K2=0.3 时 K_doc = 1 - 0.8×0.7 = 0.44。"""
    product = (1.0 - 0.2) * (1.0 - 0.3)

    assert 1.0 - product == pytest.approx(0.44)


def test_k_doc_is_not_the_average_of_steps() -> None:
    """K_doc 不是各步 K_i 的简单平均值。"""
    result = aggregate_document_masses(
        [
            _mass("d1", 0.8, 0.1, 0.1),
            _mass("d2", 0.1, 0.8, 0.1),
            _mass("d3", 0.1, 0.8, 0.1),
        ]
    )

    conflicts = [step.conflict for step in result.steps]
    average = sum(conflicts) / len(conflicts)

    assert result.k_doc == pytest.approx(_expected_k_doc(result))
    assert result.k_doc != pytest.approx(average)


def test_k_doc_stays_in_unit_interval_and_masses_sum_to_one() -> None:
    """遍历多组输入，K_doc 在 [0, 1]，最终质量之和为 1。"""
    triples = [
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.0, 0.0, 1.0),
        (0.5, 0.2, 0.3),
        (0.34, 0.33, 0.33),
    ]
    for left in triples:
        for right in triples:
            result = aggregate_document_masses(
                [_mass("d1", *left), _mass("d2", *right)]
            )

            assert 0.0 <= result.k_doc <= 1.0
            assert result.mass is not None
            total = (
                result.mass.m_support + result.mass.m_refute + result.mass.m_theta
            )
            assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 16-19. 输入一致性校验
# --------------------------------------------------------------------------


def test_mixed_sample_id_is_rejected() -> None:
    """混合 sample_id 时抛出 ValueError。"""
    with pytest.raises(ValueError, match="sample_id"):
        aggregate_document_masses(
            [_mass("d1", 0.6, 0.1, 0.3), _mass("d2", 0.6, 0.1, 0.3, sample_id="s2")]
        )


def test_mixed_claim_id_is_rejected() -> None:
    """混合 claim_id 时抛出 ValueError。"""
    with pytest.raises(ValueError, match="claim_id"):
        aggregate_document_masses(
            [_mass("d1", 0.6, 0.1, 0.3), _mass("d2", 0.6, 0.1, 0.3, claim_id="c2")]
        )


def test_mixed_evaluator_is_rejected() -> None:
    """混合 evaluator 时抛出 ValueError。"""
    with pytest.raises(ValueError, match="evaluator"):
        aggregate_document_masses(
            [_mass("d1", 0.6, 0.1, 0.3), _mass("d2", 0.6, 0.1, 0.3, evaluator="other")]
        )


def test_duplicate_doc_id_is_rejected() -> None:
    """重复 doc_id 时抛出 ValueError。"""
    with pytest.raises(ValueError, match="doc_id"):
        aggregate_document_masses(
            [_mass("d1", 0.6, 0.1, 0.3), _mass("d1", 0.5, 0.2, 0.3)]
        )


# --------------------------------------------------------------------------
# 20-22. 顺序
# --------------------------------------------------------------------------


def test_input_order_is_preserved_in_document_ids() -> None:
    """document_ids 保留输入顺序，不按 doc_id 重排。"""
    result = aggregate_document_masses(
        [
            _mass("d3", 0.7, 0.1, 0.2),
            _mass("d1", 0.6, 0.2, 0.2),
            _mass("d2", 0.5, 0.1, 0.4),
        ]
    )

    assert result.document_ids == ("d3", "d1", "d2")


def test_reordering_inputs_gives_approximately_the_same_result() -> None:
    """改变输入顺序后最终 BPA 与 K_doc 近似相同。"""
    masses = [
        _mass("d1", 0.7, 0.1, 0.2),
        _mass("d2", 0.2, 0.6, 0.2),
        _mass("d3", 0.5, 0.2, 0.3),
    ]
    forward = aggregate_document_masses(masses)
    reversed_result = aggregate_document_masses(list(reversed(masses)))

    assert forward.mass is not None
    assert reversed_result.mass is not None
    assert reversed_result.mass.m_support == pytest.approx(forward.mass.m_support)
    assert reversed_result.mass.m_refute == pytest.approx(forward.mass.m_refute)
    assert reversed_result.mass.m_theta == pytest.approx(forward.mass.m_theta)
    assert reversed_result.k_doc == pytest.approx(forward.k_doc)


# --------------------------------------------------------------------------
# 23-28. 完全冲突
# --------------------------------------------------------------------------


def _totally_conflicting() -> list[MassFunction]:
    """一条完全支持、一条完全反驳，外加一条永远轮不到的文档。"""
    return [
        _mass("d1", 1.0, 0.0, 0.0),
        _mass("d2", 0.0, 1.0, 0.0),
        _mass("d3", 0.5, 0.2, 0.3),
    ]


def test_total_conflict_result_fields() -> None:
    """完全冲突时 mass=None、k_doc=1、is_total_conflict=True。"""
    result = aggregate_document_masses(_totally_conflicting())

    assert result.mass is None
    assert result.k_doc == pytest.approx(1.0)
    assert result.is_total_conflict is True


def test_total_conflict_step_is_recorded() -> None:
    """完全冲突的那一步被完整记录。"""
    result = aggregate_document_masses(_totally_conflicting())

    step = result.steps[-1]
    assert step.is_total_conflict is True
    assert step.conflict == pytest.approx(1.0)
    assert step.normalization_denominator == pytest.approx(0.0)
    assert step.result_mass is None
    assert step.incoming_doc_id == "d2"


def test_total_conflict_stops_further_documents() -> None:
    """完全冲突后不再处理剩余文档，但已知的文档 ID 全部保留。"""
    result = aggregate_document_masses(_totally_conflicting())

    assert len(result.steps) == 1
    assert result.document_ids == ("d1", "d2", "d3")


def test_total_conflict_is_not_faked_as_full_ignorance() -> None:
    """完全冲突不会被伪造成 m_theta=1 的全无知 BPA。"""
    result = aggregate_document_masses(_totally_conflicting())

    assert result.mass is None  # 而不是 CombinedMass(0, 0, 1)


# --------------------------------------------------------------------------
# 29-30. 输入不被修改 / 不重复折扣
# --------------------------------------------------------------------------


def test_inputs_are_not_modified() -> None:
    """聚合不修改传入的 MassFunction。"""
    masses = [_mass("d1", 0.7, 0.1, 0.2, reliability_applied=0.9), _mass("d2", 0.2, 0.6, 0.2)]
    before = [m.model_dump() for m in masses]

    aggregate_document_masses(masses)

    assert [m.model_dump() for m in masses] == before


@pytest.mark.parametrize("reliability_applied", [0.0, 0.3, 0.9, 1.0])
def test_reliability_applied_is_not_reapplied(reliability_applied: float) -> None:
    """已折扣标记不会被再次应用 —— 聚合层不碰可靠性。"""
    masses = [
        _mass("d1", 0.6, 0.1, 0.3, reliability_applied=reliability_applied),
        _mass("d2", 0.6, 0.1, 0.3, reliability_applied=reliability_applied),
    ]
    reference = aggregate_document_masses(
        [_mass("d1", 0.6, 0.1, 0.3), _mass("d2", 0.6, 0.1, 0.3)]
    )

    result = aggregate_document_masses(masses)

    assert result.mass == reference.mass
    assert result.k_doc == pytest.approx(reference.k_doc)


# --------------------------------------------------------------------------
# 结果模型自身的约束
# --------------------------------------------------------------------------


def test_result_is_immutable() -> None:
    """DocumentAggregationResult 不可变。"""
    result = aggregate_document_masses([_mass("d1", 0.6, 0.1, 0.3)])

    with pytest.raises(ValidationError):
        result.k_doc = 1.0  # type: ignore[misc]


def test_result_model_rejects_inconsistent_total_conflict() -> None:
    """手工构造时，完全冲突与 mass / k_doc 必须一致。"""
    with pytest.raises(ValidationError, match="完全冲突时 mass 必须为 None"):
        DocumentAggregationResult(
            sample_id="s1",
            claim_id="c1",
            evaluator="e",
            document_ids=("d1",),
            mass=CombinedMass(m_support=0.5, m_refute=0.2, m_theta=0.3),
            k_doc=1.0,
            steps=(),
            is_total_conflict=True,
        )

    with pytest.raises(ValidationError, match="mass 不能为 None"):
        DocumentAggregationResult(
            sample_id="s1",
            claim_id="c1",
            evaluator="e",
            document_ids=("d1",),
            mass=None,
            k_doc=0.0,
            steps=(),
            is_total_conflict=False,
        )


def test_result_model_rejects_duplicate_document_ids() -> None:
    """结果模型拒绝重复的 document_ids。"""
    with pytest.raises(ValidationError, match="不允许重复"):
        DocumentAggregationResult(
            sample_id="s1",
            claim_id="c1",
            evaluator="e",
            document_ids=("d1", "d1"),
            mass=CombinedMass(m_support=0.5, m_refute=0.2, m_theta=0.3),
            k_doc=0.0,
            steps=(),
            is_total_conflict=False,
        )


def test_step_model_requires_index_from_one() -> None:
    """step_index 必须大于等于 1。"""
    with pytest.raises(ValidationError, match="step_index"):
        DocumentCombinationStep(
            step_index=0,
            accumulated_doc_ids=("d1",),
            incoming_doc_id="d2",
            conflict=0.1,
            normalization_denominator=0.9,
            result_mass=CombinedMass(m_support=0.5, m_refute=0.2, m_theta=0.3),
        )


# --------------------------------------------------------------------------
# 十一、使用 demo 数据的集成测试
# --------------------------------------------------------------------------


def _document_masses_for(
    sample: RAGSample, predictions: list[RelationPrediction], claim_id: str
) -> list[MassFunction]:
    """按 doc_id 把某条 claim 的预测与 ContextChunk 配对，生成折扣后的 BPA。

    必须同时按 sample_id 与 claim_id 过滤：文档聚合要求同一 claim 下
    doc_id 唯一，混入其他 claim 的预测会直接被拒绝。

    这里只在测试中手工串联，不实现正式 pipeline。
    """
    contexts = {chunk.doc_id: chunk for chunk in sample.contexts}
    return [
        document_discounted_mass_from_prediction(prediction, contexts[prediction.doc_id])
        for prediction in predictions
        if prediction.sample_id == sample.sample_id
        and prediction.claim_id == claim_id
    ]


def test_demo_conflicting_sample_has_higher_k_doc_than_supported() -> None:
    """conflicting 样例的 K_doc 明显高于 supported 单文档样例。"""
    samples = {s.sample_id: s for s in load_samples(DEMO_PATH)}
    predictions = load_relation_predictions(MOCK_PATH)

    conflicting_sample = samples["demo-004"]
    conflicting_masses = _document_masses_for(
        conflicting_sample, predictions, "demo-004-c1"
    )
    assert len(conflicting_masses) == 2  # 两条相互矛盾的文档

    conflicting = aggregate_document_masses(conflicting_masses)
    supported = aggregate_document_masses(
        _document_masses_for(samples["demo-001"], predictions, "demo-001-c1")
    )

    assert len(conflicting.steps) == 1
    assert conflicting.k_doc > 0.4
    assert supported.k_doc < 0.1  # 一条支持、一条无信息，几乎无冲突
    assert conflicting.k_doc > supported.k_doc

    assert conflicting.evaluator == "mock_evaluator"
    assert conflicting.document_ids == ("demo-004-d1", "demo-004-d2")


def test_demo_integration_does_not_use_gold_state() -> None:
    """把 gold_state 全部抹掉，聚合结果完全不变。"""
    samples = {s.sample_id: s for s in load_samples(DEMO_PATH)}
    predictions = load_relation_predictions(MOCK_PATH)
    sample = samples["demo-004"]

    with_label = aggregate_document_masses(
        _document_masses_for(sample, predictions, "demo-004-c1")
    )

    stripped = RAGSample(
        sample_id=sample.sample_id,
        question=sample.question,
        answer=sample.answer,
        claims=list(sample.claims),
        contexts=list(sample.contexts),
        gold_state=None,
    )
    without_label = aggregate_document_masses(
        _document_masses_for(stripped, predictions, "demo-004-c1")
    )

    assert without_label.mass == with_label.mass
    assert without_label.k_doc == pytest.approx(with_label.k_doc)
