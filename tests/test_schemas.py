"""第二阶段数据模型的校验测试。

本文件只测试数据契约，不涉及任何评估算法。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_ds.schemas import (
    PROBABILITY_SUM_TOLERANCE,
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

DEMO_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "demo.jsonl"


def _load_demo_records() -> list[dict[str, Any]]:
    """逐行读取 demo.jsonl，返回原始 dict 列表。"""
    text = DEMO_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _valid_sample_payload() -> dict[str, Any]:
    """返回一份合法的 RAGSample 输入，供各用例按需修改。"""
    return {
        "sample_id": "s1",
        "question": "青蒿素由谁提取？",
        "answer": "由屠呦呦团队提取。",
        "reference_answer": "屠呦呦。",
        "claims": [{"claim_id": "c1", "text": "青蒿素由屠呦呦团队提取。"}],
        "contexts": [
            {
                "doc_id": "d1",
                "text": "屠呦呦团队于 1972 年首次提取出青蒿素。",
                "retrieval_score": 0.9,
                "reliability": 0.95,
            }
        ],
        "gold_state": "supported",
    }


def _valid_prediction_payload() -> dict[str, Any]:
    """返回一份合法的 RelationPrediction 输入。"""
    return {
        "sample_id": "s1",
        "claim_id": "c1",
        "doc_id": "d1",
        "evaluator": "dummy",
        "p_support": 0.7,
        "p_refute": 0.1,
        "p_unknown": 0.2,
    }


# --------------------------------------------------------------------------
# 1. 正常构造
# --------------------------------------------------------------------------


def test_valid_rag_sample_can_be_created() -> None:
    """合法输入可以构造 RAGSample，且字段被正确解析。"""
    sample = RAGSample.model_validate(_valid_sample_payload())

    assert sample.sample_id == "s1"
    assert sample.gold_state is EvidenceState.SUPPORTED
    assert len(sample.claims) == 1
    assert isinstance(sample.claims[0], Claim)
    assert isinstance(sample.contexts[0], ContextChunk)
    assert sample.contexts[0].retrieval_score == pytest.approx(0.9)


def test_optional_fields_have_expected_defaults() -> None:
    """claims / contexts 默认为空列表，可选字段默认为 None 或 1.0。"""
    sample = RAGSample(sample_id="s1", question="问题？", answer="答案。")

    assert sample.claims == ()
    assert sample.contexts == ()
    assert sample.reference_answer is None
    assert sample.gold_state is None

    chunk = ContextChunk(doc_id="d1", text="正文")
    assert chunk.reliability == pytest.approx(1.0)
    assert chunk.retrieval_score is None


def test_string_fields_are_stripped() -> None:
    """必填字符串字段会自动去除首尾空白。"""
    claim = Claim(claim_id="  c1  ", text="\t断言内容 \n")

    assert claim.claim_id == "c1"
    assert claim.text == "断言内容"


# --------------------------------------------------------------------------
# 2-3. demo.jsonl 样例数据
# --------------------------------------------------------------------------


def test_demo_file_has_four_lines() -> None:
    """demo.jsonl 恰好包含四条样例。"""
    assert DEMO_PATH.is_file(), f"缺少样例文件：{DEMO_PATH}"
    assert len(_load_demo_records()) == 4


@pytest.mark.parametrize("record", _load_demo_records(), ids=lambda r: r["sample_id"])
def test_demo_records_pass_validation(record: dict[str, Any]) -> None:
    """demo.jsonl 中每一条数据都能通过 RAGSample 校验。"""
    sample = RAGSample.model_validate(record)

    assert sample.claims, "样例应至少包含一条 claim"
    assert sample.contexts, "样例应至少包含一段上下文"
    assert sample.gold_state is not None


def test_demo_covers_all_evidence_states() -> None:
    """四种 EvidenceState 都在样例数据中出现。"""
    states = {RAGSample.model_validate(r).gold_state for r in _load_demo_records()}

    assert states == set(EvidenceState)


def test_conflicting_sample_has_multiple_contexts() -> None:
    """conflicting 样例必须包含相互矛盾的多段文档。"""
    conflicting = [
        RAGSample.model_validate(r)
        for r in _load_demo_records()
        if r["gold_state"] == EvidenceState.CONFLICTING.value
    ]

    assert conflicting, "样例中缺少 conflicting 用例"
    assert all(len(sample.contexts) >= 2 for sample in conflicting)


# --------------------------------------------------------------------------
# 4-5. 样本内 ID 唯一性
# --------------------------------------------------------------------------


def test_duplicate_claim_id_is_rejected() -> None:
    """同一样本内重复的 claim_id 会被拒绝。"""
    payload = _valid_sample_payload()
    payload["claims"] = [
        {"claim_id": "c1", "text": "第一条断言。"},
        {"claim_id": "c1", "text": "第二条断言。"},
    ]

    with pytest.raises(ValidationError, match="claim_id"):
        RAGSample.model_validate(payload)


def test_duplicate_doc_id_is_rejected() -> None:
    """同一样本内重复的 doc_id 会被拒绝。"""
    payload = _valid_sample_payload()
    payload["contexts"] = [
        {"doc_id": "d1", "text": "第一段文档。"},
        {"doc_id": "d1", "text": "第二段文档。"},
    ]

    with pytest.raises(ValidationError, match="doc_id"):
        RAGSample.model_validate(payload)


def test_distinct_ids_are_accepted() -> None:
    """ID 不重复时不应报错。"""
    payload = _valid_sample_payload()
    payload["claims"] = [
        {"claim_id": "c1", "text": "第一条断言。"},
        {"claim_id": "c2", "text": "第二条断言。"},
    ]

    assert len(RAGSample.model_validate(payload).claims) == 2


# --------------------------------------------------------------------------
# 6-7. 数值范围
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, 2.0])
def test_reliability_out_of_range_is_rejected(bad_value: float) -> None:
    """reliability 超出 [0, 1] 会被拒绝。"""
    with pytest.raises(ValidationError, match="reliability"):
        ContextChunk(doc_id="d1", text="正文", reliability=bad_value)


@pytest.mark.parametrize("bad_value", [-0.01, 1.5])
def test_retrieval_score_out_of_range_is_rejected(bad_value: float) -> None:
    """retrieval_score 超出 [0, 1] 会被拒绝。"""
    with pytest.raises(ValidationError, match="retrieval_score"):
        ContextChunk(doc_id="d1", text="正文", retrieval_score=bad_value)


@pytest.mark.parametrize("good_value", [0.0, 0.5, 1.0])
def test_unit_interval_boundaries_are_accepted(good_value: float) -> None:
    """边界值 0.0 与 1.0 属于合法取值。"""
    chunk = ContextChunk(
        doc_id="d1", text="正文", retrieval_score=good_value, reliability=good_value
    )

    assert chunk.reliability == pytest.approx(good_value)


def test_evaluator_reliability_out_of_range_is_rejected() -> None:
    """evaluator_reliability 超出 [0, 1] 会被拒绝。"""
    payload = _valid_prediction_payload()
    payload["evaluator_reliability"] = 1.2

    with pytest.raises(ValidationError, match="evaluator_reliability"):
        RelationPrediction.model_validate(payload)


# --------------------------------------------------------------------------
# 8-9. RelationPrediction 约束
# --------------------------------------------------------------------------


def test_relation_prediction_accepts_normalised_probabilities() -> None:
    """概率和为 1 时可以正常构造。"""
    prediction = RelationPrediction.model_validate(_valid_prediction_payload())

    assert prediction.evaluator_reliability == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("p_support", "p_refute", "p_unknown"),
    [
        (0.5, 0.3, 0.1),  # 和为 0.9
        (0.6, 0.3, 0.3),  # 和为 1.2
        (0.0, 0.0, 0.0),  # 和为 0
    ],
)
def test_relation_prediction_rejects_bad_probability_sum(
    p_support: float, p_refute: float, p_unknown: float
) -> None:
    """概率和不为 1 时会被拒绝。"""
    payload = _valid_prediction_payload()
    payload.update(p_support=p_support, p_refute=p_refute, p_unknown=p_unknown)

    with pytest.raises(ValidationError, match="必须等于 1"):
        RelationPrediction.model_validate(payload)


def test_relation_prediction_tolerates_small_float_error() -> None:
    """允许 1e-6 以内的浮点误差。"""
    payload = _valid_prediction_payload()
    payload.update(p_support=1.0 / 3, p_refute=1.0 / 3, p_unknown=1.0 / 3)

    prediction = RelationPrediction.model_validate(payload)

    total = prediction.p_support + prediction.p_refute + prediction.p_unknown
    assert abs(total - 1.0) <= PROBABILITY_SUM_TOLERANCE


def test_relation_prediction_rejects_error_above_tolerance() -> None:
    """超出容差的偏差仍会被拒绝。"""
    payload = _valid_prediction_payload()
    payload.update(p_support=0.5, p_refute=0.3, p_unknown=0.2 + 1e-4)

    with pytest.raises(ValidationError, match="必须等于 1"):
        RelationPrediction.model_validate(payload)


def test_relation_prediction_rejects_unknown_field() -> None:
    """出现未定义字段时会被拒绝。"""
    payload = _valid_prediction_payload()
    payload["p_conflict"] = 0.0

    with pytest.raises(ValidationError, match="p_conflict"):
        RelationPrediction.model_validate(payload)


# --------------------------------------------------------------------------
# 10. 空字符串
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
@pytest.mark.parametrize("field", ["sample_id", "question", "answer"])
def test_empty_sample_fields_are_rejected(field: str, empty: str) -> None:
    """RAGSample 的必填字符串字段不接受空串或纯空白。"""
    payload = _valid_sample_payload()
    payload[field] = empty

    with pytest.raises(ValidationError, match=field):
        RAGSample.model_validate(payload)


@pytest.mark.parametrize("field", ["claim_id", "text"])
def test_empty_claim_fields_are_rejected(field: str) -> None:
    """Claim 的字段不接受空串。"""
    payload: dict[str, Any] = {"claim_id": "c1", "text": "断言。"}
    payload[field] = ""

    with pytest.raises(ValidationError, match=field):
        Claim.model_validate(payload)


@pytest.mark.parametrize("field", ["doc_id", "text"])
def test_empty_context_fields_are_rejected(field: str) -> None:
    """ContextChunk 的字段不接受纯空白串。"""
    payload: dict[str, Any] = {"doc_id": "d1", "text": "正文。"}
    payload[field] = "  "

    with pytest.raises(ValidationError, match=field):
        ContextChunk.model_validate(payload)


# --------------------------------------------------------------------------
# 其他约束
# --------------------------------------------------------------------------


def test_rag_sample_rejects_unknown_field() -> None:
    """RAGSample 同样禁止未定义字段。"""
    payload = _valid_sample_payload()
    payload["golden_state"] = "supported"  # 常见拼写错误

    with pytest.raises(ValidationError, match="golden_state"):
        RAGSample.model_validate(payload)


def test_invalid_gold_state_is_rejected() -> None:
    """gold_state 只接受四种枚举值。"""
    payload = _valid_sample_payload()
    payload["gold_state"] = "partially_supported"

    with pytest.raises(ValidationError, match="gold_state"):
        RAGSample.model_validate(payload)


def test_evidence_state_values() -> None:
    """枚举取值与数据文件中的字符串一致。"""
    assert {state.value for state in EvidenceState} == {
        "supported",
        "refuted",
        "insufficient",
        "conflicting",
    }


def test_schema_models_are_deeply_immutable() -> None:
    """冻结外层模型与内部 tuple，避免实验输入在运行中被原地改写。"""
    sample = RAGSample.model_validate(_valid_sample_payload())

    with pytest.raises(ValidationError, match="frozen"):
        sample.answer = "被改写的答案"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        sample.claims[0].text = "被改写的 claim"  # type: ignore[misc]
    assert isinstance(sample.claims, tuple)
    assert isinstance(sample.contexts, tuple)
