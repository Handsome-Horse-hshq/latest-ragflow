"""第四阶段假关系评估器的测试。

所有临时文件都写在 pytest 的 ``tmp_path`` 下。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ds.data_io import load_relation_predictions, load_samples, write_relation_predictions
from rag_ds.relation_evaluation import (
    MissingMockPredictionError,
    MockRelationEvaluator,
    RelationEvaluator,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_PATH = _DATA_DIR / "demo.jsonl"
MOCK_PATH = _DATA_DIR / "mock_relations.jsonl"

EVALUATOR_NAME = "mock_evaluator"


@pytest.fixture
def mock_predictions() -> list[RelationPrediction]:
    """从 mock_relations.jsonl 装载预设结果。"""
    return load_relation_predictions(MOCK_PATH)


@pytest.fixture
def evaluator(mock_predictions: list[RelationPrediction]) -> MockRelationEvaluator:
    """基于预设文件构造的假评估器。"""
    return MockRelationEvaluator(EVALUATOR_NAME, mock_predictions)


@pytest.fixture
def demo_samples() -> dict[str, RAGSample]:
    """按 sample_id 索引的 demo 样本。"""
    return {sample.sample_id: sample for sample in load_samples(DEMO_PATH)}


def _first_pair(sample: RAGSample) -> tuple[Claim, ContextChunk]:
    """取样本的第一个 claim 与第一段文档。"""
    return sample.claims[0], sample.contexts[0]


def _grid_sample(claim_count: int = 2, context_count: int = 3) -> RAGSample:
    """构造一个 claim × context 的网格样本，用于测试遍历顺序。"""
    return RAGSample(
        sample_id="grid",
        question="问题？",
        answer="答案。",
        claims=[
            Claim(claim_id=f"c{i}", text=f"断言 {i}。")
            for i in range(1, claim_count + 1)
        ],
        contexts=[
            ContextChunk(doc_id=f"d{j}", text=f"文档 {j}。")
            for j in range(1, context_count + 1)
        ],
    )


def _grid_predictions(
    sample: RAGSample, evaluator_name: str = EVALUATOR_NAME
) -> list[RelationPrediction]:
    """为网格样本的每个组合生成一条预设结果。"""
    return [
        RelationPrediction(
            sample_id=sample.sample_id,
            claim_id=claim.claim_id,
            doc_id=context.doc_id,
            evaluator=evaluator_name,
            p_support=0.7,
            p_refute=0.2,
            p_unknown=0.1,
        )
        for claim in sample.claims
        for context in sample.contexts
    ]


# --------------------------------------------------------------------------
# 1. 接口
# --------------------------------------------------------------------------


def test_mock_evaluator_implements_the_interface(
    evaluator: MockRelationEvaluator,
) -> None:
    """MockRelationEvaluator 是 RelationEvaluator 的子类。"""
    assert issubclass(MockRelationEvaluator, RelationEvaluator)
    assert isinstance(evaluator, RelationEvaluator)
    assert evaluator.name == EVALUATOR_NAME


def test_abstract_base_cannot_be_instantiated() -> None:
    """抽象基类本身不可实例化。"""
    with pytest.raises(TypeError):
        RelationEvaluator()  # type: ignore[abstract]


def test_empty_name_is_rejected(mock_predictions: list[RelationPrediction]) -> None:
    """评估器名称不能为空。"""
    with pytest.raises(ValueError, match="name 不能为空"):
        MockRelationEvaluator("   ", mock_predictions)


# --------------------------------------------------------------------------
# 2-5. 四种预设结果
# --------------------------------------------------------------------------


def test_supported_sample_returns_support_probabilities(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """supported 样例返回预设的高支持概率。"""
    sample = demo_samples["demo-001"]
    claim, context = _first_pair(sample)

    prediction = evaluator.evaluate(sample, claim, context)

    assert prediction.p_support == pytest.approx(0.90)
    assert prediction.p_refute == pytest.approx(0.05)
    assert prediction.p_unknown == pytest.approx(0.05)
    assert prediction.evaluator == EVALUATOR_NAME
    assert prediction.evaluator_reliability == pytest.approx(1.0)


def test_refuted_sample_returns_refute_probabilities(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """refuted 样例返回预设的高反驳概率。"""
    sample = demo_samples["demo-002"]
    claim, context = _first_pair(sample)

    prediction = evaluator.evaluate(sample, claim, context)

    assert prediction.p_refute == pytest.approx(0.90)
    assert prediction.p_support == pytest.approx(0.05)


def test_insufficient_sample_returns_unknown_probabilities(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """insufficient 样例返回预设的高未知概率。"""
    sample = demo_samples["demo-003"]
    claim, context = _first_pair(sample)

    prediction = evaluator.evaluate(sample, claim, context)

    assert prediction.p_unknown == pytest.approx(0.90)
    assert prediction.p_support == pytest.approx(0.05)
    assert prediction.p_refute == pytest.approx(0.05)


def test_conflicting_sample_returns_one_support_and_one_refute(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """conflicting 样例的两段文档分别给出支持与反驳。"""
    sample = demo_samples["demo-004"]

    predictions = evaluator.evaluate_sample(sample)  # 1 claim x 2 docs

    assert len(predictions) == 2
    supporting, refuting = predictions
    assert supporting.doc_id == "demo-004-d1"
    assert supporting.p_support == pytest.approx(0.90)
    assert refuting.doc_id == "demo-004-d2"
    assert refuting.p_refute == pytest.approx(0.90)


def test_every_preset_probability_triple_sums_to_one(
    mock_predictions: list[RelationPrediction],
) -> None:
    """预设结果的三元概率都归一（由模型保证，此处再确认一次）。"""
    for prediction in mock_predictions:
        total = prediction.p_support + prediction.p_refute + prediction.p_unknown
        assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 6-9. evaluate_sample 的数量与顺序
# --------------------------------------------------------------------------


def test_evaluate_sample_returns_full_grid() -> None:
    """evaluate_sample 返回 len(claims) * len(contexts) 条结果。"""
    sample = _grid_sample(claim_count=2, context_count=3)
    mock = MockRelationEvaluator(EVALUATOR_NAME, _grid_predictions(sample))

    assert len(mock.evaluate_sample(sample)) == 6


def test_evaluate_sample_order_is_claims_then_contexts() -> None:
    """遍历顺序固定为外层 claims、内层 contexts。"""
    sample = _grid_sample(claim_count=2, context_count=3)
    mock = MockRelationEvaluator(EVALUATOR_NAME, _grid_predictions(sample))

    pairs = [(p.claim_id, p.doc_id) for p in mock.evaluate_sample(sample)]

    assert pairs == [
        ("c1", "d1"),
        ("c1", "d2"),
        ("c1", "d3"),
        ("c2", "d1"),
        ("c2", "d2"),
        ("c2", "d3"),
    ]


def test_sample_without_claims_returns_empty_list(
    evaluator: MockRelationEvaluator,
) -> None:
    """样本没有 claim 时返回空列表。"""
    sample = RAGSample(
        sample_id="demo-001",
        question="问题？",
        answer="答案。",
        contexts=[ContextChunk(doc_id="demo-001-d1", text="文档。")],
    )

    assert evaluator.evaluate_sample(sample) == []


def test_sample_without_contexts_returns_empty_list(
    evaluator: MockRelationEvaluator,
) -> None:
    """样本没有 context 时返回空列表。"""
    sample = RAGSample(
        sample_id="demo-001",
        question="问题？",
        answer="答案。",
        claims=[Claim(claim_id="demo-001-c1", text="断言。")],
    )

    assert evaluator.evaluate_sample(sample) == []


# --------------------------------------------------------------------------
# 10-12. 缺失与重复
# --------------------------------------------------------------------------


def test_missing_preset_raises_dedicated_error(
    mock_predictions: list[RelationPrediction], demo_samples: dict[str, RAGSample]
) -> None:
    """没有预设结果的组合抛出 MissingMockPredictionError。

    mock_relations.jsonl 现已覆盖 demo.jsonl 的完整 claim × 文档网格，
    因此这里刻意只装载第一条预设，让其余组合成为缺失用例。
    """
    partial = MockRelationEvaluator(EVALUATOR_NAME, mock_predictions[:1])
    sample = demo_samples["demo-001"]

    with pytest.raises(MissingMockPredictionError):
        partial.evaluate(sample, sample.claims[1], sample.contexts[1])

    with pytest.raises(MissingMockPredictionError):
        partial.evaluate_sample(sample)


def test_missing_preset_error_reports_all_four_ids(
    mock_predictions: list[RelationPrediction], demo_samples: dict[str, RAGSample]
) -> None:
    """错误信息与异常属性都包含完整的查询 ID。"""
    partial = MockRelationEvaluator(EVALUATOR_NAME, mock_predictions[:1])
    sample = demo_samples["demo-001"]
    claim = sample.claims[1]
    context = sample.contexts[1]

    with pytest.raises(MissingMockPredictionError) as excinfo:
        partial.evaluate(sample, claim, context)

    error = excinfo.value
    assert error.evaluator == EVALUATOR_NAME
    assert error.sample_id == sample.sample_id
    assert error.claim_id == claim.claim_id
    assert error.doc_id == context.doc_id

    message = str(error)
    for token in (EVALUATOR_NAME, sample.sample_id, claim.claim_id, context.doc_id):
        assert token in message


def test_duplicate_lookup_key_is_rejected(
    mock_predictions: list[RelationPrediction],
) -> None:
    """同一查询键出现两次时立即抛出 ValueError。"""
    duplicated = [*mock_predictions, mock_predictions[0].model_copy(deep=True)]

    with pytest.raises(ValueError, match="重复查询键"):
        MockRelationEvaluator(EVALUATOR_NAME, duplicated)


def test_predictions_from_other_evaluators_are_ignored(
    mock_predictions: list[RelationPrediction], demo_samples: dict[str, RAGSample]
) -> None:
    """只装载 evaluator 字段与 name 匹配的记录。"""
    foreign = mock_predictions[0].model_copy(deep=True, update={"evaluator": "other"})
    mock = MockRelationEvaluator(EVALUATOR_NAME, [*mock_predictions, foreign])

    assert len(mock) == len(mock_predictions)

    empty = MockRelationEvaluator("other_evaluator", mock_predictions)
    assert len(empty) == 0
    sample = demo_samples["demo-002"]
    with pytest.raises(MissingMockPredictionError):
        empty.evaluate(sample, *_first_pair(sample))


# --------------------------------------------------------------------------
# 13-14. 隔离性与数据泄漏防线
# --------------------------------------------------------------------------


def test_returned_prediction_is_a_detached_copy(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """返回不可变的独立副本，调用方不能污染下一次查询。"""
    sample = demo_samples["demo-001"]
    claim, context = _first_pair(sample)

    first = evaluator.evaluate(sample, claim, context)
    with pytest.raises(ValidationError, match="frozen"):
        first.p_support = 0.0  # type: ignore[misc]

    second = evaluator.evaluate(sample, claim, context)

    assert second.p_support == pytest.approx(0.90)
    assert second.evaluator_reliability == pytest.approx(1.0)
    assert first is not second


def test_mutating_source_predictions_does_not_affect_the_table(
    mock_predictions: list[RelationPrediction], demo_samples: dict[str, RAGSample]
) -> None:
    """构造后替换传入列表元素不会影响内部表。"""
    mock = MockRelationEvaluator(EVALUATOR_NAME, mock_predictions)
    mock_predictions[0] = mock_predictions[0].model_copy(
        update={"p_support": 0.0, "p_unknown": 0.95}
    )

    sample = demo_samples["demo-001"]
    assert mock.evaluate(sample, *_first_pair(sample)).p_support == pytest.approx(0.90)


def test_gold_state_does_not_influence_output(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """改变 gold_state 不会改变评估器输出 —— 防止标签泄漏。"""
    sample = demo_samples["demo-001"]

    before = evaluator.evaluate_sample(
        RAGSample(
            sample_id=sample.sample_id,
            question=sample.question,
            answer=sample.answer,
            claims=[sample.claims[0]],
            contexts=[sample.contexts[0]],
            gold_state=EvidenceState.SUPPORTED,
        )
    )
    sample_flipped = RAGSample(
        sample_id=sample.sample_id,
        question=sample.question,
        answer=sample.answer,
        claims=[sample.claims[0]],
        contexts=[sample.contexts[0]],
        gold_state=EvidenceState.REFUTED,
    )
    after = evaluator.evaluate_sample(sample_flipped)

    assert before == after

    # 通过不可变副本移除标签也一样。
    sample_without_gold = sample_flipped.model_copy(update={"gold_state": None})
    assert evaluator.evaluate_sample(sample_without_gold) == before


def test_text_content_does_not_influence_output(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """改写全部文本字段不会改变输出 —— 证明结果只来自 ID 查表。"""
    sample = demo_samples["demo-002"]
    original = evaluator.evaluate_sample(sample)

    rewritten = RAGSample(
        sample_id=sample.sample_id,
        question="完全无关的问题？",
        answer="完全无关的答案。",
        claims=[Claim(claim_id=sample.claims[0].claim_id, text="被替换的断言。")],
        contexts=[
            ContextChunk(doc_id=sample.contexts[0].doc_id, text="被替换的文档正文。")
        ],
    )

    assert evaluator.evaluate_sample(rewritten) == original


def test_repeated_calls_are_deterministic(
    evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """同一输入多次调用结果完全一致。"""
    sample = demo_samples["demo-004"]

    runs = [evaluator.evaluate_sample(sample) for _ in range(5)]

    assert all(run == runs[0] for run in runs)


# --------------------------------------------------------------------------
# 15-16. 预设文件与 JSONL 往返
# --------------------------------------------------------------------------


def test_mock_relations_file_covers_the_full_demo_grid(
    mock_predictions: list[RelationPrediction], demo_samples: dict[str, RAGSample]
) -> None:
    """mock_relations.jsonl 覆盖 demo.jsonl 的完整 claim × 文档网格。"""
    expected = {
        (sample.sample_id, claim.claim_id, chunk.doc_id)
        for sample in demo_samples.values()
        for claim in sample.claims
        for chunk in sample.contexts
    }
    actual = {(p.sample_id, p.claim_id, p.doc_id) for p in mock_predictions}

    assert actual == expected
    assert len(mock_predictions) == 8
    assert all(isinstance(p, RelationPrediction) for p in mock_predictions)
    assert {p.evaluator for p in mock_predictions} == {EVALUATOR_NAME}


def test_preset_ids_match_demo_file(
    mock_predictions: list[RelationPrediction], demo_samples: dict[str, RAGSample]
) -> None:
    """预设结果的每个 ID 都能在 demo.jsonl 中找到对应项。"""
    for prediction in mock_predictions:
        sample = demo_samples[prediction.sample_id]
        assert prediction.claim_id in {c.claim_id for c in sample.claims}
        assert prediction.doc_id in {c.doc_id for c in sample.contexts}


def test_relation_prediction_round_trip(
    tmp_path: Path, mock_predictions: list[RelationPrediction]
) -> None:
    """RelationPrediction 写入再读取后内容保持一致。"""
    target = tmp_path / "round_trip.jsonl"

    assert write_relation_predictions(target, mock_predictions) == 8
    restored = load_relation_predictions(target)

    assert restored == mock_predictions


def test_evaluator_can_be_rebuilt_from_written_file(
    tmp_path: Path, evaluator: MockRelationEvaluator, demo_samples: dict[str, RAGSample]
) -> None:
    """把评估结果写盘再读回，可以重建出行为相同的评估器。"""
    sample = demo_samples["demo-004"]
    predictions = evaluator.evaluate_sample(sample)

    target = tmp_path / "produced.jsonl"
    write_relation_predictions(target, predictions)
    rebuilt = MockRelationEvaluator(EVALUATOR_NAME, load_relation_predictions(target))

    assert rebuilt.evaluate_sample(sample) == predictions
