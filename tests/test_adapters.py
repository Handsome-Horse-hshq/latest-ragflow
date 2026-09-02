"""第 11–12 步适配器的测试：claim 抽取、RAGChecker、RAGAS。

这些适配器**不导入也不调用**第三方库，因此测试全部离线。
"""

from __future__ import annotations

import ast
import json
import pathlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ds.baselines.ragas_adapter import (
    RagasGranularity,
    RagasMetric,
    RagasScore,
    RagasScoreTable,
    load_ragas_scores,
)
from rag_ds.claim_extraction import (
    ClaimExtractor,
    MissingMockClaimsError,
    MockClaimExtractor,
    RAGCheckerClaimAdapter,
    claims_from_texts,
)
from rag_ds.relation_evaluation import (
    LLMRelationEvaluator,
    LabelProbabilityMapping,
    MissingRAGCheckerJudgementError,
    RAGCheckerLabel,
    RAGCheckerRelationAdapter,
    RelationEvaluator,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "rag_ds"


def _sample(gold_state: EvidenceState | None = None) -> RAGSample:
    """构造一个样本。"""
    return RAGSample(
        sample_id="s1",
        question="水的沸点是多少？",
        answer="标准大气压下水的沸点是 100 摄氏度。",
        claims=[Claim(claim_id="c1", text="水的沸点是 100 摄氏度。")],
        contexts=[ContextChunk(doc_id="d1", text="纯水在 100 摄氏度沸腾。")],
        gold_state=gold_state,
    )


# --------------------------------------------------------------------------
# 不导入第三方库
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "claim_extraction/ragchecker_adapter.py",
        "relation_evaluation/ragchecker_adapter.py",
        "relation_evaluation/llm_evaluator.py",
        "baselines/ragas_adapter.py",
    ],
)
def test_adapters_do_not_import_third_party_libraries(module_path: str) -> None:
    """适配器不得 import ragchecker / ragas / 任何模型 SDK。"""
    banned = {
        "ragchecker",
        "ragas",
        "openai",
        "anthropic",
        "langchain",
        "requests",
        "httpx",
        "urllib",
    }
    tree = ast.parse((_SRC / module_path).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not (imported & banned), f"{module_path} 导入了 {imported & banned}"


# --------------------------------------------------------------------------
# claim 抽取
# --------------------------------------------------------------------------


def test_mock_claim_extractor_round_trip() -> None:
    """查表返回预设 claim，且是深拷贝。"""
    sample = _sample()
    extractor = MockClaimExtractor.from_samples([sample])

    assert isinstance(extractor, ClaimExtractor)
    assert len(extractor) == 1

    claims = extractor.extract(sample)
    assert [c.claim_id for c in claims] == ["c1"]
    claims[0] = Claim(claim_id="changed", text="改过了")
    assert extractor.extract(sample)[0].claim_id == "c1"


def test_mock_claim_extractor_missing_sample() -> None:
    """没有预设时报出专门的错误，不返回空列表。"""
    extractor = MockClaimExtractor("mock_extractor", {})

    with pytest.raises(MissingMockClaimsError, match="s1"):
        extractor.extract(_sample())


def test_with_claims_replaces_claims_and_keeps_the_rest() -> None:
    """with_claims 只替换 claims，其余字段原样保留，原样本不变。"""
    sample = _sample(EvidenceState.SUPPORTED)
    extractor = MockClaimExtractor(
        "mock_extractor", {"s1": [Claim(claim_id="x1", text="新断言。")]}
    )

    updated = extractor.with_claims(sample)

    assert [c.claim_id for c in updated.claims] == ["x1"]
    assert updated.contexts == sample.contexts
    assert updated.gold_state is EvidenceState.SUPPORTED
    assert [c.claim_id for c in sample.claims] == ["c1"]  # 原样本未被修改


def test_claims_from_texts_generates_unique_ids() -> None:
    """按模板生成 claim_id，空白文本被跳过。"""
    claims = claims_from_texts("s1", ["第一条。", "  ", "第二条。", ""])

    assert [c.claim_id for c in claims] == ["s1-c1", "s1-c2"]
    assert [c.text for c in claims] == ["第一条。", "第二条。"]


def test_ragchecker_claim_adapter_accepts_strings_and_mappings() -> None:
    """载荷条目可以是字符串，也可以是含 text/claim/content 键的映射。"""
    adapter = RAGCheckerClaimAdapter.from_payload(
        {"s1": ["断言一。", {"text": "断言二。"}, {"claim": "断言三。"}]}
    )

    claims = adapter.extract(_sample())
    assert [c.text for c in claims] == ["断言一。", "断言二。", "断言三。"]
    assert [c.claim_id for c in claims] == ["s1-c1", "s1-c2", "s1-c3"]
    assert adapter.name == "ragchecker"


def test_ragchecker_claim_adapter_rejects_unknown_sample() -> None:
    """载荷里没有的样本会报错，不静默返回空列表。"""
    adapter = RAGCheckerClaimAdapter.from_payload({"other": ["断言。"]})

    with pytest.raises(KeyError, match="s1"):
        adapter.extract(_sample())


def test_ragchecker_claim_adapter_rejects_bad_entry() -> None:
    """条目缺少文本字段或类型不对时报错。"""
    with pytest.raises(ValueError, match="text / claim / content"):
        RAGCheckerClaimAdapter.from_payload({"s1": [{"wrong": "x"}]})

    with pytest.raises(TypeError, match="字符串或映射"):
        RAGCheckerClaimAdapter.from_payload({"s1": [123]})


# --------------------------------------------------------------------------
# RAGChecker 关系适配器
# --------------------------------------------------------------------------


def test_label_mapping_defaults() -> None:
    """默认标签映射与规格一致，且每组都归一。"""
    mapping = LabelProbabilityMapping()

    assert mapping.probabilities(RAGCheckerLabel.ENTAILMENT) == (0.90, 0.05, 0.05)
    assert mapping.probabilities(RAGCheckerLabel.CONTRADICTION) == (0.05, 0.90, 0.05)
    assert mapping.probabilities(RAGCheckerLabel.NEUTRAL) == (0.05, 0.05, 0.90)
    for label in RAGCheckerLabel:
        assert sum(mapping.probabilities(label)) == pytest.approx(1.0)


def test_label_mapping_rejects_unnormalised_triple() -> None:
    """标签概率不归一时被拒绝。"""
    with pytest.raises(ValidationError, match="必须等于 1"):
        LabelProbabilityMapping(entailment=(0.9, 0.9, 0.9))


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (RAGCheckerLabel.ENTAILMENT, (0.90, 0.05, 0.05)),
        (RAGCheckerLabel.CONTRADICTION, (0.05, 0.90, 0.05)),
        (RAGCheckerLabel.NEUTRAL, (0.05, 0.05, 0.90)),
    ],
)
def test_ragchecker_relation_adapter_converts_labels(
    label: RAGCheckerLabel, expected: tuple[float, float, float]
) -> None:
    """离散标签被转换成对应的三元概率。"""
    sample = _sample()
    adapter = RAGCheckerRelationAdapter.from_payload(
        [
            {
                "sample_id": "s1",
                "claim_id": "c1",
                "doc_id": "d1",
                "label": label.value,
            }
        ]
    )

    prediction = adapter.evaluate(sample, sample.claims[0], sample.contexts[0])

    assert isinstance(adapter, RelationEvaluator)
    assert (
        prediction.p_support,
        prediction.p_refute,
        prediction.p_unknown,
    ) == pytest.approx(expected)
    assert prediction.evaluator == "ragchecker"


def test_ragchecker_relation_adapter_is_interchangeable_with_mock() -> None:
    """RAGChecker 适配器可以直接喂给 D-S 链路，核心代码无需改动。"""
    from rag_ds import (
        DiagnosticThresholds,
        aggregate_document_masses,
        document_discounted_mass_from_prediction,
    )

    sample = _sample()
    adapter = RAGCheckerRelationAdapter.from_payload(
        [
            {
                "sample_id": "s1",
                "claim_id": "c1",
                "doc_id": "d1",
                "label": "entailment",
            }
        ]
    )
    predictions = adapter.evaluate_sample(sample)
    contexts = {c.doc_id: c for c in sample.contexts}
    result = aggregate_document_masses(
        [
            document_discounted_mass_from_prediction(p, contexts[p.doc_id])
            for p in predictions
        ]
    )

    assert result.mass is not None
    assert result.mass.m_support == pytest.approx(0.90)
    assert DiagnosticThresholds().theta_threshold == pytest.approx(0.5)


def test_ragchecker_relation_adapter_missing_judgement() -> None:
    """缺少判断时报错，不退化为 neutral。"""
    sample = _sample()
    adapter = RAGCheckerRelationAdapter.from_payload([])

    with pytest.raises(MissingRAGCheckerJudgementError, match="d1"):
        adapter.evaluate(sample, sample.claims[0], sample.contexts[0])


def test_ragchecker_relation_adapter_rejects_duplicates() -> None:
    """载荷中查询键重复时报错。"""
    record = {
        "sample_id": "s1",
        "claim_id": "c1",
        "doc_id": "d1",
        "label": "entailment",
    }

    with pytest.raises(ValueError, match="重复的查询键"):
        RAGCheckerRelationAdapter.from_payload([record, dict(record)])


def test_ragchecker_adapter_does_not_read_gold_state() -> None:
    """改变 gold_state 不影响转换结果。"""
    adapter = RAGCheckerRelationAdapter.from_payload(
        [
            {
                "sample_id": "s1",
                "claim_id": "c1",
                "doc_id": "d1",
                "label": "entailment",
            }
        ]
    )
    supported = _sample(EvidenceState.SUPPORTED)
    refuted = _sample(EvidenceState.REFUTED)

    assert adapter.evaluate(
        supported, supported.claims[0], supported.contexts[0]
    ) == adapter.evaluate(refuted, refuted.claims[0], refuted.contexts[0])


# --------------------------------------------------------------------------
# LLM 评估器接口
# --------------------------------------------------------------------------


def test_llm_evaluator_wraps_an_external_caller() -> None:
    """外部调用者返回的概率被包装成 RelationPrediction。"""
    seen: list[tuple[str, str, str]] = []

    def caller(question: str, claim_text: str, document_text: str):
        seen.append((question, claim_text, document_text))
        return (0.7, 0.2, 0.1)

    sample = _sample(EvidenceState.CONFLICTING)
    evaluator = LLMRelationEvaluator("llm_judge", caller)

    prediction = evaluator.evaluate(sample, sample.claims[0], sample.contexts[0])

    assert prediction.p_support == pytest.approx(0.7)
    assert prediction.evaluator == "llm_judge"
    # 只看到三段文本，看不到 gold_state。
    assert seen == [(sample.question, sample.claims[0].text, sample.contexts[0].text)]


def test_llm_evaluator_rejects_unnormalised_output() -> None:
    """模型输出不归一时大声报错，不悄悄归一化。"""
    sample = _sample()
    evaluator = LLMRelationEvaluator("llm_judge", lambda q, c, d: (0.7, 0.7, 0.7))

    with pytest.raises(ValidationError, match="必须等于 1"):
        evaluator.evaluate(sample, sample.claims[0], sample.contexts[0])


# --------------------------------------------------------------------------
# RAGAS 适配器
# --------------------------------------------------------------------------


def test_ragas_answer_level_score_forbids_claim_id() -> None:
    """答案级记录不能带 claim_id，claim 级必须带。"""
    with pytest.raises(ValidationError, match="claim 级记录必须给出 claim_id"):
        RagasScore(
            sample_id="s1",
            metric=RagasMetric.FAITHFULNESS,
            score=0.8,
            granularity=RagasGranularity.ANSWER,
            claim_id="c1",
        )

    with pytest.raises(ValidationError, match="claim 级记录必须给出 claim_id"):
        RagasScore(
            sample_id="s1",
            metric=RagasMetric.FAITHFULNESS,
            score=0.8,
            granularity=RagasGranularity.CLAIM,
        )


def test_ragas_answer_level_is_not_spread_to_claims() -> None:
    """答案级分数不会被摊到 claim 上冒充 claim-level 结果。"""
    table = RagasScoreTable(
        [
            RagasScore(
                sample_id="s1",
                metric=RagasMetric.FAITHFULNESS,
                score=0.8,
                granularity=RagasGranularity.ANSWER,
            )
        ]
    )

    assert table.answer_level(RagasMetric.FAITHFULNESS) == {"s1": 0.8}
    assert table.claim_level(RagasMetric.FAITHFULNESS) == {}
    assert table.granularity_of(RagasMetric.FAITHFULNESS) == {
        RagasGranularity.ANSWER
    }


def test_ragas_claim_level_scores_are_kept() -> None:
    """claim 级分数按 (sample_id, claim_id) 取出。"""
    table = RagasScoreTable(
        [
            RagasScore(
                sample_id="s1",
                metric=RagasMetric.FACTUAL_CORRECTNESS,
                score=0.6,
                granularity=RagasGranularity.CLAIM,
                claim_id="c1",
            )
        ]
    )

    assert table.claim_level(RagasMetric.FACTUAL_CORRECTNESS) == {("s1", "c1"): 0.6}


def test_ragas_table_rejects_duplicate_keys() -> None:
    """重复查询键被拒绝。"""
    score = RagasScore(
        sample_id="s1",
        metric=RagasMetric.FAITHFULNESS,
        score=0.8,
        granularity=RagasGranularity.ANSWER,
    )

    with pytest.raises(ValueError, match="重复的查询键"):
        RagasScoreTable([score, score])


def test_load_ragas_scores(tmp_path: Path) -> None:
    """从 JSONL 读取分数。"""
    target = tmp_path / "ragas.jsonl"
    target.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                {
                    "sample_id": "s1",
                    "metric": "faithfulness",
                    "score": 0.8,
                    "granularity": "answer",
                },
                {
                    "sample_id": "s1",
                    "metric": "factual_correctness",
                    "score": 0.6,
                    "granularity": "claim",
                    "claim_id": "c1",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    table = load_ragas_scores(target)

    assert len(table) == 2
    assert table.get("s1", RagasMetric.FAITHFULNESS) is not None
    assert table.get("s1", RagasMetric.FACTUAL_CORRECTNESS, "c1") is not None


def test_load_ragas_scores_missing_file(tmp_path: Path) -> None:
    """文件不存在时报错。"""
    with pytest.raises(FileNotFoundError):
        load_ragas_scores(tmp_path / "nope.jsonl")
