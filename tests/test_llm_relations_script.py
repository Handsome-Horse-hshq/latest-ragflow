"""大模型关系评估器脚本的测试（不联网）。

钉住三件事：模型一次只看到一篇文档；它口头写出的三个数被正确归一化并排成
NLI 路径的顺序；拼出的输出能被现有的摄取函数原样读入。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rag_ds.ragchecker_ingest import parse_checking_output_probabilities
from rag_ds.schemas import Claim, ContextChunk, RAGSample

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # 脚本之间用同目录导入
_spec = importlib.util.spec_from_file_location("run_llm_relations", SCRIPTS / "run_llm_relations.py")
relations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relations)  # type: ignore[union-attr]


def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"model": "deepseek-flash", "max_tokens": 8000, **overrides})


def _body(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "req-1",
        "model": "deepseek-flash",
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 40},
    }


class FakePost:
    def __init__(self, *bodies: dict) -> None:
        self.bodies = list(bodies)
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return self.bodies.pop(0)


ITEM = {
    "query_id": "s1",
    "query": "问题？",
    "gt_answer": "(not used)",
    "response": "The claim text.",
    "retrieved_context": [
        {"doc_id": "s1-d1", "text": "First document."},
        {"doc_id": "s1-d2", "text": "Second document."},
    ],
}


class TestParsing:
    def test_probabilities_are_normalised_in_nli_order(self) -> None:
        text = '{"contradiction": 0.2, "entailment": 0.6, "neutral": 0.2}'

        assert relations.parse_probabilities(text, "x") == pytest.approx([0.6, 0.2, 0.2])

    def test_loosely_normalised_numbers_are_rescaled(self) -> None:
        text = '{"entailment": 0.33, "neutral": 0.33, "contradiction": 0.33}'

        values = relations.parse_probabilities(text, "x")

        assert sum(values) == pytest.approx(1.0)
        assert values == pytest.approx([1 / 3] * 3)

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            '{"entailment": 0.5, "neutral": 0.5}',
            '{"entailment": "high", "neutral": 0, "contradiction": 0}',
            '{"entailment": -0.1, "neutral": 0.6, "contradiction": 0.5}',
            '{"entailment": 1.2, "neutral": 0, "contradiction": 0}',
            '{"entailment": 0, "neutral": 0, "contradiction": 0}',
        ],
    )
    def test_malformed_output_is_a_retryable_failure(self, text: str) -> None:
        with pytest.raises(relations.RetryableJudgeError):
            relations.parse_probabilities(text, "x")


class TestJudgePair:
    def test_the_model_sees_one_document_only(self) -> None:
        post = FakePost(_body('{"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}'))

        record = relations.judge_pair(
            post, "s1", "s1-d1", ITEM["response"], "First document.", _args()
        )

        (payload,) = post.payloads
        system, user = payload["messages"]
        assert system["content"] == relations.SYSTEM_PROMPT
        assert user["content"] == "Claim: The claim text.\n\nDocument: First document."
        assert "Second document." not in json.dumps(payload)
        assert payload["response_format"] == {"type": "json_object"}
        assert record == {
            "query_id": "s1",
            "doc_id": "s1-d1",
            "probabilities": pytest.approx([0.7, 0.2, 0.1]),
            "requested_model": "deepseek-flash",
            "served_model": "deepseek-flash",
            "prompt_sha256": relations.prompt_digest(),
            "input_tokens": 150,
            "output_tokens": 40,
        }

    def test_bad_output_is_asked_again(self) -> None:
        post = FakePost(
            _body(""), _body('{"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}')
        )

        record = relations.judge_pair(post, "s1", "s1-d1", "claim", "doc", _args())

        assert len(post.payloads) == 2
        assert record["probabilities"] == pytest.approx([0.1, 0.8, 0.1])

    def test_truncated_output_is_not_parsed(self) -> None:
        post = FakePost(_body('{"entailment": 0.', finish_reason="length"))

        with pytest.raises(relations.RetryableJudgeError, match="length"):
            relations.judge_pair(post, "s1", "s1-d1", "claim", "doc", _args())


def _done() -> dict:
    def record(doc_id: str, triple: list[float]) -> dict:
        return {
            "query_id": "s1",
            "doc_id": doc_id,
            "probabilities": triple,
            "requested_model": "deepseek-flash",
            "served_model": "deepseek-flash",
            "prompt_sha256": relations.prompt_digest(),
            "input_tokens": 100,
            "output_tokens": 30,
        }

    return {
        ("s1", "s1-d1"): record("s1-d1", [0.7, 0.2, 0.1]),
        ("s1", "s1-d2"): record("s1-d2", [0.05, 0.15, 0.8]),
    }


class TestAssemble:
    def test_output_has_the_shape_of_the_nli_runner(self) -> None:
        payload = relations.assemble([ITEM], _done(), "deepseek-flash")

        (result,) = payload["results"]
        assert result["query_id"] == "s1"
        assert result["retrieved2response"] == [["Entailment", "Contradiction"]]
        assert result["retrieved2response_probabilities"] == [[[0.7, 0.2, 0.1], [0.05, 0.15, 0.8]]]
        info = payload["rag_ds_run_info"]
        assert info["producer"] == "llm-relation-checker"
        assert info["probability_source"] == "verbalized"
        assert info["probability_label_order"] == ["Entailment", "Neutral", "Contradiction"]
        assert (info["pairs"], info["input_tokens"], info["output_tokens"]) == (2, 200, 60)

    def test_existing_ingest_reads_it_unchanged(self) -> None:
        """摄取函数把 (E, N, C) 换成本项目的 (支持, 反驳, 不确定)。"""
        sample = RAGSample(
            sample_id="s1",
            question="问题？",
            answer="The claim text.",
            claims=(Claim(claim_id="s1-c1", text="The claim text."),),
            contexts=(
                ContextChunk(doc_id="s1-d1", text="First document."),
                ContextChunk(doc_id="s1-d2", text="Second document."),
            ),
        )
        payload = relations.assemble([ITEM], _done(), "deepseek-flash")

        first, second = parse_checking_output_probabilities(
            payload, [sample], evaluator="deepseek_flash"
        )

        assert (first.p_support, first.p_refute, first.p_unknown) == pytest.approx((0.7, 0.1, 0.2))
        assert (second.p_support, second.p_refute, second.p_unknown) == pytest.approx(
            (0.05, 0.8, 0.15)
        )
        assert first.evaluator == "deepseek_flash"


class TestResume:
    def test_results_are_indexed_by_pair(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json.pairs.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in _done().values()), encoding="utf-8"
        )

        assert set(relations.load_done(path, "deepseek-flash")) == {
            ("s1", "s1-d1"),
            ("s1", "s1-d2"),
        }

    def test_another_model_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json.pairs.jsonl"
        path.write_text(json.dumps(next(iter(_done().values()))) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="requested_model"):
            relations.load_done(path, "deepseek-v4-pro")

    def test_progress_file_sits_next_to_the_output(self) -> None:
        assert relations.pairs_path(Path("a/out.json")) == Path("a/out.json.pairs.jsonl")
