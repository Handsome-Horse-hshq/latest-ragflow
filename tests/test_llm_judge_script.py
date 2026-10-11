"""LLM 判四类脚本的纯逻辑测试（不联网，用假客户端）。

钉住四件容易悄悄出错的事：

1. 请求里带的是结构化输出与约定的参数，拒答回退默认开启；
2. 拒答不被当成某一类，而是记为 ``undetermined``；输出不完整时报错而不是瞎猜；
3. 续跑时拒绝与另一份提示词 / 另一个模型的旧结果混在同一个文件里；
4. 汇总时与交叉验证结果按 claim 配对，覆盖范围对不上就拒绝比较。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_judge.py"
_spec = importlib.util.spec_from_file_location("run_llm_judge", SCRIPT)
judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge)  # type: ignore[union-attr]


def _sample(sample_id: str = "s1", gold: EvidenceState = EvidenceState.SUPPORTED) -> RAGSample:
    return RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="The claim text.",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="The claim text."),),
        contexts=(
            ContextChunk(doc_id=f"{sample_id}-d1", text="First document."),
            ContextChunk(doc_id=f"{sample_id}-d2", text="Second document."),
        ),
        gold_state=gold,
    )


def _args(**overrides) -> argparse.Namespace:
    values = {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "effort": "medium",
        "max_tokens": 4096,
        "no_fallbacks": False,
    }
    return argparse.Namespace(**{**values, **overrides})


def _response(text: str = '{"label": "supported"}', stop_reason: str = "end_turn"):
    return SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        _request_id="req_test",
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


class FakeClient:
    """记录收到的请求，按预设返回响应。"""

    def __init__(self, response) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.messages = SimpleNamespace(create=self._recorder("stable", response))
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=self._recorder("beta", response))
        )

    def _recorder(self, endpoint: str, response):
        def create(**kwargs):
            self.calls.append((endpoint, kwargs))
            return response

        return create


class TestRequest:
    def test_user_message_lists_claim_and_numbered_documents(self) -> None:
        sample = _sample()

        text = judge.build_user_message(sample, sample.claims[0])

        assert text.splitlines() == [
            "Claim: The claim text.",
            "",
            "Evidence documents:",
            "[1] First document.",
            "[2] Second document.",
        ]
        # 金标准不能出现在提示词里。
        assert "supported" not in text.lower()

    def test_structured_output_and_fallbacks_are_requested_by_default(self) -> None:
        client = FakeClient(_response())
        sample = _sample()

        judge.judge_one(client, sample, sample.claims[0], _args())

        ((endpoint, request),) = client.calls
        assert endpoint == "beta"
        assert request["betas"] == [judge.FALLBACK_BETA]
        assert request["fallbacks"] == "default"
        assert request["model"] == "claude-opus-5-5"
        assert request["system"] == judge.SYSTEM_PROMPT
        assert request["output_config"]["effort"] == "medium"
        schema = request["output_config"]["format"]["schema"]
        assert schema["properties"]["label"]["enum"] == [
            "supported",
            "refuted",
            "insufficient",
            "conflicting",
        ]
        assert schema["additionalProperties"] is False
        # 这些模型上思考不能关、采样参数会被拒绝：请求里不应出现。
        assert not {"thinking", "temperature", "top_p", "top_k"} & set(request)

    def test_no_fallbacks_uses_the_stable_endpoint(self) -> None:
        client = FakeClient(_response())
        sample = _sample()

        judge.judge_one(client, sample, sample.claims[0], _args(no_fallbacks=True))

        ((endpoint, request),) = client.calls
        assert endpoint == "stable"
        assert "fallbacks" not in request and "betas" not in request


class TestResponseHandling:
    def test_record_carries_label_usage_and_provenance(self) -> None:
        sample = _sample(gold=EvidenceState.REFUTED)

        record = judge.judge_one(
            FakeClient(_response('{"label": "conflicting"}')), sample, sample.claims[0], _args()
        )

        assert record == {
            "sample_id": "s1",
            "claim_id": "s1-c1",
            "gold_label": "refuted",
            "requested_model": "claude-opus-5-5",
            "served_model": "claude-opus-5-5",
            "effort": "medium",
            "prompt_sha256": judge.prompt_digest(),
            "stop_reason": "end_turn",
            "request_id": "req_test",
            "input_tokens": 120,
            "output_tokens": 30,
            "predicted_label": "conflicting",
        }

    def test_refusal_is_undetermined_not_a_class(self) -> None:
        sample = _sample()
        response = _response(stop_reason="refusal")
        response.content = []

        record = judge.judge_one(FakeClient(response), sample, sample.claims[0], _args())

        assert record["predicted_label"] == "undetermined"
        assert record["stop_reason"] == "refusal"

    def test_truncated_output_raises_instead_of_guessing(self) -> None:
        sample = _sample()

        with pytest.raises(RuntimeError, match="max_tokens"):
            judge.judge_one(
                FakeClient(_response(stop_reason="max_tokens")), sample, sample.claims[0], _args()
            )

    def test_unknown_label_raises(self) -> None:
        sample = _sample()

        with pytest.raises(RuntimeError, match="未知标签"):
            judge.judge_one(
                FakeClient(_response('{"label": "maybe"}')), sample, sample.claims[0], _args()
            )


def _record(sample_id: str, gold: str, predicted: str, **overrides) -> dict:
    return {
        "sample_id": sample_id,
        "claim_id": f"{sample_id}-c1",
        "gold_label": gold,
        "requested_model": "claude-opus-5-5",
        "served_model": "claude-opus-5-5",
        "effort": "medium",
        "prompt_sha256": judge.prompt_digest(),
        "stop_reason": "end_turn",
        "request_id": "req",
        "input_tokens": 100,
        "output_tokens": 20,
        "predicted_label": predicted,
        **overrides,
    }


class TestResume:
    def _write(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_existing_results_are_indexed_by_claim(self, tmp_path: Path) -> None:
        path = tmp_path / "judge_predictions.jsonl"
        self._write(path, [_record("s1", "supported", "supported")])

        done = judge.load_done(path, judge.prompt_digest(), "claude-opus-5-5", "medium")

        assert set(done) == {("s1", "s1-c1")}

    def test_missing_file_means_nothing_done(self, tmp_path: Path) -> None:
        assert judge.load_done(tmp_path / "none.jsonl", "x", "m", "medium") == {}

    @pytest.mark.parametrize(
        ("field", "arguments"),
        [
            ("prompt_sha256", ("0" * 64, "claude-opus-5-5", "medium")),
            ("requested_model", (None, "claude-sonnet-5-5", "medium")),
            ("effort", (None, "claude-opus-5-5", "high")),
        ],
    )
    def test_mixing_settings_is_refused(self, tmp_path: Path, field: str, arguments) -> None:
        path = tmp_path / "judge_predictions.jsonl"
        self._write(path, [_record("s1", "supported", "supported")])
        digest, model, effort = arguments

        with pytest.raises(ValueError, match=field):
            judge.load_done(path, digest or judge.prompt_digest(), model, effort)


class TestSummary:
    RECORDS = [
        _record("s1", "supported", "supported"),
        _record("s2", "refuted", "refuted"),
        _record("s3", "insufficient", "supported"),
        _record("s4", "conflicting", "undetermined", stop_reason="refusal"),
    ]

    def test_metrics_usage_and_refusals(self) -> None:
        summary = judge.summarise(self.RECORDS, None, resamples=50, seed=1)

        assert summary["claim_count"] == 4
        assert summary["report"]["accuracy"] == pytest.approx(0.5)
        assert summary["refusals"] == 1
        assert summary["input_tokens"] == 400
        assert summary["output_tokens"] == 80
        assert summary["served_models"] == {"claude-opus-5-5": 4}
        assert "comparisons" not in summary

    def _write_cv(self, path: Path, sample_ids: list[str]) -> None:
        rows = ["method,sample_id,claim_id,predicted_label,gold_label\n"]
        rows += [f"ds,{sid},{sid}-c1,supported,supported\n" for sid in sample_ids]
        path.write_text("".join(rows), encoding="utf-8-sig")

    def test_paired_comparison_with_cross_validation_predictions(self, tmp_path: Path) -> None:
        cv = tmp_path / "cv_predictions.csv"
        self._write_cv(cv, ["s1", "s2", "s3", "s4"])

        summary = judge.summarise(self.RECORDS, cv, resamples=50, seed=1)

        (comparison,) = summary["comparisons"]
        assert (comparison["method_a"], comparison["method_b"]) == ("llm_judge", "ds")
        assert comparison["sample_count"] == 4

    def test_mismatched_coverage_is_refused(self, tmp_path: Path) -> None:
        cv = tmp_path / "cv_predictions.csv"
        self._write_cv(cv, ["s1", "s2", "s3"])

        with pytest.raises(ValueError, match="无法配对"):
            judge.summarise(self.RECORDS, cv, resamples=50, seed=1)


def test_prompt_digest_tracks_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = judge.prompt_digest()

    monkeypatch.setattr(judge, "SYSTEM_PROMPT", judge.SYSTEM_PROMPT + " ")

    assert judge.prompt_digest() != before
