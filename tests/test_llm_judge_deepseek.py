"""LLM 判四类脚本的 DeepSeek 接口测试（不联网）。

除了请求与解析，重点钉住密钥的处理：只出现在请求头里，不进请求体、不进报错
信息；以及哪些 HTTP 错误应当让整批停下、哪些只是这一条稍后重跑。
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_judge.py"
_spec = importlib.util.spec_from_file_location("run_llm_judge_deepseek", SCRIPT)
judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge)  # type: ignore[union-attr]


def _sample() -> RAGSample:
    return RAGSample(
        sample_id="s1",
        question="问题？",
        answer="The claim text.",
        claims=(Claim(claim_id="s1-c1", text="The claim text."),),
        contexts=(
            ContextChunk(doc_id="s1-d1", text="First document."),
            ContextChunk(doc_id="s1-d2", text="Second document."),
        ),
        gold_state=EvidenceState.SUPPORTED,
    )


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        provider="deepseek",
        model="deepseek-v4-pro",
        effort="default",
        max_tokens=4096,
        no_fallbacks=False,
    )


def _body(content: str = '{"label": "refuted"}', finish_reason: str = "stop") -> dict:
    return {
        "id": "ds-req-1",
        "model": "deepseek-v4-pro",
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50},
    }


class FakePost:
    """按顺序返回预设响应，并记下每次的请求体。"""

    def __init__(self, *bodies: dict) -> None:
        self.bodies = list(bodies)
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return self.bodies.pop(0)


class TestJudging:
    def test_request_uses_json_mode_and_the_shared_prompt(self) -> None:
        post = FakePost(_body())
        sample = _sample()

        record = judge.judge_one(post, sample, sample.claims[0], _args())

        (payload,) = post.payloads
        assert payload["model"] == "deepseek-v4-pro"
        assert payload["response_format"] == {"type": "json_object"}
        system, user = payload["messages"]
        # 同一份任务说明，后面只多一句 JSON 格式要求（DeepSeek 的 JSON 模式需要）。
        assert system["content"].startswith(judge.SYSTEM_PROMPT)
        assert "json" in system["content"].lower()
        assert user == {
            "role": "user",
            "content": judge.build_user_message(sample, sample.claims[0]),
        }
        assert record["predicted_label"] == "refuted"
        assert record["gold_label"] == "supported"
        assert record["served_model"] == "deepseek-v4-pro"
        assert (record["input_tokens"], record["output_tokens"]) == (200, 50)
        assert record["prompt_sha256"] == judge.prompt_digest("deepseek")

    def test_the_two_providers_have_different_prompt_digests(self) -> None:
        assert judge.prompt_digest("deepseek") != judge.prompt_digest("anthropic")

    def test_empty_content_is_asked_again(self) -> None:
        post = FakePost(_body(content=""), _body('{"label": "supported"}'))
        sample = _sample()

        record = judge.judge_one(post, sample, sample.claims[0], _args())

        assert len(post.payloads) == 2
        assert record["predicted_label"] == "supported"

    def test_persistently_bad_output_is_a_retryable_failure(self) -> None:
        attempts = judge.DEEPSEEK_CONTENT_ATTEMPTS
        post = FakePost(*[_body('{"label": "maybe"}') for _ in range(attempts)])
        sample = _sample()

        with pytest.raises(judge.RetryableJudgeError, match="未知标签"):
            judge.judge_one(post, sample, sample.claims[0], _args())
        assert len(post.payloads) == attempts

    def test_content_filter_is_recorded_as_a_refusal(self) -> None:
        post = FakePost(_body(content="", finish_reason="content_filter"))
        sample = _sample()

        record = judge.judge_one(post, sample, sample.claims[0], _args())

        assert (record["stop_reason"], record["predicted_label"]) == (
            "refusal",
            "undetermined",
        )

    def test_truncated_output_is_not_parsed(self) -> None:
        post = FakePost(_body(content='{"label": "supp', finish_reason="length"))
        sample = _sample()

        with pytest.raises(judge.RetryableJudgeError, match="length"):
            judge.judge_one(post, sample, sample.claims[0], _args())


class _FakeHTTPResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        judge.DEEPSEEK_URL, code, "error", {}, io.BytesIO(b'{"error": {"message": "nope"}}')
    )


class TestTransport:
    def test_key_goes_in_the_bearer_header_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}

        def fake_urlopen(request, timeout):
            seen["auth"] = request.get_header("Authorization")
            seen["url"] = request.full_url
            seen["body"] = request.data.decode("utf-8")
            return _FakeHTTPResponse({"ok": True})

        monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)

        assert judge.make_deepseek_post("secret-key")({"model": "m"}) == {"ok": True}
        assert seen["auth"] == "Bearer secret-key"
        assert seen["url"] == "https://api.deepseek.com/chat/completions"
        assert "secret-key" not in seen["body"]

    @pytest.mark.parametrize("code", [400, 401, 402, 404, 422])
    def test_client_errors_stop_the_whole_run(
        self, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        def fake_urlopen(request, timeout):
            raise _http_error(code)

        monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(judge.FatalJudgeError, match=str(code)) as caught:
            judge.make_deepseek_post("secret-key")({"model": "m"})
        assert "secret-key" not in str(caught.value)

    def test_rate_limits_are_retried_with_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        def fake_urlopen(request, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _http_error(429)
            return _FakeHTTPResponse({"ok": True})

        monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(judge.time, "sleep", sleeps.append)

        assert judge.make_deepseek_post("k")({"model": "m"}) == {"ok": True}
        assert calls["n"] == 3
        assert sleeps == [1.0, 2.0]

    def test_exhausted_retries_are_a_retryable_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_urlopen(request, timeout):
            raise _http_error(503)

        monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(judge.time, "sleep", lambda seconds: None)

        with pytest.raises(judge.RetryableJudgeError, match="503"):
            judge.make_deepseek_post("k", max_retries=2)({"model": "m"})


class TestSecrets:
    NAME = "DEEPSEEK_API_KEY"

    def test_environment_variable_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / ".env").write_text(f"{self.NAME}=from-file\n", encoding="utf-8")
        monkeypatch.setattr(judge, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv(self.NAME, "from-env")

        assert judge._secret(self.NAME) == "from-env"

    def test_dotenv_file_is_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / ".env").write_text(
            f'# 本地密钥\nOTHER=1\n{self.NAME} = "from-file"\n', encoding="utf-8"
        )
        monkeypatch.setattr(judge, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv(self.NAME, raising=False)

        assert judge._secret(self.NAME) == "from-file"

    def test_missing_secret_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(judge, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv(self.NAME, raising=False)

        assert judge._secret(self.NAME) is None


class TestArguments:
    BASE = ["--manifest", "m.json", "--out-dir", "o"]

    def test_provider_defaults(self) -> None:
        deepseek = judge._parse_args([*self.BASE, "--provider", "deepseek"])
        anthropic = judge._parse_args(self.BASE)

        assert (deepseek.model, deepseek.effort) == ("deepseek-v4-pro", "default")
        assert (anthropic.model, anthropic.effort) == ("claude-opus-5-5", "medium")

    def test_effort_is_rejected_for_deepseek(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            judge._parse_args([*self.BASE, "--provider", "deepseek", "--effort", "high"])
        assert "--effort" in capsys.readouterr().err
