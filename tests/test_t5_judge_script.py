"""flan-t5 评判器脚本的纯逻辑测试（不加载模型，用假分词器）。

早期版本有两个缺陷：对候选词 token 对数概率**求和**导致长度偏置，以及用了
非 FLAN 原生的提示格式，结果 99.8% 判成 unknown。修正版改为 FLAN MNLI 模板 +
选项首 token 打分。这组测试钉住三件修正后必须成立的事：

1. 选项首词必须各为单个 token（首 token 打分的前提）；
2. 超长时只截前提，模板与选项永远完整；
3. 续跑时拒绝与旧评估器 / 旧打分版本混在一起。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_t5_judge_chunked.py"
_spec = importlib.util.spec_from_file_location("run_t5_judge_chunked", SCRIPT)
t5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t5)  # type: ignore[union-attr]


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class FakeTokenizer:
    """按空白切词、每个词一个 token 的假分词器；可指定某些词为多 token。"""

    def __init__(self, multi_token: dict[str, int] | None = None) -> None:
        self.multi_token = multi_token or {}
        self.vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, len(self.vocab) + 10)

    def __call__(self, text: str, add_special_tokens: bool = True) -> _Encoded:
        ids: list[int] = []
        for word in text.split():
            ids.extend([self._id(word)] * self.multi_token.get(word, 1))
        if add_special_tokens:
            ids.append(1)  # </s>
        return _Encoded(ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        inverse = {v: k for k, v in self.vocab.items()}
        return " ".join(inverse[i] for i in ids if i in inverse)


class TestOptionTokens:
    def test_single_token_options_are_accepted(self) -> None:
        ids = t5._option_token_ids(FakeTokenizer())

        assert len(ids) == 3
        assert len(set(ids)) == 3

    def test_multi_token_option_is_rejected(self) -> None:
        """早期版本的「entailment」是 4 个 token —— 正是长度偏置的来源。"""
        with pytest.raises(ValueError, match="恰好是单个 token"):
            t5._option_token_ids(FakeTokenizer(multi_token={"yes": 4}))

    def test_option_mapping_covers_all_three_fields(self) -> None:
        fields = {field for _, field in t5.OPTION_FIRST_WORDS}

        assert fields == {"p_support", "p_refute", "p_unknown"}

    def test_yes_means_support_and_no_means_refute(self) -> None:
        mapping = dict(t5.OPTION_FIRST_WORDS)

        assert mapping["yes"] == "p_support"
        assert mapping["no"] == "p_refute"
        assert mapping["it"] == "p_unknown"


class TestPrompt:
    def test_short_prompt_is_untouched(self) -> None:
        prompt = t5.build_prompt(FakeTokenizer(), "the sky is blue", "sky blue")

        assert "Premise: the sky is blue" in prompt
        assert prompt.endswith("- no")

    def test_long_premise_is_truncated_but_options_survive(self, monkeypatch) -> None:
        monkeypatch.setattr(t5, "MAX_INPUT_TOKENS", 40)
        premise = " ".join(f"w{i}" for i in range(200))

        prompt = t5.build_prompt(FakeTokenizer(), premise, "a short claim")

        assert len(FakeTokenizer()(prompt).input_ids) <= 40
        assert prompt.endswith("OPTIONS:\n- yes\n- it is not possible to tell\n- no")
        assert "Hypothesis: a short claim" in prompt
        assert "w199" not in prompt  # 截的是前提尾部

    def test_hypothesis_alone_too_long_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(t5, "MAX_INPUT_TOKENS", 10)
        hypothesis = " ".join(f"h{i}" for i in range(50))

        with pytest.raises(ValueError, match="无法在保留选项"):
            t5.build_prompt(FakeTokenizer(), "premise", hypothesis)


class TestResumeGuard:
    def _record(self, evaluator: str, version: str | None) -> dict:
        record = {"sample_id": "s1", "evaluator": evaluator}
        if version is not None:
            record["scoring_version"] = version
        return record

    def test_same_version_is_resumable(self) -> None:
        records = [self._record(t5.EVALUATOR_NAME, t5.SCORING_VERSION)]

        t5._check_resumable(records, Path("x.partial.jsonl"))

    def test_old_evaluator_name_is_refused(self) -> None:
        """作废版本的输出评估器名是 flan_t5_judge，绝不能被续跑进新文件。"""
        records = [self._record("flan_t5_judge", None)]

        with pytest.raises(ValueError, match="不能混进同一个文件"):
            t5._check_resumable(records, Path("x.partial.jsonl"))

    def test_other_scoring_version_is_refused(self) -> None:
        records = [self._record(t5.EVALUATOR_NAME, "some-other-version")]

        with pytest.raises(ValueError, match="不能混进同一个文件"):
            t5._check_resumable(records, Path("x.partial.jsonl"))

    def test_evaluator_name_differs_from_the_invalid_one(self) -> None:
        assert t5.EVALUATOR_NAME != "flan_t5_judge"
