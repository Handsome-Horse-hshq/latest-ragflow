"""RAGAS 指标的适配器 —— 作为对照方法，**不参与 D-S 融合**。

.. important::
    本模块**不 import ``ragas``，也不调用它的任何 API**。这里只把
    **你已经跑出来的** RAGAS 分数读进来、做基本校验、并保持其原有粒度。

粒度必须诚实
------------
RAGAS 的 ``Faithfulness`` 等指标是**答案级**的：一个样本一个分数。本模块
用 :class:`RagasScore` 的 ``granularity`` 字段把这件事显式记下来，
**绝不**把答案级分数复制到该样本的每条 claim 上冒充 claim-level 结果 ——
那会让 RAGAS 在 claim 级对比里凭空获得「所有 claim 判断完全一致」的优势，
是一种不公平比较。

如果某个指标确实能给出 claim 级分数，就在记录里写明 ``claim_id``，
:func:`aggregate_to_claim_level` 才会把它当作 claim 级使用。

作对比时的正确做法
------------------
* 答案级指标 → 只与同为答案级的量比较，或在报告中明确标注粒度差异；
* claim 级指标 → 才可以直接与 D-S 的 claim 级诊断放在同一张表里。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from rag_ds.schemas import NonEmptyStr, UnitFloat

__all__ = [
    "RagasGranularity",
    "RagasMetric",
    "RagasScore",
    "RagasScoreTable",
    "load_ragas_scores",
]


class RagasMetric(str, Enum):
    """本项目使用的 RAGAS 指标。"""

    #: 答案相对检索上下文的忠实度。
    FAITHFULNESS = "faithfulness"
    #: 答案相对参考答案的事实正确性。
    FACTUAL_CORRECTNESS = "factual_correctness"


class RagasGranularity(str, Enum):
    """分数的粒度。"""

    #: 一个样本一个分数。
    ANSWER = "answer"
    #: 每条 claim 一个分数。
    CLAIM = "claim"


class RagasScore(BaseModel):
    """一条 RAGAS 分数记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    metric: RagasMetric
    score: UnitFloat
    granularity: RagasGranularity
    #: 仅 claim 级记录有值；答案级必须为 ``None``。
    claim_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check_granularity_matches_claim_id(self) -> RagasScore:
        """粒度与 ``claim_id`` 必须一致。"""
        has_claim = self.claim_id is not None
        if (self.granularity is RagasGranularity.CLAIM) != has_claim:
            raise ValueError(
                "claim 级记录必须给出 claim_id，答案级记录必须不给；"
                f"当前 granularity={self.granularity.value!r}, "
                f"claim_id={self.claim_id!r}"
            )
        return self


class RagasScoreTable:
    """按 ``(sample_id, metric, claim_id)`` 索引的 RAGAS 分数表。

    Args:
        scores: 分数记录。

    Raises:
        ValueError: 出现重复的查询键。
    """

    def __init__(self, scores: Iterable[RagasScore]) -> None:
        self._scores: list[RagasScore] = []
        self._table: dict[tuple[str, str, str | None], RagasScore] = {}
        for score in scores:
            key = (score.sample_id, score.metric.value, score.claim_id)
            if key in self._table:
                raise ValueError(f"RAGAS 分数出现重复的查询键：{key}")
            self._table[key] = score
            self._scores.append(score)

    def __len__(self) -> int:
        """记录条数。"""
        return len(self._scores)

    @property
    def scores(self) -> list[RagasScore]:
        """全部记录，顺序与输入一致。"""
        return list(self._scores)

    def get(
        self,
        sample_id: str,
        metric: RagasMetric,
        claim_id: str | None = None,
    ) -> RagasScore | None:
        """按查询键取一条记录；不存在返回 ``None``。"""
        return self._table.get((sample_id, metric.value, claim_id))

    def answer_level(self, metric: RagasMetric) -> dict[str, float]:
        """取出该指标的全部**答案级**分数，``{sample_id: score}``。"""
        return {
            score.sample_id: score.score
            for score in self._scores
            if score.metric is metric
            and score.granularity is RagasGranularity.ANSWER
        }

    def claim_level(self, metric: RagasMetric) -> dict[tuple[str, str], float]:
        """取出该指标的全部**claim 级**分数，``{(sample_id, claim_id): score}``。

        答案级记录**不会**被摊到 claim 上 —— 见模块文档字符串。
        """
        return {
            (score.sample_id, score.claim_id): score.score
            for score in self._scores
            if score.metric is metric
            and score.granularity is RagasGranularity.CLAIM
            and score.claim_id is not None
        }

    def granularity_of(self, metric: RagasMetric) -> set[RagasGranularity]:
        """该指标在本表中出现过的粒度集合，用于报告时如实标注。"""
        return {
            score.granularity for score in self._scores if score.metric is metric
        }


def load_ragas_scores(path: str | Path) -> RagasScoreTable:
    """从 JSONL 读取整理好的 RAGAS 分数。

    每行一条记录，字段与 :class:`RagasScore` 一致。这是本项目**自己定义的
    中间格式** —— 你需要写一小段胶水代码把 RAGAS 的实际输出整理成这个形状，
    具体字段名请查你所用版本的文档。

    Args:
        path: JSONL 文件路径。

    Returns:
        :class:`RagasScoreTable`。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 某行不是合法 JSON、不符合模型，或查询键重复。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"RAGAS 分数文件不存在：{file_path}")

    scores: list[RagasScore] = []
    text = file_path.read_text(encoding="utf-8-sig")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{file_path}: 第 {line_number} 行不是合法 JSON —— {exc.msg}"
            ) from exc
        scores.append(RagasScore.model_validate(payload))

    return RagasScoreTable(scores)
