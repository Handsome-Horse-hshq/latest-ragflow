"""RAGChecker 关系判断结果的适配器。

.. important::
    本模块**不 import ``ragchecker``，也不调用它的任何 API**。这里只把
    **你已经跑出来的** RAGChecker 细粒度关系判断，转换成本项目统一的
    :class:`~rag_ds.schemas.RelationPrediction`。

RAGChecker 通常给出离散标签（entailment / contradiction / neutral），而
D-S 链路需要三元概率。转换表默认为::

    entailment    -> (0.90, 0.05, 0.05)
    contradiction -> (0.05, 0.90, 0.05)
    neutral       -> (0.05, 0.05, 0.90)

.. warning::
    **这三组数字不是最终参数。** 它们只是让链路先跑起来的占位值，必须在
    **验证集**上校准 —— 一个把 entailment 判得很准但 neutral 很松的评估器，
    与一个反过来的评估器，理应得到不同的映射。校准前得到的任何实验数字都
    不能写进论文结论。

如果你的 RAGChecker 版本能给出连续置信度，请直接用
:func:`prediction_from_probabilities` 传入真实概率，不要经过标签映射 ——
离散化会白白丢掉信息。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from rag_ds.relation_evaluation.base import RelationEvaluator
from rag_ds.schemas import (
    PROBABILITY_SUM_TOLERANCE,
    Claim,
    ContextChunk,
    RAGSample,
    RelationPrediction,
    UnitFloat,
)

__all__ = [
    "DEFAULT_LABEL_MAPPING",
    "LabelProbabilityMapping",
    "MissingRAGCheckerJudgementError",
    "RAGCheckerLabel",
    "RAGCheckerRelationAdapter",
    "prediction_from_probabilities",
]


class RAGCheckerLabel(str, Enum):
    """RAGChecker 常见的三种离散关系标签。"""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class MissingRAGCheckerJudgementError(LookupError):
    """载荷里缺少某个 (claim, document) 组合的判断。

    不会退化成 neutral 或跳过：「RAGChecker 没判」与「RAGChecker 判为中立」
    是两回事，混为一谈会让缺失数据伪装成有效证据。
    """

    def __init__(self, evaluator: str, sample_id: str, claim_id: str, doc_id: str) -> None:
        self.evaluator = evaluator
        self.sample_id = sample_id
        self.claim_id = claim_id
        self.doc_id = doc_id
        super().__init__(
            "RAGChecker 载荷中缺少关系判断："
            f"evaluator={evaluator!r}, sample_id={sample_id!r}, "
            f"claim_id={claim_id!r}, doc_id={doc_id!r}"
        )


class LabelProbabilityMapping(BaseModel):
    """离散标签到三元概率的映射表。

    .. warning::
        默认值仅为占位，必须在验证集上校准后才能用于正式实验。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: entailment 对应的 ``(p_support, p_refute, p_unknown)``。
    entailment: tuple[UnitFloat, UnitFloat, UnitFloat] = (0.90, 0.05, 0.05)
    #: contradiction 对应的三元概率。
    contradiction: tuple[UnitFloat, UnitFloat, UnitFloat] = (0.05, 0.90, 0.05)
    #: neutral 对应的三元概率。
    neutral: tuple[UnitFloat, UnitFloat, UnitFloat] = (0.05, 0.05, 0.90)

    @model_validator(mode="after")
    def _check_each_triple_sums_to_one(self) -> LabelProbabilityMapping:
        """三个标签的概率三元组都必须归一。"""
        for label in RAGCheckerLabel:
            triple = getattr(self, label.value)
            total = sum(triple)
            if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
                raise ValueError(
                    f"{label.value} 的三元概率之和必须等于 1，当前为 {total!r}"
                )
        return self

    def probabilities(
        self, label: RAGCheckerLabel
    ) -> tuple[float, float, float]:
        """取出该标签对应的三元概率。"""
        return getattr(self, label.value)


#: 模块级默认映射，等价于 ``LabelProbabilityMapping()``。
DEFAULT_LABEL_MAPPING = LabelProbabilityMapping()


def prediction_from_probabilities(
    sample_id: str,
    claim_id: str,
    doc_id: str,
    evaluator: str,
    probabilities: tuple[float, float, float],
    evaluator_reliability: float = 1.0,
) -> RelationPrediction:
    """由三元概率直接构造 :class:`RelationPrediction`。

    有连续置信度时优先走这条路径，不要先离散化成标签再查表。

    Args:
        sample_id: 样本 ID。
        claim_id: claim ID。
        doc_id: 文档 ID。
        evaluator: 评估器名称。
        probabilities: ``(p_support, p_refute, p_unknown)``，须归一。
        evaluator_reliability: 评估器可靠性。

    Returns:
        :class:`RelationPrediction`。
    """
    p_support, p_refute, p_unknown = probabilities
    return RelationPrediction(
        sample_id=sample_id,
        claim_id=claim_id,
        doc_id=doc_id,
        evaluator=evaluator,
        p_support=p_support,
        p_refute=p_refute,
        p_unknown=p_unknown,
        evaluator_reliability=evaluator_reliability,
    )


#: ``(sample_id, claim_id, doc_id)``。
_JudgementKey = tuple[str, str, str]


class RAGCheckerRelationAdapter(RelationEvaluator):
    """把整理好的 RAGChecker 关系标签接入本项目的评估器接口。

    实现 :class:`~rag_ds.relation_evaluation.base.RelationEvaluator`，因此
    与 :class:`~rag_ds.relation_evaluation.mock.MockRelationEvaluator` 可以
    直接互换 —— **D-S 核心代码一行都不用改**。

    Args:
        judgements: ``{(sample_id, claim_id, doc_id): RAGCheckerLabel}``。
        name: 评估器名称，会写进每条输出的 ``evaluator`` 字段。
        mapping: 标签到三元概率的映射表。
        evaluator_reliability: 该评估器的可靠性，写进每条输出。

    Raises:
        ValueError: ``name`` 为空。
    """

    def __init__(
        self,
        judgements: Mapping[_JudgementKey, RAGCheckerLabel],
        name: str = "ragchecker",
        mapping: LabelProbabilityMapping | None = None,
        evaluator_reliability: float = 1.0,
    ) -> None:
        evaluator_name = name.strip()
        if not evaluator_name:
            raise ValueError("评估器 name 不能为空")
        self._name = evaluator_name
        self._mapping = mapping or DEFAULT_LABEL_MAPPING
        self._evaluator_reliability = evaluator_reliability
        self._table: dict[_JudgementKey, RAGCheckerLabel] = dict(judgements)

    @property
    def name(self) -> str:
        """评估器名称。"""
        return self._name

    @property
    def mapping(self) -> LabelProbabilityMapping:
        """当前使用的标签映射表。"""
        return self._mapping

    def __len__(self) -> int:
        """已装载的判断条数。"""
        return len(self._table)

    def evaluate(
        self, sample: RAGSample, claim: Claim, context: ContextChunk
    ) -> RelationPrediction:
        """查表取出标签，按映射表转成三元概率。

        只使用三个 ID；不读取任何文本，也不读取 ``sample.gold_state``。

        Raises:
            MissingRAGCheckerJudgementError: 该组合没有判断结果。
        """
        key: _JudgementKey = (sample.sample_id, claim.claim_id, context.doc_id)
        label = self._table.get(key)
        if label is None:
            raise MissingRAGCheckerJudgementError(self._name, *key)

        return prediction_from_probabilities(
            sample_id=key[0],
            claim_id=key[1],
            doc_id=key[2],
            evaluator=self._name,
            probabilities=self._mapping.probabilities(label),
            evaluator_reliability=self._evaluator_reliability,
        )

    @classmethod
    def from_payload(
        cls,
        payload: Iterable[Mapping[str, str]],
        name: str = "ragchecker",
        mapping: LabelProbabilityMapping | None = None,
        evaluator_reliability: float = 1.0,
    ) -> RAGCheckerRelationAdapter:
        """从一串记录构造适配器。

        每条记录需含 ``sample_id`` / ``claim_id`` / ``doc_id`` / ``label``
        四个键。这是本项目**自己定义的中间格式** —— 你需要写一小段胶水代码
        把 RAGChecker 的实际输出整理成这个形状。

        Args:
            payload: 记录序列。
            name: 评估器名称。
            mapping: 标签映射表。
            evaluator_reliability: 评估器可靠性。

        Returns:
            :class:`RAGCheckerRelationAdapter`。

        Raises:
            KeyError: 某条记录缺少必需键。
            ValueError: ``label`` 不是三种已知标签之一，或查询键重复。
        """
        table: dict[_JudgementKey, RAGCheckerLabel] = {}
        for record in payload:
            key: _JudgementKey = (
                record["sample_id"],
                record["claim_id"],
                record["doc_id"],
            )
            if key in table:
                raise ValueError(f"RAGChecker 载荷中出现重复的查询键：{key}")
            table[key] = RAGCheckerLabel(record["label"])
        return cls(
            table,
            name=name,
            mapping=mapping,
            evaluator_reliability=evaluator_reliability,
        )
