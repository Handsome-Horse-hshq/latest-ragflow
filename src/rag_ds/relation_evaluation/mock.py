"""可控的假关系评估器。

:class:`MockRelationEvaluator` 完全不分析文本：它只按
``(evaluator, sample_id, claim_id, doc_id)`` 查一张预设结果表。这样下游的
BPA 映射与 D-S 融合模块可以在完全确定、可复现的输入上开发和调试，
不受大模型输出波动的干扰。

数据泄漏防线
------------
本模块**不读取** ``sample.gold_state``，也不读取 ``question``、``answer``、
``claim.text``、``context.text``。``gold_state`` 是实验标签，用它生成预测
等于把答案抄进输入，会让后续所有指标失去意义。
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_ds.relation_evaluation.base import RelationEvaluator
from rag_ds.schemas import Claim, ContextChunk, RAGSample, RelationPrediction

__all__ = ["MissingMockPredictionError", "MockRelationEvaluator"]

#: 预设结果表的查询键：(evaluator, sample_id, claim_id, doc_id)。
_PredictionKey = tuple[str, str, str, str]


class MissingMockPredictionError(LookupError):
    """假评估器找不到某个组合对应的预设结果。

    四个 ID 同时以属性形式保留，方便调用方定位缺失的预设数据。
    """

    def __init__(
        self,
        evaluator: str,
        sample_id: str,
        claim_id: str,
        doc_id: str,
    ) -> None:
        self.evaluator = evaluator
        self.sample_id = sample_id
        self.claim_id = claim_id
        self.doc_id = doc_id
        super().__init__(
            "找不到预设的关系预测："
            f"evaluator={evaluator!r}, sample_id={sample_id!r}, "
            f"claim_id={claim_id!r}, doc_id={doc_id!r}"
        )


class MockRelationEvaluator(RelationEvaluator):
    """按预设结果查表的假关系评估器。

    构造时把 ``predictions`` 中 ``evaluator`` 字段与 ``name`` 相同的记录
    建成索引；其余记录被忽略，因此可以把多个评估器的预设结果放在同一个
    文件里，再按名字分别装载。

    Args:
        name: 评估器名称，会写进每条输出的 ``evaluator`` 字段。
        predictions: 预设的关系预测，可以是任意可迭代对象。

    Raises:
        ValueError: ``name`` 为空，或同一查询键出现多条预设结果。
    """

    def __init__(self, name: str, predictions: Iterable[RelationPrediction]) -> None:
        evaluator_name = name.strip()
        if not evaluator_name:
            raise ValueError("评估器 name 不能为空")

        self._name = evaluator_name
        self._table: dict[_PredictionKey, RelationPrediction] = {}

        for prediction in predictions:
            if prediction.evaluator != evaluator_name:
                continue  # 属于别的评估器，跳过

            key: _PredictionKey = (
                prediction.evaluator,
                prediction.sample_id,
                prediction.claim_id,
                prediction.doc_id,
            )
            if key in self._table:
                raise ValueError(
                    "预设结果存在重复查询键："
                    f"evaluator={key[0]!r}, sample_id={key[1]!r}, "
                    f"claim_id={key[2]!r}, doc_id={key[3]!r}"
                )
            # 存入副本，使调用方之后修改传入对象不会影响这张表。
            self._table[key] = prediction.model_copy(deep=True)

    @property
    def name(self) -> str:
        """评估器名称。"""
        return self._name

    def __len__(self) -> int:
        """已装载的预设结果条数。"""
        return len(self._table)

    def evaluate(
        self,
        sample: RAGSample,
        claim: Claim,
        context: ContextChunk,
    ) -> RelationPrediction:
        """按 ID 查表返回预设结果。

        只使用 ``sample.sample_id``、``claim.claim_id`` 与 ``context.doc_id``
        三个标识符；不读取任何文本字段，也不读取 ``sample.gold_state``。

        Args:
            sample: claim 与文档所属的样本。
            claim: 待判断的断言。
            context: 与之比对的检索文档。

        Returns:
            预设结果的深拷贝 —— 调用方修改返回对象不会影响内部表。

        Raises:
            MissingMockPredictionError: 该组合没有预设结果。
        """
        key: _PredictionKey = (
            self._name,
            sample.sample_id,
            claim.claim_id,
            context.doc_id,
        )
        prediction = self._table.get(key)
        if prediction is None:
            raise MissingMockPredictionError(*key)
        return prediction.model_copy(deep=True)
