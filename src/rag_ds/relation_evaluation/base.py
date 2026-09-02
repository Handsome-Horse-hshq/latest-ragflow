"""关系评估器的统一抽象接口。

所有关系评估器 —— 后续的 RAGChecker、RAGAS、大模型评估器，以及本阶段的
假评估器 —— 都实现同一个 :class:`RelationEvaluator` 接口，输出统一的
:class:`~rag_ds.schemas.RelationPrediction`。这样上层的 BPA 映射与证据
融合模块不必关心预测是怎么产生的。

本模块只定义接口与遍历顺序，不包含任何判断规则。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_ds.schemas import Claim, ContextChunk, RAGSample, RelationPrediction

__all__ = ["RelationEvaluator"]


class RelationEvaluator(ABC):
    """判断 claim 与检索文档之间关系的评估器接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """评估器名称，例如 ``"mock_evaluator"``、``"ragchecker"``。

        该名称会写进每条 :class:`RelationPrediction` 的 ``evaluator``
        字段，用于区分不同来源的预测。
        """

    @abstractmethod
    def evaluate(
        self,
        sample: RAGSample,
        claim: Claim,
        context: ContextChunk,
    ) -> RelationPrediction:
        """判断单个 (claim, document) 对的关系。

        Args:
            sample: claim 与文档所属的样本。
            claim: 待判断的断言。
            context: 与之比对的检索文档。

        Returns:
            该组合对应的 :class:`RelationPrediction`。
        """

    def evaluate_sample(self, sample: RAGSample) -> list[RelationPrediction]:
        """对样本内所有 (claim, document) 组合逐一调用 :meth:`evaluate`。

        遍历顺序固定为「外层 claims、内层 contexts」，与 ``sample.claims``
        和 ``sample.contexts`` 的原始顺序一致，因此同一样本的输出顺序在
        任何一次运行中都相同。

        Args:
            sample: 待评估的样本。

        Returns:
            长度为 ``len(claims) * len(contexts)`` 的预测列表；样本没有
            claim 或没有 context 时返回空列表。
        """
        return [
            self.evaluate(sample, claim, context)
            for claim in sample.claims
            for context in sample.contexts
        ]
