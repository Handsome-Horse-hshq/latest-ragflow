"""基于大模型的关系评估器 —— 只有接口，没有任何供应商实现。

.. important::
    本模块**不导入任何模型 SDK，不发起网络请求，不读取 API Key**。
    它只定义一个「调用者」协议：你自己实现一个函数，负责真正去问模型，
    本模块负责把返回的三元概率包装成统一的
    :class:`~rag_ds.schemas.RelationPrediction`。

    这样拆分是刻意的：供应商 SDK 的签名变化频繁，把调用细节留在项目外面，
    D-S 链路就永远不需要因为换模型而改动。

使用方式::

    def my_caller(question, claim_text, document_text):
        # 你自己的实现：拼 prompt、调模型、解析出三个概率
        return (0.8, 0.1, 0.1)

    evaluator = LLMRelationEvaluator("gpt_judge", my_caller)
    prediction = evaluator.evaluate(sample, claim, context)

调用者返回的三元概率必须归一（容差同
:data:`~rag_ds.schemas.PROBABILITY_SUM_TOLERANCE`），否则会在构造
``RelationPrediction`` 时被拒绝 —— 模型输出不归一是常见故障，这里刻意让它
大声报错而不是悄悄归一化。
"""

from __future__ import annotations

from typing import Protocol

from rag_ds.relation_evaluation.base import RelationEvaluator
from rag_ds.schemas import Claim, ContextChunk, RAGSample, RelationPrediction

__all__ = ["LLMRelationEvaluator", "RelationProbabilityCaller"]


class RelationProbabilityCaller(Protocol):
    """真正去问模型的调用者协议。

    实现方负责：拼 prompt、调用模型、解析输出、处理重试与限流。
    本项目不对这些做任何假设。
    """

    def __call__(
        self, question: str, claim_text: str, document_text: str
    ) -> tuple[float, float, float]:
        """返回 ``(p_support, p_refute, p_unknown)``，三者之和须为 1。"""
        ...


class LLMRelationEvaluator(RelationEvaluator):
    """把外部大模型调用包装成统一的关系评估器。

    实现 :class:`RelationEvaluator`，因此可与 mock、RAGChecker 适配器直接
    互换，D-S 核心代码无需改动。

    Args:
        name: 评估器名称，会写进每条输出的 ``evaluator`` 字段。
        caller: 满足 :class:`RelationProbabilityCaller` 的可调用对象。
        evaluator_reliability: 该评估器的可靠性，写进每条输出。

    Raises:
        ValueError: ``name`` 为空。
    """

    def __init__(
        self,
        name: str,
        caller: RelationProbabilityCaller,
        evaluator_reliability: float = 1.0,
    ) -> None:
        evaluator_name = name.strip()
        if not evaluator_name:
            raise ValueError("评估器 name 不能为空")
        self._name = evaluator_name
        self._caller = caller
        self._evaluator_reliability = evaluator_reliability

    @property
    def name(self) -> str:
        """评估器名称。"""
        return self._name

    def evaluate(
        self, sample: RAGSample, claim: Claim, context: ContextChunk
    ) -> RelationPrediction:
        """调用外部实现，把三元概率包装成 RelationPrediction。

        传给调用者的只有 ``question`` / ``claim.text`` / ``context.text``
        三段文本 —— ``gold_state`` 不在其中，模型看不到标签。

        Raises:
            pydantic.ValidationError: 调用者返回的概率不在 [0, 1] 或不归一。
        """
        p_support, p_refute, p_unknown = self._caller(
            sample.question, claim.text, context.text
        )
        return RelationPrediction(
            sample_id=sample.sample_id,
            claim_id=claim.claim_id,
            doc_id=context.doc_id,
            evaluator=self._name,
            p_support=p_support,
            p_refute=p_refute,
            p_unknown=p_unknown,
            evaluator_reliability=self._evaluator_reliability,
        )
