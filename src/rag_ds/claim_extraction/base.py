"""claim 抽取器的统一抽象接口。

把答案拆成原子断言，是整条链路的第一步。所有抽取器 —— 查表式的
mock、RAGChecker 适配器、以后可能的大模型抽取器 —— 都实现同一个接口，
输出统一的 :class:`~rag_ds.schemas.Claim`，这样下游的关系评估与 D-S
融合不必关心 claim 是怎么来的。

本模块只定义接口，不含任何抽取规则。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_ds.schemas import Claim, RAGSample

__all__ = ["ClaimExtractor"]


class ClaimExtractor(ABC):
    """把答案拆成原子断言的抽取器接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """抽取器名称，例如 ``"mock_extractor"``、``"ragchecker"``。"""

    @abstractmethod
    def extract(self, sample: RAGSample) -> list[Claim]:
        """从样本的答案中抽取 claim。

        实现**不应**读取 ``sample.claims``（那是已有的标注或上一次抽取的
        结果），也**不应**读取 ``sample.gold_state``。

        Args:
            sample: 待抽取的样本；只应使用 ``question`` 与 ``answer``。

        Returns:
            抽取出的 claim 列表；``claim_id`` 在同一样本内必须唯一。
        """

    def with_claims(self, sample: RAGSample) -> RAGSample:
        """返回一个 claims 被替换为抽取结果的新样本。

        原样本不会被修改。``contexts``、``gold_state`` 等字段原样保留 ——
        ``gold_state`` 只是被搬运，抽取过程从不读取它。

        Args:
            sample: 待处理的样本。

        Returns:
            claims 已替换的新 :class:`RAGSample`。

        Raises:
            ValueError: 抽取结果中 ``claim_id`` 重复（由 RAGSample 校验）。
        """
        return RAGSample(
            sample_id=sample.sample_id,
            question=sample.question,
            answer=sample.answer,
            reference_answer=sample.reference_answer,
            claims=self.extract(sample),
            contexts=list(sample.contexts),
            gold_state=sample.gold_state,
        )
