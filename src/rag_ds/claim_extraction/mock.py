"""查表式的假 claim 抽取器。

与 :class:`~rag_ds.relation_evaluation.mock.MockRelationEvaluator` 同一思路：
不分析文本，只按 ``sample_id`` 查一张预设表。用途是让下游模块在完全确定、
可复现的输入上开发和调试。

**不读取** ``sample.gold_state``，也不根据 ``answer`` 文本推断任何东西。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rag_ds.claim_extraction.base import ClaimExtractor
from rag_ds.schemas import Claim, RAGSample

__all__ = ["MissingMockClaimsError", "MockClaimExtractor"]


class MissingMockClaimsError(LookupError):
    """假抽取器找不到某个样本的预设 claim。"""

    def __init__(self, extractor: str, sample_id: str) -> None:
        self.extractor = extractor
        self.sample_id = sample_id
        super().__init__(
            f"找不到预设的 claim：extractor={extractor!r}, sample_id={sample_id!r}"
        )


class MockClaimExtractor(ClaimExtractor):
    """按 ``sample_id`` 查表返回预设 claim。

    Args:
        name: 抽取器名称。
        claims_by_sample: ``{sample_id: [Claim, ...]}`` 预设表。

    Raises:
        ValueError: ``name`` 为空。
    """

    def __init__(
        self, name: str, claims_by_sample: Mapping[str, Iterable[Claim]]
    ) -> None:
        extractor_name = name.strip()
        if not extractor_name:
            raise ValueError("抽取器 name 不能为空")
        self._name = extractor_name
        self._table: dict[str, list[Claim]] = {
            sample_id: [claim.model_copy(deep=True) for claim in claims]
            for sample_id, claims in claims_by_sample.items()
        }

    @property
    def name(self) -> str:
        """抽取器名称。"""
        return self._name

    def __len__(self) -> int:
        """已装载预设的样本数。"""
        return len(self._table)

    def extract(self, sample: RAGSample) -> list[Claim]:
        """按 ``sample_id`` 返回预设 claim 的深拷贝。

        Raises:
            MissingMockClaimsError: 该样本没有预设 claim。
        """
        claims = self._table.get(sample.sample_id)
        if claims is None:
            raise MissingMockClaimsError(self._name, sample.sample_id)
        return [claim.model_copy(deep=True) for claim in claims]

    @classmethod
    def from_samples(
        cls, samples: Iterable[RAGSample], name: str = "mock_extractor"
    ) -> MockClaimExtractor:
        """用一批样本自带的 claims 建表。

        方便在离线调试时把「标注好的 claims」当成抽取结果重放。
        """
        return cls(name, {s.sample_id: list(s.claims) for s in samples})
