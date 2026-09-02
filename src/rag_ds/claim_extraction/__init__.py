"""claim 抽取：统一接口、查表式 mock，以及 RAGChecker 结果适配器。

本子包不调用任何大模型，也不 import ``ragchecker``。
"""

from rag_ds.claim_extraction.base import ClaimExtractor
from rag_ds.claim_extraction.mock import (
    MissingMockClaimsError,
    MockClaimExtractor,
)
from rag_ds.claim_extraction.ragchecker_adapter import (
    DEFAULT_CLAIM_ID_TEMPLATE,
    RAGCheckerClaimAdapter,
    claims_from_texts,
)

__all__ = [
    "DEFAULT_CLAIM_ID_TEMPLATE",
    "ClaimExtractor",
    "MissingMockClaimsError",
    "MockClaimExtractor",
    "RAGCheckerClaimAdapter",
    "claims_from_texts",
]
