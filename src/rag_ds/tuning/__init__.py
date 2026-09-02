"""阈值搜索。

只允许在验证集上搜索：用测试集选阈值再用测试集报告结果没有意义，
:func:`~rag_ds.tuning.threshold_search.search_thresholds` 会直接拒绝。
"""

from rag_ds.tuning.threshold_search import (
    SplitName,
    ThresholdCandidate,
    ThresholdGrid,
    ThresholdSearchResult,
    predicted_label,
    rediagnose,
    search_thresholds,
)

__all__ = [
    "SplitName",
    "ThresholdCandidate",
    "ThresholdGrid",
    "ThresholdSearchResult",
    "predicted_label",
    "rediagnose",
    "search_thresholds",
]
