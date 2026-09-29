"""阈值搜索与标签映射校准。

两者都只允许在验证集上做：用测试集选参数再用测试集报告结果没有意义，
:func:`~rag_ds.tuning.threshold_search.search_thresholds` 与
:func:`~rag_ds.tuning.label_calibration.calibrate_label_mapping` 都会直接拒绝。
"""

from rag_ds.tuning.baseline_threshold_search import (
    DEFAULT_DECISION_THRESHOLDS,
    BaselineThresholdCandidate,
    BaselineThresholdSearchResult,
    search_baseline_thresholds,
)
from rag_ds.tuning.label_calibration import (
    LabelCalibrationError,
    LabelCalibrationResult,
    LabelCalibrationStat,
    calibrate_label_mapping,
)
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
    "DEFAULT_DECISION_THRESHOLDS",
    "BaselineThresholdCandidate",
    "BaselineThresholdSearchResult",
    "LabelCalibrationError",
    "LabelCalibrationResult",
    "LabelCalibrationStat",
    "SplitName",
    "ThresholdCandidate",
    "ThresholdGrid",
    "ThresholdSearchResult",
    "calibrate_label_mapping",
    "predicted_label",
    "search_baseline_thresholds",
    "rediagnose",
    "search_thresholds",
]
