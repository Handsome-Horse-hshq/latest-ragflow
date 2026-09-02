"""统一评估指标：四分类报告与二分类检测报告。

标签口径的定义见 :mod:`rag_ds.metrics.classification` 的模块文档字符串 ——
baseline 无法输出 ``conflicting`` 这件事必须在指标层显式处理，不能糊弄。
"""

from rag_ds.metrics.classification import (
    GOLD_LABELS,
    UNDETERMINED_LABEL,
    ClassificationReport,
    ClassMetrics,
    classification_report,
    default_label_universe,
)
from rag_ds.metrics.detection import DetectionReport, detection_report

__all__ = [
    "GOLD_LABELS",
    "UNDETERMINED_LABEL",
    "ClassMetrics",
    "ClassificationReport",
    "DetectionReport",
    "classification_report",
    "default_label_universe",
    "detection_report",
]
