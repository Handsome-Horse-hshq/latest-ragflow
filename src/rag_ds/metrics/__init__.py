"""统一评估指标：四分类报告、二分类检测报告与显著性检验。

标签口径的定义见 :mod:`rag_ds.metrics.classification` 的模块文档字符串 ——
baseline 无法输出 ``conflicting`` 这件事必须在指标层显式处理，不能糊弄。

方法之间的差异必须配上置信区间，见 :mod:`rag_ds.metrics.significance`：
测试集只有 80 条 claim 时，0.08 的 Macro-F1 差距其 95% 区间是跨 0 的。
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
from rag_ds.metrics.significance import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    BootstrapComparison,
    BootstrapInterval,
    MetricName,
    bootstrap_interval,
    paired_bootstrap,
)

__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "GOLD_LABELS",
    "UNDETERMINED_LABEL",
    "BootstrapComparison",
    "BootstrapInterval",
    "ClassMetrics",
    "ClassificationReport",
    "DetectionReport",
    "MetricName",
    "bootstrap_interval",
    "classification_report",
    "default_label_universe",
    "detection_report",
    "paired_bootstrap",
]
