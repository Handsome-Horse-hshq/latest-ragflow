"""对比实验与消融实验的编排。

四个方法共用同一批输入与同一套完整性检查，公平比较是结构性保证。
"""

from rag_ds.experiments.ablation import (
    CLASSIFICATION_ABLATION_VARIANTS,
    AblationResult,
    AblationVariant,
    run_ablation,
)
from rag_ds.experiments.comparison import (
    DS_METHOD,
    ExperimentReport,
    MethodPrediction,
    collect_predictions,
    run_comparison,
)
from rag_ds.experiments.export import (
    plot_confusion_matrix,
    plot_diagnostic_scatter,
    plot_threshold_sensitivity,
    write_ablation_csv,
    write_main_results_csv,
    write_predictions_csv,
)

__all__ = [
    "DS_METHOD",
    "AblationResult",
    "AblationVariant",
    "CLASSIFICATION_ABLATION_VARIANTS",
    "ExperimentReport",
    "MethodPrediction",
    "collect_predictions",
    "plot_confusion_matrix",
    "plot_diagnostic_scatter",
    "plot_threshold_sensitivity",
    "run_ablation",
    "run_comparison",
    "write_ablation_csv",
    "write_main_results_csv",
    "write_predictions_csv",
]
