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
from rag_ds.experiments.cross_validation import (
    CrossValidationResult,
    FoldThresholds,
    make_stratified_folds,
    run_nested_cross_validation,
)
from rag_ds.experiments.cv_ablation import (
    CV_ABLATION_VARIANTS,
    AblationFoldThresholds,
    NestedAblationResult,
    run_nested_ablation,
)
from rag_ds.experiments.detection_proxies import (
    CONFLICT_PROXIES,
    INSUFFICIENCY_PROXIES,
    ClaimDetectionScores,
    DetectionComparison,
    compare_detection,
    detection_scores,
)
from rag_ds.experiments.reliability_sensitivity import (
    MAX_VOTE_ENTROPY,
    ReliabilitySensitivityPoint,
    apply_document_reliability,
    oracle_reliability_from_provenance,
    run_reliability_sensitivity,
    uniform_reliability_sweep,
)
from rag_ds.experiments.export import (
    plot_confusion_matrix,
    plot_diagnostic_scatter,
    plot_reliability_sensitivity,
    plot_threshold_sensitivity,
    write_ablation_csv,
    write_main_results_csv,
    write_predictions_csv,
    write_reliability_sensitivity_csv,
)

__all__ = [
    "CONFLICT_PROXIES",
    "CV_ABLATION_VARIANTS",
    "DS_METHOD",
    "INSUFFICIENCY_PROXIES",
    "MAX_VOTE_ENTROPY",
    "AblationFoldThresholds",
    "ClaimDetectionScores",
    "CrossValidationResult",
    "DetectionComparison",
    "FoldThresholds",
    "AblationResult",
    "AblationVariant",
    "CLASSIFICATION_ABLATION_VARIANTS",
    "ExperimentReport",
    "MethodPrediction",
    "NestedAblationResult",
    "ReliabilitySensitivityPoint",
    "apply_document_reliability",
    "collect_predictions",
    "compare_detection",
    "detection_scores",
    "make_stratified_folds",
    "oracle_reliability_from_provenance",
    "plot_confusion_matrix",
    "plot_diagnostic_scatter",
    "plot_reliability_sensitivity",
    "plot_threshold_sensitivity",
    "run_ablation",
    "run_comparison",
    "run_nested_ablation",
    "run_nested_cross_validation",
    "run_reliability_sensitivity",
    "uniform_reliability_sweep",
    "write_ablation_csv",
    "write_main_results_csv",
    "write_predictions_csv",
    "write_reliability_sensitivity_csv",
]
