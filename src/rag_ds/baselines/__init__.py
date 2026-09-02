"""三个不使用证据理论的对照 baseline。

* **Weighted Average** —— 按 文档可靠性 × 评估器可靠性 加权平均三个概率；
* **Majority Vote** —— 每条关系预测一票，不使用任何可靠性；
* **Single Evaluator** —— 只用一个指定评估器，按文档可靠性加权平均。

三者**都不会输出** ``conflicting``：朴素聚合把「两条针锋相对的证据」压成
一个低分或一个平局，无法与「谁都说不清楚」区分开。这正是与 D-S 诊断方法
对比时要展示的核心局限。
"""

from rag_ds.baselines.config import (
    BaselineConfig,
    BaselineOptions,
    load_baseline_config,
)
from rag_ds.baselines.decision import decide_baseline_state
from rag_ds.baselines.majority_vote import cast_vote, predict_majority_vote
from rag_ds.baselines.models import (
    BASELINE_SCORE_SUM_TOLERANCE,
    BaselineDecisionReason,
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
    MissingBaselineEvaluatorError,
)
from rag_ds.baselines.runner import (
    BASELINE_CSV_COLUMNS,
    BaselineRunSummary,
    run_baselines,
    run_baselines_from_config,
    write_baseline_csv,
    write_baseline_jsonl,
)
from rag_ds.baselines.single_evaluator import predict_single_evaluator
from rag_ds.baselines.weighted_average import predict_weighted_average

__all__ = [
    "BASELINE_CSV_COLUMNS",
    "BASELINE_SCORE_SUM_TOLERANCE",
    "BaselineConfig",
    "BaselineDecisionReason",
    "BaselineMethod",
    "BaselineOptions",
    "BaselinePrediction",
    "BaselineRunSummary",
    "BaselineThresholds",
    "MissingBaselineEvaluatorError",
    "cast_vote",
    "decide_baseline_state",
    "load_baseline_config",
    "predict_majority_vote",
    "predict_single_evaluator",
    "predict_weighted_average",
    "run_baselines",
    "run_baselines_from_config",
    "write_baseline_csv",
    "write_baseline_jsonl",
]
