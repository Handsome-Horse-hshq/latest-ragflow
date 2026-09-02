"""把 D-S 诊断与三个 baseline 放在同一批数据上对比。

公平比较的前提（总规划 §13）在这里是**结构性保证**而非约定：四个方法
共用同一批 ``samples``、同一批 ``RelationPrediction``、同一套完整性检查，
因此不存在「某个方法悄悄用了不同输入」的可能。

粒度与标签口径见 :mod:`rag_ds.metrics.classification`。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.baselines.models import BaselineMethod, BaselineThresholds
from rag_ds.baselines.runner import run_baselines
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.metrics.classification import ClassificationReport, classification_report
from rag_ds.metrics.detection import DetectionReport, detection_report
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import EvidenceState, NonEmptyStr, RAGSample, RelationPrediction
from rag_ds.tuning.threshold_search import predicted_label

__all__ = [
    "DS_METHOD",
    "ExperimentReport",
    "MethodPrediction",
    "collect_predictions",
    "run_comparison",
]

#: D-S 方法在报告中的名称。
DS_METHOD = "ds"


class MethodPrediction(BaseModel):
    """某个方法对某条 claim 的预测，外加用于检测实验的连续分数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: NonEmptyStr
    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    predicted_label: NonEmptyStr
    gold_label: NonEmptyStr
    #: 证据不足信号：D-S 用 ``m_theta``，baseline 用 ``score_unknown``。
    insufficiency_score: float = Field(ge=0.0, le=1.0)
    #: 冲突信号：D-S 用 ``k_doc``；baseline 没有冲突量，用
    #: ``1 - |score_support - score_refute|`` 作为**代理**（见模块说明）。
    conflict_score: float = Field(ge=0.0, le=1.0)


def collect_predictions(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    ds_thresholds: DiagnosticThresholds,
    baseline_thresholds: BaselineThresholds,
    single_evaluator: str,
) -> list[MethodPrediction]:
    """在同一批输入上跑 D-S 与三个 baseline，汇总成统一记录。

    Args:
        samples: 数据集样本，每条须带 ``gold_state``。
        predictions: 关系评估器输出。
        ds_thresholds: 二维门控阈值。
        baseline_thresholds: baseline 判定阈值。
        single_evaluator: single-evaluator baseline 使用的评估器。

    Returns:
        每个 (方法, claim) 一条 :class:`MethodPrediction`。

    Raises:
        ValueError: 存在缺少 ``gold_state`` 的样本。
    """
    missing = [s.sample_id for s in samples if s.gold_state is None]
    if missing:
        raise ValueError(
            f"以下样本缺少 gold_state，无法参与对比实验：{sorted(missing)[:5]}"
        )

    records: list[MethodPrediction] = []

    for result in run_pipeline(samples, predictions, ds_thresholds):
        diagnostic = result.diagnostic
        records.append(
            MethodPrediction(
                method=DS_METHOD,
                sample_id=result.sample_id,
                claim_id=result.claim_id,
                predicted_label=predicted_label(diagnostic),
                gold_label=result.gold_state.value,  # type: ignore[union-attr]
                # 完全冲突时质量未定义：证据不足信号取 0（它并非「不知道」），
                # 冲突信号取 k_doc 本身（此时为 1）。
                insufficiency_score=(
                    0.0 if diagnostic.m_theta is None else diagnostic.m_theta
                ),
                conflict_score=diagnostic.k_doc,
            )
        )

    for prediction in run_baselines(
        samples, predictions, baseline_thresholds, single_evaluator
    ):
        records.append(
            MethodPrediction(
                method=prediction.method.value,
                sample_id=prediction.sample_id,
                claim_id=prediction.claim_id,
                predicted_label=prediction.predicted_state.value,
                gold_label=prediction.gold_state.value,  # type: ignore[union-attr]
                insufficiency_score=prediction.score_unknown,
                conflict_score=1.0
                - abs(prediction.score_support - prediction.score_refute),
            )
        )

    return records


class ExperimentReport(BaseModel):
    """一次对比实验的全部报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 实验 1：四分类能力。
    classification: tuple[ClassificationReport, ...]
    #: 实验 2：证据不足识别（正类 ``insufficient``）。
    insufficiency_detection: tuple[DetectionReport, ...]
    #: 实验 3：文档冲突识别（正类 ``conflicting``）。
    conflict_detection: tuple[DetectionReport, ...]
    claim_count: int = Field(ge=0)
    methods: tuple[str, ...]

    def classification_for(self, method: str) -> ClassificationReport | None:
        """按方法名取分类报告。"""
        return next((r for r in self.classification if r.method == method), None)


#: 报告中方法的固定顺序。
_METHOD_ORDER = (
    DS_METHOD,
    BaselineMethod.WEIGHTED_AVERAGE.value,
    BaselineMethod.MAJORITY_VOTE.value,
    BaselineMethod.SINGLE_EVALUATOR.value,
)


def run_comparison(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    ds_thresholds: DiagnosticThresholds,
    baseline_thresholds: BaselineThresholds,
    single_evaluator: str,
) -> tuple[ExperimentReport, list[MethodPrediction]]:
    """跑完实验 1–3 并汇总报告。

    Returns:
        ``(报告, 逐条预测记录)``；后者用于绘图与导出明细。
    """
    records = collect_predictions(
        samples, predictions, ds_thresholds, baseline_thresholds, single_evaluator
    )
    by_method: dict[str, list[MethodPrediction]] = {}
    for record in records:
        by_method.setdefault(record.method, []).append(record)

    methods = [m for m in _METHOD_ORDER if m in by_method]
    methods += sorted(set(by_method) - set(methods))

    classification = tuple(
        classification_report(
            [r.gold_label for r in by_method[method]],
            [r.predicted_label for r in by_method[method]],
            method=method,
        )
        for method in methods
    )
    insufficiency = tuple(
        detection_report(
            [r.insufficiency_score for r in by_method[method]],
            [
                r.gold_label == EvidenceState.INSUFFICIENT.value
                for r in by_method[method]
            ],
            score_name=(
                "m_theta" if method == DS_METHOD else f"{method}.score_unknown"
            ),
            positive_label=EvidenceState.INSUFFICIENT.value,
        )
        for method in methods
    )
    conflict = tuple(
        detection_report(
            [r.conflict_score for r in by_method[method]],
            [
                r.gold_label == EvidenceState.CONFLICTING.value
                for r in by_method[method]
            ],
            score_name=(
                "k_doc" if method == DS_METHOD else f"{method}.support_refute_closeness"
            ),
            positive_label=EvidenceState.CONFLICTING.value,
        )
        for method in methods
    )

    claim_count = len(by_method[methods[0]]) if methods else 0
    return (
        ExperimentReport(
            classification=classification,
            insufficiency_detection=insufficiency,
            conflict_detection=conflict,
            claim_count=claim_count,
            methods=tuple(methods),
        ),
        records,
    )
