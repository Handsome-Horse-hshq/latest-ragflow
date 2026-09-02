"""Baseline 批量运行与结果输出。

对每条 claim 依次运行三个 baseline，顺序固定为::

    样本顺序 -> claim 顺序 -> weighted_average -> majority_vote -> single_evaluator

输入完整性沿用 :mod:`rag_ds.integrity` 里 D-S pipeline 用的同一份检查，
不另写一套 —— 两条链路对「什么算合法输入」必须理解一致，否则实验对比就
失去了共同前提。
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.baselines.majority_vote import predict_majority_vote
from rag_ds.baselines.models import (
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.baselines.single_evaluator import predict_single_evaluator
from rag_ds.baselines.weighted_average import predict_weighted_average
from rag_ds.baselines.config import BaselineConfig, load_baseline_config
from rag_ds.data_io import (
    _write_records,
    load_relation_predictions,
    load_samples,
)
from rag_ds.integrity import (
    check_samples,
    evaluator_reliability_for,
    evaluators_for_claim,
    index_predictions,
    predictions_for_claim,
)
from rag_ds.schemas import RAGSample, RelationPrediction

__all__ = [
    "BASELINE_CSV_COLUMNS",
    "BaselineRunSummary",
    "run_baselines",
    "run_baselines_from_config",
    "write_baseline_csv",
    "write_baseline_jsonl",
]

#: CSV 摘要的列顺序。
BASELINE_CSV_COLUMNS = (
    "sample_id",
    "claim_id",
    "method",
    "evaluator",
    "score_support",
    "score_refute",
    "score_unknown",
    "predicted_state",
    "reason",
    "input_count",
    "gold_state",
)


def _claim_predictions(
    table, sample: RAGSample, claim_id: str
) -> list[RelationPrediction]:
    """按「评估器名排序 × 文档原始顺序」收齐该 claim 的全部预测。

    顺序确定，且沿途复用 :mod:`rag_ds.integrity` 的缺失与可靠性一致性检查。
    """
    collected: list[RelationPrediction] = []
    for evaluator in evaluators_for_claim(table, sample.sample_id, claim_id):
        predictions = predictions_for_claim(table, sample, claim_id, evaluator)
        # 可靠性一致性检查与 D-S 链路保持同一套标准。
        evaluator_reliability_for(
            predictions, sample.sample_id, claim_id, evaluator
        )
        collected.extend(predictions)
    return collected


def run_baselines(
    samples: Iterable[RAGSample],
    predictions: Iterable[RelationPrediction],
    thresholds: BaselineThresholds,
    single_evaluator: str,
) -> list[BaselinePrediction]:
    """对每条 claim 运行三个 baseline。

    Args:
        samples: 待判定的样本。
        predictions: 关系评估器的输出。
        thresholds: baseline 判定阈值。
        single_evaluator: single-evaluator baseline 使用的评估器名称。

    Returns:
        ``claim 数 × 3`` 条结果，顺序为
        「样本 → claim → weighted_average → majority_vote → single_evaluator」。

    Raises:
        PipelineError: 输入数据不满足完整性要求（缺失预测、引用错误等）。
        MissingBaselineEvaluatorError: 某 claim 下没有指定评估器的预测。
    """
    sample_list = list(samples)
    prediction_list = list(predictions)

    check_samples(sample_list)
    table = index_predictions(sample_list, prediction_list)

    results: list[BaselinePrediction] = []
    for sample in sample_list:
        for claim in sample.claims:
            claim_predictions = _claim_predictions(table, sample, claim.claim_id)
            results.append(
                predict_weighted_average(
                    sample, claim, claim_predictions, thresholds
                )
            )
            results.append(
                predict_majority_vote(sample, claim, claim_predictions, thresholds)
            )
            results.append(
                predict_single_evaluator(
                    sample, claim, claim_predictions, single_evaluator, thresholds
                )
            )
    return results


def write_baseline_jsonl(
    path: str | Path,
    results: Iterable[BaselinePrediction],
    overwrite: bool = False,
) -> int:
    """把 baseline 结果写成 JSONL，每个 claim-method 一行。

    复用 :mod:`rag_ds.data_io` 的原子写入机制：先写同目录临时文件，成功后
    再 ``os.replace``；中文原样保留，不转义。
    """
    return _write_records(path, results, BaselinePrediction, overwrite)


def _csv_row(result: BaselinePrediction) -> dict[str, str]:
    """把一条 baseline 结果压平成 CSV 行。"""
    return {
        "sample_id": result.sample_id,
        "claim_id": result.claim_id,
        "method": result.method.value,
        "evaluator": result.evaluator or "",
        "score_support": repr(result.score_support),
        "score_refute": repr(result.score_refute),
        "score_unknown": repr(result.score_unknown),
        "predicted_state": result.predicted_state.value,
        "reason": result.reason.value,
        "input_count": str(result.input_count),
        "gold_state": "" if result.gold_state is None else result.gold_state.value,
    }


def write_baseline_csv(
    path: str | Path,
    results: Iterable[BaselinePrediction],
    overwrite: bool = False,
) -> int:
    """把 baseline 结果写成扁平 CSV。

    标准库 :mod:`csv`，UTF-8 with BOM，先临时文件再原子替换。
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"目标文件已存在：{target}；如需覆盖请传入 overwrite=True"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    written = 0
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(BASELINE_CSV_COLUMNS))
            writer.writeheader()
            for result in results:
                writer.writerow(_csv_row(result))
                written += 1
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return written


class BaselineRunSummary(BaseModel):
    """一次 baseline 批量运行的统计摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    method_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    single_evaluator: str
    #: ``{方法名: {预测状态: 次数}}``。
    state_counts_by_method: dict[str, dict[str, int]]
    #: ``{方法名: {判定原因: 次数}}``。
    reason_counts_by_method: dict[str, dict[str, int]]
    output_jsonl: str | None = None
    output_csv: str | None = None


def _summarise(
    results: list[BaselinePrediction],
    sample_count: int,
    claim_count: int,
    config: BaselineConfig,
) -> BaselineRunSummary:
    """统计各方法的预测分布。"""
    methods = [method.value for method in BaselineMethod]
    return BaselineRunSummary(
        sample_count=sample_count,
        claim_count=claim_count,
        method_count=len(methods),
        record_count=len(results),
        single_evaluator=config.baseline.single_evaluator,
        state_counts_by_method={
            method: dict(
                Counter(
                    r.predicted_state.value
                    for r in results
                    if r.method.value == method
                )
            )
            for method in methods
        },
        reason_counts_by_method={
            method: dict(
                Counter(r.reason.value for r in results if r.method.value == method)
            )
            for method in methods
        },
        output_jsonl=str(config.paths.output_jsonl),
        output_csv=str(config.paths.output_csv),
    )


def run_baselines_from_config(
    config_path: str | Path,
    overwrite: bool | None = None,
) -> BaselineRunSummary:
    """按 YAML 配置运行三个 baseline 并写出两份结果。

    与 D-S pipeline 一样：所有 claim 计算成功后才写盘，且在写第一个文件之前
    就检查两个目标文件的覆盖许可，避免半成品状态。
    """
    config = load_baseline_config(config_path)
    allow_overwrite = config.output.overwrite if overwrite is None else overwrite

    if not allow_overwrite:
        for output_path in (config.paths.output_jsonl, config.paths.output_csv):
            if output_path.exists():
                raise FileExistsError(
                    f"输出文件已存在：{output_path}；"
                    "如需覆盖请传入 overwrite=True 或改配置 output.overwrite"
                )

    samples = load_samples(config.paths.samples)
    predictions = load_relation_predictions(config.paths.relation_predictions)

    results = run_baselines(
        samples,
        predictions,
        config.baseline.thresholds,
        config.baseline.single_evaluator,
    )

    write_baseline_jsonl(config.paths.output_jsonl, results, overwrite=True)
    write_baseline_csv(config.paths.output_csv, results, overwrite=True)

    claim_count = sum(len(sample.claims) for sample in samples)
    return _summarise(results, len(samples), claim_count, config)
