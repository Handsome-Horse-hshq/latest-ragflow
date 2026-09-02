"""离线 MVP pipeline：把已有模块串成一条完整链路。

链路::

    RAGSample + RelationPrediction
      -> 文档可靠性折扣    document_discounted_mass_from_prediction
      -> 文档融合与 K_doc  aggregate_document_masses
      -> 评估器可靠性折扣  （由 aggregate_evaluators 内部只施加一次）
      -> 评估器融合与 K_eval
      -> 二维门控          diagnose_evaluator_result
      -> JSONL / CSV 输出

本模块**只做编排**：不含任何数学公式，不复制 D-S 计算，不接入
RAGChecker / RAGAS / 大模型，也不自动抽取 claim 或补全缺失的关系预测。
``gold_state`` 全程不参与计算，只被原样带到结果里供事后对比。
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from rag_ds.config import PipelineConfig, load_pipeline_config
from rag_ds.data_io import (
    _write_records,
    load_relation_predictions,
    load_samples,
)
from rag_ds.diagnostics.gating import (
    diagnose_document_total_conflict,
    diagnose_evaluator_result,
    diagnose_no_evidence,
)
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.ds.discount import document_discounted_mass_from_prediction
from rag_ds.ds.document_aggregation import (
    DocumentAggregationResult,
    aggregate_document_masses,
)
from rag_ds.ds.evaluator_aggregation import (
    EvaluatorEvidence,
    aggregate_evaluators,
)
from rag_ds.integrity import (
    DuplicateRelationPredictionError,
    InconsistentEvaluatorReliabilityError,
    MissingRelationPredictionError,
    NoClaimsError,
    PipelineError,
    PredictionKey,
    ReferentialIntegrityError,
    check_samples,
    evaluator_reliability_for,
    evaluators_for_claim,
    index_predictions,
    predictions_for_claim,
)
from rag_ds.pipeline_results import (
    ClaimPipelineResult,
    PipelineRunSummary,
    PipelineStatus,
)
from rag_ds.schemas import RAGSample, RelationPrediction

__all__ = [
    "CSV_COLUMNS",
    "DuplicateRelationPredictionError",
    "InconsistentEvaluatorReliabilityError",
    "MissingRelationPredictionError",
    "NoClaimsError",
    "PipelineError",
    "ReferentialIntegrityError",
    "run_pipeline",
    "run_pipeline_from_config",
    "write_pipeline_csv",
    "write_pipeline_jsonl",
]

#: CSV 摘要的列顺序。
CSV_COLUMNS = (
    "sample_id",
    "claim_id",
    "claim_text",
    "status",
    "context_count",
    "evaluators",
    "m_support",
    "m_refute",
    "m_theta",
    "k_doc",
    "k_eval",
    "region",
    "verdict",
    "primary_state",
    "evidence_insufficient",
    "document_conflict",
    "evaluator_disagreement",
    "gold_state",
)

#: CSV 中多个评估器名称的连接符。
EVALUATOR_SEPARATOR = "|"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _run_claim(
    sample: RAGSample,
    claim_id: str,
    claim_text: str,
    table: dict[PredictionKey, RelationPrediction],
    thresholds: DiagnosticThresholds,
) -> ClaimPipelineResult:
    """处理单条 claim。"""
    common = {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "answer": sample.answer,
        "reference_answer": sample.reference_answer,
        "gold_state": sample.gold_state,  # 只带走，不参与计算
        "claim_id": claim_id,
        "claim_text": claim_text,
        "context_count": len(sample.contexts),
    }

    if not sample.contexts:
        return ClaimPipelineResult(
            **common,
            evaluators=(),
            status=PipelineStatus.NO_CONTEXTS,
            diagnostic=diagnose_no_evidence(sample.sample_id, claim_id, thresholds),
            document_results=(),
            evaluator_result=None,
        )

    evaluators = evaluators_for_claim(table, sample.sample_id, claim_id)
    if not evaluators:
        raise MissingRelationPredictionError(
            f"claim 有 {len(sample.contexts)} 条检索文档却没有任何关系预测："
            f"sample_id={sample.sample_id!r}, claim_id={claim_id!r}"
        )

    document_results: list[DocumentAggregationResult] = []
    reliabilities: list[float] = []
    for evaluator in evaluators:
        predictions = predictions_for_claim(table, sample, claim_id, evaluator)
        reliabilities.append(
            evaluator_reliability_for(
                predictions, sample.sample_id, claim_id, evaluator
            )
        )
        contexts = {chunk.doc_id: chunk for chunk in sample.contexts}
        document_results.append(
            aggregate_document_masses(
                [
                    document_discounted_mass_from_prediction(
                        prediction, contexts[prediction.doc_id]
                    )
                    for prediction in predictions
                ]
            )
        )

    conflicted = next(
        (result for result in document_results if result.is_total_conflict), None
    )
    if conflicted is not None:
        # 文档级完全冲突：不跳过该评估器，也不让别的评估器把它盖过去。
        return ClaimPipelineResult(
            **common,
            evaluators=evaluators,
            status=PipelineStatus.DOCUMENT_TOTAL_CONFLICT,
            diagnostic=diagnose_document_total_conflict(conflicted, thresholds),
            document_results=tuple(document_results),
            evaluator_result=None,
        )

    evaluator_result = aggregate_evaluators(
        [
            EvaluatorEvidence(
                document_result=document_result,
                evaluator_reliability=reliability,
            )
            for document_result, reliability in zip(document_results, reliabilities)
        ]
    )
    status = (
        PipelineStatus.EVALUATOR_TOTAL_CONFLICT
        if evaluator_result.is_total_conflict
        else PipelineStatus.NORMAL
    )

    return ClaimPipelineResult(
        **common,
        evaluators=evaluators,
        status=status,
        diagnostic=diagnose_evaluator_result(evaluator_result, thresholds),
        document_results=tuple(document_results),
        evaluator_result=evaluator_result,
    )


def run_pipeline(
    samples: Iterable[RAGSample],
    predictions: Iterable[RelationPrediction],
    thresholds: DiagnosticThresholds,
) -> list[ClaimPipelineResult]:
    """对每个样本的每条 claim 独立走完整条链路。

    先做全量完整性检查，再逐 claim 计算 —— 数据有问题时不会先算出一半结果
    再失败。

    Args:
        samples: 待诊断的样本。
        predictions: 关系评估器的输出。
        thresholds: 二维门控阈值。

    Returns:
        每条 claim 一个 :class:`ClaimPipelineResult`，顺序为
        「样本顺序 × claim 顺序」。

    Raises:
        NoClaimsError: 某个样本没有 claim。
        DuplicateRelationPredictionError: 关系预测查询键重复。
        ReferentialIntegrityError: 预测引用了不存在的 sample / claim / doc。
        MissingRelationPredictionError: 某评估器未覆盖全部检索文档。
        InconsistentEvaluatorReliabilityError: 同一评估器的可靠性记录不一致。
    """
    sample_list = list(samples)
    prediction_list = list(predictions)

    check_samples(sample_list)
    table = index_predictions(sample_list, prediction_list)

    return [
        _run_claim(sample, claim.claim_id, claim.text, table, thresholds)
        for sample in sample_list
        for claim in sample.claims
    ]


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def write_pipeline_jsonl(
    path: str | Path,
    results: Iterable[ClaimPipelineResult],
    overwrite: bool = False,
) -> int:
    """把逐 claim 结果写成 JSONL，保留完整嵌套的中间过程。

    复用 :mod:`rag_ds.data_io` 的原子写入机制：先写同目录临时文件，成功后
    再 ``os.replace``；中文原样保留，不转义成 Unicode 码点。

    Args:
        path: 目标文件路径，缺失的父目录会被自动创建。
        results: 待写入的结果。
        overwrite: 目标文件已存在时是否允许覆盖。

    Returns:
        实际写入的行数（每条 claim 一行）。

    Raises:
        FileExistsError: 目标文件已存在且 ``overwrite`` 为 ``False``。
    """
    return _write_records(path, results, ClaimPipelineResult, overwrite)


def _csv_row(result: ClaimPipelineResult) -> dict[str, str]:
    """把一条结果压平成 CSV 行。

    完全冲突状态下三个质量为 ``None``，对应单元格留空 —— 不写 0，也不写
    "None"，避免下游把「未定义」误读成「零质量」。
    """
    diagnostic = result.diagnostic

    def _number(value: float | None) -> str:
        return "" if value is None else repr(value)

    return {
        "sample_id": result.sample_id,
        "claim_id": result.claim_id,
        "claim_text": result.claim_text,
        "status": result.status.value,
        "context_count": str(result.context_count),
        "evaluators": EVALUATOR_SEPARATOR.join(result.evaluators),
        "m_support": _number(diagnostic.m_support),
        "m_refute": _number(diagnostic.m_refute),
        "m_theta": _number(diagnostic.m_theta),
        "k_doc": repr(diagnostic.k_doc),
        "k_eval": repr(diagnostic.k_eval),
        "region": diagnostic.region.value,
        "verdict": diagnostic.verdict.value,
        "primary_state": (
            "" if diagnostic.primary_state is None else diagnostic.primary_state.value
        ),
        "evidence_insufficient": str(diagnostic.evidence_insufficient),
        "document_conflict": str(diagnostic.document_conflict),
        "evaluator_disagreement": str(diagnostic.evaluator_disagreement),
        "gold_state": "" if result.gold_state is None else result.gold_state.value,
    }


def write_pipeline_csv(
    path: str | Path,
    results: Iterable[ClaimPipelineResult],
    overwrite: bool = False,
) -> int:
    """把逐 claim 结果写成扁平 CSV 摘要。

    使用标准库 :mod:`csv`（不使用 pandas），编码为 UTF-8 with BOM，
    方便 Windows Excel 直接打开中文。写入同样先落临时文件再原子替换。

    Args:
        path: 目标文件路径，缺失的父目录会被自动创建。
        results: 待写入的结果。
        overwrite: 目标文件已存在时是否允许覆盖。

    Returns:
        实际写入的数据行数（不含表头）。

    Raises:
        FileExistsError: 目标文件已存在且 ``overwrite`` 为 ``False``。
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
        with os.fdopen(
            handle_fd, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for result in results:
                writer.writerow(_csv_row(result))
                written += 1
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return written


# --------------------------------------------------------------------------
# 配置驱动的入口
# --------------------------------------------------------------------------


def _summarise(
    results: list[ClaimPipelineResult],
    sample_count: int,
    config: PipelineConfig,
) -> PipelineRunSummary:
    """统计一次运行的结果分布。"""
    statuses = Counter(result.status for result in results)
    return PipelineRunSummary(
        sample_count=sample_count,
        claim_count=len(results),
        normal_count=statuses[PipelineStatus.NORMAL],
        no_contexts_count=statuses[PipelineStatus.NO_CONTEXTS],
        document_total_conflict_count=statuses[PipelineStatus.DOCUMENT_TOTAL_CONFLICT],
        evaluator_total_conflict_count=statuses[
            PipelineStatus.EVALUATOR_TOTAL_CONFLICT
        ],
        region_counts=dict(
            Counter(result.diagnostic.region.value for result in results)
        ),
        primary_state_counts=dict(
            Counter(
                "none"
                if result.diagnostic.primary_state is None
                else result.diagnostic.primary_state.value
                for result in results
            )
        ),
        output_jsonl=str(config.paths.output_jsonl),
        output_csv=str(config.paths.output_csv),
    )


def run_pipeline_from_config(
    config_path: str | Path,
    overwrite: bool | None = None,
) -> PipelineRunSummary:
    """按 YAML 配置跑完整条链路并写出两份结果。

    **所有 claim 全部计算成功之后**才开始写盘，且在写第一个文件之前就检查
    两个目标文件的覆盖许可 —— 否则可能出现「JSONL 写好了、CSV 因不允许覆盖
    而失败」的半成品状态。两个写入本身也都是先临时文件再原子替换。

    Args:
        config_path: YAML 配置文件路径。
        overwrite: 覆盖开关；``None`` 表示沿用配置里的 ``output.overwrite``。

    Returns:
        :class:`PipelineRunSummary`。

    Raises:
        PipelineError: 输入数据不满足 pipeline 的前置条件。
        FileExistsError: 输出文件已存在且不允许覆盖。
    """
    config = load_pipeline_config(config_path)
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

    results = run_pipeline(samples, predictions, config.diagnostics)

    write_pipeline_jsonl(config.paths.output_jsonl, results, overwrite=True)
    write_pipeline_csv(config.paths.output_csv, results, overwrite=True)

    return _summarise(results, len(samples), config)
