"""Pipeline 的逐 claim 结果与运行摘要模型。

这些模型只承载结果，不含任何计算逻辑。``gold_state`` 被原样保留下来是为了
**运行结束后的对比**，pipeline 的计算过程从不读取它。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.diagnostics.models import DiagnosticResult
from rag_ds.ds.document_aggregation import DocumentAggregationResult
from rag_ds.ds.evaluator_aggregation import EvaluatorAggregationResult
from rag_ds.schemas import EvidenceState, NonEmptyStr

__all__ = [
    "ClaimPipelineResult",
    "PipelineRunSummary",
    "PipelineStatus",
]


class PipelineStatus(str, Enum):
    """一条 claim 走完 pipeline 后的状态。"""

    #: 正常完成全部融合与门控。
    NORMAL = "normal"
    #: 该 claim 没有任何检索文档。
    NO_CONTEXTS = "no_contexts"
    #: 至少一个评估器的文档融合发生完全冲突。
    DOCUMENT_TOTAL_CONFLICT = "document_total_conflict"
    #: 多个评估器之间发生完全冲突。
    EVALUATOR_TOTAL_CONFLICT = "evaluator_total_conflict"


class ClaimPipelineResult(BaseModel):
    """单条 claim 的完整 pipeline 结果，含全部中间过程。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    question: NonEmptyStr
    answer: NonEmptyStr
    reference_answer: str | None = None
    #: 人工标注，**仅供运行后对比**，计算过程从不读取。
    gold_state: EvidenceState | None = None

    claim_id: NonEmptyStr
    claim_text: NonEmptyStr
    context_count: int = Field(ge=0)
    #: 参与该 claim 的评估器名称，顺序与融合顺序一致。
    evaluators: tuple[NonEmptyStr, ...]

    status: PipelineStatus
    diagnostic: DiagnosticResult

    #: 每个评估器的文档级聚合结果；无文档时为空。
    document_results: tuple[DocumentAggregationResult, ...]
    #: 评估器级聚合结果；无文档或文档完全冲突时为 ``None``。
    evaluator_result: EvaluatorAggregationResult | None = None


class PipelineRunSummary(BaseModel):
    """一次命令行运行的统计摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    normal_count: int = Field(ge=0)
    no_contexts_count: int = Field(ge=0)
    document_total_conflict_count: int = Field(ge=0)
    evaluator_total_conflict_count: int = Field(ge=0)
    #: 各 DiagnosticRegion 取值的出现次数。
    region_counts: dict[str, int]
    #: 各 primary_state 取值的出现次数；``None`` 记为 ``"none"``。
    primary_state_counts: dict[str, int]
    output_jsonl: str | None = None
    output_csv: str | None = None
