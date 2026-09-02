"""统一数据模型：claim-level RAG 证据状态诊断。

本模块只定义数据契约，不包含任何评估算法。RAGChecker、RAGAS 与
Dempster-Shafer 融合模块在后续阶段共用这里的模型，以避免接口反复变更。

设计约定
--------
* 所有模型均设置 ``extra="forbid"`` 与 ``frozen=True``：拼写错误的字段会
  立即报错，实例创建后也不能被改写。这对论文实验数据的可信度很重要。
* 所有必填字符串字段会自动去除首尾空白，且去空白后不得为空。
* 所有概率与可靠度字段取值范围为 [0, 1]。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Iterable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__ = [
    "PROBABILITY_SUM_TOLERANCE",
    "Claim",
    "ContextChunk",
    "EvidenceState",
    "RAGSample",
    "RelationPrediction",
]

#: 三元概率求和允许的浮点误差。
PROBABILITY_SUM_TOLERANCE: float = 1e-6

#: 去除首尾空白后不得为空的字符串。
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: 闭区间 [0, 1] 上的浮点数，用于概率与可靠度。
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class EvidenceState(str, Enum):
    """claim 相对于检索文档集合的证据状态。"""

    #: 检索文档充分支持 claim。
    SUPPORTED = "supported"
    #: 检索文档明确反驳 claim。
    REFUTED = "refuted"
    #: 检索文档没有足够信息判断 claim。
    INSUFFICIENT = "insufficient"
    #: 不同检索文档相互矛盾。
    CONFLICTING = "conflicting"


class _StrictModel(BaseModel):
    """所有数据模型的公共配置基类。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_duplicates(values: Iterable[str], field_name: str) -> None:
    """当 ``values`` 中出现重复项时抛出 ``ValueError``。"""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(
            f"同一样本内 {field_name} 必须唯一，重复值：{sorted(duplicates)}"
        )


class Claim(_StrictModel):
    """从答案中拆分出的一条原子断言。"""

    claim_id: NonEmptyStr
    text: NonEmptyStr


class ContextChunk(_StrictModel):
    """一段检索到的上下文文档。"""

    doc_id: NonEmptyStr
    text: NonEmptyStr
    #: 检索器给出的相关性分数；缺失时为 ``None``。
    retrieval_score: UnitFloat | None = None
    #: 文档来源的先验可靠度，默认完全可信。
    reliability: UnitFloat = 1.0


class RAGSample(_StrictModel):
    """一条完整的 RAG 评估样本。"""

    sample_id: NonEmptyStr
    question: NonEmptyStr
    answer: NonEmptyStr
    #: 参考答案；若不提供请使用 ``None`` 而不是空字符串。
    reference_answer: NonEmptyStr | None = None
    # 使用 tuple 而不是 list，避免 frozen 模型内部仍可通过 append 被修改。
    claims: tuple[Claim, ...] = Field(default_factory=tuple)
    contexts: tuple[ContextChunk, ...] = Field(default_factory=tuple)
    #: 人工标注的证据状态，用于评估；预测阶段可为 ``None``。
    gold_state: EvidenceState | None = None

    @model_validator(mode="after")
    def _check_unique_ids(self) -> RAGSample:
        """样本内的 ``claim_id`` 与 ``doc_id`` 都必须唯一。"""
        _reject_duplicates((claim.claim_id for claim in self.claims), "claim_id")
        _reject_duplicates((chunk.doc_id for chunk in self.contexts), "doc_id")
        return self


class RelationPrediction(_StrictModel):
    """关系评估器对单个 (claim, document) 对的输出。

    这是后续 RAGChecker / RAGAS 等评估器的统一输出格式；本阶段只定义
    结构，不实现任何评估器。
    """

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    doc_id: NonEmptyStr
    #: 产生该预测的评估器名称，例如 "ragchecker"。
    evaluator: NonEmptyStr
    p_support: UnitFloat
    p_refute: UnitFloat
    p_unknown: UnitFloat
    #: 该评估器自身的可靠度，供后续证据融合使用。
    evaluator_reliability: UnitFloat = 1.0

    @model_validator(mode="after")
    def _check_probabilities_sum_to_one(self) -> RelationPrediction:
        """三元概率之和必须为 1（允许 ``PROBABILITY_SUM_TOLERANCE`` 误差）。"""
        total = self.p_support + self.p_refute + self.p_unknown
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                "p_support + p_refute + p_unknown 必须等于 1，"
                f"当前为 {total!r}（允许误差 {PROBABILITY_SUM_TOLERANCE}）"
            )
        return self
