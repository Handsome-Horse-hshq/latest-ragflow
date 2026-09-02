"""三个 baseline 共用的小工具。

只包含「按 doc_id 找 ContextChunk」和「无证据时的空结果」两件事，
不含任何聚合规则 —— 各 baseline 的算法差异必须留在各自模块里。
"""

from __future__ import annotations

from rag_ds.baselines.models import (
    BaselineDecisionReason,
    BaselineMethod,
    BaselinePrediction,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample

__all__ = ["contexts_by_doc_id", "no_evidence_prediction"]


def contexts_by_doc_id(sample: RAGSample) -> dict[str, ContextChunk]:
    """按 ``doc_id`` 索引样本的检索文档。"""
    return {chunk.doc_id: chunk for chunk in sample.contexts}


def no_evidence_prediction(
    sample: RAGSample,
    claim: Claim,
    method: BaselineMethod,
    input_count: int,
    evaluator: str | None = None,
) -> BaselinePrediction:
    """构造「没有任何可用证据」时的结果。

    分数固定为 ``(0, 0, 1)``：全部质量落在不确定上。这与 D-S 侧的完全无知
    含义一致，但 baseline 无论如何都给不出 ``conflicting``。
    """
    return BaselinePrediction(
        sample_id=sample.sample_id,
        claim_id=claim.claim_id,
        method=method,
        evaluator=evaluator,
        score_support=0.0,
        score_refute=0.0,
        score_unknown=1.0,
        predicted_state=EvidenceState.INSUFFICIENT,
        reason=BaselineDecisionReason.NO_EVIDENCE,
        input_count=input_count,
        gold_state=sample.gold_state,  # 只带走，不参与计算
    )
