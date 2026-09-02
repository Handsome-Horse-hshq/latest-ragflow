"""二维门控诊断。

根据 ``m_theta``（证据不足）与 ``K_doc``（文档冲突）划分区域，并用
``K_eval``（评估器意见冲突）生成额外的分歧警报。

本子包不含阈值搜索或训练：阈值由调用方显式给出，
:class:`~rag_ds.diagnostics.models.DiagnosticThresholds` 的默认值仅供调试。
"""

from rag_ds.diagnostics.gating import (
    determine_verdict,
    diagnose_document_total_conflict,
    diagnose_evaluator_result,
    diagnose_no_evidence,
)
from rag_ds.diagnostics.models import (
    ClaimVerdict,
    DiagnosticRegion,
    DiagnosticResult,
    DiagnosticThresholds,
)

__all__ = [
    "ClaimVerdict",
    "DiagnosticRegion",
    "DiagnosticResult",
    "DiagnosticThresholds",
    "determine_verdict",
    "diagnose_document_total_conflict",
    "diagnose_evaluator_result",
    "diagnose_no_evidence",
]
