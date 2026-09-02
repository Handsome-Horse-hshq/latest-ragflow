"""Dempster-Shafer 证据理论模块。

完整链路::

    关系概率
      -> 文档可靠性折扣      （每条文档一次）
      -> 同一评估器内融合文档 -> 评估器级 BPA 与该评估器的 K_doc
      -> 评估器可靠性折扣    （每个评估器只有一次）
      -> 融合多个评估器      -> 最终 BPA、K_eval 与加权 K_doc

二维门控、最终诊断类别与阈值搜索尚未实现。
"""

from rag_ds.ds.combination import (
    TOTAL_CONFLICT_EPSILON,
    CombinedMass,
    MassLike,
    PairwiseCombinationResult,
    TotalConflictError,
    combine_two_masses,
)
from rag_ds.ds.discount import (
    discount_combined_mass,
    discount_mass,
    discounted_mass_from_prediction,
    document_discounted_mass_from_prediction,
    effective_reliability,
)
from rag_ds.ds.document_aggregation import (
    DocumentAggregationResult,
    DocumentCombinationStep,
    EmptyEvidenceError,
    aggregate_document_masses,
)
from rag_ds.ds.evaluator_aggregation import (
    EvaluatorAggregationResult,
    EvaluatorCombinationStep,
    EvaluatorDocumentDiagnostic,
    EvaluatorEvidence,
    UndefinedDocumentMassError,
    aggregate_evaluators,
)
from rag_ds.ds.mass import MASS_SUM_TOLERANCE, MassFunction, mass_from_prediction

__all__ = [
    "MASS_SUM_TOLERANCE",
    "TOTAL_CONFLICT_EPSILON",
    "CombinedMass",
    "DocumentAggregationResult",
    "DocumentCombinationStep",
    "EmptyEvidenceError",
    "EvaluatorAggregationResult",
    "EvaluatorCombinationStep",
    "EvaluatorDocumentDiagnostic",
    "EvaluatorEvidence",
    "MassFunction",
    "MassLike",
    "PairwiseCombinationResult",
    "TotalConflictError",
    "UndefinedDocumentMassError",
    "aggregate_document_masses",
    "aggregate_evaluators",
    "combine_two_masses",
    "discount_combined_mass",
    "discount_mass",
    "discounted_mass_from_prediction",
    "document_discounted_mass_from_prediction",
    "effective_reliability",
    "mass_from_prediction",
]
