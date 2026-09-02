"""关系评估器：统一接口与可控的假实现。

本子包只定义接口和查表式的假评估器，不包含 RAGChecker、RAGAS、
大模型调用或任何文本判断逻辑。
"""

from rag_ds.relation_evaluation.base import RelationEvaluator
from rag_ds.relation_evaluation.llm_evaluator import (
    LLMRelationEvaluator,
    RelationProbabilityCaller,
)
from rag_ds.relation_evaluation.mock import (
    MissingMockPredictionError,
    MockRelationEvaluator,
)
from rag_ds.relation_evaluation.ragchecker_adapter import (
    DEFAULT_LABEL_MAPPING,
    LabelProbabilityMapping,
    MissingRAGCheckerJudgementError,
    RAGCheckerLabel,
    RAGCheckerRelationAdapter,
    prediction_from_probabilities,
)

__all__ = [
    "DEFAULT_LABEL_MAPPING",
    "LLMRelationEvaluator",
    "LabelProbabilityMapping",
    "MissingMockPredictionError",
    "MissingRAGCheckerJudgementError",
    "MockRelationEvaluator",
    "RAGCheckerLabel",
    "RAGCheckerRelationAdapter",
    "RelationEvaluator",
    "RelationProbabilityCaller",
    "prediction_from_probabilities",
]
