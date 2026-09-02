"""Baseline 的枚举、阈值与结果模型。

三个 baseline 都**不使用** Dempster-Shafer 证据理论：它们把关系概率直接
压成三个分数再取最大值。这是刻意的对照设计 —— 实验要展示的正是这类朴素
聚合无法区分「证据不足」与「证据冲突」：两条针锋相对的文档在平均或投票
之后，只会表现为一个低分或一个平局，看不出「有人说是、有人说否」。

因此 :class:`BaselinePrediction` 的 ``predicted_state`` **在结构上禁止**
取 ``CONFLICTING`` —— 这不是实现偷懒，而是被比较对象的固有局限。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.integrity import PipelineError
from rag_ds.schemas import EvidenceState, NonEmptyStr, UnitFloat

__all__ = [
    "BASELINE_SCORE_SUM_TOLERANCE",
    "BaselineDecisionReason",
    "BaselineMethod",
    "BaselinePrediction",
    "BaselineThresholds",
    "MissingBaselineEvaluatorError",
]

#: 三个分数求和允许的浮点误差。
BASELINE_SCORE_SUM_TOLERANCE: float = 1e-6

#: baseline 允许输出的状态；``CONFLICTING`` 被刻意排除在外。
ALLOWED_BASELINE_STATES = frozenset(
    {
        EvidenceState.SUPPORTED,
        EvidenceState.REFUTED,
        EvidenceState.INSUFFICIENT,
    }
)


class MissingBaselineEvaluatorError(PipelineError):
    """single-evaluator baseline 指定的评估器在数据中不存在。

    不会退化成「用别的评估器代替」或「返回空结果」：指定了哪个评估器就必须
    用哪个，否则实验里的「单评估器」这一条对照就名不副实。
    """


class BaselineMethod(str, Enum):
    """三种 baseline 方法。"""

    #: 按 文档可靠性 × 评估器可靠性 加权平均三个概率。
    WEIGHTED_AVERAGE = "weighted_average"
    #: 每条关系预测一票，少数服从多数。
    MAJORITY_VOTE = "majority_vote"
    #: 只使用一个指定评估器，按文档可靠性加权平均。
    SINGLE_EVALUATOR = "single_evaluator"


class BaselineDecisionReason(str, Enum):
    """得出该结论的直接原因。"""

    #: 最高分明确胜出且达到阈值。
    DECIDED = "decided"
    #: unknown 分数最高。
    UNKNOWN_HIGHEST = "unknown_highest"
    #: 最高分低于 decision_threshold。
    BELOW_THRESHOLD = "below_threshold"
    #: 最高分之间差距在容差内，分不出胜负。
    SCORE_TIE = "score_tie"
    #: 没有任何可用证据（权重和为零或无票）。
    NO_EVIDENCE = "no_evidence"


class BaselineThresholds(BaseModel):
    """baseline 判定使用的阈值。

    .. warning::
        默认值 **0.5 / 1e-6 仅供调试**，不是通过任何数据选出的正式取值。
        正式实验必须在**验证集**上选择，测试集不得参与选择。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 最高分低于该值即判为证据不足。调试默认值。
    decision_threshold: UnitFloat = 0.5
    #: 最高分之间差距不超过该值时视为平局。
    tie_tolerance: UnitFloat = 1e-6


class BaselinePrediction(BaseModel):
    """单条 claim 在单个 baseline 方法下的预测结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    method: BaselineMethod
    #: 仅 single_evaluator 方法有值，其余为 ``None``。
    evaluator: NonEmptyStr | None = None

    score_support: UnitFloat
    score_refute: UnitFloat
    score_unknown: UnitFloat

    predicted_state: EvidenceState
    reason: BaselineDecisionReason

    #: 参与本次计算的关系预测条数。
    input_count: int = Field(ge=0)
    #: 人工标注，**仅供运行后对比**，计算过程从不读取。
    gold_state: EvidenceState | None = None

    @model_validator(mode="after")
    def _check_scores_and_state(self) -> BaselinePrediction:
        """三个分数须归一，且状态不得为 ``CONFLICTING``。"""
        total = self.score_support + self.score_refute + self.score_unknown
        if abs(total - 1.0) > BASELINE_SCORE_SUM_TOLERANCE:
            raise ValueError(
                "score_support + score_refute + score_unknown 必须等于 1，"
                f"当前为 {total!r}（允许误差 {BASELINE_SCORE_SUM_TOLERANCE}）"
            )

        if self.predicted_state not in ALLOWED_BASELINE_STATES:
            raise ValueError(
                f"baseline 不允许输出 {self.predicted_state.value!r}；"
                "只能是 supported / refuted / insufficient —— "
                "无法识别证据冲突正是这类朴素方法的固有局限"
            )

        if (self.method is BaselineMethod.SINGLE_EVALUATOR) != (
            self.evaluator is not None
        ):
            raise ValueError(
                "只有 single_evaluator 方法可以（且必须）给出 evaluator，"
                f"当前 method={self.method.value!r}, evaluator={self.evaluator!r}"
            )

        return self
