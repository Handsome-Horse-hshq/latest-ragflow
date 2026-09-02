"""二维门控的阈值、区域与诊断结果模型。

三个诊断量各自衡量不同的东西，互不替代：

* ``m_theta`` —— 证据不足程度（无知）；
* ``K_doc`` —— 文档之间的冲突程度；
* ``K_eval`` —— 评估器之间的意见冲突程度。

二维门控**只使用前两个**作为坐标轴：横轴 ``m_theta``、纵轴 ``K_doc``。
``K_eval`` 仅作为额外警报，不参与区域划分。

本模块只定义数据结构，判定逻辑在 :mod:`rag_ds.diagnostics.gating`。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.ds.mass import MASS_SUM_TOLERANCE
from rag_ds.schemas import EvidenceState, NonEmptyStr, UnitFloat

__all__ = [
    "ClaimVerdict",
    "DiagnosticRegion",
    "DiagnosticResult",
    "DiagnosticThresholds",
]

#: ``m_support - m_refute`` 的取值范围。
SignedUnitFloat = Annotated[float, Field(ge=-1.0, le=1.0)]


class DiagnosticRegion(str, Enum):
    """二维门控区域，外加两种完全冲突状态。"""

    #: m_theta 低、K_doc 低 —— 证据充分且一致。
    SUFFICIENT_CONSISTENT = "sufficient_consistent"
    #: m_theta 高、K_doc 低 —— 证据不足。
    INSUFFICIENT = "insufficient"
    #: m_theta 低、K_doc 高 —— 文档之间冲突。
    DOCUMENT_CONFLICT = "document_conflict"
    #: m_theta 高、K_doc 高 —— 既证据不足又存在文档冲突。
    INSUFFICIENT_AND_CONFLICTING = "insufficient_and_conflicting"
    #: 文档融合发生完全冲突，文档级质量未定义。
    DOCUMENT_TOTAL_CONFLICT = "document_total_conflict"
    #: 评估器融合发生完全冲突，评估器级质量未定义。
    EVALUATOR_TOTAL_CONFLICT = "evaluator_total_conflict"


class ClaimVerdict(str, Enum):
    """最终质量的支持/反驳倾向。

    这与 :class:`DiagnosticRegion` 是两件事：一个 claim 可能落在
    ``document_conflict`` 区域（文档之间打架），融合后的质量却仍轻微
    偏向 ``refuted``。区域说的是「证据状况如何」，verdict 说的是
    「质量偏向哪边」。
    """

    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNDETERMINED = "undetermined"


class DiagnosticThresholds(BaseModel):
    """二维门控与分歧警报使用的阈值。

    .. warning::
        这里的默认值 **0.5 / 0.4 / 0.4 仅供调试**，不是通过任何数据选出的
        正式取值。正式实验必须在**验证集**上选择阈值，且**不得**使用测试集
        参与选择。本阶段不实现任何阈值搜索或训练。

    判定「高」时统一使用 ``value >= threshold``（含等号），三个阈值一致，
    不混用 ``>`` 与 ``>=``。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: m_theta 达到该值即视为证据不足。调试默认值。
    theta_threshold: UnitFloat = 0.5
    #: K_doc 达到该值即视为存在文档冲突。调试默认值。
    document_conflict_threshold: UnitFloat = 0.4
    #: K_eval 达到该值即触发评估器分歧警报。调试默认值。
    evaluator_conflict_threshold: UnitFloat = 0.4
    #: ``|m_support - m_refute|`` 不超过该值时判为平局。
    tie_tolerance: UnitFloat = 1e-6


class DiagnosticResult(BaseModel):
    """单个 claim 的完整诊断结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    #: 最终 BPA 的三个焦元质量；完全冲突时为 ``None``。
    m_support: UnitFloat | None
    m_refute: UnitFloat | None
    m_theta: UnitFloat | None
    #: 文档冲突程度（评估器级诊断中为可靠性加权的 K_doc）。
    k_doc: UnitFloat
    #: 评估器意见冲突程度。
    k_eval: UnitFloat
    region: DiagnosticRegion
    verdict: ClaimVerdict
    #: 用于后续四分类比较的主状态；无法给出合理分类时为 ``None``。
    primary_state: EvidenceState | None
    #: ``m_theta >= theta_threshold``。
    evidence_insufficient: bool
    #: ``k_doc >= document_conflict_threshold``。
    document_conflict: bool
    #: ``k_eval >= evaluator_conflict_threshold``。仅为额外警报。
    evaluator_disagreement: bool
    #: ``m_support - m_refute``；正数偏向支持，负数偏向反驳。
    support_refute_margin: SignedUnitFloat | None
    #: 本次诊断实际使用的阈值，随结果一起保存以便复现。
    thresholds: DiagnosticThresholds

    @model_validator(mode="after")
    def _check_mass_consistency(self) -> DiagnosticResult:
        """三个质量要么同时存在并归一，要么同时缺失。"""
        masses = (self.m_support, self.m_refute, self.m_theta)
        missing = [value is None for value in masses]

        if any(missing):
            if not all(missing):
                raise ValueError("三个质量必须同时存在或同时为 None")
            if self.region not in _TOTAL_CONFLICT_REGIONS:
                raise ValueError(
                    f"只有完全冲突区域允许质量为 None，当前 region={self.region.value}"
                )
            if self.support_refute_margin is not None:
                raise ValueError("质量为 None 时 support_refute_margin 也必须为 None")
            return self

        if self.region in _TOTAL_CONFLICT_REGIONS:
            raise ValueError(
                f"完全冲突区域 {self.region.value} 的三个质量必须为 None"
            )

        total = self.m_support + self.m_refute + self.m_theta  # type: ignore[operator]
        if abs(total - 1.0) > MASS_SUM_TOLERANCE:
            raise ValueError(
                "m_support + m_refute + m_theta 必须等于 1，"
                f"当前为 {total!r}（允许误差 {MASS_SUM_TOLERANCE}）"
            )

        if self.support_refute_margin is None:
            raise ValueError("质量存在时必须给出 support_refute_margin")
        expected_margin = self.m_support - self.m_refute  # type: ignore[operator]
        if abs(self.support_refute_margin - expected_margin) > MASS_SUM_TOLERANCE:
            raise ValueError(
                "support_refute_margin 必须等于 m_support - m_refute，"
                f"当前为 {self.support_refute_margin!r}，期望 {expected_margin!r}"
            )

        return self


#: 质量允许缺失的区域。
_TOTAL_CONFLICT_REGIONS = frozenset(
    {
        DiagnosticRegion.DOCUMENT_TOTAL_CONFLICT,
        DiagnosticRegion.EVALUATOR_TOTAL_CONFLICT,
    }
)
