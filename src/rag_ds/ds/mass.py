"""基本概率分配（BPA）及其数据模型。

识别框架固定为两个互斥假设::

    Theta = {Support, Refute}

于是幂集上只有三个可能承载质量的焦元：``{Support}``、``{Refute}`` 和
``Theta`` 本身（空集质量恒为 0）。三者的质量之和为 1。

关于 ``m_theta``
----------------
``m_theta`` 是分配给**整个识别框架** ``{Support, Refute}`` 的质量，表示
「当前证据尚不能区分支持与反驳」，即无知（ignorance）。

它**不是**与 Support、Refute 并列的第三个互斥类别。二者的区别在后续的
Dempster 组合中会体现出来：Theta 上的质量可以与任一焦元相交并把质量让渡
给对方，而三个互斥类别之间只会产生冲突。把 Theta 当成「第三类」会得到
完全错误的融合结果。

本模块只做概率到 BPA 的搬运，不含可靠性折扣（见 :mod:`rag_ds.ds.discount`），
也不含任何证据融合。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from rag_ds.schemas import (
    PROBABILITY_SUM_TOLERANCE,
    NonEmptyStr,
    RelationPrediction,
    UnitFloat,
)

__all__ = [
    "MASS_SUM_TOLERANCE",
    "MassFunction",
    "mass_from_prediction",
]

#: 三个焦元质量求和允许的浮点误差。
#:
#: 与 :data:`rag_ds.schemas.PROBABILITY_SUM_TOLERANCE` 刻意取同一个值：
#: BPA 直接由三元概率搬运而来，两者的归一化要求必须一致，否则勉强通过
#: ``RelationPrediction`` 校验的记录会在建 BPA 时莫名其妙地被拒。
MASS_SUM_TOLERANCE: float = PROBABILITY_SUM_TOLERANCE


class MassFunction(BaseModel):
    """单个 (claim, document, evaluator) 组合上的基本概率分配。

    识别框架为 ``Theta = {Support, Refute}``，三个焦元的质量满足::

        m_support + m_refute + m_theta = 1

    ``m_theta`` 是分配给整个框架 ``{Support, Refute}`` 的质量，表示证据
    尚不足以区分支持与反驳，**不是**第三个互斥类别。

    对象是不可变的（``frozen=True``）：任何变换都返回新实例，不会就地
    修改已有的 BPA。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    doc_id: NonEmptyStr
    #: 产生该 BPA 的评估器名称。
    evaluator: NonEmptyStr
    #: 焦元 ``{Support}`` 上的质量。
    m_support: UnitFloat
    #: 焦元 ``{Refute}`` 上的质量。
    m_refute: UnitFloat
    #: 焦元 ``Theta = {Support, Refute}`` 上的质量，即无知程度。
    m_theta: UnitFloat
    #: 已经作用在这个 BPA 上的累计可靠性；未折扣时为 1.0。
    reliability_applied: UnitFloat = 1.0

    @model_validator(mode="after")
    def _check_masses_sum_to_one(self) -> MassFunction:
        """三个焦元的质量之和必须为 1（允许 :data:`MASS_SUM_TOLERANCE` 误差）。"""
        total = self.m_support + self.m_refute + self.m_theta
        if abs(total - 1.0) > MASS_SUM_TOLERANCE:
            raise ValueError(
                "m_support + m_refute + m_theta 必须等于 1，"
                f"当前为 {total!r}（允许误差 {MASS_SUM_TOLERANCE}）"
            )
        return self


def mass_from_prediction(prediction: RelationPrediction) -> MassFunction:
    """把关系评估器的三元概率原样搬成基础 BPA。

    映射规则::

        m_support = p_support
        m_refute  = p_refute
        m_theta   = p_unknown

    ``p_unknown`` 落到 ``m_theta`` 而不是某个第三类别，正是因为「不知道」
    在证据理论里就是把质量留在整个框架 ``{Support, Refute}`` 上。

    本函数**不应用任何可靠性**：``reliability_applied`` 固定为 ``1.0``，
    ``prediction.evaluator_reliability`` 在这里被刻意忽略，折扣由
    :func:`rag_ds.ds.discount.discount_mass` 单独负责。

    Args:
        prediction: 关系评估器的输出。

    Returns:
        未经折扣的 :class:`MassFunction`；四个标识字段原样保留。

    Note:
        传入的 ``prediction`` 不会被修改，返回的是全新对象。
    """
    return MassFunction(
        sample_id=prediction.sample_id,
        claim_id=prediction.claim_id,
        doc_id=prediction.doc_id,
        evaluator=prediction.evaluator,
        m_support=prediction.p_support,
        m_refute=prediction.p_refute,
        m_theta=prediction.p_unknown,
        reliability_applied=1.0,
    )
