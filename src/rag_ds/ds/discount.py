"""Shafer 折扣：把可靠性作用到 BPA 上。

折扣算子把「证据源不完全可信」表达成「一部分质量退回到无知」::

    m'(S)     = r * m(S)
    m'(R)     = r * m(R)
    m'(Theta) = 1 - m'(S) - m'(R)

质量只会从确定的焦元流向 ``Theta``，绝不会反向流动：可靠性越低，
支持与反驳的质量越少，无知越多；``r = 0`` 时 BPA 退化为完全无知
（``m_theta = 1``），该证据源在后续融合中不产生任何影响。

两级可靠性的作用位置
--------------------
完整链路是::

    关系概率
      -> 文档可靠性折扣   （每条文档一次，用 context.reliability）
      -> 同一评估器内融合文档
      -> 评估器可靠性折扣 （每个评估器只有一次，用 evaluator_reliability）
      -> 融合多个评估器

**评估器可靠性必须在文档融合完成之后只作用一次。** 若在每条文档上都乘
一遍 ``evaluator_reliability``，它会随文档数量被重复计入：同一个评估器
看了 5 篇文档，其可靠性就被折了 5 次，结果凭空受文档数量影响。
:func:`document_discounted_mass_from_prediction` 因此只使用
``context.reliability``；评估器级折扣由 :func:`discount_combined_mass`
在聚合层单独完成。

本模块不含任何证据融合。
"""

from __future__ import annotations

import warnings

from rag_ds.ds._numeric import clamp_unit
from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.mass import MassFunction, mass_from_prediction
from rag_ds.schemas import ContextChunk, RelationPrediction

__all__ = [
    "discount_combined_mass",
    "discount_mass",
    "discounted_mass_from_prediction",
    "document_discounted_mass_from_prediction",
    "effective_reliability",
]

#: 允许被夹回区间端点的浮点噪声上限。
#:
#: 这里只吸收 IEEE-754 舍入产生的、量级在 1e-16 附近的偏差，例如
#: ``1 - 0.7 - 0.3`` 得到 ``-1.1e-16``。真正越界的数值仍会照常报错 ——
#: 本模块不做静默归一化。
_FLOAT_SLACK = 1e-9


def _check_reliability(reliability: float) -> None:
    """确认可靠性位于 ``[0, 1]``；``NaN`` 同样被拒绝。"""
    if not 0.0 <= reliability <= 1.0:
        raise ValueError(f"reliability 必须位于 [0, 1]，收到 {reliability!r}")


def _discounted_triple(
    m_support: float, m_refute: float, reliability: float
) -> tuple[float, float, float]:
    """折扣公式的唯一实现，供所有折扣函数共用。

    抽成一处是为了保证 :func:`discount_mass` 与
    :func:`discount_combined_mass` 永远给出完全相同的数值 —— 两套并行
    维护的折扣公式迟早会漂移。

    Args:
        m_support: 折扣前的 ``m(S)``。
        m_refute: 折扣前的 ``m(R)``。
        reliability: 折扣系数，调用方需先校验范围。

    Returns:
        折扣后的 ``(m(S), m(R), m(Theta))``。
    """
    discounted_support = clamp_unit(reliability * m_support, "m_support", _FLOAT_SLACK)
    discounted_refute = clamp_unit(reliability * m_refute, "m_refute", _FLOAT_SLACK)
    # 折扣损失的确定质量全部回流到 Theta；不会有质量从 Theta 流出。
    discounted_theta = clamp_unit(
        1.0 - discounted_support - discounted_refute, "m_theta", _FLOAT_SLACK
    )
    return discounted_support, discounted_refute, discounted_theta


def discount_mass(mass: MassFunction, reliability: float) -> MassFunction:
    """对带 ID 的文档级 BPA 施加 Shafer 折扣。

    计算规则::

        m'(S)     = reliability * m(S)
        m'(R)     = reliability * m(R)
        m'(Theta) = 1 - m'(S) - m'(R)

    累计可靠性按乘法累积::

        reliability_applied' = reliability_applied * reliability

    因此连续折扣与一次性折扣其乘积等价：``discount(discount(m, a), b)``
    与 ``discount(m, a * b)`` 结果相同。

    Args:
        mass: 待折扣的 BPA；不会被修改。
        reliability: 折扣系数，必须位于 ``[0, 1]``。

    Returns:
        折扣后的新 :class:`MassFunction`。``reliability = 1`` 时数值不变；
        ``reliability = 0`` 时退化为 ``m_theta = 1`` 的完全无知状态。

    Raises:
        ValueError: ``reliability`` 不在 ``[0, 1]`` 内（``NaN`` 同样被拒绝）。
    """
    _check_reliability(reliability)
    support, refute, theta = _discounted_triple(
        mass.m_support, mass.m_refute, reliability
    )

    return MassFunction(
        sample_id=mass.sample_id,
        claim_id=mass.claim_id,
        doc_id=mass.doc_id,
        evaluator=mass.evaluator,
        m_support=support,
        m_refute=refute,
        m_theta=theta,
        reliability_applied=clamp_unit(
            mass.reliability_applied * reliability,
            "reliability_applied",
            _FLOAT_SLACK,
        ),
    )


def discount_combined_mass(mass: CombinedMass, reliability: float) -> CombinedMass:
    """对融合后的 BPA 施加 Shafer 折扣。

    与 :func:`discount_mass` 共用同一份折扣公式（:func:`_discounted_triple`），
    只是操作的模型不携带 ID 与 ``reliability_applied``。

    典型用途是评估器级折扣：某个评估器的所有文档融合完成后，用它自己的
    ``evaluator_reliability`` **只折一次**。

    Args:
        mass: 待折扣的融合 BPA；不会被修改。
        reliability: 折扣系数，必须位于 ``[0, 1]``。

    Returns:
        折扣后的新 :class:`CombinedMass`。``reliability = 1`` 时数值不变；
        ``reliability = 0`` 时退化为 ``(0, 0, 1)``。

    Raises:
        ValueError: ``reliability`` 不在 ``[0, 1]`` 内。
    """
    _check_reliability(reliability)
    support, refute, theta = _discounted_triple(
        mass.m_support, mass.m_refute, reliability
    )
    return CombinedMass(m_support=support, m_refute=refute, m_theta=theta)


def document_discounted_mass_from_prediction(
    prediction: RelationPrediction,
    context: ContextChunk,
) -> MassFunction:
    """由一条关系预测和它对应的文档，得到**文档级**折扣后的 BPA。

    处理顺序：

    1. 确认 ``prediction.doc_id`` 与 ``context.doc_id`` 指向同一篇文档；
    2. 用 :func:`~rag_ds.ds.mass.mass_from_prediction` 得到基础 BPA；
    3. 用 ``context.reliability`` 施加折扣。

    公式::

        m_doc(S)     = context.reliability * p_support
        m_doc(R)     = context.reliability * p_refute
        m_doc(Theta) = 1 - m_doc(S) - m_doc(R)

    此时 ``reliability_applied == context.reliability``。

    ``prediction.evaluator_reliability`` **在这里不被使用**，它要留到该
    评估器的所有文档融合完毕之后，由 :func:`discount_combined_mass` 只
    施加一次；在每条文档上都乘一遍会让评估器可靠性随文档数量被重复计入。

    ``retrieval_score`` 同样不参与 —— 检索相关性衡量「这段文档与问题有多
    相关」，与「这段文档有多可信」是两回事。

    Args:
        prediction: 关系评估器对该 (claim, document) 对的输出。
        context: 该预测所针对的检索文档。

    Returns:
        只应用了文档可靠性的 :class:`MassFunction`。

    Raises:
        ValueError: 两个 ``doc_id`` 不一致 —— 说明调用方把预测和文档配错了，
            这是静默算错结果的常见来源，因此直接拒绝。

    Note:
        本函数不接收 :class:`~rag_ds.schemas.RAGSample`，因此不可能读到
        ``gold_state``；也不读取任何文本字段与 ``retrieval_score``。
    """
    if prediction.doc_id != context.doc_id:
        raise ValueError(
            "prediction 与 context 指向不同的文档："
            f"prediction.doc_id={prediction.doc_id!r}, "
            f"context.doc_id={context.doc_id!r}"
        )

    base_mass = mass_from_prediction(prediction)
    return discount_mass(base_mass, context.reliability)


def discounted_mass_from_prediction(
    prediction: RelationPrediction,
    context: ContextChunk,
) -> MassFunction:
    """已废弃：请改用 :func:`document_discounted_mass_from_prediction`。

    早期版本在这里同时乘上 ``context.reliability`` 与
    ``prediction.evaluator_reliability``，会让评估器可靠性随文档数量被
    重复计入。本函数现已与新名字行为完全一致（**只应用文档可靠性**），
    保留仅为兼容旧调用，会发出 :class:`DeprecationWarning`。

    .. deprecated::
        改用 :func:`document_discounted_mass_from_prediction`。
    """
    warnings.warn(
        "discounted_mass_from_prediction 已废弃，请改用 "
        "document_discounted_mass_from_prediction；文档级折扣只应用 "
        "context.reliability，评估器可靠性在文档融合之后单独施加一次。",
        DeprecationWarning,
        stacklevel=2,
    )
    return document_discounted_mass_from_prediction(prediction, context)


def effective_reliability(
    prediction: RelationPrediction, context: ContextChunk
) -> float:
    """已废弃：文档可靠性与评估器可靠性不应在同一处相乘。

    这个乘积正是本阶段要修掉的错误做法 —— 两级可靠性作用在链路的不同
    位置：``context.reliability`` 在每条文档上、``evaluator_reliability``
    在该评估器的文档融合完成之后。把它们提前相乘，等价于在每条文档上都
    折一次评估器可靠性。

    .. deprecated::
        文档级请用 ``context.reliability``；评估器级请把
        ``evaluator_reliability`` 交给 :func:`discount_combined_mass`。
    """
    warnings.warn(
        "effective_reliability 已废弃：两级可靠性作用在链路的不同位置，"
        "提前相乘会让评估器可靠性随文档数量被重复计入。",
        DeprecationWarning,
        stacklevel=2,
    )
    return clamp_unit(
        context.reliability * prediction.evaluator_reliability,
        "effective_reliability",
        _FLOAT_SLACK,
    )
