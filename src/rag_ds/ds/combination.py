"""两条 BPA 的标准归一化 Dempster 组合规则，以及单次冲突量 K。

识别框架仍是 ``Theta = {Support, Refute}``。两条证据 ``m1``、``m2`` 的
组合分三步：

1. 冲突量（**归一化之前**计算）::

       K = m1(S) * m2(R) + m1(R) * m2(S)

2. 未归一化质量::

       S_raw     = m1(S)m2(S) + m1(S)m2(Theta) + m1(Theta)m2(S)
       R_raw     = m1(R)m2(R) + m1(R)m2(Theta) + m1(Theta)m2(R)
       Theta_raw = m1(Theta)m2(Theta)

3. 归一化（分母为 ``1 - K``）::

       m(S)     = S_raw / (1 - K)
       m(R)     = R_raw / (1 - K)
       m(Theta) = Theta_raw / (1 - K)

关于 K 必须单独保留
-------------------
K 是**归一化之前**被两条证据判定为互相矛盾的那部分质量。归一化把它从
分子中抹掉、再把剩余质量放大回和为 1，因此融合结果 ``m(S)``、``m(R)``、
``m(Theta)`` 本身**无法反映原始冲突有多大**：两条温和一致的证据与两条
剧烈矛盾的证据完全可能给出相近的融合质量。这正是 Zadeh 反例的根源。

所以 :class:`PairwiseCombinationResult` 必须把 K 与融合结果一起保存 ——
丢掉 K 就等于丢掉「这个结论有多可疑」这一信息。

关于命名
--------
本模块的 ``conflict`` 只是**两条 BPA 的单次冲突**，不是 ``K_doc``，
也不是 ``K_eval``。后者是聚合层按证据来源（同一评估器下的多篇文档 /
同一文档下的多个评估器）分别累计出来的量，将在后续阶段实现。在这里
把它命名为 K_doc 会把两个不同层次的量混为一谈。

本模块只实现标准归一化 Dempster 规则，不含 Yager、Dubois-Prade 或任何
未归一化变体，也不含多证据循环融合。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from rag_ds.ds._numeric import clamp_unit
from rag_ds.ds.mass import MASS_SUM_TOLERANCE, MassFunction
from rag_ds.schemas import UnitFloat

__all__ = [
    "TOTAL_CONFLICT_EPSILON",
    "CombinedMass",
    "MassLike",
    "PairwiseCombinationResult",
    "TotalConflictError",
    "combine_two_masses",
]

#: 判定「完全冲突」的分母下界。``1 - K <= TOTAL_CONFLICT_EPSILON`` 时
#: 归一化没有定义，直接抛 :class:`TotalConflictError`。
TOTAL_CONFLICT_EPSILON: float = 1e-12

#: 归一化环节允许被夹回区间端点的浮点噪声上限。
_FLOAT_SLACK = 1e-12

#: ``normalization_denominator`` 与 ``1 - conflict`` 允许的偏差。
_DENOMINATOR_CONSISTENCY_TOLERANCE = 1e-9


class CombinedMass(BaseModel):
    """融合后的 BPA，只保存三个焦元的质量。

    这里**刻意不携带** ``sample_id`` / ``claim_id`` / ``doc_id`` /
    ``evaluator``：融合结果来自两条证据，任何单一 ID 都是伪造的，
    而 ``"doc1+doc2"``、``"combined"`` 这类拼接值会让下游误以为它是一篇
    真实文档。融合来源与业务元数据由后续聚合层单独记录。

    ``m_theta`` 的含义与 :class:`~rag_ds.ds.mass.MassFunction` 一致：
    分配给整个识别框架 ``{Support, Refute}`` 的质量，即无知，不是第三个
    互斥类别。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    m_support: UnitFloat
    m_refute: UnitFloat
    m_theta: UnitFloat

    @model_validator(mode="after")
    def _check_masses_sum_to_one(self) -> CombinedMass:
        """三个焦元的质量之和必须为 1（允许 ``MASS_SUM_TOLERANCE`` 误差）。"""
        total = self.m_support + self.m_refute + self.m_theta
        if abs(total - 1.0) > MASS_SUM_TOLERANCE:
            raise ValueError(
                "m_support + m_refute + m_theta 必须等于 1，"
                f"当前为 {total!r}（允许误差 {MASS_SUM_TOLERANCE}）"
            )
        return self


#: 可以参与组合的 BPA：第五阶段带 ID 的 BPA，或本阶段的融合结果。
MassLike = MassFunction | CombinedMass


class PairwiseCombinationResult(BaseModel):
    """一次两两组合的完整结果。

    除融合后的 BPA 外，还保存归一化之前的冲突量 ``conflict``，因为它
    在归一化后无法从 ``mass`` 反推出来。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 融合后的 BPA。
    mass: CombinedMass
    #: 归一化**之前**计算的单次冲突量 K。不是 K_doc，也不是 K_eval。
    conflict: UnitFloat
    #: 归一化分母，等于 ``1 - conflict``。
    normalization_denominator: UnitFloat

    @model_validator(mode="after")
    def _check_denominator_matches_conflict(self) -> PairwiseCombinationResult:
        """``normalization_denominator`` 必须等于 ``1 - conflict``。"""
        expected = 1.0 - self.conflict
        if abs(self.normalization_denominator - expected) > (
            _DENOMINATOR_CONSISTENCY_TOLERANCE
        ):
            raise ValueError(
                "normalization_denominator 必须等于 1 - conflict，"
                f"当前 conflict={self.conflict!r}，"
                f"normalization_denominator={self.normalization_denominator!r}"
            )
        return self


class TotalConflictError(ArithmeticError):
    """两条证据完全冲突，标准归一化 Dempster 规则没有定义。

    ``K`` 与 ``1 - K`` 同时以属性形式保留，方便调用方决定后续策略
    （例如改用其他组合规则，或把该组合标记为不可判定）。
    """

    def __init__(self, conflict: float, denominator: float) -> None:
        self.conflict = conflict
        self.denominator = denominator
        super().__init__(
            "两条证据完全冲突，标准归一化 Dempster 组合规则在此处没有定义："
            f"K={conflict!r}，1-K={denominator!r}"
            f"（阈值 {TOTAL_CONFLICT_EPSILON}）。"
            "本项目不会用 epsilon 替代分母强行计算，"
            "也不会返回全零质量、m_theta=1 或任一侧证据 —— "
            "那些做法都会悄悄改变算法含义。"
        )


def combine_two_masses(left: MassLike, right: MassLike) -> PairwiseCombinationResult:
    """用标准归一化 Dempster 规则组合两条 BPA。

    只读取 ``m_support`` / ``m_refute`` / ``m_theta`` 三个字段，因此
    :class:`~rag_ds.ds.mass.MassFunction` 与 :class:`CombinedMass` 可以
    任意搭配。``reliability_applied``、各类 ID、``evaluator`` 名称以及
    任何文本都**不参与**公式 —— 可靠性折扣是上一阶段的事，在这里再折一次
    就重复计入了。

    公式见模块文档字符串。冲突量 K 在归一化之前算出并原样保留。

    Args:
        left: 第一条证据。
        right: 第二条证据。两个参数可交换，结果相同。

    Returns:
        :class:`PairwiseCombinationResult`，含融合后的 BPA、K 与 ``1 - K``。

    Raises:
        TotalConflictError: ``1 - K <= TOTAL_CONFLICT_EPSILON``，即两条
            证据完全（或数值上无法区分于完全）冲突。

    Note:
        传入的两个对象都不会被修改；两者本身也都是不可变模型。

        数值提醒：``1 - K`` 由两个接近 1 的数相减得到，K 逼近 1 时会发生
        灾难性抵消。实测在 ``1 - K`` 约处于 ``[1e-12, 2e-11]`` 区间时，
        归一化后三个质量之和偏离 1 已超过 ``MASS_SUM_TOLERANCE``，此时
        :class:`CombinedMass` 会抛出校验错误而不是返回结果。这是刻意的：
        分母本身的有效位数已所剩无几，宁可大声报错，也不返回一个不可信
        的数值。``1 - K`` 大于该区间时归一化稳定，和与 1 的偏差在 1e-8
        以内。
    """
    # ---- 第一步：归一化之前计算冲突量 K ----
    conflict = clamp_unit(
        left.m_support * right.m_refute + left.m_refute * right.m_support,
        "conflict",
        _FLOAT_SLACK,
    )
    denominator = 1.0 - conflict

    # ---- 第二步：完全冲突检测，必须在做除法之前 ----
    if denominator <= TOTAL_CONFLICT_EPSILON:
        raise TotalConflictError(conflict, denominator)

    # ---- 第三步：未归一化质量 ----
    support_raw = (
        left.m_support * right.m_support
        + left.m_support * right.m_theta
        + left.m_theta * right.m_support
    )
    refute_raw = (
        left.m_refute * right.m_refute
        + left.m_refute * right.m_theta
        + left.m_theta * right.m_refute
    )
    theta_raw = left.m_theta * right.m_theta

    # ---- 第四步：归一化 ----
    mass = CombinedMass(
        m_support=clamp_unit(support_raw / denominator, "m_support", _FLOAT_SLACK),
        m_refute=clamp_unit(refute_raw / denominator, "m_refute", _FLOAT_SLACK),
        m_theta=clamp_unit(theta_raw / denominator, "m_theta", _FLOAT_SLACK),
    )

    return PairwiseCombinationResult(
        mass=mass,
        conflict=conflict,
        normalization_denominator=denominator,
    )
