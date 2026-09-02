"""``ds`` 子包内部共用的顺序融合工具。

文档级聚合与评估器级聚合的折叠过程完全同构：都是「以第一条证据为初始
累计 BPA，按输入顺序逐条调用 :func:`combine_two_masses`，记录每一步的
冲突量，遇到完全冲突则停止」。这里只实现这段公共骨架，两个聚合模块各自
把结果映射成自己的步骤模型。

本模块不实现任何组合公式 —— Dempster 规则由
:func:`rag_ds.ds.combination.combine_two_masses` 独家负责。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_ds.ds._numeric import clamp_unit
from rag_ds.ds.combination import (
    CombinedMass,
    MassLike,
    TotalConflictError,
    combine_two_masses,
)

__all__ = ["FoldStep", "accumulate_conflict", "sequential_combine", "to_combined_mass"]

#: 冲突累计允许被夹回区间端点的浮点噪声上限。
_FLOAT_SLACK = 1e-12


@dataclass(frozen=True)
class FoldStep:
    """一次顺序融合的原始记录，尚未映射成具体的步骤模型。"""

    #: 从 1 开始的步骤编号。
    index: int
    #: 本步的单次冲突量 K_i。完全冲突时按约定记为 1.0。
    conflict: float
    #: 本步的 1 - K_i。完全冲突时按约定记为 0.0。
    denominator: float
    #: 本步融合后的 BPA；完全冲突时为 ``None``。
    result_mass: CombinedMass | None
    #: 本步是否发生完全冲突。
    is_total_conflict: bool


def to_combined_mass(mass: MassLike) -> CombinedMass:
    """把任意 BPA 转成只含三个焦元质量的 :class:`CombinedMass`。

    :class:`~rag_ds.ds.mass.MassFunction` 携带的 ID 与
    ``reliability_applied`` 在融合结果里没有意义，因此被丢弃而不是
    伪造成拼接值。

    Args:
        mass: 待转换的 BPA。

    Returns:
        等值的 :class:`CombinedMass`；输入对象不会被修改。
    """
    if isinstance(mass, CombinedMass):
        return mass
    return CombinedMass(
        m_support=mass.m_support,
        m_refute=mass.m_refute,
        m_theta=mass.m_theta,
    )


def sequential_combine(
    masses: Sequence[MassLike],
) -> tuple[CombinedMass | None, list[FoldStep], bool]:
    """按输入顺序把一串 BPA 依次两两融合。

    第一条证据直接作为初始累计 BPA，**不调用** :func:`combine_two_masses`，
    因此只有一条证据时不会产生任何步骤。

    遇到 :class:`~rag_ds.ds.combination.TotalConflictError` 时立即停止：
    该步按约定记为 ``conflict=1.0``、``denominator=0.0``、
    ``result_mass=None``，后续证据不再处理。其他异常一律向上抛出，
    不会被吞掉。

    Args:
        masses: 至少含一条 BPA 的序列，顺序即融合顺序。

    Returns:
        ``(最终 BPA, 步骤列表, 是否完全冲突)``。完全冲突时最终 BPA 为
        ``None``。
    """
    accumulated = to_combined_mass(masses[0])
    steps: list[FoldStep] = []

    for index, incoming in enumerate(masses[1:], start=1):
        try:
            result = combine_two_masses(accumulated, incoming)
        except TotalConflictError:
            # 只捕获完全冲突；其他错误（含数值校验失败）继续向上抛。
            steps.append(
                FoldStep(
                    index=index,
                    conflict=1.0,
                    denominator=0.0,
                    result_mass=None,
                    is_total_conflict=True,
                )
            )
            return None, steps, True

        steps.append(
            FoldStep(
                index=index,
                conflict=result.conflict,
                denominator=result.normalization_denominator,
                result_mass=result.mass,
                is_total_conflict=False,
            )
        )
        accumulated = result.mass

    return accumulated, steps, False


def accumulate_conflict(conflicts: Iterable[float]) -> float:
    """把各步单次冲突量累计成一个总冲突指标。

    公式::

        K_total = 1 - (1 - K_1)(1 - K_2) ... (1 - K_n)

    这**不是** K_i 的平均值：它衡量的是「至少在某一步发生过冲突」的
    累计程度。没有任何步骤时（只有一条证据）结果为 0；任意一步 K_i = 1
    时乘积为 0，结果为 1。

    Args:
        conflicts: 各步的单次冲突量，取值均在 ``[0, 1]``。

    Returns:
        位于 ``[0, 1]`` 的累计冲突量。
    """
    non_conflict_product = 1.0
    for conflict in conflicts:
        non_conflict_product *= 1.0 - conflict
    return clamp_unit(1.0 - non_conflict_product, "累计冲突量", _FLOAT_SLACK)
