"""``ds`` 子包内部共用的数值工具。

这里只处理 IEEE-754 舍入噪声，不做任何静默归一化：真正越界的数值一律
报错，绝不悄悄拉回合法区间。
"""

from __future__ import annotations

__all__ = ["clamp_unit"]


def clamp_unit(value: float, label: str, slack: float) -> float:
    """把紧邻 0 或 1 的浮点噪声夹回端点，其余越界值一律报错。

    ``slack`` 由调用方显式给出，因为不同环节可容忍的噪声量级不同：
    可靠性折扣只做乘法与一次减法，噪声在 1e-16 量级；Dempster 归一化
    要除以可能很小的分母，规格要求更严的 1e-12。

    Args:
        value: 待检查的数值。
        label: 出错信息中使用的字段名。
        slack: 允许被夹回端点的最大偏差。

    Returns:
        位于 ``[0, 1]`` 内的数值。

    Raises:
        ValueError: 偏离区间超过 ``slack``，说明是真的算错了，而不是
            浮点噪声。
    """
    if -slack <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + slack:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} 超出 [0, 1]：{value!r}")
    return value
