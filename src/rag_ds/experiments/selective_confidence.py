"""选择性回答用的置信度：衡量「对所预测的那一类有多确信」。

为什么要单独定义
----------------
最初的风险分数对所有方法一律取 ``max(insufficiency_score, conflict_score)``。
这对 baseline 勉强说得通（那两个量确实是它们的"说不清"信号），但**对 D-S 是
错的**：``m_theta`` 与 ``K_doc`` 不是不确定性，它们是四类里**两类的证据本身**。

* ``m_theta`` 高 → D-S 有把握判 ``insufficient``；
* ``K_doc`` 高 → D-S 有把握判 ``conflicting``。

用它们当风险，等于让 D-S **优先弃答自己最有把握的预测**。实测 616 条里 170 条
``insufficient`` 预测的风险分均值高达 0.94，被整体顶到弃答队列最前 —— 风险排序
实际在排「是不是判了 insufficient」，而不是「有多不确定」。

本模块因此按**预测类别**取该类自己的证据作为置信度，两侧对称：

====================  ==============================  ==============================
预测类别              D-S 的置信度                    baseline 的置信度
====================  ==============================  ==============================
``supported``         ``m_support``                   ``score_support``
``refuted``           ``m_refute``                    ``score_refute``
``insufficient``      ``m_theta``                     ``score_unknown``
``conflicting``       ``K_doc``                       ``min(score_support, score_refute)``
``undetermined``      0（没有认领任何一类）           不适用
====================  ==============================  ==============================

``conflicting`` 一行对 baseline 取 ``min(s, r)``，是因为 ``conflict_aware`` 的
冲突规则正是对这个量设阈值的 —— 两个方向都必须真的有分量才算冲突。

风险 = ``1 - 置信度``。

.. note::
    换用这个定义会改变**所要回答的问题**：旧定义问的是「证据不足/冲突信号能否
    预测错误」，新定义问的是「方法对自己的判断有多确信」。后者才是选择性回答
    文献里的标准口径（max-softmax 置信度的直接类比）。两套数字都应报告。
"""

from __future__ import annotations

from rag_ds.metrics.classification import UNDETERMINED_LABEL
from rag_ds.schemas import EvidenceState

__all__ = [
    "baseline_confidence",
    "ds_confidence",
]


def ds_confidence(
    predicted_label: str,
    m_support: float | None,
    m_refute: float | None,
    m_theta: float | None,
    k_doc: float,
) -> float:
    """D-S 对其所预测类别的置信度。

    Args:
        predicted_label: 四类之一或 ``undetermined``。
        m_support: 融合后支持质量；完全冲突时为 ``None``。
        m_refute: 融合后反驳质量。
        m_theta: 融合后未分配质量。
        k_doc: 文档冲突量。

    Returns:
        ``[0, 1]`` 内的置信度。完全冲突（三个质量均为 ``None``）时，
        ``conflicting`` 仍可用 ``k_doc`` 表达置信，其余类别记为 0。

    Raises:
        ValueError: ``predicted_label`` 不是已知取值。
    """
    if predicted_label == EvidenceState.SUPPORTED.value:
        return float(m_support or 0.0)
    if predicted_label == EvidenceState.REFUTED.value:
        return float(m_refute or 0.0)
    if predicted_label == EvidenceState.INSUFFICIENT.value:
        return float(m_theta or 0.0)
    if predicted_label == EvidenceState.CONFLICTING.value:
        return float(k_doc)
    if predicted_label == UNDETERMINED_LABEL:
        # 没有认领任何一类，置信度为 0：这类预测应当最先被弃答。
        return 0.0
    raise ValueError(f"未知的预测标签：{predicted_label!r}")


def baseline_confidence(
    predicted_label: str,
    score_support: float,
    score_refute: float,
    score_unknown: float,
) -> float:
    """baseline 对其所预测类别的置信度。

    Args:
        predicted_label: 四类之一。
        score_support: 加权后的支持分。
        score_refute: 加权后的反驳分。
        score_unknown: 加权后的未知分。

    Returns:
        ``[0, 1]`` 内的置信度。

    Raises:
        ValueError: ``predicted_label`` 不是已知取值。
    """
    if predicted_label == EvidenceState.SUPPORTED.value:
        return float(score_support)
    if predicted_label == EvidenceState.REFUTED.value:
        return float(score_refute)
    if predicted_label == EvidenceState.INSUFFICIENT.value:
        return float(score_unknown)
    if predicted_label == EvidenceState.CONFLICTING.value:
        # conflict_aware 的冲突规则就是对 min(s, r) 设阈值：两个方向都要有分量。
        return float(min(score_support, score_refute))
    if predicted_label == UNDETERMINED_LABEL:
        return 0.0
    raise ValueError(f"未知的预测标签：{predicted_label!r}")
