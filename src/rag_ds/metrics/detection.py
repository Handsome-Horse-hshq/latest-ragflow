"""二分类检测指标：把某一类当正类，评估一个连续分数的判别能力。

用于总规划 §15 的实验 2（证据不足识别，正类 ``insufficient``）与实验 3
（文档冲突识别，正类 ``conflicting``）。

只在正负两类都出现时才计算 AUROC / AUPRC —— 单一类别下这两个量没有定义，
本模块返回 ``None`` 而不是编一个 0.5 出来。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["DetectionReport", "detection_report"]


class DetectionReport(BaseModel):
    """一次二分类检测评估的结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 分数名称，例如 ``"m_theta"``、``"k_doc"``。
    score_name: str
    #: 正类名称，例如 ``"insufficient"``。
    positive_label: str
    #: 正负两类都存在时才有值；否则 AUROC / AUPRC 没有定义。
    auroc: float | None = None
    auprc: float | None = None
    #: 在所有候选阈值中取到的最好 F1，及对应阈值。
    best_f1: float = Field(ge=0.0, le=1.0)
    best_threshold: float
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)


def detection_report(
    scores: Sequence[float],
    is_positive: Sequence[bool],
    score_name: str,
    positive_label: str,
) -> DetectionReport:
    """把一个连续分数当作检测器来评估。

    判定规则为 ``score >= threshold`` 判正，与二维门控保持一致（含等号）。
    候选阈值取所有出现过的分数值。

    Args:
        scores: 连续分数序列。
        is_positive: 与之等长的布尔序列，``True`` 表示该条属于正类。
        score_name: 分数名称。
        positive_label: 正类名称。

    Returns:
        :class:`DetectionReport`。

    Raises:
        ValueError: 两个序列长度不同或为空。
    """
    if len(scores) != len(is_positive):
        raise ValueError(
            f"scores 与 is_positive 长度不同：{len(scores)} vs {len(is_positive)}"
        )
    if not scores:
        raise ValueError("不能在空序列上计算检测指标")

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(is_positive, dtype=bool)
    positive_count = int(label_array.sum())
    negative_count = int(len(label_array) - positive_count)

    auroc: float | None = None
    auprc: float | None = None
    if positive_count > 0 and negative_count > 0:
        auroc = float(roc_auc_score(label_array, score_array))
        auprc = float(average_precision_score(label_array, score_array))

    best_f1, best_threshold = 0.0, float(score_array.min())
    for threshold in np.unique(score_array):
        predicted = score_array >= threshold
        true_positive = int((predicted & label_array).sum())
        if true_positive == 0:
            continue
        precision = true_positive / int(predicted.sum())
        recall = true_positive / positive_count
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)

    return DetectionReport(
        score_name=score_name,
        positive_label=positive_label,
        auroc=auroc,
        auprc=auprc,
        best_f1=best_f1,
        best_threshold=best_threshold,
        positive_count=positive_count,
        negative_count=negative_count,
    )
