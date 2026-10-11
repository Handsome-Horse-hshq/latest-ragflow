"""二分类检测指标：把某一类当正类，评估一个连续分数的判别能力。

用于总规划 §15 的实验 2（证据不足识别，正类 ``insufficient``）与实验 3
（文档冲突识别，正类 ``conflicting``）。

只在正负两类都出现时才计算 AUROC / AUPRC —— 单一类别下这两个量没有定义，
本模块返回 ``None`` 而不是编一个 0.5 出来。
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from rag_ds.metrics.significance import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    _resample_indices,
)

__all__ = [
    "AurocComparison",
    "DetectionReport",
    "detection_report",
    "paired_auroc_bootstrap",
]


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


class AurocComparison(BaseModel):
    """参照分数与另一个分数的 AUROC 之差及其配对自助置信区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_label: str
    reference: str
    rival: str
    auroc_reference: float = Field(ge=0.0, le=1.0)
    auroc_rival: float = Field(ge=0.0, le=1.0)
    #: ``auroc_reference - auroc_rival``。
    observed_delta: float = Field(ge=-1.0, le=1.0)
    ci_low: float = Field(ge=-1.0, le=1.0)
    ci_high: float = Field(ge=-1.0, le=1.0)
    #: 双侧自助 p 值，**近似值**；结论应以置信区间为准。
    p_value: float = Field(ge=0.0, le=1.0)
    #: 实际用上的重采样次数：抽到单一类别的那几次 AUROC 没有定义，被跳过。
    n_resamples: int = Field(ge=1)
    seed: int
    sample_count: int = Field(ge=1)

    @property
    def is_significant(self) -> bool:
        """95% 置信区间是否不跨 0。"""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_auroc_bootstrap(
    is_positive: Sequence[bool],
    reference_scores: Sequence[float],
    rival_scores: Mapping[str, Sequence[float]],
    reference: str,
    positive_label: str,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[AurocComparison]:
    """把参照分数与多个对手分数逐一做配对自助 AUROC 比较。

    与 :func:`~rag_ds.metrics.significance.paired_bootstrap` 同一套抽样：
    同一种子预先抽好下标，所有分数共用 —— 它们是同一批 claim 上的不同分数，
    配对才能消掉「这批 claim 有多难分」这个共同因素。参照分数每次重采样只算
    一次 AUROC。

    Args:
        is_positive: 每条是否属于正类。
        reference_scores: 参照分数（例如 ``K_doc``）。
        rival_scores: ``对手名 -> 分数``。
        reference: 参照分数的名称。
        positive_label: 正类名称，只用于记录。
        n_resamples: 重采样次数。
        seed: 随机种子。

    Returns:
        每个对手一条 :class:`AurocComparison`，顺序与 ``rival_scores`` 相同。

    Raises:
        ValueError: 序列为空、长度不一致，或全体数据只有一个类别。
    """
    labels = np.asarray(is_positive, dtype=bool)
    size = len(labels)
    if size == 0:
        raise ValueError("不能在空序列上做自助重采样")
    if labels.all() or not labels.any():
        raise ValueError("正负两类都出现时 AUROC 才有定义")
    reference_array = np.asarray(reference_scores, dtype=float)
    rivals = {name: np.asarray(s, dtype=float) for name, s in rival_scores.items()}
    for name, scores in [(reference, reference_array), *rivals.items()]:
        if len(scores) != size:
            raise ValueError(f"{name} 的长度与标签不同：{len(scores)} vs {size}")

    deltas: dict[str, list[float]] = {name: [] for name in rivals}
    for idx in _resample_indices(random.Random(seed), size, n_resamples):
        chosen = labels[idx]
        if chosen.all() or not chosen.any():
            continue
        reference_auroc = roc_auc_score(chosen, reference_array[idx])
        for name, scores in rivals.items():
            deltas[name].append(reference_auroc - roc_auc_score(chosen, scores[idx]))

    observed_reference = float(roc_auc_score(labels, reference_array))
    comparisons: list[AurocComparison] = []
    for name, scores in rivals.items():
        values = sorted(deltas[name])
        below = sum(1 for d in values if d <= 0.0)
        above = sum(1 for d in values if d >= 0.0)
        observed_rival = float(roc_auc_score(labels, scores))
        comparisons.append(
            AurocComparison(
                positive_label=positive_label,
                reference=reference,
                rival=name,
                auroc_reference=observed_reference,
                auroc_rival=observed_rival,
                observed_delta=observed_reference - observed_rival,
                ci_low=values[int(0.025 * len(values))],
                ci_high=values[min(int(0.975 * len(values)), len(values) - 1)],
                p_value=min(1.0, 2.0 * min(below, above) / len(values)),
                n_resamples=len(values),
                seed=seed,
                sample_count=size,
            )
        )
    return comparisons
