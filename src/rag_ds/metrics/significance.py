"""配对自助重采样：给方法差异配上置信区间。

为什么必须有
------------
本项目的测试集只有 80 条 claim。实测 D-S 与最好的 baseline 相差
0.08 Macro-F1，看上去不小，但 95% 置信区间是 ``[-0.03, +0.19]`` —— **包含 0**。
只报一个点估计、不给区间，等于把「可能只是噪声」说成了「方法更好」。

三个必须讲清楚的设计点
----------------------
1. **配对重采样。** 所有方法都在**同一批 claim** 上评估，因此每次重采样
   抽出的下标对全部方法共用。这样就把「这批 claim 本身有多难」这个共同
   因素消掉了，比两边各自独立重采样的检验力高得多。

2. **固定 Macro-F1 的标签集。** :func:`~rag_ds.metrics.classification.classification_report`
   默认只在「金标准中实际出现过的类」上平均。重采样时若沿用这个默认，
   某次恰好抽不到某一类，平均的分母就变了，估计会有偏。因此这里**一次性
   固定标签集**，每次重采样都用同一组类。

3. **p 值是自助近似，不是精确检验。** 报告时应以置信区间为主：
   区间是否跨过 0 才是结论，p 值只作参考。

本模块只做统计，不含任何 D-S 逻辑。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import f1_score

from rag_ds.metrics.classification import GOLD_LABELS

__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "BootstrapComparison",
    "BootstrapInterval",
    "MetricName",
    "bootstrap_interval",
    "paired_bootstrap",
]

#: 默认重采样次数。
DEFAULT_RESAMPLES: int = 5000
#: 默认随机种子，保证结果可复现。
DEFAULT_SEED: int = 42

MetricName = Literal["macro_f1", "accuracy"]


class BootstrapInterval(BaseModel):
    """单个方法在某个指标上的自助置信区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    metric: MetricName
    observed: float = Field(ge=0.0, le=1.0)
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)
    n_resamples: int = Field(ge=1)
    seed: int
    sample_count: int = Field(ge=1)


class BootstrapComparison(BaseModel):
    """两个方法之差的自助置信区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_a: str
    method_b: str
    metric: MetricName
    observed_a: float = Field(ge=0.0, le=1.0)
    observed_b: float = Field(ge=0.0, le=1.0)
    #: ``observed_a - observed_b``。
    observed_delta: float = Field(ge=-1.0, le=1.0)
    ci_low: float = Field(ge=-1.0, le=1.0)
    ci_high: float = Field(ge=-1.0, le=1.0)
    #: 双侧自助 p 值，**近似值**；结论应以置信区间为准。
    p_value: float = Field(ge=0.0, le=1.0)
    n_resamples: int = Field(ge=1)
    seed: int
    sample_count: int = Field(ge=1)

    @property
    def is_significant(self) -> bool:
        """95% 置信区间是否不跨 0。"""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: MetricName,
    labels: Sequence[str],
) -> float:
    """按固定标签集计算指标。"""
    if metric == "accuracy":
        return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    return float(
        f1_score(
            list(y_true),
            list(y_pred),
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


def _check_inputs(y_true: Sequence[str], *predictions: Sequence[str]) -> None:
    """长度一致且非空。"""
    if not y_true:
        raise ValueError("不能在空序列上做自助重采样")
    for predicted in predictions:
        if len(predicted) != len(y_true):
            raise ValueError(
                f"预测序列长度与金标准不同：{len(predicted)} vs {len(y_true)}"
            )


def _resample_indices(
    rng: random.Random, size: int, n_resamples: int
) -> list[list[int]]:
    """预先抽好全部重采样下标，供多个方法共用（配对的关键）。"""
    return [
        [rng.randrange(size) for _ in range(size)] for _ in range(n_resamples)
    ]


def bootstrap_interval(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    method: str,
    *,
    metric: MetricName = "macro_f1",
    labels: Sequence[str] = GOLD_LABELS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> BootstrapInterval:
    """给单个方法的指标算 95% 自助置信区间。

    Args:
        y_true: 金标准标签。
        y_pred: 预测标签。
        method: 方法名。
        metric: ``macro_f1`` 或 ``accuracy``。
        labels: 参与 Macro-F1 平均的**固定**标签集。
        n_resamples: 重采样次数。
        seed: 随机种子。

    Returns:
        :class:`BootstrapInterval`。

    Raises:
        ValueError: 序列为空或长度不一致。
    """
    _check_inputs(y_true, y_pred)
    rng = random.Random(seed)
    size = len(y_true)
    scores = [
        _score([y_true[i] for i in idx], [y_pred[i] for i in idx], metric, labels)
        for idx in _resample_indices(rng, size, n_resamples)
    ]
    scores.sort()
    return BootstrapInterval(
        method=method,
        metric=metric,
        observed=_score(y_true, y_pred, metric, labels),
        ci_low=scores[int(0.025 * len(scores))],
        ci_high=scores[min(int(0.975 * len(scores)), len(scores) - 1)],
        n_resamples=n_resamples,
        seed=seed,
        sample_count=size,
    )


def paired_bootstrap(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
    method_a: str,
    method_b: str,
    *,
    metric: MetricName = "macro_f1",
    labels: Sequence[str] = GOLD_LABELS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> BootstrapComparison:
    """配对自助重采样，给 ``a - b`` 的差异配上置信区间与 p 值。

    两个方法在每次重采样里**共用同一组下标** —— 它们本就在同一批 claim 上
    评估，配对能消掉「这批 claim 有多难」这个共同因素。

    Args:
        y_true: 金标准标签。
        y_pred_a: 方法 A 的预测。
        y_pred_b: 方法 B 的预测。
        method_a: 方法 A 名称。
        method_b: 方法 B 名称。
        metric: ``macro_f1`` 或 ``accuracy``。
        labels: 参与 Macro-F1 平均的**固定**标签集。
        n_resamples: 重采样次数。
        seed: 随机种子。

    Returns:
        :class:`BootstrapComparison`。

    Raises:
        ValueError: 序列为空或长度不一致。
    """
    _check_inputs(y_true, y_pred_a, y_pred_b)
    rng = random.Random(seed)
    size = len(y_true)

    deltas: list[float] = []
    for idx in _resample_indices(rng, size, n_resamples):
        truth = [y_true[i] for i in idx]
        deltas.append(
            _score(truth, [y_pred_a[i] for i in idx], metric, labels)
            - _score(truth, [y_pred_b[i] for i in idx], metric, labels)
        )
    deltas.sort()

    below = sum(1 for d in deltas if d <= 0.0)
    above = sum(1 for d in deltas if d >= 0.0)
    p_value = min(1.0, 2.0 * min(below, above) / len(deltas))

    observed_a = _score(y_true, y_pred_a, metric, labels)
    observed_b = _score(y_true, y_pred_b, metric, labels)
    return BootstrapComparison(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        observed_a=observed_a,
        observed_b=observed_b,
        observed_delta=observed_a - observed_b,
        ci_low=deltas[int(0.025 * len(deltas))],
        ci_high=deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)],
        p_value=p_value,
        n_resamples=n_resamples,
        seed=seed,
        sample_count=size,
    )
