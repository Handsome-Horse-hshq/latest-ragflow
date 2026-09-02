"""四分类评估指标。

标签口径
--------
金标准 ``gold_state`` 有四类：``supported`` / ``refuted`` /
``insufficient`` / ``conflicting``。但两侧方法的**输出空间并不相同**：

* D-S 诊断可以给出全部四类，也可能给出 ``None``（证据充分却分不出方向）；
* 三个 baseline **在结构上无法输出** ``conflicting``。

这带来一个必须显式定义、不能糊弄过去的问题：baseline 的 ``conflicting``
一行在混淆矩阵里必然全部落到别的列。本模块的处理是：

1. **混淆矩阵**使用完整标签集（四类 + ``undetermined``），行为真实标签、
   列为预测标签，方阵，任何一格都不隐藏；
2. **Macro-F1 默认只在「金标准中实际出现过的类」上平均**。这样
   ``undetermined`` 不会作为一个凭空多出来的类拉低所有方法的分数，而
   baseline 在 ``conflicting`` 上的 0 分**照常计入** —— 那是它真实的能力
   缺口，不该被抹掉。

论文里必须写明这一口径，否则「baseline 的 Macro-F1 为什么这么低」会成为
审稿人的第一个问题。若要给出「只比三类」的补充视角，请显式传入
``macro_labels``，并在报告中同时给出两套数字。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from rag_ds.schemas import EvidenceState

__all__ = [
    "GOLD_LABELS",
    "UNDETERMINED_LABEL",
    "ClassMetrics",
    "ClassificationReport",
    "classification_report",
    "default_label_universe",
]

#: 金标准的四个类别，顺序固定以保证混淆矩阵可比。
GOLD_LABELS: tuple[str, ...] = (
    EvidenceState.SUPPORTED.value,
    EvidenceState.REFUTED.value,
    EvidenceState.INSUFFICIENT.value,
    EvidenceState.CONFLICTING.value,
)

#: D-S 侧「证据充分却分不出方向」时使用的预测标签。
UNDETERMINED_LABEL = "undetermined"


def default_label_universe() -> tuple[str, ...]:
    """混淆矩阵使用的完整标签集：四个金标准类 + ``undetermined``。"""
    return (*GOLD_LABELS, UNDETERMINED_LABEL)


class ClassMetrics(BaseModel):
    """单个类别的 P / R / F1 与样本数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    #: 该类在**金标准**中的样本数。
    support: int = Field(ge=0)


class ClassificationReport(BaseModel):
    """一次分类评估的完整报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 方法名称，例如 ``"ds"``、``"weighted_average"``。
    method: str
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    #: 参与 Macro-F1 平均的类别。
    macro_labels: tuple[str, ...]
    #: 每个类别的指标，顺序与 ``labels`` 一致。
    per_class: tuple[ClassMetrics, ...]
    #: 混淆矩阵的标签集（行=真实，列=预测）。
    labels: tuple[str, ...]
    #: ``matrix[i][j]`` = 真实为 labels[i] 且预测为 labels[j] 的条数。
    matrix: tuple[tuple[int, ...], ...]
    sample_count: int = Field(ge=0)

    def class_metrics(self, label: str) -> ClassMetrics | None:
        """按标签取出单类指标；不存在返回 ``None``。"""
        return next((m for m in self.per_class if m.label == label), None)


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    method: str,
    labels: Sequence[str] | None = None,
    macro_labels: Sequence[str] | None = None,
) -> ClassificationReport:
    """计算 Accuracy、Macro-F1、各类 P/R/F1 与混淆矩阵。

    Args:
        y_true: 金标准标签序列。
        y_pred: 预测标签序列，长度须与 ``y_true`` 相同。
        method: 方法名称，写进报告。
        labels: 混淆矩阵的标签集；默认为四个金标准类 + ``undetermined``。
        macro_labels: 参与 Macro-F1 平均的类别；默认为**金标准中实际出现过
            的类**（见模块文档字符串对该口径的说明）。

    Returns:
        :class:`ClassificationReport`。

    Raises:
        ValueError: 两个序列长度不同、为空，或出现不在 ``labels`` 中的标签。
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true 与 y_pred 长度不同：{len(y_true)} vs {len(y_pred)}"
        )
    if not y_true:
        raise ValueError("不能在空序列上计算指标")

    label_list = (
        list(labels) if labels is not None else list(default_label_universe())
    )
    unknown = sorted((set(y_true) | set(y_pred)) - set(label_list))
    if unknown:
        raise ValueError(f"出现不在标签集中的标签：{unknown}")

    if macro_labels is None:
        # 只在金标准中实际出现过的类上平均。
        macro_list = [label for label in label_list if label in set(y_true)]
    else:
        macro_list = list(macro_labels)
    if not macro_list:
        raise ValueError("macro_labels 为空，无法计算 Macro-F1")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, zero_division=0
    )
    per_class = tuple(
        ClassMetrics(
            label=label,
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(label_list)
    )

    by_label = {metrics.label: metrics for metrics in per_class}
    missing = [label for label in macro_list if label not in by_label]
    if missing:
        raise ValueError(f"macro_labels 含不在标签集中的类：{missing}")

    macro_f1 = float(np.mean([by_label[label].f1 for label in macro_list]))
    accuracy = float(
        np.mean([true == pred for true, pred in zip(y_true, y_pred)])
    )
    matrix = _sk_confusion_matrix(y_true, y_pred, labels=label_list)

    return ClassificationReport(
        method=method,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_labels=tuple(macro_list),
        per_class=per_class,
        labels=tuple(label_list),
        matrix=tuple(tuple(int(cell) for cell in row) for row in matrix),
        sample_count=len(y_true),
    )
