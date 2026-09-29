"""在验证集上搜索二维门控阈值。

.. warning::
    **只能在验证集上搜索。** 用测试集选阈值再用测试集报告结果，等于把测试
    集当训练集，得到的数字没有意义。本模块的 API 强制调用方显式说明数据
    划分（:class:`SplitName`），并在选中测试集时直接报错。

为什么可以只重跑门控
--------------------
阈值只影响**最后一步二维门控**，不影响前面的 BPA 映射、可靠性折扣、文档
融合与评估器融合。因此搜索时先把 pipeline 跑一遍、把
:class:`~rag_ds.pipeline_results.ClaimPipelineResult` 缓存下来，之后每个
网格点只需重新调用一次门控函数 —— 而不是把整条 D-S 链路重算几百遍。

这也保证了搜索与正式运行用的是**同一套门控代码**，不会出现「调参用一套、
报告用另一套」的偏差。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from itertools import product

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.diagnostics.gating import (
    diagnose_document_total_conflict,
    diagnose_evaluator_result,
    diagnose_no_evidence,
)
from rag_ds.diagnostics.models import DiagnosticResult, DiagnosticThresholds
from rag_ds.metrics.classification import (
    UNDETERMINED_LABEL,
    classification_report,
)
from rag_ds.pipeline_results import ClaimPipelineResult, PipelineStatus

__all__ = [
    "SplitName",
    "ThresholdCandidate",
    "ThresholdGrid",
    "ThresholdSearchResult",
    "predicted_label",
    "rediagnose",
    "search_thresholds",
]


class SplitName(str, Enum):
    """数据划分名称。"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def rediagnose(
    result: ClaimPipelineResult, thresholds: DiagnosticThresholds
) -> DiagnosticResult:
    """用一组新阈值，对已算好的 pipeline 结果重新做门控。

    不重算任何 D-S 数学：按 ``status`` 分派到与 pipeline 完全相同的三个
    门控函数。

    Args:
        result: 已有的逐 claim 结果。
        thresholds: 新的阈值。

    Returns:
        重新计算的 :class:`DiagnosticResult`。

    Raises:
        ValueError: 结果的 ``status`` 与其携带的中间量不一致（正常流程下
            不会发生，由 pipeline 保证）。
    """
    if result.status is PipelineStatus.NO_CONTEXTS:
        return diagnose_no_evidence(result.sample_id, result.claim_id, thresholds)

    if result.status is PipelineStatus.DOCUMENT_TOTAL_CONFLICT:
        conflicted = next(
            (r for r in result.document_results if r.is_total_conflict), None
        )
        if conflicted is None:
            raise ValueError(
                f"{result.claim_id!r} 标记为文档完全冲突，"
                "但 document_results 中没有对应记录"
            )
        return diagnose_document_total_conflict(conflicted, thresholds)

    if result.evaluator_result is None:
        raise ValueError(
            f"{result.claim_id!r} 的 status={result.status.value!r} "
            "却缺少 evaluator_result"
        )
    return diagnose_evaluator_result(result.evaluator_result, thresholds)


def predicted_label(diagnostic: DiagnosticResult) -> str:
    """把诊断结果转成用于指标计算的预测标签。

    ``primary_state`` 为 ``None``（证据充分却分不出方向）时映射为
    :data:`~rag_ds.metrics.classification.UNDETERMINED_LABEL`，而不是硬塞进
    四类中的某一类 —— 那会凭空制造一次「猜对」或「猜错」。
    """
    if diagnostic.primary_state is None:
        return UNDETERMINED_LABEL
    return diagnostic.primary_state.value


class ThresholdGrid(BaseModel):
    """四分类门控阈值的候选网格。

    ``K_eval`` 只控制 ``evaluator_disagreement`` 警报，不改变四分类标签，
    因此其阈值在这里固定而不参与以 Macro-F1 为目标的搜索。若将它作为第三
    个搜索维度，同一组分类结果会被无意义地重复五遍。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    theta_values: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    document_conflict_values: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
    #: 额外警报阈值；固定记录在候选中，但不以四分类 Macro-F1 调参。
    evaluator_conflict_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    #: 平局容差不参与搜索，固定沿用基准阈值。
    tie_tolerance: float = 1e-6

    @model_validator(mode="after")
    def _check_non_empty_and_in_range(self) -> ThresholdGrid:
        """三组候选都不能为空，且取值须在 [0, 1]。"""
        for field in ("theta_values", "document_conflict_values"):
            values = getattr(self, field)
            if not values:
                raise ValueError(f"{field} 不能为空")
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"{field} 的取值必须位于 [0, 1]：{values}")
        if not 0.0 <= self.tie_tolerance <= 1.0:
            raise ValueError("tie_tolerance 必须位于 [0, 1]")
        return self

    @classmethod
    def from_observed(
        cls,
        theta_values: Sequence[float],
        document_conflict_values: Sequence[float],
        *,
        steps: int = 5,
        evaluator_conflict_threshold: float = 0.4,
        tie_tolerance: float = 1e-6,
    ) -> ThresholdGrid:
        """按验证集上**实际观测到的** m_theta / K_doc 分位数构造网格。

        固定网格 ``(0.3, 0.4, ..., 0.7)`` 隐含了一个假设：融合后的 m_theta
        会落在 0.3 以上。这个假设只在概率比较尖锐时成立 —— 换成校准后的软
        概率、或文档数变多时，m_theta 会被 Dempster 组合压到很小，整条 theta
        轴就全部落在观测范围之外，门控**一次都不会触发**，而搜索结果看上去
        却一切正常（所有网格点 Macro-F1 完全相同）。

        本方法让网格跟着数据走：取观测值的等距分位数作为候选阈值。

        Args:
            theta_values: 验证集上观测到的 m_theta。
            document_conflict_values: 验证集上观测到的 K_doc。
            steps: 每个轴的候选点数。
            evaluator_conflict_threshold: 固定的 K_eval 告警阈值。
            tie_tolerance: 平局容差。

        Returns:
            :class:`ThresholdGrid`。

        Raises:
            ValueError: 任一序列为空，或 ``steps`` 小于 1。
        """
        if steps < 1:
            raise ValueError("steps 必须是正整数")

        def _quantile_candidates(values: Sequence[float]) -> tuple[float, ...]:
            if not values:
                raise ValueError("观测值序列不能为空")
            ordered = sorted(values)
            last = len(ordered) - 1
            picked: list[float] = []
            for index in range(steps):
                # 分位点**含两端**：取不到上沿的话，「阈值高到门控几乎不触发」
                # 这种配置就永远进不了网格，而它恰恰是有意义的一种取值
                # （消融里的 no_doc_conflict_gate 就是把阈值抬到 1.0）。
                position = 0.5 if steps == 1 else index / (steps - 1)
                raw = ordered[min(round(position * last), last)]
                value = round(min(max(raw, 0.0), 1.0), 6)
                if value not in picked:
                    picked.append(value)
            return tuple(picked)

        return cls(
            theta_values=_quantile_candidates(theta_values),
            document_conflict_values=_quantile_candidates(document_conflict_values),
            evaluator_conflict_threshold=evaluator_conflict_threshold,
            tie_tolerance=tie_tolerance,
        )

    def __len__(self) -> int:
        """网格点总数。"""
        return len(self.theta_values) * len(self.document_conflict_values)

    def candidates(self) -> Iterable[DiagnosticThresholds]:
        """按 (theta, k_doc) 的字典序枚举全部四分类阈值组合。"""
        for theta, doc in product(self.theta_values, self.document_conflict_values):
            yield DiagnosticThresholds(
                theta_threshold=theta,
                document_conflict_threshold=doc,
                evaluator_conflict_threshold=self.evaluator_conflict_threshold,
                tie_tolerance=self.tie_tolerance,
            )


class ThresholdCandidate(BaseModel):
    """一个网格点的评估结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thresholds: DiagnosticThresholds
    macro_f1: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)


class ThresholdSearchResult(BaseModel):
    """一次阈值搜索的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 搜索所用的数据划分，必须是 ``validation``。
    split: SplitName
    best: ThresholdCandidate
    #: 全部网格点，按 Macro-F1 降序、阈值升序排列（结果可复现）。
    candidates: tuple[ThresholdCandidate, ...]
    claim_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_split_is_validation(self) -> ThresholdSearchResult:
        """搜索结果只允许来自验证集。"""
        if self.split is not SplitName.VALIDATION:
            raise ValueError(
                f"阈值只能在验证集上搜索，收到 split={self.split.value!r}"
            )
        return self


def search_thresholds(
    results: Sequence[ClaimPipelineResult],
    split: SplitName,
    grid: ThresholdGrid | None = None,
) -> ThresholdSearchResult:
    """在验证集上网格搜索使 Macro-F1 最大的阈值组合。

    Args:
        results: 验证集上已算好的逐 claim pipeline 结果。每条必须带
            ``gold_state``，否则无法评分。
        split: 数据划分名称；**必须**为 :attr:`SplitName.VALIDATION`。
        grid: 候选网格；``None`` 表示使用默认网格。

    Returns:
        :class:`ThresholdSearchResult`，含最优阈值与全部网格点得分。

    Raises:
        ValueError: ``split`` 不是验证集、``results`` 为空，或存在缺少
            ``gold_state`` 的条目。

    Note:
        并列时取**阈值字典序最小**的那一组，保证结果可复现。
    """
    if split is not SplitName.VALIDATION:
        raise ValueError(
            f"阈值只能在验证集上搜索，收到 split={split.value!r}；"
            "用测试集选阈值再用测试集报告结果没有意义"
        )
    if not results:
        raise ValueError("阈值搜索收到空的验证集")

    missing = [r.claim_id for r in results if r.gold_state is None]
    if missing:
        raise ValueError(
            f"以下 claim 缺少 gold_state，无法用于阈值搜索：{sorted(missing)[:5]}"
        )

    y_true = [r.gold_state.value for r in results]  # type: ignore[union-attr]
    search_grid = grid or ThresholdGrid()

    candidates: list[ThresholdCandidate] = []
    for thresholds in search_grid.candidates():
        y_pred = [predicted_label(rediagnose(r, thresholds)) for r in results]
        report = classification_report(y_true, y_pred, method="ds")
        candidates.append(
            ThresholdCandidate(
                thresholds=thresholds,
                macro_f1=report.macro_f1,
                accuracy=report.accuracy,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda c: (
            -c.macro_f1,
            c.thresholds.theta_threshold,
            c.thresholds.document_conflict_threshold,
            c.thresholds.evaluator_conflict_threshold,
        ),
    )

    return ThresholdSearchResult(
        split=split,
        best=ordered[0],
        candidates=tuple(ordered),
        claim_count=len(results),
    )
