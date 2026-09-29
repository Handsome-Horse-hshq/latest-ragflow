"""在验证集上为每个 baseline 单独搜索判定阈值。

为什么必须有这一步
------------------
D-S 的 ``theta`` / ``K_doc`` 阈值是在验证集上搜出来的；如果 baseline 还用着
``decision_threshold = 0.5`` 这个调试默认值，两边就不是在同一个条件下比较 ——
**这不是 baseline 弱，是 baseline 没调参**，把这种对比写进论文会被直接质疑。

实测中这个差别是决定性的：关系概率经校准后偏软，最高分几乎都够不到 0.5，
于是 ``weighted_average`` 与 ``single_evaluator`` 把测试集 80 条**全部**判成
``insufficient``，Macro-F1 退化成 0.1。给它们同样的调参机会之后，比较才有意义。

平台取中位数
------------
Macro-F1 常常在一整段阈值上完全持平：阈值低到一定程度后，「最高分不够高」
这条规则就不再生效，判定完全由平局与 unknown 最高两条规则决定。这种情况下取
平台边缘的值很脆弱（再动一点就掉出平台），因此统一取**平台中位数**。

每个方法单独搜
--------------
三个 baseline 的分数量纲不同：``majority_vote`` 的分数是投票比例，
``weighted_average`` 与 ``single_evaluator`` 是概率加权平均。用同一个阈值
等于给其中一方设了不合适的刻度，因此这里**每个方法各搜各的**，每个方法都拿到
它在验证集上的最优配置。

与 :mod:`~rag_ds.tuning.threshold_search` 一样，只允许在验证集上搜索。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.baselines.conflict_aware import decide_conflict_aware_state
from rag_ds.baselines.models import BaselineMethod, BaselineThresholds
from rag_ds.baselines.runner import run_baselines
from rag_ds.metrics.classification import classification_report
from rag_ds.schemas import RAGSample, RelationPrediction
from rag_ds.tuning.threshold_search import SplitName

__all__ = [
    "BaselineThresholdCandidate",
    "BaselineThresholdSearchResult",
    "DEFAULT_CONFLICT_THRESHOLDS",
    "DEFAULT_DECISION_THRESHOLDS",
    "search_baseline_thresholds",
]

#: 默认候选：0.05 步长扫完 [0.2, 0.9]。
DEFAULT_DECISION_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.20 + 0.05 * step, 2) for step in range(15)
)

#: conflict_aware 的 conflict_threshold 候选。下限到 0.001：实测最优值持续顶在
#: 网格下界，需要把「规则尽量放宽」的极限行为纳入搜索；上限 0.45 ——
#: min(s, r) 最大也只可能到 0.5，更高的阈值永远不会触发。
DEFAULT_CONFLICT_THRESHOLDS: tuple[float, ...] = (0.001, 0.005) + tuple(
    round(0.01 + 0.01 * step, 2) for step in range(4)
) + tuple(round(0.05 + 0.05 * step, 2) for step in range(9))


class BaselineThresholdCandidate(BaseModel):
    """一个候选阈值在某个方法上的表现。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: BaselineMethod
    decision_threshold: float = Field(ge=0.0, le=1.0)
    #: 仅 conflict_aware 方法有值：本次候选使用的冲突分数下限。
    conflict_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    #: 预测落在各标签上的条数，用来识别「全判成同一类」的退化解。
    label_counts: dict[str, int]

    @property
    def is_degenerate(self) -> bool:
        """所有 claim 都被判成同一类时为 ``True``。"""
        return len([c for c in self.label_counts.values() if c > 0]) <= 1


class BaselineThresholdSearchResult(BaseModel):
    """一次 baseline 阈值搜索的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 搜索所用的数据划分，必须是 ``validation``。
    split: SplitName
    #: 每个方法各自的最优候选。
    best: dict[str, BaselineThresholdCandidate]
    #: 全部候选，按方法、阈值升序。
    candidates: tuple[BaselineThresholdCandidate, ...]
    claim_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_split_is_validation(self) -> BaselineThresholdSearchResult:
        """搜索结果只允许来自验证集。"""
        if self.split is not SplitName.VALIDATION:
            raise ValueError(
                f"baseline 阈值只能在验证集上搜索，收到 split={self.split.value!r}"
            )
        return self


def search_baseline_thresholds(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    split: SplitName,
    single_evaluator: str,
    *,
    decision_thresholds: Sequence[float] = DEFAULT_DECISION_THRESHOLDS,
    conflict_thresholds: Sequence[float] = DEFAULT_CONFLICT_THRESHOLDS,
    tie_tolerance: float = 1e-6,
    conflict_margin: float = 0.1,
) -> BaselineThresholdSearchResult:
    """为每个 baseline 在验证集上搜索使 Macro-F1 最大的判定阈值。

    前三个方法只搜 ``decision_threshold``；``conflict_aware`` 额外搜
    ``conflict_threshold``（二维网格），``conflict_margin`` 固定不参与
    搜索。

    Args:
        samples: 验证集样本，每条须带 ``gold_state``。
        predictions: 验证集关系预测。
        split: 数据划分名称；**必须**为 :attr:`SplitName.VALIDATION`。
        single_evaluator: ``single_evaluator`` baseline 使用的评估器名。
        decision_thresholds: 判定阈值候选。
        conflict_thresholds: conflict_aware 的冲突分数下限候选。
        tie_tolerance: 平局容差，不参与搜索。
        conflict_margin: 冲突平衡度，固定值，不参与搜索。

    Returns:
        :class:`BaselineThresholdSearchResult`。

    Raises:
        ValueError: ``split`` 不是 validation、候选为空，或样本缺少 ``gold_state``。
    """
    if split is not SplitName.VALIDATION:
        raise ValueError(
            f"baseline 阈值只能在验证集上搜索，收到 split={split.value!r}"
        )
    if not decision_thresholds:
        raise ValueError("decision_thresholds 不能为空")
    if not conflict_thresholds:
        raise ValueError("conflict_thresholds 不能为空")
    missing = [s.sample_id for s in samples if s.gold_state is None]
    if missing:
        raise ValueError(
            f"以下样本缺少 gold_state，无法搜索阈值：{sorted(missing)[:5]}"
        )

    gold = {
        (s.sample_id, claim.claim_id): s.gold_state.value  # type: ignore[union-attr]
        for s in samples
        for claim in s.claims
    }

    candidates: list[BaselineThresholdCandidate] = []
    conflict_aware_rows: list = []
    for threshold in sorted(decision_thresholds):
        results = run_baselines(
            samples,
            predictions,
            BaselineThresholds(
                decision_threshold=threshold, tie_tolerance=tie_tolerance
            ),
            single_evaluator,
        )
        by_method: dict[BaselineMethod, list] = {}
        for item in results:
            by_method.setdefault(item.method, []).append(item)

        for method, items in by_method.items():
            if method is BaselineMethod.CONFLICT_AWARE:
                # conflict_aware 的分数与 decision_threshold 无关，只需记一次；
                # 它的候选由下面的二维网格单独生成。
                if not conflict_aware_rows:
                    conflict_aware_rows = list(items)
                continue
            truth = [gold[(i.sample_id, i.claim_id)] for i in items]
            predicted = [i.predicted_state.value for i in items]
            report = classification_report(truth, predicted, method=method.value)
            counts: dict[str, int] = {}
            for label in predicted:
                counts[label] = counts.get(label, 0) + 1
            candidates.append(
                BaselineThresholdCandidate(
                    method=method,
                    decision_threshold=threshold,
                    macro_f1=report.macro_f1,
                    accuracy=report.accuracy,
                    label_counts=counts,
                )
            )

    # conflict_aware 的二维网格：分数只算一次，判定规则对每个
    # (decision_threshold, conflict_threshold) 组合重新应用。
    for decision_threshold in sorted(decision_thresholds):
        for conflict_threshold in sorted(conflict_thresholds):
            thresholds = BaselineThresholds(
                decision_threshold=decision_threshold,
                tie_tolerance=tie_tolerance,
                conflict_threshold=conflict_threshold,
                conflict_margin=conflict_margin,
            )
            predicted = [
                decide_conflict_aware_state(
                    item.score_support, item.score_refute, item.score_unknown,
                    thresholds,
                )[0].value
                for item in conflict_aware_rows
            ]
            truth = [
                gold[(i.sample_id, i.claim_id)] for i in conflict_aware_rows
            ]
            report = classification_report(
                truth, predicted, method=BaselineMethod.CONFLICT_AWARE.value
            )
            counts = {}
            for label in predicted:
                counts[label] = counts.get(label, 0) + 1
            candidates.append(
                BaselineThresholdCandidate(
                    method=BaselineMethod.CONFLICT_AWARE,
                    decision_threshold=decision_threshold,
                    conflict_threshold=conflict_threshold,
                    macro_f1=report.macro_f1,
                    accuracy=report.accuracy,
                    label_counts=counts,
                )
            )

    best: dict[str, BaselineThresholdCandidate] = {}
    for method_name in {c.method.value for c in candidates}:
        same_method = [c for c in candidates if c.method.value == method_name]
        top = max(c.macro_f1 for c in same_method)
        # Macro-F1 常常在一整段阈值上完全持平（低阈值时 below_threshold 规则
        # 不再生效）。取平台**中位数**而不是边缘：边缘值再动一点就会掉出平台，
        # 中位数对数据扰动更稳健，且取法唯一、可复现。
        plateau = sorted(
            (c for c in same_method if c.macro_f1 == top),
            key=lambda c: (c.decision_threshold, c.conflict_threshold or 0.0),
        )
        best[method_name] = plateau[(len(plateau) - 1) // 2]

    claim_count = sum(len(s.claims) for s in samples)
    return BaselineThresholdSearchResult(
        split=split,
        best=best,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda c: (
                    c.method.value,
                    c.decision_threshold,
                    c.conflict_threshold or 0.0,
                ),
            )
        ),
        claim_count=claim_count,
    )
