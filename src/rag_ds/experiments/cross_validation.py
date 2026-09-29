"""嵌套交叉验证：让每条 claim 都当一次测试点。

为什么需要
----------
单次留出划分下测试集只有 80–124 条 claim。实测 D-S 与最好的 baseline 相差
约 0.08 Macro-F1，而 95% 自助置信区间**跨过 0**（见
:mod:`rag_ds.metrics.significance`）。样本量撑不起观测到的差距，主结论就立不住。

交叉验证把全部样本都用上：每条 claim 恰好在一个外折里被预测一次，汇总起来
就有 n = 全体样本数 个评估点，置信区间随 ``sqrt(n)`` 收窄。

嵌套在哪里
----------
本方法不训练模型，需要在数据上选的只有**阈值**。所以每个外折的做法是：

1. 留出该折作为测试；
2. 其余各折合起来当**验证集**，在上面搜 D-S 门控阈值与 baseline 判定阈值；
3. 用搜出的阈值预测该折。

测试折全程不参与任何阈值选择，因此没有泄漏。每折实际使用的阈值都会记录下来，
折与折之间的阈值差异本身就是稳定性的证据。

汇总口径
--------
各折的预测**汇总后统一算一次指标**（micro-pooling），而不是把各折的 Macro-F1
再平均一次。后者在小折上会因为某一类样本过少而剧烈抖动，且无法直接喂给配对
自助检验；汇总后每条 claim 恰好一条预测，可以直接复用现有的指标与显著性工具。
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.baselines.models import BaselineThresholds
from rag_ds.baselines.runner import run_baselines
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments.comparison import DS_METHOD, MethodPrediction
from rag_ds.experiments.selective_confidence import (
    baseline_confidence,
    ds_confidence,
)
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import RAGSample, RelationPrediction
from rag_ds.tuning.baseline_threshold_search import (
    DEFAULT_DECISION_THRESHOLDS,
    search_baseline_thresholds,
)
from rag_ds.tuning.threshold_search import (
    SplitName,
    ThresholdGrid,
    predicted_label,
    search_thresholds,
)

__all__ = [
    "CrossValidationResult",
    "FoldThresholds",
    "make_stratified_folds",
    "run_nested_cross_validation",
]


class FoldThresholds(BaseModel):
    """某个外折实际使用的阈值，以及它是在多少条上选出来的。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold: int = Field(ge=0)
    test_size: int = Field(ge=1)
    inner_size: int = Field(ge=1)
    ds_thresholds: DiagnosticThresholds
    baseline_decision_threshold: float = Field(ge=0.0, le=1.0)
    #: conflict_aware 在该折内层搜出的冲突分数下限；无该方法的候选时为 None。
    baseline_conflict_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    #: 该折在内层验证集上取得的 Macro-F1，仅供观察调参是否稳定。
    inner_macro_f1: float = Field(ge=0.0, le=1.0)


class CrossValidationResult(BaseModel):
    """一次嵌套交叉验证的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_folds: int = Field(ge=2)
    seed: int
    #: 汇总后的逐条预测，每条 claim 恰好一条，可直接喂给指标与显著性工具。
    predictions: tuple[MethodPrediction, ...]
    #: 每个外折使用的阈值。
    folds: tuple[FoldThresholds, ...]
    claim_count: int = Field(ge=1)
    methods: tuple[str, ...]


def make_stratified_folds(
    samples: Sequence[RAGSample], n_folds: int, seed: int = 42
) -> list[list[int]]:
    """按 ``gold_state`` 分层切成 ``n_folds`` 折，返回每折的样本下标。

    同一类的样本先打散再轮流分配，保证各折的类别分布尽量接近 —— 四类平衡的
    数据集上若不分层，小折很容易缺类，Macro-F1 会无法计算或剧烈抖动。

    Args:
        samples: 样本，每条须带 ``gold_state``。
        n_folds: 折数。
        seed: 随机种子。

    Returns:
        ``n_folds`` 个下标列表，每个下标恰好出现一次。

    Raises:
        ValueError: 折数不合法、样本缺少 ``gold_state``，或某折为空。
    """
    if n_folds < 2:
        raise ValueError(f"n_folds 至少为 2，收到 {n_folds}")
    if len(samples) < n_folds:
        raise ValueError(
            f"样本数 {len(samples)} 少于折数 {n_folds}，无法划分"
        )
    missing = [s.sample_id for s in samples if s.gold_state is None]
    if missing:
        raise ValueError(
            f"以下样本缺少 gold_state，无法分层划分：{sorted(missing)[:5]}"
        )

    by_label: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_label.setdefault(sample.gold_state.value, []).append(index)  # type: ignore[union-attr]

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % n_folds].append(index)

    empty = [i for i, fold in enumerate(folds) if not fold]
    if empty:
        raise ValueError(f"第 {empty} 折为空，请减小 n_folds")
    return [sorted(fold) for fold in folds]


def _subset(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    indices: Sequence[int],
) -> tuple[list[RAGSample], list[RelationPrediction]]:
    """取出给定下标的样本及其对应的关系预测。"""
    chosen = [samples[i] for i in indices]
    keep = {s.sample_id for s in chosen}
    return chosen, [p for p in predictions if p.sample_id in keep]


def run_nested_cross_validation(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    single_evaluator: str,
    *,
    n_folds: int = 5,
    seed: int = 42,
    grid_steps: int = 9,
    evaluator_conflict_threshold: float = 0.4,
    baseline_decision_thresholds: Sequence[float] = DEFAULT_DECISION_THRESHOLDS,
) -> CrossValidationResult:
    """跑嵌套交叉验证，返回汇总后的逐条预测与每折阈值。

    每个外折的阈值都只在**其余各折**上选，测试折不参与选择。

    Args:
        samples: 全部样本，每条须带 ``gold_state``。
        predictions: 全部关系预测。
        single_evaluator: ``single_evaluator`` baseline 使用的评估器名。
        n_folds: 外折数。
        seed: 分折随机种子。
        grid_steps: D-S 阈值网格每个轴的候选点数（按内层观测值分位数构造）。
        evaluator_conflict_threshold: 固定的 K_eval 告警阈值。
        baseline_decision_thresholds: baseline 判定阈值候选。

    Returns:
        :class:`CrossValidationResult`。

    Raises:
        ValueError: 分折失败，或某折内层数据不足以搜索阈值。
    """
    folds = make_stratified_folds(samples, n_folds, seed)

    pooled: list[MethodPrediction] = []
    records: list[FoldThresholds] = []
    methods: list[str] = []

    for fold_index, test_indices in enumerate(folds):
        inner_indices = [
            i for position, fold in enumerate(folds) if position != fold_index
            for i in fold
        ]
        inner_samples, inner_predictions = _subset(samples, predictions, inner_indices)
        test_samples, test_predictions = _subset(samples, predictions, test_indices)

        # --- 内层：只在其余各折上选阈值 ---
        base = DiagnosticThresholds(
            evaluator_conflict_threshold=evaluator_conflict_threshold
        )
        inner_results = run_pipeline(inner_samples, inner_predictions, base)
        observed_theta = [
            r.diagnostic.m_theta
            for r in inner_results
            if r.diagnostic.m_theta is not None
        ]
        observed_k_doc = [r.diagnostic.k_doc for r in inner_results]
        if not observed_theta:
            raise ValueError(
                f"第 {fold_index} 折的内层数据没有可用的 m_theta，无法搜索阈值"
            )
        grid = ThresholdGrid.from_observed(
            observed_theta,
            observed_k_doc,
            steps=grid_steps,
            evaluator_conflict_threshold=evaluator_conflict_threshold,
        )
        search = search_thresholds(inner_results, SplitName.VALIDATION, grid)
        ds_thresholds = search.best.thresholds

        baseline_search = search_baseline_thresholds(
            inner_samples,
            inner_predictions,
            SplitName.VALIDATION,
            single_evaluator,
            decision_thresholds=baseline_decision_thresholds,
        )
        # 取各方法最优平台的公共取值；没有交集时退回 D-S 之外最常见的那个。
        decision_threshold = _shared_decision_threshold(baseline_search)
        # conflict_aware 的冲突分数下限用它自己在内层搜出的值。
        ca_best = baseline_search.best.get("conflict_aware")
        conflict_threshold = (
            ca_best.conflict_threshold if ca_best is not None else None
        )

        # --- 外层：用选好的阈值预测测试折 ---
        test_results = run_pipeline(test_samples, test_predictions, ds_thresholds)
        gold = {
            (r.sample_id, r.claim_id): r.gold_state.value  # type: ignore[union-attr]
            for r in test_results
        }
        for result in test_results:
            pooled.append(
                MethodPrediction(
                    method=DS_METHOD,
                    sample_id=result.sample_id,
                    claim_id=result.claim_id,
                    predicted_label=predicted_label(result.diagnostic),
                    gold_label=gold[(result.sample_id, result.claim_id)],
                    insufficiency_score=result.diagnostic.m_theta or 0.0,
                    conflict_score=result.diagnostic.k_doc,
                    confidence=ds_confidence(
                        predicted_label(result.diagnostic),
                        result.diagnostic.m_support,
                        result.diagnostic.m_refute,
                        result.diagnostic.m_theta,
                        result.diagnostic.k_doc,
                    ),
                )
            )

        baseline_thresholds = BaselineThresholds(
            decision_threshold=decision_threshold,
            **(
                {"conflict_threshold": conflict_threshold}
                if conflict_threshold is not None
                else {}
            ),
        )
        baseline_results = run_baselines(
            test_samples,
            test_predictions,
            baseline_thresholds,
            single_evaluator,
        )
        for item in baseline_results:
            pooled.append(
                MethodPrediction(
                    method=item.method.value,
                    sample_id=item.sample_id,
                    claim_id=item.claim_id,
                    predicted_label=item.predicted_state.value,
                    gold_label=gold[(item.sample_id, item.claim_id)],
                    insufficiency_score=item.score_unknown,
                    # 与 collect_predictions 保持同一口径：baseline 没有冲突量，
                    # 用 1 - |支持分 - 反驳分| 作为代理。
                    conflict_score=1.0
                    - abs(item.score_support - item.score_refute),
                    confidence=baseline_confidence(
                        item.predicted_state.value,
                        item.score_support,
                        item.score_refute,
                        item.score_unknown,
                    ),
                )
            )

        records.append(
            FoldThresholds(
                fold=fold_index,
                test_size=len(test_samples),
                inner_size=len(inner_samples),
                ds_thresholds=ds_thresholds,
                baseline_decision_threshold=decision_threshold,
                baseline_conflict_threshold=conflict_threshold,
                inner_macro_f1=search.best.macro_f1,
            )
        )
        if not methods:
            methods = [DS_METHOD] + sorted(
                {item.method.value for item in baseline_results}
            )

    claim_count = len({(p.sample_id, p.claim_id) for p in pooled})
    return CrossValidationResult(
        n_folds=n_folds,
        seed=seed,
        predictions=tuple(pooled),
        folds=tuple(records),
        claim_count=claim_count,
        methods=tuple(methods),
    )


def _shared_decision_threshold(search) -> float:
    """取三个 baseline 最优平台的交集中点；无交集时取各自最优的中位数。"""
    plateaus: dict[str, tuple[float, float]] = {}
    for method, best in search.best.items():
        tied = [
            c.decision_threshold
            for c in search.candidates
            if c.method.value == method and c.macro_f1 == best.macro_f1
        ]
        plateaus[method] = (min(tied), max(tied))

    low = max(bounds[0] for bounds in plateaus.values())
    high = min(bounds[1] for bounds in plateaus.values())
    if low <= high:
        return round((low + high) / 2, 6)
    # 没有交集：单一阈值无法让三者同时最优，取各自最优值的中位数并如实记录。
    values = sorted(best.decision_threshold for best in search.best.values())
    return values[len(values) // 2]
