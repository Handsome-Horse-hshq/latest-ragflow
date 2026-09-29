"""在验证集上校准 RAGChecker 的「标签 → 三元概率」映射表。

为什么必须校准
--------------
``LabelProbabilityMapping`` 的默认值 ``(0.90, 0.05, 0.05)`` 等三组数字是
**占位值**，不是从任何数据里得出的。一个 entailment 判得很准、neutral 判得
很松的评估器，与一个反过来的评估器，理应得到不同的映射。不校准就拿去跑测试
集，得到的数字不能写进论文。

校准口径
--------
CLIMATE-FEVER 的人工证据投票给出了每个 (claim, document) 组合上的三元分布
（``*_relations.jsonl``，evaluator 名为 ``climate_fever_human_vote_distribution``）。
本模块把它当作**验证集上的监督信号**，对每个标签 L 求条件平均::

    mapping[L] = sum_pairs  w_L(pair) * oracle_triple(pair)
                 ----------------------------------------
                 sum_pairs  w_L(pair)

其中 ``w_L(pair)`` 是 RAGChecker 在该组合上投给标签 L 的权重（子 claim 只有
一条时就是 0 或 1）。这就是「RAGChecker 说 L 时，人工投票平均长什么样」。

三条红线
--------
1. **只能在验证集上校准。** :class:`LabelCalibrationResult` 会拒绝非
   validation 的 split，与 :func:`~rag_ds.tuning.threshold_search.search_thresholds`
   的约束一致。
2. **测试集全程不接触 oracle。** 校准完成后，测试集上的关系概率完全由
   RAGChecker 的标签经该映射表换算得到，人工投票不参与。
3. **论文必须写明**：映射表是用验证集的人工投票标定的，这是方法的一部分，
   不是测试集泄漏。

覆盖不到的标签
--------------
某个标签在验证集上一次都没出现时，无法估计它的映射。默认**直接报错**，
不静默沿用占位值 —— 那会让「校准过的参数」与「没校准的占位值」在结果里
混在一起看不出来。确需继续时显式传 ``allow_default_for_unseen=True``，
沿用的标签会记录在 :attr:`LabelCalibrationResult.used_default` 里。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.ragchecker_ingest import ClaimLabelDistribution
from rag_ds.relation_evaluation.ragchecker_adapter import (
    DEFAULT_LABEL_MAPPING,
    LabelProbabilityMapping,
    RAGCheckerLabel,
)
from rag_ds.schemas import RelationPrediction
from rag_ds.tuning.threshold_search import SplitName

__all__ = [
    "LabelCalibrationError",
    "LabelCalibrationResult",
    "LabelCalibrationStat",
    "calibrate_label_mapping",
]


class LabelCalibrationError(ValueError):
    """校准输入与 oracle 对不上，或某个标签没有可用样本。"""


#: ``(sample_id, claim_id, doc_id)``。
_PairKey = tuple[str, str, str]


class LabelCalibrationStat(BaseModel):
    """单个标签的校准统计量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: RAGCheckerLabel
    #: 该标签累计到的权重质量；子 claim 均为单条时等于组合数。
    weight_mass: float = Field(ge=0.0)
    #: 该标签为唯一多数票的组合数，便于报告一个整数规模。
    majority_pairs: int = Field(ge=0)
    #: 校准得到的 ``(p_support, p_refute, p_unknown)``。
    probabilities: tuple[float, float, float]
    #: 该标签无样本、沿用占位默认值时为 ``True``。
    used_default: bool = False


class LabelCalibrationResult(BaseModel):
    """一次标签映射校准的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 校准所用的数据划分，必须是 ``validation``。
    split: SplitName
    #: 校准出的映射表，可直接交给摄取脚本。
    mapping: LabelProbabilityMapping
    #: 参与校准的 (claim, doc) 组合数。
    pair_count: int = Field(ge=0)
    stats: tuple[LabelCalibrationStat, ...]
    #: 因验证集上无样本而沿用占位默认值的标签。
    used_default: tuple[RAGCheckerLabel, ...] = ()
    #: RAGChecker 多数标签 × oracle 三元分布 argmax 的列联表，供论文报告。
    contingency: dict[str, dict[str, int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_split_is_validation(self) -> LabelCalibrationResult:
        """映射表只允许在验证集上校准。"""
        if self.split is not SplitName.VALIDATION:
            raise ValueError(
                f"标签映射只能在验证集上校准，收到 split={self.split.value!r}"
            )
        return self


def _oracle_index(
    oracle: Iterable[RelationPrediction], expected_evaluator: str | None
) -> dict[_PairKey, RelationPrediction]:
    """按 (sample, claim, doc) 建立 oracle 索引。"""
    table: dict[_PairKey, RelationPrediction] = {}
    for prediction in oracle:
        if (
            expected_evaluator is not None
            and prediction.evaluator != expected_evaluator
        ):
            continue
        key: _PairKey = (
            prediction.sample_id,
            prediction.claim_id,
            prediction.doc_id,
        )
        if key in table:
            raise LabelCalibrationError(f"oracle 中出现重复的组合：{key}")
        table[key] = prediction
    if not table:
        raise LabelCalibrationError(
            "没有任何可用的 oracle 关系预测"
            + (f"（evaluator 过滤为 {expected_evaluator!r}）" if expected_evaluator else "")
        )
    return table


def _argmax_state(prediction: RelationPrediction) -> str:
    """oracle 三元分布的 argmax；并列时记为 ``tie``。"""
    scored = [
        (prediction.p_support, "support"),
        (prediction.p_refute, "refute"),
        (prediction.p_unknown, "unknown"),
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] == scored[1][0]:
        return "tie"
    return scored[0][1]


def calibrate_label_mapping(
    distributions: Sequence[ClaimLabelDistribution],
    oracle: Sequence[RelationPrediction],
    split: SplitName,
    *,
    oracle_evaluator: str | None = None,
    allow_default_for_unseen: bool = False,
) -> LabelCalibrationResult:
    """用验证集的人工投票标定标签映射表。

    Args:
        distributions: 验证集上 RAGChecker 的标签分布（由
            :func:`~rag_ds.ragchecker_ingest.parse_checking_outputs` 产出）。
        oracle: 验证集的人工投票关系预测。
        split: 数据划分名称；**必须**为 :attr:`SplitName.VALIDATION`。
        oracle_evaluator: 只使用该评估器名下的 oracle 记录；``None`` 表示全用。
        allow_default_for_unseen: 某标签在验证集上没出现时是否沿用占位默认值。

    Returns:
        :class:`LabelCalibrationResult`。

    Raises:
        LabelCalibrationError: 组合对不上，或某标签无样本且未允许沿用默认值。
        ValueError: ``split`` 不是 validation。
    """
    if split is not SplitName.VALIDATION:
        raise ValueError(
            f"标签映射只能在验证集上校准，收到 split={split.value!r}"
        )
    if not distributions:
        raise LabelCalibrationError("没有任何 RAGChecker 标签分布可供校准")

    table = _oracle_index(oracle, oracle_evaluator)

    sums: dict[RAGCheckerLabel, list[float]] = {
        label: [0.0, 0.0, 0.0] for label in RAGCheckerLabel
    }
    mass: dict[RAGCheckerLabel, float] = {label: 0.0 for label in RAGCheckerLabel}
    majority: dict[RAGCheckerLabel, int] = {label: 0 for label in RAGCheckerLabel}
    contingency: dict[str, dict[str, int]] = {
        label.value: {"support": 0, "refute": 0, "unknown": 0, "tie": 0}
        for label in RAGCheckerLabel
    }

    for item in distributions:
        key: _PairKey = (item.sample_id, item.claim_id, item.doc_id)
        target = table.get(key)
        if target is None:
            raise LabelCalibrationError(
                f"oracle 中缺少组合 {key}；校准要求两边覆盖同一批 (claim, doc)"
            )
        triple = (target.p_support, target.p_refute, target.p_unknown)
        for label, weight in item.weights.items():
            if weight == 0.0:
                continue
            mass[label] += weight
            bucket = sums[label]
            bucket[0] += weight * triple[0]
            bucket[1] += weight * triple[1]
            bucket[2] += weight * triple[2]
        winner = item.majority_label
        if winner is not None:
            majority[winner] += 1
            contingency[winner.value][_argmax_state(target)] += 1

    unseen = [label for label in RAGCheckerLabel if mass[label] == 0.0]
    if unseen and not allow_default_for_unseen:
        raise LabelCalibrationError(
            "以下标签在验证集上一次都没出现，无法校准："
            f"{[label.value for label in unseen]}；"
            "确需沿用占位默认值请显式传 allow_default_for_unseen=True"
        )

    probabilities: dict[str, tuple[float, float, float]] = {}
    stats: list[LabelCalibrationStat] = []
    for label in RAGCheckerLabel:
        if mass[label] == 0.0:
            triple = DEFAULT_LABEL_MAPPING.probabilities(label)
            used_default = True
        else:
            bucket = sums[label]
            total = mass[label]
            triple = (bucket[0] / total, bucket[1] / total, bucket[2] / total)
            # oracle 三元组已归一，凸组合理论上仍归一；这里只兜浮点尾差。
            scale = sum(triple)
            triple = (triple[0] / scale, triple[1] / scale, triple[2] / scale)
            used_default = False
        probabilities[label.value] = triple
        stats.append(
            LabelCalibrationStat(
                label=label,
                weight_mass=mass[label],
                majority_pairs=majority[label],
                probabilities=triple,
                used_default=used_default,
            )
        )

    return LabelCalibrationResult(
        split=split,
        mapping=LabelProbabilityMapping(**probabilities),
        pair_count=len(distributions),
        stats=tuple(stats),
        used_default=tuple(label for label in RAGCheckerLabel if mass[label] == 0.0),
        contingency=contingency,
    )
