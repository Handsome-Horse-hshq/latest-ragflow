"""Statistic Gate：不用 D-S 的两维判定对照组。

为什么需要它
------------
D-S 相对朴素 baseline 的优势可以拆成两层：能输出 ``conflicting`` 的**输出
空间**（``conflict_aware`` 已经把这一层分离出来），以及剩下的部分。剩下的
部分里又混着两件事：

1. **两维判定框架**：先问「证据够不够」、再问「文档打不打架」，最后才看
   支持还是反驳；
2. **Dempster 融合本身**：用 ``m_theta`` 与 ``K_doc`` 作为这两个维度的分数。

本方法保留第 1 件、拿掉第 2 件：判定规则、区域到类别的映射（共用
:func:`~rag_ds.diagnostics.gating.gate_primary_state`）、阈值网格的构造方式
与搜索目标都和 D-S **完全相同**，只把两个坐标轴换成不做任何证据组合的简单
统计量。它与 D-S 打平，就说明优势来自框架而不是融合；D-S 仍显著更好，
才能把功劳记在 Dempster 融合上。

两个坐标轴
----------
横轴（证据不足），由 :class:`InsufficiencyStatistic` 指定：

* ``mean_unknown``：加权平均 ``p_unknown``，与 ``weighted_average`` 的
  ``score_unknown`` 逐字相同。
* ``unknown_product``：折扣后 ``p_unknown`` 的连乘 —— **所有**文档都说不清时
  才高，只要有一篇给出明确意见就被压低。单评估器、可靠性全为 1 时
  ``m_theta = 连乘 / (1 - K_doc)`` 精确成立，所以它就是「去掉 Dempster 归一化
  的 m_theta」。

纵轴（文档冲突），由 :class:`ConflictStatistic` 指定：

* ``sr_std``：各条关系的 ``p_support - p_refute`` 的加权标准差 —— 文档之间
  「偏向哪边」分歧有多大。这是不借助 D-S 时对「文档冲突」最直接的定义。
* ``max_product``：按可靠性折扣后的最强支持 × 最强反驳 —— 有没有一篇强
  支持、同时另一篇强反驳。

两个轴的组合由 :class:`StatGateVariant` 表示，共四个变体，实验里**全部报告**。

权重与 ``weighted_average`` 一致：文档可靠性 × 评估器可靠性。取最大值或连乘
的统计量用同一权重给概率打折（支持、反驳乘以权重，不确定取
``1 - 权重 × (1 - p_unknown)``），与 D-S 的可靠性折扣对应。

多评估器时，所有 (文档, 评估器) 组合的关系一起参与统计，纵轴因此同时反映
文档之间与评估器之间的分歧；本方法不区分 ``K_doc`` 与 ``K_eval``。

``retrieval_score`` 与 ``gold_state`` 都不参与计算。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from rag_ds.baselines._shared import contexts_by_doc_id
from rag_ds.baselines.runner import _claim_predictions
from rag_ds.diagnostics.gating import determine_verdict, gate_primary_state
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.integrity import check_samples, index_predictions
from rag_ds.schemas import (
    EvidenceState,
    NonEmptyStr,
    RAGSample,
    RelationPrediction,
    UnitFloat,
)

__all__ = [
    "STAT_GATE_VARIANTS",
    "ConflictStatistic",
    "InsufficiencyStatistic",
    "StatGateScores",
    "StatGateVariant",
    "decide_stat_gate",
    "stat_gate_method_name",
    "stat_gate_scores",
]


class ConflictStatistic(str, Enum):
    """Statistic Gate 纵轴使用的冲突统计量。"""

    #: 各条关系 ``p_support - p_refute`` 的加权标准差。
    SUPPORT_REFUTE_STD = "sr_std"
    #: 折扣后的最强支持 × 最强反驳。
    MAX_PRODUCT = "max_product"


class InsufficiencyStatistic(str, Enum):
    """Statistic Gate 横轴使用的证据不足统计量。"""

    #: 加权平均 ``p_unknown``。
    MEAN_UNKNOWN = "mean_unknown"
    #: 折扣后 ``p_unknown`` 的连乘。
    UNKNOWN_PRODUCT = "unknown_product"


class StatGateVariant(BaseModel):
    """一个 Statistic Gate 变体：纵轴统计量 + 横轴统计量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conflict: ConflictStatistic
    insufficiency: InsufficiencyStatistic = InsufficiencyStatistic.MEAN_UNKNOWN

    @property
    def name(self) -> str:
        """命令行与结果文件里用的短名，例如 ``sr_std``、``max_product_uprod``。"""
        if self.insufficiency is InsufficiencyStatistic.UNKNOWN_PRODUCT:
            return f"{self.conflict.value}_uprod"
        return self.conflict.value

    @property
    def method_name(self) -> str:
        """该变体在结果里的方法名，例如 ``stat_gate_sr_std``。"""
        return f"stat_gate_{self.name}"

    @classmethod
    def of(cls, value: StatGateVariant | ConflictStatistic) -> StatGateVariant:
        """只给纵轴统计量时，横轴取平均 ``p_unknown``。"""
        return value if isinstance(value, StatGateVariant) else cls(conflict=value)


#: 全部四个变体，按短名索引。
STAT_GATE_VARIANTS: dict[str, StatGateVariant] = {
    variant.name: variant
    for variant in (
        StatGateVariant(conflict=conflict, insufficiency=insufficiency)
        for insufficiency in InsufficiencyStatistic
        for conflict in ConflictStatistic
    )
}


def stat_gate_method_name(variant: StatGateVariant | ConflictStatistic) -> str:
    """该变体在结果里的方法名，例如 ``stat_gate_sr_std``。"""
    return StatGateVariant.of(variant).method_name


class StatGateScores(BaseModel):
    """一条 claim 在 Statistic Gate 下的全部分数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    variant: StatGateVariant
    #: 加权平均的三个概率；倾向由 ``score_support`` 与 ``score_refute`` 之差决定。
    score_support: UnitFloat
    score_refute: UnitFloat
    score_unknown: UnitFloat
    #: 横轴：所选证据不足统计量的取值。
    insufficiency: UnitFloat
    #: 纵轴：所选冲突统计量的取值。
    conflict: UnitFloat
    #: 参与本次计算的关系预测条数。
    input_count: int = Field(ge=0)


def _claim_scores(
    sample: RAGSample,
    claim_id: str,
    predictions: list[RelationPrediction],
    variant: StatGateVariant,
) -> StatGateScores:
    """在一条 claim 的全部关系预测上算出三个平均分与两个坐标。"""
    contexts = contexts_by_doc_id(sample)
    weights = [
        contexts[p.doc_id].reliability * p.evaluator_reliability for p in predictions
    ]
    total = sum(weights)
    if total <= 0.0:
        # 没有可用证据：与 baseline 的 no_evidence 一致，全部落在不确定上；
        # 没有证据谈不上冲突。
        return StatGateScores(
            sample_id=sample.sample_id,
            claim_id=claim_id,
            variant=variant,
            score_support=0.0,
            score_refute=0.0,
            score_unknown=1.0,
            insufficiency=1.0,
            conflict=0.0,
            input_count=len(predictions),
        )

    support = sum(w * p.p_support for w, p in zip(weights, predictions)) / total
    refute = sum(w * p.p_refute for w, p in zip(weights, predictions)) / total
    unknown = sum(w * p.p_unknown for w, p in zip(weights, predictions)) / total

    if variant.insufficiency is InsufficiencyStatistic.MEAN_UNKNOWN:
        insufficiency = unknown
    else:
        insufficiency = math.prod(
            1.0 - w * (1.0 - p.p_unknown) for w, p in zip(weights, predictions)
        )

    if variant.conflict is ConflictStatistic.SUPPORT_REFUTE_STD:
        margins = [p.p_support - p.p_refute for p in predictions]
        mean = sum(w * m for w, m in zip(weights, margins)) / total
        variance = sum(w * (m - mean) ** 2 for w, m in zip(weights, margins)) / total
        conflict = math.sqrt(max(variance, 0.0))
    else:
        conflict = max(w * p.p_support for w, p in zip(weights, predictions)) * max(
            w * p.p_refute for w, p in zip(weights, predictions)
        )

    return StatGateScores(
        sample_id=sample.sample_id,
        claim_id=claim_id,
        variant=variant,
        score_support=support,
        score_refute=refute,
        score_unknown=unknown,
        # 两个坐标理论上都落在 [0, 1]；夹一下只防浮点噪声。
        insufficiency=min(max(insufficiency, 0.0), 1.0),
        conflict=min(conflict, 1.0),
        input_count=len(predictions),
    )


def stat_gate_scores(
    samples: Iterable[RAGSample],
    predictions: Iterable[RelationPrediction],
    variant: StatGateVariant | ConflictStatistic,
) -> list[StatGateScores]:
    """对每条 claim 算出 Statistic Gate 的分数。

    输入完整性沿用 baseline 与 D-S 共用的同一份检查，收齐预测的顺序也与
    :func:`~rag_ds.baselines.runner.run_baselines` 相同。

    Args:
        samples: 待判定的样本。
        predictions: 关系评估器的输出。
        variant: 要用的变体；只给 :class:`ConflictStatistic` 时横轴取平均
            ``p_unknown``。

    Returns:
        每条 claim 一条 :class:`StatGateScores`，按「样本 → claim」顺序。

    Raises:
        PipelineError: 输入数据不满足完整性要求。
    """
    resolved = StatGateVariant.of(variant)
    sample_list = list(samples)
    prediction_list = list(predictions)
    check_samples(sample_list)
    table = index_predictions(sample_list, prediction_list)

    return [
        _claim_scores(
            sample,
            claim.claim_id,
            _claim_predictions(table, sample, claim.claim_id),
            resolved,
        )
        for sample in sample_list
        for claim in sample.claims
    ]


def decide_stat_gate(
    scores: StatGateScores, thresholds: DiagnosticThresholds
) -> EvidenceState | None:
    """用与 D-S 相同的两维门控规则给出主状态。

    ``theta_threshold`` 作用于横轴 ``insufficiency``，
    ``document_conflict_threshold`` 作用于纵轴 ``conflict``，判「高」一律用
    ``>=``；倾向由平均支持分与平均反驳分之差决定。

    Returns:
        四类之一；证据充分却分不出倾向时为 ``None``，调用方应记为
        ``undetermined``（与 D-S 一致）。
    """
    verdict = determine_verdict(
        scores.score_support, scores.score_refute, thresholds.tie_tolerance
    )
    return gate_primary_state(
        scores.insufficiency >= thresholds.theta_threshold,
        scores.conflict >= thresholds.document_conflict_threshold,
        verdict,
    )
