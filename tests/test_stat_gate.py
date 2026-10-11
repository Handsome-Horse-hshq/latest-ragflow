"""Statistic Gate 测试：不用 D-S 的两维判定对照组。

这个对照组的意义全在于「只换坐标轴、其余与 D-S 完全相同」。所以这组测试
钉住三件事：两个统计量算得对；判定规则与 D-S 用的是同一张区域表；阈值搜索
与 D-S 同样只允许在验证集上做。
"""

from __future__ import annotations

import pytest

from rag_ds.baselines.stat_gate import (
    STAT_GATE_VARIANTS,
    ConflictStatistic,
    InsufficiencyStatistic,
    StatGateVariant,
    decide_stat_gate,
    stat_gate_method_name,
    stat_gate_scores,
)
from rag_ds.diagnostics.gating import gate_primary_state
from rag_ds.diagnostics.models import ClaimVerdict, DiagnosticThresholds
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction
from rag_ds.tuning import SplitName, search_stat_gate_thresholds, stat_gate_label

EVALUATOR = "e1"


def _sample(
    sample_id: str,
    gold: EvidenceState | None = EvidenceState.SUPPORTED,
    reliabilities: tuple[float, float] = (1.0, 1.0),
) -> RAGSample:
    return RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=(
            ContextChunk(doc_id=f"{sample_id}-d1", text="文档一。", reliability=reliabilities[0]),
            ContextChunk(doc_id=f"{sample_id}-d2", text="文档二。", reliability=reliabilities[1]),
        ),
        gold_state=gold,
    )


def _predictions(sample_id: str, first, second) -> list[RelationPrediction]:
    return [
        RelationPrediction(
            sample_id=sample_id,
            claim_id=f"{sample_id}-c1",
            doc_id=f"{sample_id}-d{index}",
            evaluator=EVALUATOR,
            p_support=triple[0],
            p_refute=triple[1],
            p_unknown=triple[2],
        )
        for index, triple in enumerate((first, second), start=1)
    ]


def _thresholds(theta: float, doc: float) -> DiagnosticThresholds:
    return DiagnosticThresholds(theta_threshold=theta, document_conflict_threshold=doc)


class TestScores:
    def test_opposing_documents(self) -> None:
        """一篇 0.7 支持、一篇 0.7 反驳：均值相等，margin ±0.5，标准差 0.5。"""
        sample = _sample("s1")
        predictions = _predictions("s1", (0.7, 0.2, 0.1), (0.2, 0.7, 0.1))

        (std,) = stat_gate_scores([sample], predictions, ConflictStatistic.SUPPORT_REFUTE_STD)
        (product,) = stat_gate_scores([sample], predictions, ConflictStatistic.MAX_PRODUCT)

        assert std.score_support == pytest.approx(0.45)
        assert std.score_refute == pytest.approx(0.45)
        assert std.score_unknown == pytest.approx(0.1)
        assert std.conflict == pytest.approx(0.5)
        assert product.conflict == pytest.approx(0.49)
        assert std.input_count == 2

    def test_agreeing_documents_have_zero_spread(self) -> None:
        sample = _sample("s1")
        predictions = _predictions("s1", (0.6, 0.1, 0.3), (0.6, 0.1, 0.3))

        (scores,) = stat_gate_scores([sample], predictions, ConflictStatistic.SUPPORT_REFUTE_STD)

        assert scores.conflict == pytest.approx(0.0)

    def test_zero_reliability_document_is_ignored(self) -> None:
        """可靠性为 0 的文档权重为 0：均值与离散度只由另一篇决定。"""
        sample = _sample("s1", reliabilities=(1.0, 0.0))
        predictions = _predictions("s1", (0.7, 0.2, 0.1), (0.2, 0.7, 0.1))

        (std,) = stat_gate_scores([sample], predictions, ConflictStatistic.SUPPORT_REFUTE_STD)
        (product,) = stat_gate_scores([sample], predictions, ConflictStatistic.MAX_PRODUCT)

        assert std.score_support == pytest.approx(0.7)
        assert std.conflict == pytest.approx(0.0)
        # 折扣后的最强反驳是 0.2（来自可靠的那篇），不是 0.7。
        assert product.conflict == pytest.approx(0.7 * 0.2)

    def test_gold_state_is_not_read(self) -> None:
        predictions = _predictions("s1", (0.7, 0.2, 0.1), (0.2, 0.7, 0.1))

        with_gold = stat_gate_scores(
            [_sample("s1", EvidenceState.SUPPORTED)], predictions, ConflictStatistic.SUPPORT_REFUTE_STD
        )
        without_gold = stat_gate_scores(
            [_sample("s1", None)], predictions, ConflictStatistic.SUPPORT_REFUTE_STD
        )

        assert with_gold == without_gold

    def test_method_name(self) -> None:
        assert stat_gate_method_name(ConflictStatistic.SUPPORT_REFUTE_STD) == "stat_gate_sr_std"


class TestDecision:
    @pytest.mark.parametrize(
        ("insufficient", "conflict", "verdict", "expected"),
        [
            (False, False, ClaimVerdict.SUPPORTED, EvidenceState.SUPPORTED),
            (False, False, ClaimVerdict.REFUTED, EvidenceState.REFUTED),
            (False, False, ClaimVerdict.UNDETERMINED, None),
            (True, False, ClaimVerdict.SUPPORTED, EvidenceState.INSUFFICIENT),
            (False, True, ClaimVerdict.SUPPORTED, EvidenceState.CONFLICTING),
            # 混合区域与 D-S 一样映射为 conflicting，不丢冲突信息。
            (True, True, ClaimVerdict.REFUTED, EvidenceState.CONFLICTING),
        ],
    )
    def test_shared_region_mapping(self, insufficient, conflict, verdict, expected) -> None:
        assert gate_primary_state(insufficient, conflict, verdict) is expected

    def test_high_conflict_statistic_gives_conflicting(self) -> None:
        (scores,) = stat_gate_scores(
            [_sample("s1")],
            _predictions("s1", (0.7, 0.2, 0.1), (0.2, 0.7, 0.1)),
            ConflictStatistic.SUPPORT_REFUTE_STD,
        )

        assert decide_stat_gate(scores, _thresholds(0.5, 0.4)) is EvidenceState.CONFLICTING
        # 冲突门槛抬到统计量之上，就只剩方向判断；两边相等 → 分不出倾向。
        assert decide_stat_gate(scores, _thresholds(0.5, 0.6)) is None
        assert stat_gate_label(None) == "undetermined"

    def test_high_unknown_gives_insufficient(self) -> None:
        (scores,) = stat_gate_scores(
            [_sample("s1")],
            _predictions("s1", (0.1, 0.1, 0.8), (0.15, 0.05, 0.8)),
            ConflictStatistic.SUPPORT_REFUTE_STD,
        )

        assert decide_stat_gate(scores, _thresholds(0.7, 0.5)) is EvidenceState.INSUFFICIENT
        assert decide_stat_gate(scores, _thresholds(0.9, 0.5)) is EvidenceState.SUPPORTED

    def test_threshold_uses_greater_or_equal(self) -> None:
        """与 D-S 一致：恰好等于阈值算「高」。"""
        (scores,) = stat_gate_scores(
            [_sample("s1")],
            _predictions("s1", (0.1, 0.1, 0.8), (0.1, 0.1, 0.8)),
            ConflictStatistic.SUPPORT_REFUTE_STD,
        )

        assert decide_stat_gate(scores, _thresholds(scores.score_unknown, 0.5)) is (
            EvidenceState.INSUFFICIENT
        )


def _search_dataset():
    samples = [
        _sample("s1", EvidenceState.SUPPORTED),
        _sample("s2", EvidenceState.REFUTED),
        _sample("s3", EvidenceState.INSUFFICIENT),
        _sample("s4", EvidenceState.CONFLICTING),
    ]
    predictions = (
        _predictions("s1", (0.6, 0.1, 0.3), (0.5, 0.1, 0.4))
        + _predictions("s2", (0.1, 0.6, 0.3), (0.1, 0.5, 0.4))
        + _predictions("s3", (0.1, 0.1, 0.8), (0.05, 0.05, 0.9))
        + _predictions("s4", (0.7, 0.2, 0.1), (0.2, 0.7, 0.1))
    )
    scores = stat_gate_scores(samples, predictions, ConflictStatistic.SUPPORT_REFUTE_STD)
    gold = {(s.sample_id, f"{s.sample_id}-c1"): s.gold_state.value for s in samples}
    return scores, gold


class TestSearch:
    @pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.TEST])
    def test_non_validation_split_is_rejected(self, split: SplitName) -> None:
        scores, gold = _search_dataset()

        with pytest.raises(ValueError, match="只能在验证集上搜索"):
            search_stat_gate_thresholds(scores, gold, split)

    def test_separable_data_is_solved(self) -> None:
        scores, gold = _search_dataset()

        result = search_stat_gate_thresholds(scores, gold, SplitName.VALIDATION, steps=4)

        assert result.best.macro_f1 == pytest.approx(1.0)
        assert result.claim_count == 4
        # 两个轴各 4 个分位点，去重后各剩 3 个观测值：(0.1, 0.35, 0.85) 与 (0, 0.05, 0.5)。
        assert len(result.candidates) == 9

    def test_ties_break_to_smallest_thresholds(self) -> None:
        scores, gold = _search_dataset()

        result = search_stat_gate_thresholds(scores, gold, SplitName.VALIDATION, steps=4)
        top = [c for c in result.candidates if c.macro_f1 == result.best.macro_f1]

        assert result.best.thresholds == min(
            (c.thresholds for c in top),
            key=lambda t: (t.theta_threshold, t.document_conflict_threshold),
        )

    def test_missing_gold_is_rejected(self) -> None:
        scores, gold = _search_dataset()
        gold.pop(("s1", "s1-c1"))

        with pytest.raises(ValueError, match="缺少金标准"):
            search_stat_gate_thresholds(scores, gold, SplitName.VALIDATION)


class TestProductVariant:
    """横轴换成 p_unknown 连乘：所有文档都说不清才算证据不足。"""

    PRODUCT = StatGateVariant(
        conflict=ConflictStatistic.MAX_PRODUCT,
        insufficiency=InsufficiencyStatistic.UNKNOWN_PRODUCT,
    )

    def test_four_variants_with_stable_names(self) -> None:
        assert list(STAT_GATE_VARIANTS) == [
            "sr_std",
            "max_product",
            "sr_std_uprod",
            "max_product_uprod",
        ]
        assert self.PRODUCT.method_name == "stat_gate_max_product_uprod"
        assert STAT_GATE_VARIANTS["max_product_uprod"] == self.PRODUCT

    def test_bare_statistic_means_average_axis(self) -> None:
        variant = StatGateVariant.of(ConflictStatistic.SUPPORT_REFUTE_STD)

        assert variant.insufficiency is InsufficiencyStatistic.MEAN_UNKNOWN
        assert variant.method_name == stat_gate_method_name(
            ConflictStatistic.SUPPORT_REFUTE_STD
        )

    def test_one_informative_document_pulls_the_product_down(self) -> None:
        """一篇明确支持 + 一篇说不清：平均仍有 0.5，连乘只剩 0.09。"""
        sample = _sample("s1")
        predictions = _predictions("s1", (0.9, 0.0, 0.1), (0.05, 0.05, 0.9))

        (mean,) = stat_gate_scores([sample], predictions, ConflictStatistic.MAX_PRODUCT)
        (product,) = stat_gate_scores([sample], predictions, self.PRODUCT)

        assert mean.insufficiency == pytest.approx(0.5)
        assert product.insufficiency == pytest.approx(0.09)
        # 其余分数不受横轴选择影响。
        assert product.score_unknown == pytest.approx(0.5)
        assert product.conflict == mean.conflict
        # 同一组阈值下，平均判「证据不足」，连乘让那篇明确支持的文档说了算。
        thresholds = _thresholds(0.4, 0.9)
        assert decide_stat_gate(mean, thresholds) is EvidenceState.INSUFFICIENT
        assert decide_stat_gate(product, thresholds) is EvidenceState.SUPPORTED

    def test_product_uses_discounted_unknown(self) -> None:
        """可靠性 0.5 的文档：不确定被折扣抬到 1 - 0.5 × (1 - 0.8) = 0.9。"""
        sample = _sample("s1", reliabilities=(0.5, 1.0))
        predictions = _predictions("s1", (0.1, 0.1, 0.8), (0.1, 0.1, 0.8))

        (scores,) = stat_gate_scores([sample], predictions, self.PRODUCT)

        assert scores.insufficiency == pytest.approx(0.9 * 0.8)

    def test_search_grid_follows_the_chosen_axis(self) -> None:
        samples = [_sample("s1", EvidenceState.SUPPORTED), _sample("s2", EvidenceState.INSUFFICIENT)]
        predictions = _predictions("s1", (0.9, 0.0, 0.1), (0.05, 0.05, 0.9)) + _predictions(
            "s2", (0.1, 0.1, 0.8), (0.05, 0.05, 0.9)
        )
        gold = {(s.sample_id, f"{s.sample_id}-c1"): s.gold_state.value for s in samples}

        result = search_stat_gate_thresholds(
            stat_gate_scores(samples, predictions, self.PRODUCT),
            gold,
            SplitName.VALIDATION,
            steps=2,
        )

        # 候选阈值来自连乘的观测值（0.09 与 0.72），不是平均值（0.5 与 0.85）。
        assert {c.thresholds.theta_threshold for c in result.candidates} == {0.09, 0.72}
