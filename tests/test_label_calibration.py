"""标签映射校准测试：验证集约束、条件平均与覆盖检查。"""

from __future__ import annotations

import pytest

from rag_ds.ragchecker_ingest import ClaimLabelDistribution
from rag_ds.relation_evaluation.ragchecker_adapter import (
    DEFAULT_LABEL_MAPPING,
    RAGCheckerLabel,
)
from rag_ds.schemas import RelationPrediction
from rag_ds.tuning import SplitName, calibrate_label_mapping
from rag_ds.tuning.label_calibration import (
    LabelCalibrationError,
    LabelCalibrationResult,
)

ORACLE = "climate_fever_human_vote_distribution"


def _distribution(doc_id: str, **counts: int) -> ClaimLabelDistribution:
    return ClaimLabelDistribution(
        sample_id="s1", claim_id="s1-c1", doc_id=doc_id, **counts
    )


def _oracle(
    doc_id: str, p_support: float, p_refute: float, p_unknown: float
) -> RelationPrediction:
    return RelationPrediction(
        sample_id="s1",
        claim_id="s1-c1",
        doc_id=doc_id,
        evaluator=ORACLE,
        p_support=p_support,
        p_refute=p_refute,
        p_unknown=p_unknown,
    )


def _three_label_case() -> tuple[list[ClaimLabelDistribution], list[RelationPrediction]]:
    """每个标签各两条，便于手算条件平均。"""
    distributions = [
        _distribution("d1", entailment=1),
        _distribution("d2", entailment=1),
        _distribution("d3", contradiction=1),
        _distribution("d4", contradiction=1),
        _distribution("d5", neutral=1),
        _distribution("d6", neutral=1),
    ]
    oracle = [
        _oracle("d1", 1.0, 0.0, 0.0),
        _oracle("d2", 0.6, 0.2, 0.2),
        _oracle("d3", 0.0, 1.0, 0.0),
        _oracle("d4", 0.2, 0.6, 0.2),
        _oracle("d5", 0.0, 0.0, 1.0),
        _oracle("d6", 0.25, 0.25, 0.5),
    ]
    return distributions, oracle


class TestValidationOnly:
    """参数只能在验证集上选。"""

    @pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.TEST])
    def test_non_validation_split_is_rejected(self, split: SplitName) -> None:
        distributions, oracle = _three_label_case()

        with pytest.raises(ValueError, match="只能在验证集上校准"):
            calibrate_label_mapping(distributions, oracle, split)

    def test_result_model_also_refuses_non_validation(self) -> None:
        distributions, oracle = _three_label_case()
        result = calibrate_label_mapping(distributions, oracle, SplitName.VALIDATION)

        with pytest.raises(ValueError, match="只能在验证集上校准"):
            LabelCalibrationResult(
                split=SplitName.TEST,
                mapping=result.mapping,
                pair_count=result.pair_count,
                stats=result.stats,
            )


class TestConditionalAverage:
    """校准值就是「该标签下人工投票的平均分布」。"""

    def test_each_label_is_the_mean_of_its_oracle_triples(self) -> None:
        distributions, oracle = _three_label_case()

        result = calibrate_label_mapping(distributions, oracle, SplitName.VALIDATION)

        assert result.mapping.entailment == pytest.approx((0.8, 0.1, 0.1))
        assert result.mapping.contradiction == pytest.approx((0.1, 0.8, 0.1))
        assert result.mapping.neutral == pytest.approx((0.125, 0.125, 0.75))
        assert result.pair_count == 6

    def test_every_calibrated_triple_is_normalised(self) -> None:
        distributions, oracle = _three_label_case()

        result = calibrate_label_mapping(distributions, oracle, SplitName.VALIDATION)

        for label in RAGCheckerLabel:
            assert sum(result.mapping.probabilities(label)) == pytest.approx(1.0)

    def test_multi_subclaim_votes_are_weighted(self) -> None:
        """一条组合抽出 2 个子 claim 时，两个标签各记 0.5 权重。"""
        distributions = [
            _distribution("d1", entailment=1, neutral=1),
            _distribution("d2", entailment=1),
        ]
        oracle = [_oracle("d1", 0.0, 0.0, 1.0), _oracle("d2", 1.0, 0.0, 0.0)]

        result = calibrate_label_mapping(
            distributions,
            oracle,
            SplitName.VALIDATION,
            # 这个小用例里没有 contradiction，覆盖检查本身另有测试。
            allow_default_for_unseen=True,
        )

        stats = {stat.label: stat for stat in result.stats}
        assert stats[RAGCheckerLabel.ENTAILMENT].weight_mass == pytest.approx(1.5)
        assert stats[RAGCheckerLabel.NEUTRAL].weight_mass == pytest.approx(0.5)
        # entailment 权重：d1 占 0.5（全 unknown），d2 占 1.0（全 support）
        assert result.mapping.entailment == pytest.approx((2 / 3, 0.0, 1 / 3))

    def test_contingency_counts_majority_label_against_oracle_argmax(self) -> None:
        distributions, oracle = _three_label_case()

        result = calibrate_label_mapping(distributions, oracle, SplitName.VALIDATION)

        assert result.contingency["entailment"]["support"] == 2
        assert result.contingency["contradiction"]["refute"] == 2
        assert result.contingency["neutral"]["unknown"] == 2


class TestCoverage:
    """覆盖不到就报错，不静默沿用占位值。"""

    def test_unseen_label_raises_by_default(self) -> None:
        distributions = [_distribution("d1", entailment=1)]
        oracle = [_oracle("d1", 1.0, 0.0, 0.0)]

        with pytest.raises(LabelCalibrationError, match="一次都没出现"):
            calibrate_label_mapping(distributions, oracle, SplitName.VALIDATION)

    def test_unseen_label_can_fall_back_explicitly(self) -> None:
        distributions = [_distribution("d1", entailment=1)]
        oracle = [_oracle("d1", 1.0, 0.0, 0.0)]

        result = calibrate_label_mapping(
            distributions,
            oracle,
            SplitName.VALIDATION,
            allow_default_for_unseen=True,
        )

        assert set(result.used_default) == {
            RAGCheckerLabel.NEUTRAL,
            RAGCheckerLabel.CONTRADICTION,
        }
        assert result.mapping.neutral == pytest.approx(DEFAULT_LABEL_MAPPING.neutral)
        assert result.mapping.entailment == pytest.approx((1.0, 0.0, 0.0))
        flagged = {stat.label for stat in result.stats if stat.used_default}
        assert RAGCheckerLabel.ENTAILMENT not in flagged

    def test_missing_oracle_pair_is_rejected(self) -> None:
        distributions, oracle = _three_label_case()

        with pytest.raises(LabelCalibrationError, match="oracle 中缺少组合"):
            calibrate_label_mapping(distributions, oracle[:-1], SplitName.VALIDATION)

    def test_duplicate_oracle_pair_is_rejected(self) -> None:
        distributions, oracle = _three_label_case()

        with pytest.raises(LabelCalibrationError, match="重复的组合"):
            calibrate_label_mapping(
                distributions, [*oracle, oracle[0]], SplitName.VALIDATION
            )

    def test_evaluator_filter_excludes_other_evaluators(self) -> None:
        distributions, oracle = _three_label_case()
        foreign = [p.model_copy(update={"evaluator": "someone_else"}) for p in oracle]

        with pytest.raises(LabelCalibrationError, match="没有任何可用的 oracle"):
            calibrate_label_mapping(
                distributions,
                foreign,
                SplitName.VALIDATION,
                oracle_evaluator=ORACLE,
            )

    def test_empty_distributions_are_rejected(self) -> None:
        _, oracle = _three_label_case()

        with pytest.raises(LabelCalibrationError, match="没有任何 RAGChecker 标签分布"):
            calibrate_label_mapping([], oracle, SplitName.VALIDATION)
