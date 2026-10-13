"""两份关系输入的配对比较。

钉住三件事：只在人工多数票唯一的组合上比；argmax 相同的两个评估器，概率排序
不同时 AUROC 能分出高下；两次交叉验证只在逐条对得上的时候才配对。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_ds.experiments.relation_input_comparison import (
    compare_prediction_runs,
    read_cv_predictions,
    relation_level_comparison,
)
from rag_ds.schemas import RelationPrediction


def _relations(evaluator: str, triples: list[tuple[float, float, float]]) -> list[RelationPrediction]:
    return [
        RelationPrediction(
            sample_id=f"s{n}", claim_id=f"s{n}-c1", doc_id=f"s{n}-d1",
            evaluator=evaluator, p_support=s, p_refute=r, p_unknown=u,
        )
        for n, (s, r, u) in enumerate(triples)
    ]


# 六个组合：两条支持、两条反驳、一条不确定，最后一条人工两票对半（不参与比较）。
ORACLE = _relations(
    "human",
    [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
     (0.5, 0.0, 0.5)],
)
# 两个评估器都把所有组合判成「不确定」（argmax 相同）；
# sharp 的支持/反驳概率排序与人工一致，flat 完全不区分。
SHARP = _relations(
    "sharp",
    [(0.4, 0.0, 0.6), (0.3, 0.0, 0.7), (0.0, 0.4, 0.6), (0.0, 0.3, 0.7), (0.05, 0.05, 0.9),
     (0.1, 0.0, 0.9)],
)
FLAT = _relations("flat", [(0.1, 0.1, 0.8)] * 6)


def _compare(**overrides) -> dict:
    arguments = {
        "oracle": ORACLE, "reference": SHARP, "rival": FLAT,
        "reference_name": "sharp", "rival_name": "flat", "n_resamples": 200, "seed": 1,
    }
    arguments.update(overrides)
    return relation_level_comparison(
        arguments.pop("oracle"), arguments.pop("reference"), arguments.pop("rival"), **arguments
    )


class TestRelationLevel:
    def test_only_pairs_with_a_unique_human_majority_are_scored(self) -> None:
        result = _compare()

        assert result["pair_count"] == 6
        assert result["pairs_with_unique_majority"] == 5
        assert result["majority_counts"] == {"support": 2, "refute": 2, "unknown": 1}

    def test_ranking_separates_evaluators_that_agree_on_the_argmax(self) -> None:
        result = _compare()

        # 两者都只答对「不确定」那一条，argmax 完全一致 ……
        assert result["evaluators"]["sharp"]["argmax_accuracy"] == pytest.approx(0.2)
        assert result["evaluators"]["flat"]["argmax_accuracy"] == pytest.approx(0.2)
        assert result["argmax_agreement"] == pytest.approx(1.0)
        # …… 但概率排序差得很远。
        by_relation = {c["positive_label"]: c for c in result["auroc_comparisons"]}
        assert by_relation["support"]["auroc_reference"] == pytest.approx(1.0)
        assert by_relation["support"]["auroc_rival"] == pytest.approx(0.5)
        assert by_relation["refute"]["observed_delta"] == pytest.approx(0.5)
        assert by_relation["support"]["reference"] == "sharp"
        assert by_relation["support"]["rival"] == "flat"

    def test_a_tied_argmax_counts_as_wrong(self) -> None:
        tied = _relations("tied", [(0.5, 0.5, 0.0)] * 6)

        result = _compare(rival=tied, rival_name="tied")

        assert result["evaluators"]["tied"]["argmax_accuracy"] == 0.0
        assert result["evaluators"]["tied"]["argmax_counts"]["tie"] == 5

    def test_mismatched_coverage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="无法配对"):
            _compare(rival=FLAT[:-1])

    def test_duplicate_pairs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="重复"):
            _compare(reference=[*SHARP, SHARP[0]])

    def test_same_name_for_both_evaluators_is_refused(self) -> None:
        with pytest.raises(ValueError, match="同名"):
            _compare(rival_name="sharp")

    def test_no_unique_majority_is_refused(self) -> None:
        undecided = _relations("human", [(0.5, 0.5, 0.0)] * 6)

        with pytest.raises(ValueError, match="多数票"):
            _compare(oracle=undecided)


GOLD = ["supported", "refuted", "insufficient", "conflicting"]


def _run(predicted: list[str], gold: list[str] = GOLD, method: str = "ds") -> dict:
    return {
        method: {(f"s{n}", f"s{n}-c1"): (p, g) for n, (p, g) in enumerate(zip(predicted, gold, strict=True))}
    }


class TestDecisionLevel:
    def test_same_method_is_paired_across_the_two_runs(self) -> None:
        strong = {**_run(GOLD), **_run(GOLD, method="only_here")}
        weak = _run(["supported", "supported", "supported", "supported"])

        (comparison,) = compare_prediction_runs(
            strong, weak, reference_name="llm", rival_name="nli", n_resamples=100, seed=1
        )

        assert (comparison.method_a, comparison.method_b) == ("ds@llm", "ds@nli")
        assert comparison.observed_a == pytest.approx(1.0)
        assert comparison.observed_delta > 0
        assert comparison.sample_count == 4

    def test_no_common_method_is_refused(self) -> None:
        with pytest.raises(ValueError, match="同名"):
            compare_prediction_runs(
                _run(GOLD, method="a"), _run(GOLD, method="b"),
                reference_name="llm", rival_name="nli",
            )

    def test_different_claims_are_refused(self) -> None:
        shorter = {"ds": dict(list(_run(GOLD)["ds"].items())[:3])}

        with pytest.raises(ValueError, match="无法配对"):
            compare_prediction_runs(_run(GOLD), shorter, reference_name="llm", rival_name="nli")

    def test_different_gold_labels_are_refused(self) -> None:
        other_gold = _run(GOLD, gold=["refuted", "refuted", "insufficient", "conflicting"])

        with pytest.raises(ValueError, match="金标准"):
            compare_prediction_runs(_run(GOLD), other_gold, reference_name="llm", rival_name="nli")


class TestReadCvPredictions:
    HEADER = "method,sample_id,claim_id,predicted_label,gold_label,correct\n"

    def test_rows_are_grouped_by_method(self, tmp_path: Path) -> None:
        path = tmp_path / "cv_predictions.csv"
        path.write_text(
            self.HEADER + "ds,s1,s1-c1,supported,refuted,0\nmajority_vote,s1,s1-c1,refuted,refuted,1\n",
            encoding="utf-8-sig",
        )

        assert read_cv_predictions(path) == {
            "ds": {("s1", "s1-c1"): ("supported", "refuted")},
            "majority_vote": {("s1", "s1-c1"): ("refuted", "refuted")},
        }

    def test_a_claim_listed_twice_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "cv_predictions.csv"
        path.write_text(self.HEADER + "ds,s1,s1-c1,supported,refuted,0\n" * 2, encoding="utf-8-sig")

        with pytest.raises(ValueError, match="重复"):
            read_cv_predictions(path)
