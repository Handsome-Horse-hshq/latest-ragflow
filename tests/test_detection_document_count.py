"""「证据篇数」混杂因素对照的测试。

K_doc 随文档数累积变大。冲突类 claim 恰好配了更多证据的数据集上，它能靠
「数篇数」得分 —— 这组测试钉住：篇数本身进了对照表，且分层比较确实把它
控制掉了。
"""

from __future__ import annotations

import pytest

from rag_ds.experiments.detection_proxies import (
    CONFLICT_PROXIES,
    compare_detection,
    conflict_auroc_by_document_count,
    detection_scores,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction

NEUTRAL = (0.05, 0.05, 0.90)
MILD = (0.30, 0.25, 0.45)  # 每篇文档都略带一点支持与反驳


def _claim(index: int, gold: EvidenceState, triples: list[tuple[float, float, float]]):
    sample_id = f"s{index:03d}"
    sample = RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=tuple(
            ContextChunk(doc_id=f"{sample_id}-d{n}", text=f"文档 {n}。")
            for n in range(1, len(triples) + 1)
        ),
        gold_state=gold,
    )
    predictions = [
        RelationPrediction(
            sample_id=sample_id,
            claim_id=f"{sample_id}-c1",
            doc_id=f"{sample_id}-d{n}",
            evaluator="e1",
            p_support=s,
            p_refute=r,
            p_unknown=u,
        )
        for n, (s, r, u) in enumerate(triples, start=1)
    ]
    return sample, predictions


def _dataset(conflicting_docs: int, other_docs: int, per_class: int = 4):
    """冲突类与其余类的每篇文档完全一样，只是篇数不同。"""
    classes = (
        (EvidenceState.CONFLICTING, conflicting_docs),
        (EvidenceState.SUPPORTED, other_docs),
        (EvidenceState.INSUFFICIENT, other_docs),
    )
    samples, predictions = [], []
    for index in range(per_class):
        for offset, (gold, docs) in enumerate(classes):
            sample, preds = _claim(index * len(classes) + offset, gold, [MILD] * docs)
            samples.append(sample)
            predictions.extend(preds)
    return samples, predictions


class TestDocumentCountProxy:
    def test_count_is_recorded_and_listed_as_a_conflict_control(self) -> None:
        samples, predictions = _dataset(conflicting_docs=5, other_docs=2)

        rows = detection_scores(samples, predictions)

        assert CONFLICT_PROXIES[-1] == "document_count"
        assert {r.document_count for r in rows} == {2, 5}
        assert all(r.conflict["document_count"] == r.document_count for r in rows)

    def test_k_doc_separates_claims_that_differ_only_in_document_count(self) -> None:
        """每篇文档一模一样，K_doc 仍然把篇数多的排在前面：这正是要防的假象。"""
        samples, predictions = _dataset(conflicting_docs=5, other_docs=2)

        result = compare_detection(detection_scores(samples, predictions), n_resamples=50, seed=1)
        reports = {r.score_name: r for r in result["conflict"].reports}

        assert reports["k_doc"].auroc == pytest.approx(1.0)
        assert reports["document_count"].auroc == pytest.approx(1.0)
        # 不随篇数累积的统计量看不出差别。
        assert reports["sr_std"].auroc == pytest.approx(0.5)
        assert reports["max_product"].auroc == pytest.approx(0.5)

    def test_constant_document_count_is_uninformative(self) -> None:
        samples, predictions = _dataset(conflicting_docs=3, other_docs=3)

        result = compare_detection(detection_scores(samples, predictions), n_resamples=50, seed=1)
        reports = {r.score_name: r for r in result["conflict"].reports}

        assert reports["document_count"].auroc == pytest.approx(0.5)


class TestStratifiedComparison:
    def test_no_table_when_every_claim_has_the_same_count(self) -> None:
        samples, predictions = _dataset(conflicting_docs=3, other_docs=3)

        assert conflict_auroc_by_document_count(detection_scores(samples, predictions)) == []

    def test_within_a_count_group_only_the_scores_themselves_matter(self) -> None:
        samples, predictions = [], []
        index = 0
        for docs in (2, 3):
            for _ in range(3):
                # 冲突：一篇支持一篇反驳；非冲突：全部说不清。篇数在组内相同。
                conflicting = [(0.8, 0.1, 0.1), (0.1, 0.8, 0.1)] + [NEUTRAL] * (docs - 2)
                for gold, triples in (
                    (EvidenceState.CONFLICTING, conflicting),
                    (EvidenceState.INSUFFICIENT, [NEUTRAL] * docs),
                ):
                    sample, preds = _claim(index, gold, triples)
                    samples.append(sample)
                    predictions.extend(preds)
                    index += 1

        table = conflict_auroc_by_document_count(
            detection_scores(samples, predictions), min_claims=6, min_per_class=3
        )

        assert [row["document_count"] for row in table] == [2, 3]
        for row in table:
            assert (row["claims"], row["conflicting"]) == (6, 3)
            assert row["k_doc"] == pytest.approx(1.0)
            assert row["sr_std"] == pytest.approx(1.0)
            assert row["max_product"] == pytest.approx(1.0)

    def test_small_or_one_sided_groups_are_skipped(self) -> None:
        samples, predictions = _dataset(conflicting_docs=5, other_docs=2)

        # 每个篇数组里只有一类，AUROC 没有定义。
        assert (
            conflict_auroc_by_document_count(
                detection_scores(samples, predictions), min_claims=1, min_per_class=1
            )
            == []
        )
