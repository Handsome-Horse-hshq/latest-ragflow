"""统一证据篇数（控制混杂因素）的测试。"""

from __future__ import annotations

import pytest

from rag_ds.experiments.data_loading import restrict_to_first_documents
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction


def _claim(sample_id: str, documents: int):
    sample = RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id=f"{sample_id}-c1", text="答案。"),),
        contexts=tuple(
            ContextChunk(doc_id=f"{sample_id}-d{n}", text=f"文档 {n}。")
            for n in range(1, documents + 1)
        ),
        gold_state=EvidenceState.SUPPORTED,
    )
    predictions = [
        RelationPrediction(
            sample_id=sample_id, claim_id=f"{sample_id}-c1", doc_id=f"{sample_id}-d{n}",
            evaluator="e1", p_support=0.6, p_refute=0.1, p_unknown=0.3,
        )
        for n in range(1, documents + 1)
    ]
    return sample, predictions


def _dataset(*document_counts: int):
    samples, predictions = [], []
    for index, documents in enumerate(document_counts):
        sample, preds = _claim(f"s{index}", documents)
        samples.append(sample)
        predictions.extend(preds)
    return samples, predictions


def test_short_claims_are_dropped_and_long_ones_truncated() -> None:
    samples, predictions = _dataset(1, 2, 4)

    kept, kept_predictions = restrict_to_first_documents(samples, predictions, 2)

    assert [s.sample_id for s in kept] == ["s1", "s2"]
    assert all(len(s.contexts) == 2 for s in kept)
    # 保留的是原始顺序里的前两篇，不做挑选。
    assert [c.doc_id for c in kept[1].contexts] == ["s2-d1", "s2-d2"]
    assert sorted((p.sample_id, p.doc_id) for p in kept_predictions) == [
        ("s1", "s1-d1"),
        ("s1", "s1-d2"),
        ("s2", "s2-d1"),
        ("s2", "s2-d2"),
    ]


def test_inputs_are_not_modified() -> None:
    samples, predictions = _dataset(4)

    restrict_to_first_documents(samples, predictions, 2)

    assert len(samples[0].contexts) == 4
    assert len(predictions) == 4


def test_gold_labels_and_claims_are_untouched() -> None:
    samples, predictions = _dataset(3)

    (kept,), _ = restrict_to_first_documents(samples, predictions, 2)

    assert kept.gold_state is samples[0].gold_state
    assert kept.claims == samples[0].claims


@pytest.mark.parametrize("count", [0, -1])
def test_non_positive_count_is_rejected(count: int) -> None:
    samples, predictions = _dataset(2)

    with pytest.raises(ValueError, match="正整数"):
        restrict_to_first_documents(samples, predictions, count)


def test_no_claim_reaching_the_count_is_an_error() -> None:
    samples, predictions = _dataset(1, 2)

    with pytest.raises(ValueError, match="达到 3 篇"):
        restrict_to_first_documents(samples, predictions, 3)
