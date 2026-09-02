"""真实数据构建、数据谱系和 validation 身份保护测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    file_sha256,
    load_dataset_manifest,
    verify_split_artifacts,
    verify_validation_artifacts,
)
from rag_ds.datasets import build_climate_fever_dataset
from rag_ds.schemas import EvidenceState

LIVE_DATASET = Path(__file__).resolve().parents[1] / "data" / "processed" / "climate_fever_v1"


def _source_record(claim_id: int, label: str) -> dict:
    evidence_label = {
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "DISPUTED": "SUPPORTS" if claim_id % 2 == 0 else "REFUTES",
    }[label]
    opposite = "REFUTES" if evidence_label == "SUPPORTS" else "SUPPORTS"
    return {
        "claim_id": str(claim_id),
        "claim": f"Climate claim {claim_id}",
        "claim_label": label,
        "evidences": [
            {
                "evidence_id": f"Article:{claim_id}",
                "evidence_label": evidence_label,
                "article": "Article",
                "evidence": f"Evidence for claim {claim_id}.",
                "entropy": 0.0,
                "votes": [evidence_label, opposite, None, None, None],
            }
        ],
    }


def _write_source(path: Path) -> None:
    labels = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")
    rows = [
        _source_record(index * 10 + offset, label)
        for index, label in enumerate(labels)
        for offset in range(3)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_build_is_balanced_disjoint_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_source(source)

    manifest = build_climate_fever_dataset(
        source,
        first,
        seed=7,
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )
    build_climate_fever_dataset(
        source,
        second,
        seed=7,
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )

    all_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        samples = load_samples(first / f"{split}.jsonl")
        predictions = load_relation_predictions(first / f"{split}_relations.jsonl")
        assert len(samples) == 4
        assert len(predictions) == 4
        assert {sample.gold_state for sample in samples} == set(EvidenceState)
        split_ids = {sample.sample_id for sample in samples}
        assert all_ids.isdisjoint(split_ids)
        all_ids.update(split_ids)
        assert (first / f"{split}.jsonl").read_bytes() == (
            second / f"{split}.jsonl"
        ).read_bytes()

    assert manifest.relation_predictions_kind == "annotation_oracle"
    assert "must not be reported as model" in manifest.relation_predictions_warning
    assert load_dataset_manifest(first / "manifest.json") == manifest


def test_validation_verification_rejects_wrong_or_modified_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "dataset"
    _write_source(source)
    build_climate_fever_dataset(
        source,
        output,
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
    )

    manifest = output / "manifest.json"
    validation = output / "validation.jsonl"
    relations = output / "validation_relations.jsonl"
    verify_validation_artifacts(manifest, validation, relations)
    verify_split_artifacts(
        manifest,
        output / "test.jsonl",
        output / "test_relations.jsonl",
        "test",
    )

    with pytest.raises(DatasetManifestError, match="不是清单登记"):
        verify_validation_artifacts(
            manifest, output / "train.jsonl", relations
        )

    validation.write_text(
        validation.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetManifestError, match="SHA-256"):
        verify_validation_artifacts(manifest, validation, relations)


@pytest.mark.skipif(
    not (LIVE_DATASET / "manifest.json").is_file(),
    reason="live climate_fever_v1 is not present",
)
def test_live_climate_fever_v1_matches_manifest() -> None:
    """仓库内 400 条正式数据与清单一致：平衡、互斥、摘要未改。"""
    manifest_path = LIVE_DATASET / "manifest.json"
    manifest = load_dataset_manifest(manifest_path)
    assert manifest.dataset_name == "rag-ds-climate-fever-balanced-v1"
    assert manifest.per_class == 100
    assert manifest.split_per_class == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert manifest.relation_predictions_kind == "annotation_oracle"
    assert "must not be reported as model" in manifest.relation_predictions_warning

    all_ids: set[str] = set()
    all_claims: set[str] = set()
    for split, per_class in manifest.split_per_class.items():
        verify_split_artifacts(
            manifest_path,
            LIVE_DATASET / f"{split}.jsonl",
            LIVE_DATASET / f"{split}_relations.jsonl",
            split,
        )
        samples = load_samples(LIVE_DATASET / f"{split}.jsonl")
        predictions = load_relation_predictions(
            LIVE_DATASET / f"{split}_relations.jsonl"
        )
        provenance = LIVE_DATASET / f"{split}_provenance.jsonl"
        split_info = manifest.splits[split]
        assert len(samples) == split_info.samples.records
        assert len(predictions) == split_info.relation_predictions.records
        assert file_sha256(provenance) == split_info.provenance.sha256
        assert {sample.gold_state for sample in samples} == set(EvidenceState)
        for state in EvidenceState:
            assert split_info.label_counts[state] == per_class
        assert {len(sample.contexts) for sample in samples} == {5}
        ids = {sample.sample_id for sample in samples}
        claims = {sample.answer.strip().casefold() for sample in samples}
        assert all_ids.isdisjoint(ids)
        assert all_claims.isdisjoint(claims)
        all_ids.update(ids)
        all_claims.update(claims)

    assert len(all_ids) == 400
    assert len(all_claims) == 400
