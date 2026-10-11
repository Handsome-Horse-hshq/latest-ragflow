"""AVeriTeC 数据构建测试：问答对转文档、均衡抽样、以及「没有 oracle」的清单。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_ds.data_io import load_samples
from rag_ds.dataset_manifest import (
    DatasetManifest,
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
    verify_split_artifacts,
)
from rag_ds.datasets import AveritecDataError, build_averitec_dataset
from rag_ds.schemas import EvidenceState

LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


def _answer(text: str, answer_type: str = "Extractive", **extra) -> dict:
    return {
        "answer": text,
        "answer_type": answer_type,
        "source_url": "https://example.org/page",
        "source_medium": "Web text",
        **extra,
    }


def _record(claim: str, label: str, questions: list[dict] | None = None) -> dict:
    return {
        "claim": claim,
        "label": label,
        "justification": "why",
        "questions": questions
        or [{"question": f"Is it true that {claim}?", "answers": [_answer("Some answer.")]}],
    }


def _write_raw(directory: Path, train: list[dict], dev: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (directory / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
    return directory


def _balanced(per_class: int, prefix: str) -> list[dict]:
    return [
        _record(f"{prefix} claim {label} {index}", label)
        for label in LABELS
        for index in range(per_class)
    ]


def _build(tmp_path: Path, name: str = "out", **kwargs) -> tuple[DatasetManifest, Path]:
    raw = _write_raw(tmp_path / "averitec", _balanced(3, "train"), _balanced(1, "dev"))
    out = tmp_path / name
    manifest = build_averitec_dataset(
        raw, out, seed=7, train_per_class=2, validation_per_class=1, test_per_class=1, **kwargs
    )
    return manifest, out


def test_build_is_balanced_disjoint_and_reproducible(tmp_path: Path) -> None:
    manifest, first = _build(tmp_path, "first")
    _, second = _build(tmp_path, "second")

    seen: set[str] = set()
    for split, expected in (("train", 8), ("validation", 4), ("test", 4)):
        samples = load_samples(first / f"{split}.jsonl")
        assert len(samples) == expected
        assert {s.gold_state for s in samples} == set(EvidenceState)
        assert manifest.splits[split].label_counts == {
            state: expected // 4 for state in EvidenceState
        }
        ids = {s.sample_id for s in samples}
        assert not (ids & seen)
        seen |= ids
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_manifest_registers_no_oracle_relations(tmp_path: Path) -> None:
    manifest, out = _build(tmp_path)

    loaded = load_dataset_manifest(out / "manifest.json")

    assert loaded == manifest
    assert loaded.relation_predictions_kind == "none"
    assert all(split.relation_predictions is None for split in loaded.splits.values())
    assert not list(out.glob("*_relations.jsonl"))
    # 原始数据是两个文件：train 记在主字段，dev 记在 extra_raw_files。
    assert loaded.source.raw_file == "averitec/train.json"
    assert loaded.source.raw_records == 12
    assert [(f.path, f.records) for f in loaded.source.extra_raw_files] == [
        ("averitec/dev.json", 4)
    ]
    # 样本文件仍然可以按清单校验。
    verify_artifact(out, loaded.splits["test"].samples)


def test_oracle_verification_is_refused_for_a_dataset_without_oracle(tmp_path: Path) -> None:
    _, out = _build(tmp_path)

    with pytest.raises(DatasetManifestError, match="没有登记 annotation-oracle"):
        verify_split_artifacts(
            out / "manifest.json", out / "test.jsonl", out / "test.jsonl", "test"
        )


def test_question_answer_pairs_become_documents(tmp_path: Path) -> None:
    special = _record(
        "The special claim",
        "Supported",
        [
            {
                "question": "Did it happen?",
                "answers": [
                    _answer("Yes", "Boolean", boolean_explanation="Two sources confirm it."),
                    _answer("No answer could be found.", "Unanswerable"),
                ],
            },
            {"question": "  ", "answers": [_answer("An answer without a question.")]},
        ],
    )
    train = [special] + _balanced(3, "train")[1:]
    raw = _write_raw(tmp_path / "averitec", train, _balanced(1, "dev"))
    out = tmp_path / "out"

    build_averitec_dataset(
        raw, out, seed=7, train_per_class=2, validation_per_class=1, test_per_class=1
    )
    samples = [s for split in ("train", "validation", "test") for s in load_samples(out / f"{split}.jsonl")]
    (sample,) = [s for s in samples if s.answer == "The special claim"]

    assert sample.sample_id == "averitec-train-0000"
    assert [c.text for c in sample.contexts] == [
        "Question: Did it happen? Answer: Yes. Two sources confirm it.",
        # 「查过但没查到」保留为文档：它本身就是证据不足的证据。
        "Question: Did it happen? Answer: No answer could be found.",
        "An answer without a question.",
    ]
    assert [c.doc_id for c in sample.contexts] == [
        "averitec-train-0000-d1",
        "averitec-train-0000-d2",
        "averitec-train-0000-d3",
    ]
    assert sample.claims[0].text == sample.answer
    assert all(c.reliability == 1.0 for c in sample.contexts)


def test_empty_claims_are_skipped_but_counted_as_raw(tmp_path: Path) -> None:
    train = _balanced(3, "train") + [_record("   ", "Supported")]
    raw = _write_raw(tmp_path / "averitec", train, _balanced(1, "dev"))

    manifest = build_averitec_dataset(
        raw, tmp_path / "out", seed=7, train_per_class=2, validation_per_class=1, test_per_class=1
    )

    assert manifest.source.raw_records == 13
    samples = [
        s for split in ("train", "validation", "test")
        for s in load_samples(tmp_path / "out" / f"{split}.jsonl")
    ]
    assert all(s.answer.strip() for s in samples)


def test_duplicate_claims_are_used_once(tmp_path: Path) -> None:
    train = _balanced(3, "train")
    dev = _balanced(1, "dev") + [_record("TRAIN  claim supported 0", "Refuted")]
    raw = _write_raw(tmp_path / "averitec", train, dev)

    build_averitec_dataset(
        raw, tmp_path / "out", seed=7, train_per_class=2, validation_per_class=1, test_per_class=1
    )
    samples = [
        s for split in ("train", "validation", "test")
        for s in load_samples(tmp_path / "out" / f"{split}.jsonl")
    ]

    # 大小写与空白不同的重复 claim 只保留先出现的那条（train 里标为 Supported）。
    matching = [s for s in samples if s.answer.casefold().split() == "train claim supported 0".split()]
    assert len(matching) <= 1
    assert all(s.gold_state is EvidenceState.SUPPORTED for s in matching)


def test_unknown_label_is_rejected(tmp_path: Path) -> None:
    raw = _write_raw(
        tmp_path / "averitec", _balanced(3, "train") + [_record("odd", "Half True")], _balanced(1, "dev")
    )

    with pytest.raises(AveritecDataError, match="标签未知"):
        build_averitec_dataset(raw, tmp_path / "out")


def test_too_few_claims_in_a_class_is_rejected(tmp_path: Path) -> None:
    raw = _write_raw(tmp_path / "averitec", _balanced(3, "train"), _balanced(1, "dev"))

    with pytest.raises(AveritecDataError, match="无法抽取"):
        build_averitec_dataset(raw, tmp_path / "out", train_per_class=10)


def test_missing_raw_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "averitec").mkdir()

    with pytest.raises(FileNotFoundError):
        build_averitec_dataset(tmp_path / "averitec", tmp_path / "out")


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    _build(tmp_path)

    with pytest.raises(FileExistsError):
        _build(tmp_path)


class TestManifestOracleConsistency:
    def test_kind_none_with_registered_oracle_is_rejected(self, tmp_path: Path) -> None:
        manifest, _ = _build(tmp_path)
        payload = json.loads(manifest.model_dump_json(exclude_none=True))
        payload["splits"]["test"]["relation_predictions"] = payload["splits"]["test"]["samples"]

        with pytest.raises(ValueError, match="不应登记 oracle"):
            DatasetManifest.model_validate(payload)

    def test_annotation_oracle_requires_every_split_to_register_one(self, tmp_path: Path) -> None:
        manifest, _ = _build(tmp_path)
        payload = json.loads(manifest.model_dump_json(exclude_none=True))
        payload["relation_predictions_kind"] = "annotation_oracle"

        with pytest.raises(ValueError, match="都必须登记 oracle"):
            DatasetManifest.model_validate(payload)
