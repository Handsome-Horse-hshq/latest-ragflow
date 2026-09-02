"""把官方 CLIMATE-FEVER JSONL 转成项目的四类正式数据集。"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_ds.data_io import write_relation_predictions, write_samples
from rag_ds.dataset_manifest import (
    ArtifactDigest,
    DatasetManifest,
    DatasetSource,
    SplitArtifacts,
    file_sha256,
    write_dataset_manifest,
)
from rag_ds.schemas import (
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

__all__ = [
    "CLIMATE_FEVER_DOWNLOAD_URL",
    "ClimateFeverDataError",
    "build_climate_fever_dataset",
]

CLIMATE_FEVER_HOMEPAGE = "https://github.com/tdiggelm/climate-fever-dataset"
CLIMATE_FEVER_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/"
    "main/dataset/climate-fever.jsonl"
)
CLIMATE_FEVER_PAPER_URL = "https://arxiv.org/abs/2012.00614"
CLIMATE_FEVER_CITATION = (
    "Diggelmann et al. (2020), CLIMATE-FEVER: A Dataset for Verification of "
    "Real-World Climate Claims, arXiv:2012.00614"
)
LICENSE_NOTE = (
    "The official repository does not include an explicit dataset license. "
    "Use for research with the paper citation and comply with the licensing "
    "and attribution terms of the underlying English Wikipedia evidence."
)
ORACLE_WARNING = (
    "These relation predictions are derived from CLIMATE-FEVER human evidence "
    "votes. They are annotation-oracle inputs for pipeline validation only and "
    "must not be reported as model predictions or model performance."
)
QUESTION_TEMPLATE = (
    "Determine whether the following climate claim is supported, refuted, "
    "insufficiently evidenced, or disputed by the retrieved evidence: {claim}"
)


class ClimateFeverDataError(ValueError):
    """原始 CLIMATE-FEVER 文件不符合预期格式。"""


class SourceLabel(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    DISPUTED = "DISPUTED"


class EvidenceLabel(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


LABEL_MAPPING: dict[SourceLabel, EvidenceState] = {
    SourceLabel.SUPPORTS: EvidenceState.SUPPORTED,
    SourceLabel.REFUTES: EvidenceState.REFUTED,
    SourceLabel.NOT_ENOUGH_INFO: EvidenceState.INSUFFICIENT,
    SourceLabel.DISPUTED: EvidenceState.CONFLICTING,
}


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    evidence_label: EvidenceLabel
    article: str
    evidence: str
    entropy: float
    votes: tuple[EvidenceLabel | None, ...]


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    claim: str
    claim_label: SourceLabel
    evidences: tuple[SourceEvidence, ...]


def _load_source(path: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = SourceRecord.model_validate_json(line)
            except Exception as error:
                raise ClimateFeverDataError(
                    f"{path} 第 {line_number} 行无效：{error}"
                ) from error
            if record.claim_id in seen_ids:
                raise ClimateFeverDataError(
                    f"{path} 第 {line_number} 行重复 claim_id={record.claim_id!r}"
                )
            if not record.claim.strip() or not record.evidences:
                raise ClimateFeverDataError(
                    f"{path} 第 {line_number} 行 claim 或 evidences 为空"
                )
            if any(not evidence.evidence.strip() for evidence in record.evidences):
                raise ClimateFeverDataError(
                    f"{path} 第 {line_number} 行包含空 evidence 文本"
                )
            seen_ids.add(record.claim_id)
            records.append(record)
    if not records:
        raise ClimateFeverDataError(f"原始数据为空：{path}")
    return records


def _normalise_claim(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _select_splits(
    records: list[SourceRecord],
    seed: int,
    split_per_class: dict[str, int],
) -> dict[str, list[SourceRecord]]:
    needed = sum(split_per_class.values())
    by_label: dict[SourceLabel, list[SourceRecord]] = defaultdict(list)
    seen_claims: set[str] = set()
    for record in records:
        normalised = _normalise_claim(record.claim)
        if normalised in seen_claims:
            continue
        seen_claims.add(normalised)
        by_label[record.claim_label].append(record)

    rng = random.Random(seed)
    result: dict[str, list[SourceRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in SourceLabel:
        candidates = sorted(by_label[label], key=lambda item: int(item.claim_id))
        if len(candidates) < needed:
            raise ClimateFeverDataError(
                f"{label.value} 去重后只有 {len(candidates)} 条，无法抽取 {needed} 条"
            )
        rng.shuffle(candidates)
        selected = candidates[:needed]
        offset = 0
        for split in ("train", "validation", "test"):
            count = split_per_class[split]
            result[split].extend(selected[offset : offset + count])
            offset += count

    for split in result:
        result[split].sort(key=lambda item: int(item.claim_id))
    return result


def _sample_id(record: SourceRecord) -> str:
    return f"climate-fever-{int(record.claim_id):04d}"


def _to_sample(record: SourceRecord) -> RAGSample:
    sample_id = _sample_id(record)
    return RAGSample(
        sample_id=sample_id,
        question=QUESTION_TEMPLATE.format(claim=record.claim.strip()),
        answer=record.claim.strip(),
        reference_answer=None,
        claims=(Claim(claim_id=f"{sample_id}-c1", text=record.claim.strip()),),
        contexts=tuple(
            ContextChunk(
                doc_id=f"{sample_id}-d{index}",
                text=evidence.evidence.strip(),
                retrieval_score=None,
                reliability=1.0,
            )
            for index, evidence in enumerate(record.evidences, start=1)
        ),
        gold_state=LABEL_MAPPING[record.claim_label],
    )


def _vote_probabilities(evidence: SourceEvidence) -> tuple[float, float, float]:
    votes = [vote for vote in evidence.votes if vote is not None]
    if not votes:
        votes = [evidence.evidence_label]
    counts = Counter(votes)
    total = len(votes)
    return (
        counts[EvidenceLabel.SUPPORTS] / total,
        counts[EvidenceLabel.REFUTES] / total,
        counts[EvidenceLabel.NOT_ENOUGH_INFO] / total,
    )


def _to_predictions(record: SourceRecord) -> tuple[RelationPrediction, ...]:
    sample_id = _sample_id(record)
    predictions: list[RelationPrediction] = []
    for index, evidence in enumerate(record.evidences, start=1):
        support, refute, unknown = _vote_probabilities(evidence)
        predictions.append(
            RelationPrediction(
                sample_id=sample_id,
                claim_id=f"{sample_id}-c1",
                doc_id=f"{sample_id}-d{index}",
                evaluator="climate_fever_human_vote_distribution",
                p_support=support,
                p_refute=refute,
                p_unknown=unknown,
                evaluator_reliability=1.0,
            )
        )
    return tuple(predictions)


def _provenance(record: SourceRecord, split: str) -> dict[str, Any]:
    return {
        "sample_id": _sample_id(record),
        "split": split,
        "source_dataset": "CLIMATE-FEVER",
        "source_claim_id": record.claim_id,
        "source_claim_label": record.claim_label.value,
        "mapped_gold_state": LABEL_MAPPING[record.claim_label].value,
        "source_homepage": CLIMATE_FEVER_HOMEPAGE,
        "contexts": [
            {
                "doc_id": f"{_sample_id(record)}-d{index}",
                "source_evidence_id": evidence.evidence_id,
                "article": evidence.article,
                "evidence_label": evidence.evidence_label.value,
                "entropy": evidence.entropy,
                "votes": [None if vote is None else vote.value for vote in evidence.votes],
            }
            for index, evidence in enumerate(record.evidences, start=1)
        ],
    }


def _write_dict_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _artifact(path: Path, root: Path, records: int) -> ArtifactDigest:
    return ArtifactDigest(
        path=path.relative_to(root).as_posix(),
        sha256=file_sha256(path),
        records=records,
    )


def build_climate_fever_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    train_per_class: int = 60,
    validation_per_class: int = 20,
    test_per_class: int = 20,
    overwrite: bool = False,
) -> DatasetManifest:
    """构建平衡、分层、带摘要和 provenance 的 CLIMATE-FEVER 子集。"""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    split_per_class = {
        "train": train_per_class,
        "validation": validation_per_class,
        "test": test_per_class,
    }
    if any(count < 0 for count in split_per_class.values()):
        raise ValueError("各 split 的每类数量不能为负数")
    per_class = sum(split_per_class.values())
    if per_class < 1:
        raise ValueError("每类总数量必须大于 0")

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    targets = [manifest_path]
    for split in split_per_class:
        targets.extend(
            [
                root / f"{split}.jsonl",
                root / f"{split}_relations.jsonl",
                root / f"{split}_provenance.jsonl",
            ]
        )
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"输出文件已存在：{existing[0]}")

    records = _load_source(source)
    splits = _select_splits(records, seed, split_per_class)
    artifacts: dict[str, SplitArtifacts] = {}
    for split, selected in splits.items():
        samples = [_to_sample(record) for record in selected]
        predictions = [
            prediction
            for record in selected
            for prediction in _to_predictions(record)
        ]
        provenance = [_provenance(record, split) for record in selected]
        sample_path = root / f"{split}.jsonl"
        prediction_path = root / f"{split}_relations.jsonl"
        provenance_path = root / f"{split}_provenance.jsonl"
        write_samples(sample_path, samples, overwrite=overwrite)
        write_relation_predictions(
            prediction_path, predictions, overwrite=overwrite
        )
        _write_dict_jsonl(provenance_path, provenance, overwrite=overwrite)
        artifacts[split] = SplitArtifacts(
            samples=_artifact(sample_path, root, len(samples)),
            relation_predictions=_artifact(
                prediction_path, root, len(predictions)
            ),
            provenance=_artifact(provenance_path, root, len(provenance)),
            label_counts=dict(Counter(sample.gold_state for sample in samples)),
        )

    manifest = DatasetManifest(
        dataset_name="rag-ds-climate-fever-balanced-v1",
        source=DatasetSource(
            name="CLIMATE-FEVER",
            homepage=CLIMATE_FEVER_HOMEPAGE,
            download_url=CLIMATE_FEVER_DOWNLOAD_URL,
            paper_url=CLIMATE_FEVER_PAPER_URL,
            citation=CLIMATE_FEVER_CITATION,
            license_note=LICENSE_NOTE,
            raw_file=source.name,
            raw_sha256=file_sha256(source),
            raw_records=len(records),
        ),
        random_seed=seed,
        per_class=per_class,
        split_per_class=split_per_class,
        label_mapping={key.value: value for key, value in LABEL_MAPPING.items()},
        relation_predictions_warning=ORACLE_WARNING,
        splits=artifacts,
    )
    write_dataset_manifest(manifest_path, manifest, overwrite=overwrite)
    return manifest
