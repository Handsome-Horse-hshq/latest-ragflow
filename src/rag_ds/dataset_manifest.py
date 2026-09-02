"""可复现实验数据的清单、摘要校验与验证集身份检查。"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from rag_ds.schemas import EvidenceState, NonEmptyStr

__all__ = [
    "ArtifactDigest",
    "DatasetManifest",
    "DatasetManifestError",
    "DatasetSource",
    "SplitArtifacts",
    "file_sha256",
    "load_dataset_manifest",
    "verify_split_artifacts",
    "verify_validation_artifacts",
    "write_dataset_manifest",
]

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SplitKey = Literal["train", "validation", "test"]


class DatasetManifestError(ValueError):
    """数据清单缺失、与输入文件不一致或摘要校验失败。"""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactDigest(_FrozenModel):
    """一个 JSONL 实验文件的相对路径、记录数与内容摘要。"""

    path: NonEmptyStr
    sha256: Sha256
    records: int = Field(ge=0)


class SplitArtifacts(_FrozenModel):
    """一个数据划分的样本、annotation-oracle 关系和类别分布。"""

    samples: ArtifactDigest
    relation_predictions: ArtifactDigest
    provenance: ArtifactDigest
    label_counts: dict[EvidenceState, int]

    @model_validator(mode="after")
    def _check_counts(self) -> SplitArtifacts:
        if any(count < 0 for count in self.label_counts.values()):
            raise ValueError("label_counts 不能包含负数")
        if sum(self.label_counts.values()) != self.samples.records:
            raise ValueError("label_counts 之和必须等于 samples.records")
        if self.provenance.records != self.samples.records:
            raise ValueError("provenance.records 必须等于 samples.records")
        return self


class DatasetSource(_FrozenModel):
    """原始数据源及其固定内容摘要。"""

    name: NonEmptyStr
    homepage: NonEmptyStr
    download_url: NonEmptyStr
    paper_url: NonEmptyStr
    citation: NonEmptyStr
    license_note: NonEmptyStr
    raw_file: NonEmptyStr
    raw_sha256: Sha256
    raw_records: int = Field(ge=1)


class DatasetManifest(_FrozenModel):
    """正式数据集的版本化构建清单。"""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: NonEmptyStr
    source: DatasetSource
    random_seed: int
    per_class: int = Field(ge=1)
    split_per_class: dict[SplitKey, int]
    label_mapping: dict[NonEmptyStr, EvidenceState]
    relation_predictions_kind: Literal["annotation_oracle"] = "annotation_oracle"
    relation_predictions_warning: NonEmptyStr
    splits: dict[SplitKey, SplitArtifacts]

    @model_validator(mode="after")
    def _check_splits(self) -> DatasetManifest:
        expected = {"train", "validation", "test"}
        if set(self.splits) != expected or set(self.split_per_class) != expected:
            raise ValueError("splits 与 split_per_class 必须恰好包含 train/validation/test")
        if sum(self.split_per_class.values()) != self.per_class:
            raise ValueError("split_per_class 之和必须等于 per_class")
        if any(count < 0 for count in self.split_per_class.values()):
            raise ValueError("split_per_class 不能包含负数")
        return self


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _non_empty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """读取并校验数据清单。"""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise DatasetManifestError(f"数据清单不存在：{manifest_path}")
    try:
        return DatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise DatasetManifestError(f"数据清单无效：{manifest_path}：{error}") from error


def _verify_artifact(manifest_dir: Path, artifact: ArtifactDigest) -> Path:
    path = (manifest_dir / artifact.path).resolve()
    if not path.is_file():
        raise DatasetManifestError(f"清单中的文件不存在：{path}")
    actual_hash = file_sha256(path)
    if actual_hash != artifact.sha256:
        raise DatasetManifestError(
            f"文件 SHA-256 与清单不一致：{path}；"
            f"expected={artifact.sha256}, actual={actual_hash}"
        )
    actual_records = _non_empty_line_count(path)
    if actual_records != artifact.records:
        raise DatasetManifestError(
            f"文件记录数与清单不一致：{path}；"
            f"expected={artifact.records}, actual={actual_records}"
        )
    return path


def verify_split_artifacts(
    manifest_path: str | Path,
    samples_path: str | Path,
    predictions_path: str | Path,
    split_name: SplitKey,
) -> DatasetManifest:
    """确认输入正是清单登记且摘要未变的指定 split 文件。"""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_dataset_manifest(manifest_file)
    split = manifest.splits[split_name]
    registered_samples = _verify_artifact(manifest_file.parent, split.samples)
    registered_predictions = _verify_artifact(
        manifest_file.parent, split.relation_predictions
    )
    requested_samples = Path(samples_path).resolve()
    requested_predictions = Path(predictions_path).resolve()
    if requested_samples != registered_samples:
        raise DatasetManifestError(
            f"--samples 不是清单登记的 {split_name} 样本："
            f"{requested_samples} != {registered_samples}"
        )
    if requested_predictions != registered_predictions:
        raise DatasetManifestError(
            f"--predictions 不是清单登记的 {split_name} 关系文件："
            f"{requested_predictions} != {registered_predictions}"
        )
    return manifest


def verify_validation_artifacts(
    manifest_path: str | Path,
    samples_path: str | Path,
    predictions_path: str | Path,
) -> DatasetManifest:
    """确认调参输入正是清单登记且摘要未变的 validation 文件。"""
    return verify_split_artifacts(
        manifest_path, samples_path, predictions_path, "validation"
    )


def write_dataset_manifest(
    path: str | Path, manifest: DatasetManifest, overwrite: bool = False
) -> None:
    """原子写入 UTF-8 JSON 清单。"""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"数据清单已存在：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(manifest.model_dump_json(indent=2))
            handle.write("\n")
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
