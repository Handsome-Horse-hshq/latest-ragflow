"""模型产出的关系文件的谱系清单。

为什么需要单独一份清单
----------------------
:mod:`rag_ds.dataset_manifest` 登记的 ``*_relations.jsonl`` 是**人工标注
oracle**，``verify_split_artifacts`` 会要求关系文件恰好是清单里登记的那一份。
而 RAGChecker 跑出来的关系文件是**模型预测**，既不在数据集清单里，也不应该
被塞进去 —— 数据集是固定的，模型输出会随模型、随版本、随温度变化。

于是模型产出的关系文件配一份自己的 :class:`ModelRunManifest`，它同时钉住：

* 这批关系**属于哪个 split 的哪份样本**（samples 摘要，与数据集清单一致）；
* 关系文件本身的摘要与记录数；
* 产出它的**评估器与模型名**（extractor / checker），以及原始输出文件摘要；
* 换算用的**标签映射表**及其来源（验证集校准 / 占位默认值）。

``relation_predictions_kind`` 固定为 ``model_prediction``，与数据集清单里的
``annotation_oracle`` 形成对照：下游只要读这个字段，就能知道手上的数字能不能
当作模型性能报告。
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.dataset_manifest import (
    ArtifactDigest,
    DatasetManifestError,
    file_sha256,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.relation_evaluation.ragchecker_adapter import LabelProbabilityMapping
from rag_ds.schemas import NonEmptyStr

__all__ = [
    "LabelMappingSource",
    "ModelRunError",
    "ModelRunManifest",
    "SourceArtifact",
    "build_model_run_manifest",
    "load_model_run_manifest",
    "verify_model_run_artifacts",
    "write_model_run_manifest",
]

SplitKey = Literal["train", "validation", "test"]

#: 标签映射表的来源。
#:
#: * ``calibrated_on_validation`` —— 在验证集上用人工投票标定过；
#: * ``placeholder_default`` —— **没有**校准，用的是占位值；
#: * ``continuous_probabilities`` —— 根本没用映射表：关系概率直接取自评估器
#:   输出的连续置信度，不经过离散标签这一步。
LabelMappingSource = Literal[
    "calibrated_on_validation",
    "placeholder_default",
    "continuous_probabilities",
]


class ModelRunError(ValueError):
    """模型运行清单缺失、与输入文件不一致或摘要校验失败。"""


class SourceArtifact(BaseModel):
    """产出这批关系的原始文件（如 ``checking_outputs.json``）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: NonEmptyStr
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelRunManifest(BaseModel):
    """一次模型关系评估运行的完整谱系。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    #: 对应的数据集名称，须与数据集清单一致。
    dataset_name: NonEmptyStr
    split: SplitKey
    relation_predictions_kind: Literal["model_prediction"] = "model_prediction"

    #: 写进每条关系记录的评估器名。
    evaluator: NonEmptyStr
    #: 产出工具，例如 ``ragchecker``。
    producer: NonEmptyStr
    producer_version: str | None = None
    #: RAGChecker 的 claim 抽取模型；未使用时为 ``None``。
    extractor_name: str | None = None
    #: RAGChecker 的 entailment 判定模型。
    checker_name: NonEmptyStr

    #: 这批关系所属的样本文件，摘要须与数据集清单一致。
    samples: ArtifactDigest
    #: 关系文件本身。
    relation_predictions: ArtifactDigest
    #: 原始输出文件。
    source_output: SourceArtifact

    #: 标签 → 三元概率的映射表。
    label_mapping: LabelProbabilityMapping
    label_mapping_source: LabelMappingSource
    #: 校准结果文件的相对路径；占位默认值时为 ``None``。
    calibration_path: str | None = None
    #: 子 claim 条数不为 1 时的处理策略。
    multi_claim_policy: Literal["mixture", "strict"] = "mixture"
    #: 写进每条关系记录的评估器可靠性。
    evaluator_reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: NonEmptyStr

    @model_validator(mode="after")
    def _check_calibration_reference(self) -> ModelRunManifest:
        """校准来源与校准文件必须自洽。"""
        if self.label_mapping_source == "calibrated_on_validation":
            if not self.calibration_path:
                raise ValueError(
                    "label_mapping_source 为 calibrated_on_validation 时，"
                    "必须记录 calibration_path"
                )
        elif self.calibration_path:
            raise ValueError(
                f"label_mapping_source 为 {self.label_mapping_source} 时，"
                "不应记录 calibration_path"
            )
        return self

    @property
    def is_calibrated(self) -> bool:
        """关系概率是否来自可直接报告的来源。

        连续概率路径不经过映射表，也就无所谓「映射表有没有校准」——
        它没有任何未校准的占位参数，因此同样算作可报告。
        """
        return self.label_mapping_source in {
            "calibrated_on_validation",
            "continuous_probabilities",
        }


def _non_empty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def build_model_run_manifest(
    *,
    dataset_name: str,
    split: SplitKey,
    evaluator: str,
    producer: str,
    checker_name: str,
    samples: ArtifactDigest,
    predictions_path: str | Path,
    predictions_relative_path: str,
    source_output_path: str | Path,
    source_output_relative_path: str,
    label_mapping: LabelProbabilityMapping,
    label_mapping_source: LabelMappingSource,
    calibration_path: str | None = None,
    extractor_name: str | None = None,
    producer_version: str | None = None,
    multi_claim_policy: Literal["mixture", "strict"] = "mixture",
    evaluator_reliability: float = 1.0,
) -> ModelRunManifest:
    """就地计算摘要并装配 :class:`ModelRunManifest`。"""
    predictions = Path(predictions_path)
    return ModelRunManifest(
        dataset_name=dataset_name,
        split=split,
        evaluator=evaluator,
        producer=producer,
        producer_version=producer_version,
        extractor_name=extractor_name,
        checker_name=checker_name,
        samples=samples,
        relation_predictions=ArtifactDigest(
            path=predictions_relative_path,
            sha256=file_sha256(predictions),
            records=_non_empty_line_count(predictions),
        ),
        source_output=SourceArtifact(
            path=source_output_relative_path,
            sha256=file_sha256(source_output_path),
        ),
        label_mapping=label_mapping,
        label_mapping_source=label_mapping_source,
        calibration_path=calibration_path,
        multi_claim_policy=multi_claim_policy,
        evaluator_reliability=evaluator_reliability,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def load_model_run_manifest(path: str | Path) -> ModelRunManifest:
    """读取并校验模型运行清单。"""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ModelRunError(f"模型运行清单不存在：{manifest_path}")
    try:
        return ModelRunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise ModelRunError(f"模型运行清单无效：{manifest_path}：{error}") from error


def write_model_run_manifest(
    path: str | Path, manifest: ModelRunManifest, overwrite: bool = False
) -> None:
    """原子写入 UTF-8 JSON 清单。"""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"模型运行清单已存在：{target}")
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


def verify_model_run_artifacts(
    model_run_path: str | Path,
    dataset_manifest_path: str | Path,
    samples_path: str | Path,
    predictions_path: str | Path,
    split_name: SplitKey,
) -> ModelRunManifest:
    """确认样本来自数据集清单、关系文件来自这次模型运行，且两者摘要都未变。

    与 :func:`~rag_ds.dataset_manifest.verify_split_artifacts` 的分工：
    那一个要求关系文件**就是**清单登记的 oracle；这一个允许关系文件是模型
    产出，但仍然钉死「样本必须是登记的那一份」。

    Raises:
        ModelRunError: split 不符、样本与数据集清单不一致，或任一摘要变化。
        DatasetManifestError: 数据集清单本身有问题。
    """
    run = load_model_run_manifest(model_run_path)
    if run.split != split_name:
        raise ModelRunError(
            f"模型运行清单记录的 split 是 {run.split!r}，请求的是 {split_name!r}"
        )

    dataset_file = Path(dataset_manifest_path).resolve()
    dataset = load_dataset_manifest(dataset_file)
    if dataset.dataset_name != run.dataset_name:
        raise ModelRunError(
            f"模型运行清单属于数据集 {run.dataset_name!r}，"
            f"当前数据集清单是 {dataset.dataset_name!r}"
        )

    registered = dataset.splits[split_name].samples
    if (
        run.samples.sha256 != registered.sha256
        or run.samples.records != registered.records
    ):
        raise ModelRunError(
            f"模型运行清单登记的 {split_name} 样本与数据集清单不一致："
            f"run={run.samples.sha256[:12]}…/{run.samples.records} 条，"
            f"dataset={registered.sha256[:12]}…/{registered.records} 条；"
            "样本文件被改过，之前跑出的关系不再对应当前数据"
        )

    registered_samples = verify_artifact(dataset_file.parent, registered)
    requested_samples = Path(samples_path).resolve()
    if requested_samples != registered_samples:
        raise ModelRunError(
            f"--samples 不是清单登记的 {split_name} 样本："
            f"{requested_samples} != {registered_samples}"
        )

    run_dir = Path(model_run_path).resolve().parent
    try:
        registered_predictions = verify_artifact(run_dir, run.relation_predictions)
    except DatasetManifestError as error:
        raise ModelRunError(f"模型关系文件校验失败：{error}") from error
    requested_predictions = Path(predictions_path).resolve()
    if requested_predictions != registered_predictions:
        raise ModelRunError(
            "--predictions 不是这次模型运行产出的关系文件："
            f"{requested_predictions} != {registered_predictions}"
        )
    return run
