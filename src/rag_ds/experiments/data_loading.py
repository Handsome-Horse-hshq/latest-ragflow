"""把一个数据集的全部 split 连同模型关系文件一起加载，并逐个校验谱系。

交叉验证、嵌套消融与检测实验都要「全体样本 + 全体关系」，而且都必须确认
读进来的正是清单登记、摘要未变的那几份文件。校验逻辑集中在这里，几个脚本
对「什么算合法输入」的理解就不会分叉。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import DatasetManifest, load_dataset_manifest, verify_artifact
from rag_ds.model_runs import verify_model_run_artifacts
from rag_ds.schemas import RAGSample, RelationPrediction

__all__ = ["ALL_SPLITS", "load_verified_dataset"]

ALL_SPLITS: tuple[str, ...] = ("train", "validation", "test")


def load_verified_dataset(
    manifest_path: str | Path,
    relations_template: str,
    evaluators: Sequence[str],
    model_run_template: str | None = None,
    splits: Sequence[str] = ALL_SPLITS,
) -> tuple[DatasetManifest, list[RAGSample], list[RelationPrediction]]:
    """按清单加载各 split 的样本与各评估器的关系预测。

    Args:
        manifest_path: 数据集清单路径。
        relations_template: 关系文件路径模板，可用 ``{split}`` 与 ``{evaluator}``。
        evaluators: 要汇总的评估器名。
        model_run_template: 谱系文件路径模板；给出时逐个校验身份与摘要。
        splits: 要加载的 split。

    Returns:
        ``(清单, 全部样本, 全部关系预测)``。

    Raises:
        DatasetManifestError: 清单无效，或样本文件摘要与清单不符。
        ModelRunError: 关系文件与谱系记录不符。
        OSError: 文件读取失败。
    """
    manifest_file = Path(manifest_path).resolve()
    manifest = load_dataset_manifest(manifest_file)
    samples: list[RAGSample] = []
    predictions: list[RelationPrediction] = []
    for split in splits:
        samples_path = verify_artifact(
            manifest_file.parent, manifest.splits[split].samples
        )
        samples.extend(load_samples(samples_path))
        for evaluator in evaluators:
            relations = Path(relations_template.format(split=split, evaluator=evaluator))
            if model_run_template:
                run_path = Path(
                    model_run_template.format(split=split, evaluator=evaluator)
                )
                verify_model_run_artifacts(
                    run_path, manifest_file, samples_path, relations, split
                )
            predictions.extend(load_relation_predictions(relations))
    return manifest, samples, predictions
