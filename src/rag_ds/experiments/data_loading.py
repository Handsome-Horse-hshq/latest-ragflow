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

__all__ = ["ALL_SPLITS", "load_verified_dataset", "restrict_to_first_documents"]

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


def restrict_to_first_documents(
    samples: Sequence[RAGSample],
    predictions: Sequence[RelationPrediction],
    count: int,
) -> tuple[list[RAGSample], list[RelationPrediction]]:
    """把每条样本的证据篇数统一成 ``count``：不足的样本丢弃，多出的文档截掉。

    用途是**控制「证据篇数」这个混杂因素**。篇数不固定的数据集上，某一类 claim
    可能恰好配了更多证据（AVeriTeC 的冲突类平均 3.95 篇，其余 2.6–3.0 篇），
    会随篇数变大的分数就能靠「数篇数」得分。统一篇数之后，剩下的差别才属于
    分数本身。

    保留的是每条样本**原始顺序里的前** ``count`` 篇，不做任何挑选；关系预测
    同步只留这些文档的。

    Args:
        samples: 样本。
        predictions: 关系预测。
        count: 统一后的篇数，须为正整数。

    Returns:
        ``(保留并截断后的样本, 对应的关系预测)``。

    Raises:
        ValueError: ``count`` 不是正整数，或没有任何样本达到这个篇数。
    """
    if count < 1:
        raise ValueError(f"count 必须是正整数，收到 {count}")
    kept: list[RAGSample] = []
    kept_docs: set[tuple[str, str]] = set()
    for sample in samples:
        if len(sample.contexts) < count:
            continue
        contexts = list(sample.contexts[:count])
        kept.append(sample.model_copy(update={"contexts": tuple(contexts)}))
        kept_docs.update((sample.sample_id, chunk.doc_id) for chunk in contexts)
    if not kept:
        raise ValueError(f"没有任何样本的证据达到 {count} 篇")
    return kept, [p for p in predictions if (p.sample_id, p.doc_id) in kept_docs]
