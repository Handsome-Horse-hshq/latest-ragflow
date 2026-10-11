"""把官方 AVeriTeC（train + dev）转成项目的四类正式数据集。

为什么需要第二个数据集
----------------------
CLIMATE-FEVER 的 claim 标签是由逐条证据的多数票**按规则合成**的（至少一篇
支持且其余不确定 → 支持；全部不确定 → 证据不足；既有支持又有反驳 → 冲突）。
两维判定在结构上复刻了这条规则，因此「两维判定有效」可能只是贴合了造标签的
方式。AVeriTeC 的四类判定（Supported / Refuted / Not Enough Evidence /
Conflicting Evidence/Cherrypicking）由事实核查员**直接给出**，不是由证据级
标签合成，可以检验这个质疑。

与 CLIMATE-FEVER 的三处不同（解读结果时必须记住）
------------------------------------------------
1. **证据是问答对。** 每条 claim 配若干「问题 + 答案」，答案来自网页。这里把
   每个 (问题, 答案) 当作一篇文档，文本为 ``Question: ... Answer: ...``；
   布尔型答案把标注员写的解释接在 Yes / No 后面；``Unanswerable`` 的答案
   （「No answer could be found.」）**保留**为文档 —— 「查过但没查到」本身
   就是证据不足的证据。
2. **证据数量不固定**：每条 claim 1–16 篇，平均约 3 篇（CLIMATE-FEVER 固定 5 篇）。
3. **没有逐条证据的立场标注**，因此不存在 annotation-oracle 关系，清单里
   ``relation_predictions_kind`` 为 ``none``，也就做不了「人工标注输入的上限」
   那一类分析。另外 ``Conflicting Evidence/Cherrypicking`` 把「断章取义」也算
   在内（claim 字面属实但误导），并不总是「文档之间互相矛盾」。

官方 test 集不使用：它在 2024 年的共享任务后才公开标签，且 train + dev 已足够
构成四类均衡子集。
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_ds.data_io import write_samples
from rag_ds.dataset_manifest import (
    ArtifactDigest,
    DatasetManifest,
    DatasetSource,
    SplitArtifacts,
    file_sha256,
    write_dataset_manifest,
)
from rag_ds.datasets.climate_fever import (
    _artifact,
    _normalise_claim,
    _write_dict_jsonl,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample

__all__ = [
    "AVERITEC_COMMIT",
    "AVERITEC_DOWNLOAD_URLS",
    "AveritecDataError",
    "build_averitec_dataset",
]

AVERITEC_HOMEPAGE = "https://huggingface.co/chenxwh/AVeriTeC"
#: 下载时固定的仓库版本，保证原始文件可复现。
AVERITEC_COMMIT = "2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031"
AVERITEC_DOWNLOAD_URLS = {
    split: f"{AVERITEC_HOMEPAGE}/resolve/{AVERITEC_COMMIT}/data/{split}.json"
    for split in ("train", "dev")
}
AVERITEC_PAPER_URL = "https://arxiv.org/abs/2305.13117"
AVERITEC_CITATION = (
    "Schlichtkrull, Guo and Vlachos (2023), AVeriTeC: A Dataset for Real-world "
    "Claim Verification with Evidence from the Web, NeurIPS Datasets and Benchmarks"
)
LICENSE_NOTE = (
    "AVeriTeC is released under CC-BY-NC-4.0: non-commercial use with attribution."
)
NO_ORACLE_NOTE = (
    "AVeriTeC has no evidence-level stance annotations, so no annotation-oracle "
    "relation predictions are registered. Relation predictions for this dataset "
    "only exist as model outputs with their own model-run manifests."
)
QUESTION_TEMPLATE = (
    "Determine whether the following claim is supported, refuted, "
    "insufficiently evidenced, or met with conflicting evidence: {claim}"
)

LABEL_MAPPING: dict[str, EvidenceState] = {
    "Supported": EvidenceState.SUPPORTED,
    "Refuted": EvidenceState.REFUTED,
    "Not Enough Evidence": EvidenceState.INSUFFICIENT,
    "Conflicting Evidence/Cherrypicking": EvidenceState.CONFLICTING,
}

#: 原始文件名（不含扩展名）及读取顺序；样本 ID 里带这个名字。
SOURCE_SPLITS: tuple[str, ...] = ("train", "dev")


class AveritecDataError(ValueError):
    """原始 AVeriTeC 文件不符合预期格式。"""


class SourceDocument(BaseModel):
    """一个 (问题, 答案) 对，即一篇证据文档。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer: str
    answer_type: str
    boolean_explanation: str | None
    source_url: str
    source_medium: str

    @property
    def text(self) -> str:
        """喂给关系评估器的文档文本。"""
        answer = self.answer.strip()
        if self.boolean_explanation and self.boolean_explanation.strip():
            answer = f"{answer}. {self.boolean_explanation.strip()}"
        if not self.question.strip():
            # 原始数据里有 1 条问题文本为空（train 第 438 条），答案仍然可用。
            return answer
        return f"Question: {self.question.strip()} Answer: {answer}"


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_split: str
    index: int
    claim: str
    label: str
    documents: tuple[SourceDocument, ...]


def _load_source(path: Path, source_split: str) -> tuple[list[SourceRecord], int]:
    """读取一个官方文件，返回 ``(可用记录, 原始条数)``。

    claim 文本为空的条目无法判定任何东西，直接跳过（原始数据里有 1 条：
    train 第 1948 条）；其余格式问题一律报错，不静默丢弃。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise AveritecDataError(f"{path} 不是合法 JSON：{error}") from error
    if not isinstance(raw, list) or not raw:
        raise AveritecDataError(f"{path} 应为非空 JSON 数组")

    records: list[SourceRecord] = []
    for index, item in enumerate(raw):
        try:
            label = item["label"]
            documents = tuple(
                SourceDocument(
                    question=question["question"],
                    answer=str(answer["answer"]),
                    answer_type=answer["answer_type"],
                    boolean_explanation=answer.get("boolean_explanation"),
                    source_url=str(answer.get("source_url") or ""),
                    source_medium=str(answer.get("source_medium") or ""),
                )
                for question in item["questions"]
                for answer in question["answers"]
            )
            record = SourceRecord(
                source_split=source_split,
                index=index,
                claim=item["claim"],
                label=label,
                documents=documents,
            )
        except Exception as error:
            raise AveritecDataError(f"{path} 第 {index} 条无效：{error}") from error
        if label not in LABEL_MAPPING:
            raise AveritecDataError(f"{path} 第 {index} 条的标签未知：{label!r}")
        if not record.claim.strip():
            continue
        if not record.documents:
            raise AveritecDataError(f"{path} 第 {index} 条没有任何证据")
        if any(not d.answer.strip() for d in documents):
            raise AveritecDataError(f"{path} 第 {index} 条包含空的答案")
        records.append(record)
    return records, len(raw)


def _select_splits(
    records: list[SourceRecord],
    seed: int,
    split_per_class: dict[str, int],
) -> dict[str, list[SourceRecord]]:
    """按 claim 去重后，每类随机抽取并切成 train / validation / test。"""
    needed = sum(split_per_class.values())
    by_label: dict[str, list[SourceRecord]] = defaultdict(list)
    seen_claims: set[str] = set()
    for record in records:  # 已按 (train, dev) × 原始顺序排好，去重保留先出现的
        normalised = _normalise_claim(record.claim)
        if normalised in seen_claims:
            continue
        seen_claims.add(normalised)
        by_label[record.label].append(record)

    rng = random.Random(seed)
    result: dict[str, list[SourceRecord]] = {"train": [], "validation": [], "test": []}
    for label in LABEL_MAPPING:
        candidates = list(by_label[label])
        if len(candidates) < needed:
            raise AveritecDataError(
                f"{label} 去重后只有 {len(candidates)} 条，无法抽取 {needed} 条"
            )
        rng.shuffle(candidates)
        selected = candidates[:needed]
        offset = 0
        for split in ("train", "validation", "test"):
            count = split_per_class[split]
            result[split].extend(selected[offset : offset + count])
            offset += count

    for split in result:
        result[split].sort(key=lambda item: (SOURCE_SPLITS.index(item.source_split), item.index))
    return result


def _sample_id(record: SourceRecord) -> str:
    return f"averitec-{record.source_split}-{record.index:04d}"


def _to_sample(record: SourceRecord) -> RAGSample:
    sample_id = _sample_id(record)
    claim = record.claim.strip()
    return RAGSample(
        sample_id=sample_id,
        question=QUESTION_TEMPLATE.format(claim=claim),
        answer=claim,
        reference_answer=None,
        claims=(Claim(claim_id=f"{sample_id}-c1", text=claim),),
        contexts=tuple(
            ContextChunk(
                doc_id=f"{sample_id}-d{index}",
                text=document.text,
                retrieval_score=None,
                reliability=1.0,
            )
            for index, document in enumerate(record.documents, start=1)
        ),
        gold_state=LABEL_MAPPING[record.label],
    )


def _provenance(record: SourceRecord, split: str) -> dict[str, Any]:
    sample_id = _sample_id(record)
    return {
        "sample_id": sample_id,
        "split": split,
        "source_dataset": "AVeriTeC",
        "source_split": record.source_split,
        "source_index": record.index,
        "source_label": record.label,
        "mapped_gold_state": LABEL_MAPPING[record.label].value,
        "source_homepage": AVERITEC_HOMEPAGE,
        "contexts": [
            {
                "doc_id": f"{sample_id}-d{index}",
                "answer_type": document.answer_type,
                "source_url": document.source_url,
                "source_medium": document.source_medium,
            }
            for index, document in enumerate(record.documents, start=1)
        ],
    }


def build_averitec_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    train_per_class: int = 138,
    validation_per_class: int = 46,
    test_per_class: int = 46,
    overwrite: bool = False,
    dataset_name: str = "rag-ds-averitec-balanced-v1",
) -> DatasetManifest:
    """构建四类均衡、分层、带摘要和 provenance 的 AVeriTeC 子集。

    Args:
        source_dir: 放着官方 ``train.json`` 与 ``dev.json`` 的目录。
        output_dir: 输出目录。
        seed: 抽样随机种子。
        train_per_class: 每类进 train 的条数。
        validation_per_class: 每类进 validation 的条数。
        test_per_class: 每类进 test 的条数。
        overwrite: 是否允许覆盖已有输出。
        dataset_name: 写进清单的数据集名。

    Returns:
        写出的 :class:`DatasetManifest`。

    Raises:
        FileNotFoundError: 原始文件不存在。
        FileExistsError: 输出已存在且未指定 ``overwrite``。
        AveritecDataError: 原始数据格式不符，或某一类去重后数量不够。
    """
    source = Path(source_dir)
    raw_paths = {split: source / f"{split}.json" for split in SOURCE_SPLITS}
    for path in raw_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
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
        targets.extend([root / f"{split}.jsonl", root / f"{split}_provenance.jsonl"])
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"输出文件已存在：{existing[0]}")

    loaded: dict[str, list[SourceRecord]] = {}
    raw_counts: dict[str, int] = {}
    for split in SOURCE_SPLITS:
        loaded[split], raw_counts[split] = _load_source(raw_paths[split], split)
    records = [record for split in SOURCE_SPLITS for record in loaded[split]]
    splits = _select_splits(records, seed, split_per_class)

    artifacts: dict[str, SplitArtifacts] = {}
    for split, selected in splits.items():
        samples = [_to_sample(record) for record in selected]
        provenance = [_provenance(record, split) for record in selected]
        sample_path = root / f"{split}.jsonl"
        provenance_path = root / f"{split}_provenance.jsonl"
        write_samples(sample_path, samples, overwrite=overwrite)
        _write_dict_jsonl(provenance_path, provenance, overwrite=overwrite)
        artifacts[split] = SplitArtifacts(
            samples=_artifact(sample_path, root, len(samples)),
            provenance=_artifact(provenance_path, root, len(provenance)),
            label_counts=dict(Counter(sample.gold_state for sample in samples)),
        )

    primary, *rest = SOURCE_SPLITS
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        source=DatasetSource(
            name="AVeriTeC",
            homepage=AVERITEC_HOMEPAGE,
            download_url=AVERITEC_DOWNLOAD_URLS[primary],
            paper_url=AVERITEC_PAPER_URL,
            citation=AVERITEC_CITATION,
            license_note=LICENSE_NOTE,
            raw_file=f"{source.name}/{primary}.json",
            raw_sha256=file_sha256(raw_paths[primary]),
            raw_records=raw_counts[primary],
            extra_raw_files=tuple(
                ArtifactDigest(
                    path=f"{source.name}/{split}.json",
                    sha256=file_sha256(raw_paths[split]),
                    records=raw_counts[split],
                )
                for split in rest
            ),
        ),
        random_seed=seed,
        per_class=per_class,
        split_per_class=split_per_class,
        label_mapping=dict(LABEL_MAPPING),
        relation_predictions_kind="none",
        relation_predictions_warning=NO_ORACLE_NOTE,
        splits=artifacts,
    )
    write_dataset_manifest(manifest_path, manifest, overwrite=overwrite)
    return manifest
