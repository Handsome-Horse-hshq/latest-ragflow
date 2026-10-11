"""把 CONFACT（HumC 子集）转成「每篇网页一段最相关段落 + 来源可信度」的样本。

为什么用这个数据集
------------------
CLIMATE-FEVER 与 AVeriTeC 里所有文档的可靠性都是 1.0，D-S 的**文档级可靠性
折扣**从来没有被真实数据检验过。CONFACT（Ge et al., IJCAI 2025）给每条 claim
配了约 12 篇互相冲突的网页，并附带 Media Bias/Fact Check 对 6100 家媒体的
可信度分级（high / medium / low）—— 正好是「文档可靠性确有差异」的设定。

它能回答、不能回答的问题
------------------------
* claim 的标签只有 ``Supported`` / ``Refuted`` 两类（冲突是数据集的前提，不是
  标签），所以这里检验的是**倾向判断**（支持还是反驳），不是四类证据状态。
* 没有逐篇文档的立场标注，不存在 annotation oracle。

三个必须写进论文的处理
----------------------
1. **整页网页 → 一段话。** 网页中位数约 940 词，NLI 模型一次只能看 512 个
   词元。每篇网页切成重叠的词窗，取与 claim 的 TF-IDF 余弦相似度最高的那一段
   作为该文档的文本。这是最朴素的稀疏检索，不做任何调参。
2. **可信度 → 可靠性**用事先定好的映射 :data:`CREDIBILITY_RELIABILITY`；没有
   评级的来源取 :data:`UNRATED_RELIABILITY`。实验脚本会对这两个取值做敏感性
   扫描并全部报告。
3. **域名匹配**先按完整主机名（去掉 ``www.``）查，查不到再退到最后两段。

安全
----
原始数据是 pickle。``pickle.load`` 会执行文件里指定的任意可调用对象，来源
不明的 pickle 不能直接加载；:func:`load_untrusted_pickle` 用一个**拒绝导入
任何类**的 Unpickler，只接受内置的列表、字典、字符串与数字。

原仓库没有声明许可证：数据只在本地使用，不随本仓库分发。
"""

from __future__ import annotations

import gzip
import io
import json
import pickle
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rag_ds.data_io import write_samples
from rag_ds.dataset_manifest import file_sha256
from rag_ds.datasets.climate_fever import _write_dict_jsonl
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample

__all__ = [
    "CONFACT_COMMIT",
    "CONFACT_HOMEPAGE",
    "CREDIBILITY_RELIABILITY",
    "UNRATED_RELIABILITY",
    "ConfactDataError",
    "build_confact_dataset",
    "load_untrusted_pickle",
    "lookup_credibility",
    "select_passage",
    "source_domain",
    "split_passages",
]

CONFACT_HOMEPAGE = "https://github.com/zoeyyes/CONFACT"
#: 下载时固定的仓库版本。
CONFACT_COMMIT = "05a1227b4821ebf1b843f0ce5824f9420006cfdb"
CONFACT_PAPER_URL = "https://arxiv.org/abs/2505.17762"

#: MBFC 可信度分级到文档可靠性的映射（事先定好，不在数据上调）。
CREDIBILITY_RELIABILITY: dict[str, float] = {"high": 1.0, "medium": 0.6, "low": 0.2}
#: 没有 MBFC 评级的来源：按「中等」对待。
UNRATED_RELIABILITY: float = 0.6

LABEL_MAPPING: dict[str, EvidenceState] = {
    "Supported": EvidenceState.SUPPORTED,
    "Refuted": EvidenceState.REFUTED,
}


class ConfactDataError(ValueError):
    """原始 CONFACT 文件不符合预期格式。"""


class _NoImportUnpickler(pickle.Unpickler):
    """拒绝导入任何类或函数的 Unpickler。"""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"拒绝加载：pickle 试图导入 {module}.{name}。"
            "这里只接受由内置容器、字符串与数字组成的数据。"
        )


def load_untrusted_pickle(path: str | Path) -> Any:
    """安全读取一个来源不明的 pickle（``.gz`` 会先解压）。

    Raises:
        pickle.UnpicklingError: 文件试图导入任何模块里的对象。
    """
    raw = Path(path).read_bytes()
    data = gzip.decompress(raw) if str(path).endswith(".gz") else raw
    return _NoImportUnpickler(io.BytesIO(data)).load()


def source_domain(url: str) -> str:
    """取 URL 的主机名（小写，去掉 ``www.``）。"""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def lookup_credibility(domain: str, media: Mapping[str, Mapping[str, Any]]) -> str | None:
    """查某个域名的 MBFC 可信度；先查完整主机名，再退到最后两段。"""
    for candidate in (domain, ".".join(domain.split(".")[-2:])):
        entry = media.get(candidate)
        if entry is not None:
            credibility = str(entry.get("credibility", "")).strip().lower()
            if credibility in CREDIBILITY_RELIABILITY:
                return credibility
    return None


def split_passages(
    text: str, *, window: int = 120, stride: int = 60, max_words: int = 30000
) -> list[str]:
    """按词切成重叠的窗口；只看前 ``max_words`` 个词（个别网页有几十万词）。"""
    if window < 1 or stride < 1:
        raise ValueError("window 与 stride 必须是正整数")
    words = re.sub(r"\s+", " ", text).strip().split(" ")[:max_words]
    if words == [""]:
        return []
    passages: list[str] = []
    for start in range(0, len(words), stride):
        passages.append(" ".join(words[start : start + window]))
        if start + window >= len(words):
            break
    return passages


def select_passage(claim: str, passages_by_doc: Sequence[Sequence[str]]) -> list[int]:
    """为每篇文档选出与 claim 的 TF-IDF 余弦相似度最高的段落下标。

    TF-IDF 在该 claim 的全部段落上拟合。某篇文档的所有段落相似度都是 0 时取
    第一段（网页开头）。

    Args:
        claim: claim 文本（可以把问题接在后面一起传入）。
        passages_by_doc: 每篇文档的段落列表，均不为空。

    Returns:
        与 ``passages_by_doc`` 等长的下标列表。
    """
    # 只有这里需要 scikit-learn；延迟导入，``import rag_ds.datasets`` 保持轻量。
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    flat = [passage for passages in passages_by_doc for passage in passages]
    try:
        matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(
            [claim, *flat]
        )
    except ValueError:
        # 词表为空（全是停用词或符号）：无从比较，统一取第一段。
        return [0] * len(passages_by_doc)
    similarity = linear_kernel(matrix[0:1], matrix[1:]).ravel()

    chosen: list[int] = []
    cursor = 0
    for passages in passages_by_doc:
        scores = similarity[cursor : cursor + len(passages)]
        chosen.append(int(scores.argmax()))  # 并列时 argmax 取最靠前的
        cursor += len(passages)
    return chosen


def build_confact_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    subset: str = "HumC",
    window: int = 120,
    stride: int = 60,
    max_words: int = 30000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """构建 CONFACT 样本、谱系与元数据。

    Args:
        source_dir: 放着 ``<subset>.pkl.gz`` 与 ``mbfc_media_data.pkl`` 的目录。
        output_dir: 输出目录；写出 ``samples.jsonl``、``provenance.jsonl``、
            ``meta.json``。
        subset: ``HumC``（人工撰写的冲突证据）。
        window: 段落的词数。
        stride: 相邻段落起点的间隔词数。
        max_words: 每篇网页只看前多少个词。
        overwrite: 是否允许覆盖已有输出。

    Returns:
        写进 ``meta.json`` 的元数据。

    Raises:
        FileNotFoundError: 原始文件不存在。
        FileExistsError: 输出已存在且未指定 ``overwrite``。
        ConfactDataError: 原始数据格式不符。
        pickle.UnpicklingError: pickle 里含有内置类型之外的对象。
    """
    source = Path(source_dir)
    claims_path = source / f"{subset}.pkl.gz"
    media_path = source / "mbfc_media_data.pkl"
    for path in (claims_path, media_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    root = Path(output_dir)
    samples_path = root / "samples.jsonl"
    provenance_path = root / "provenance.jsonl"
    meta_path = root / "meta.json"
    if not overwrite:
        existing = [p for p in (samples_path, provenance_path, meta_path) if p.exists()]
        if existing:
            raise FileExistsError(f"输出文件已存在：{existing[0]}")

    raw_claims = load_untrusted_pickle(claims_path)
    media = load_untrusted_pickle(media_path)
    if not isinstance(raw_claims, list) or not isinstance(media, dict):
        raise ConfactDataError("CONFACT 原始数据的顶层结构不符合预期")

    samples: list[RAGSample] = []
    provenance: list[dict[str, Any]] = []
    skipped_documents = 0
    truncated_documents = 0
    credibility_counts: Counter[str] = Counter()
    for item in raw_claims:
        try:
            label = str(item["label"]).strip()
            claim_text = str(item["claim"]).strip()
            question = str(item.get("question") or "").strip()
            documents = list(item["evidence_url"])
            identifier = int(item["id"])
        except Exception as error:
            raise ConfactDataError(f"CONFACT 条目无效：{error}") from error
        if label not in LABEL_MAPPING:
            raise ConfactDataError(f"id={identifier} 的标签未知：{label!r}")
        if not claim_text:
            raise ConfactDataError(f"id={identifier} 的 claim 为空")

        kept: list[tuple[dict[str, Any], list[str]]] = []
        for document in documents:
            content = str(document.get("content") or "")
            passages = split_passages(
                content, window=window, stride=stride, max_words=max_words
            )
            if not passages:
                skipped_documents += 1
                continue
            if len(content.split()) > max_words:
                truncated_documents += 1
            kept.append((document, passages))
        if not kept:
            raise ConfactDataError(f"id={identifier} 没有任何可用的证据文档")

        sample_id = f"confact-{subset.lower()}-{identifier:04d}"
        query = f"{claim_text} {question}".strip()
        chosen = select_passage(query, [passages for _, passages in kept])
        contexts: list[ContextChunk] = []
        context_provenance: list[dict[str, Any]] = []
        for index, ((document, passages), passage_index) in enumerate(
            zip(kept, chosen), start=1
        ):
            url = str(document.get("original_link") or "")
            domain = source_domain(url)
            credibility = lookup_credibility(domain, media)
            credibility_counts[credibility or "unrated"] += 1
            doc_id = f"{sample_id}-d{index}"
            contexts.append(
                ContextChunk(
                    doc_id=doc_id,
                    text=passages[passage_index],
                    retrieval_score=None,
                    reliability=(
                        CREDIBILITY_RELIABILITY[credibility]
                        if credibility is not None
                        else UNRATED_RELIABILITY
                    ),
                )
            )
            context_provenance.append(
                {
                    "doc_id": doc_id,
                    "source_evidence_id": str(document.get("evidence_id") or ""),
                    "url": url,
                    "domain": domain,
                    "credibility": credibility,
                    "passage_index": passage_index,
                    "passage_count": len(passages),
                }
            )

        samples.append(
            RAGSample(
                sample_id=sample_id,
                question=question or claim_text,
                answer=claim_text,
                reference_answer=None,
                claims=(Claim(claim_id=f"{sample_id}-c1", text=claim_text),),
                contexts=tuple(contexts),
                gold_state=LABEL_MAPPING[label],
            )
        )
        provenance.append(
            {
                "sample_id": sample_id,
                "source_dataset": f"CONFACT-{subset}",
                "source_id": identifier,
                "source_label": label,
                "contexts": context_provenance,
            }
        )

    write_samples(samples_path, samples, overwrite=overwrite)
    _write_dict_jsonl(provenance_path, provenance, overwrite=overwrite)
    meta = {
        "dataset_name": f"rag-ds-confact-{subset.lower()}-v1",
        "source": {
            "name": f"CONFACT-{subset}",
            "homepage": CONFACT_HOMEPAGE,
            "commit": CONFACT_COMMIT,
            "paper_url": CONFACT_PAPER_URL,
            "license_note": (
                "The upstream repository declares no license. The raw files are "
                "used locally for research and are not redistributed here."
            ),
            "claims_file": claims_path.name,
            "claims_sha256": file_sha256(claims_path),
            "media_file": media_path.name,
            "media_sha256": file_sha256(media_path),
        },
        "passage_selection": {
            "method": "tfidf_cosine_top1",
            "window_words": window,
            "stride_words": stride,
            "max_words_per_document": max_words,
        },
        "credibility_reliability": CREDIBILITY_RELIABILITY,
        "unrated_reliability": UNRATED_RELIABILITY,
        "claims": len(samples),
        "label_counts": dict(Counter(s.gold_state.value for s in samples)),  # type: ignore[union-attr]
        "documents": sum(len(s.contexts) for s in samples),
        "documents_skipped_empty": skipped_documents,
        "documents_truncated": truncated_documents,
        "document_credibility_counts": dict(credibility_counts),
        "samples": {"path": samples_path.name, "sha256": file_sha256(samples_path)},
        "provenance": {
            "path": provenance_path.name,
            "sha256": file_sha256(provenance_path),
        },
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return meta
