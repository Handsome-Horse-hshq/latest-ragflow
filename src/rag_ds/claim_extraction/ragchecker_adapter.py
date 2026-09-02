"""RAGChecker claim 抽取结果的适配器。

.. important::
    本模块**不 import ``ragchecker``，也不调用它的任何 API**。RAGChecker 的
    具体函数签名与输出结构随版本变化，凭记忆写出来的调用几乎一定是错的。
    这里只做一件事：把**你已经跑出来的** RAGChecker 输出，转换成本项目统一的
    :class:`~rag_ds.schemas.Claim`。

使用方式::

    # 1. 你自己按 RAGChecker 的文档跑它，把结果存成 JSON/JSONL
    # 2. 用本模块把结果读进来
    payload = json.loads(Path("ragchecker_claims.json").read_text("utf-8"))
    extractor = RAGCheckerClaimAdapter.from_payload(payload)
    sample_with_claims = extractor.with_claims(sample)

假定的载荷契约
--------------
``from_payload`` 接受一个 ``{sample_id: [claim 文本, ...]}`` 的映射，
其中 claim 也可以是含 ``text``（或 ``claim``）键的字典。这是本项目**自己
定义的中间格式**，不是 RAGChecker 的原生格式 —— 你需要写一小段胶水代码把
RAGChecker 的实际输出整理成这个形状，具体字段名请查你所用版本的文档。

这样拆分的好处是：RAGChecker 换版本时，只有那段胶水代码要改，本模块与
下游的 D-S 全链路都不受影响。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag_ds.claim_extraction.base import ClaimExtractor
from rag_ds.schemas import Claim, RAGSample

__all__ = [
    "DEFAULT_CLAIM_ID_TEMPLATE",
    "RAGCheckerClaimAdapter",
    "claims_from_texts",
]

#: 生成 ``claim_id`` 的模板，占位符为 ``sample_id`` 与 1 起编号的 ``index``。
DEFAULT_CLAIM_ID_TEMPLATE = "{sample_id}-c{index}"


def claims_from_texts(
    sample_id: str,
    texts: Iterable[str],
    claim_id_template: str = DEFAULT_CLAIM_ID_TEMPLATE,
) -> list[Claim]:
    """把一串 claim 文本转成带 ID 的 :class:`Claim` 列表。

    ID 由模板生成而非取自外部 —— RAGChecker 不保证给出稳定的 claim 标识，
    自行按顺序编号可以确保同一样本内唯一且可复现。

    Args:
        sample_id: 所属样本 ID，用于拼接 claim_id。
        texts: claim 文本，按原始顺序。
        claim_id_template: 含 ``{sample_id}`` 与 ``{index}`` 的模板。

    Returns:
        :class:`Claim` 列表；空白文本会被跳过。
    """
    claims: list[Claim] = []
    for index, text in enumerate(texts, start=1):
        if not text or not text.strip():
            continue  # 跳过空串，而不是生成一个非法的 Claim
        claims.append(
            Claim(
                claim_id=claim_id_template.format(
                    sample_id=sample_id, index=len(claims) + 1
                ),
                text=text,
            )
        )
    return claims


def _claim_text(entry: Any) -> str:
    """从一个载荷条目中取出 claim 文本。

    接受纯字符串，或含 ``text`` / ``claim`` / ``content`` 键的映射 ——
    这三个键名覆盖了整理 RAGChecker 输出时最常见的几种写法。
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        for key in ("text", "claim", "content"):
            value = entry.get(key)
            if isinstance(value, str):
                return value
        raise ValueError(
            f"claim 条目缺少 text / claim / content 键：{sorted(entry)}"
        )
    raise TypeError(f"claim 条目必须是字符串或映射，收到 {type(entry).__name__}")


class RAGCheckerClaimAdapter(ClaimExtractor):
    """把整理好的 RAGChecker claim 输出接入本项目的抽取器接口。

    实现 :class:`ClaimExtractor`，因此可以和
    :class:`~rag_ds.claim_extraction.mock.MockClaimExtractor` 直接互换，
    下游代码无需改动。

    Args:
        claims_by_sample: ``{sample_id: [Claim, ...]}``。
        name: 抽取器名称，默认 ``"ragchecker"``。

    Raises:
        ValueError: ``name`` 为空。
    """

    def __init__(
        self,
        claims_by_sample: Mapping[str, Iterable[Claim]],
        name: str = "ragchecker",
    ) -> None:
        extractor_name = name.strip()
        if not extractor_name:
            raise ValueError("抽取器 name 不能为空")
        self._name = extractor_name
        self._table: dict[str, list[Claim]] = {
            sample_id: [claim.model_copy(deep=True) for claim in claims]
            for sample_id, claims in claims_by_sample.items()
        }

    @property
    def name(self) -> str:
        """抽取器名称。"""
        return self._name

    def __len__(self) -> int:
        """已装载的样本数。"""
        return len(self._table)

    def extract(self, sample: RAGSample) -> list[Claim]:
        """返回该样本的 claim 深拷贝。

        Raises:
            KeyError: 该样本不在载荷中 —— 不会静默返回空列表，否则
                「RAGChecker 没跑这个样本」会被误读成「这个答案没有断言」。
        """
        claims = self._table.get(sample.sample_id)
        if claims is None:
            raise KeyError(
                f"RAGChecker 载荷中没有 sample_id={sample.sample_id!r} 的 claim；"
                "不会静默返回空列表"
            )
        return [claim.model_copy(deep=True) for claim in claims]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Iterable[Any]],
        name: str = "ragchecker",
        claim_id_template: str = DEFAULT_CLAIM_ID_TEMPLATE,
    ) -> RAGCheckerClaimAdapter:
        """从 ``{sample_id: [claim, ...]}`` 载荷构造适配器。

        Args:
            payload: 整理好的 RAGChecker 输出，见模块文档字符串。
            name: 抽取器名称。
            claim_id_template: claim_id 生成模板。

        Returns:
            :class:`RAGCheckerClaimAdapter`。

        Raises:
            ValueError: 某个条目缺少文本字段。
            TypeError: 某个条目既不是字符串也不是映射。
        """
        return cls(
            {
                sample_id: claims_from_texts(
                    sample_id,
                    (_claim_text(entry) for entry in entries),
                    claim_id_template,
                )
                for sample_id, entries in payload.items()
            },
            name=name,
        )
