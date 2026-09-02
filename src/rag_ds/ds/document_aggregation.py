"""同一评估器下的多文档 BPA 融合，以及文档冲突指标 K_doc。

处理范围
--------
输入的所有 BPA 必须属于同一个 ``sample_id``、``claim_id`` 与
``evaluator``，但来自**不同**的 ``doc_id``。跨样本、跨 claim 或跨评估器
的混合输入会被拒绝 —— 那是聚合层调用方配错了数据，静默融合会得到毫无
意义的结果。

融合顺序
--------
以第一条文档的 BPA 作为初始累计 BPA，按**输入原始顺序**依次加入后续
文档，每一步调用第六阶段的 :func:`~rag_ds.ds.combination.combine_two_masses`
并记录该步的单次冲突量 K_i。输入顺序不会按 ``retrieval_score`` 或
``doc_id`` 重排。

K_doc 的定义
------------
::

    K_doc = 1 - (1 - K_1)(1 - K_2) ... (1 - K_n)

其中 K_i 是**累计 BPA 与第 i+1 条文档 BPA** 的单次冲突量。

K_doc **不是** K_i 的平均值，也**不是** ``m_theta``：

* 高 ``m_theta`` 表示证据不足、无知 —— 没人给出明确意见；
* 高 K_doc 表示文档之间互相矛盾 —— 有人说支持、有人说反驳。

两者可以同时高，也可以一高一低，不能互相替代。

本模块只做同一评估器内的文档融合，不含多评估器融合、K_eval 或二维门控。
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.ds._aggregation import (
    accumulate_conflict,
    sequential_combine,
    to_combined_mass,
)
from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.mass import MassFunction
from rag_ds.schemas import NonEmptyStr, UnitFloat

__all__ = [
    "DocumentAggregationResult",
    "DocumentCombinationStep",
    "EmptyEvidenceError",
    "aggregate_document_masses",
]


class EmptyEvidenceError(ValueError):
    """聚合函数收到了空的证据集合。

    这里刻意**不**返回一条全无知 BPA：「一条文档都没检索到」是检索环节
    的问题，应由 pipeline 单独诊断；在数学层把它伪造成一条 ``m_theta=1``
    的证据，会让「没有证据」和「证据说不清楚」这两种完全不同的情况
    在下游变得无法区分。
    """


class DocumentCombinationStep(BaseModel):
    """一次文档融合的过程记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 本次融合编号，从 1 开始。
    step_index: int = Field(ge=1)
    #: 本次融合**之前**已经进入累计 BPA 的文档 ID。
    accumulated_doc_ids: tuple[NonEmptyStr, ...]
    #: 本次新加入的文档 ID。
    incoming_doc_id: NonEmptyStr
    #: 本次融合的单次冲突量 K_i。完全冲突时按约定记为 1.0。
    conflict: UnitFloat
    #: 本次融合的 1 - K_i。完全冲突时按约定记为 0.0。
    normalization_denominator: UnitFloat
    #: 本次融合后的 BPA；完全冲突时为 ``None``。
    result_mass: CombinedMass | None
    #: 本次是否发生完全冲突。
    is_total_conflict: bool = False

    @model_validator(mode="after")
    def _check_total_conflict_consistency(self) -> DocumentCombinationStep:
        """完全冲突与 ``result_mass`` 必须一致。"""
        if self.is_total_conflict and self.result_mass is not None:
            raise ValueError("完全冲突的步骤 result_mass 必须为 None")
        if not self.is_total_conflict and self.result_mass is None:
            raise ValueError("非完全冲突的步骤必须给出 result_mass")
        return self


class DocumentAggregationResult(BaseModel):
    """同一评估器下多文档融合的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    evaluator: NonEmptyStr
    #: 参与融合的文档 ID，顺序与输入一致。
    document_ids: tuple[NonEmptyStr, ...]
    #: 全部文档融合后的 BPA；发生完全冲突时为 ``None``。
    mass: CombinedMass | None
    #: 多文档累计冲突程度，见模块文档字符串。
    k_doc: UnitFloat
    #: 每一步的融合记录，含各步的 K_i。
    steps: tuple[DocumentCombinationStep, ...]
    #: 融合过程中是否出现标准 Dempster 规则无法处理的完全冲突。
    is_total_conflict: bool = False

    @model_validator(mode="after")
    def _check_result_consistency(self) -> DocumentAggregationResult:
        """文档 ID 唯一，且完全冲突与 ``mass`` / ``k_doc`` 一致。"""
        if not self.document_ids:
            raise ValueError("document_ids 至少要包含一个文档")

        seen: set[str] = set()
        duplicates = sorted({d for d in self.document_ids if d in seen or seen.add(d)})
        if duplicates:
            raise ValueError(f"document_ids 不允许重复，重复值：{duplicates}")

        if self.is_total_conflict:
            if self.mass is not None:
                raise ValueError("完全冲突时 mass 必须为 None")
            if self.k_doc != 1.0:
                raise ValueError(f"完全冲突时 k_doc 必须为 1.0，当前为 {self.k_doc!r}")
        elif self.mass is None:
            raise ValueError("未发生完全冲突时 mass 不能为 None")

        return self


def _check_homogeneous(masses: list[MassFunction]) -> None:
    """确认所有 BPA 同属一个 sample / claim / evaluator，且文档不重复。"""
    for field in ("sample_id", "claim_id", "evaluator"):
        values = {getattr(mass, field) for mass in masses}
        if len(values) > 1:
            raise ValueError(
                f"多文档聚合要求所有 BPA 的 {field} 相同，"
                f"收到 {sorted(values)}"
            )

    seen: set[str] = set()
    duplicates = sorted({m.doc_id for m in masses if m.doc_id in seen or seen.add(m.doc_id)})
    if duplicates:
        raise ValueError(f"多文档聚合不允许重复的 doc_id，重复值：{duplicates}")


def aggregate_document_masses(
    masses: Iterable[MassFunction],
) -> DocumentAggregationResult:
    """把同一评估器对多条文档产生的 BPA 依次融合，并汇总 K_doc。

    输入的每个 :class:`~rag_ds.ds.mass.MassFunction` 都必须**已经完成
    文档可靠性折扣**；本函数不会再次应用任何可靠性。

    只有一条文档时不调用 :func:`combine_two_masses`：直接把三个质量复制成
    :class:`~rag_ds.ds.combination.CombinedMass`，``k_doc`` 为 0，
    ``steps`` 为空元组。

    Args:
        masses: 同一 ``sample_id`` / ``claim_id`` / ``evaluator`` 下、
            来自不同 ``doc_id`` 的已折扣 BPA。顺序即融合顺序，不会被
            按 ``retrieval_score`` 或 ``doc_id`` 重排。

    Returns:
        :class:`DocumentAggregationResult`，含最终 BPA、K_doc 与逐步记录。

    Raises:
        EmptyEvidenceError: 输入为空。
        ValueError: ``sample_id`` / ``claim_id`` / ``evaluator`` 不一致，
            或 ``doc_id`` 重复。

    Note:
        传入的对象都不会被修改（它们本身也是不可变模型）。
    """
    ordered = list(masses)
    if not ordered:
        raise EmptyEvidenceError(
            "多文档聚合收到空输入；「没有检索到文档」应由 pipeline 单独诊断，"
            "不会在数学层伪造成一条全无知证据"
        )

    _check_homogeneous(ordered)

    document_ids = tuple(mass.doc_id for mass in ordered)
    reference = ordered[0]

    final_mass, fold_steps, is_total_conflict = sequential_combine(ordered)

    steps = tuple(
        DocumentCombinationStep(
            step_index=fold.index,
            accumulated_doc_ids=document_ids[: fold.index],
            incoming_doc_id=document_ids[fold.index],
            conflict=fold.conflict,
            normalization_denominator=fold.denominator,
            result_mass=fold.result_mass,
            is_total_conflict=fold.is_total_conflict,
        )
        for fold in fold_steps
    )

    return DocumentAggregationResult(
        sample_id=reference.sample_id,
        claim_id=reference.claim_id,
        evaluator=reference.evaluator,
        document_ids=document_ids,
        mass=None if is_total_conflict else to_combined_mass(final_mass),
        k_doc=accumulate_conflict(step.conflict for step in steps),
        steps=steps,
        is_total_conflict=is_total_conflict,
    )
