"""多评估器融合、评估器冲突指标 K_eval 与 K_doc 的可靠性加权汇总。

在链路中的位置
--------------
::

    关系概率
      -> 文档可靠性折扣      （每条文档一次）
      -> 同一评估器内融合文档 -> 得到评估器级 BPA 与该评估器的 K_doc
      -> 评估器可靠性折扣    （每个评估器只有一次）   <- 本模块
      -> 融合多个评估器      -> 得到最终 BPA 与 K_eval

评估器可靠性在这里、而不是在每条文档上施加，否则它会随文档数量被重复
计入：同一个评估器看了 5 篇文档，其可靠性就被折了 5 次，结果凭空受文档
数量影响。

三个指标互不等价
----------------
* ``m_theta`` —— 融合后仍未被分配给支持或反驳的质量，衡量**无知**：
  没人给出明确意见，或证据本身太弱。
* ``K_doc`` —— 同一评估器内**文档之间**的冲突：有的文档说支持、有的说
  反驳。
* ``K_eval`` —— **评估器之间**的冲突：不同评估器对同一 claim 给出相反
  结论。

三者可以任意组合出现，不能互相替代。例如两个评估器各自内部毫无冲突
（K_doc = 0）却彼此对立（K_eval 高）；也可能所有证据都很弱
（m_theta 高）而谁也不与谁矛盾（K_doc = K_eval = 0）。

本模块不含二维门控、最终诊断类别或阈值搜索。
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.ds._aggregation import accumulate_conflict, sequential_combine
from rag_ds.ds._numeric import clamp_unit
from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.discount import discount_combined_mass
from rag_ds.ds.document_aggregation import (
    DocumentAggregationResult,
    EmptyEvidenceError,
)
from rag_ds.schemas import NonEmptyStr, UnitFloat

__all__ = [
    "EvaluatorAggregationResult",
    "EvaluatorCombinationStep",
    "EvaluatorDocumentDiagnostic",
    "EvaluatorEvidence",
    "UndefinedDocumentMassError",
    "aggregate_evaluators",
]

#: 加权汇总允许被夹回区间端点的浮点噪声上限。
_FLOAT_SLACK = 1e-12


class UndefinedDocumentMassError(ValueError):
    """某个评估器的文档级 BPA 因完全冲突而未定义，无法参与评估器融合。

    这里刻意**不**跳过该评估器、也不把它替换成全无知或任一侧质量：文档级
    完全冲突是一个需要被下游直接诊断的结论，把它悄悄抹平会让「文档之间
    彻底矛盾」和「证据不足」变得无法区分。
    """

    def __init__(
        self, sample_id: str, claim_id: str, evaluator: str, k_doc: float
    ) -> None:
        self.sample_id = sample_id
        self.claim_id = claim_id
        self.evaluator = evaluator
        self.k_doc = k_doc
        super().__init__(
            f"评估器 {evaluator!r} 的文档级 BPA 未定义（文档间完全冲突，"
            f"K_doc={k_doc!r}），因此无法计算 K_eval："
            f"sample_id={sample_id!r}, claim_id={claim_id!r}。"
            "该情况应由 pipeline 直接诊断为文档完全冲突，"
            "本函数不会跳过该评估器，也不会把它替换成全无知或任一侧质量。"
        )


class EvaluatorEvidence(BaseModel):
    """一个评估器带入评估器级融合的全部信息。

    评估器名称、文档列表与 K_doc 都从 ``document_result`` 读取，不在这里
    重复保存，以免两份数据不一致。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 该评估器的文档级聚合结果。
    document_result: DocumentAggregationResult
    #: 该评估器自身的可靠性，在文档融合完成后**只施加一次**。
    evaluator_reliability: UnitFloat


class EvaluatorCombinationStep(BaseModel):
    """一次评估器融合的过程记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 本次融合编号，从 1 开始。
    step_index: int = Field(ge=1)
    #: 本次融合**之前**已经进入累计 BPA 的评估器名称。
    accumulated_evaluators: tuple[NonEmptyStr, ...]
    #: 本次新加入的评估器名称。
    incoming_evaluator: NonEmptyStr
    #: 本次融合的单次冲突量 K_i。完全冲突时按约定记为 1.0。
    conflict: UnitFloat
    #: 本次融合的 1 - K_i。完全冲突时按约定记为 0.0。
    normalization_denominator: UnitFloat
    #: 本次融合后的 BPA；完全冲突时为 ``None``。
    result_mass: CombinedMass | None
    #: 本次是否发生完全冲突。
    is_total_conflict: bool = False

    @model_validator(mode="after")
    def _check_total_conflict_consistency(self) -> EvaluatorCombinationStep:
        """完全冲突与 ``result_mass`` 必须一致。"""
        if self.is_total_conflict and self.result_mass is not None:
            raise ValueError("完全冲突的步骤 result_mass 必须为 None")
        if not self.is_total_conflict and self.result_mass is None:
            raise ValueError("非完全冲突的步骤必须给出 result_mass")
        return self


class EvaluatorDocumentDiagnostic(BaseModel):
    """单个评估器的中间量记录，供后续消融实验使用。

    保留每个评估器**自己的** K_doc，而不是只留加权平均值 —— 加权平均把
    「某一个评估器内部剧烈冲突」和「所有评估器都轻微冲突」压成了同一个
    数字，做方法对比时必须能拆开看。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator: NonEmptyStr
    evaluator_reliability: UnitFloat
    #: 该评估器自身的文档冲突指标。
    k_doc: UnitFloat
    #: 该评估器参与融合的文档 ID，顺序与文档聚合时一致。
    document_ids: tuple[NonEmptyStr, ...]
    #: 施加评估器可靠性折扣**之前**的评估器级 BPA。
    mass_before_evaluator_discount: CombinedMass
    #: 施加评估器可靠性折扣**之后**的评估器级 BPA。
    mass_after_evaluator_discount: CombinedMass


class EvaluatorAggregationResult(BaseModel):
    """多评估器融合的完整结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    claim_id: NonEmptyStr
    #: 参与融合的评估器名称，顺序与输入一致。
    evaluators: tuple[NonEmptyStr, ...]
    #: 全部评估器融合后的 BPA；发生完全冲突时为 ``None``。
    mass: CombinedMass | None
    #: 评估器之间的累计冲突程度，见模块文档字符串。
    k_eval: UnitFloat
    #: 各评估器 K_doc 的可靠性加权平均。
    k_doc_weighted: UnitFloat
    #: 每个评估器的中间量，含各自原始的 K_doc。
    evaluator_diagnostics: tuple[EvaluatorDocumentDiagnostic, ...]
    #: 每一步评估器融合的记录，含各步的 K_i。
    steps: tuple[EvaluatorCombinationStep, ...]
    #: 融合过程中是否出现标准 Dempster 规则无法处理的完全冲突。
    is_total_conflict: bool = False

    @model_validator(mode="after")
    def _check_result_consistency(self) -> EvaluatorAggregationResult:
        """评估器名称唯一，且完全冲突与 ``mass`` / ``k_eval`` 一致。"""
        if not self.evaluators:
            raise ValueError("evaluators 至少要包含一个评估器")

        seen: set[str] = set()
        duplicates = sorted(
            {name for name in self.evaluators if name in seen or seen.add(name)}
        )
        if duplicates:
            raise ValueError(f"evaluators 不允许重复，重复值：{duplicates}")

        if self.is_total_conflict:
            if self.mass is not None:
                raise ValueError("完全冲突时 mass 必须为 None")
            if self.k_eval != 1.0:
                raise ValueError(
                    f"完全冲突时 k_eval 必须为 1.0，当前为 {self.k_eval!r}"
                )
        elif self.mass is None:
            raise ValueError("未发生完全冲突时 mass 不能为 None")

        return self


def _check_homogeneous(evidences: list[EvaluatorEvidence]) -> None:
    """确认所有证据同属一个 sample / claim，且评估器名称不重复。"""
    for field in ("sample_id", "claim_id"):
        values = {getattr(e.document_result, field) for e in evidences}
        if len(values) > 1:
            raise ValueError(
                f"多评估器聚合要求所有证据的 {field} 相同，收到 {sorted(values)}"
            )

    names = [e.document_result.evaluator for e in evidences]
    seen: set[str] = set()
    duplicates = sorted({n for n in names if n in seen or seen.add(n)})
    if duplicates:
        raise ValueError(f"多评估器聚合不允许重复的 evaluator，重复值：{duplicates}")


def _require_defined_mass(evidence: EvaluatorEvidence) -> CombinedMass:
    """取出文档级 BPA；未定义时抛 :class:`UndefinedDocumentMassError`。"""
    document_result = evidence.document_result
    if document_result.mass is None:
        raise UndefinedDocumentMassError(
            sample_id=document_result.sample_id,
            claim_id=document_result.claim_id,
            evaluator=document_result.evaluator,
            k_doc=document_result.k_doc,
        )
    return document_result.mass


def _weighted_k_doc(evidences: list[EvaluatorEvidence]) -> float:
    """各评估器 K_doc 的可靠性加权平均。

    公式::

        k_doc_weighted = sum(r_e * K_doc_e) / sum(r_e)

    所有可靠性之和为 0 时（每个评估器都完全不可信）结果定义为 0：此时
    没有任何证据可以加权，冲突也就无从谈起 —— 与之配套的最终质量是
    ``m_theta = 1`` 的完全无知。
    """
    reliability_sum = sum(e.evaluator_reliability for e in evidences)
    if reliability_sum <= 0.0:
        return 0.0

    weighted_sum = sum(
        e.evaluator_reliability * e.document_result.k_doc for e in evidences
    )
    return clamp_unit(weighted_sum / reliability_sum, "k_doc_weighted", _FLOAT_SLACK)


def aggregate_evaluators(
    evidences: Iterable[EvaluatorEvidence],
) -> EvaluatorAggregationResult:
    """对多个评估器的文档级 BPA 各折扣一次，再依次融合。

    处理顺序：

    1. 校验所有证据同属一个 ``sample_id`` / ``claim_id``，评估器名不重复；
    2. 校验每个 ``document_result.mass`` 都已定义；
    3. 用各自的 ``evaluator_reliability`` 对文档级 BPA **各折扣一次**；
    4. 记录每个评估器折扣前后的质量与原始 K_doc；
    5. 以第一个折扣后 BPA 为初始累计质量，按输入顺序依次融合；
    6. 由各步 K_i 得到 ``K_eval = 1 - ∏(1 - K_i)``；
    7. 由各评估器的 K_doc 得到可靠性加权的 ``k_doc_weighted``。

    只有一个评估器时不调用 ``combine_two_masses``：``k_eval`` 为 0，
    ``steps`` 为空，``mass`` 就是那一次折扣后的质量。

    Args:
        evidences: 同一 ``sample_id`` / ``claim_id`` 下、来自不同评估器的
            文档级聚合结果与各自的可靠性。顺序即融合顺序。

    Returns:
        :class:`EvaluatorAggregationResult`。

    Raises:
        EmptyEvidenceError: 输入为空。
        ValueError: ``sample_id`` / ``claim_id`` 不一致，或评估器名重复。
        UndefinedDocumentMassError: 某个评估器的文档级 BPA 因完全冲突
            而未定义。

    Note:
        传入的对象都不会被修改（它们本身也是不可变模型）。

        ``evaluator_diagnostics`` 与 ``k_doc_weighted`` 覆盖**全部**输入
        证据，即使融合在中途因完全冲突而停止 —— 各评估器的 K_doc 在融合
        之前就已确定，与融合是否走完无关。
    """
    ordered = list(evidences)
    if not ordered:
        raise EmptyEvidenceError(
            "多评估器聚合收到空输入；「没有任何评估器给出结果」应由 pipeline "
            "单独诊断，不会在数学层伪造成一条全无知证据"
        )

    _check_homogeneous(ordered)

    reference = ordered[0].document_result
    evaluator_names = tuple(e.document_result.evaluator for e in ordered)

    # 评估器可靠性在这里、且只在这里施加一次。
    diagnostics: list[EvaluatorDocumentDiagnostic] = []
    discounted_masses: list[CombinedMass] = []
    for evidence in ordered:
        before = _require_defined_mass(evidence)
        after = discount_combined_mass(before, evidence.evaluator_reliability)
        discounted_masses.append(after)
        diagnostics.append(
            EvaluatorDocumentDiagnostic(
                evaluator=evidence.document_result.evaluator,
                evaluator_reliability=evidence.evaluator_reliability,
                k_doc=evidence.document_result.k_doc,
                document_ids=evidence.document_result.document_ids,
                mass_before_evaluator_discount=before,
                mass_after_evaluator_discount=after,
            )
        )

    final_mass, fold_steps, is_total_conflict = sequential_combine(discounted_masses)

    steps = tuple(
        EvaluatorCombinationStep(
            step_index=fold.index,
            accumulated_evaluators=evaluator_names[: fold.index],
            incoming_evaluator=evaluator_names[fold.index],
            conflict=fold.conflict,
            normalization_denominator=fold.denominator,
            result_mass=fold.result_mass,
            is_total_conflict=fold.is_total_conflict,
        )
        for fold in fold_steps
    )

    return EvaluatorAggregationResult(
        sample_id=reference.sample_id,
        claim_id=reference.claim_id,
        evaluators=evaluator_names,
        mass=None if is_total_conflict else final_mass,
        k_eval=accumulate_conflict(step.conflict for step in steps),
        k_doc_weighted=_weighted_k_doc(ordered),
        evaluator_diagnostics=tuple(diagnostics),
        steps=steps,
        is_total_conflict=is_total_conflict,
    )
