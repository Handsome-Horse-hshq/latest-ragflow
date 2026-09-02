"""二维门控与评估器分歧警报。

门控坐标只有两个::

    横轴：m_theta   证据不足程度
    纵轴：K_doc     文档冲突程度

四个区域::

    m_theta 低 + K_doc 低  ->  sufficient_consistent
    m_theta 高 + K_doc 低  ->  insufficient
    m_theta 低 + K_doc 高  ->  document_conflict
    m_theta 高 + K_doc 高  ->  insufficient_and_conflicting

``K_eval`` **不是坐标轴**：它只翻转 ``evaluator_disagreement`` 这一个布尔
警报，不改变 ``region``、``primary_state``、``evidence_insufficient`` 或
``document_conflict``。评估器之间意见不合，与「证据够不够」「文档打不打架」
是三件独立的事，把它们揉进同一个区域划分会让诊断结果无法解释。

判定「高」统一使用 ``value >= threshold``（含等号），三处一致，不混用
``>`` 与 ``>=``。

本模块全部是纯函数：不修改输入、不读取 ``gold_state``、不自动调整阈值、
不做任何阈值搜索或训练。
"""

from __future__ import annotations

from rag_ds.diagnostics.models import (
    ClaimVerdict,
    DiagnosticRegion,
    DiagnosticResult,
    DiagnosticThresholds,
)
from rag_ds.ds.document_aggregation import DocumentAggregationResult
from rag_ds.ds.evaluator_aggregation import EvaluatorAggregationResult
from rag_ds.schemas import EvidenceState

__all__ = [
    "determine_verdict",
    "diagnose_no_evidence",
    "diagnose_document_total_conflict",
    "diagnose_evaluator_result",
]

#: ``(证据不足, 文档冲突)`` 到区域的映射。
_REGION_BY_FLAGS: dict[tuple[bool, bool], DiagnosticRegion] = {
    (False, False): DiagnosticRegion.SUFFICIENT_CONSISTENT,
    (True, False): DiagnosticRegion.INSUFFICIENT,
    (False, True): DiagnosticRegion.DOCUMENT_CONFLICT,
    (True, True): DiagnosticRegion.INSUFFICIENT_AND_CONFLICTING,
}


def determine_verdict(
    m_support: float, m_refute: float, tie_tolerance: float
) -> ClaimVerdict:
    """根据支持与反驳质量的差值判定倾向。

    规则::

        margin = m_support - m_refute
        margin >  tie_tolerance  ->  supported
        margin < -tie_tolerance  ->  refuted
        否则                      ->  undetermined

    ``m_theta`` **不参与**这个判断：无知程度决定的是「证据够不够」，
    而不是「偏向哪一边」，二者由不同的量分别表达。平局一律返回
    ``undetermined``，不随机打破，也不默认判为 ``supported``。

    Args:
        m_support: 支持焦元的质量。
        m_refute: 反驳焦元的质量。
        tie_tolerance: 判为平局的差值上限，必须非负。

    Returns:
        对应的 :class:`ClaimVerdict`。

    Raises:
        ValueError: ``tie_tolerance`` 为负数 —— 那会让任何微小差值都被判定
            为有倾向，与「容差」的语义相反。
    """
    if tie_tolerance < 0.0:
        raise ValueError(f"tie_tolerance 不能为负数，收到 {tie_tolerance!r}")

    margin = m_support - m_refute
    if margin > tie_tolerance:
        return ClaimVerdict.SUPPORTED
    if margin < -tie_tolerance:
        return ClaimVerdict.REFUTED
    return ClaimVerdict.UNDETERMINED


def _primary_state(
    region: DiagnosticRegion, verdict: ClaimVerdict
) -> EvidenceState | None:
    """把区域与倾向映射成用于四分类比较的主状态。

    ``insufficient_and_conflicting`` 映射为 ``CONFLICTING`` 而不是
    ``INSUFFICIENT``：混合区域若映射成「证据不足」，冲突信息就在四分类里
    彻底丢失了。两个布尔字段 ``evidence_insufficient`` 与
    ``document_conflict`` 仍然同时为 ``True``，完整信息不会丢。
    """
    if region is DiagnosticRegion.SUFFICIENT_CONSISTENT:
        if verdict is ClaimVerdict.SUPPORTED:
            return EvidenceState.SUPPORTED
        if verdict is ClaimVerdict.REFUTED:
            return EvidenceState.REFUTED
        return None  # 证据充分却分不出倾向，无法给出合理的四分类
    if region is DiagnosticRegion.INSUFFICIENT:
        return EvidenceState.INSUFFICIENT
    if region in (
        DiagnosticRegion.DOCUMENT_CONFLICT,
        DiagnosticRegion.INSUFFICIENT_AND_CONFLICTING,
        DiagnosticRegion.DOCUMENT_TOTAL_CONFLICT,
    ):
        return EvidenceState.CONFLICTING
    return None  # EVALUATOR_TOTAL_CONFLICT


def diagnose_evaluator_result(
    result: EvaluatorAggregationResult,
    thresholds: DiagnosticThresholds,
) -> DiagnosticResult:
    """对多评估器融合结果做二维门控诊断。

    正常情况下::

        theta_high        = result.mass.m_theta  >= thresholds.theta_threshold
        doc_conflict_high = result.k_doc_weighted >= thresholds.document_conflict_threshold
        eval_conflict_high = result.k_eval       >= thresholds.evaluator_conflict_threshold

    区域由前两个布尔量决定；``eval_conflict_high`` 只写进
    ``evaluator_disagreement``。

    评估器融合发生完全冲突时（``result.is_total_conflict``）：区域为
    ``evaluator_total_conflict``，三个质量为 ``None``，``verdict`` 为
    ``undetermined``，``primary_state`` 为 ``None``，
    ``evaluator_disagreement`` 为 ``True``；``k_eval`` 与
    ``k_doc_weighted`` 原样保留。此时**不会**伪造 ``m_theta = 1``，
    也**不会**把它写成「证据不足」—— 完全冲突意味着证据明确但无法通过标准
    Dempster 规则归一化，与完全无知是两回事。

    Args:
        result: 多评估器融合结果。
        thresholds: 本次诊断使用的阈值。

    Returns:
        :class:`DiagnosticResult`。

    Note:
        纯函数：不修改 ``result``（它本身也是不可变模型），不读取
        ``gold_state``，不调整阈值。
    """
    doc_conflict_high = (
        result.k_doc_weighted >= thresholds.document_conflict_threshold
    )
    eval_conflict_high = result.k_eval >= thresholds.evaluator_conflict_threshold

    if result.is_total_conflict:
        # 评估器级完全冲突：质量未定义，不做二维划分。
        return DiagnosticResult(
            sample_id=result.sample_id,
            claim_id=result.claim_id,
            m_support=None,
            m_refute=None,
            m_theta=None,
            k_doc=result.k_doc_weighted,
            k_eval=result.k_eval,
            region=DiagnosticRegion.EVALUATOR_TOTAL_CONFLICT,
            verdict=ClaimVerdict.UNDETERMINED,
            primary_state=None,
            evidence_insufficient=False,
            document_conflict=doc_conflict_high,
            evaluator_disagreement=True,
            support_refute_margin=None,
            thresholds=thresholds,
        )

    mass = result.mass
    assert mass is not None  # 由 EvaluatorAggregationResult 的校验器保证

    theta_high = mass.m_theta >= thresholds.theta_threshold
    region = _REGION_BY_FLAGS[(theta_high, doc_conflict_high)]
    verdict = determine_verdict(
        mass.m_support, mass.m_refute, thresholds.tie_tolerance
    )

    return DiagnosticResult(
        sample_id=result.sample_id,
        claim_id=result.claim_id,
        m_support=mass.m_support,
        m_refute=mass.m_refute,
        m_theta=mass.m_theta,
        k_doc=result.k_doc_weighted,
        k_eval=result.k_eval,
        region=region,
        verdict=verdict,
        primary_state=_primary_state(region, verdict),
        evidence_insufficient=theta_high,
        document_conflict=doc_conflict_high,
        evaluator_disagreement=eval_conflict_high,
        support_refute_margin=mass.m_support - mass.m_refute,
        thresholds=thresholds,
    )


def diagnose_no_evidence(
    sample_id: str,
    claim_id: str,
    thresholds: DiagnosticThresholds,
) -> DiagnosticResult:
    """给出「该 claim 没有任何检索文档」的诊断。

    输出固定为完全无知的 BPA::

        m_support = 0,  m_refute = 0,  m_theta = 1
        k_doc = 0,      k_eval = 0
        region = insufficient,  verdict = undetermined
        primary_state = INSUFFICIENT

    这里的 ``m_theta = 1`` 有确切来源：一条证据都没有，所有质量当然留在
    整个识别框架上。这与**文档完全冲突**是两回事 —— 后者证据非常明确、
    只是彼此对立到无法归一化，因此那种情况下三个质量一律为 ``None``
    （见 :func:`diagnose_document_total_conflict`），绝不写成 ``m_theta = 1``。

    Args:
        sample_id: 所属样本 ID。
        claim_id: 该 claim 的 ID。
        thresholds: 本次诊断使用的阈值，随结果保存以便复现。

    Returns:
        :class:`DiagnosticResult`。
    """
    return DiagnosticResult(
        sample_id=sample_id,
        claim_id=claim_id,
        m_support=0.0,
        m_refute=0.0,
        m_theta=1.0,
        k_doc=0.0,
        k_eval=0.0,
        region=DiagnosticRegion.INSUFFICIENT,
        verdict=ClaimVerdict.UNDETERMINED,
        primary_state=EvidenceState.INSUFFICIENT,
        evidence_insufficient=True,
        document_conflict=False,
        evaluator_disagreement=False,
        support_refute_margin=0.0,
        thresholds=thresholds,
    )


def diagnose_document_total_conflict(
    result: DocumentAggregationResult,
    thresholds: DiagnosticThresholds,
) -> DiagnosticResult:
    """对**文档级完全冲突**的聚合结果给出专门诊断。

    只接受 ``is_total_conflict=True``、``mass=None``、``k_doc=1`` 的结果；
    其他输入一律拒绝 —— 正常结果应该先做评估器聚合，再走
    :func:`diagnose_evaluator_result`。

    输出固定为 ``region=document_total_conflict``、
    ``verdict=undetermined``、``primary_state=CONFLICTING``、三个质量为
    ``None``、``k_doc=1``、``k_eval=0``、``document_conflict=True``，
    ``evidence_insufficient`` 与 ``evaluator_disagreement`` 均为 ``False``。

    这里**不会**把结果转换成 ``m_theta = 1``：``m_theta = 1`` 表示完全无知
    （谁也没给出意见），而文档完全冲突表示证据非常明确、只是彼此对立到
    标准 Dempster 规则无法归一化。两者含义完全不同，混为一谈会让下游把
    「文档吵翻了」误读成「什么都没查到」。

    Args:
        result: 文档级聚合结果，必须处于完全冲突状态。
        thresholds: 本次诊断使用的阈值，随结果保存以便复现。

    Returns:
        :class:`DiagnosticResult`。

    Raises:
        ValueError: 输入不是文档完全冲突结果。
    """
    if not (result.is_total_conflict and result.mass is None and result.k_doc == 1.0):
        raise ValueError(
            "diagnose_document_total_conflict 只接受文档完全冲突的结果，"
            f"收到 is_total_conflict={result.is_total_conflict!r}, "
            f"mass is None={result.mass is None!r}, k_doc={result.k_doc!r}；"
            "正常结果请先做评估器聚合，再调用 diagnose_evaluator_result"
        )

    return DiagnosticResult(
        sample_id=result.sample_id,
        claim_id=result.claim_id,
        m_support=None,
        m_refute=None,
        m_theta=None,
        k_doc=1.0,
        k_eval=0.0,
        region=DiagnosticRegion.DOCUMENT_TOTAL_CONFLICT,
        verdict=ClaimVerdict.UNDETERMINED,
        primary_state=EvidenceState.CONFLICTING,
        evidence_insufficient=False,
        document_conflict=True,
        evaluator_disagreement=False,
        support_refute_margin=None,
        thresholds=thresholds,
    )
