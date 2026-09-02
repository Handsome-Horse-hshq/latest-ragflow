"""第九阶段二维门控与评估器分歧警报的测试。"""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.diagnostics.gating import (
    determine_verdict,
    diagnose_document_total_conflict,
    diagnose_evaluator_result,
)
from rag_ds.diagnostics.models import (
    ClaimVerdict,
    DiagnosticRegion,
    DiagnosticResult,
    DiagnosticThresholds,
)
from rag_ds.ds.combination import CombinedMass
from rag_ds.ds.discount import document_discounted_mass_from_prediction
from rag_ds.ds.document_aggregation import aggregate_document_masses
from rag_ds.ds.evaluator_aggregation import (
    EvaluatorAggregationResult,
    EvaluatorEvidence,
    aggregate_evaluators,
)
from rag_ds.ds.mass import MassFunction
from rag_ds.schemas import EvidenceState, RAGSample

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_PATH = _DATA_DIR / "demo.jsonl"
MOCK_PATH = _DATA_DIR / "mock_relations.jsonl"

THRESHOLDS = DiagnosticThresholds()


def _evaluator_result(
    m_support: float,
    m_refute: float,
    m_theta: float,
    k_doc_weighted: float = 0.0,
    k_eval: float = 0.0,
) -> EvaluatorAggregationResult:
    """直接构造一个评估器聚合结果，用于精确控制三个诊断量。"""
    return EvaluatorAggregationResult(
        sample_id="s1",
        claim_id="c1",
        evaluators=("eval_a",),
        mass=CombinedMass(m_support=m_support, m_refute=m_refute, m_theta=m_theta),
        k_eval=k_eval,
        k_doc_weighted=k_doc_weighted,
        evaluator_diagnostics=(),
        steps=(),
        is_total_conflict=False,
    )


def _total_conflict_evaluator_result(
    k_doc_weighted: float = 0.0,
) -> EvaluatorAggregationResult:
    """构造一个评估器级完全冲突的结果。"""
    return EvaluatorAggregationResult(
        sample_id="s1",
        claim_id="c1",
        evaluators=("eval_a", "eval_b"),
        mass=None,
        k_eval=1.0,
        k_doc_weighted=k_doc_weighted,
        evaluator_diagnostics=(),
        steps=(),
        is_total_conflict=True,
    )


def _mass(doc_id: str, m_support: float, m_refute: float, m_theta: float) -> MassFunction:
    """构造一条已完成文档折扣的 BPA。"""
    return MassFunction(
        sample_id="s1",
        claim_id="c1",
        doc_id=doc_id,
        evaluator="eval_a",
        m_support=m_support,
        m_refute=m_refute,
        m_theta=m_theta,
    )


# --------------------------------------------------------------------------
# 1-3. DiagnosticThresholds
# --------------------------------------------------------------------------


def test_default_thresholds() -> None:
    """默认阈值为调试值 0.5 / 0.4 / 0.4 / 1e-6。"""
    thresholds = DiagnosticThresholds()

    assert thresholds.theta_threshold == pytest.approx(0.5)
    assert thresholds.document_conflict_threshold == pytest.approx(0.4)
    assert thresholds.evaluator_conflict_threshold == pytest.approx(0.4)
    assert thresholds.tie_tolerance == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "field",
    [
        "theta_threshold",
        "document_conflict_threshold",
        "evaluator_conflict_threshold",
        "tie_tolerance",
    ],
)
@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_thresholds_out_of_range_are_rejected(field: str, bad_value: float) -> None:
    """阈值超出 [0, 1] 时被拒绝。"""
    with pytest.raises(ValidationError, match=field):
        DiagnosticThresholds(**{field: bad_value})


def test_thresholds_reject_unknown_field_and_are_immutable() -> None:
    """阈值模型禁止未定义字段且不可变。"""
    with pytest.raises(ValidationError, match="tau_theta"):
        DiagnosticThresholds(tau_theta=0.5)

    thresholds = DiagnosticThresholds()
    with pytest.raises(ValidationError):
        thresholds.theta_threshold = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------
# 4-7. determine_verdict
# --------------------------------------------------------------------------


def test_verdict_supported() -> None:
    """m_support 明显大于 m_refute 时判为 supported。"""
    assert determine_verdict(0.7, 0.1, 1e-6) is ClaimVerdict.SUPPORTED


def test_verdict_refuted() -> None:
    """m_refute 明显大于 m_support 时判为 refuted。"""
    assert determine_verdict(0.1, 0.7, 1e-6) is ClaimVerdict.REFUTED


def test_exact_tie_is_undetermined() -> None:
    """完全相等时判为 undetermined，不默认归为 supported。"""
    assert determine_verdict(0.4, 0.4, 1e-6) is ClaimVerdict.UNDETERMINED


@pytest.mark.parametrize("margin", [0.0, 1e-7, -1e-7, 1e-6, -1e-6])
def test_within_tie_tolerance_is_undetermined(margin: float) -> None:
    """差值绝对值不超过 tie_tolerance 时判为 undetermined。"""
    assert determine_verdict(0.4 + margin, 0.4, 1e-6) is ClaimVerdict.UNDETERMINED


@pytest.mark.parametrize("margin", [2e-6, -2e-6])
def test_just_outside_tie_tolerance_is_decided(margin: float) -> None:
    """刚超出容差即给出倾向。"""
    verdict = determine_verdict(0.4 + margin, 0.4, 1e-6)

    assert verdict is not ClaimVerdict.UNDETERMINED


def test_verdict_ignores_theta() -> None:
    """m_theta 不影响倾向判断 —— 它只表达证据够不够。"""
    assert determine_verdict(0.05, 0.01, 1e-6) is ClaimVerdict.SUPPORTED


def test_negative_tie_tolerance_is_rejected() -> None:
    """负的 tie_tolerance 与「容差」语义相反，被拒绝。"""
    with pytest.raises(ValueError, match="tie_tolerance 不能为负数"):
        determine_verdict(0.4, 0.4, -1e-6)


# --------------------------------------------------------------------------
# 8-12. 二维区域
# --------------------------------------------------------------------------


def test_low_theta_low_conflict_is_sufficient_consistent() -> None:
    """m_theta 低、K_doc 低 -> sufficient_consistent。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.7, 0.1, 0.2, k_doc_weighted=0.1), THRESHOLDS
    )

    assert result.region is DiagnosticRegion.SUFFICIENT_CONSISTENT
    assert result.evidence_insufficient is False
    assert result.document_conflict is False


def test_high_theta_low_conflict_is_insufficient() -> None:
    """m_theta 高、K_doc 低 -> insufficient。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.2, 0.1, 0.7, k_doc_weighted=0.1), THRESHOLDS
    )

    assert result.region is DiagnosticRegion.INSUFFICIENT
    assert result.evidence_insufficient is True
    assert result.document_conflict is False


def test_low_theta_high_conflict_is_document_conflict() -> None:
    """m_theta 低、K_doc 高 -> document_conflict。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.5, 0.3, 0.2, k_doc_weighted=0.7), THRESHOLDS
    )

    assert result.region is DiagnosticRegion.DOCUMENT_CONFLICT
    assert result.evidence_insufficient is False
    assert result.document_conflict is True


def test_high_theta_high_conflict_is_mixed_region() -> None:
    """m_theta 高、K_doc 高 -> insufficient_and_conflicting，两个布尔都为 True。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.2, 0.1, 0.7, k_doc_weighted=0.7), THRESHOLDS
    )

    assert result.region is DiagnosticRegion.INSUFFICIENT_AND_CONFLICTING
    assert result.evidence_insufficient is True
    assert result.document_conflict is True


def test_values_exactly_at_thresholds_count_as_high() -> None:
    """数值恰好等于阈值时按「高」处理（统一使用 >=）。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.3, 0.2, 0.5, k_doc_weighted=0.4, k_eval=0.4), THRESHOLDS
    )

    assert result.m_theta == pytest.approx(0.5)
    assert result.evidence_insufficient is True
    assert result.document_conflict is True
    assert result.evaluator_disagreement is True
    assert result.region is DiagnosticRegion.INSUFFICIENT_AND_CONFLICTING


def test_values_just_below_thresholds_count_as_low() -> None:
    """略低于阈值时按「低」处理。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.3, 0.2, 0.5 - 1e-9, k_doc_weighted=0.4 - 1e-9, k_eval=0.4 - 1e-9),
        THRESHOLDS,
    )

    assert result.evidence_insufficient is False
    assert result.document_conflict is False
    assert result.evaluator_disagreement is False
    assert result.region is DiagnosticRegion.SUFFICIENT_CONSISTENT


# --------------------------------------------------------------------------
# 13-18. primary_state 映射
# --------------------------------------------------------------------------


def test_sufficient_consistent_supported_maps_to_supported() -> None:
    """证据充分且倾向支持 -> SUPPORTED。"""
    result = diagnose_evaluator_result(_evaluator_result(0.7, 0.1, 0.2), THRESHOLDS)

    assert result.verdict is ClaimVerdict.SUPPORTED
    assert result.primary_state is EvidenceState.SUPPORTED


def test_sufficient_consistent_refuted_maps_to_refuted() -> None:
    """证据充分且倾向反驳 -> REFUTED。"""
    result = diagnose_evaluator_result(_evaluator_result(0.1, 0.7, 0.2), THRESHOLDS)

    assert result.verdict is ClaimVerdict.REFUTED
    assert result.primary_state is EvidenceState.REFUTED


def test_sufficient_consistent_tie_has_no_primary_state() -> None:
    """证据充分却分不出倾向 -> primary_state 为 None。"""
    result = diagnose_evaluator_result(_evaluator_result(0.4, 0.4, 0.2), THRESHOLDS)

    assert result.region is DiagnosticRegion.SUFFICIENT_CONSISTENT
    assert result.verdict is ClaimVerdict.UNDETERMINED
    assert result.primary_state is None


def test_insufficient_maps_to_insufficient() -> None:
    """insufficient 区域 -> INSUFFICIENT。"""
    result = diagnose_evaluator_result(_evaluator_result(0.2, 0.1, 0.7), THRESHOLDS)

    assert result.primary_state is EvidenceState.INSUFFICIENT


def test_document_conflict_maps_to_conflicting() -> None:
    """document_conflict 区域 -> CONFLICTING。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.5, 0.3, 0.2, k_doc_weighted=0.7), THRESHOLDS
    )

    assert result.primary_state is EvidenceState.CONFLICTING


def test_mixed_region_maps_to_conflicting_and_keeps_both_flags() -> None:
    """混合区域 -> CONFLICTING，冲突信息不因「证据不足」而丢失。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.2, 0.1, 0.7, k_doc_weighted=0.7), THRESHOLDS
    )

    assert result.primary_state is EvidenceState.CONFLICTING
    assert result.evidence_insufficient is True
    assert result.document_conflict is True


def test_verdict_and_region_are_independent() -> None:
    """区域为 document_conflict 时，融合质量仍可有明确倾向。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.3, 0.5, 0.2, k_doc_weighted=0.7), THRESHOLDS
    )

    assert result.region is DiagnosticRegion.DOCUMENT_CONFLICT
    assert result.verdict is ClaimVerdict.REFUTED
    assert result.primary_state is EvidenceState.CONFLICTING


# --------------------------------------------------------------------------
# 19-23. K_eval 只作为额外警报
# --------------------------------------------------------------------------


def test_low_k_eval_gives_no_disagreement() -> None:
    """K_eval 低于阈值时 evaluator_disagreement=False。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.7, 0.1, 0.2, k_eval=0.1), THRESHOLDS
    )

    assert result.evaluator_disagreement is False


def test_k_eval_exactly_at_threshold_triggers_disagreement() -> None:
    """K_eval 等于阈值时触发警报（>=）。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.7, 0.1, 0.2, k_eval=0.4), THRESHOLDS
    )

    assert result.evaluator_disagreement is True


def test_high_k_eval_triggers_disagreement() -> None:
    """K_eval 高于阈值时触发警报。"""
    result = diagnose_evaluator_result(
        _evaluator_result(0.7, 0.1, 0.2, k_eval=0.8), THRESHOLDS
    )

    assert result.evaluator_disagreement is True


@pytest.mark.parametrize(
    ("m_support", "m_refute", "m_theta", "k_doc"),
    [
        (0.7, 0.1, 0.2, 0.1),
        (0.2, 0.1, 0.7, 0.1),
        (0.5, 0.3, 0.2, 0.7),
        (0.2, 0.1, 0.7, 0.7),
    ],
)
def test_changing_k_eval_only_flips_the_disagreement_flag(
    m_support: float, m_refute: float, m_theta: float, k_doc: float
) -> None:
    """把 K_eval 从 0.1 改到 0.8，只有 evaluator_disagreement 变化。"""
    low = diagnose_evaluator_result(
        _evaluator_result(m_support, m_refute, m_theta, k_doc, k_eval=0.1), THRESHOLDS
    )
    high = diagnose_evaluator_result(
        _evaluator_result(m_support, m_refute, m_theta, k_doc, k_eval=0.8), THRESHOLDS
    )

    assert low.evaluator_disagreement is False
    assert high.evaluator_disagreement is True

    # 除 k_eval 与 evaluator_disagreement 外，其余字段逐一相同。
    changed = {
        field
        for field in DiagnosticResult.model_fields
        if getattr(low, field) != getattr(high, field)
    }
    assert changed == {"k_eval", "evaluator_disagreement"}


# --------------------------------------------------------------------------
# 24. support_refute_margin
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("m_support", "m_refute", "m_theta", "expected"),
    [
        (0.7, 0.1, 0.2, 0.6),
        (0.1, 0.7, 0.2, -0.6),
        (0.4, 0.4, 0.2, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, -1.0),
    ],
)
def test_support_refute_margin(
    m_support: float, m_refute: float, m_theta: float, expected: float
) -> None:
    """margin = m_support - m_refute，正数偏支持、负数偏反驳。"""
    result = diagnose_evaluator_result(
        _evaluator_result(m_support, m_refute, m_theta), THRESHOLDS
    )

    assert result.support_refute_margin == pytest.approx(expected)


# --------------------------------------------------------------------------
# 25-27. 评估器完全冲突
# --------------------------------------------------------------------------


def test_evaluator_total_conflict_region() -> None:
    """评估器完全冲突 -> evaluator_total_conflict。"""
    result = diagnose_evaluator_result(_total_conflict_evaluator_result(), THRESHOLDS)

    assert result.region is DiagnosticRegion.EVALUATOR_TOTAL_CONFLICT
    assert result.verdict is ClaimVerdict.UNDETERMINED
    assert result.primary_state is None
    assert result.evaluator_disagreement is True
    assert result.k_eval == pytest.approx(1.0)


def test_evaluator_total_conflict_masses_are_none() -> None:
    """评估器完全冲突时三个质量与 margin 均为 None。"""
    result = diagnose_evaluator_result(_total_conflict_evaluator_result(), THRESHOLDS)

    assert result.m_support is None
    assert result.m_refute is None
    assert result.m_theta is None
    assert result.support_refute_margin is None


def test_evaluator_total_conflict_is_not_full_ignorance() -> None:
    """完全冲突不会被伪造成 m_theta=1，也不会被写成「证据不足」。"""
    result = diagnose_evaluator_result(_total_conflict_evaluator_result(), THRESHOLDS)

    assert result.m_theta is None  # 而不是 1.0
    assert result.evidence_insufficient is False
    assert result.region is not DiagnosticRegion.INSUFFICIENT


def test_evaluator_total_conflict_preserves_k_doc() -> None:
    """k_doc_weighted 被原样保留，对应的布尔标志也随之设置。"""
    result = diagnose_evaluator_result(
        _total_conflict_evaluator_result(k_doc_weighted=0.65), THRESHOLDS
    )

    assert result.k_doc == pytest.approx(0.65)
    assert result.document_conflict is True


# --------------------------------------------------------------------------
# 28-31. 文档完全冲突
# --------------------------------------------------------------------------


def _document_total_conflict():
    """构造一个文档级完全冲突的聚合结果。"""
    result = aggregate_document_masses(
        [_mass("d1", 1.0, 0.0, 0.0), _mass("d2", 0.0, 1.0, 0.0)]
    )
    assert result.is_total_conflict is True
    return result


def test_document_total_conflict_region_and_state() -> None:
    """文档完全冲突 -> document_total_conflict，primary_state=CONFLICTING。"""
    result = diagnose_document_total_conflict(_document_total_conflict(), THRESHOLDS)

    assert result.region is DiagnosticRegion.DOCUMENT_TOTAL_CONFLICT
    assert result.primary_state is EvidenceState.CONFLICTING
    assert result.verdict is ClaimVerdict.UNDETERMINED


def test_document_total_conflict_fields() -> None:
    """三个质量为 None，k_doc=1、k_eval=0，布尔标志按规格设置。"""
    result = diagnose_document_total_conflict(_document_total_conflict(), THRESHOLDS)

    assert result.m_support is None
    assert result.m_refute is None
    assert result.m_theta is None
    assert result.support_refute_margin is None
    assert result.k_doc == pytest.approx(1.0)
    assert result.k_eval == pytest.approx(0.0)
    assert result.evidence_insufficient is False
    assert result.document_conflict is True
    assert result.evaluator_disagreement is False


def test_document_total_conflict_is_not_full_ignorance() -> None:
    """文档完全冲突不等于完全无知：m_theta 为 None 而不是 1。"""
    result = diagnose_document_total_conflict(_document_total_conflict(), THRESHOLDS)

    assert result.m_theta is None
    assert result.region is not DiagnosticRegion.INSUFFICIENT


def test_non_total_conflict_document_result_is_rejected() -> None:
    """正常的文档聚合结果传给专用函数时被拒绝。"""
    normal = aggregate_document_masses([_mass("d1", 0.6, 0.1, 0.3)])

    with pytest.raises(ValueError, match="只接受文档完全冲突的结果"):
        diagnose_document_total_conflict(normal, THRESHOLDS)


# --------------------------------------------------------------------------
# 32-33. 纯函数性质
# --------------------------------------------------------------------------


def test_gating_module_never_reads_gold_state() -> None:
    """门控模块的实际代码中不出现 gold_state / retrieval_score。"""
    banned = {"gold_state", "retrieval_score", "text", "question", "answer"}
    for name in ("gating", "models"):
        source = pathlib.Path(f"src/rag_ds/diagnostics/{name}.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        stripped = ast.parse(ast.unparse(tree))
        attrs = {n.attr for n in ast.walk(stripped) if isinstance(n, ast.Attribute)}
        names = {n.id for n in ast.walk(stripped) if isinstance(n, ast.Name)}

        assert not (attrs & banned), f"{name}.py 访问了 {attrs & banned}"
        assert not (names & banned), f"{name}.py 引用了 {names & banned}"


def test_diagnostic_result_has_no_gold_state_field() -> None:
    """诊断结果模型里没有 gold_state 字段。"""
    assert "gold_state" not in DiagnosticResult.model_fields


def test_inputs_are_not_modified() -> None:
    """诊断不修改输入对象。"""
    evaluator_result = _evaluator_result(0.5, 0.3, 0.2, k_doc_weighted=0.7, k_eval=0.5)
    document_result = _document_total_conflict()
    thresholds = DiagnosticThresholds()
    before = (
        evaluator_result.model_dump(),
        document_result.model_dump(),
        thresholds.model_dump(),
    )

    diagnose_evaluator_result(evaluator_result, thresholds)
    diagnose_document_total_conflict(document_result, thresholds)

    assert (
        evaluator_result.model_dump(),
        document_result.model_dump(),
        thresholds.model_dump(),
    ) == before


def test_diagnosis_is_deterministic() -> None:
    """相同输入多次诊断结果完全一致。"""
    source = _evaluator_result(0.5, 0.3, 0.2, k_doc_weighted=0.7, k_eval=0.5)

    runs = [diagnose_evaluator_result(source, THRESHOLDS) for _ in range(5)]

    assert all(run == runs[0] for run in runs)


def test_thresholds_are_recorded_in_the_result() -> None:
    """结果里保存了本次实际使用的阈值，便于复现。"""
    custom = DiagnosticThresholds(theta_threshold=0.8, document_conflict_threshold=0.9)

    result = diagnose_evaluator_result(
        _evaluator_result(0.2, 0.1, 0.7, k_doc_weighted=0.7), custom
    )

    assert result.thresholds == custom
    # theta 0.7 < 0.8、k_doc 0.7 < 0.9，因此在这套阈值下都算「低」。
    assert result.region is DiagnosticRegion.SUFFICIENT_CONSISTENT


# --------------------------------------------------------------------------
# 结果模型自身的约束
# --------------------------------------------------------------------------


def test_result_model_rejects_partial_masses() -> None:
    """三个质量不能只缺一部分。"""
    with pytest.raises(ValidationError, match="同时存在或同时为 None"):
        DiagnosticResult(
            sample_id="s1",
            claim_id="c1",
            m_support=0.5,
            m_refute=None,
            m_theta=0.5,
            k_doc=0.0,
            k_eval=0.0,
            region=DiagnosticRegion.SUFFICIENT_CONSISTENT,
            verdict=ClaimVerdict.SUPPORTED,
            primary_state=EvidenceState.SUPPORTED,
            evidence_insufficient=False,
            document_conflict=False,
            evaluator_disagreement=False,
            support_refute_margin=0.0,
            thresholds=THRESHOLDS,
        )


def test_result_model_rejects_masses_in_total_conflict_region() -> None:
    """完全冲突区域不允许带有质量。"""
    with pytest.raises(ValidationError, match="三个质量必须为 None"):
        DiagnosticResult(
            sample_id="s1",
            claim_id="c1",
            m_support=0.5,
            m_refute=0.3,
            m_theta=0.2,
            k_doc=1.0,
            k_eval=0.0,
            region=DiagnosticRegion.DOCUMENT_TOTAL_CONFLICT,
            verdict=ClaimVerdict.UNDETERMINED,
            primary_state=EvidenceState.CONFLICTING,
            evidence_insufficient=False,
            document_conflict=True,
            evaluator_disagreement=False,
            support_refute_margin=0.2,
            thresholds=THRESHOLDS,
        )


def test_result_model_rejects_wrong_margin() -> None:
    """margin 必须等于 m_support - m_refute。"""
    with pytest.raises(ValidationError, match="support_refute_margin 必须等于"):
        DiagnosticResult(
            sample_id="s1",
            claim_id="c1",
            m_support=0.7,
            m_refute=0.1,
            m_theta=0.2,
            k_doc=0.0,
            k_eval=0.0,
            region=DiagnosticRegion.SUFFICIENT_CONSISTENT,
            verdict=ClaimVerdict.SUPPORTED,
            primary_state=EvidenceState.SUPPORTED,
            evidence_insufficient=False,
            document_conflict=False,
            evaluator_disagreement=False,
            support_refute_margin=0.1,
            thresholds=THRESHOLDS,
        )


def test_result_is_immutable() -> None:
    """DiagnosticResult 不可变。"""
    result = diagnose_evaluator_result(_evaluator_result(0.7, 0.1, 0.2), THRESHOLDS)

    with pytest.raises(ValidationError):
        result.region = DiagnosticRegion.INSUFFICIENT  # type: ignore[misc]


# --------------------------------------------------------------------------
# 十三、四类 demo 数据的端到端集成测试
# --------------------------------------------------------------------------


def _diagnose_demo(
    sample: RAGSample,
    predictions,
    claim_id: str,
    evaluator_reliability: float = 1.0,
):
    """手工串联已有模块走完整链路，不构建正式 pipeline。

    按 sample_id 与 claim_id 同时过滤：文档聚合要求同一 claim 下 doc_id
    唯一，混入其他 claim 的预测会被直接拒绝。

    gold_state 全程不参与计算 —— 本函数只接收 contexts 与 predictions。
    """
    contexts = {chunk.doc_id: chunk for chunk in sample.contexts}
    document_masses = [
        document_discounted_mass_from_prediction(p, contexts[p.doc_id])
        for p in predictions
        if p.sample_id == sample.sample_id and p.claim_id == claim_id
    ]
    document_result = aggregate_document_masses(document_masses)
    evaluator_result = aggregate_evaluators(
        [
            EvaluatorEvidence(
                document_result=document_result,
                evaluator_reliability=evaluator_reliability,
            )
        ]
    )
    return diagnose_evaluator_result(evaluator_result, THRESHOLDS)


@pytest.mark.parametrize(
    ("sample_id", "claim_id", "expected_region", "expected_verdict", "expected_state"),
    [
        (
            "demo-001",
            "demo-001-c1",
            DiagnosticRegion.SUFFICIENT_CONSISTENT,
            ClaimVerdict.SUPPORTED,
            EvidenceState.SUPPORTED,
        ),
        (
            "demo-001",
            "demo-001-c2",
            DiagnosticRegion.SUFFICIENT_CONSISTENT,
            ClaimVerdict.SUPPORTED,
            EvidenceState.SUPPORTED,
        ),
        (
            "demo-002",
            "demo-002-c1",
            DiagnosticRegion.SUFFICIENT_CONSISTENT,
            ClaimVerdict.REFUTED,
            EvidenceState.REFUTED,
        ),
        (
            "demo-003",
            "demo-003-c1",
            DiagnosticRegion.INSUFFICIENT,
            ClaimVerdict.UNDETERMINED,
            EvidenceState.INSUFFICIENT,
        ),
        (
            "demo-004",
            "demo-004-c1",
            DiagnosticRegion.DOCUMENT_CONFLICT,
            ClaimVerdict.REFUTED,
            EvidenceState.CONFLICTING,
        ),
    ],
)
def test_demo_end_to_end_diagnosis(
    sample_id: str,
    claim_id: str,
    expected_region: DiagnosticRegion,
    expected_verdict: ClaimVerdict,
    expected_state: EvidenceState,
) -> None:
    """五条 demo claim 走完整链路后的诊断结果符合预期。"""
    samples = {s.sample_id: s for s in load_samples(DEMO_PATH)}
    predictions = load_relation_predictions(MOCK_PATH)

    result = _diagnose_demo(samples[sample_id], predictions, claim_id)

    assert result.region is expected_region
    assert result.verdict is expected_verdict
    assert result.primary_state is expected_state


def test_demo_primary_state_matches_gold_state() -> None:
    """四类 demo 的 primary_state 与标注一致。

    gold_state 只在这里、诊断完成之后用于比较，计算过程从未接触它。
    """
    samples = {s.sample_id: s for s in load_samples(DEMO_PATH)}
    predictions = load_relation_predictions(MOCK_PATH)

    for sample_id, sample in samples.items():
        for claim in sample.claims:
            result = _diagnose_demo(sample, predictions, claim.claim_id)

            assert result.primary_state is sample.gold_state, (
                f"{sample_id}/{claim.claim_id}"
            )


def test_demo_diagnosis_is_unaffected_by_gold_state() -> None:
    """把 gold_state 抹成 None，四类 demo 的诊断结果完全不变。"""
    samples = {s.sample_id: s for s in load_samples(DEMO_PATH)}
    predictions = load_relation_predictions(MOCK_PATH)

    for sample in samples.values():
        stripped = RAGSample(
            sample_id=sample.sample_id,
            question=sample.question,
            answer=sample.answer,
            claims=list(sample.claims),
            contexts=list(sample.contexts),
            gold_state=None,
        )

        for claim in sample.claims:
            assert _diagnose_demo(
                stripped, predictions, claim.claim_id
            ) == _diagnose_demo(sample, predictions, claim.claim_id)
