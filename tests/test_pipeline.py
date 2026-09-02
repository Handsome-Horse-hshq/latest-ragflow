"""第十阶段离线 MVP pipeline 的测试。

所有临时输出都写在 pytest 的 ``tmp_path`` 下，不污染项目的
``outputs/`` 目录。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rag_ds.config import load_pipeline_config
from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.diagnostics.models import DiagnosticRegion, DiagnosticThresholds
from rag_ds.pipeline import (
    CSV_COLUMNS,
    DuplicateRelationPredictionError,
    InconsistentEvaluatorReliabilityError,
    MissingRelationPredictionError,
    NoClaimsError,
    ReferentialIntegrityError,
    run_pipeline,
    run_pipeline_from_config,
    write_pipeline_csv,
    write_pipeline_jsonl,
)
from rag_ds.pipeline_results import ClaimPipelineResult, PipelineStatus
from rag_ds.schemas import (
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data" / "samples"
DEMO_PATH = _DATA_DIR / "demo.jsonl"
MOCK_PATH = _DATA_DIR / "mock_relations.jsonl"

THRESHOLDS = DiagnosticThresholds()


@pytest.fixture
def demo_samples() -> list[RAGSample]:
    """demo.jsonl 的全部样本。"""
    return load_samples(DEMO_PATH)


@pytest.fixture
def demo_predictions() -> list[RelationPrediction]:
    """mock_relations.jsonl 的全部预设。"""
    return load_relation_predictions(MOCK_PATH)


@pytest.fixture
def demo_results(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> list[ClaimPipelineResult]:
    """跑一遍完整 pipeline。"""
    return run_pipeline(demo_samples, demo_predictions, THRESHOLDS)


def _by_claim(results: list[ClaimPipelineResult]) -> dict[str, ClaimPipelineResult]:
    """按 claim_id 索引结果。"""
    return {result.claim_id: result for result in results}


def _sample(
    sample_id: str = "s1",
    claim_ids: tuple[str, ...] = ("c1",),
    doc_ids: tuple[str, ...] = ("d1",),
    reliability: float = 1.0,
    gold_state: EvidenceState | None = None,
) -> RAGSample:
    """构造一个可控的样本。"""
    return RAGSample(
        sample_id=sample_id,
        question="问题？",
        answer="答案。",
        claims=[Claim(claim_id=cid, text=f"断言 {cid}。") for cid in claim_ids],
        contexts=[
            ContextChunk(doc_id=did, text=f"文档 {did}。", reliability=reliability)
            for did in doc_ids
        ],
        gold_state=gold_state,
    )


def _prediction(
    doc_id: str,
    probabilities: tuple[float, float, float] = (0.8, 0.1, 0.1),
    sample_id: str = "s1",
    claim_id: str = "c1",
    evaluator: str = "mock_a",
    evaluator_reliability: float = 1.0,
) -> RelationPrediction:
    """构造一条关系预测。"""
    return RelationPrediction(
        sample_id=sample_id,
        claim_id=claim_id,
        doc_id=doc_id,
        evaluator=evaluator,
        p_support=probabilities[0],
        p_refute=probabilities[1],
        p_unknown=probabilities[2],
        evaluator_reliability=evaluator_reliability,
    )


def _write_config(
    tmp_path: Path,
    *,
    overwrite: bool = False,
    samples: Path = DEMO_PATH,
    predictions: Path = MOCK_PATH,
) -> Path:
    """在 tmp_path 下写一份配置，输出也指向 tmp_path。"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "paths:\n"
        f"  samples: {samples.as_posix()}\n"
        f"  relation_predictions: {predictions.as_posix()}\n"
        "  output_jsonl: ../out/diagnostics.jsonl\n"
        "  output_csv: ../out/diagnostics.csv\n"
        "diagnostics:\n"
        "  theta_threshold: 0.5\n"
        "  document_conflict_threshold: 0.4\n"
        "  evaluator_conflict_threshold: 0.4\n"
        "  tie_tolerance: 0.000001\n"
        "output:\n"
        f"  overwrite: {str(overwrite).lower()}\n",
        encoding="utf-8",
    )
    return config_path


# --------------------------------------------------------------------------
# 1-8. demo 数据的正常结果
# --------------------------------------------------------------------------


def test_demo_produces_one_result_per_claim(
    demo_samples: list[RAGSample], demo_results: list[ClaimPipelineResult]
) -> None:
    """每条 claim 独立产出一条结果。"""
    expected = sum(len(sample.claims) for sample in demo_samples)

    assert len(demo_results) == expected == 5
    assert [r.claim_id for r in demo_results] == [
        "demo-001-c1",
        "demo-001-c2",
        "demo-002-c1",
        "demo-003-c1",
        "demo-004-c1",
    ]


@pytest.mark.parametrize(
    ("claim_id", "expected_state", "expected_region"),
    [
        ("demo-001-c1", EvidenceState.SUPPORTED, DiagnosticRegion.SUFFICIENT_CONSISTENT),
        ("demo-001-c2", EvidenceState.SUPPORTED, DiagnosticRegion.SUFFICIENT_CONSISTENT),
        ("demo-002-c1", EvidenceState.REFUTED, DiagnosticRegion.SUFFICIENT_CONSISTENT),
        ("demo-003-c1", EvidenceState.INSUFFICIENT, DiagnosticRegion.INSUFFICIENT),
        ("demo-004-c1", EvidenceState.CONFLICTING, DiagnosticRegion.DOCUMENT_CONFLICT),
    ],
)
def test_demo_primary_states(
    demo_results: list[ClaimPipelineResult],
    claim_id: str,
    expected_state: EvidenceState,
    expected_region: DiagnosticRegion,
) -> None:
    """四类 demo 的 primary_state 与区域符合预期。"""
    result = _by_claim(demo_results)[claim_id]

    assert result.status is PipelineStatus.NORMAL
    assert result.diagnostic.primary_state is expected_state
    assert result.diagnostic.region is expected_region


def test_demo_primary_state_matches_gold_state(
    demo_results: list[ClaimPipelineResult],
) -> None:
    """诊断结论与标注一致（gold_state 只在此处比较）。"""
    for result in demo_results:
        assert result.diagnostic.primary_state is result.gold_state, result.claim_id


def test_single_evaluator_gives_zero_k_eval(
    demo_results: list[ClaimPipelineResult],
) -> None:
    """demo 只有一个评估器，K_eval 恒为 0。"""
    for result in demo_results:
        assert result.evaluators == ("mock_evaluator",)
        assert result.diagnostic.k_eval == pytest.approx(0.0)
        assert result.diagnostic.evaluator_disagreement is False


def test_conflicting_sample_has_higher_k_doc(
    demo_results: list[ClaimPipelineResult],
) -> None:
    """conflicting 样例的 K_doc 明显高于 supported 样例。"""
    by_claim = _by_claim(demo_results)

    assert by_claim["demo-004-c1"].diagnostic.k_doc > 0.4
    assert by_claim["demo-001-c1"].diagnostic.k_doc < 0.1
    assert (
        by_claim["demo-004-c1"].diagnostic.k_doc
        > by_claim["demo-001-c1"].diagnostic.k_doc
    )


def test_results_keep_intermediate_document_results(
    demo_results: list[ClaimPipelineResult],
) -> None:
    """每条结果都保留文档级与评估器级中间结果。"""
    for result in demo_results:
        assert len(result.document_results) == len(result.evaluators)
        assert result.evaluator_result is not None
        assert result.evaluator_result.evaluators == result.evaluators
        for document_result in result.document_results:
            assert document_result.claim_id == result.claim_id


def test_document_order_follows_sample_contexts(
    demo_samples: list[RAGSample], demo_results: list[ClaimPipelineResult]
) -> None:
    """文档处理顺序与 sample.contexts 原始顺序一致。"""
    samples = {sample.sample_id: sample for sample in demo_samples}

    for result in demo_results:
        expected = tuple(c.doc_id for c in samples[result.sample_id].contexts)
        for document_result in result.document_results:
            assert document_result.document_ids == expected


def test_gold_state_does_not_affect_the_pipeline(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> None:
    """抹掉 gold_state 后，诊断结果逐字段不变。"""
    stripped = [
        RAGSample(
            sample_id=sample.sample_id,
            question=sample.question,
            answer=sample.answer,
            reference_answer=sample.reference_answer,
            claims=list(sample.claims),
            contexts=list(sample.contexts),
            gold_state=None,
        )
        for sample in demo_samples
    ]

    with_label = run_pipeline(demo_samples, demo_predictions, THRESHOLDS)
    without_label = run_pipeline(stripped, demo_predictions, THRESHOLDS)

    assert [r.diagnostic for r in without_label] == [r.diagnostic for r in with_label]
    assert all(r.gold_state is None for r in without_label)


# --------------------------------------------------------------------------
# 9-12. 无文档 / 无 claim
# --------------------------------------------------------------------------


def test_claim_without_contexts_gives_no_contexts_status() -> None:
    """没有检索文档时状态为 no_contexts，且不做任何聚合。"""
    results = run_pipeline([_sample(doc_ids=())], [], THRESHOLDS)

    result = results[0]
    assert result.status is PipelineStatus.NO_CONTEXTS
    assert result.context_count == 0
    assert result.evaluators == ()
    assert result.document_results == ()
    assert result.evaluator_result is None


def test_no_contexts_gives_full_ignorance() -> None:
    """无文档时 m_theta=1，区域为 insufficient。"""
    diagnostic = run_pipeline([_sample(doc_ids=())], [], THRESHOLDS)[0].diagnostic

    assert diagnostic.m_support == pytest.approx(0.0)
    assert diagnostic.m_refute == pytest.approx(0.0)
    assert diagnostic.m_theta == pytest.approx(1.0)
    assert diagnostic.region is DiagnosticRegion.INSUFFICIENT
    assert diagnostic.primary_state is EvidenceState.INSUFFICIENT
    assert diagnostic.evidence_insufficient is True
    assert diagnostic.document_conflict is False


def test_sample_without_claims_is_rejected() -> None:
    """样本没有 claim 时抛出 NoClaimsError，不做自动抽取。"""
    empty = RAGSample(sample_id="s1", question="问题？", answer="答案。")

    with pytest.raises(NoClaimsError, match="s1"):
        run_pipeline([empty], [], THRESHOLDS)


# --------------------------------------------------------------------------
# 13-18. 完整性检查
# --------------------------------------------------------------------------


def test_missing_relation_prediction_is_rejected() -> None:
    """评估器未覆盖全部文档时抛出 MissingRelationPredictionError。"""
    sample = _sample(doc_ids=("d1", "d2", "d3"))
    predictions = [_prediction("d1"), _prediction("d2")]  # 缺 d3

    with pytest.raises(MissingRelationPredictionError, match="d3"):
        run_pipeline([sample], predictions, THRESHOLDS)


def test_missing_predictions_for_second_evaluator_is_rejected() -> None:
    """有两个评估器时，每个都必须覆盖全部文档。"""
    sample = _sample(doc_ids=("d1", "d2"))
    predictions = [
        _prediction("d1", evaluator="mock_a"),
        _prediction("d2", evaluator="mock_a"),
        _prediction("d1", evaluator="mock_b"),  # mock_b 缺 d2
    ]

    with pytest.raises(MissingRelationPredictionError, match="mock_b"):
        run_pipeline([sample], predictions, THRESHOLDS)


def test_claim_with_contexts_but_no_predictions_is_rejected() -> None:
    """有文档却完全没有预测时同样报错，不会静默产出空结果。"""
    with pytest.raises(MissingRelationPredictionError, match="没有任何关系预测"):
        run_pipeline([_sample(doc_ids=("d1",))], [], THRESHOLDS)


def test_duplicate_prediction_is_rejected() -> None:
    """同一查询键重复时抛出 DuplicateRelationPredictionError。"""
    sample = _sample(doc_ids=("d1",))
    predictions = [_prediction("d1"), _prediction("d1", (0.1, 0.8, 0.1))]

    with pytest.raises(DuplicateRelationPredictionError, match="d1"):
        run_pipeline([sample], predictions, THRESHOLDS)


def test_unknown_sample_reference_is_rejected() -> None:
    """预测引用不存在的 sample_id 时被拒绝。"""
    with pytest.raises(ReferentialIntegrityError, match="sample_id"):
        run_pipeline(
            [_sample(doc_ids=("d1",))],
            [_prediction("d1"), _prediction("d1", sample_id="ghost")],
            THRESHOLDS,
        )


def test_unknown_claim_reference_is_rejected() -> None:
    """预测引用不存在的 claim_id 时被拒绝。"""
    with pytest.raises(ReferentialIntegrityError, match="claim_id"):
        run_pipeline(
            [_sample(doc_ids=("d1",))],
            [_prediction("d1"), _prediction("d1", claim_id="ghost")],
            THRESHOLDS,
        )


def test_unknown_document_reference_is_rejected() -> None:
    """预测引用不存在的 doc_id 时被拒绝。"""
    with pytest.raises(ReferentialIntegrityError, match="doc_id"):
        run_pipeline(
            [_sample(doc_ids=("d1",))],
            [_prediction("d1"), _prediction("ghost")],
            THRESHOLDS,
        )


def test_inconsistent_evaluator_reliability_is_rejected() -> None:
    """同一评估器在不同文档记录了不同可靠性时被拒绝。"""
    sample = _sample(doc_ids=("d1", "d2"))
    predictions = [
        _prediction("d1", evaluator_reliability=0.9),
        _prediction("d2", evaluator_reliability=0.5),
    ]

    with pytest.raises(InconsistentEvaluatorReliabilityError, match="mock_a"):
        run_pipeline([sample], predictions, THRESHOLDS)


def test_duplicate_sample_id_is_rejected() -> None:
    """sample_id 重复时被拒绝。"""
    with pytest.raises(Exception, match="sample_id 重复"):
        run_pipeline([_sample(doc_ids=()), _sample(doc_ids=())], [], THRESHOLDS)


# --------------------------------------------------------------------------
# 19-22. 完全冲突
# --------------------------------------------------------------------------


def test_document_total_conflict_status() -> None:
    """文档级完全冲突产生 document_total_conflict，且不再做评估器融合。"""
    sample = _sample(doc_ids=("d1", "d2"))
    predictions = [
        _prediction("d1", (1.0, 0.0, 0.0)),
        _prediction("d2", (0.0, 1.0, 0.0)),
    ]

    result = run_pipeline([sample], predictions, THRESHOLDS)[0]

    assert result.status is PipelineStatus.DOCUMENT_TOTAL_CONFLICT
    assert result.evaluator_result is None
    assert result.diagnostic.region is DiagnosticRegion.DOCUMENT_TOTAL_CONFLICT
    assert result.diagnostic.primary_state is EvidenceState.CONFLICTING
    assert result.diagnostic.m_theta is None  # 不伪造成完全无知
    assert result.document_results[0].is_total_conflict is True


def test_evaluator_total_conflict_status() -> None:
    """评估器级完全冲突产生 evaluator_total_conflict。"""
    sample = _sample(doc_ids=("d1",))
    predictions = [
        _prediction("d1", (1.0, 0.0, 0.0), evaluator="mock_a"),
        _prediction("d1", (0.0, 1.0, 0.0), evaluator="mock_b"),
    ]

    result = run_pipeline([sample], predictions, THRESHOLDS)[0]

    assert result.status is PipelineStatus.EVALUATOR_TOTAL_CONFLICT
    assert result.evaluators == ("mock_a", "mock_b")
    assert result.diagnostic.region is DiagnosticRegion.EVALUATOR_TOTAL_CONFLICT
    assert result.diagnostic.primary_state is None
    assert result.diagnostic.m_theta is None
    assert result.diagnostic.k_eval == pytest.approx(1.0)


def test_two_evaluators_are_folded_in_sorted_order() -> None:
    """多评估器按名称排序融合，与预测文件行序无关。"""
    sample = _sample(doc_ids=("d1",))
    forward = [
        _prediction("d1", evaluator="mock_b"),
        _prediction("d1", (0.7, 0.2, 0.1), evaluator="mock_a"),
    ]

    result = run_pipeline([sample], forward, THRESHOLDS)[0]

    assert result.evaluators == ("mock_a", "mock_b")
    assert run_pipeline([sample], list(reversed(forward)), THRESHOLDS)[0] == result


# --------------------------------------------------------------------------
# 23-29. 输出
# --------------------------------------------------------------------------


def test_jsonl_has_one_line_per_claim(
    tmp_path: Path, demo_results: list[ClaimPipelineResult]
) -> None:
    """JSONL 每条 claim 一行。"""
    target = tmp_path / "out.jsonl"

    assert write_pipeline_jsonl(target, demo_results) == 5

    lines = [
        line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 5


def test_jsonl_round_trips_and_keeps_nested_details(
    tmp_path: Path, demo_results: list[ClaimPipelineResult]
) -> None:
    """JSONL 能重新解析回模型，且保留嵌套中间结果与中文。"""
    target = tmp_path / "out.jsonl"
    write_pipeline_jsonl(target, demo_results)

    raw = target.read_text(encoding="utf-8")
    assert "\\u" not in raw
    assert "青蒿素" in raw

    restored = [
        ClaimPipelineResult.model_validate(json.loads(line))
        for line in raw.splitlines()
        if line.strip()
    ]
    assert restored == demo_results
    assert restored[0].document_results[0].steps
    assert restored[0].evaluator_result is not None


def test_csv_columns_and_content(
    tmp_path: Path, demo_results: list[ClaimPipelineResult]
) -> None:
    """CSV 含规定字段，每条 claim 一行。"""
    target = tmp_path / "out.csv"

    assert write_pipeline_csv(target, demo_results) == 5

    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 5
    assert list(rows[0].keys()) == list(CSV_COLUMNS)
    assert rows[0]["sample_id"] == "demo-001"
    assert rows[0]["evaluators"] == "mock_evaluator"
    assert rows[0]["primary_state"] == "supported"
    assert rows[0]["gold_state"] == "supported"
    assert float(rows[0]["m_support"]) > 0.5


def test_csv_leaves_mass_columns_empty_on_total_conflict(tmp_path: Path) -> None:
    """完全冲突状态下三个质量列留空，而不是写 0 或 None。"""
    sample = _sample(doc_ids=("d1", "d2"))
    predictions = [
        _prediction("d1", (1.0, 0.0, 0.0)),
        _prediction("d2", (0.0, 1.0, 0.0)),
    ]
    results = run_pipeline([sample], predictions, THRESHOLDS)
    target = tmp_path / "out.csv"
    write_pipeline_csv(target, results)

    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(iter(csv.DictReader(handle)))

    assert row["m_support"] == ""
    assert row["m_refute"] == ""
    assert row["m_theta"] == ""
    assert row["status"] == "document_total_conflict"
    assert row["primary_state"] == "conflicting"
    assert row["gold_state"] == ""


def test_csv_joins_multiple_evaluators(tmp_path: Path) -> None:
    """多个评估器用 | 连接。"""
    sample = _sample(doc_ids=("d1",))
    predictions = [
        _prediction("d1", evaluator="mock_a"),
        _prediction("d1", (0.7, 0.2, 0.1), evaluator="mock_b"),
    ]
    target = tmp_path / "out.csv"
    write_pipeline_csv(target, run_pipeline([sample], predictions, THRESHOLDS))

    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(iter(csv.DictReader(handle)))

    assert row["evaluators"] == "mock_a|mock_b"


@pytest.mark.parametrize("writer", [write_pipeline_jsonl, write_pipeline_csv])
def test_writers_refuse_to_overwrite_by_default(
    tmp_path: Path, demo_results: list[ClaimPipelineResult], writer
) -> None:
    """默认拒绝覆盖已存在的文件。"""
    target = tmp_path / "out.dat"
    writer(target, demo_results)

    with pytest.raises(FileExistsError):
        writer(target, demo_results)

    assert writer(target, demo_results, overwrite=True) == 5


def test_failed_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    """写入中途失败时清理临时文件，也不产生目标文件。"""

    def exploding():
        yield from ()
        raise RuntimeError("模拟写入中途失败")

    for writer in (write_pipeline_jsonl, write_pipeline_csv):
        target = tmp_path / "never.dat"
        with pytest.raises(RuntimeError, match="模拟写入中途失败"):
            writer(target, exploding())

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# 30-32. 配置与摘要
# --------------------------------------------------------------------------


def test_config_paths_resolve_against_the_config_directory(tmp_path: Path) -> None:
    """相对路径以配置文件所在目录为基准解析。"""
    config_path = _write_config(tmp_path)

    config = load_pipeline_config(config_path)

    assert config.paths.output_jsonl == (tmp_path / "out" / "diagnostics.jsonl")
    assert config.paths.output_csv == (tmp_path / "out" / "diagnostics.csv")
    assert config.paths.output_jsonl.parent.is_dir()  # 输出目录被自动创建


def test_config_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """切换当前工作目录不改变配置解析结果。"""
    config_path = _write_config(tmp_path)
    from_root = load_pipeline_config(config_path)

    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    assert load_pipeline_config(config_path) == from_root


def test_config_rejects_unknown_field(tmp_path: Path) -> None:
    """配置中出现未知字段时被拒绝。"""
    config_path = _write_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "api_key: secret\n", encoding="utf-8"
    )

    with pytest.raises(Exception, match="api_key"):
        load_pipeline_config(config_path)


def test_config_rejects_missing_input_file(tmp_path: Path) -> None:
    """输入文件不存在时报出清晰错误。"""
    config_path = _write_config(tmp_path, samples=tmp_path / "ghost.jsonl")

    with pytest.raises(FileNotFoundError, match="paths.samples"):
        load_pipeline_config(config_path)


def test_run_from_config_produces_both_outputs_and_summary(tmp_path: Path) -> None:
    """按配置运行后生成两份输出，摘要统计正确。"""
    config_path = _write_config(tmp_path)

    summary = run_pipeline_from_config(config_path)

    assert summary.sample_count == 4
    assert summary.claim_count == 5
    assert summary.normal_count == 5
    assert summary.no_contexts_count == 0
    assert summary.document_total_conflict_count == 0
    assert summary.evaluator_total_conflict_count == 0
    assert summary.primary_state_counts == {
        "supported": 2,
        "refuted": 1,
        "insufficient": 1,
        "conflicting": 1,
    }
    assert summary.region_counts == {
        "sufficient_consistent": 3,
        "insufficient": 1,
        "document_conflict": 1,
    }
    assert Path(summary.output_jsonl).is_file()
    assert Path(summary.output_csv).is_file()


def test_run_from_config_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    """默认不覆盖已有输出；overwrite=True 时可以覆盖。"""
    config_path = _write_config(tmp_path)
    run_pipeline_from_config(config_path)

    with pytest.raises(FileExistsError):
        run_pipeline_from_config(config_path)

    assert run_pipeline_from_config(config_path, overwrite=True).claim_count == 5


def test_run_from_config_honours_overwrite_in_yaml(tmp_path: Path) -> None:
    """配置里的 output.overwrite 生效。"""
    config_path = _write_config(tmp_path, overwrite=True)
    run_pipeline_from_config(config_path)

    assert run_pipeline_from_config(config_path).claim_count == 5


def test_failed_run_leaves_no_partial_output(tmp_path: Path) -> None:
    """计算阶段失败时不写出任何文件。"""
    broken = tmp_path / "broken_samples.jsonl"
    sample = _sample(doc_ids=("d1",))
    broken.write_text(
        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    empty_predictions = tmp_path / "empty.jsonl"
    empty_predictions.write_text("", encoding="utf-8")

    config_path = _write_config(
        tmp_path, samples=broken, predictions=empty_predictions
    )

    with pytest.raises(MissingRelationPredictionError):
        run_pipeline_from_config(config_path)

    output_dir = tmp_path / "out"
    assert list(output_dir.iterdir()) == []


def test_run_from_config_does_not_half_write_on_overwrite_conflict(
    tmp_path: Path,
) -> None:
    """只有 CSV 已存在时，JSONL 也不会被抢先写出。"""
    config_path = _write_config(tmp_path)
    csv_path = tmp_path / "out" / "diagnostics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("占位\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_pipeline_from_config(config_path)

    assert not (tmp_path / "out" / "diagnostics.jsonl").exists()
    assert csv_path.read_text(encoding="utf-8") == "占位\n"


def test_summary_counts_special_statuses(tmp_path: Path) -> None:
    """摘要正确统计无文档与完全冲突状态。"""
    samples_path = tmp_path / "samples.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    no_context = _sample(sample_id="s-empty", doc_ids=())
    conflicted = _sample(sample_id="s-conflict", doc_ids=("d1", "d2"))
    samples_path.write_text(
        "\n".join(
            json.dumps(s.model_dump(mode="json"), ensure_ascii=False)
            for s in (no_context, conflicted)
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(
            json.dumps(p.model_dump(mode="json"), ensure_ascii=False)
            for p in (
                _prediction("d1", (1.0, 0.0, 0.0), sample_id="s-conflict"),
                _prediction("d2", (0.0, 1.0, 0.0), sample_id="s-conflict"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_pipeline_from_config(
        _write_config(tmp_path, samples=samples_path, predictions=predictions_path)
    )

    assert summary.sample_count == 2
    assert summary.claim_count == 2
    assert summary.no_contexts_count == 1
    assert summary.document_total_conflict_count == 1
    assert summary.normal_count == 0
    assert summary.region_counts == {
        "insufficient": 1,
        "document_total_conflict": 1,
    }


# --------------------------------------------------------------------------
# 项目自带的 demo 配置
# --------------------------------------------------------------------------


def test_shipped_demo_config_is_loadable() -> None:
    """configs/demo.yaml 可以直接加载，指向项目内的真实文件。"""
    config = load_pipeline_config(_PROJECT_ROOT / "configs" / "demo.yaml")

    assert config.paths.samples == DEMO_PATH
    assert config.paths.relation_predictions == MOCK_PATH
    assert config.output.overwrite is False
    assert config.diagnostics == DiagnosticThresholds()
