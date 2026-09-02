"""第 13–14 步的测试：指标、阈值搜索、对比实验、消融与导出。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ds.baselines.models import BaselineThresholds
from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments import (
    CLASSIFICATION_ABLATION_VARIANTS,
    DS_METHOD,
    AblationVariant,
    plot_confusion_matrix,
    plot_diagnostic_scatter,
    plot_threshold_sensitivity,
    run_ablation,
    run_comparison,
    write_ablation_csv,
    write_main_results_csv,
    write_predictions_csv,
)
from rag_ds.metrics import (
    GOLD_LABELS,
    UNDETERMINED_LABEL,
    classification_report,
    default_label_universe,
    detection_report,
)
from rag_ds.pipeline import run_pipeline
from rag_ds.tuning import (
    SplitName,
    ThresholdGrid,
    predicted_label,
    rediagnose,
    search_thresholds,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = _PROJECT_ROOT / "data" / "samples" / "demo.jsonl"
MOCK_PATH = _PROJECT_ROOT / "data" / "samples" / "mock_relations.jsonl"

SINGLE_EVALUATOR = "mock_evaluator"


@pytest.fixture
def demo_inputs():
    """demo 样本与预设关系预测。"""
    return load_samples(DEMO_PATH), load_relation_predictions(MOCK_PATH)


# --------------------------------------------------------------------------
# 分类指标
# --------------------------------------------------------------------------


def test_perfect_prediction() -> None:
    """全对时 accuracy 与 macro-F1 都是 1。"""
    labels = list(GOLD_LABELS)
    report = classification_report(labels, labels, method="perfect")

    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.macro_labels == GOLD_LABELS
    assert report.sample_count == 4


def test_macro_labels_default_to_classes_present_in_gold() -> None:
    """默认只在金标准出现过的类上平均，undetermined 不算作一类。"""
    report = classification_report(
        ["supported", "refuted"], ["supported", UNDETERMINED_LABEL], method="m"
    )

    assert report.macro_labels == ("supported", "refuted")
    assert UNDETERMINED_LABEL in report.labels  # 但仍出现在混淆矩阵里


def test_baseline_cannot_score_on_conflicting() -> None:
    """baseline 无法输出 conflicting，该类 F1 为 0 且计入 Macro-F1。"""
    y_true = ["supported", "refuted", "insufficient", "conflicting"]
    y_pred = ["supported", "refuted", "insufficient", "insufficient"]

    report = classification_report(y_true, y_pred, method="baseline")

    conflicting = report.class_metrics("conflicting")
    assert conflicting is not None
    assert conflicting.f1 == pytest.approx(0.0)
    assert conflicting.support == 1
    assert report.macro_f1 < 1.0


def test_restricting_macro_labels_gives_the_three_class_view() -> None:
    """显式传 macro_labels 可得到「只比三类」的补充视角。"""
    y_true = ["supported", "refuted", "insufficient", "conflicting"]
    y_pred = ["supported", "refuted", "insufficient", "insufficient"]

    three_class = classification_report(
        y_true,
        y_pred,
        method="baseline",
        macro_labels=("supported", "refuted", "insufficient"),
    )

    assert three_class.macro_labels == ("supported", "refuted", "insufficient")
    assert three_class.macro_f1 > classification_report(
        y_true, y_pred, method="baseline"
    ).macro_f1


def test_confusion_matrix_is_square_over_the_label_universe() -> None:
    """混淆矩阵行列都用完整标签集。"""
    universe = default_label_universe()
    report = classification_report(["supported"], [UNDETERMINED_LABEL], method="m")

    assert report.labels == universe
    assert len(report.matrix) == len(universe)
    assert all(len(row) == len(universe) for row in report.matrix)
    assert sum(sum(row) for row in report.matrix) == 1


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (["supported"], ["supported", "refuted"], "长度不同"),
        ([], [], "空序列"),
        (["ghost"], ["supported"], "不在标签集中"),
    ],
)
def test_classification_report_rejects_bad_input(
    y_true: list[str], y_pred: list[str], message: str
) -> None:
    """非法输入被拒绝。"""
    with pytest.raises(ValueError, match=message):
        classification_report(y_true, y_pred, method="m")


# --------------------------------------------------------------------------
# 检测指标
# --------------------------------------------------------------------------


def test_detection_perfect_separation() -> None:
    """完全可分时 AUROC = AUPRC = 1。"""
    report = detection_report(
        [0.9, 0.8, 0.2, 0.1], [True, True, False, False], "m_theta", "insufficient"
    )

    assert report.auroc == pytest.approx(1.0)
    assert report.auprc == pytest.approx(1.0)
    assert report.best_f1 == pytest.approx(1.0)
    assert report.positive_count == 2
    assert report.negative_count == 2


def test_detection_without_both_classes_has_no_auroc() -> None:
    """只有单一类别时 AUROC / AUPRC 没有定义，返回 None 而不是 0.5。"""
    report = detection_report([0.9, 0.8], [True, True], "m_theta", "insufficient")

    assert report.auroc is None
    assert report.auprc is None
    assert report.negative_count == 0


def test_detection_rejects_bad_input() -> None:
    """长度不同或为空时报错。"""
    with pytest.raises(ValueError, match="长度不同"):
        detection_report([0.5], [True, False], "s", "p")
    with pytest.raises(ValueError, match="空序列"):
        detection_report([], [], "s", "p")


# --------------------------------------------------------------------------
# 阈值搜索
# --------------------------------------------------------------------------


def test_rediagnose_matches_a_full_rerun(demo_inputs) -> None:
    """重跑门控的结果与整条 pipeline 重算完全一致。"""
    samples, predictions = demo_inputs
    thresholds = DiagnosticThresholds(theta_threshold=0.3, document_conflict_threshold=0.2)

    cached = run_pipeline(samples, predictions, DiagnosticThresholds())
    recomputed = run_pipeline(samples, predictions, thresholds)

    assert [rediagnose(r, thresholds) for r in cached] == [
        r.diagnostic for r in recomputed
    ]


def test_threshold_grid_defaults() -> None:
    """默认四分类网格为 5×5；K_eval 警报阈值固定。"""
    grid = ThresholdGrid()

    candidates = list(grid.candidates())
    assert len(grid) == 25
    assert len(candidates) == 25
    assert {c.evaluator_conflict_threshold for c in candidates} == {0.4}


def test_threshold_grid_rejects_empty_or_out_of_range() -> None:
    """候选为空或越界时被拒绝。"""
    with pytest.raises(ValidationError, match="不能为空"):
        ThresholdGrid(theta_values=())
    with pytest.raises(ValidationError, match="必须位于"):
        ThresholdGrid(theta_values=(1.5,))


def test_search_rejects_the_test_split(demo_inputs) -> None:
    """用测试集选阈值会被直接拒绝。"""
    samples, predictions = demo_inputs
    results = run_pipeline(samples, predictions, DiagnosticThresholds())

    for split in (SplitName.TEST, SplitName.TRAIN):
        with pytest.raises(ValueError, match="只能在验证集上搜索"):
            search_thresholds(results, split)


def test_search_requires_gold_state() -> None:
    """缺少 gold_state 时无法搜索。"""
    from rag_ds.schemas import Claim, ContextChunk, RAGSample

    sample = RAGSample(
        sample_id="s1",
        question="问题？",
        answer="答案。",
        claims=[Claim(claim_id="c1", text="断言。")],
        contexts=[ContextChunk(doc_id="d1", text="文档。")],
    )
    from rag_ds.schemas import RelationPrediction

    predictions = [
        RelationPrediction(
            sample_id="s1",
            claim_id="c1",
            doc_id="d1",
            evaluator="e",
            p_support=0.8,
            p_refute=0.1,
            p_unknown=0.1,
        )
    ]
    results = run_pipeline([sample], predictions, DiagnosticThresholds())

    with pytest.raises(ValueError, match="缺少 gold_state"):
        search_thresholds(results, SplitName.VALIDATION)


def test_search_is_deterministic_and_sorted(demo_inputs) -> None:
    """结果按 Macro-F1 降序排列，并列时取阈值字典序最小者。"""
    samples, predictions = demo_inputs
    results = run_pipeline(samples, predictions, DiagnosticThresholds())

    first = search_thresholds(results, SplitName.VALIDATION)
    second = search_thresholds(results, SplitName.VALIDATION)

    assert first == second
    scores = [c.macro_f1 for c in first.candidates]
    assert scores == sorted(scores, reverse=True)
    assert first.best is first.candidates[0]


def test_predicted_label_maps_none_to_undetermined() -> None:
    """primary_state 为 None 时映射为 undetermined，不硬塞进四类。"""
    from rag_ds.diagnostics.models import (
        ClaimVerdict,
        DiagnosticRegion,
        DiagnosticResult,
    )

    diagnostic = DiagnosticResult(
        sample_id="s1",
        claim_id="c1",
        m_support=0.4,
        m_refute=0.4,
        m_theta=0.2,
        k_doc=0.0,
        k_eval=0.0,
        region=DiagnosticRegion.SUFFICIENT_CONSISTENT,
        verdict=ClaimVerdict.UNDETERMINED,
        primary_state=None,
        evidence_insufficient=False,
        document_conflict=False,
        evaluator_disagreement=False,
        support_refute_margin=0.0,
        thresholds=DiagnosticThresholds(),
    )

    assert predicted_label(diagnostic) == UNDETERMINED_LABEL


# --------------------------------------------------------------------------
# 对比实验
# --------------------------------------------------------------------------


def test_comparison_covers_all_methods(demo_inputs) -> None:
    """四个方法都在同一批数据上跑，记录数一致。"""
    samples, predictions = demo_inputs

    report, records = run_comparison(
        samples,
        predictions,
        DiagnosticThresholds(),
        BaselineThresholds(),
        SINGLE_EVALUATOR,
    )

    assert report.methods == (
        DS_METHOD,
        "weighted_average",
        "majority_vote",
        "single_evaluator",
    )
    assert len(records) == 5 * 4
    assert report.claim_count == 5
    assert len(report.classification) == 4
    assert len(report.insufficiency_detection) == 4
    assert len(report.conflict_detection) == 4


def test_ds_beats_baselines_on_the_demo_set(demo_inputs) -> None:
    """demo 上 D-S 的 Macro-F1 高于三个 baseline。"""
    samples, predictions = demo_inputs
    report, _ = run_comparison(
        samples,
        predictions,
        DiagnosticThresholds(),
        BaselineThresholds(),
        SINGLE_EVALUATOR,
    )

    ds = report.classification_for(DS_METHOD)
    assert ds is not None
    others = [r for r in report.classification if r.method != DS_METHOD]
    assert all(ds.macro_f1 > r.macro_f1 for r in others)


def test_comparison_requires_gold_state() -> None:
    """缺少 gold_state 的样本无法参与对比。"""
    from rag_ds.schemas import Claim, RAGSample

    sample = RAGSample(
        sample_id="s1",
        question="问题？",
        answer="答案。",
        claims=[Claim(claim_id="c1", text="断言。")],
    )

    with pytest.raises(ValueError, match="缺少 gold_state"):
        run_comparison(
            [sample], [], DiagnosticThresholds(), BaselineThresholds(), "e"
        )


# --------------------------------------------------------------------------
# 消融实验
# --------------------------------------------------------------------------


def test_ablation_runs_all_variants(demo_inputs) -> None:
    """全部变体都能跑出结果，full 排在最前且 delta 为 0。"""
    samples, predictions = demo_inputs

    results = run_ablation(samples, predictions, DiagnosticThresholds())

    assert len(results) == len(CLASSIFICATION_ABLATION_VARIANTS)
    assert results[0].variant is AblationVariant.FULL
    assert results[0].macro_f1_delta == pytest.approx(0.0)


def test_classification_ablation_rejects_eval_alert_variant(demo_inputs) -> None:
    """K_eval 只控制告警，不能伪装成四分类 Macro-F1 消融。"""
    samples, predictions = demo_inputs

    with pytest.raises(ValueError, match="不能用四分类 Macro-F1"):
        run_ablation(
            samples,
            predictions,
            DiagnosticThresholds(),
            variants=(
                AblationVariant.FULL,
                AblationVariant.NO_EVAL_CONFLICT_ALERT,
            ),
        )


def test_removing_gates_hurts_macro_f1(demo_inputs) -> None:
    """去掉门控维度会让 Macro-F1 下降 —— 这正是消融要展示的。"""
    samples, predictions = demo_inputs
    results = {r.variant: r for r in run_ablation(samples, predictions, DiagnosticThresholds())}

    assert results[AblationVariant.NO_THETA_GATE].macro_f1_delta < 0
    assert results[AblationVariant.NO_DOC_CONFLICT_GATE].macro_f1_delta < 0
    assert (
        results[AblationVariant.NO_TWO_DIMENSIONAL_GATE].macro_f1_delta
        < results[AblationVariant.NO_THETA_GATE].macro_f1_delta
    )


def test_ablation_does_not_modify_inputs(demo_inputs) -> None:
    """消融不修改传入的样本与预测。"""
    samples, predictions = demo_inputs
    before = (
        [s.model_dump() for s in samples],
        [p.model_dump() for p in predictions],
    )

    run_ablation(samples, predictions, DiagnosticThresholds())

    assert (
        [s.model_dump() for s in samples],
        [p.model_dump() for p in predictions],
    ) == before


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def test_export_csv_files(tmp_path: Path, demo_inputs) -> None:
    """三个 CSV 都能写出且字段完整。"""
    samples, predictions = demo_inputs
    report, records = run_comparison(
        samples,
        predictions,
        DiagnosticThresholds(),
        BaselineThresholds(),
        SINGLE_EVALUATOR,
    )
    ablation = run_ablation(samples, predictions, DiagnosticThresholds())

    main_rows = write_main_results_csv(tmp_path / "main.csv", report)
    ablation_rows = write_ablation_csv(tmp_path / "ablation.csv", ablation)
    detail_rows = write_predictions_csv(tmp_path / "detail.csv", records)

    assert main_rows == 4 * len(default_label_universe())
    assert ablation_rows == len(CLASSIFICATION_ABLATION_VARIANTS)
    assert detail_rows == 20

    with (tmp_path / "main.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(iter(csv.DictReader(handle)))
    for column in ("method", "macro_f1", "label", "f1", "conflict_auroc"):
        assert column in row


def test_export_figures(tmp_path: Path, demo_inputs) -> None:
    """三张图都能生成且非空。"""
    samples, predictions = demo_inputs
    report, records = run_comparison(
        samples,
        predictions,
        DiagnosticThresholds(),
        BaselineThresholds(),
        SINGLE_EVALUATOR,
    )
    ds_report = report.classification_for(DS_METHOD)
    assert ds_report is not None
    search = search_thresholds(
        run_pipeline(samples, predictions, DiagnosticThresholds()),
        SplitName.VALIDATION,
    )

    paths = [
        plot_confusion_matrix(tmp_path / "cm.png", ds_report),
        plot_diagnostic_scatter(tmp_path / "scatter.png", records),
        plot_threshold_sensitivity(tmp_path / "sensitivity.png", search),
    ]

    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 1000
