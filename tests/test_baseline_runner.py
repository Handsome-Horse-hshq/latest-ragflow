"""第十一阶段 baseline 批量运行与输出的测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rag_ds.baselines.config import load_baseline_config
from rag_ds.baselines.models import (
    BaselineMethod,
    BaselinePrediction,
    BaselineThresholds,
)
from rag_ds.baselines.runner import (
    BASELINE_CSV_COLUMNS,
    run_baselines,
    run_baselines_from_config,
    write_baseline_csv,
    write_baseline_jsonl,
)
from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.integrity import MissingRelationPredictionError
from rag_ds.pipeline import run_pipeline
from rag_ds.schemas import EvidenceState, RAGSample, RelationPrediction

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data" / "samples"
DEMO_PATH = _DATA_DIR / "demo.jsonl"
MOCK_PATH = _DATA_DIR / "mock_relations.jsonl"

THRESHOLDS = BaselineThresholds()
SINGLE_EVALUATOR = "mock_evaluator"


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
) -> list[BaselinePrediction]:
    """跑一遍三个 baseline。"""
    return run_baselines(
        demo_samples, demo_predictions, THRESHOLDS, SINGLE_EVALUATOR
    )


def _write_config(
    tmp_path: Path,
    *,
    overwrite: bool = False,
    single_evaluator: str = SINGLE_EVALUATOR,
) -> Path:
    """在 tmp_path 下写一份 baseline 配置。"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text(
        "paths:\n"
        f"  samples: {DEMO_PATH.as_posix()}\n"
        f"  relation_predictions: {MOCK_PATH.as_posix()}\n"
        "  output_jsonl: ../out/baselines.jsonl\n"
        "  output_csv: ../out/baselines.csv\n"
        "baseline:\n"
        "  decision_threshold: 0.5\n"
        "  tie_tolerance: 0.000001\n"
        f"  single_evaluator: {single_evaluator}\n"
        "output:\n"
        f"  overwrite: {str(overwrite).lower()}\n",
        encoding="utf-8",
    )
    return config_path


# --------------------------------------------------------------------------
# 21-22. 记录数量与顺序
# --------------------------------------------------------------------------


def test_every_claim_gets_three_records(
    demo_samples: list[RAGSample], demo_results: list[BaselinePrediction]
) -> None:
    """claim 数 × 3 条结果。"""
    claim_count = sum(len(sample.claims) for sample in demo_samples)

    assert claim_count == 5
    assert len(demo_results) == claim_count * 3 == 15


def test_result_order_is_stable(demo_results: list[BaselinePrediction]) -> None:
    """顺序固定为 样本 → claim → 三个方法。"""
    order = [(r.claim_id, r.method.value) for r in demo_results]

    assert order[:3] == [
        ("demo-001-c1", "weighted_average"),
        ("demo-001-c1", "majority_vote"),
        ("demo-001-c1", "single_evaluator"),
    ]
    assert [claim for claim, _ in order[::3]] == [
        "demo-001-c1",
        "demo-001-c2",
        "demo-002-c1",
        "demo-003-c1",
        "demo-004-c1",
    ]


def test_repeated_runs_are_identical(
    demo_samples: list[RAGSample], demo_results: list[BaselinePrediction]
) -> None:
    """多次运行结果完全一致。"""
    again = run_baselines(
        demo_samples, load_relation_predictions(MOCK_PATH), THRESHOLDS, SINGLE_EVALUATOR
    )

    assert again == demo_results


def test_single_evaluator_field_is_set_only_for_that_method(
    demo_results: list[BaselinePrediction],
) -> None:
    """只有 single_evaluator 方法带 evaluator 字段。"""
    for result in demo_results:
        if result.method is BaselineMethod.SINGLE_EVALUATOR:
            assert result.evaluator == SINGLE_EVALUATOR
        else:
            assert result.evaluator is None


# --------------------------------------------------------------------------
# 6 / 20. 不输出 conflicting，冲突被压缩
# --------------------------------------------------------------------------


def test_no_baseline_ever_outputs_conflicting(
    demo_results: list[BaselinePrediction],
) -> None:
    """三个 baseline 都不会输出 conflicting。"""
    assert all(
        result.predicted_state is not EvidenceState.CONFLICTING
        for result in demo_results
    )
    assert {r.predicted_state for r in demo_results} <= {
        EvidenceState.SUPPORTED,
        EvidenceState.REFUTED,
        EvidenceState.INSUFFICIENT,
    }


def test_conflicting_demo_is_compressed_to_insufficient(
    demo_results: list[BaselinePrediction],
) -> None:
    """标注为 conflicting 的 demo-004 被三个 baseline 都判成 insufficient。"""
    conflicting = [r for r in demo_results if r.claim_id == "demo-004-c1"]

    assert len(conflicting) == 3
    for result in conflicting:
        assert result.gold_state is EvidenceState.CONFLICTING
        assert result.predicted_state is EvidenceState.INSUFFICIENT


def test_ds_pipeline_still_identifies_the_conflict(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> None:
    """同一条数据，D-S 链路能识别为 conflicting —— 这就是对比的核心。"""
    ds_results = {
        r.claim_id: r
        for r in run_pipeline(demo_samples, demo_predictions, DiagnosticThresholds())
    }

    assert (
        ds_results["demo-004-c1"].diagnostic.primary_state
        is EvidenceState.CONFLICTING
    )
    assert ds_results["demo-004-c1"].diagnostic.k_doc > 0.4


# --------------------------------------------------------------------------
# 18-19. gold_state 不参与计算
# --------------------------------------------------------------------------


def test_gold_state_does_not_change_predictions(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> None:
    """抹掉 gold_state 后，除该字段外结果逐条相同。"""
    stripped = [
        RAGSample(
            sample_id=s.sample_id,
            question=s.question,
            answer=s.answer,
            reference_answer=s.reference_answer,
            claims=list(s.claims),
            contexts=list(s.contexts),
            gold_state=None,
        )
        for s in demo_samples
    ]

    with_label = run_baselines(
        demo_samples, demo_predictions, THRESHOLDS, SINGLE_EVALUATOR
    )
    without_label = run_baselines(
        stripped, demo_predictions, THRESHOLDS, SINGLE_EVALUATOR
    )

    assert [
        r.model_dump(exclude={"gold_state"}) for r in without_label
    ] == [r.model_dump(exclude={"gold_state"}) for r in with_label]
    assert all(r.gold_state is None for r in without_label)


def test_inputs_are_not_modified(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> None:
    """批量运行不修改输入对象。"""
    before = (
        [s.model_dump() for s in demo_samples],
        [p.model_dump() for p in demo_predictions],
    )

    run_baselines(demo_samples, demo_predictions, THRESHOLDS, SINGLE_EVALUATOR)

    assert (
        [s.model_dump() for s in demo_samples],
        [p.model_dump() for p in demo_predictions],
    ) == before


# --------------------------------------------------------------------------
# 完整性检查复用
# --------------------------------------------------------------------------


def test_missing_prediction_is_rejected_by_the_shared_check(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> None:
    """缺失预测时报出与 D-S 链路相同的错误类型。"""
    incomplete = [p for p in demo_predictions if p.doc_id != "demo-004-d2"]

    with pytest.raises(MissingRelationPredictionError, match="demo-004-d2"):
        run_baselines(demo_samples, incomplete, THRESHOLDS, SINGLE_EVALUATOR)


# --------------------------------------------------------------------------
# 23-26. 输出
# --------------------------------------------------------------------------


def test_jsonl_round_trips(
    tmp_path: Path, demo_results: list[BaselinePrediction]
) -> None:
    """JSONL 每条一行，能重新解析回模型，中文不转义。"""
    target = tmp_path / "out.jsonl"

    assert write_baseline_jsonl(target, demo_results) == 15

    raw = target.read_text(encoding="utf-8")
    assert "\\u" not in raw
    lines = [line for line in raw.splitlines() if line.strip()]
    assert len(lines) == 15
    assert [
        BaselinePrediction.model_validate(json.loads(line)) for line in lines
    ] == demo_results


def test_csv_columns_and_content(
    tmp_path: Path, demo_results: list[BaselinePrediction]
) -> None:
    """CSV 字段完整，内容正确。"""
    target = tmp_path / "out.csv"

    assert write_baseline_csv(target, demo_results) == 15

    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 15
    assert list(rows[0].keys()) == list(BASELINE_CSV_COLUMNS)
    assert rows[0]["method"] == "weighted_average"
    assert rows[0]["evaluator"] == ""
    assert rows[2]["method"] == "single_evaluator"
    assert rows[2]["evaluator"] == SINGLE_EVALUATOR
    assert all(row["predicted_state"] != "conflicting" for row in rows)


@pytest.mark.parametrize("writer", [write_baseline_jsonl, write_baseline_csv])
def test_writers_refuse_to_overwrite_by_default(
    tmp_path: Path, demo_results: list[BaselinePrediction], writer
) -> None:
    """默认拒绝覆盖，overwrite=True 时可以覆盖。"""
    target = tmp_path / "out.dat"
    writer(target, demo_results)

    with pytest.raises(FileExistsError):
        writer(target, demo_results)

    assert writer(target, demo_results, overwrite=True) == 15


def test_failed_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    """写入中途失败时清理临时文件。"""

    def exploding():
        yield from ()
        raise RuntimeError("模拟写入中途失败")

    for writer in (write_baseline_jsonl, write_baseline_csv):
        target = tmp_path / "never.dat"
        with pytest.raises(RuntimeError, match="模拟写入中途失败"):
            writer(target, exploding())

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# 配置驱动运行
# --------------------------------------------------------------------------


def test_config_paths_resolve_against_the_config_directory(tmp_path: Path) -> None:
    """相对路径以配置文件所在目录为基准解析。"""
    config = load_baseline_config(_write_config(tmp_path))

    assert config.paths.output_jsonl == (tmp_path / "out" / "baselines.jsonl")
    assert config.baseline.single_evaluator == SINGLE_EVALUATOR
    assert config.baseline.thresholds == THRESHOLDS


def test_config_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """切换当前工作目录不改变配置解析结果。"""
    config_path = _write_config(tmp_path)
    from_root = load_baseline_config(config_path)

    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    assert load_baseline_config(config_path) == from_root


def test_run_from_config_produces_outputs_and_summary(tmp_path: Path) -> None:
    """按配置运行后生成两份输出，摘要统计正确。"""
    summary = run_baselines_from_config(_write_config(tmp_path))

    assert summary.sample_count == 4
    assert summary.claim_count == 5
    assert summary.method_count == 3
    assert summary.record_count == 15
    assert summary.single_evaluator == SINGLE_EVALUATOR
    assert set(summary.state_counts_by_method) == {
        "weighted_average",
        "majority_vote",
        "single_evaluator",
    }
    for counts in summary.state_counts_by_method.values():
        assert "conflicting" not in counts
        assert sum(counts.values()) == 5
    assert Path(summary.output_jsonl).is_file()
    assert Path(summary.output_csv).is_file()


def test_run_from_config_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    """默认不覆盖已有输出；overwrite=True 时可以。"""
    config_path = _write_config(tmp_path)
    run_baselines_from_config(config_path)

    with pytest.raises(FileExistsError):
        run_baselines_from_config(config_path)

    assert run_baselines_from_config(config_path, overwrite=True).record_count == 15


def test_run_from_config_does_not_half_write_on_conflict(tmp_path: Path) -> None:
    """只有 CSV 已存在时，JSONL 也不会被抢先写出。"""
    config_path = _write_config(tmp_path)
    csv_path = tmp_path / "out" / "baselines.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("占位\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_baselines_from_config(config_path)

    assert not (tmp_path / "out" / "baselines.jsonl").exists()


def test_shipped_baseline_config_is_loadable() -> None:
    """configs/baselines_demo.yaml 可直接加载。"""
    config = load_baseline_config(_PROJECT_ROOT / "configs" / "baselines_demo.yaml")

    assert config.paths.samples == DEMO_PATH
    assert config.baseline.single_evaluator == SINGLE_EVALUATOR
    assert config.output.overwrite is False


# --------------------------------------------------------------------------
# 28. D-S pipeline 不受影响
# --------------------------------------------------------------------------


def test_ds_pipeline_output_is_unchanged(
    demo_samples: list[RAGSample], demo_predictions: list[RelationPrediction]
) -> None:
    """跑过 baseline 之后，D-S pipeline 结果不变。"""
    before = run_pipeline(demo_samples, demo_predictions, DiagnosticThresholds())
    run_baselines(demo_samples, demo_predictions, THRESHOLDS, SINGLE_EVALUATOR)
    after = run_pipeline(demo_samples, demo_predictions, DiagnosticThresholds())

    assert after == before
    assert len(after) == 5
