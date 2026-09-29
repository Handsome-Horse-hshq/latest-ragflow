"""谱系登记脚本测试：直接产出的关系文件必须完整、单一评估器、与 run_info 一致。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rag_ds.data_io import load_samples, write_relation_predictions
from rag_ds.model_runs import load_model_run_manifest, verify_model_run_artifacts
from rag_ds.schemas import RelationPrediction

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "climate_fever_v1" / "manifest.json"
SCRIPT = ROOT / "scripts" / "register_model_run.py"

_spec = importlib.util.spec_from_file_location("register_model_run", SCRIPT)
register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(register)  # type: ignore[union-attr]

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="需要已构建的 climate_fever_v1 数据集"
)


def _full_grid(evaluator: str = "judge_x") -> list[RelationPrediction]:
    samples = load_samples(MANIFEST.parent / "test.jsonl")
    return [
        RelationPrediction(
            sample_id=s.sample_id,
            claim_id=c.claim_id,
            doc_id=d.doc_id,
            evaluator=evaluator,
            p_support=0.2,
            p_refute=0.3,
            p_unknown=0.5,
        )
        for s in samples
        for c in s.claims
        for d in s.contexts
    ]


def _run_info(path: Path, evaluator: str = "judge_x") -> Path:
    info = path / "run_info.json"
    info.write_text(
        json.dumps(
            {
                "evaluator": evaluator,
                "producer": "some-judge",
                "checker_model": "some/model",
                "scoring_version": "v1",
            }
        ),
        encoding="utf-8",
    )
    return info


def _args(tmp_path: Path, relations: Path, info: Path) -> list[str]:
    return [
        "--manifest", str(MANIFEST),
        "--split", "test",
        "--relations", str(relations),
        "--run-info", str(info),
        "--out", str(tmp_path / "test_model_run.json"),
    ]


class TestRegister:
    def test_complete_grid_is_registered_and_verifiable(self, tmp_path: Path) -> None:
        relations = tmp_path / "test_relations.jsonl"
        write_relation_predictions(relations, _full_grid())
        info = _run_info(tmp_path)

        assert register.main(_args(tmp_path, relations, info)) == 0

        run = load_model_run_manifest(tmp_path / "test_model_run.json")
        assert run.evaluator == "judge_x"
        assert run.label_mapping_source == "continuous_probabilities"
        assert run.producer_version == "v1"
        verify_model_run_artifacts(
            tmp_path / "test_model_run.json",
            MANIFEST,
            MANIFEST.parent / "test.jsonl",
            relations,
            "test",
        )

    def test_incomplete_grid_is_refused(self, tmp_path: Path) -> None:
        relations = tmp_path / "test_relations.jsonl"
        write_relation_predictions(relations, _full_grid()[:-1])
        info = _run_info(tmp_path)

        assert register.main(_args(tmp_path, relations, info)) == 1
        assert not (tmp_path / "test_model_run.json").exists()

    def test_evaluator_mismatch_with_run_info_is_refused(self, tmp_path: Path) -> None:
        relations = tmp_path / "test_relations.jsonl"
        write_relation_predictions(relations, _full_grid("judge_x"))
        info = _run_info(tmp_path, evaluator="judge_y")

        assert register.main(_args(tmp_path, relations, info)) == 1

    def test_mixed_evaluators_are_refused(self, tmp_path: Path) -> None:
        grid = _full_grid("judge_x")
        grid[0] = grid[0].model_copy(update={"evaluator": "judge_y"})
        relations = tmp_path / "test_relations.jsonl"
        write_relation_predictions(relations, grid)
        info = _run_info(tmp_path)

        assert register.main(_args(tmp_path, relations, info)) == 1

    def test_existing_output_is_not_overwritten_by_default(self, tmp_path: Path) -> None:
        relations = tmp_path / "test_relations.jsonl"
        write_relation_predictions(relations, _full_grid())
        info = _run_info(tmp_path)
        assert register.main(_args(tmp_path, relations, info)) == 0

        assert register.main(_args(tmp_path, relations, info)) == 1
