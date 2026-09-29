"""模型关系文件谱系测试：身份、摘要与 oracle/模型两类输入的区分。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_ds.data_io import write_relation_predictions
from rag_ds.dataset_manifest import (
    ArtifactDigest,
    DatasetManifestError,
    file_sha256,
    load_dataset_manifest,
    verify_split_artifacts,
)
from rag_ds.model_runs import (
    ModelRunError,
    ModelRunManifest,
    build_model_run_manifest,
    load_model_run_manifest,
    verify_model_run_artifacts,
    write_model_run_manifest,
)
from rag_ds.relation_evaluation.ragchecker_adapter import LabelProbabilityMapping
from rag_ds.schemas import RelationPrediction

LIVE_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "climate_fever_v1"
)
MANIFEST = LIVE_DATASET / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="需要已构建的 climate_fever_v1 数据集"
)


def _predictions(count: int = 3) -> list[RelationPrediction]:
    return [
        RelationPrediction(
            sample_id=f"s{i}",
            claim_id=f"s{i}-c1",
            doc_id=f"s{i}-d1",
            evaluator="ragchecker",
            p_support=0.8,
            p_refute=0.1,
            p_unknown=0.1,
        )
        for i in range(count)
    ]


@pytest.fixture
def model_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """造一份 test split 的模型关系文件与谱系清单。"""
    manifest = load_dataset_manifest(MANIFEST)
    relations = tmp_path / "test_relations_ragchecker.jsonl"
    write_relation_predictions(relations, _predictions())
    source = tmp_path / "test_checking_outputs.json"
    source.write_text(json.dumps({"results": []}), encoding="utf-8")

    run = build_model_run_manifest(
        dataset_name=manifest.dataset_name,
        split="test",
        evaluator="ragchecker",
        producer="ragchecker",
        checker_name="some-checker-model",
        extractor_name="some-extractor-model",
        samples=manifest.splits["test"].samples,
        predictions_path=relations,
        predictions_relative_path=relations.name,
        source_output_path=source,
        source_output_relative_path=source.name,
        label_mapping=LabelProbabilityMapping(),
        label_mapping_source="placeholder_default",
    )
    run_path = tmp_path / "test_model_run.json"
    write_model_run_manifest(run_path, run)
    return run_path, relations, LIVE_DATASET / "test.jsonl"


class TestRoundTrip:
    """写入、读回与字段含义。"""

    def test_manifest_round_trips(self, model_run: tuple[Path, Path, Path]) -> None:
        run_path, relations, _ = model_run

        loaded = load_model_run_manifest(run_path)

        assert loaded.relation_predictions_kind == "model_prediction"
        assert loaded.relation_predictions.records == 3
        assert loaded.relation_predictions.sha256 == file_sha256(relations)
        assert loaded.is_calibrated is False

    def test_verify_accepts_matching_files(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, relations, samples = model_run

        run = verify_model_run_artifacts(run_path, MANIFEST, samples, relations, "test")

        assert run.evaluator == "ragchecker"
        assert run.checker_name == "some-checker-model"

    def test_refuses_overwrite_without_flag(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, _, _ = model_run

        with pytest.raises(FileExistsError):
            write_model_run_manifest(run_path, load_model_run_manifest(run_path))


class TestIdentityChecks:
    """任何一处对不上都必须报错。"""

    def test_wrong_split_is_rejected(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, relations, _ = model_run

        with pytest.raises(ModelRunError, match="记录的 split"):
            verify_model_run_artifacts(
                run_path,
                MANIFEST,
                LIVE_DATASET / "validation.jsonl",
                relations,
                "validation",
            )

    def test_tampered_predictions_file_is_detected(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, relations, samples = model_run
        write_relation_predictions(relations, _predictions(2), overwrite=True)

        with pytest.raises(ModelRunError, match="模型关系文件校验失败"):
            verify_model_run_artifacts(run_path, MANIFEST, samples, relations, "test")

    def test_samples_digest_must_match_dataset_manifest(
        self, tmp_path: Path, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, relations, samples = model_run
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        payload["samples"]["sha256"] = "0" * 64
        run_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ModelRunError, match="样本文件被改过"):
            verify_model_run_artifacts(run_path, MANIFEST, samples, relations, "test")

    def test_predictions_path_must_be_the_recorded_one(
        self, tmp_path: Path, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, relations, samples = model_run
        impostor = tmp_path / "impostor.jsonl"
        impostor.write_bytes(relations.read_bytes())

        with pytest.raises(ModelRunError, match="不是这次模型运行产出"):
            verify_model_run_artifacts(run_path, MANIFEST, samples, impostor, "test")


class TestCalibrationBookkeeping:
    """校准来源与校准文件必须自洽。"""

    def test_calibrated_source_requires_a_path(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, _, _ = model_run
        payload = load_model_run_manifest(run_path).model_dump()
        payload["label_mapping_source"] = "calibrated_on_validation"

        with pytest.raises(ValueError, match="必须记录 calibration_path"):
            ModelRunManifest.model_validate(payload)

    def test_placeholder_source_must_not_carry_a_path(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        run_path, _, _ = model_run
        payload = load_model_run_manifest(run_path).model_dump()
        payload["calibration_path"] = "somewhere.json"

        with pytest.raises(ValueError, match="不应记录 calibration_path"):
            ModelRunManifest.model_validate(payload)


class TestOracleStaysSeparate:
    """模型关系文件不会被 oracle 校验路径接受，反之亦然。"""

    def test_oracle_verifier_rejects_model_predictions(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        _, relations, samples = model_run

        with pytest.raises(DatasetManifestError, match="不是清单登记的 test 关系文件"):
            verify_split_artifacts(MANIFEST, samples, relations, "test")

    def test_dataset_manifest_still_marks_its_own_relations_as_oracle(self) -> None:
        manifest = load_dataset_manifest(MANIFEST)

        assert manifest.relation_predictions_kind == "annotation_oracle"

    def test_digest_helper_is_shared_with_dataset_manifest(
        self, model_run: tuple[Path, Path, Path]
    ) -> None:
        _, relations, _ = model_run
        run = load_model_run_manifest(model_run[0])

        assert run.relation_predictions == ArtifactDigest(
            path=relations.name,
            sha256=file_sha256(relations),
            records=3,
        )
