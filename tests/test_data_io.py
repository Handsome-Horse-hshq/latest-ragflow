"""第三阶段 JSONL 读写模块的测试。

所有临时文件都写在 pytest 的 ``tmp_path`` 下，不会污染项目的
``data/`` 或 ``outputs/`` 目录。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_ds.data_io import (
    JsonlDataError,
    iter_relation_predictions,
    iter_samples,
    load_relation_predictions,
    load_samples,
    write_relation_predictions,
    write_samples,
)
from rag_ds.schemas import (
    Claim,
    ContextChunk,
    EvidenceState,
    RAGSample,
    RelationPrediction,
)

DEMO_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "demo.jsonl"


def _chinese_sample(sample_id: str = "zh-001") -> RAGSample:
    """构造一条含中文的样本，用于验证序列化不做 Unicode 转义。"""
    return RAGSample(
        sample_id=sample_id,
        question="水的沸点是多少？",
        answer="在标准大气压下，水的沸点是 100 摄氏度。",
        reference_answer="100 摄氏度。",
        claims=[
            Claim(claim_id=f"{sample_id}-c1", text="水的沸点在标准大气压下为 100 摄氏度。")
        ],
        contexts=[
            ContextChunk(
                doc_id=f"{sample_id}-d1",
                text="地球的天然卫星是月球；在标准大气压下水的沸点为 100 摄氏度。",
                retrieval_score=0.9,
                reliability=0.95,
            )
        ],
        gold_state=EvidenceState.SUPPORTED,
    )


def _minimal_sample(sample_id: str) -> RAGSample:
    """构造一条最小合法样本。"""
    return RAGSample(sample_id=sample_id, question="问题？", answer="答案。")


# --------------------------------------------------------------------------
# 1-3. 读取 demo.jsonl
# --------------------------------------------------------------------------


def test_iter_samples_streams_demo_file() -> None:
    """iter_samples 返回迭代器，可以逐条取出 demo.jsonl 的样本。"""
    iterator = iter_samples(DEMO_PATH)

    assert isinstance(iterator, Iterator)

    first = next(iterator)
    assert isinstance(first, RAGSample)
    assert first.sample_id == "demo-001"

    remaining = list(iterator)
    assert len(remaining) == 3


def test_load_samples_reads_all_four_demo_records() -> None:
    """load_samples 读取 demo.jsonl 得到四条样本，覆盖四种证据状态。"""
    samples = load_samples(DEMO_PATH)

    assert len(samples) == 4
    assert [s.sample_id for s in samples] == [
        "demo-001",
        "demo-002",
        "demo-003",
        "demo-004",
    ]
    assert {s.gold_state for s in samples} == set(EvidenceState)


def test_loaded_objects_are_rag_samples() -> None:
    """读取结果全部是 RAGSample 实例，嵌套字段也已解析为模型。"""
    samples = load_samples(DEMO_PATH)

    assert all(isinstance(s, RAGSample) for s in samples)
    assert all(isinstance(c, Claim) for s in samples for c in s.claims)
    assert all(isinstance(c, ContextChunk) for s in samples for c in s.contexts)


def test_load_samples_accepts_str_path() -> None:
    """路径参数同时接受 str 与 Path。"""
    assert len(load_samples(str(DEMO_PATH))) == 4


# --------------------------------------------------------------------------
# 4-5. 空白行与空文件
# --------------------------------------------------------------------------


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    """空行、纯空白行都会被跳过。"""
    source = load_samples(DEMO_PATH)
    target = tmp_path / "with_blanks.jsonl"
    lines = [json.dumps(s.model_dump(mode="json"), ensure_ascii=False) for s in source]
    target.write_text(
        "\n".join(["", lines[0], "   ", "\t", lines[1], ""]) + "\n", encoding="utf-8"
    )

    samples = load_samples(target)

    assert [s.sample_id for s in samples] == ["demo-001", "demo-002"]


def test_empty_file_returns_empty_list(tmp_path: Path) -> None:
    """空文件返回空列表。"""
    target = tmp_path / "empty.jsonl"
    target.write_text("", encoding="utf-8")

    assert load_samples(target) == []


def test_whitespace_only_file_returns_empty_list(tmp_path: Path) -> None:
    """只有空白字符的文件同样返回空列表。"""
    target = tmp_path / "blank.jsonl"
    target.write_text("\n  \n\t\n", encoding="utf-8")

    assert load_samples(target) == []


def test_utf8_bom_file_is_readable(tmp_path: Path) -> None:
    """带 BOM 的 UTF-8 文件可以正常读取。"""
    target = tmp_path / "bom.jsonl"
    payload = json.dumps(
        _chinese_sample().model_dump(mode="json"), ensure_ascii=False
    )
    target.write_text(payload + "\n", encoding="utf-8-sig")

    samples = load_samples(target)

    assert len(samples) == 1
    assert samples[0].sample_id == "zh-001"


# --------------------------------------------------------------------------
# 6-9. 错误处理
# --------------------------------------------------------------------------


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """文件不存在时立即抛出 FileNotFoundError，而不是等到迭代。"""
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        iter_samples(missing)

    with pytest.raises(FileNotFoundError):
        load_samples(missing)


def test_invalid_json_raises_jsonl_data_error(tmp_path: Path) -> None:
    """非法 JSON 行抛出 JsonlDataError。"""
    target = tmp_path / "broken.jsonl"
    target.write_text("{ 这不是 JSON\n", encoding="utf-8")

    with pytest.raises(JsonlDataError) as excinfo:
        load_samples(target)

    assert "不是合法 JSON" in str(excinfo.value)


def test_error_reports_correct_path_and_line_number(tmp_path: Path) -> None:
    """错误信息与异常属性都包含准确的文件路径和物理行号。"""
    good = json.dumps(
        _minimal_sample("ok-1").model_dump(mode="json"), ensure_ascii=False
    )
    target = tmp_path / "line_three_is_bad.jsonl"
    # 第 1 行合法、第 2 行空白、第 3 行非法 —— 空行不应打乱行号。
    target.write_text(f"{good}\n\n{{ broken\n", encoding="utf-8")

    with pytest.raises(JsonlDataError) as excinfo:
        load_samples(target)

    error = excinfo.value
    assert error.line_number == 3
    assert error.path == target
    message = str(error)
    assert str(target) in message
    assert "第 3 行" in message


def test_schema_violation_raises_jsonl_data_error(tmp_path: Path) -> None:
    """JSON 合法但不符合 RAGSample 时抛出 JsonlDataError，并指出字段。"""
    target = tmp_path / "bad_schema.jsonl"
    target.write_text(
        json.dumps({"sample_id": "x", "question": "问题？"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlDataError) as excinfo:
        load_samples(target)

    error = excinfo.value
    assert error.line_number == 1
    assert "不符合 RAGSample 模型" in str(error)
    assert "answer" in str(error)


def test_non_object_line_is_rejected(tmp_path: Path) -> None:
    """每行必须是 JSON 对象，数组或标量会被拒绝。"""
    target = tmp_path / "not_object.jsonl"
    target.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(JsonlDataError, match="必须是一个 JSON 对象"):
        load_samples(target)


def test_iteration_is_lazy_until_the_bad_line(tmp_path: Path) -> None:
    """坏行之前的样本仍可正常产出，说明读取确实是流式的。"""
    good = json.dumps(
        _minimal_sample("ok-1").model_dump(mode="json"), ensure_ascii=False
    )
    target = tmp_path / "half_broken.jsonl"
    target.write_text(f"{good}\n{{ broken\n", encoding="utf-8")

    iterator = iter_samples(target)
    assert next(iterator).sample_id == "ok-1"

    with pytest.raises(JsonlDataError) as excinfo:
        next(iterator)
    assert excinfo.value.line_number == 2


def test_error_message_does_not_dump_whole_file(tmp_path: Path) -> None:
    """错误信息不应包含整份文件内容。"""
    good = json.dumps(
        _minimal_sample("ok-1").model_dump(mode="json"), ensure_ascii=False
    )
    target = tmp_path / "long.jsonl"
    target.write_text(f"{good}\n" * 50 + "{ broken\n", encoding="utf-8")

    with pytest.raises(JsonlDataError) as excinfo:
        load_samples(target)

    assert "ok-1" not in str(excinfo.value)


# --------------------------------------------------------------------------
# 10-15. 写入
# --------------------------------------------------------------------------


def test_write_samples_creates_missing_parent_directories(tmp_path: Path) -> None:
    """父目录不存在时会被自动创建。"""
    target = tmp_path / "deep" / "nested" / "out.jsonl"

    count = write_samples(target, [_chinese_sample()])

    assert count == 1
    assert target.is_file()


def test_write_samples_returns_written_count(tmp_path: Path) -> None:
    """返回值等于实际写入的样本数量。"""
    samples = [_minimal_sample(f"s{i}") for i in range(5)]

    assert write_samples(tmp_path / "five.jsonl", samples) == 5
    assert write_samples(tmp_path / "none.jsonl", []) == 0
    assert (tmp_path / "none.jsonl").read_text(encoding="utf-8") == ""


def test_round_trip_preserves_content(tmp_path: Path) -> None:
    """写入再读取后，数据内容与原始样本完全一致。"""
    original = load_samples(DEMO_PATH)
    target = tmp_path / "round_trip.jsonl"

    assert write_samples(target, original) == 4
    restored = load_samples(target)

    assert restored == original
    assert [s.model_dump() for s in restored] == [s.model_dump() for s in original]


def test_written_file_is_valid_jsonl(tmp_path: Path) -> None:
    """每个样本占一行，且文件末尾保留换行符。"""
    target = tmp_path / "lines.jsonl"
    write_samples(target, [_minimal_sample(f"s{i}") for i in range(3)])

    raw = target.read_text(encoding="utf-8")

    assert raw.endswith("\n")
    assert "\r" not in raw
    lines = raw.splitlines()
    assert len(lines) == 3
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_chinese_is_not_unicode_escaped(tmp_path: Path) -> None:
    """中文按原样写入，不会被转义成 \\uXXXX。"""
    target = tmp_path / "chinese.jsonl"
    write_samples(target, [_chinese_sample()])

    raw = target.read_text(encoding="utf-8")

    assert "水的沸点" in raw
    assert "地球的天然卫星" in raw
    assert "\\u" not in raw


def test_existing_file_is_not_overwritten_by_default(tmp_path: Path) -> None:
    """目标文件已存在且 overwrite=False 时拒绝写入，且原内容不变。"""
    target = tmp_path / "existing.jsonl"
    target.write_text("原有内容\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_samples(target, [_minimal_sample("s1")])

    assert target.read_text(encoding="utf-8") == "原有内容\n"


def test_overwrite_flag_allows_replacing_the_file(tmp_path: Path) -> None:
    """overwrite=True 时可以覆盖已有文件。"""
    target = tmp_path / "existing.jsonl"
    write_samples(target, [_minimal_sample("s1")])

    count = write_samples(target, [_minimal_sample("s2"), _minimal_sample("s3")], overwrite=True)

    assert count == 2
    assert [s.sample_id for s in load_samples(target)] == ["s2", "s3"]


def test_write_samples_does_not_mutate_inputs(tmp_path: Path) -> None:
    """写入过程不修改传入的样本对象。"""
    samples = load_samples(DEMO_PATH)
    before = [s.model_dump() for s in samples]

    write_samples(tmp_path / "out.jsonl", samples)

    assert [s.model_dump() for s in samples] == before


def test_write_samples_accepts_a_generator(tmp_path: Path) -> None:
    """samples 可以是任意可迭代对象，包括生成器。"""
    target = tmp_path / "from_generator.jsonl"

    count = write_samples(target, (s for s in load_samples(DEMO_PATH)))

    assert count == 4


def test_failed_write_leaves_no_partial_output(tmp_path: Path) -> None:
    """写入中途失败时清理临时文件，且不产生目标文件。"""

    def exploding_samples():
        yield _minimal_sample("s1")
        raise RuntimeError("模拟写入中途失败")

    target = tmp_path / "never_created.jsonl"

    with pytest.raises(RuntimeError, match="模拟写入中途失败"):
        write_samples(target, exploding_samples())

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_overwrite_keeps_original_file(tmp_path: Path) -> None:
    """覆盖写入失败时，原文件保持不变。"""
    target = tmp_path / "existing.jsonl"
    write_samples(target, [_minimal_sample("keep-me")])

    def exploding_samples():
        yield _minimal_sample("s1")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_samples(target, exploding_samples(), overwrite=True)

    assert [s.sample_id for s in load_samples(target)] == ["keep-me"]


def test_write_samples_rejects_non_rag_sample(tmp_path: Path) -> None:
    """传入非 RAGSample 元素时报出明确的 TypeError。"""
    target = tmp_path / "bad_type.jsonl"

    with pytest.raises(TypeError, match="只接受 RAGSample"):
        write_samples(target, [{"sample_id": "s1"}])  # type: ignore[list-item]

    assert not target.exists()


# --------------------------------------------------------------------------
# RelationPrediction 读写：与 RAGSample 行为保持一致
# --------------------------------------------------------------------------


def _prediction(claim_id: str = "c1") -> RelationPrediction:
    """构造一条合法的关系预测。"""
    return RelationPrediction(
        sample_id="s1",
        claim_id=claim_id,
        doc_id="d1",
        evaluator="mock_evaluator",
        p_support=0.7,
        p_refute=0.2,
        p_unknown=0.1,
    )


def test_relation_prediction_round_trip(tmp_path: Path) -> None:
    """写入再读取后内容一致，且返回正确条数。"""
    target = tmp_path / "preds.jsonl"
    predictions = [_prediction(f"c{i}") for i in range(3)]

    assert write_relation_predictions(target, predictions) == 3
    assert load_relation_predictions(target) == predictions


def test_iter_relation_predictions_is_an_iterator(tmp_path: Path) -> None:
    """iter_relation_predictions 返回迭代器并逐条产出。"""
    target = tmp_path / "preds.jsonl"
    write_relation_predictions(target, [_prediction("c1"), _prediction("c2")])

    iterator = iter_relation_predictions(target)

    assert isinstance(iterator, Iterator)
    assert next(iterator).claim_id == "c1"
    assert [p.claim_id for p in iterator] == ["c2"]


def test_relation_prediction_empty_file(tmp_path: Path) -> None:
    """空文件返回空列表。"""
    target = tmp_path / "empty.jsonl"
    target.write_text("", encoding="utf-8")

    assert load_relation_predictions(target) == []


def test_relation_prediction_missing_file(tmp_path: Path) -> None:
    """文件不存在时同样立即抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        iter_relation_predictions(tmp_path / "nope.jsonl")


def test_relation_prediction_schema_error_names_the_model(tmp_path: Path) -> None:
    """校验失败时错误信息指出的是 RelationPrediction 而不是 RAGSample。"""
    target = tmp_path / "bad.jsonl"
    target.write_text(
        json.dumps({"sample_id": "s1", "claim_id": "c1"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(JsonlDataError) as excinfo:
        load_relation_predictions(target)

    error = excinfo.value
    assert error.line_number == 1
    assert "不符合 RelationPrediction 模型" in str(error)


def test_relation_prediction_respects_overwrite_flag(tmp_path: Path) -> None:
    """默认拒绝覆盖，overwrite=True 时才允许。"""
    target = tmp_path / "preds.jsonl"
    write_relation_predictions(target, [_prediction("c1")])

    with pytest.raises(FileExistsError):
        write_relation_predictions(target, [_prediction("c2")])

    assert write_relation_predictions(target, [_prediction("c2")], overwrite=True) == 1
    assert [p.claim_id for p in load_relation_predictions(target)] == ["c2"]


def test_relation_prediction_rejects_wrong_model_type(tmp_path: Path) -> None:
    """写入端拒绝非 RelationPrediction 元素。"""
    target = tmp_path / "mixed.jsonl"

    with pytest.raises(TypeError, match="只接受 RelationPrediction"):
        write_relation_predictions(target, [_minimal_sample("s1")])  # type: ignore[list-item]

    assert not target.exists()


def test_sample_and_prediction_writers_do_not_interfere(tmp_path: Path) -> None:
    """两套读写函数互不影响，模型不会被张冠李戴。"""
    samples_path = tmp_path / "samples.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    write_samples(samples_path, [_minimal_sample("s1")])
    write_relation_predictions(preds_path, [_prediction("c1")])

    with pytest.raises(JsonlDataError, match="不符合 RelationPrediction 模型"):
        load_relation_predictions(samples_path)
    with pytest.raises(JsonlDataError, match="不符合 RAGSample 模型"):
        load_samples(preds_path)
