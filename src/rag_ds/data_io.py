"""JSONL 数据读写。

本模块只负责把 Pydantic 模型安全地读进来、写出去，不包含 claim 提取、
关系评估、BPA 映射或证据融合等任何算法。

读写逻辑对模型类型是泛型的：内部只有一份实现，
:class:`~rag_ds.schemas.RAGSample` 与
:class:`~rag_ds.schemas.RelationPrediction` 共用同一套解析、校验、
错误报告与原子写入机制。

读取端一律逐行处理并逐行校验：一行坏数据只会报告该行的行号与原因，
不会把整份文件打印到错误信息里。写入端采用「先写临时文件、成功后再替换」
的方式，避免中途失败留下半截文件。
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from rag_ds.schemas import RAGSample, RelationPrediction

__all__ = [
    "JsonlDataError",
    "iter_relation_predictions",
    "iter_samples",
    "load_relation_predictions",
    "load_samples",
    "write_relation_predictions",
    "write_samples",
]

#: 校验失败时最多展示的字段错误条数，避免错误信息无限膨胀。
_MAX_REPORTED_ERRORS = 3

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class JsonlDataError(Exception):
    """JSONL 文件中某一行无法解析或无法通过数据模型校验。

    异常同时以属性形式保留 :attr:`path`、:attr:`line_number` 与
    :attr:`reason`，方便调用方在不解析字符串的情况下定位问题。
    """

    def __init__(self, path: str | Path, line_number: int, reason: str) -> None:
        self.path = Path(path)
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"{self.path}: 第 {line_number} 行: {reason}")


def _describe_validation_error(exc: ValidationError, model: type[BaseModel]) -> str:
    """把 Pydantic 的校验错误压缩成一行简短说明。"""
    details = exc.errors()
    parts: list[str] = []
    for detail in details[:_MAX_REPORTED_ERRORS]:
        location = ".".join(str(item) for item in detail["loc"]) or "<root>"
        parts.append(f"{location}: {detail['msg']}")
    remaining = len(details) - _MAX_REPORTED_ERRORS
    if remaining > 0:
        parts.append(f"（另有 {remaining} 处错误未显示）")
    return f"不符合 {model.__name__} 模型 —— " + "; ".join(parts)


def _iter_validated(file_path: Path, model: type[_ModelT]) -> Iterator[_ModelT]:
    """逐行读取并按 ``model`` 校验，供 :func:`_iter_records` 使用的生成器。"""
    # utf-8-sig 兼容带 BOM 的文件；无 BOM 时行为与 utf-8 一致。
    with file_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue  # 纯空白行直接跳过，但行号继续累加

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlDataError(
                    file_path, line_number, f"不是合法 JSON —— {exc.msg}（列 {exc.colno}）"
                ) from exc

            if not isinstance(payload, dict):
                raise JsonlDataError(
                    file_path,
                    line_number,
                    f"每行必须是一个 JSON 对象，实际得到 {type(payload).__name__}",
                )

            try:
                yield model.model_validate(payload)
            except ValidationError as exc:
                raise JsonlDataError(
                    file_path, line_number, _describe_validation_error(exc, model)
                ) from exc


def _iter_records(path: str | Path, model: type[_ModelT]) -> Iterator[_ModelT]:
    """所有流式读取函数的共同实现。

    存在性检查在调用时立即执行，而不是等到第一次迭代 —— 生成器函数在被
    迭代前不会执行函数体，若把检查放进生成器里，调用方将无法及时拿到
    ``FileNotFoundError``。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), str(file_path)
        )
    return _iter_validated(file_path, model)


def _write_records(
    path: str | Path,
    records: Iterable[_ModelT],
    model: type[_ModelT],
    overwrite: bool,
) -> int:
    """所有写入函数的共同实现。

    先写入同目录下的临时文件，全部写完后再用 :func:`os.replace` 原子替换
    目标文件；任何阶段失败都会清理临时文件，不会留下半截结果。
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            errno.EEXIST,
            f"{os.strerror(errno.EEXIST)}；如需覆盖请传入 overwrite=True",
            str(target),
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    handle_fd, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    written = 0
    try:
        # newline="\n" 阻止 Windows 把换行改写成 CRLF。
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                if not isinstance(record, model):
                    raise TypeError(
                        f"只接受 {model.__name__}，收到 {type(record).__name__}"
                    )
                line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
                handle.write(line)
                handle.write("\n")
                written += 1
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return written


# --------------------------------------------------------------------------
# RAGSample
# --------------------------------------------------------------------------


def iter_samples(path: str | Path) -> Iterator[RAGSample]:
    """流式读取 JSONL 文件，逐条产出校验通过的 :class:`RAGSample`。

    文件按行读取，不会一次性载入内存；纯空白行被忽略。

    Args:
        path: JSONL 文件路径。

    Returns:
        产出 :class:`RAGSample` 的迭代器。

    Raises:
        FileNotFoundError: 文件不存在，调用时立即抛出。
        JsonlDataError: 某一行不是合法 JSON，或无法通过校验；异常信息
            包含文件路径与出错行号。
    """
    return _iter_records(path, RAGSample)


def load_samples(path: str | Path) -> list[RAGSample]:
    """一次性读取整个 JSONL 文件为 :class:`RAGSample` 列表。

    内部完全复用 :func:`iter_samples`，不重复实现解析与校验逻辑。

    Args:
        path: JSONL 文件路径。

    Returns:
        校验通过的样本列表；空文件返回空列表。

    Raises:
        FileNotFoundError: 文件不存在。
        JsonlDataError: 任意一行解析或校验失败。
    """
    return list(iter_samples(path))


def write_samples(
    path: str | Path,
    samples: Iterable[RAGSample],
    overwrite: bool = False,
) -> int:
    """把 :class:`RAGSample` 写成 JSONL 文件。

    中文按原样保留（``ensure_ascii=False``），不会被转义成 ``\\uXXXX``。
    换行统一为 ``\\n``，文件末尾保留换行符。传入的对象不会被修改。

    Args:
        path: 目标文件路径，缺失的父目录会被自动创建。
        samples: 待写入的样本，可以是任意可迭代对象（含生成器）。
        overwrite: 目标文件已存在时是否允许覆盖。

    Returns:
        实际写入的样本数量。

    Raises:
        FileExistsError: 目标文件已存在且 ``overwrite`` 为 ``False``。
        TypeError: ``samples`` 中含有非 :class:`RAGSample` 元素。
    """
    return _write_records(path, samples, RAGSample, overwrite)


# --------------------------------------------------------------------------
# RelationPrediction
# --------------------------------------------------------------------------


def iter_relation_predictions(path: str | Path) -> Iterator[RelationPrediction]:
    """流式读取 JSONL 文件，逐条产出 :class:`RelationPrediction`。

    行为与 :func:`iter_samples` 完全一致，只是校验用的模型不同。

    Args:
        path: JSONL 文件路径。

    Returns:
        产出 :class:`RelationPrediction` 的迭代器。

    Raises:
        FileNotFoundError: 文件不存在，调用时立即抛出。
        JsonlDataError: 某一行不是合法 JSON，或无法通过校验。
    """
    return _iter_records(path, RelationPrediction)


def load_relation_predictions(path: str | Path) -> list[RelationPrediction]:
    """一次性读取整个 JSONL 文件为 :class:`RelationPrediction` 列表。

    Args:
        path: JSONL 文件路径。

    Returns:
        校验通过的预测列表；空文件返回空列表。

    Raises:
        FileNotFoundError: 文件不存在。
        JsonlDataError: 任意一行解析或校验失败。
    """
    return list(iter_relation_predictions(path))


def write_relation_predictions(
    path: str | Path,
    predictions: Iterable[RelationPrediction],
    overwrite: bool = False,
) -> int:
    """把 :class:`RelationPrediction` 写成 JSONL 文件。

    行为与 :func:`write_samples` 完全一致（原子替换、保留中文、默认拒绝
    覆盖），只是接受的模型不同。

    Args:
        path: 目标文件路径，缺失的父目录会被自动创建。
        predictions: 待写入的预测，可以是任意可迭代对象（含生成器）。
        overwrite: 目标文件已存在时是否允许覆盖。

    Returns:
        实际写入的预测数量。

    Raises:
        FileExistsError: 目标文件已存在且 ``overwrite`` 为 ``False``。
        TypeError: ``predictions`` 中含有非 :class:`RelationPrediction` 元素。
    """
    return _write_records(path, predictions, RelationPrediction, overwrite)
