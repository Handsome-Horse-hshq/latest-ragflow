"""Pipeline 的 YAML 配置。

配置里的相对路径**以配置文件所在目录为基准**解析，不依赖终端的当前工作
目录 —— 从项目根目录、从 ``scripts/`` 里、还是从任意别处运行，得到的都是
同一批文件。

配置中不含任何模型供应商设置，也不读取 API Key：本阶段的关系概率全部来自
预设的 mock 数据。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from rag_ds.diagnostics.models import DiagnosticThresholds

__all__ = [
    "OutputOptions",
    "PipelineConfig",
    "PipelinePaths",
    "load_pipeline_config",
    "read_yaml_mapping",
    "resolve_paths",
]


class PipelinePaths(BaseModel):
    """输入与输出文件路径。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: RAGSample 的 JSONL 文件。
    samples: Path
    #: RelationPrediction 的 JSONL 文件。
    relation_predictions: Path
    #: 逐 claim 的完整诊断结果（含中间过程）。
    output_jsonl: Path
    #: 逐 claim 的扁平诊断摘要。
    output_csv: Path


class OutputOptions(BaseModel):
    """输出行为选项。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 目标文件已存在时是否允许覆盖。
    overwrite: bool = False


class PipelineConfig(BaseModel):
    """一次离线运行的完整配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: PipelinePaths
    #: 二维门控阈值；默认值仅供调试，正式实验须在验证集上选择。
    diagnostics: DiagnosticThresholds = DiagnosticThresholds()
    output: OutputOptions = OutputOptions()


def _resolve(base_dir: Path, value: Path) -> Path:
    """把相对路径解析到 ``base_dir`` 之下；绝对路径原样保留。"""
    return value if value.is_absolute() else (base_dir / value).resolve()


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """读取 YAML 配置，解析路径并做基本检查。

    步骤：

    1. 解析 YAML（顶层必须是映射）；
    2. 按模型校验，未知字段一律拒绝；
    3. 把四个相对路径解析到**配置文件所在目录**之下；
    4. 确认两个输入文件存在；
    5. 为两个输出文件创建缺失的父目录。

    Args:
        path: YAML 配置文件路径。

    Returns:
        路径已解析为绝对路径的 :class:`PipelineConfig`。

    Raises:
        FileNotFoundError: 配置文件本身或某个输入文件不存在。
        ValueError: YAML 顶层不是映射。
        pydantic.ValidationError: 配置结构不合法或含未知字段。

    Note:
        本函数会**创建输出目录**（不创建文件）。这是刻意的：等到写盘那一刻
        才发现目录不存在，前面所有计算就白做了。
    """
    config_path, raw = read_yaml_mapping(path)
    config = PipelineConfig.model_validate(raw)
    resolved = resolve_paths(config.paths, config_path.parent)
    return PipelineConfig(
        paths=resolved,
        diagnostics=config.diagnostics,
        output=config.output,
    )


def read_yaml_mapping(path: str | Path) -> tuple[Path, dict]:
    """读取 YAML 并确认顶层是映射，返回 ``(绝对路径, 内容)``。"""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"配置文件顶层必须是映射，实际得到 {type(raw).__name__}：{config_path}"
        )
    return config_path, raw


def resolve_paths(paths: PipelinePaths, base_dir: Path) -> PipelinePaths:
    """把四个相对路径解析到 ``base_dir``，检查输入并创建输出目录。"""
    resolved = PipelinePaths(
        samples=_resolve(base_dir, paths.samples),
        relation_predictions=_resolve(base_dir, paths.relation_predictions),
        output_jsonl=_resolve(base_dir, paths.output_jsonl),
        output_csv=_resolve(base_dir, paths.output_csv),
    )

    for field in ("samples", "relation_predictions"):
        input_path: Path = getattr(resolved, field)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"配置 paths.{field} 指向的输入文件不存在：{input_path}"
                f"（相对路径以配置文件目录 {base_dir} 为基准解析）"
            )

    for field in ("output_jsonl", "output_csv"):
        output_path: Path = getattr(resolved, field)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    return resolved

