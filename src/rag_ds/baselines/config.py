"""Baseline 运行的 YAML 配置。

路径解析、输入检查与输出目录创建都复用 :mod:`rag_ds.config` 里的同一套
辅助函数，与 D-S pipeline 的行为完全一致。

依赖方向是单向的：``baselines`` 依赖 ``config``，反之不成立 —— 否则
``config`` 与 ``baselines`` 会互相导入。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rag_ds.baselines.models import BaselineThresholds
from rag_ds.config import (
    OutputOptions,
    PipelinePaths,
    read_yaml_mapping,
    resolve_paths,
)

__all__ = ["BaselineConfig", "BaselineOptions", "load_baseline_config"]


class BaselineOptions(BaseModel):
    """baseline 运行选项。

    ``decision_threshold`` 与 ``tie_tolerance`` 直接展开在这一层，而不是
    再嵌套一层子映射 —— YAML 里少一级缩进更好读。内部再打包成
    :class:`~rag_ds.baselines.models.BaselineThresholds`。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 最高分低于该值即判为证据不足。调试默认值。
    decision_threshold: float = 0.5
    #: 最高分之间差距不超过该值时视为平局。
    tie_tolerance: float = 1e-6
    #: single-evaluator baseline 使用的评估器名称。
    single_evaluator: str

    @property
    def thresholds(self) -> BaselineThresholds:
        """打包成 baseline 判定函数需要的阈值模型。"""
        return BaselineThresholds(
            decision_threshold=self.decision_threshold,
            tie_tolerance=self.tie_tolerance,
        )


class BaselineConfig(BaseModel):
    """一次 baseline 批量运行的完整配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: PipelinePaths
    baseline: BaselineOptions
    output: OutputOptions = OutputOptions()


def load_baseline_config(path: str | Path) -> BaselineConfig:
    """读取 baseline 的 YAML 配置。

    Args:
        path: YAML 配置文件路径。

    Returns:
        路径已解析为绝对路径的 :class:`BaselineConfig`。

    Raises:
        FileNotFoundError: 配置文件本身或某个输入文件不存在。
        ValueError: YAML 顶层不是映射。
        pydantic.ValidationError: 配置结构不合法或含未知字段。
    """
    config_path, raw = read_yaml_mapping(path)
    config = BaselineConfig.model_validate(raw)
    return BaselineConfig(
        paths=resolve_paths(config.paths, config_path.parent),
        baseline=config.baseline,
        output=config.output,
    )
