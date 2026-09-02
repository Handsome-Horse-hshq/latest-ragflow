"""从实验结果生成论文用图表。

用法::

    python scripts/export_results.py --config configs/experiment.yaml \
        --threshold-search outputs/metrics/threshold_search.json

产出（写入 outputs/figures/）::

    confusion_matrix.png        D-S 的混淆矩阵
    diagnostic_scatter.png      二维诊断散点图（x=m_theta, y=K_doc）
    threshold_sensitivity.png   阈值敏感性曲线

不联网，不调用任何大模型；matplotlib 使用 Agg 后端，不弹窗。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from rag_ds.baselines.models import BaselineThresholds
from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments import (
    DS_METHOD,
    plot_confusion_matrix,
    plot_diagnostic_scatter,
    plot_threshold_sensitivity,
    run_comparison,
)
from rag_ds.integrity import PipelineError
from rag_ds.tuning import ThresholdSearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成论文用图表")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_ROOT / "outputs" / "figures"
    )
    parser.add_argument(
        "--threshold-search",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "threshold_search.json",
        help="由 tune_thresholds.py 在验证集上生成的搜索结果",
    )
    return parser.parse_args(argv)


def _load(config_path: Path) -> dict:
    """读取实验配置并解析相对路径。"""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    paths = {
        key: (base / value).resolve() if not Path(value).is_absolute() else Path(value)
        for key, value in raw["paths"].items()
    }
    return {"paths": paths, **{k: v for k, v in raw.items() if k != "paths"}}


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    config = _load(args.config.resolve())
    paths = config["paths"]

    ds_thresholds = DiagnosticThresholds(**config["diagnostics"])
    baseline_config = dict(config["baseline"])
    single_evaluator = baseline_config.pop("single_evaluator")

    try:
        samples = load_samples(paths["samples"])
        predictions = load_relation_predictions(paths["relation_predictions"])
        report, records = run_comparison(
            samples,
            predictions,
            ds_thresholds,
            BaselineThresholds(**baseline_config),
            single_evaluator,
        )
        search = ThresholdSearchResult.model_validate_json(
            args.threshold_search.read_text(encoding="utf-8")
        )
    except (PipelineError, OSError, ValueError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    ds_report = report.classification_for(DS_METHOD)
    if ds_report is None:
        print("[错误] 报告中缺少 D-S 结果", file=sys.stderr)
        return 1

    figures_dir: Path = args.figures_dir
    written = [
        plot_confusion_matrix(figures_dir / "confusion_matrix.png", ds_report),
        plot_diagnostic_scatter(figures_dir / "diagnostic_scatter.png", records),
        plot_threshold_sensitivity(
            figures_dir / "threshold_sensitivity.png", search
        ),
    ]

    print("图表生成完成。")
    for path in written:
        print(f"  {path}")
    print()
    print(f"  阈值敏感性图读取已保存的验证集搜索结果：{args.threshold_search}")
    print("  当前分类/散点数据未被用于重新选择阈值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
