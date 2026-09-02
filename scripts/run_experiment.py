"""对比实验入口：D-S 诊断 vs 三个 baseline。

用法::

    python scripts/run_experiment.py --config configs/experiment.yaml --overwrite

产出（写入 outputs/）::

    metrics/main_results.csv        实验 1-3 的主结果
    metrics/ablation_results.csv    实验 4 消融
    predictions/experiment_predictions.csv  逐条预测明细

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from rag_ds.baselines.models import BaselineThresholds
from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import DatasetManifestError, verify_split_artifacts
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments import (
    run_ablation,
    run_comparison,
    write_ablation_csv,
    write_main_results_csv,
    write_predictions_csv,
)
from rag_ds.integrity import PipelineError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 D-S 与 baseline 的对比实验")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics",
        help="指标 CSV 的输出目录",
    )
    return parser.parse_args(argv)


def _load(config_path: Path) -> dict:
    """读取实验配置并把相对路径解析到配置文件目录。"""
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
    baseline_thresholds = BaselineThresholds(**baseline_config)

    metrics_dir: Path = args.metrics_dir
    main_csv = metrics_dir / "main_results.csv"
    ablation_csv = metrics_dir / "ablation_results.csv"
    detail_csv = paths["output_csv"]
    allow_overwrite = args.overwrite or bool(config.get("output", {}).get("overwrite"))
    if not allow_overwrite:
        existing = [p for p in (main_csv, ablation_csv, detail_csv) if p.exists()]
        if existing:
            print(f"[输出文件已存在] {existing[0]}", file=sys.stderr)
            print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
            return 1

    try:
        manifest = None
        if "manifest" in paths:
            split_name = config.get("data", {}).get("split")
            if split_name not in {"train", "validation", "test"}:
                raise DatasetManifestError(
                    "配置包含 manifest 时，data.split 必须是 "
                    "train/validation/test 之一"
                )
            manifest = verify_split_artifacts(
                paths["manifest"],
                paths["samples"],
                paths["relation_predictions"],
                split_name,
            )
        samples = load_samples(paths["samples"])
        predictions = load_relation_predictions(paths["relation_predictions"])
        report, records = run_comparison(
            samples,
            predictions,
            ds_thresholds,
            baseline_thresholds,
            single_evaluator,
        )
        ablation = run_ablation(samples, predictions, ds_thresholds)
    except (PipelineError, DatasetManifestError, ValueError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    write_main_results_csv(main_csv, report)
    write_ablation_csv(ablation_csv, ablation)
    write_predictions_csv(detail_csv, records)

    print("对比实验完成。")
    print(f"  样本 {len(samples)} 条 / claim {report.claim_count} 条")
    print()
    print("  实验 1 —— 四分类能力：")
    print(f"    {'方法':<20}{'Accuracy':>10}{'Macro-F1':>10}")
    for item in report.classification:
        print(f"    {item.method:<20}{item.accuracy:>10.4f}{item.macro_f1:>10.4f}")
    print()
    print("  实验 2 —— 证据不足识别（正类 insufficient）：")
    for item in report.insufficiency_detection:
        auroc = "n/a" if item.auroc is None else f"{item.auroc:.4f}"
        print(f"    {item.score_name:<38}AUROC={auroc}  bestF1={item.best_f1:.4f}")
    print()
    print("  实验 3 —— 文档冲突识别（正类 conflicting）：")
    for item in report.conflict_detection:
        auroc = "n/a" if item.auroc is None else f"{item.auroc:.4f}"
        print(f"    {item.score_name:<38}AUROC={auroc}  bestF1={item.best_f1:.4f}")
    print()
    print("  实验 4 —— 消融：")
    for item in ablation:
        print(
            f"    {item.variant.value:<28}macroF1={item.report.macro_f1:.4f}"
            f"  Δ={item.macro_f1_delta:+.4f}"
        )
    print()
    print("  输出文件：")
    print(f"    {main_csv}")
    print(f"    {ablation_csv}")
    print(f"    {detail_csv}")
    print()
    if manifest is None:
        print("  提醒：本次输入没有 manifest 身份与摘要校验。")
    elif manifest.relation_predictions_kind == "annotation_oracle":
        print("  重要：关系输入来自人工标注 oracle；这些结果只用于验证融合链路，")
        print("  不能报告为关系评估模型的真实性能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
