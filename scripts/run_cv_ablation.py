"""嵌套交叉验证下的门控消融，每个变体都与完整方法做配对检验。

用法::

    python scripts/run_cv_ablation.py \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" \
        --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" \
        --evaluator refchecker_nli \
        --out-dir outputs/metrics/cross_validation_ablation

为什么要跑
----------
原消融只在 v1 的 80 条测试集上做过，而且离散版与连续版对 theta 门控给出
相反结论。这里用与主实验完全相同的分折、种子与网格，让每个变体在每折内层
**各自重新搜阈值**，616 条汇总后与 ``full`` 做配对自助检验 ——「两个维度
各自有用」要靠这一步才能说。

``full`` 与主实验 D-S 逐条一致（有测试钉住），所以这里的 Δ 就是相对论文
主表那个方法的变化。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rag_ds.dataset_manifest import DatasetManifestError
from rag_ds.experiments.cv_ablation import run_nested_ablation
from rag_ds.experiments.data_loading import load_verified_dataset
from rag_ds.integrity import PipelineError
from rag_ds.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    GOLD_LABELS,
    bootstrap_interval,
    classification_report,
    paired_bootstrap,
)
from rag_ds.model_runs import ModelRunError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "full"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="嵌套交叉验证消融 + 配对检验")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relations-template", required=True)
    parser.add_argument("--model-run-template", default=None)
    parser.add_argument("--evaluator", nargs="+", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-steps", type=int, default=9)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_write(path: Path, write) -> None:
    """先写同目录临时文件，成功后再替换。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8", newline="") as handle:
            write(handle)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    out_dir: Path = args.out_dir or (
        PROJECT_ROOT / "outputs" / "metrics" / "cross_validation_ablation"
    )
    summary_path = out_dir / "ablation_summary.json"
    table_path = out_dir / "ablation_results.csv"
    if not args.overwrite:
        existing = [p for p in (summary_path, table_path) if p.exists()]
        if existing:
            print(f"[输出文件已存在] {existing[0]}", file=sys.stderr)
            print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
            return 1

    try:
        manifest, samples, predictions = load_verified_dataset(
            args.manifest,
            args.relations_template,
            args.evaluator,
            args.model_run_template,
        )
        result = run_nested_ablation(
            samples,
            predictions,
            n_folds=args.n_folds,
            seed=args.seed,
            grid_steps=args.grid_steps,
        )
    except (
        PipelineError,
        DatasetManifestError,
        ModelRunError,
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    by_variant: dict[str, list] = defaultdict(list)
    for record in result.predictions:
        by_variant[record.method].append(record)
    ordered = {
        variant: sorted(items, key=lambda r: (r.sample_id, r.claim_id))
        for variant, items in by_variant.items()
    }
    truth = [r.gold_label for r in ordered[REFERENCE]]

    reports = {
        variant: classification_report(
            truth, [r.predicted_label for r in items], method=variant
        )
        for variant, items in ordered.items()
    }
    intervals = {
        variant: bootstrap_interval(
            truth,
            [r.predicted_label for r in items],
            variant,
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.bootstrap_seed,
        )
        for variant, items in ordered.items()
    }
    # 方向：variant - full，与原消融的 Δ 口径一致（负数表示去掉后变差）。
    comparisons = {
        variant: paired_bootstrap(
            truth,
            [r.predicted_label for r in items],
            [r.predicted_label for r in ordered[REFERENCE]],
            variant,
            REFERENCE,
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.bootstrap_seed,
        )
        for variant, items in ordered.items()
        if variant != REFERENCE
    }

    def _write_summary(handle) -> None:
        json.dump(
            {
                "dataset_name": manifest.dataset_name,
                "evaluators": list(args.evaluator),
                "n_folds": result.n_folds,
                "fold_seed": result.seed,
                "claim_count": len(truth),
                "macro_labels": list(GOLD_LABELS),
                "variants": list(result.variants),
                "folds": [json.loads(f.model_dump_json()) for f in result.folds],
                "reports": {v: json.loads(r.model_dump_json()) for v, r in reports.items()},
                "intervals": {v: json.loads(i.model_dump_json()) for v, i in intervals.items()},
                "comparisons_vs_full": [
                    json.loads(c.model_dump_json()) for c in comparisons.values()
                ],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    def _write_table(handle) -> None:
        writer = csv.writer(handle)
        writer.writerow(
            ["variant", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
             "delta_vs_full", "delta_ci_low", "delta_ci_high", "p_value", "significant"]
        )
        for variant in result.variants:
            report, interval = reports[variant], intervals[variant]
            row = [variant, f"{report.accuracy:.6f}", f"{report.macro_f1:.6f}",
                   f"{interval.ci_low:.6f}", f"{interval.ci_high:.6f}"]
            item = comparisons.get(variant)
            if item is None:
                row += ["", "", "", "", ""]
            else:
                row += [f"{item.observed_delta:+.6f}", f"{item.ci_low:+.6f}",
                        f"{item.ci_high:+.6f}", f"{item.p_value:.4f}", str(item.is_significant)]
            writer.writerow(row)

    _atomic_write(summary_path, _write_summary)
    _atomic_write(table_path, _write_table)

    print("嵌套交叉验证消融完成。")
    print(f"  数据集   {manifest.dataset_name}")
    print(f"  评估器   {', '.join(args.evaluator)}")
    print(f"  折数     {result.n_folds}（分折种子 {result.seed}），{len(truth)} 条 claim")
    print()
    print("  各折各变体选出的阈值（theta / K_doc）：")
    header = "".join(f"{v:>26}" for v in result.variants)
    print(f"    {'折':<4}{header}")
    for fold in result.folds:
        cells = "".join(
            f"{fold.thresholds[v].theta_threshold:>13.4f}"
            f"{fold.thresholds[v].document_conflict_threshold:>13.4f}"
            for v in result.variants
        )
        print(f"    {fold.fold:<4}{cells}")
    print()
    print(f"    {'变体':<26}{'Accuracy':>10}{'Macro-F1':>10}{'Δ vs full':>11}{'CI 下界':>10}{'CI 上界':>10}{'p':>8}   结论")
    for variant in result.variants:
        report = reports[variant]
        item = comparisons.get(variant)
        if item is None:
            print(f"    {variant:<26}{report.accuracy:>10.4f}{report.macro_f1:>10.4f}")
            continue
        verdict = "显著" if item.is_significant else "不显著（区间跨 0）"
        print(
            f"    {variant:<26}{report.accuracy:>10.4f}{report.macro_f1:>10.4f}"
            f"{item.observed_delta:>+11.4f}{item.ci_low:>+10.4f}{item.ci_high:>+10.4f}"
            f"{item.p_value:>8.3f}   {verdict}"
        )
    print()
    print(f"  输出 {summary_path}")
    print(f"       {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
