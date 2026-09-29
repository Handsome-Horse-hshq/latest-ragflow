"""跨分折种子重跑交叉验证，量化主结论的脆弱性。

为什么要跑
----------
单次 5 折 CV 给出 D-S 相对最强 baseline（``conflict_aware``）的
Δ Macro-F1 = +0.0461，95% 置信区间 ``[+0.0007, +0.0906]`` —— **下界离 0 只有
0.0007**。这种"刚好显著"的结论极不稳健：换一个分折种子就可能翻到不显著。

本脚本用多个分折种子重跑同一套协议，把每个种子的结果**全部**报出来，
并给出跨种子的均值、标准差与"有多少个种子达到显著"。

.. warning::
    这是**稳健性检验**，不是挑种子。论文里必须报告全部种子的结果，
    绝不能只报对自己有利的那个。脚本因此不提供"取最优种子"的选项，
    输出 CSV 里每个种子一行，全部保留。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.experiments import run_nested_cross_validation
from rag_ds.integrity import PipelineError
from rag_ds.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    GOLD_LABELS,
    classification_report,
    paired_bootstrap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="跨分折种子的 CV 稳健性检验")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relations-template", required=True)
    parser.add_argument("--evaluator", nargs="+", required=True)
    parser.add_argument("--single-evaluator", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4],
        help="要跑的分折种子；全部结果都会被报告",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--grid-steps", type=int, default=9)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--reference", default="ds", help="作为基准的方法名"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """先写临时文件再原子替换。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    out_path: Path = args.out or (
        PROJECT_ROOT / "outputs" / "metrics" / "cv_seed_robustness.csv"
    )
    if out_path.exists() and not args.overwrite:
        print(f"[输出文件已存在] {out_path}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    manifest_path = args.manifest.resolve()
    try:
        manifest = load_dataset_manifest(manifest_path)
        samples, predictions = [], []
        for split in SPLITS:
            samples_path = verify_artifact(
                manifest_path.parent, manifest.splits[split].samples
            )
            samples.extend(load_samples(samples_path))
            for evaluator in args.evaluator:
                predictions.extend(
                    load_relation_predictions(
                        Path(
                            args.relations_template.format(
                                split=split, evaluator=evaluator
                            )
                        )
                    )
                )
    except (DatasetManifestError, ValueError, OSError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    per_method_f1: dict[str, list[float]] = defaultdict(list)
    per_rival_delta: dict[str, list[float]] = defaultdict(list)
    per_rival_significant: dict[str, int] = defaultdict(int)

    for seed in args.seeds:
        try:
            result = run_nested_cross_validation(
                samples,
                predictions,
                args.single_evaluator,
                n_folds=args.n_folds,
                seed=seed,
                grid_steps=args.grid_steps,
            )
        except (PipelineError, ValueError) as error:
            print(f"[种子 {seed} 失败] {error}", file=sys.stderr)
            return 1

        ordered = {}
        for record in result.predictions:
            ordered.setdefault(record.method, []).append(record)
        for method in ordered:
            ordered[method] = sorted(
                ordered[method], key=lambda r: (r.sample_id, r.claim_id)
            )
        truth = [r.gold_label for r in ordered[args.reference]]

        reports = {
            method: classification_report(
                [r.gold_label for r in items],
                [r.predicted_label for r in items],
                method=method,
            )
            for method, items in ordered.items()
        }
        for method, report in reports.items():
            per_method_f1[method].append(report.macro_f1)

        for method, items in ordered.items():
            if method == args.reference:
                continue
            comparison = paired_bootstrap(
                truth,
                [r.predicted_label for r in ordered[args.reference]],
                [r.predicted_label for r in items],
                args.reference,
                method,
                labels=GOLD_LABELS,
                n_resamples=args.resamples,
                seed=DEFAULT_SEED,
            )
            per_rival_delta[method].append(comparison.observed_delta)
            per_rival_significant[method] += int(comparison.is_significant)
            rows.append(
                {
                    "fold_seed": seed,
                    "reference": args.reference,
                    "rival": method,
                    "reference_macro_f1": round(reports[args.reference].macro_f1, 6),
                    "rival_macro_f1": round(reports[method].macro_f1, 6),
                    "delta": round(comparison.observed_delta, 6),
                    "ci_low": round(comparison.ci_low, 6),
                    "ci_high": round(comparison.ci_high, 6),
                    "p_value": round(comparison.p_value, 6),
                    "significant": comparison.is_significant,
                }
            )
        seed_rows = [r for r in rows if r["fold_seed"] == seed]
        detail = "  ".join(
            f"{r['rival']}:Δ{r['delta']:+.4f}{'*' if r['significant'] else ''}"
            for r in sorted(seed_rows, key=lambda r: r["rival"])
        )
        print(
            f"  种子 {seed:<4} {args.reference} macroF1="
            f"{reports[args.reference].macro_f1:.4f}   {detail}"
        )

    _atomic_write_csv(out_path, list(rows[0].keys()), rows)

    print()
    print("跨分折种子稳健性检验完成。")
    print(f"  种子：{args.seeds}   折数：{args.n_folds}   评估器：{', '.join(args.evaluator)}")
    print()
    print(f"  {args.reference} 的 Macro-F1：")
    values = per_method_f1[args.reference]
    print(
        f"    均值 {statistics.mean(values):.4f}   标准差 "
        f"{statistics.pstdev(values):.4f}   范围 [{min(values):.4f}, {max(values):.4f}]"
    )
    print()
    print(f"  与各 baseline 的差异（{len(args.seeds)} 个种子）：")
    print(f"    {'对手':<20}{'Δ均值':>9}{'Δ标准差':>10}{'Δ最小':>9}{'Δ最大':>9}{'显著种子数':>12}")
    for method in sorted(per_rival_delta):
        deltas = per_rival_delta[method]
        print(
            f"    {method:<20}{statistics.mean(deltas):>+9.4f}"
            f"{statistics.pstdev(deltas):>10.4f}{min(deltas):>+9.4f}{max(deltas):>+9.4f}"
            f"{per_rival_significant[method]:>8}/{len(args.seeds)}"
        )
    print()
    print(f"  逐种子明细已写入 {out_path}")
    print()
    fragile = [
        m for m in per_rival_delta
        if 0 < per_rival_significant[m] < len(args.seeds)
    ]
    if fragile:
        print("  ⚠ 以下对比**随种子翻转**，论文里必须如实说明，不能只报显著的那次：")
        for method in fragile:
            print(
                f"    {method}：{per_rival_significant[method]}/{len(args.seeds)} 个种子显著"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
