"""给主对比结果配上置信区间与显著性判断。

用法::

    python scripts/run_significance.py \
        --predictions outputs/predictions/climate_fever_refchecker_nli_probs_test.csv

读取 ``run_experiment.py`` 产出的逐条预测明细，对每个方法算 95% 自助置信区间，
并把 D-S 与三个 baseline 逐一做**配对**自助检验。

为什么必须报区间
----------------
测试集只有 80 条 claim。D-S 与最好的 baseline 相差约 0.08 Macro-F1，看上去不小，
但 95% 区间跨 0 —— 只报点估计等于把「可能只是噪声」说成「方法更好」。

区间是否跨 0 才是结论；p 值是自助近似，只作参考。

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

from rag_ds.experiments.comparison import DS_METHOD
from rag_ds.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    GOLD_LABELS,
    bootstrap_interval,
    paired_bootstrap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="给对比结果配上置信区间")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="run_experiment.py 产出的逐条预测 CSV",
    )
    parser.add_argument(
        "--reference",
        default=DS_METHOD,
        help="作为对照基准的方法名，其余方法与它逐一配对比较",
    )
    parser.add_argument(
        "--metric", choices=("macro_f1", "accuracy"), default="macro_f1"
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None, help="结果 JSON 路径")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    out_path: Path = args.out or (
        PROJECT_ROOT / "outputs" / "metrics" / f"{args.predictions.stem}_significance.json"
    )
    if out_path.exists() and not args.overwrite:
        print(f"[输出文件已存在] {out_path}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    try:
        rows = list(csv.DictReader(args.predictions.open(encoding="utf-8-sig")))
    except OSError as error:
        print(f"[读取失败] {error}", file=sys.stderr)
        return 1
    if not rows:
        print("[输入错误] 预测明细为空", file=sys.stderr)
        return 1

    by_method: dict[str, dict[tuple[str, str], tuple[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["sample_id"], row["claim_id"])
        by_method[row["method"]][key] = (row["gold_label"], row["predicted_label"])

    if args.reference not in by_method:
        print(
            f"[输入错误] 明细里没有基准方法 {args.reference!r}；"
            f"现有方法：{sorted(by_method)}",
            file=sys.stderr,
        )
        return 1

    # 所有方法必须覆盖同一批 claim —— 配对检验的前提。
    keys = sorted(by_method[args.reference])
    for method, table in by_method.items():
        if sorted(table) != keys:
            print(
                f"[输入错误] 方法 {method!r} 覆盖的 claim 与基准不一致，无法配对比较",
                file=sys.stderr,
            )
            return 1

    truth = [by_method[args.reference][k][0] for k in keys]

    intervals = [
        bootstrap_interval(
            truth,
            [by_method[method][k][1] for k in keys],
            method,
            metric=args.metric,
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        for method in sorted(by_method)
    ]
    comparisons = [
        paired_bootstrap(
            truth,
            [by_method[args.reference][k][1] for k in keys],
            [by_method[method][k][1] for k in keys],
            args.reference,
            method,
            metric=args.metric,
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        for method in sorted(by_method)
        if method != args.reference
    ]

    print("显著性检验完成（配对自助重采样）。")
    print(f"  指标 {args.metric}   claim 数量 {len(keys)}")
    print(f"  重采样 {args.resamples} 次，种子 {args.seed}")
    print(f"  Macro-F1 固定标签集：{'|'.join(GOLD_LABELS)}")
    print()
    print(f"  各方法 95% 置信区间：")
    print(f"    {'方法':<20}{'观测值':>10}{'CI 下界':>10}{'CI 上界':>10}")
    for item in intervals:
        print(
            f"    {item.method:<20}{item.observed:>10.4f}"
            f"{item.ci_low:>10.4f}{item.ci_high:>10.4f}"
        )
    print()
    print(f"  与 {args.reference} 的配对比较：")
    print(f"    {'对手':<20}{'Δ':>10}{'CI 下界':>10}{'CI 上界':>10}{'p':>8}   结论")
    any_significant = False
    for item in comparisons:
        verdict = "显著" if item.is_significant else "不显著（区间跨 0）"
        any_significant = any_significant or item.is_significant
        print(
            f"    {item.method_b:<20}{item.observed_delta:>+10.4f}"
            f"{item.ci_low:>+10.4f}{item.ci_high:>+10.4f}{item.p_value:>8.3f}   {verdict}"
        )

    _atomic_write_json(
        out_path,
        {
            "metric": args.metric,
            "reference": args.reference,
            "claim_count": len(keys),
            "macro_labels": list(GOLD_LABELS),
            "intervals": [json.loads(i.model_dump_json()) for i in intervals],
            "comparisons": [json.loads(c.model_dump_json()) for c in comparisons],
        },
    )
    print()
    print(f"  输出 {out_path}")
    if not any_significant:
        print()
        print("  ⚠ 没有任何一项差异达到显著：当前样本量撑不起观测到的差距。")
        print("    论文里必须如实写明，或者扩大评估样本（如交叉验证）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
