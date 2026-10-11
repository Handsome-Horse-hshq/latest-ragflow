"""嵌套交叉验证：把全部样本都用上，给主结论配上置信区间。

用法::

    python scripts/run_cross_validation.py \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" \
        --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" \
        --evaluator refchecker_nli_roberta \
        --single-evaluator refchecker_nli_roberta

为什么要跑
----------
单次留出划分下测试集只有 80–124 条 claim，而 D-S 与最好的 baseline 相差约
0.08 Macro-F1 时，95% 置信区间是**跨 0** 的。交叉验证让每条 claim 都当一次
测试点，n 提升到全体样本数，区间随 ``sqrt(n)`` 收窄。

阈值只在每个外折的**其余各折**上选，测试折全程不参与，因此没有泄漏。

多评估器
--------
``--evaluator`` 可以给多个：各自的关系文件会被**汇总**送进同一条链路，于是
``K_eval``（评估器之间的冲突）才会真正被激活。只给一个评估器时 K_eval 恒为 0，
多评估器融合那一整套机制就没有任何实验支撑。

补充对照
--------
``--stat-gate sr_std max_product sr_std_uprod max_product_uprod``：加入不用
D-S 的两维判定对照组（:mod:`rag_ds.baselines.stat_gate`），判定规则与 D-S
相同，只换坐标轴分数，用来区分「两维判定框架」与「Dempster 融合」各自的
贡献。带 ``_uprod`` 的变体横轴用 ``p_unknown`` 连乘，否则用平均。

``--conflict-margins 0.05 0.1 0.2 0.3 0.5 1.0``：conflict_aware 在三维网格上
单独调好判定阈值、冲突下限与平衡度，而不是固定 margin = 0.1。

两个选项都不给时，输出与原协议逐位一致。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rag_ds.baselines.stat_gate import ALL_STAT_GATE_VARIANTS
from rag_ds.dataset_manifest import DatasetManifestError
from rag_ds.experiments import run_nested_cross_validation
from rag_ds.experiments.data_loading import (
    load_verified_dataset,
    restrict_to_first_documents,
)
from rag_ds.experiments.export import write_predictions_csv
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="嵌套交叉验证 + 显著性检验")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--relations-template",
        required=True,
        help="关系文件路径模板，可用 {split} 与 {evaluator} 占位",
    )
    parser.add_argument(
        "--model-run-template",
        default=None,
        help="谱系文件路径模板；给出时逐个校验身份与摘要",
    )
    parser.add_argument(
        "--evaluator",
        nargs="+",
        required=True,
        help="要汇总的评估器名，可给多个以激活 K_eval",
    )
    parser.add_argument(
        "--single-evaluator",
        required=True,
        help="single_evaluator baseline 使用的评估器名",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-steps", type=int, default=9)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--stat-gate",
        nargs="+",
        default=[],
        choices=list(ALL_STAT_GATE_VARIANTS),
        help="加入不用 D-S 的两维判定对照组，可给多个纵轴统计量",
    )
    parser.add_argument(
        "--conflict-margins",
        nargs="+",
        type=float,
        default=None,
        help="conflict_aware 的平衡度候选；给出时三个参数一起搜",
    )
    parser.add_argument(
        "--first-documents",
        type=int,
        default=None,
        help=(
            "只保留证据至少有这么多篇的 claim，并且每条只用前这么多篇；"
            "用来控制「证据篇数」这个混杂因素"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
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
    out_dir: Path = args.out_dir or (
        PROJECT_ROOT / "outputs" / "metrics" / "cross_validation"
    )
    summary_path = out_dir / "cv_summary.json"
    predictions_path = out_dir / "cv_predictions.csv"
    if not args.overwrite:
        existing = [p for p in (summary_path, predictions_path) if p.exists()]
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
        if args.first_documents is not None:
            samples, predictions = restrict_to_first_documents(
                samples, predictions, args.first_documents
            )
        result = run_nested_cross_validation(
            samples,
            predictions,
            args.single_evaluator,
            n_folds=args.n_folds,
            seed=args.seed,
            grid_steps=args.grid_steps,
            stat_gate_statistics=[ALL_STAT_GATE_VARIANTS[s] for s in args.stat_gate],
            conflict_margins=args.conflict_margins,
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

    by_method: dict[str, list] = defaultdict(list)
    for record in result.predictions:
        by_method[record.method].append(record)
    ordered = {
        method: sorted(items, key=lambda r: (r.sample_id, r.claim_id))
        for method, items in by_method.items()
    }
    truth = [r.gold_label for r in ordered[result.methods[0]]]

    reports = {
        method: classification_report(
            [r.gold_label for r in items], [r.predicted_label for r in items], method=method
        )
        for method, items in ordered.items()
    }
    intervals = {
        method: bootstrap_interval(
            truth,
            [r.predicted_label for r in items],
            method,
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.bootstrap_seed,
        )
        for method, items in ordered.items()
    }
    reference = result.methods[0]
    comparisons = [
        paired_bootstrap(
            truth,
            [r.predicted_label for r in ordered[reference]],
            [r.predicted_label for r in items],
            reference,
            method,
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.bootstrap_seed,
        )
        for method, items in ordered.items()
        if method != reference
    ]
    # 分解用：不用 D-S 的两维判定 vs 只加一条冲突规则的 conflict_aware ——
    # 回答「光有两维判定框架、不做 Dempster 融合，能比 conflict_aware 好多少」。
    stat_gate_methods = [m for m in result.methods if m.startswith("stat_gate_")]
    framework_comparisons = [
        paired_bootstrap(
            truth,
            [r.predicted_label for r in ordered[method]],
            [r.predicted_label for r in ordered["conflict_aware"]],
            method,
            "conflict_aware",
            labels=GOLD_LABELS,
            n_resamples=args.resamples,
            seed=args.bootstrap_seed,
        )
        for method in stat_gate_methods
        if "conflict_aware" in ordered
    ]

    write_predictions_csv(predictions_path, list(result.predictions))
    summary = {
        "dataset_name": manifest.dataset_name,
        "evaluators": list(args.evaluator),
        "n_folds": result.n_folds,
        "fold_seed": result.seed,
        "claim_count": result.claim_count,
        "macro_labels": list(GOLD_LABELS),
        # exclude_none：没开补充对照时不多出空字段，与原协议的输出逐字一致。
        "folds": [
            json.loads(f.model_dump_json(exclude_none=True)) for f in result.folds
        ],
        "reports": {m: json.loads(r.model_dump_json()) for m, r in reports.items()},
        "intervals": {m: json.loads(i.model_dump_json()) for m, i in intervals.items()},
        "comparisons": [json.loads(c.model_dump_json()) for c in comparisons],
    }
    if args.stat_gate:
        summary["stat_gate"] = list(args.stat_gate)
        summary["framework_comparisons"] = [
            json.loads(c.model_dump_json()) for c in framework_comparisons
        ]
    if args.conflict_margins is not None:
        summary["conflict_margins"] = list(args.conflict_margins)
    if args.first_documents is not None:
        summary["first_documents"] = args.first_documents
        summary["gold_label_counts"] = {
            label: truth.count(label) for label in GOLD_LABELS
        }
    _atomic_write_json(summary_path, summary)

    print("嵌套交叉验证完成。")
    print(f"  数据集   {manifest.dataset_name}")
    print(f"  评估器   {', '.join(args.evaluator)}")
    print(f"  折数     {result.n_folds}（分折种子 {result.seed}）")
    print(f"  评估点   {result.claim_count} 条 claim（每条恰好被预测一次）")
    if args.first_documents is not None:
        counts = "，".join(f"{label} {truth.count(label)}" for label in GOLD_LABELS)
        print(f"  篇数控制 只保留证据 ≥ {args.first_documents} 篇的 claim，各用前 "
              f"{args.first_documents} 篇（{counts}）")
    print()
    print("  各折选出的阈值：")
    print(f"    {'折':<4}{'test':>6}{'inner':>7}{'theta':>11}{'K_doc':>11}{'baseline':>10}{'innerF1':>9}")
    for fold in result.folds:
        print(
            f"    {fold.fold:<4}{fold.test_size:>6}{fold.inner_size:>7}"
            f"{fold.ds_thresholds.theta_threshold:>11.4f}"
            f"{fold.ds_thresholds.document_conflict_threshold:>11.4f}"
            f"{fold.baseline_decision_threshold:>10.3f}{fold.inner_macro_f1:>9.4f}"
        )
    if args.conflict_margins is not None:
        print()
        print("  conflict_aware 各折单独选出的参数：")
        print(f"    {'折':<4}{'decision':>10}{'conflict':>10}{'margin':>10}")
        for fold in result.folds:
            print(
                f"    {fold.fold:<4}{fold.conflict_aware_decision_threshold:>10.3f}"
                f"{fold.baseline_conflict_threshold:>10.3f}"
                f"{fold.conflict_aware_margin:>10.3f}"
            )
    for method in stat_gate_methods:
        print()
        print(f"  {method} 各折选出的阈值（横轴证据不足统计量，纵轴冲突统计量）：")
        print(f"    {'折':<4}{'unknown':>11}{'conflict':>11}")
        for fold in result.folds:
            chosen = fold.stat_gate_thresholds[method]
            print(
                f"    {fold.fold:<4}{chosen.theta_threshold:>11.4f}"
                f"{chosen.document_conflict_threshold:>11.4f}"
            )
    print()
    print("  汇总结果（95% 自助置信区间）：")
    print(f"    {'方法':<20}{'Accuracy':>10}{'Macro-F1':>10}{'CI 下界':>10}{'CI 上界':>10}")
    for method in result.methods:
        report, interval = reports[method], intervals[method]
        print(
            f"    {method:<20}{report.accuracy:>10.4f}{report.macro_f1:>10.4f}"
            f"{interval.ci_low:>10.4f}{interval.ci_high:>10.4f}"
        )
    print()
    print(f"  与 {reference} 的配对比较：")
    print(f"    {'对手':<20}{'Δ':>10}{'CI 下界':>10}{'CI 上界':>10}{'p':>8}   结论")
    significant = 0
    for item in comparisons:
        verdict = "显著" if item.is_significant else "不显著（区间跨 0）"
        significant += int(item.is_significant)
        print(
            f"    {item.method_b:<20}{item.observed_delta:>+10.4f}"
            f"{item.ci_low:>+10.4f}{item.ci_high:>+10.4f}{item.p_value:>8.3f}   {verdict}"
        )
    if framework_comparisons:
        print()
        print("  两维判定框架本身（不用 D-S）与 conflict_aware 的配对比较：")
        print(f"    {'方法':<24}{'Δ':>10}{'CI 下界':>10}{'CI 上界':>10}{'p':>8}   结论")
        for item in framework_comparisons:
            verdict = "显著" if item.is_significant else "不显著（区间跨 0）"
            print(
                f"    {item.method_a:<24}{item.observed_delta:>+10.4f}"
                f"{item.ci_low:>+10.4f}{item.ci_high:>+10.4f}{item.p_value:>8.3f}   {verdict}"
            )
    print()
    print(f"  输出 {summary_path}")
    print(f"       {predictions_path}")
    if significant == 0:
        print()
        print("  ⚠ 仍然没有任何一项差异达到显著。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
