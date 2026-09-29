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

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.experiments import run_nested_cross_validation
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
from rag_ds.model_runs import ModelRunError, verify_model_run_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


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

    manifest_path = args.manifest.resolve()
    try:
        manifest = load_dataset_manifest(manifest_path)
        samples = []
        predictions = []
        for split in SPLITS:
            samples_path = verify_artifact(
                manifest_path.parent, manifest.splits[split].samples
            )
            samples.extend(load_samples(samples_path))
            for evaluator in args.evaluator:
                relations = Path(
                    args.relations_template.format(split=split, evaluator=evaluator)
                )
                if args.model_run_template:
                    run_path = Path(
                        args.model_run_template.format(
                            split=split, evaluator=evaluator
                        )
                    )
                    verify_model_run_artifacts(
                        run_path, manifest_path, samples_path, relations, split
                    )
                predictions.extend(load_relation_predictions(relations))

        result = run_nested_cross_validation(
            samples,
            predictions,
            args.single_evaluator,
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

    write_predictions_csv(predictions_path, list(result.predictions))
    _atomic_write_json(
        summary_path,
        {
            "dataset_name": manifest.dataset_name,
            "evaluators": list(args.evaluator),
            "n_folds": result.n_folds,
            "fold_seed": result.seed,
            "claim_count": result.claim_count,
            "macro_labels": list(GOLD_LABELS),
            "folds": [json.loads(f.model_dump_json()) for f in result.folds],
            "reports": {m: json.loads(r.model_dump_json()) for m, r in reports.items()},
            "intervals": {m: json.loads(i.model_dump_json()) for m, i in intervals.items()},
            "comparisons": [json.loads(c.model_dump_json()) for c in comparisons],
        },
    )

    print("嵌套交叉验证完成。")
    print(f"  数据集   {manifest.dataset_name}")
    print(f"  评估器   {', '.join(args.evaluator)}")
    print(f"  折数     {result.n_folds}（分折种子 {result.seed}）")
    print(f"  评估点   {result.claim_count} 条 claim（每条恰好被预测一次）")
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
    print()
    print(f"  输出 {summary_path}")
    print(f"       {predictions_path}")
    if significant == 0:
        print()
        print("  ⚠ 仍然没有任何一项差异达到显著。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
