"""检测实验的公平对照：m_theta / K_doc 与不用 D-S 的简单统计量比 AUROC。

用法::

    python scripts/run_detection_proxies.py \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" \
        --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" \
        --evaluator refchecker_nli \
        --out-dir outputs/metrics/detection_proxies

为什么要跑
----------
原实验 3 里 baseline 的冲突分数 ``1 - |s - r|`` 会把「全都说不清」当成冲突，
AUROC 低于随机，``K_doc`` 赢它说明不了什么。这里把一组定义合理的统计量
（见 :mod:`rag_ds.experiments.detection_proxies`）事先全部列出、全部报告，
逐一与 D-S 分数做配对自助 AUROC 比较。

所有分数都与阈值无关，因此直接在全体 claim 上计算，不需要交叉验证。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from rag_ds.dataset_manifest import DatasetManifestError
from rag_ds.experiments.data_loading import load_verified_dataset
from rag_ds.experiments.detection_proxies import compare_detection, detection_scores
from rag_ds.integrity import PipelineError
from rag_ds.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED
from rag_ds.model_runs import ModelRunError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="检测实验：D-S 分数 vs 简单统计量")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relations-template", required=True)
    parser.add_argument("--model-run-template", default=None)
    parser.add_argument("--evaluator", nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_write(path: Path, write, encoding: str) -> None:
    """先写同目录临时文件，成功后再替换。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            write(handle)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    out_dir: Path = args.out_dir or (
        PROJECT_ROOT / "outputs" / "metrics" / "detection_proxies"
    )
    summary_path = out_dir / "detection_summary.json"
    table_path = out_dir / "detection_results.csv"
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
        rows = detection_scores(samples, predictions)
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

    results = compare_detection(
        rows, n_resamples=args.resamples, seed=args.bootstrap_seed
    )

    def _write_summary(handle) -> None:
        json.dump(
            {
                "dataset_name": manifest.dataset_name,
                "evaluators": list(args.evaluator),
                "claim_count": len(rows),
                "tasks": {
                    task: json.loads(result.model_dump_json())
                    for task, result in results.items()
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    def _write_table(handle) -> None:
        writer = csv.writer(handle)
        writer.writerow(
            ["task", "positive_label", "score", "auroc", "auprc", "best_f1",
             "delta_reference_minus_score", "delta_ci_low", "delta_ci_high",
             "p_value", "significant"]
        )
        for task, result in results.items():
            by_rival = {c.rival: c for c in result.comparisons}
            for report in result.reports:
                row = [task, result.positive_label, report.score_name,
                       f"{report.auroc:.6f}", f"{report.auprc:.6f}", f"{report.best_f1:.6f}"]
                item = by_rival.get(report.score_name)
                if item is None:
                    row += ["", "", "", "", ""]
                else:
                    row += [f"{item.observed_delta:+.6f}", f"{item.ci_low:+.6f}",
                            f"{item.ci_high:+.6f}", f"{item.p_value:.4f}",
                            str(item.is_significant)]
                writer.writerow(row)

    _atomic_write(summary_path, _write_summary, "utf-8")
    _atomic_write(table_path, _write_table, "utf-8-sig")

    print("检测实验（公平对照）完成。")
    print(f"  数据集   {manifest.dataset_name}")
    print(f"  评估器   {', '.join(args.evaluator)}")
    print(f"  评估点   {len(rows)} 条 claim（分数与阈值无关，不需要交叉验证）")
    for task, result in results.items():
        by_rival = {c.rival: c for c in result.comparisons}
        positives = result.reports[0].positive_count
        print()
        print(f"  {task}：正类 {result.positive_label}（{positives} 条），参照 {result.reference}")
        print(f"    {'分数':<24}{'AUROC':>8}{'AUPRC':>8}{'参照−它':>10}{'CI 下界':>10}{'CI 上界':>10}{'p':>8}   结论")
        for report in result.reports:
            item = by_rival.get(report.score_name)
            line = f"    {report.score_name:<24}{report.auroc:>8.4f}{report.auprc:>8.4f}"
            if item is not None:
                verdict = "显著" if item.is_significant else "不显著（区间跨 0）"
                line += (
                    f"{item.observed_delta:>+10.4f}{item.ci_low:>+10.4f}"
                    f"{item.ci_high:>+10.4f}{item.p_value:>8.3f}   {verdict}"
                )
            print(line)
    print()
    print(f"  输出 {summary_path}")
    print(f"       {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
