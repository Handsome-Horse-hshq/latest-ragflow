"""CONFACT 文档级可靠性实验：来源可信度确有差异时，可靠性折扣有没有用。

三步（第 2 步在 .venv-refchecker 里跑，其余在主环境）::

    # 1. 导出 NLI 输入
    python scripts/run_confact_reliability.py export

    # 2. 跑 NLI（与其他数据集同一个脚本、同一个模型）
    set HF_HOME=.model-cache
    .venv-refchecker\\Scripts\\python.exe scripts/run_refchecker_nli_chunked.py ^
        --input outputs/ragchecker_confact/checking_inputs.json ^
        --output outputs/ragchecker_confact/checking_outputs_probs.json ^
        --emit-probabilities

    # 3. 分析
    python scripts/run_confact_reliability.py analyse

回答的两个问题见 :mod:`rag_ds.experiments.reliability_verdict`：可靠性有没有
用；D-S 的折扣是否优于把可靠性当权重的加权平均。没有需要调的阈值，所以
不做交叉验证，直接在全体 claim 上评估并配对自助检验。

事先定好的可靠性映射是主结果；``low`` 与「无评级」两个取值的敏感性扫描
**全部**写进 ``sensitivity.csv``，不挑。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from itertools import product
from pathlib import Path

from rag_ds.data_io import load_samples
from rag_ds.dataset_manifest import file_sha256
from rag_ds.datasets.confact import CREDIBILITY_RELIABILITY, UNRATED_RELIABILITY
from rag_ds.experiments.reliability_verdict import (
    VERDICT_LABELS,
    VERDICT_METHODS,
    VerdictScore,
    reliability_from_credibility,
    verdict_scores,
)
from rag_ds.integrity import PipelineError
from rag_ds.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    classification_report,
    paired_auroc_bootstrap,
    paired_bootstrap,
)
from rag_ds.metrics.detection import detection_report
from rag_ds.ragchecker_ingest import (
    RAGCheckerOutputError,
    parse_checking_output_probabilities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "confact_humc_v1"
DEFAULT_NLI_DIR = PROJECT_ROOT / "outputs" / "ragchecker_confact"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "metrics" / "confact_reliability"
EVALUATOR = "refchecker_nli"
POSITIVE = VERDICT_LABELS[0]  # supported

#: 敏感性扫描：low 与「无评级」的可靠性取值（high 固定 1.0、medium 固定 0.6）。
SENSITIVITY_LOW = (0.0, 0.2, 0.4)
SENSITIVITY_UNRATED = (0.4, 0.6, 0.8, 1.0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CONFACT 文档级可靠性实验")
    parser.add_argument("command", choices=("export", "analyse"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--nli-dir", type=Path, default=DEFAULT_NLI_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_write(path: Path, write, encoding: str = "utf-8") -> None:
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


def _export(args: argparse.Namespace) -> int:
    """把样本写成 RefChecker NLI 脚本的输入格式。"""
    out_path = args.nli_dir / "checking_inputs.json"
    if out_path.exists() and not args.overwrite:
        print(f"[输出文件已存在] {out_path}（加 --overwrite 可覆盖）", file=sys.stderr)
        return 1
    samples = load_samples(args.data_dir / "samples.jsonl")
    results = [
        {
            "query_id": sample.sample_id,
            "query": sample.question,
            # 只跑 faithfulness，不需要参考答案；写占位说明而不是复制 claim。
            "gt_answer": "(not used: only the claim-document relation grid is needed)",
            "response": sample.answer,
            "retrieved_context": [
                {"doc_id": chunk.doc_id, "text": chunk.text} for chunk in sample.contexts
            ],
        }
        for sample in samples
    ]
    _atomic_write(
        out_path,
        lambda handle: json.dump({"results": results}, handle, ensure_ascii=False, indent=2),
    )
    pairs = sum(len(item["retrieved_context"]) for item in results)
    print(f"NLI 输入已导出：{len(results)} 条 claim，{pairs} 个 (claim, 文档) 组合")
    print(f"  输出 {out_path}")
    return 0


def _metrics(scores: list[VerdictScore]) -> dict:
    """一个 (条件, 方法) 的分类指标与 margin 的 AUROC。"""
    truth = [s.gold_label for s in scores]
    predicted = [s.predicted_label for s in scores]
    report = classification_report(
        truth, predicted, method=scores[0].method, macro_labels=VERDICT_LABELS
    )
    detection = detection_report(
        [s.margin for s in scores], [t == POSITIVE for t in truth], "margin", POSITIVE
    )
    return {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "auroc": detection.auroc,
        "undetermined": sum(1 for p in predicted if p not in VERDICT_LABELS),
        "ds_undefined": sum(1 for s in scores if s.ds_undefined),
        "predicted_supported": sum(1 for p in predicted if p == POSITIVE),
    }


def _compare(a: list[VerdictScore], b: list[VerdictScore], name_a: str, name_b: str, args) -> dict:
    """a 减 b：Macro-F1 与 AUROC 的配对自助比较（两者覆盖同一批 claim、同序）。"""
    assert [(s.sample_id, s.claim_id) for s in a] == [(s.sample_id, s.claim_id) for s in b]
    truth = [s.gold_label for s in a]
    f1 = paired_bootstrap(
        truth,
        [s.predicted_label for s in a],
        [s.predicted_label for s in b],
        name_a,
        name_b,
        labels=VERDICT_LABELS,
        n_resamples=args.resamples,
        seed=args.bootstrap_seed,
    )
    (auroc,) = paired_auroc_bootstrap(
        [t == POSITIVE for t in truth],
        [s.margin for s in a],
        {name_b: [s.margin for s in b]},
        name_a,
        POSITIVE,
        n_resamples=args.resamples,
        seed=args.bootstrap_seed,
    )
    return {
        "a": name_a,
        "b": name_b,
        "macro_f1": json.loads(f1.model_dump_json()),
        "macro_f1_significant": f1.is_significant,
        "auroc": json.loads(auroc.model_dump_json()),
        "auroc_significant": auroc.is_significant,
    }


def _analyse(args: argparse.Namespace) -> int:
    """读 NLI 输出，比较有无可靠性、D-S 与加权平均。"""
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"[输出文件已存在] {summary_path}（加 --overwrite 可覆盖）", file=sys.stderr)
        return 1

    samples_path = args.data_dir / "samples.jsonl"
    outputs_path = args.nli_dir / "checking_outputs_probs.json"
    try:
        meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
        if file_sha256(samples_path) != meta["samples"]["sha256"]:
            raise ValueError(f"{samples_path} 与 meta.json 记录的摘要不一致")
        samples = load_samples(samples_path)
        credibility = {
            context["doc_id"]: context["credibility"]
            for line in (args.data_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
            for context in json.loads(line)["contexts"]
        }
        payload = json.loads(outputs_path.read_text(encoding="utf-8-sig"))
        predictions = parse_checking_output_probabilities(
            payload, samples, evaluator=EVALUATOR
        )
    except (OSError, ValueError, KeyError, RAGCheckerOutputError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    def _run(reliability) -> dict[str, list[VerdictScore]]:
        by_method: dict[str, list[VerdictScore]] = {m: [] for m in VERDICT_METHODS}
        for score in verdict_scores(samples, predictions, reliability):
            by_method[score.method].append(score)
        return by_method

    try:
        conditions = {
            "none": _run(1.0),
            "credibility": _run(
                reliability_from_credibility(
                    credibility, CREDIBILITY_RELIABILITY, UNRATED_RELIABILITY
                )
            ),
        }
    except (PipelineError, ValueError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    results = {
        condition: {method: _metrics(scores) for method, scores in by_method.items()}
        for condition, by_method in conditions.items()
    }
    none, cred = conditions["none"], conditions["credibility"]
    comparisons = [
        # 可靠性有没有用：同一方法，有可靠性 − 无可靠性。
        _compare(cred["ds"], none["ds"], "ds@credibility", "ds@none", args),
        _compare(
            cred["weighted_average"], none["weighted_average"],
            "weighted_average@credibility", "weighted_average@none", args,
        ),
        # D-S 的折扣是否优于加权：同一份可靠性，D-S − 加权平均。
        _compare(cred["ds"], cred["weighted_average"],
                 "ds@credibility", "weighted_average@credibility", args),
        _compare(none["ds"], none["weighted_average"], "ds@none", "weighted_average@none", args),
        _compare(cred["ds"], none["majority_vote"], "ds@credibility", "majority_vote", args),
    ]

    sensitivity = []
    for low, unrated in product(SENSITIVITY_LOW, SENSITIVITY_UNRATED):
        mapping = {**CREDIBILITY_RELIABILITY, "low": low}
        by_method = _run(reliability_from_credibility(credibility, mapping, unrated))
        for method in ("ds", "weighted_average"):
            sensitivity.append(
                {"low": low, "unrated": unrated, "method": method, **_metrics(by_method[method])}
            )

    def _write_summary(handle) -> None:
        json.dump(
            {
                "dataset_name": meta["dataset_name"],
                "evaluator": EVALUATOR,
                "checker_model": (payload.get("rag_ds_run_info") or {}).get("checker_model"),
                "claim_count": len(samples),
                "label_counts": meta["label_counts"],
                "document_credibility_counts": meta["document_credibility_counts"],
                "credibility_reliability": CREDIBILITY_RELIABILITY,
                "unrated_reliability": UNRATED_RELIABILITY,
                "samples_sha256": meta["samples"]["sha256"],
                "nli_outputs_sha256": file_sha256(outputs_path),
                "results": results,
                "comparisons": comparisons,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    def _write_results(handle) -> None:
        writer = csv.writer(handle)
        writer.writerow(["reliability", "method", "accuracy", "macro_f1", "auroc_margin",
                         "undetermined", "ds_undefined", "predicted_supported"])
        for condition, by_method in results.items():
            for method, m in by_method.items():
                writer.writerow([condition, method, f"{m['accuracy']:.6f}", f"{m['macro_f1']:.6f}",
                                 f"{m['auroc']:.6f}", m["undetermined"], m["ds_undefined"],
                                 m["predicted_supported"]])

    def _write_sensitivity(handle) -> None:
        writer = csv.writer(handle)
        writer.writerow(["low", "unrated", "method", "accuracy", "macro_f1", "auroc_margin"])
        for row in sensitivity:
            writer.writerow([row["low"], row["unrated"], row["method"], f"{row['accuracy']:.6f}",
                             f"{row['macro_f1']:.6f}", f"{row['auroc']:.6f}"])

    def _write_scores(handle) -> None:
        writer = csv.writer(handle)
        writer.writerow(["reliability", "method", "sample_id", "claim_id", "gold_label",
                         "margin", "predicted_label"])
        for condition, by_method in conditions.items():
            for scores in by_method.values():
                for s in scores:
                    writer.writerow([condition, s.method, s.sample_id, s.claim_id, s.gold_label,
                                     f"{s.margin:.6f}", s.predicted_label])

    _atomic_write(summary_path, _write_summary)
    _atomic_write(args.out_dir / "results.csv", _write_results, "utf-8-sig")
    _atomic_write(args.out_dir / "sensitivity.csv", _write_sensitivity, "utf-8-sig")
    _atomic_write(args.out_dir / "verdict_scores.csv", _write_scores, "utf-8-sig")

    print("CONFACT 文档级可靠性实验完成。")
    print(f"  {len(samples)} 条 claim，标签分布 {meta['label_counts']}")
    print(f"  文档可信度 {meta['document_credibility_counts']}")
    print(f"  可靠性映射 {CREDIBILITY_RELIABILITY}，无评级 {UNRATED_RELIABILITY}")
    print()
    print(f"    {'可靠性':<14}{'方法':<20}{'Accuracy':>10}{'Macro-F1':>10}{'AUROC':>8}"
          f"{'未定':>6}{'D-S 无定义':>12}")
    for condition, by_method in results.items():
        for method, m in by_method.items():
            print(f"    {condition:<14}{method:<20}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}"
                  f"{m['auroc']:>8.4f}{m['undetermined']:>6}{m['ds_undefined']:>12}")
    print()
    print(f"    {'比较（前者 - 后者）':<52}{'ΔMacro-F1':>10}{'95% CI':>20}{'ΔAUROC':>9}{'95% CI':>20}")
    for item in comparisons:
        f1, auroc = item["macro_f1"], item["auroc"]
        f1_ci = f"[{f1['ci_low']:+.3f},{f1['ci_high']:+.3f}]{'*' if item['macro_f1_significant'] else ' '}"
        auroc_ci = (f"[{auroc['ci_low']:+.3f},{auroc['ci_high']:+.3f}]"
                    f"{'*' if item['auroc_significant'] else ' '}")
        print(f"    {item['a'] + ' - ' + item['b']:<52}{f1['observed_delta']:>+10.4f}{f1_ci:>20}"
              f"{auroc['observed_delta']:>+9.4f}{auroc_ci:>20}")
    print("    * = 95% 区间不跨 0")
    print()
    print(f"  输出目录 {args.out_dir}（含 sensitivity.csv：low × 无评级 的 12 组取值）")
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    return _export(args) if args.command == "export" else _analyse(args)


if __name__ == "__main__":
    raise SystemExit(main())
