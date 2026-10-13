"""换了关系评估器之后的两层配对比较：关系这一层，以及四分类判定这一层。

用法::

    python scripts/compare_relation_inputs.py \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" \
        --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" \
        --reference deepseek_flash --rival refchecker_nli --with-oracle \
        --reference-cv outputs/metrics/cross_validation_deepseek_flash_controls/cv_predictions.csv \
        --rival-cv outputs/metrics/cross_validation_controls/cv_predictions.csv \
        --out outputs/metrics/relation_input_comparison/summary.json

``--with-oracle`` 只适用于清单里登记了人工逐篇标注的数据集（CLIMATE-FEVER）；
两个 ``--*-cv`` 要么都给、要么都不给。两部分至少要有一部分。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from rag_ds.data_io import load_relation_predictions
from rag_ds.dataset_manifest import DatasetManifestError, verify_artifact
from rag_ds.experiments.data_loading import ALL_SPLITS, load_verified_dataset
from rag_ds.experiments.relation_input_comparison import (
    compare_prediction_runs,
    read_cv_predictions,
    relation_level_comparison,
)
from rag_ds.integrity import PipelineError
from rag_ds.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED
from rag_ds.model_runs import ModelRunError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="两份关系输入的配对比较")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relations-template", required=True)
    parser.add_argument("--model-run-template", default=None)
    parser.add_argument("--reference", required=True, help="参照评估器名（差值 = 参照 − 对手）")
    parser.add_argument("--rival", required=True, help="对手评估器名")
    parser.add_argument(
        "--with-oracle",
        action="store_true",
        help="用清单里登记的人工逐篇标注，在关系这一层比较两个评估器",
    )
    parser.add_argument("--reference-cv", type=Path, default=None, help="参照评估器的 cv_predictions.csv")
    parser.add_argument("--rival-cv", type=Path, default=None, help="对手评估器的 cv_predictions.csv")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (args.reference_cv is None) != (args.rival_cv is None):
        parser.error("--reference-cv 与 --rival-cv 要么都给、要么都不给")
    if not args.with_oracle and args.reference_cv is None:
        parser.error("至少要有一部分可比：--with-oracle，或两个 --*-cv")
    if args.reference == args.rival:
        parser.error("--reference 与 --rival 不能相同")
    return args


def _atomic_write_json(path: Path, payload: dict) -> None:
    """先写同目录临时文件，成功后再替换。"""
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
    if args.out.exists() and not args.overwrite:
        print(f"[输出文件已存在] {args.out}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    summary: dict = {"reference": args.reference, "rival": args.rival}
    try:
        manifest, _, reference = load_verified_dataset(
            args.manifest, args.relations_template, [args.reference], args.model_run_template
        )
        _, _, rival = load_verified_dataset(
            args.manifest, args.relations_template, [args.rival], args.model_run_template
        )
        summary["dataset_name"] = manifest.dataset_name

        if args.with_oracle:
            oracle = []
            for split in ALL_SPLITS:
                artifact = manifest.splits[split].relation_predictions
                if artifact is None:
                    raise ValueError(
                        f"清单没有登记 {split} 的人工逐篇标注，不能用 --with-oracle"
                    )
                oracle.extend(
                    load_relation_predictions(
                        verify_artifact(args.manifest.resolve().parent, artifact)
                    )
                )
            summary["relation_level"] = relation_level_comparison(
                oracle,
                reference,
                rival,
                reference_name=args.reference,
                rival_name=args.rival,
                n_resamples=args.resamples,
                seed=args.bootstrap_seed,
            )

        if args.reference_cv is not None:
            comparisons = compare_prediction_runs(
                read_cv_predictions(args.reference_cv),
                read_cv_predictions(args.rival_cv),
                reference_name=args.reference,
                rival_name=args.rival,
                n_resamples=args.resamples,
                seed=args.bootstrap_seed,
            )
            summary["reference_cv"] = str(args.reference_cv)
            summary["rival_cv"] = str(args.rival_cv)
            summary["decision_level"] = [json.loads(c.model_dump_json()) for c in comparisons]
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

    _atomic_write_json(args.out, summary)

    print(f"{summary['dataset_name']}：{args.reference} - {args.rival}")
    level = summary.get("relation_level")
    if level:
        print(f"  关系这一层（人工多数票唯一的 {level['pairs_with_unique_majority']} 个组合，"
              f"共 {level['pair_count']} 个）")
        for name, stats in level["evaluators"].items():
            print(f"    {name:<24} argmax 准确率 {stats['argmax_accuracy']:.4f}  {stats['argmax_counts']}")
        print(f"    两者 argmax 一致率 {level['argmax_agreement']:.4f}")
        for c in level["auroc_comparisons"]:
            print(f"    AUROC {c['positive_label']:<8} {c['auroc_reference']:.4f} vs {c['auroc_rival']:.4f}"
                  f"  Δ {c['observed_delta']:+.4f}  [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
                  f"  p={c['p_value']:.3f}")
    for c in summary.get("decision_level", []):
        print(f"  {c['method_a'].split('@')[0]:<30} {c['observed_a']:.4f} vs {c['observed_b']:.4f}"
              f"  Δ {c['observed_delta']:+.4f}  [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
              f"  p={c['p_value']:.3f}")
    print(f"  输出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
