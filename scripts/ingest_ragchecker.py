"""把 RAGChecker 跑出的 checking_outputs.json 摄取成本项目的关系文件。

用法::

    python scripts/ingest_ragchecker.py --split test \
        --outputs outputs/ragchecker/test_checking_outputs.json \
        --calibration outputs/metrics/ragchecker_label_calibration.json \
        --checker-name <判定模型> --extractor-name <抽取模型>

产出::

    outputs/ragchecker/<split>_relations_<evaluator>.jsonl   关系预测
    outputs/ragchecker/<split>_model_run.json                谱系清单

本脚本**不 import ragchecker，也不调用任何模型**，只读它写在磁盘上的结果。

映射表必须校准
--------------
默认要求传 ``--calibration``（由 scripts/calibrate_label_mapping.py 在**验证集**
上产出）。确实只想先跑通链路时可以传 ``--allow-placeholder-mapping``，届时
谱系清单里会把 ``label_mapping_source`` 记成 ``placeholder_default``，
**这种结果不能写进论文**。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rag_ds.data_io import load_samples, write_relation_predictions
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.model_runs import build_model_run_manifest, write_model_run_manifest
from rag_ds.ragchecker_ingest import (
    RAGCheckerOutputError,
    distributions_to_predictions,
    parse_checking_output_probabilities,
    parse_checking_outputs,
)
from rag_ds.relation_evaluation.ragchecker_adapter import (
    DEFAULT_LABEL_MAPPING,
    LabelProbabilityMapping,
    RAGCheckerLabel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "climate_fever_v1" / "manifest.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "ragchecker"


def _project_relative(path: Path) -> str:
    """项目内的路径记成相对项目根目录的 POSIX 路径，项目外的保留绝对路径。

    谱系清单会随仓库分发，写进本机绝对路径既不可移植，也会暴露本地目录结构。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="摄取 RAGChecker 输出，产出关系预测与谱系清单"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument(
        "--outputs", type=Path, required=True, help="RAGChecker 的 checking_outputs.json"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="验证集标签校准结果 JSON（scripts/calibrate_label_mapping.py 产出）",
    )
    parser.add_argument(
        "--allow-placeholder-mapping",
        action="store_true",
        help="允许使用未校准的占位映射表；这种结果不能写进论文",
    )
    parser.add_argument(
        "--use-probabilities",
        action="store_true",
        help=(
            "改用输出里的连续概率（retrieved2response_probabilities），"
            "完全绕开离散标签与映射表。离散标签会把诊断空间压成有限格点，"
            "连续概率没有这个问题"
        ),
    )
    parser.add_argument("--evaluator", default="ragchecker", help="写进关系记录的评估器名")
    parser.add_argument(
        "--evaluator-reliability", type=float, default=1.0, help="评估器可靠性，取值 [0,1]"
    )
    parser.add_argument("--checker-name", required=True, help="RAGChecker 的 checker 模型名")
    parser.add_argument("--extractor-name", default=None, help="RAGChecker 的 extractor 模型名")
    parser.add_argument(
        "--producer",
        default=None,
        help=(
            "产出这批关系的工具名，写进谱系清单。默认读取输出文件里的 "
            "rag_ds_run_info.producer（run_refchecker_nli.py 会写成 refchecker-nli），"
            "读不到时回落为 ragchecker"
        ),
    )
    parser.add_argument("--producer-version", default=None, help="产出工具的版本号")
    parser.add_argument(
        "--strict-single-claim",
        action="store_true",
        help="抽出的子 claim 条数不等于 1 时直接报错，而不是按标签分布做凸组合",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="只解析并打印统计，不写任何文件"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许输出只覆盖部分样本（冒烟测试用）；强制进入 --dry-run",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    return parser.parse_args(argv)


def _load_mapping(args: argparse.Namespace) -> tuple[LabelProbabilityMapping, str, str | None]:
    """按参数取出映射表，返回 (映射表, 来源, 校准文件相对路径)。"""
    if args.calibration is not None:
        payload = json.loads(args.calibration.read_text(encoding="utf-8"))
        split = payload.get("split")
        if split != "validation":
            raise ValueError(
                f"校准文件记录的 split 是 {split!r}，标签映射只能在验证集上校准"
            )
        mapping = LabelProbabilityMapping.model_validate(payload["mapping"])
        return mapping, "calibrated_on_validation", _project_relative(args.calibration)
    if not args.allow_placeholder_mapping:
        raise ValueError(
            "没有传 --calibration。标签映射必须先在验证集上校准；"
            "确实只想跑通链路请显式传 --allow-placeholder-mapping"
        )
    return DEFAULT_LABEL_MAPPING, "placeholder_default", None


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    dry_run = args.dry_run or args.allow_partial

    manifest_path = args.manifest.resolve()
    try:
        if args.use_probabilities:
            # 连续概率不经过映射表；仍记下默认表只是为了让谱系字段完整。
            mapping = DEFAULT_LABEL_MAPPING
            mapping_source = "continuous_probabilities"
            calibration_path = None
            if args.calibration is not None:
                raise ValueError(
                    "--use-probabilities 不经过标签映射表，不应再传 --calibration"
                )
        else:
            mapping, mapping_source, calibration_path = _load_mapping(args)
        manifest = load_dataset_manifest(manifest_path)
        samples_digest = manifest.splits[args.split].samples
        samples_path = verify_artifact(manifest_path.parent, samples_digest)
        samples = load_samples(samples_path)
        payload = json.loads(args.outputs.read_text(encoding="utf-8-sig"))
        run_info = payload.get("rag_ds_run_info") or {}
        # 产出工具以输出文件自报的为准：ragchecker-cli 与 run_refchecker_nli.py
        # 走的是不同的判定路径，谱系里必须能区分。
        producer = args.producer or run_info.get("producer") or "ragchecker"

        if args.allow_partial:
            present = {
                str(item.get("query_id", "")).strip()
                for item in payload.get("results", [])
                if isinstance(item, dict)
            }
            samples = [s for s in samples if s.sample_id in present]
            if not samples:
                raise RAGCheckerOutputError("输出里没有任何属于本 split 的样本")

        distributions = parse_checking_outputs(
            payload, samples, strict_single_claim=args.strict_single_claim
        )
        if args.use_probabilities:
            predictions = parse_checking_output_probabilities(
                payload,
                samples,
                evaluator=args.evaluator,
                evaluator_reliability=args.evaluator_reliability,
            )
        else:
            predictions = distributions_to_predictions(
                distributions,
                mapping,
                evaluator=args.evaluator,
                evaluator_reliability=args.evaluator_reliability,
            )
    except (
        RAGCheckerOutputError,
        DatasetManifestError,
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(f"[摄取失败] {error}", file=sys.stderr)
        return 1

    subclaims = Counter(item.total for item in distributions)
    majority = Counter(
        item.majority_label.value if item.majority_label else "tie"
        for item in distributions
    )

    print("RAGChecker 输出解析完成。")
    print(f"  数据集     {manifest.dataset_name}  split={args.split}")
    print(f"  样本       {len(samples)} 条")
    print(f"  关系网格   {len(distributions)} 个 (claim, document) 组合")
    print(f"  子 claim   条数分布 {dict(sorted(subclaims.items()))}")
    print("  标签分布（多数票）：")
    for label in RAGCheckerLabel:
        count = majority.get(label.value, 0)
        share = count / len(distributions) if distributions else 0.0
        print(f"    {label.value:<14}{count:>6}  ({share:.1%})")
    if majority.get("tie"):
        print(f"    {'tie(并列)':<14}{majority['tie']:>6}")
    print()
    print(f"  产出工具   {producer}")
    print(f"  关系来源   {mapping_source}")
    if args.use_probabilities:
        triples = [(p.p_support, p.p_refute, p.p_unknown) for p in predictions]
        print(f"    连续概率，共 {len(triples)} 条，未经过离散标签")
        print(
            f"    p_support 均值 {sum(t[0] for t in triples)/len(triples):.4f}"
            f"  p_refute 均值 {sum(t[1] for t in triples)/len(triples):.4f}"
            f"  p_unknown 均值 {sum(t[2] for t in triples)/len(triples):.4f}"
        )
        distinct = len({tuple(round(v, 6) for v in t) for t in triples})
        print(f"    不同的三元组取值 {distinct} 种（离散标签路径只有 3 种）")
    else:
        for label in RAGCheckerLabel:
            p_s, p_r, p_u = mapping.probabilities(label)
            print(f"    {label.value:<14}({p_s:.4f}, {p_r:.4f}, {p_u:.4f})")

    if mapping_source == "placeholder_default":
        print()
        print("  警告：使用的是未经校准的占位映射表，结果不能写进论文。")

    if dry_run:
        print()
        if args.allow_partial:
            print("  --allow-partial 只用于冒烟测试，已强制 dry-run，未写任何文件。")
        else:
            print("  --dry-run：未写任何文件。")
        return 0

    out_dir: Path = args.out_dir
    suffix = "_probs" if args.use_probabilities else ""
    relations_name = f"{args.split}_relations_{args.evaluator}{suffix}.jsonl"
    relations_path = out_dir / relations_name
    # 谱系文件名必须带上评估器名：接第二个评估器时两者会写到同一个文件。
    run_path = out_dir / f"{args.split}_model_run_{args.evaluator}{suffix}.json"
    if not args.overwrite:
        existing = [p for p in (relations_path, run_path) if p.exists()]
        if existing:
            print(f"[输出文件已存在] {existing[0]}", file=sys.stderr)
            print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_relation_predictions(
        relations_path, predictions, overwrite=args.overwrite
    )
    run = build_model_run_manifest(
        dataset_name=manifest.dataset_name,
        split=args.split,
        evaluator=args.evaluator,
        producer=producer,
        producer_version=args.producer_version,
        extractor_name=args.extractor_name,
        checker_name=args.checker_name,
        samples=samples_digest,
        predictions_path=relations_path,
        predictions_relative_path=relations_name,
        source_output_path=args.outputs,
        source_output_relative_path=_project_relative(args.outputs),
        label_mapping=mapping,
        label_mapping_source=mapping_source,
        calibration_path=calibration_path,
        multi_claim_policy="strict" if args.strict_single_claim else "mixture",
        evaluator_reliability=args.evaluator_reliability,
    )
    write_model_run_manifest(run_path, run, overwrite=args.overwrite)

    print()
    print("  输出文件：")
    print(f"    {relations_path}   （{written} 条关系预测）")
    print(f"    {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
