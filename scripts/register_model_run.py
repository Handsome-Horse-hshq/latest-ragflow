"""为「直接产出的」模型关系文件登记谱系清单。

用法::

    python scripts/register_model_run.py \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --split validation \
        --relations outputs/ragchecker_v2/validation_relations_flan_t5_mnli_probs.jsonl \
        --run-info outputs/ragchecker_v2/validation_relations_flan_t5_mnli_probs.jsonl.run_info.json \
        --out outputs/ragchecker_v2/validation_model_run_flan_t5_mnli_probs.json

为什么需要
----------
RefChecker 路径的关系文件由 ``ingest_ragchecker.py`` 产出，顺带写好
``ModelRunManifest``。而有些评估器（如 ``run_t5_judge_chunked.py``）直接写出
``RelationPrediction`` JSONL，没有中间的 ``checking_outputs.json``，也就没有
谱系清单 —— 交叉验证脚本的身份与摘要校验会因此过不去，或者只能关掉校验。

本脚本补上这一步，并在登记前做三项检查：

1. 关系文件每一行都通过 ``RelationPrediction`` 校验；
2. 关系文件只含**一个**评估器，且与 run_info 自报的一致；
3. 覆盖了该 split 样本的**完整** (claim, document) 网格，不多不少。

直接产出的文件没有独立的原始输出，``source_output`` 记为关系文件本身
（摘要相同），这一点写在清单的 ``producer`` 字段旁，读者可以据此区分。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.model_runs import build_model_run_manifest, write_model_run_manifest
from rag_ds.relation_evaluation.ragchecker_adapter import DEFAULT_LABEL_MAPPING

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser = argparse.ArgumentParser(description="为直接产出的关系文件登记谱系")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument(
        "--run-info", type=Path, required=True, help="评估器脚本写出的 run_info JSON"
    )
    parser.add_argument("--out", type=Path, required=True, help="谱系清单输出路径")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.out.exists() and not args.overwrite:
        print(f"[输出文件已存在] {args.out}（加 --overwrite 可覆盖）", file=sys.stderr)
        return 1

    manifest_path = args.manifest.resolve()
    try:
        info = json.loads(args.run_info.read_text(encoding="utf-8"))
        for key in ("evaluator", "checker_model", "producer"):
            if not info.get(key):
                raise ValueError(f"run_info 缺少字段 {key!r}")

        manifest = load_dataset_manifest(manifest_path)
        samples_digest = manifest.splits[args.split].samples
        samples = load_samples(verify_artifact(manifest_path.parent, samples_digest))
        predictions = load_relation_predictions(args.relations)

        evaluators = {p.evaluator for p in predictions}
        if evaluators != {info["evaluator"]}:
            raise ValueError(
                f"关系文件里的评估器 {sorted(evaluators)} 与 run_info 自报的 "
                f"{info['evaluator']!r} 不一致"
            )

        expected = {
            (s.sample_id, c.claim_id, d.doc_id)
            for s in samples
            for c in s.claims
            for d in s.contexts
        }
        actual = [(p.sample_id, p.claim_id, p.doc_id) for p in predictions]
        if len(actual) != len(set(actual)):
            raise ValueError("关系文件里有重复的 (sample, claim, doc) 组合")
        missing = expected - set(actual)
        extra = set(actual) - expected
        if missing or extra:
            raise ValueError(
                f"关系网格不完整：缺 {len(missing)} 个、多 {len(extra)} 个组合"
                f"（缺的前几个：{sorted(missing)[:3]}）"
            )
    except (DatasetManifestError, ValueError, KeyError, OSError) as error:
        print(f"[登记失败] {error}", file=sys.stderr)
        return 1

    run = build_model_run_manifest(
        dataset_name=manifest.dataset_name,
        split=args.split,
        evaluator=info["evaluator"],
        producer=info["producer"],
        producer_version=info.get("scoring_version"),
        checker_name=info["checker_model"],
        samples=samples_digest,
        predictions_path=args.relations,
        predictions_relative_path=args.relations.name,
        # 直接产出的文件没有独立原始输出，source 记为关系文件本身。
        source_output_path=args.relations,
        source_output_relative_path=_project_relative(args.relations),
        label_mapping=DEFAULT_LABEL_MAPPING,
        label_mapping_source="continuous_probabilities",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_model_run_manifest(args.out, run, overwrite=args.overwrite)

    print(f"已登记 {args.split}：{len(predictions)} 条，评估器 {info['evaluator']}")
    print(f"  打分版本 {info.get('scoring_version')}")
    print(f"  清单 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
