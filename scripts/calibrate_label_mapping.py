"""在验证集上校准 RAGChecker 的「标签 → 三元概率」映射表。

用法::

    python scripts/calibrate_label_mapping.py \
        --outputs outputs/ragchecker/validation_checking_outputs.json \
        --out outputs/metrics/ragchecker_label_calibration.json

做了什么
--------
读取验证集上 RAGChecker 的标签，与 CLIMATE-FEVER 的人工证据投票分布对齐，
对每个标签求条件平均，得到「RAGChecker 说 entailment 时，人工投票平均长什么样」。

三条红线（脚本会强制执行前两条）
--------------------------------
1. 只接受 **validation**：样本与 oracle 都必须是数据集清单里登记的验证集文件，
   摘要对不上直接报错；
2. 某个标签在验证集上一次都没出现时报错，不静默沿用占位值
   （确需继续请传 ``--allow-default-for-unseen``）；
3. 论文必须写明映射表由验证集人工投票标定 —— 测试集全程不接触 oracle。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import DatasetManifestError, verify_validation_artifacts
from rag_ds.ragchecker_ingest import RAGCheckerOutputError, parse_checking_outputs
from rag_ds.relation_evaluation.ragchecker_adapter import RAGCheckerLabel
from rag_ds.tuning import SplitName, calibrate_label_mapping
from rag_ds.tuning.label_calibration import LabelCalibrationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "climate_fever_v1"
DEFAULT_MANIFEST = DATASET_DIR / "manifest.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "metrics" / "ragchecker_label_calibration.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在验证集上校准标签映射表")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", type=Path, default=DATASET_DIR / "validation.jsonl")
    parser.add_argument(
        "--oracle", type=Path, default=DATASET_DIR / "validation_relations.jsonl"
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="验证集上 RAGChecker 的 checking_outputs.json",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--oracle-evaluator",
        default="climate_fever_human_vote_distribution",
        help="只使用该评估器名下的 oracle 记录",
    )
    parser.add_argument(
        "--allow-default-for-unseen",
        action="store_true",
        help="某标签在验证集上没出现时沿用占位默认值，而不是报错",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
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
    if args.out.exists() and not args.overwrite:
        print(f"[输出文件已存在] {args.out}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    try:
        verify_validation_artifacts(args.manifest, args.samples, args.oracle)
        samples = load_samples(args.samples)
        oracle = load_relation_predictions(args.oracle)
        payload = json.loads(args.outputs.read_text(encoding="utf-8-sig"))
        distributions = parse_checking_outputs(payload, samples)
        result = calibrate_label_mapping(
            distributions,
            oracle,
            SplitName.VALIDATION,
            oracle_evaluator=args.oracle_evaluator,
            allow_default_for_unseen=args.allow_default_for_unseen,
        )
    except (
        RAGCheckerOutputError,
        LabelCalibrationError,
        DatasetManifestError,
        ValueError,
        OSError,
    ) as error:
        print(f"[校准失败] {error}", file=sys.stderr)
        return 1

    _atomic_write_json(args.out, json.loads(result.model_dump_json()))

    print("标签映射校准完成（验证集）。")
    print(f"  参与校准的 (claim, document) 组合：{result.pair_count}")
    print()
    print(f"  {'标签':<16}{'权重质量':>10}{'多数票组合':>12}   校准后 (p_S, p_R, p_U)")
    for stat in result.stats:
        flag = "  ← 占位默认值" if stat.used_default else ""
        p_s, p_r, p_u = stat.probabilities
        print(
            f"  {stat.label.value:<16}{stat.weight_mass:>10.1f}{stat.majority_pairs:>12}"
            f"   ({p_s:.4f}, {p_r:.4f}, {p_u:.4f}){flag}"
        )
    print()
    print("  列联表（RAGChecker 多数标签 × 人工投票 argmax）：")
    header = f"    {'':<16}{'support':>9}{'refute':>9}{'unknown':>9}{'tie':>9}"
    print(header)
    for label in RAGCheckerLabel:
        row = result.contingency.get(label.value, {})
        print(
            f"    {label.value:<16}{row.get('support', 0):>9}{row.get('refute', 0):>9}"
            f"{row.get('unknown', 0):>9}{row.get('tie', 0):>9}"
        )
    print()
    print(f"  输出 {args.out}")
    print()
    print("  论文里必须写明：映射表由验证集的人工证据投票标定；")
    print("  测试集全程不接触 oracle，其关系概率完全由 RAGChecker 标签换算而来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
