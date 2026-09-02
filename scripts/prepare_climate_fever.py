"""从官方 CLIMATE-FEVER 原始 JSONL 构建可复现的正式数据划分。"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_ds.datasets import build_climate_fever_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CLIMATE-FEVER 正式数据集")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "climate-fever.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "climate_fever_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=60)
    parser.add_argument("--validation-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_climate_fever_dataset(
        args.source,
        args.output_dir,
        seed=args.seed,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        overwrite=args.overwrite,
    )
    print(f"数据集构建完成：{manifest.dataset_name}")
    print(f"  每类总数：{manifest.per_class}")
    for split in ("train", "validation", "test"):
        item = manifest.splits[split]
        counts = ", ".join(
            f"{label.value}={count}" for label, count in item.label_counts.items()
        )
        print(
            f"  {split:<10} samples={item.samples.records:<4} "
            f"relations={item.relation_predictions.records:<5} {counts}"
        )
    print(f"  manifest：{args.output_dir / 'manifest.json'}")
    print("  注意：*_relations.jsonl 是人工标注 oracle，不是模型预测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
