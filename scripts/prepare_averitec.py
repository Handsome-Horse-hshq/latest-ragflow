"""从官方 AVeriTeC（train.json + dev.json）构建可复现的四类均衡数据划分。

原始文件需要先手动下载到 ``data/raw/averitec/``（CC-BY-NC-4.0，仅限非商业
使用并注明出处），地址固定到一个仓库版本，见
:data:`rag_ds.datasets.averitec.AVERITEC_DOWNLOAD_URLS`::

    curl -L -o data/raw/averitec/train.json <train 的地址>
    curl -L -o data/raw/averitec/dev.json   <dev 的地址>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_ds.datasets import build_averitec_dataset
from rag_ds.datasets.averitec import AVERITEC_DOWNLOAD_URLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 AVeriTeC 正式数据集")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "averitec",
        help="放着官方 train.json 与 dev.json 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "averitec_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=138)
    parser.add_argument("--validation-per-class", type=int, default=46)
    parser.add_argument("--test-per-class", type=int, default=46)
    parser.add_argument(
        "--dataset-name",
        default="rag-ds-averitec-balanced-v1",
        help="写进 manifest 的数据集名；不同规模的子集必须取不同的名字",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_averitec_dataset(
            args.source_dir,
            args.output_dir,
            seed=args.seed,
            train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class,
            test_per_class=args.test_per_class,
            overwrite=args.overwrite,
            dataset_name=args.dataset_name,
        )
    except FileNotFoundError as error:
        print(f"[缺少原始文件] {error}")
        for split, url in AVERITEC_DOWNLOAD_URLS.items():
            print(f"  {split}.json  <-  {url}")
        return 1
    print(f"数据集构建完成：{manifest.dataset_name}")
    print(f"  每类总数：{manifest.per_class}")
    for split in ("train", "validation", "test"):
        item = manifest.splits[split]
        counts = ", ".join(
            f"{label.value}={count}" for label, count in item.label_counts.items()
        )
        print(f"  {split:<10} samples={item.samples.records:<4} {counts}")
    print(f"  manifest：{args.output_dir / 'manifest.json'}")
    print("  注意：AVeriTeC 没有逐条证据的立场标注，不存在 annotation-oracle 关系。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
