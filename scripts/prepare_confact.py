"""从 CONFACT（HumC）构建「每篇网页一段最相关段落 + 来源可信度」的样本。

原始文件需要先手动下载到 ``data/raw/confact/``（上游仓库没有声明许可证，
只在本地做研究使用，不随本仓库分发），版本固定为
:data:`rag_ds.datasets.confact.CONFACT_COMMIT`::

    <主页>/raw/<commit>/data/dataset/HumC.pkl.gz
    <主页>/raw/<commit>/data/dataset/mbfc_media_data.pkl

原始数据是 pickle：本脚本用拒绝导入任何类的方式读取，见
:func:`rag_ds.datasets.confact.load_untrusted_pickle`。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from rag_ds.datasets.confact import (
    CONFACT_COMMIT,
    CONFACT_HOMEPAGE,
    ConfactDataError,
    build_confact_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CONFACT 可靠性实验数据")
    parser.add_argument(
        "--source-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "confact"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "confact_humc_v1",
    )
    parser.add_argument("--window", type=int, default=120, help="段落词数")
    parser.add_argument("--stride", type=int, default=60, help="相邻段落起点间隔词数")
    parser.add_argument("--max-words", type=int, default=30000, help="每篇网页只看前多少词")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        meta = build_confact_dataset(
            args.source_dir,
            args.output_dir,
            window=args.window,
            stride=args.stride,
            max_words=args.max_words,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as error:
        print(f"[缺少原始文件] {error}")
        print(f"  来源：{CONFACT_HOMEPAGE}（commit {CONFACT_COMMIT}）下的 data/dataset/")
        return 1
    except (ConfactDataError, pickle.UnpicklingError, FileExistsError) as error:
        print(f"[无法构建] {error}")
        return 1
    print(f"数据集构建完成：{meta['dataset_name']}")
    print(f"  claim {meta['claims']} 条，标签分布 {meta['label_counts']}")
    print(f"  文档 {meta['documents']} 篇（空文档跳过 {meta['documents_skipped_empty']} 篇，"
          f"超长截断 {meta['documents_truncated']} 篇）")
    print(f"  来源可信度 {meta['document_credibility_counts']}")
    print(f"  输出目录：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
