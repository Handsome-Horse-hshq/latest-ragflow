"""把 climate_fever_v1 的某个 split 导出成 RAGChecker 的输入 JSON。

用法::

    python scripts/export_ragchecker_input.py --split validation \
        --out outputs/ragchecker/validation_checking_inputs.json

输出格式取自 RAGChecker 仓库的 ``examples/checking_inputs.json``::

    {"results": [{"query_id", "query", "gt_answer", "response",
                  "retrieved_context": [{"doc_id", "text"}]}]}

关于 gt_answer
--------------
CLIMATE-FEVER 没有参考答案，本项目的 ``reference_answer`` 也是 ``None``。
``gt_answer`` 是 RAGChecker 输入结构的必填字段，但我们只需要 ``faithfulness``
这一个 metric，而它**只依赖 retrieved2response**（见 ragchecker/metrics.py），
完全不碰 gt_answer。

因此这里写入一句显式的占位说明，而不是把 claim 文本复制过去：一旦有人误跑
``--metrics all_metrics``，得到的 precision / recall 会是明显的垃圾值，而不是
因为「gt_answer == response」而虚高到 1.0 的假好成绩。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from rag_ds.data_io import load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "climate_fever_v1" / "manifest.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "ragchecker"

#: gt_answer 的占位文本，刻意写得一眼能看出不是参考答案。
PLACEHOLDER_GT_ANSWER = (
    "[PLACEHOLDER] CLIMATE-FEVER provides no reference answer. "
    "This field is unused by the faithfulness metric; "
    "do not run metrics that depend on gt_answer."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="导出 RAGChecker 输入 JSON（不调用任何模型）"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="数据集清单路径"
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
        help="要导出的划分",
    )
    parser.add_argument("--out", type=Path, default=None, help="输出 JSON 路径")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只导出前 N 条样本，用于先花很少的钱验证链路是否跑得通",
    )
    parser.add_argument(
        "--gt-answer",
        default=PLACEHOLDER_GT_ANSWER,
        help="写入 gt_answer 的文本；默认是显式占位说明",
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
    out_path: Path = args.out or (
        DEFAULT_OUT_DIR / f"{args.split}_checking_inputs.json"
    )
    if out_path.exists() and not args.overwrite:
        print(f"[输出文件已存在] {out_path}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    manifest_path = args.manifest.resolve()
    try:
        manifest = load_dataset_manifest(manifest_path)
        samples_path = verify_artifact(
            manifest_path.parent, manifest.splits[args.split].samples
        )
        samples = load_samples(samples_path)
    except (DatasetManifestError, ValueError, OSError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    if args.limit is not None:
        if args.limit < 1:
            print("[参数错误] --limit 必须是正整数", file=sys.stderr)
            return 1
        samples = samples[: args.limit]

    results = [
        {
            "query_id": sample.sample_id,
            "query": sample.question,
            "gt_answer": args.gt_answer,
            "response": sample.answer,
            "retrieved_context": [
                {"doc_id": chunk.doc_id, "text": chunk.text}
                for chunk in sample.contexts
            ],
        }
        for sample in samples
    ]
    _atomic_write_json(out_path, {"results": results})

    pairs = sum(len(item["retrieved_context"]) for item in results)
    print("RAGChecker 输入已导出。")
    print(f"  数据集   {manifest.dataset_name}  split={args.split}")
    print(f"  样本     {len(results)} 条")
    print(f"  判断网格 {pairs} 个 (claim, document) 组合")
    print(f"  输出     {out_path}")
    if args.limit is not None:
        print(f"  注意：这是 --limit {args.limit} 的子集，只能用于冒烟测试，")
        print("        正式实验必须导出完整 split。")
    print()
    print("  下一步（需要你自己的 LLM 凭证，本项目不代持也不代调）：")
    print("    ragchecker-cli \\")
    print(f"      --input_path={out_path} \\")
    print(f"      --output_path=<同目录>/{args.split}_checking_outputs.json \\")
    print("      --extractor_name=<模型> --checker_name=<模型> \\")
    print("      --metrics faithfulness")
    print("  只跑 faithfulness 即可拿到完整的 retrieved2response 网格。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
