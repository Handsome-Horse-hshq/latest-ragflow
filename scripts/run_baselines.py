"""三个对照 baseline 的批量运行入口。

用法::

    python scripts/run_baselines.py --config configs/baselines_demo.yaml
    python scripts/run_baselines.py --config configs/baselines_demo.yaml --overwrite

不联网，不调用任何大模型：关系概率全部来自预设的 mock 数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_ds.baselines.runner import BaselineRunSummary, run_baselines_from_config
from rag_ds.integrity import PipelineError

#: 默认配置路径按脚本位置推算，与终端当前目录无关。
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baselines_demo.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行三个对照 baseline（使用预设 mock 关系概率）"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML 配置文件路径（默认：{DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件",
    )
    return parser.parse_args(argv)


def _print_summary(summary: BaselineRunSummary) -> None:
    """打印简短中文摘要。"""
    print("Baseline 运行完成。")
    print(f"  样本数量：{summary.sample_count}")
    print(f"  claim 数量：{summary.claim_count}")
    print(f"  baseline 方法数量：{summary.method_count}")
    print(f"  输出记录数量：{summary.record_count}")
    print(f"  single_evaluator：{summary.single_evaluator}")
    print()
    print("  各方法预测类别数量：")
    for method, counts in summary.state_counts_by_method.items():
        parts = "  ".join(
            f"{state}={counts.get(state, 0)}"
            for state in ("supported", "refuted", "insufficient")
        )
        print(f"    {method:<18}{parts}")
    print()
    print("  判定原因分布：")
    for method, counts in summary.reason_counts_by_method.items():
        parts = "  ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
        print(f"    {method:<18}{parts}")
    print()
    print("  注意：三个 baseline 都不会输出 conflicting —— 这是被比较对象的固有局限。")
    print()
    print("  输出文件：")
    print(f"    JSONL  {summary.output_jsonl}")
    print(f"    CSV    {summary.output_csv}")


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    try:
        summary = run_baselines_from_config(
            args.config, overwrite=True if args.overwrite else None
        )
    except PipelineError as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1
    except FileExistsError as error:
        print(f"[输出文件已存在] {error}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
