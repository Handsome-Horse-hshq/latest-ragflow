"""离线 MVP demo 入口。

用法::

    python scripts/run_demo.py --config configs/demo.yaml
    python scripts/run_demo.py --config configs/demo.yaml --overwrite

不联网，不调用任何大模型：关系概率全部来自预设的 mock 数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_ds.pipeline import PipelineError, run_pipeline_from_config
from rag_ds.pipeline_results import PipelineRunSummary

#: 默认配置路径按脚本位置推算，与终端当前目录无关。
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行离线 MVP 诊断 pipeline（使用预设 mock 关系概率）"
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


def _print_summary(summary: PipelineRunSummary) -> None:
    """打印简短中文摘要。"""
    print("离线 MVP 运行完成。")
    print(f"  样本数量：{summary.sample_count}")
    print(f"  claim 数量：{summary.claim_count}")
    print()
    print("  处理状态：")
    print(f"    正常完成          {summary.normal_count}")
    print(f"    无检索文档        {summary.no_contexts_count}")
    print(f"    文档完全冲突      {summary.document_total_conflict_count}")
    print(f"    评估器完全冲突    {summary.evaluator_total_conflict_count}")
    print()
    print("  primary_state 分布：")
    for state in ("supported", "refuted", "insufficient", "conflicting", "none"):
        count = summary.primary_state_counts.get(state, 0)
        if count or state != "none":
            print(f"    {state:<16}{count}")
    print()
    print("  输出文件：")
    print(f"    JSONL  {summary.output_jsonl}")
    print(f"    CSV    {summary.output_csv}")


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    try:
        summary = run_pipeline_from_config(
            args.config, overwrite=True if args.overwrite else None
        )
    except PipelineError as error:
        # 输入数据问题：给简洁提示，不打印堆栈。
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
