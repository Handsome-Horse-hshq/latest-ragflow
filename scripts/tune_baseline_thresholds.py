"""在验证集上为三个 baseline 搜索判定阈值。

用法::

    python scripts/tune_baseline_thresholds.py \
        --predictions outputs/ragchecker/validation_relations_refchecker_nli.jsonl \
        --model-run outputs/ragchecker/validation_model_run.json \
        --single-evaluator refchecker_nli

为什么必须跑这一步
------------------
D-S 的门控阈值是在验证集上搜出来的。如果 baseline 还用着
``decision_threshold = 0.5`` 这个调试默认值，两边就不是在同一条件下比较 ——
**那不是 baseline 弱，是 baseline 没调参**，这种对比写进论文会被直接质疑。

输出里会同时给出每个方法的最优阈值、最优值所在的**平台区间**，以及三个方法
平台的**交集**：交集非空时，用交集里的一个值就能让三个 baseline 同时处于各自
的验证集最优，配置里只需填一个数。

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
from rag_ds.integrity import PipelineError
from rag_ds.model_runs import ModelRunError, verify_model_run_artifacts
from rag_ds.tuning import SplitName, search_baseline_thresholds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "climate_fever_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在验证集上搜索 baseline 判定阈值")
    parser.add_argument("--manifest", type=Path, default=DATASET_DIR / "manifest.json")
    parser.add_argument("--samples", type=Path, default=DATASET_DIR / "validation.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--model-run",
        type=Path,
        default=None,
        help="关系来自模型时传它的谱系清单；不传则要求关系是数据集登记的人工 oracle",
    )
    parser.add_argument("--single-evaluator", required=True)
    parser.add_argument(
        "--step", type=float, default=0.02, help="候选阈值步长，扫完 (0, 0.9]"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "baseline_threshold_search.json",
    )
    parser.add_argument("--overwrite", action="store_true")
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
    if args.step <= 0 or args.step > 0.5:
        print("[参数错误] --step 必须位于 (0, 0.5]", file=sys.stderr)
        return 1

    try:
        if args.model_run is not None:
            run = verify_model_run_artifacts(
                args.model_run,
                args.manifest,
                args.samples,
                args.predictions,
                "validation",
            )
        else:
            run = None
            verify_validation_artifacts(args.manifest, args.samples, args.predictions)
        samples = load_samples(args.samples)
        predictions = load_relation_predictions(args.predictions)
        grid = tuple(
            round(args.step * step, 6)
            for step in range(1, int(0.9 / args.step) + 1)
        )
        search = search_baseline_thresholds(
            samples,
            predictions,
            SplitName.VALIDATION,
            args.single_evaluator,
            decision_thresholds=grid,
        )
    except (PipelineError, DatasetManifestError, ModelRunError, ValueError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    print("baseline 阈值搜索完成（验证集）。")
    if run is None:
        print("  关系输入：人工标注 oracle")
    else:
        print(f"  关系输入：模型预测 evaluator={run.evaluator}")
        if not run.is_calibrated:
            print("  警告：标签映射未经校准，搜出的阈值同样不能写进论文。")
    print(f"  claim 数量：{search.claim_count}")
    print(f"  候选阈值：{len(grid)} 个，步长 {args.step}")
    print()

    plateaus: dict[str, tuple[float, float]] = {}
    print(f"  {'方法':<20}{'最优阈值':>10}{'macroF1':>10}{'acc':>8}   最优平台")
    for method, best in sorted(search.best.items()):
        tied = [
            c.decision_threshold
            for c in search.candidates
            if c.method.value == method and c.macro_f1 == best.macro_f1
        ]
        plateaus[method] = (min(tied), max(tied))
        flag = "  ← 退化：全判成同一类" if best.is_degenerate else ""
        print(
            f"  {method:<20}{best.decision_threshold:>10}{best.macro_f1:>10.4f}"
            f"{best.accuracy:>8.4f}   [{min(tied)}, {max(tied)}]{flag}"
        )

    low = max(bounds[0] for bounds in plateaus.values())
    high = min(bounds[1] for bounds in plateaus.values())
    print()
    shared: float | None = None
    if low <= high:
        shared = round((low + high) / 2, 6)
        print(f"  三个方法最优平台的交集：[{low}, {high}]")
        print(f"  => 配置里填 baseline.decision_threshold: {shared}")
        print("     这一个值能让三个 baseline 同时处于各自的验证集最优。")
    else:
        print("  三个方法的最优平台没有交集，单一阈值无法让三者同时最优。")
        print("  请在论文里说明各方法分别使用了自己的阈值。")

    _atomic_write_json(
        args.out,
        {
            **json.loads(search.model_dump_json()),
            "plateaus": {k: list(v) for k, v in plateaus.items()},
            "shared_threshold": shared,
        },
    )
    print()
    print(f"  完整结果已写入：{args.out}")
    print()
    print("  下一步：把阈值手工填回实验配置的 baseline.decision_threshold 并锁定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
