"""在验证集上搜索二维门控阈值。

用法::

    python scripts/tune_thresholds.py --manifest data/processed/dataset/manifest.json \
        --samples data/processed/dataset/validation.jsonl \
        --predictions data/processed/dataset/validation_relations.jsonl

**只接受验证集。** 用测试集选阈值再用测试集报告结果没有意义，
:func:`rag_ds.tuning.search_thresholds` 会直接拒绝。

搜索结束后请把最优阈值**手工填回** configs/experiment.yaml 并锁定，
再跑测试集 —— 刻意不自动改写配置，避免「什么时候用了哪组阈值」变成一笔糊涂账。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import DatasetManifestError, verify_validation_artifacts
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.integrity import PipelineError
from rag_ds.pipeline import run_pipeline
from rag_ds.tuning import SplitName, ThresholdGrid, search_thresholds

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在验证集上网格搜索门控阈值")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="登记 train/validation/test 及摘要的数据清单",
    )
    parser.add_argument("--samples", type=Path, required=True, help="验证集样本 JSONL")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="验证集关系预测 JSONL"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "threshold_search.json",
        help="搜索结果 JSON 的输出路径",
    )
    parser.add_argument("--top", type=int, default=10, help="打印前 N 个网格点")
    parser.add_argument(
        "--evaluator-alert-threshold",
        type=float,
        default=0.4,
        help="固定的 K_eval 告警阈值（不参与四分类 Macro-F1 调参）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    try:
        verify_validation_artifacts(args.manifest, args.samples, args.predictions)
        samples = load_samples(args.samples)
        predictions = load_relation_predictions(args.predictions)
        # 搜索只重跑门控，前面的 D-S 计算跑一次即可。
        results = run_pipeline(samples, predictions, DiagnosticThresholds())
        grid = ThresholdGrid(
            evaluator_conflict_threshold=args.evaluator_alert_threshold
        )
        search = search_thresholds(results, SplitName.VALIDATION, grid)
    except (PipelineError, DatasetManifestError, ValueError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    best = search.best.thresholds
    print("阈值搜索完成（验证集）。")
    print(f"  claim 数量：{search.claim_count}")
    print(f"  网格点数：{len(search.candidates)}")
    print()
    print("  最优阈值：")
    print(f"    theta_threshold:              {best.theta_threshold}")
    print(f"    document_conflict_threshold:  {best.document_conflict_threshold}")
    print(
        "    evaluator_conflict_threshold: "
        f"{best.evaluator_conflict_threshold}（固定告警阈值，未参与搜索）"
    )
    print(f"    -> Macro-F1 = {search.best.macro_f1:.4f}, "
          f"Accuracy = {search.best.accuracy:.4f}")
    print()
    print(f"  前 {args.top} 个网格点：")
    print(f"    {'theta':>7}{'k_doc':>8}{'macroF1':>10}{'acc':>8}")
    for candidate in search.candidates[: args.top]:
        t = candidate.thresholds
        print(
            f"    {t.theta_threshold:>7}{t.document_conflict_threshold:>8}"
            f"{candidate.macro_f1:>10.4f}{candidate.accuracy:>8.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(search.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"  完整结果已写入：{args.out}")
    print()
    print("  下一步：把最优阈值手工填回 configs/experiment.yaml 并锁定，再跑测试集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
