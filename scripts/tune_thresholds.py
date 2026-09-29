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
from rag_ds.model_runs import ModelRunError, verify_model_run_artifacts
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
    parser.add_argument(
        "--model-run",
        type=Path,
        default=None,
        help=(
            "验证集关系来自模型（如 RAGChecker）时，传它的谱系清单；"
            "不传则要求关系必须是数据集清单登记的人工 oracle"
        ),
    )
    parser.add_argument(
        "--grid",
        choices=("fixed", "observed"),
        default="fixed",
        help=(
            "fixed=沿用固定网格（默认，保持既有结果可复现）；"
            "observed=按验证集实际观测到的 m_theta / K_doc 分位数构造网格。"
            "换了关系输入（例如从 oracle 换成 RAGChecker）后建议用 observed，"
            "否则整条 theta 轴可能全部落在观测范围之外、门控一次都不触发"
        ),
    )
    parser.add_argument(
        "--grid-steps", type=int, default=5, help="observed 网格每个轴的候选点数"
    )
    parser.add_argument("--top", type=int, default=10, help="打印前 N 个网格点")
    parser.add_argument(
        "--evaluator-alert-threshold",
        type=float,
        default=0.4,
        help="固定的 K_eval 告警阈值（不参与四分类 Macro-F1 调参）",
    )
    return parser.parse_args(argv)


def _report_grid_health(
    grid: ThresholdGrid,
    search,
    observed_theta: list[float],
    observed_k_doc: list[float],
) -> None:
    """检查网格是否真的覆盖了数据，把退化情况明确打出来。

    两种退化都会让搜索**看上去正常、实际无效**：候选阈值全部落在观测范围之外
    时，那条坐标轴一次都不会改变判定，所有网格点的 Macro-F1 完全相同；最优值
    落在网格边界时，真正的最优很可能在网格之外。
    """
    problems: list[str] = []
    for name, values, observed in (
        ("theta", grid.theta_values, observed_theta),
        ("K_doc", grid.document_conflict_values, observed_k_doc),
    ):
        if not observed:
            continue
        low, high = min(observed), max(observed)
        print(
            f"    {name:<6}观测范围 [{low:.4f}, {high:.4f}]，"
            f"候选 {tuple(round(v, 4) for v in values)}"
        )
        if all(value > high for value in values):
            problems.append(
                f"{name} 的候选阈值全部高于观测最大值 {high:.4f}，"
                f"该门控一次都不会触发"
            )
        elif all(value <= low for value in values):
            problems.append(
                f"{name} 的候选阈值全部不高于观测最小值 {low:.4f}，"
                f"该门控会始终触发"
            )

    best = search.best.thresholds
    for name, chosen, values in (
        ("theta", best.theta_threshold, grid.theta_values),
        ("K_doc", best.document_conflict_threshold, grid.document_conflict_values),
    ):
        if len(values) > 1 and chosen in (min(values), max(values)):
            problems.append(
                f"{name} 的最优值 {chosen} 落在网格边界，真正的最优可能在网格之外"
            )

    distinct = {round(c.macro_f1, 12) for c in search.candidates}
    if len(distinct) == 1 and len(search.candidates) > 1:
        problems.append(
            "全部网格点的 Macro-F1 完全相同：这组阈值对结果没有任何影响"
        )

    if problems:
        print()
        print("  ⚠ 网格健康检查：")
        for item in problems:
            print(f"    - {item}")
        print("    建议改用 --grid observed 让网格跟着数据走。")


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
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
        # 搜索只重跑门控，前面的 D-S 计算跑一次即可。
        results = run_pipeline(samples, predictions, DiagnosticThresholds())
        observed_theta = [
            r.diagnostic.m_theta for r in results if r.diagnostic.m_theta is not None
        ]
        observed_k_doc = [r.diagnostic.k_doc for r in results]
        if args.grid == "observed":
            grid = ThresholdGrid.from_observed(
                observed_theta,
                observed_k_doc,
                steps=args.grid_steps,
                evaluator_conflict_threshold=args.evaluator_alert_threshold,
            )
        else:
            grid = ThresholdGrid(
                evaluator_conflict_threshold=args.evaluator_alert_threshold
            )
        search = search_thresholds(results, SplitName.VALIDATION, grid)
    except (PipelineError, DatasetManifestError, ModelRunError, ValueError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    best = search.best.thresholds
    print("阈值搜索完成（验证集）。")
    if run is None:
        print("  关系输入：人工标注 oracle")
    else:
        print(f"  关系输入：模型预测 evaluator={run.evaluator}, checker={run.checker_name}")
        if not run.is_calibrated:
            print("  警告：标签映射未经校准，搜出的阈值同样不能写进论文。")
    print(f"  claim 数量：{search.claim_count}")
    print(f"  网格点数：{len(search.candidates)}（{args.grid} 网格）")
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
    print(f"    {'theta':>10}{'k_doc':>10}{'macroF1':>10}{'acc':>8}")
    for candidate in search.candidates[: args.top]:
        t = candidate.thresholds
        print(
            f"    {t.theta_threshold:>10.4f}{t.document_conflict_threshold:>10.4f}"
            f"{candidate.macro_f1:>10.4f}{candidate.accuracy:>8.4f}"
        )

    _report_grid_health(grid, search, observed_theta, observed_k_doc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(search.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"  完整结果已写入：{args.out}")
    print()
    if run is None:
        print("  下一步：把最优阈值手工填回 configs/experiment.yaml 并锁定，再跑测试集。")
    else:
        print("  下一步：把最优阈值手工填回 configs/climate_fever_refchecker_nli_test.yaml")
        print("        并锁定，再跑测试集。阈值必须与关系输入配套 ——")
        print("        在 oracle 上搜出的阈值不能直接用在 RAGChecker 的关系上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
