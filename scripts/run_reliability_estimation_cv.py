"""真实可靠性估计实验：用数据估出来的可靠性替换恒等于 1 的占位值。

动机
----
到目前为止所有实验里 ``evaluator_reliability`` 都恒为 1，``no_reliability``
消融因而是恒等变换，可靠性折扣机制从未被真实数据检验过。本实验在每个外折的
**内层各折**上估计每个评估器的可靠性，再把估计值写回关系预测，重跑同一套
嵌套交叉验证协议。

估计方法（刻意简单、无泄漏）
---------------------------
对内层数据里的每个评估器：

1. 用 ``single_evaluator`` 同款聚合（文档可靠性加权的支持/反驳/未知平均），
   在 ``DEFAULT_DECISION_THRESHOLDS`` 上扫判定阈值，取内层 Macro-F1 最大者；
2. 把最优 Macro-F1 折算成可靠性::

       r = clip((F1 - F1_chance) / (1 - F1_chance), 0, 1),  F1_chance = 0.25

   四类均衡数据上随机猜测的 Macro-F1 约为 0.25，因此 r 是「超出随机的技能
   占比」——弱评估器（F1 接近 0.25）可靠性接近 0，完美评估器为 1。

估计只用内层折（带金标准），测试折的标签不参与；同一外折里阈值搜索、baseline
搜索与测试预测全部使用同一份改写后的可靠性，协议与 ``run_cross_validation``
完全一致，输出同形，可直接对比。

用法::

    python scripts/run_reliability_estimation_cv.py \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" \
        --evaluator refchecker_nli refchecker_nli_deberta \
        --single-evaluator refchecker_nli \
        --out-dir outputs/metrics/cross_validation_2eval_est_rel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rag_ds.baselines.models import BaselineThresholds
from rag_ds.baselines.single_evaluator import predict_single_evaluator
from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import load_dataset_manifest, verify_artifact
from rag_ds.experiments.cross_validation import (
    CrossValidationResult,
    FoldThresholds,
    make_stratified_folds,
    _shared_decision_threshold,
    _subset,
)
from rag_ds.experiments.comparison import DS_METHOD, MethodPrediction
from rag_ds.experiments.export import write_predictions_csv
from rag_ds.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    GOLD_LABELS,
    bootstrap_interval,
    classification_report,
    paired_bootstrap,
)
from rag_ds.pipeline import run_pipeline
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.schemas import RAGSample, RelationPrediction
from rag_ds.tuning.baseline_threshold_search import (
    DEFAULT_DECISION_THRESHOLDS,
    search_baseline_thresholds,
)
from rag_ds.tuning.threshold_search import (
    SplitName,
    ThresholdGrid,
    predicted_label,
    search_thresholds,
)
from rag_ds.baselines.runner import run_baselines

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
CHANCE_MACRO_F1 = 0.25


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可靠性估计 + 嵌套交叉验证")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relations-template", required=True)
    parser.add_argument("--evaluator", nargs="+", required=True)
    parser.add_argument("--single-evaluator", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-steps", type=int, default=9)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--estimator",
        choices=("macro_f1", "auroc"),
        default="auroc",
        help="macro_f1：单评估器四分类 Macro-F1 折算；auroc：按职能的判别 AUROC 折算",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _auroc(scores: list[float], labels: list[int]) -> float:
    """秩次法 AUROC（带并列修正）；只有一类时返回 NaN。"""
    import numpy as np
    import pandas as pd

    y = np.asarray(labels, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def _estimate_macro_f1(
    samples: list[RAGSample],
    predictions: list[RelationPrediction],
    evaluator: str,
) -> float:
    """单评估器四分类最优 Macro-F1 折算：r = (F1 - 0.25) / 0.75。

    注意：该口径对**结构上无法输出 conflicting** 的评估器系统性偏严——
    它们在四类金标准上的天花板本来就低于 1。实测两个 NLI 评估器在此口径下
    可靠性均约为 0（四分类 F1 贴着随机水平 0.25），会把所有证据折扣成无知。
    保留此估计器是为了在论文里如实对比两种口径。
    """
    best_f1 = 0.0
    for threshold in DEFAULT_DECISION_THRESHOLDS:
        gold, pred = [], []
        for sample in samples:
            for claim in sample.claims:
                try:
                    result = predict_single_evaluator(
                        sample,
                        claim,
                        predictions,
                        evaluator,
                        BaselineThresholds(decision_threshold=threshold),
                    )
                except Exception:
                    continue
                if sample.gold_state is None:
                    continue
                gold.append(sample.gold_state.value)
                pred.append(result.predicted_state.value)
        if not gold:
            continue
        f1 = classification_report(gold, pred, method=evaluator).macro_f1
        best_f1 = max(best_f1, f1)
    reliability = (best_f1 - CHANCE_MACRO_F1) / (1.0 - CHANCE_MACRO_F1)
    return min(1.0, max(0.0, reliability))


def _estimate_auroc(
    samples: list[RAGSample],
    predictions: list[RelationPrediction],
    evaluator: str,
) -> float:
    """按评估器职能的判别 AUROC 折算。

    评估器在链路里的实际职责是提供逐文档的支持/反驳/未知证据，因此用两个
    可观测的判别任务度量它的技能：

    - A1：claim 级（平均 p_support - 平均 p_refute）区分 supported 与
      refuted 金标准；
    - A2：claim 级平均 p_unknown 区分 insufficient 与其余三类。

    r = clip(2 * mean(A1, A2) - 1, 0, 1)，即「超出随机判别的技能占比」。
    """
    gold = {s.sample_id: s.gold_state.value for s in samples if s.gold_state is not None}
    agg: dict[str, list[float]] = {}
    for p in predictions:
        if p.evaluator != evaluator or p.sample_id not in gold:
            continue
        slot = agg.setdefault(p.sample_id, [0.0, 0.0, 0.0, 0.0])
        slot[0] += p.p_support
        slot[1] += p.p_refute
        slot[2] += p.p_unknown
        slot[3] += 1.0
    if not agg:
        return 0.0
    decisive = [
        ((v[0] - v[1]) / v[3], 1 if gold[sid] == "supported" else 0)
        for sid, v in agg.items()
        if gold[sid] in ("supported", "refuted")
    ]
    insufficiency = [
        (v[2] / v[3], 1 if gold[sid] == "insufficient" else 0)
        for sid, v in agg.items()
    ]
    a1 = _auroc([d for d, _ in decisive], [y for _, y in decisive])
    a2 = _auroc([u for u, _ in insufficiency], [y for _, y in insufficiency])
    skills = [a for a in (a1, a2) if a == a]  # 去掉 NaN
    if not skills:
        return 0.0
    return min(1.0, max(0.0, 2.0 * (sum(skills) / len(skills)) - 1.0))


def estimate_reliabilities(
    samples: list[RAGSample],
    predictions: list[RelationPrediction],
    evaluators: list[str],
    estimator: str = "auroc",
) -> dict[str, float]:
    """在带金标准的数据上估计每个评估器的可靠性（见模块文档字符串）。"""
    fn = _estimate_auroc if estimator == "auroc" else _estimate_macro_f1
    return {ev: fn(samples, predictions, ev) for ev in evaluators}


def apply_reliabilities(
    predictions: list[RelationPrediction],
    estimates: dict[str, float],
) -> list[RelationPrediction]:
    """把估计出的可靠性写回每条关系预测（不改原对象）。"""
    return [
        p.model_copy(update={"evaluator_reliability": estimates[p.evaluator]})
        if p.evaluator in estimates
        else p
        for p in predictions
    ]


def run_cv_with_estimated_reliability(
    samples: list[RAGSample],
    predictions: list[RelationPrediction],
    evaluators: list[str],
    single_evaluator: str,
    *,
    n_folds: int,
    seed: int,
    grid_steps: int,
    evaluator_conflict_threshold: float = 0.4,
    estimator: str = "auroc",
) -> tuple[CrossValidationResult, list[dict]]:
    """与 run_nested_cross_validation 同协议，但可靠性逐折从内层估计。"""
    folds = make_stratified_folds(samples, n_folds, seed)
    pooled: list[MethodPrediction] = []
    records: list[FoldThresholds] = []
    reliability_log: list[dict] = []
    methods: list[str] = []

    for fold_index, test_indices in enumerate(folds):
        inner_indices = [
            i for position, fold in enumerate(folds) if position != fold_index
            for i in fold
        ]
        inner_samples, inner_predictions = _subset(samples, predictions, inner_indices)
        test_samples, test_predictions = _subset(samples, predictions, test_indices)

        # --- 可靠性估计：只用内层数据 ---
        estimates = estimate_reliabilities(
            inner_samples, inner_predictions, evaluators, estimator
        )
        reliability_log.append({"fold": fold_index, **estimates})
        inner_predictions = apply_reliabilities(inner_predictions, estimates)
        test_predictions = apply_reliabilities(test_predictions, estimates)

        # --- 以下与原嵌套 CV 完全一致 ---
        base = DiagnosticThresholds(
            evaluator_conflict_threshold=evaluator_conflict_threshold
        )
        inner_results = run_pipeline(inner_samples, inner_predictions, base)
        observed_theta = [
            r.diagnostic.m_theta for r in inner_results if r.diagnostic.m_theta is not None
        ]
        observed_k_doc = [r.diagnostic.k_doc for r in inner_results]
        grid = ThresholdGrid.from_observed(
            observed_theta, observed_k_doc,
            steps=grid_steps,
            evaluator_conflict_threshold=evaluator_conflict_threshold,
        )
        search = search_thresholds(inner_results, SplitName.VALIDATION, grid)
        ds_thresholds = search.best.thresholds

        baseline_search = search_baseline_thresholds(
            inner_samples, inner_predictions, SplitName.VALIDATION, single_evaluator,
        )
        decision_threshold = _shared_decision_threshold(baseline_search)
        ca_best = baseline_search.best.get("conflict_aware")
        conflict_threshold = ca_best.conflict_threshold if ca_best is not None else None

        test_results = run_pipeline(test_samples, test_predictions, ds_thresholds)
        gold = {
            (r.sample_id, r.claim_id): r.gold_state.value
            for r in test_results
        }
        for result in test_results:
            pooled.append(
                MethodPrediction(
                    method=DS_METHOD,
                    sample_id=result.sample_id,
                    claim_id=result.claim_id,
                    predicted_label=predicted_label(result.diagnostic),
                    gold_label=gold[(result.sample_id, result.claim_id)],
                    insufficiency_score=result.diagnostic.m_theta or 0.0,
                    conflict_score=result.diagnostic.k_doc,
                )
            )
        baseline_thresholds = BaselineThresholds(
            decision_threshold=decision_threshold,
            **({"conflict_threshold": conflict_threshold} if conflict_threshold is not None else {}),
        )
        for item in run_baselines(test_samples, test_predictions, baseline_thresholds, single_evaluator):
            pooled.append(
                MethodPrediction(
                    method=item.method.value,
                    sample_id=item.sample_id,
                    claim_id=item.claim_id,
                    predicted_label=item.predicted_state.value,
                    gold_label=gold[(item.sample_id, item.claim_id)],
                    insufficiency_score=item.score_unknown,
                    conflict_score=1.0 - abs(item.score_support - item.score_refute),
                )
            )
        records.append(
            FoldThresholds(
                fold=fold_index,
                test_size=len(test_samples),
                inner_size=len(inner_samples),
                ds_thresholds=ds_thresholds,
                baseline_decision_threshold=decision_threshold,
                baseline_conflict_threshold=conflict_threshold,
                inner_macro_f1=search.best.macro_f1,
            )
        )
        if not methods:
            methods = [DS_METHOD] + sorted({p.method for p in pooled if p.method != DS_METHOD})

    claim_count = len({(p.sample_id, p.claim_id) for p in pooled})
    return (
        CrossValidationResult(
            n_folds=n_folds,
            seed=seed,
            predictions=tuple(pooled),
            folds=tuple(records),
            claim_count=claim_count,
            methods=tuple(methods),
        ),
        reliability_log,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    summary_path = out_dir / "cv_summary.json"
    predictions_path = out_dir / "cv_predictions.csv"
    if not args.overwrite and (summary_path.exists() or predictions_path.exists()):
        print("[输出文件已存在] 加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    manifest_path = args.manifest.resolve()
    manifest = load_dataset_manifest(manifest_path)
    samples, predictions = [], []
    for split in SPLITS:
        samples_path = verify_artifact(manifest_path.parent, manifest.splits[split].samples)
        samples.extend(load_samples(samples_path))
        for evaluator in args.evaluator:
            predictions.extend(
                load_relation_predictions(
                    Path(args.relations_template.format(split=split, evaluator=evaluator))
                )
            )

    result, reliability_log = run_cv_with_estimated_reliability(
        samples, predictions, args.evaluator, args.single_evaluator,
        n_folds=args.n_folds, seed=args.seed, grid_steps=args.grid_steps,
        estimator=args.estimator,
    )

    by_method: dict[str, list] = defaultdict(list)
    for record in result.predictions:
        by_method[record.method].append(record)
    ordered = {m: sorted(v, key=lambda r: (r.sample_id, r.claim_id)) for m, v in by_method.items()}
    truth = [r.gold_label for r in ordered[result.methods[0]]]
    reports = {
        m: classification_report([r.gold_label for r in v], [r.predicted_label for r in v], method=m)
        for m, v in ordered.items()
    }
    intervals = {
        m: bootstrap_interval(truth, [r.predicted_label for r in v], m, labels=GOLD_LABELS,
                              n_resamples=args.resamples, seed=args.bootstrap_seed)
        for m, v in ordered.items()
    }
    reference = result.methods[0]
    comparisons = [
        paired_bootstrap(truth, [r.predicted_label for r in ordered[reference]],
                         [r.predicted_label for r in v], reference, m,
                         labels=GOLD_LABELS, n_resamples=args.resamples, seed=args.bootstrap_seed)
        for m, v in ordered.items() if m != reference
    ]

    write_predictions_csv(predictions_path, list(result.predictions))
    _atomic_write_json(
        summary_path,
        {
            "dataset_name": manifest.dataset_name,
            "evaluators": list(args.evaluator),
            "reliability_estimation": {
                "estimator": args.estimator,
                "method": (
                    "inner-fold single-evaluator 4-class Macro-F1, (F1-0.25)/0.75 clipped"
                    if args.estimator == "macro_f1"
                    else "inner-fold per-function AUROC (sup-vs-ref, insuff-vs-rest), "
                         "clip(2*mean(AUROC)-1, 0, 1)"
                ),
                "per_fold": reliability_log,
            },
            "n_folds": result.n_folds,
            "fold_seed": result.seed,
            "claim_count": result.claim_count,
            "folds": [json.loads(f.model_dump_json()) for f in result.folds],
            "reports": {m: json.loads(r.model_dump_json()) for m, r in reports.items()},
            "intervals": {m: json.loads(i.model_dump_json()) for m, i in intervals.items()},
            "comparisons": [json.loads(c.model_dump_json()) for c in comparisons],
        },
    )

    print("可靠性估计（逐折）：")
    for entry in reliability_log:
        fold = entry.pop("fold")
        print(f"  折 {fold}: " + ", ".join(f"{k}={v:.3f}" for k, v in entry.items()))
    print()
    print(f"  {'方法':<20}{'Accuracy':>10}{'Macro-F1':>10}{'CI 下界':>10}{'CI 上界':>10}")
    for method in result.methods:
        report, interval = reports[method], intervals[method]
        print(f"  {method:<20}{report.accuracy:>10.4f}{report.macro_f1:>10.4f}"
              f"{interval.ci_low:>10.4f}{interval.ci_high:>10.4f}")
    print()
    for item in comparisons:
        verdict = "显著" if item.is_significant else "不显著"
        print(f"  vs {item.method_b:<18}{item.observed_delta:+.4f}  p={item.p_value:.3f}  {verdict}")
    print(f"\n  输出 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
