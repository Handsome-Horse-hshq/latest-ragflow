"""把实验结果导出为 CSV 与图表。

图表使用 matplotlib 的 ``Agg`` 后端，不弹窗、不依赖显示环境。
所有写盘都先落临时文件再原子替换，与项目其余部分一致。

图内文字一律使用**英文**：matplotlib 自带的 DejaVu Sans 没有中文字形，
用中文会渲染成方框，而依赖系统中文字体又会让图在别的机器上画不出来。
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前设置

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rag_ds.experiments.ablation import AblationResult  # noqa: E402
from rag_ds.experiments.comparison import (  # noqa: E402
    DS_METHOD,
    ExperimentReport,
    MethodPrediction,
)
from rag_ds.metrics.classification import ClassificationReport  # noqa: E402
from rag_ds.schemas import EvidenceState  # noqa: E402
from rag_ds.tuning.threshold_search import ThresholdSearchResult  # noqa: E402

__all__ = [
    "GOLD_COLORS",
    "plot_confusion_matrix",
    "plot_diagnostic_scatter",
    "plot_threshold_sensitivity",
    "write_ablation_csv",
    "write_main_results_csv",
    "write_predictions_csv",
]

#: 四类金标准在散点图中的配色（色觉友好，且黑白打印仍可区分）。
GOLD_COLORS: dict[str, str] = {
    EvidenceState.SUPPORTED.value: "#1b7837",
    EvidenceState.REFUTED.value: "#b2182b",
    EvidenceState.INSUFFICIENT.value: "#7f7f7f",
    EvidenceState.CONFLICTING.value: "#2166ac",
}


def _atomic_write_csv(
    path: str | Path, columns: Sequence[str], rows: Iterable[dict[str, object]]
) -> int:
    """先写临时文件再原子替换，返回写入的数据行数。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    written = 0
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                written += 1
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return written


def _classification_rows(reports: Sequence[ClassificationReport]):
    """把分类报告摊平成每方法每类一行。"""
    for report in reports:
        for metrics in report.per_class:
            yield {
                "method": report.method,
                "accuracy": round(report.accuracy, 6),
                "macro_f1": round(report.macro_f1, 6),
                "label": metrics.label,
                "precision": round(metrics.precision, 6),
                "recall": round(metrics.recall, 6),
                "f1": round(metrics.f1, 6),
                "support": metrics.support,
                "sample_count": report.sample_count,
                "macro_labels": "|".join(report.macro_labels),
            }


def write_main_results_csv(path: str | Path, report: ExperimentReport) -> int:
    """写出实验 1–3 的主结果。

    每行是「方法 × 类别」的分类指标，并附上该方法在实验 2、3 中的
    AUROC / AUPRC / 最佳 F1。
    """
    insufficiency = dict(zip(report.methods, report.insufficiency_detection))
    conflict = dict(zip(report.methods, report.conflict_detection))

    rows = []
    for row in _classification_rows(report.classification):
        method = str(row["method"])
        ins, con = insufficiency[method], conflict[method]
        row.update(
            {
                "insufficiency_score_name": ins.score_name,
                "insufficiency_auroc": "" if ins.auroc is None else round(ins.auroc, 6),
                "insufficiency_auprc": "" if ins.auprc is None else round(ins.auprc, 6),
                "insufficiency_best_f1": round(ins.best_f1, 6),
                "conflict_score_name": con.score_name,
                "conflict_auroc": "" if con.auroc is None else round(con.auroc, 6),
                "conflict_auprc": "" if con.auprc is None else round(con.auprc, 6),
                "conflict_best_f1": round(con.best_f1, 6),
            }
        )
        rows.append(row)

    columns = list(rows[0].keys()) if rows else ["method"]
    return _atomic_write_csv(path, columns, rows)


def write_ablation_csv(path: str | Path, results: Sequence[AblationResult]) -> int:
    """写出消融实验结果，每个变体一行。"""
    rows = [
        {
            "variant": result.variant.value,
            "accuracy": round(result.report.accuracy, 6),
            "macro_f1": round(result.report.macro_f1, 6),
            "macro_f1_delta": round(result.macro_f1_delta, 6),
            "theta_threshold": result.thresholds.theta_threshold,
            "document_conflict_threshold": result.thresholds.document_conflict_threshold,
            "evaluator_conflict_threshold": result.thresholds.evaluator_conflict_threshold,
            "sample_count": result.report.sample_count,
        }
        for result in results
    ]
    columns = list(rows[0].keys()) if rows else ["variant"]
    return _atomic_write_csv(path, columns, rows)


def write_predictions_csv(
    path: str | Path, records: Sequence[MethodPrediction]
) -> int:
    """写出逐条预测明细，供复查与绘图。"""
    rows = [
        {
            "method": r.method,
            "sample_id": r.sample_id,
            "claim_id": r.claim_id,
            "predicted_label": r.predicted_label,
            "gold_label": r.gold_label,
            "insufficiency_score": round(r.insufficiency_score, 6),
            "conflict_score": round(r.conflict_score, 6),
            "correct": int(r.predicted_label == r.gold_label),
        }
        for r in records
    ]
    columns = list(rows[0].keys()) if rows else ["method"]
    return _atomic_write_csv(path, columns, rows)


def plot_confusion_matrix(path: str | Path, report: ClassificationReport) -> Path:
    """画一张混淆矩阵热图。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(report.matrix, dtype=int)

    figure, axes = plt.subplots(figsize=(6.0, 5.2))
    image = axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(report.labels)), report.labels, rotation=45, ha="right")
    axes.set_yticks(range(len(report.labels)), report.labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("gold")
    axes.set_title(
        f"{report.method}  acc={report.accuracy:.3f}  macro-F1={report.macro_f1:.3f}"
    )
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def plot_diagnostic_scatter(
    path: str | Path, records: Sequence[MethodPrediction]
) -> Path:
    """画二维诊断散点图：x = m_theta，y = K_doc，颜色 = gold_state。

    这是论文最直观的一张图：若四类样本能在平面上分出相对清楚的区域，
    就说明「证据不足」与「文档冲突」确实是两个独立且可分的维度。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ds_records = [r for r in records if r.method == DS_METHOD]

    figure, axes = plt.subplots(figsize=(6.4, 5.4))
    for label, color in GOLD_COLORS.items():
        subset = [r for r in ds_records if r.gold_label == label]
        if not subset:
            continue
        axes.scatter(
            [r.insufficiency_score for r in subset],
            [r.conflict_score for r in subset],
            c=color,
            label=f"{label} (n={len(subset)})",
            s=70,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
        )
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.set_xlabel("m(Theta)  —  evidence insufficiency")
    axes.set_ylabel("K_doc  —  document conflict")
    axes.set_title("Two-dimensional diagnostic scatter (D-S)")
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def plot_threshold_sensitivity(
    path: str | Path, search: ThresholdSearchResult
) -> Path:
    """画四分类阈值敏感性曲线。

    K_eval 只控制额外警报，不改变分类标签，因此不会画一条必然水平的伪
    敏感性曲线。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    best = search.best.thresholds

    axis_specs = (
        ("theta_threshold", "m(Theta) threshold"),
        ("document_conflict_threshold", "K_doc threshold"),
    )
    figure, axes_list = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for axes, (field, title) in zip(axes_list, axis_specs):
        others = [f for f, _ in axis_specs if f != field]
        points = sorted(
            (
                (getattr(c.thresholds, field), c.macro_f1)
                for c in search.candidates
                if all(
                    getattr(c.thresholds, other) == getattr(best, other)
                    for other in others
                )
            )
        )
        if points:
            axes.plot(*zip(*points), marker="o", color="#2166ac")
        axes.axvline(getattr(best, field), color="#b2182b", linestyle="--", linewidth=1)
        axes.set_xlabel(title)
        axes.grid(alpha=0.25, linestyle=":")
    axes_list[0].set_ylabel("Macro-F1 (validation)")
    figure.suptitle(
        "Threshold sensitivity "
        f"(other thresholds fixed at optimum, n={search.claim_count})",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
