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
    "plot_reliability_sensitivity",
    "plot_threshold_sensitivity",
    "write_ablation_csv",
    "write_main_results_csv",
    "write_predictions_csv",
    "write_reliability_sensitivity_csv",
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
            # 该变体没有改变任何输入时为 True：此行的 Δ=0 不构成任何结论。
            "is_vacuous": result.is_vacuous,
            "theta_threshold": result.thresholds.theta_threshold,
            "document_conflict_threshold": result.thresholds.document_conflict_threshold,
            "evaluator_conflict_threshold": result.thresholds.evaluator_conflict_threshold,
            "sample_count": result.report.sample_count,
        }
        for result in results
    ]
    columns = list(rows[0].keys()) if rows else ["variant"]
    return _atomic_write_csv(path, columns, rows)


def write_reliability_sensitivity_csv(path: str | Path, points) -> int:
    """写出可靠性敏感性分析结果，每个设定一行。

    ``is_oracle`` 列把用到人工标注的设定标出来 —— 那些行只能作为上界报告，
    不能当作模型性能。
    """
    rows = [
        {
            "setting": point.setting,
            "mean_reliability": round(point.mean_reliability, 6),
            "accuracy": round(point.report.accuracy, 6),
            "macro_f1": round(point.report.macro_f1, 6),
            "macro_f1_delta": round(point.macro_f1_delta, 6),
            "mean_m_theta": round(point.mean_m_theta, 6),
            "mean_k_doc": round(point.mean_k_doc, 6),
            "is_oracle": point.is_oracle,
            "sample_count": point.report.sample_count,
        }
        for point in points
    ]
    columns = list(rows[0].keys()) if rows else ["setting"]
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
            "confidence": round(r.confidence, 6),
            "correct": int(r.predicted_label == r.gold_label),
        }
        for r in records
    ]
    columns = list(rows[0].keys()) if rows else ["method"]
    return _atomic_write_csv(path, columns, rows)


def plot_reliability_sensitivity(path: str | Path, points) -> Path:
    """画可靠性敏感性曲线：横轴 r，左轴 Macro-F1，右轴 m_theta / K_doc。

    这张图要展示的是**机制的理论性质在整条链路上依然成立**：折扣只把质量从
    确定焦元移向 Theta，所以 r 下降时 m_theta 必须单调上升、K_doc 单调下降。
    oracle 设定不在曲线上（它不是某个统一的 r），单独用一条横线标出。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sweep = sorted(
        (p for p in points if not p.is_oracle), key=lambda p: p.mean_reliability
    )
    oracle = [p for p in points if p.is_oracle]

    figure, axes = plt.subplots(figsize=(6.6, 4.6))
    levels = [p.mean_reliability for p in sweep]
    axes.plot(
        levels, [p.report.macro_f1 for p in sweep],
        marker="o", color="#1b7837", label="Macro-F1",
    )
    axes.set_xlabel("uniform document reliability  r")
    axes.set_ylabel("Macro-F1", color="#1b7837")
    axes.tick_params(axis="y", labelcolor="#1b7837")
    axes.grid(alpha=0.25, linestyle=":")

    twin = axes.twinx()
    twin.plot(
        levels, [p.mean_m_theta for p in sweep],
        marker="s", linestyle="--", color="#7f7f7f", label="mean m(Theta)",
    )
    twin.plot(
        levels, [p.mean_k_doc for p in sweep],
        marker="^", linestyle="--", color="#2166ac", label="mean K_doc",
    )
    twin.set_ylabel("mean m(Theta) / K_doc")

    if oracle:
        axes.axhline(
            oracle[0].report.macro_f1,
            color="#b2182b", linestyle=":", linewidth=1.4,
            label=f"oracle reliability (upper bound) = {oracle[0].report.macro_f1:.3f}",
        )

    handles, labels = axes.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes.legend(
        handles + twin_handles, labels + twin_labels, loc="center left", fontsize=8
    )
    axes.set_title(
        "Reliability discounting sensitivity\n"
        "sanity check: m(Theta) must rise and K_doc must fall as r decreases",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


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


def _padded_limits(values: Sequence[float], pad_ratio: float = 0.08) -> tuple[float, float]:
    """给一组取值算出带边距的坐标范围，全部相同时退化为一个小窗口。"""
    if not values:
        return -0.02, 1.02
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        pad = max(abs(low) * pad_ratio, 0.01)
    else:
        pad = span * pad_ratio
    return max(low - pad, -0.02), min(high + pad, 1.02)


def plot_diagnostic_scatter(
    path: str | Path,
    records: Sequence[MethodPrediction],
    *,
    full_range: bool = False,
) -> Path:
    """画二维诊断散点图：x = m_theta，y = K_doc，颜色 = gold_state。

    这是论文最直观的一张图：若四类样本能在平面上分出相对清楚的区域，
    就说明「证据不足」与「文档冲突」确实是两个独立且可分的维度。

    Args:
        path: 图片输出路径。
        records: 逐条预测记录。
        full_range: 强制把两轴都画成 ``[0, 1]``。默认按数据自动缩放 ——
            融合后的 m_theta 常常只落在 ``[0, 0.05]`` 这样的窄带里，
            固定成 ``[0, 1]`` 会把所有点压成左下角一个看不清的小团，
            图上什么结构都读不出来。图里会标注实际的取值范围，
            避免自动缩放让人误判量级。
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
    theta_values = [r.insufficiency_score for r in ds_records]
    conflict_values = [r.conflict_score for r in ds_records]
    if full_range or not ds_records:
        axes.set_xlim(-0.02, 1.02)
        axes.set_ylim(-0.02, 1.02)
        subtitle = "axes fixed to [0, 1]"
    else:
        axes.set_xlim(*_padded_limits(theta_values))
        axes.set_ylim(*_padded_limits(conflict_values))
        distinct = len({(round(x, 9), round(y, 9)) for x, y in zip(theta_values, conflict_values)})
        subtitle = (
            f"axes auto-scaled | m(Theta) in "
            f"[{min(theta_values):.4f}, {max(theta_values):.4f}], "
            f"K_doc in [{min(conflict_values):.4f}, {max(conflict_values):.4f}]"
            # 离散标签 + 固定文档数会让诊断点落在一个有限格点集上，
            # 重叠的点在图里看不出来，必须写明白。
            "\n"
            f"{len(ds_records)} claims occupy {distinct} distinct points "
            f"(markers overlap)"
        )
    axes.set_xlabel("m(Theta)  —  evidence insufficiency")
    axes.set_ylabel("K_doc  —  document conflict")
    # 自动缩放会放大窄带里的结构，但也容易让人误读量级，所以把真实范围
    # 作为副标题写在图上；用换行而不是额外的 text，避免与标题重叠。
    axes.set_title(
        "Two-dimensional diagnostic scatter (D-S)\n" + subtitle, fontsize=10
    )
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
