"""选择性回答（风险-覆盖率）分析：两套风险定义并列报告。

两套定义回答的是**不同的问题**，都要报，不能只留对自己有利的那套。

``max_signal``（原定义）
    ``risk = max(insufficiency_score, conflict_score)``。问的是「证据不足 /
    冲突信号能否预测错误」。

    这个定义对 D-S **系统性不公**：``m_theta`` 与 ``K_doc`` 不是不确定性，
    它们是四类里**两类的证据本身**。``m_theta`` 高意味着 D-S 有把握判
    ``insufficient``，``K_doc`` 高意味着有把握判 ``conflicting``。实测 616 条
    里 170 条 ``insufficient`` 预测的风险分均值高达 0.94，被整体顶到弃答队列
    最前 —— 排序实际在排「是不是判了 insufficient」。

``class_conditional``（修正定义）
    ``risk = 1 - 对所预测类别的置信度``，两侧对称取各自那一类的证据，
    见 :mod:`rag_ds.experiments.selective_confidence`。这是选择性回答文献里的
    标准口径（max-softmax 置信度的直接类比）。

输出（写到 outputs/metrics/selective_answer/）::

    risk_coverage_curves.csv      长表曲线数据，带 risk_definition 列
    auroc_incorrectness.csv       风险分数预测「判错」的 AUROC
    aurc_summary.csv              风险-覆盖率曲线下面积（越低越好）
    risk_coverage.png             两套定义 × 两种设定的四联图

不联网，不调用任何大模型。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "metrics" / "selective_answer"

SETTINGS = {
    "single-eval (refchecker_nli)": ROOT
    / "outputs/metrics/cross_validation/cv_predictions.csv",
    "two-eval (+deberta)": ROOT
    / "outputs/metrics/cross_validation_2eval/cv_predictions.csv",
}
METHODS = [
    "ds",
    "conflict_aware",
    "weighted_average",
    "majority_vote",
    "single_evaluator",
]
COVERAGES = np.round(np.arange(0.50, 1.0001, 0.02), 2)

#: 两套风险定义。值是「从明细行算出风险」的函数。
RISK_DEFINITIONS = {
    "max_signal": lambda sub: sub[["insufficiency_score", "conflict_score"]].max(axis=1),
    "class_conditional": lambda sub: 1.0 - sub["confidence"],
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """风险分数预测正类（判错）的 AUROC，按秩计算并处理并列。"""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 并列取平均秩，否则大量相同分数会让 AUROC 失真。
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]

    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> int:
    """命令行入口，返回进程退出码。"""
    OUT.mkdir(parents=True, exist_ok=True)
    curve_rows, auroc_rows, aurc_rows = [], [], []

    for setting, path in SETTINGS.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "confidence" not in df.columns:
            print(
                f"[缺少 confidence 列] {path}\n"
                "请先用当前版本重跑 scripts/run_cross_validation.py。"
            )
            return 1
        df["incorrect"] = 1 - df["correct"]

        for definition, compute in RISK_DEFINITIONS.items():
            for method in METHODS:
                sub = df[df["method"] == method].copy()
                if sub.empty:
                    continue
                sub["risk"] = compute(sub)
                # 并列时按 claim_id 定序，保证曲线可复现。
                sub = sub.sort_values(
                    ["risk", "claim_id"], ascending=[True, True]
                ).reset_index(drop=True)
                n = len(sub)

                auroc_rows.append(
                    {
                        "setting": setting,
                        "risk_definition": definition,
                        "method": method,
                        "auroc_incorrectness": round(
                            auroc(sub["risk"].to_numpy(), sub["incorrect"].to_numpy()),
                            4,
                        ),
                    }
                )

                accs = []
                for cov in COVERAGES:
                    k = max(1, int(round(n * cov)))
                    accuracy = float(sub["correct"].iloc[:k].mean())
                    accs.append(accuracy)
                    curve_rows.append(
                        {
                            "setting": setting,
                            "risk_definition": definition,
                            "method": method,
                            "coverage": cov,
                            "accuracy": round(accuracy, 4),
                            "n_kept": k,
                        }
                    )
                aurc_rows.append(
                    {
                        "setting": setting,
                        "risk_definition": definition,
                        "method": method,
                        # 风险即 1 - 准确率，在覆盖率上取平均；越低越好。
                        "aurc": round(float(np.mean([1 - a for a in accs])), 4),
                        "acc_100": round(accs[-1], 4),
                        "acc_80": round(accs[list(COVERAGES).index(0.80)], 4),
                        "acc_60": round(accs[list(COVERAGES).index(0.60)], 4),
                    }
                )

    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT / "risk_coverage_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(auroc_rows).to_csv(
        OUT / "auroc_incorrectness.csv", index=False, encoding="utf-8-sig"
    )
    aurc = pd.DataFrame(aurc_rows)
    aurc.to_csv(OUT / "aurc_summary.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(
        len(RISK_DEFINITIONS), len(SETTINGS), figsize=(11.5, 8.0), sharey=True
    )
    for row, definition in enumerate(RISK_DEFINITIONS):
        for col, setting in enumerate(SETTINGS):
            ax = axes[row][col]
            for method in METHODS:
                part = curves[
                    (curves["setting"] == setting)
                    & (curves["risk_definition"] == definition)
                    & (curves["method"] == method)
                ]
                if part.empty:
                    continue
                ax.plot(
                    part["coverage"],
                    part["accuracy"],
                    marker="o" if method == "ds" else None,
                    markersize=3,
                    linewidth=2.0 if method == "ds" else 1.2,
                    label=method,
                )
            ax.set_title(f"{definition}\n{setting}", fontsize=9)
            ax.set_xlabel("coverage")
            if col == 0:
                ax.set_ylabel("accuracy on kept claims")
            ax.grid(alpha=0.25, linestyle=":")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    figure.suptitle(
        "Selective answering: two risk definitions\n"
        "max_signal penalises D-S for confident insufficient/conflicting calls",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(OUT / "risk_coverage.png", dpi=150)
    plt.close(figure)

    print("选择性回答分析完成（两套风险定义并列）。")
    for definition in RISK_DEFINITIONS:
        print()
        print(f"  === 风险定义 {definition} ===")
        print(f"    {'设定':<28}{'方法':<18}{'AURC↓':>8}{'AUROC↑':>9}")
        for setting in SETTINGS:
            for method in METHODS:
                a = aurc[
                    (aurc["setting"] == setting)
                    & (aurc["risk_definition"] == definition)
                    & (aurc["method"] == method)
                ]
                u = [
                    r
                    for r in auroc_rows
                    if r["setting"] == setting
                    and r["risk_definition"] == definition
                    and r["method"] == method
                ]
                if a.empty or not u:
                    continue
                mark = "  ←" if method == "ds" else ""
                print(
                    f"    {setting:<28}{method:<18}"
                    f"{a['aurc'].iloc[0]:>8.4f}{u[0]['auroc_incorrectness']:>9.4f}{mark}"
                )
    print()
    print(f"  输出目录 {OUT}")
    print()
    print("  两套定义问的是不同的问题，论文里都要报：")
    print("    max_signal        证据不足/冲突信号能否预测错误（对 D-S 不公）")
    print("    class_conditional 方法对自己判出的那一类有多确信（标准口径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
