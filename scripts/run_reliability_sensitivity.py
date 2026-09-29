"""可靠性折扣的敏感性分析。

用法::

    python scripts/run_reliability_sensitivity.py \
        --config configs/climate_fever_refchecker_nli_probs_test.yaml \
        --provenance data/processed/climate_fever_v1/test_provenance.jsonl

为什么要跑这一步
----------------
``no_reliability`` 消融在 CLIMATE-FEVER v1 上是**恒等变换**：数据集里所有
``reliability`` 都是 1.0，"去掉折扣"等于什么都没做，Δ 必然为 0。那个 0 说明的是
消融没跑起来，不能读成「可靠性折扣没有作用」。可靠性折扣是 D-S 链路的核心机制，
不能留成空白，于是改用两组敏感性分析把它实际跑起来：

* **S1 均匀扫描**（无 oracle）—— 所有文档可靠性统一设为 r 并扫 r。检验的是
  机制本身的性质：折扣只把质量从确定焦元移向 Theta，r 下降时 ``m_theta``
  应当单调上升。可以直接报告。
* **S2 标注一致度作可靠性**（**oracle 上界**）—— 用 CLIMATE-FEVER 记录的标注
  投票熵算 ``r = 1 - H / ln(3)``。这用到了人工标注，**只能作为上界报告**，
  回答「若存在完美的可靠性估计器，折扣最多能带来多少」。

两组都是敏感性分析，不是主结果。输出 CSV 的 ``is_oracle`` 列把 S2 标了出来。

不联网，不调用任何大模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from rag_ds.data_io import load_relation_predictions, load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.diagnostics.models import DiagnosticThresholds
from rag_ds.experiments import (
    oracle_reliability_from_provenance,
    plot_reliability_sensitivity,
    run_reliability_sensitivity,
    write_reliability_sensitivity_csv,
)
from rag_ds.integrity import PipelineError
from rag_ds.model_runs import ModelRunError, verify_model_run_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="可靠性折扣的敏感性分析")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="谱系 JSONL；给出时追加 S2（oracle 上界），不给则只跑 S1",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="S1 扫描的可靠性取值，必须包含 1.0 作为基准",
    )
    parser.add_argument("--out", type=Path, default=None, help="结果 CSV 路径")
    parser.add_argument(
        "--figure", type=Path, default=None, help="敏感性曲线图路径；不给则不画"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load(config_path: Path) -> dict:
    """读取实验配置并把相对路径解析到配置文件目录。"""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    paths = {
        key: (base / value).resolve() if not Path(value).is_absolute() else Path(value)
        for key, value in raw["paths"].items()
    }
    return {"paths": paths, **{k: v for k, v in raw.items() if k != "paths"}}


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    config = _load(args.config.resolve())
    paths = config["paths"]
    out_path: Path = args.out or (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / f"{args.config.stem}_reliability_sensitivity.csv"
    )
    if out_path.exists() and not args.overwrite:
        print(f"[输出文件已存在] {out_path}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    try:
        thresholds = DiagnosticThresholds(**config["diagnostics"])
        if "manifest" in paths:
            split_name = config.get("data", {}).get("split")
            if split_name not in {"train", "validation", "test"}:
                raise DatasetManifestError(
                    "配置包含 manifest 时，data.split 必须是 train/validation/test 之一"
                )
            if "model_run" in paths:
                verify_model_run_artifacts(
                    paths["model_run"],
                    paths["manifest"],
                    paths["samples"],
                    paths["relation_predictions"],
                    split_name,
                )
            manifest = load_dataset_manifest(paths["manifest"])
            verify_artifact(
                Path(paths["manifest"]).parent, manifest.splits[split_name].samples
            )
        samples = load_samples(paths["samples"])
        predictions = load_relation_predictions(paths["relation_predictions"])

        oracle = None
        if args.provenance is not None:
            records = [
                json.loads(line)
                for line in args.provenance.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            oracle = oracle_reliability_from_provenance(records)
            covered = {c.doc_id for s in samples for c in s.contexts}
            missing = covered - set(oracle)
            if missing:
                raise ValueError(
                    f"谱系没有覆盖 {len(missing)} 篇文档的可靠性，"
                    f"前几条：{sorted(missing)[:5]}"
                )

        points = run_reliability_sensitivity(
            samples,
            predictions,
            thresholds,
            levels=tuple(args.levels),
            oracle_reliability=oracle,
        )
    except (
        PipelineError,
        DatasetManifestError,
        ModelRunError,
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    write_reliability_sensitivity_csv(out_path, points)
    if args.figure is not None:
        plot_reliability_sensitivity(args.figure, points)

    print("可靠性敏感性分析完成。")
    print(f"  配置 {args.config}")
    print(f"  claim 数量：{points[0].report.sample_count}")
    print()
    print("  S1 —— 均匀可靠性扫描（无 oracle，可直接报告）")
    print(
        f"    {'设定':<22}{'平均可靠性':>10}{'macroF1':>10}{'Δ':>9}"
        f"{'m_theta 均值':>13}{'K_doc 均值':>12}"
    )
    for point in points:
        if point.is_oracle:
            continue
        print(
            f"    {point.setting:<22}{point.mean_reliability:>10.2f}"
            f"{point.report.macro_f1:>10.4f}{point.macro_f1_delta:>+9.4f}"
            f"{point.mean_m_theta:>13.4f}{point.mean_k_doc:>12.4f}"
        )

    oracle_points = [p for p in points if p.is_oracle]
    if oracle_points:
        print()
        print("  S2 —— 标注一致度作可靠性（**oracle 上界，不是模型性能**）")
        for point in oracle_points:
            print(
                f"    {point.setting:<22}{point.mean_reliability:>10.2f}"
                f"{point.report.macro_f1:>10.4f}{point.macro_f1_delta:>+9.4f}"
                f"{point.mean_m_theta:>13.4f}{point.mean_k_doc:>12.4f}"
            )

    print()
    print(f"  输出 {out_path}")
    if args.figure is not None:
        print(f"       {args.figure}")
    print()
    print("  论文里请注意：这是敏感性分析，不是主结果；带 is_oracle=True 的行")
    print("  用到了人工标注，只能作为上界报告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
