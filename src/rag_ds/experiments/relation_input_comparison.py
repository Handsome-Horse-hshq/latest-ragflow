"""两份关系输入（例如 NLI 小模型与大模型）之间的配对比较。

换关系评估器之后，要分开回答两个问题：

1. **关系这一层**：数据集带人工逐篇标注时，哪个评估器更接近人工多数票。
   只看 argmax 标签会漏掉信息 —— 两个评估器可以在「取最大的那一类」上差不多，
   而在概率的**排序**上差很多，后者才是 D-S 与各个统计量实际用到的东西。
   所以 argmax 准确率与逐类 AUROC 都报。
2. **判定这一层**：同一批 claim、同一套分折与搜索协议下，每个方法换了关系输入
   之后 Macro-F1 变了多少。逐条配对，才能消掉「这批 claim 有多难」。
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from rag_ds.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED
from rag_ds.metrics.detection import paired_auroc_bootstrap
from rag_ds.metrics.significance import BootstrapComparison, paired_bootstrap
from rag_ds.schemas import EvidenceState, RelationPrediction

__all__ = [
    "RELATION_NAMES",
    "compare_prediction_runs",
    "read_cv_predictions",
    "relation_level_comparison",
]

#: 关系三元组的顺序，与 :class:`~rag_ds.schemas.RelationPrediction` 的字段一致。
RELATION_NAMES: tuple[str, ...] = ("support", "refute", "unknown")

_GOLD_LABELS: tuple[str, ...] = tuple(
    state.value
    for state in (
        EvidenceState.SUPPORTED,
        EvidenceState.REFUTED,
        EvidenceState.INSUFFICIENT,
        EvidenceState.CONFLICTING,
    )
)
_TIE = 1e-9

PairKey = tuple[str, str, str]
ClaimKey = tuple[str, str]


def _index(predictions: Sequence[RelationPrediction], who: str) -> dict[PairKey, tuple[float, ...]]:
    indexed: dict[PairKey, tuple[float, ...]] = {}
    for item in predictions:
        key = (item.sample_id, item.claim_id, item.doc_id)
        if key in indexed:
            raise ValueError(f"{who} 里 (claim, 文档) 组合重复：{key}")
        indexed[key] = (item.p_support, item.p_refute, item.p_unknown)
    return indexed


def _unique_argmax(triple: Sequence[float]) -> int | None:
    """唯一最大的那一类的下标；并列时返回 ``None``。"""
    top = max(triple)
    winners = [index for index, value in enumerate(triple) if top - value <= _TIE]
    return winners[0] if len(winners) == 1 else None


def relation_level_comparison(
    oracle: Sequence[RelationPrediction],
    reference: Sequence[RelationPrediction],
    rival: Sequence[RelationPrediction],
    *,
    reference_name: str,
    rival_name: str,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """在人工多数票唯一的 (claim, 文档) 组合上比较两个评估器。

    Args:
        oracle: 人工标注的关系分布。
        reference: 参照评估器的关系概率。
        rival: 对手评估器的关系概率。
        reference_name: 参照评估器名。
        rival_name: 对手评估器名。
        n_resamples: 自助重采样次数。
        seed: 随机种子。

    Returns:
        组合数、人工多数票分布、两个评估器各自的 argmax 准确率与预测分布、
        两者 argmax 的一致率，以及逐类 AUROC 的配对比较（参照 − 对手）。

    Raises:
        ValueError: 三份输入覆盖的组合不一致，或没有任何人工多数票唯一的组合。
    """
    truth = _index(oracle, "人工标注")
    scores = {
        reference_name: _index(reference, reference_name),
        rival_name: _index(rival, rival_name),
    }
    if len(scores) != 2:
        raise ValueError("参照评估器与对手评估器不能同名")
    for name, indexed in scores.items():
        if set(indexed) != set(truth):
            raise ValueError(f"{name} 覆盖的 (claim, 文档) 组合与人工标注不一致，无法配对")

    majority = {key: _unique_argmax(triple) for key, triple in truth.items()}
    keys = sorted(key for key, label in majority.items() if label is not None)
    if not keys:
        raise ValueError("没有人工多数票唯一的组合")

    evaluators = {}
    for name, indexed in scores.items():
        predicted = [_unique_argmax(indexed[key]) for key in keys]
        counts = {relation: 0 for relation in RELATION_NAMES}
        counts["tie"] = 0
        for label in predicted:
            counts["tie" if label is None else RELATION_NAMES[label]] += 1
        evaluators[name] = {
            # 并列算错：评估器没有给出唯一判断。
            "argmax_accuracy": sum(p == majority[k] for p, k in zip(predicted, keys, strict=True))
            / len(keys),
            "argmax_counts": counts,
        }

    comparisons = []
    for index, relation in enumerate(RELATION_NAMES):
        (comparison,) = paired_auroc_bootstrap(
            [majority[key] == index for key in keys],
            [scores[reference_name][key][index] for key in keys],
            {rival_name: [scores[rival_name][key][index] for key in keys]},
            reference=reference_name,
            positive_label=relation,
            n_resamples=n_resamples,
            seed=seed,
        )
        comparisons.append(comparison.model_dump())

    agree = sum(
        _unique_argmax(scores[reference_name][key]) == _unique_argmax(scores[rival_name][key])
        for key in truth
    )
    return {
        "pair_count": len(truth),
        "pairs_with_unique_majority": len(keys),
        "majority_counts": {
            relation: sum(majority[key] == index for key in keys)
            for index, relation in enumerate(RELATION_NAMES)
        },
        "evaluators": evaluators,
        "argmax_agreement": agree / len(truth),
        "auroc_comparisons": comparisons,
    }


def read_cv_predictions(path: str | Path) -> dict[str, dict[ClaimKey, tuple[str, str]]]:
    """读交叉验证的逐条预测：``方法 -> (样本, claim) -> (预测, 金标准)``。"""
    by_method: dict[str, dict[ClaimKey, tuple[str, str]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], row["claim_id"])
            rows = by_method.setdefault(row["method"], {})
            if key in rows:
                raise ValueError(f"{path} 里 {row['method']} 的 claim 重复：{key}")
            rows[key] = (row["predicted_label"], row["gold_label"])
    return by_method


def compare_prediction_runs(
    reference: Mapping[str, Mapping[ClaimKey, tuple[str, str]]],
    rival: Mapping[str, Mapping[ClaimKey, tuple[str, str]]],
    *,
    reference_name: str,
    rival_name: str,
    labels: Sequence[str] = _GOLD_LABELS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[BootstrapComparison]:
    """两次交叉验证里同名的方法逐一配对比较 Macro-F1（参照 − 对手）。

    Args:
        reference: 参照那次运行的逐条预测。
        rival: 对手那次运行的逐条预测。
        reference_name: 参照运行的名字，写进比较结果的方法名。
        rival_name: 对手运行的名字。
        labels: 参与 Macro-F1 平均的固定标签集。
        n_resamples: 自助重采样次数。
        seed: 随机种子。

    Returns:
        两次运行都有的方法各一条比较，按方法名排序。

    Raises:
        ValueError: 两次运行没有同名方法，或某个方法覆盖的 claim / 金标准不一致。
    """
    methods = sorted(set(reference) & set(rival))
    if not methods:
        raise ValueError("两次运行没有同名的方法")
    comparisons = []
    for method in methods:
        ours, theirs = reference[method], rival[method]
        if set(ours) != set(theirs):
            raise ValueError(f"{method} 在两次运行里覆盖的 claim 不一致，无法配对")
        keys = sorted(ours)
        truth = [ours[key][1] for key in keys]
        if truth != [theirs[key][1] for key in keys]:
            raise ValueError(f"{method} 在两次运行里的金标准不一致")
        comparisons.append(
            paired_bootstrap(
                truth,
                [ours[key][0] for key in keys],
                [theirs[key][0] for key in keys],
                f"{method}@{reference_name}",
                f"{method}@{rival_name}",
                labels=labels,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    return comparisons
