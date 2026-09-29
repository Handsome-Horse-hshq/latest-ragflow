"""把 RAGChecker 的真实输出解析成本项目的关系预测。

本模块**不 import ``ragchecker``**，只读取它已经跑完写在磁盘上的
``checking_outputs.json``。这样 RAGChecker 换版本时，需要改的只有这一个
文件，D-S 核心一行都不用动。

依据的输出契约（取自 RAGChecker 源码，非凭记忆）
------------------------------------------------
``ragchecker/container.py`` 的 ``RAGResult`` 中：

* ``retrieved2response: List[List[str]]`` —— "entailment results of
  retrieved -> response"，形状为 ``[claim_num][reference_num]``，
  即**外层是抽取出的 response claim，内层是检索文档**；
* 标签取值来自 ``refchecker`` 的 ``checker_base.py``：
  ``'Entailment'`` / ``'Neutral'`` / ``'Contradiction'``；
* ``ragchecker/metrics.py`` 中 ``faithfulness`` 只依赖 ``retrieved2response``，
  因此跑这一个 metric 就能拿到完整网格，不需要 ``gt_answer`` 参与计算。

claim 对齐
----------
RAGChecker 会用自己的 extractor 从 ``response`` 里抽 claim，抽出的条数 N
不一定等于 1。本项目的 ``claim_id`` 是数据集固定的，因此：

* N == 1 —— 直接一一对应；
* N > 1 —— 把 N 个子 claim 的标签看作对同一条 claim 的 N 次投票，按标签
  分布做凸组合（见 :func:`distribution_to_probabilities`）。N == 1 时该式
  精确退化为查表，两条路径不会给出不一致的结果；
* N == 0 —— extractor 没抽出任何 claim，**报错**，不伪造成 neutral。

``strict_single_claim=True`` 时 N != 1 一律报错，用于要求严格对齐的场合。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ds.relation_evaluation.ragchecker_adapter import (
    LabelProbabilityMapping,
    RAGCheckerLabel,
    prediction_from_probabilities,
)
from rag_ds.schemas import (
    PROBABILITY_SUM_TOLERANCE,
    RAGSample,
    RelationPrediction,
)

__all__ = [
    "NLI_LABEL_ORDER",
    "ClaimLabelDistribution",
    "RAGCheckerOutputError",
    "distribution_to_probabilities",
    "distributions_to_predictions",
    "parse_checking_output_probabilities",
    "parse_checking_outputs",
]

#: 连续概率字段的类别顺序，取自 refchecker/checker/nli_checker.py 的 ``LABELS``。
#:
#: 注意它与本项目三元组的顺序**不同**：NLI 的第二类是 Neutral（对应
#: ``p_unknown``），第三类才是 Contradiction（对应 ``p_refute``）。
NLI_LABEL_ORDER: tuple[str, str, str] = ("Entailment", "Neutral", "Contradiction")


class RAGCheckerOutputError(ValueError):
    """RAGChecker 输出文件与当前 split 的样本对不上，或结构不符合契约。

    一律直接报错，不做任何猜测式补全：错位一格会让整份实验结果失去意义，
    而这种错位在数字上是看不出来的。
    """


#: 大小写无关的标签解析表。
_LABEL_ALIASES: dict[str, RAGCheckerLabel] = {
    "entailment": RAGCheckerLabel.ENTAILMENT,
    "contradiction": RAGCheckerLabel.CONTRADICTION,
    "neutral": RAGCheckerLabel.NEUTRAL,
}


def _parse_label(raw: object, where: str) -> RAGCheckerLabel:
    """把 RAGChecker 的标签字符串解析成枚举。"""
    if not isinstance(raw, str):
        raise RAGCheckerOutputError(
            f"{where} 的标签必须是字符串，实际为 {type(raw).__name__}"
        )
    label = _LABEL_ALIASES.get(raw.strip().lower())
    if label is None:
        raise RAGCheckerOutputError(
            f"{where} 出现未知标签 {raw!r}；"
            "RefChecker 只会给出 Entailment / Neutral / Contradiction"
        )
    return label


class ClaimLabelDistribution(BaseModel):
    """一个 (sample, claim, doc) 组合上，RAGChecker 各标签的票数。

    子 claim 只有一条时就是 one-hot，这是最常见的情形。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    claim_id: str
    doc_id: str
    entailment: int = Field(default=0, ge=0)
    neutral: int = Field(default=0, ge=0)
    contradiction: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_non_empty(self) -> ClaimLabelDistribution:
        if self.total == 0:
            raise ValueError(
                f"({self.sample_id}, {self.claim_id}, {self.doc_id}) 没有任何标签票数"
            )
        return self

    @property
    def total(self) -> int:
        """票数总和，即 RAGChecker 为该 claim 抽出的子 claim 条数。"""
        return self.entailment + self.neutral + self.contradiction

    @property
    def weights(self) -> dict[RAGCheckerLabel, float]:
        """归一化到和为 1 的标签权重。"""
        total = float(self.total)
        return {
            RAGCheckerLabel.ENTAILMENT: self.entailment / total,
            RAGCheckerLabel.NEUTRAL: self.neutral / total,
            RAGCheckerLabel.CONTRADICTION: self.contradiction / total,
        }

    @property
    def majority_label(self) -> RAGCheckerLabel | None:
        """票数唯一最多的标签；并列时为 ``None``。"""
        counts = [
            (self.entailment, RAGCheckerLabel.ENTAILMENT),
            (self.neutral, RAGCheckerLabel.NEUTRAL),
            (self.contradiction, RAGCheckerLabel.CONTRADICTION),
        ]
        counts.sort(key=lambda item: item[0], reverse=True)
        if counts[0][0] == counts[1][0]:
            return None
        return counts[0][1]


def distribution_to_probabilities(
    distribution: ClaimLabelDistribution, mapping: LabelProbabilityMapping
) -> tuple[float, float, float]:
    """按标签权重对映射表做凸组合，得到归一的三元概率。

    子 claim 只有一条时，结果与直接查表**完全相同**（权重为 one-hot）。
    """
    support = refute = unknown = 0.0
    for label, weight in distribution.weights.items():
        if weight == 0.0:
            continue
        p_s, p_r, p_u = mapping.probabilities(label)
        support += weight * p_s
        refute += weight * p_r
        unknown += weight * p_u

    total = support + refute + unknown
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        # 凸组合理论上保持归一，这里只兜住浮点累加的尾差。
        support, refute, unknown = support / total, refute / total, unknown / total
    return support, refute, unknown


def _check_docs_align(
    sample: RAGSample, retrieved: Sequence[Mapping[str, object]]
) -> list[str]:
    """核对检索文档与样本的 contexts 逐位一致，返回 doc_id 列表。"""
    if len(retrieved) != len(sample.contexts):
        raise RAGCheckerOutputError(
            f"样本 {sample.sample_id!r} 的检索文档条数不一致："
            f"输出里有 {len(retrieved)} 段，数据集里有 {len(sample.contexts)} 段"
        )
    doc_ids: list[str] = []
    for position, (chunk, context) in enumerate(zip(retrieved, sample.contexts)):
        if not isinstance(chunk, Mapping):
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 第 {position} 段检索文档不是 JSON 对象"
            )
        raw_doc_id = chunk.get("doc_id")
        if raw_doc_id is None:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 第 {position} 段检索文档缺少 doc_id；"
                "无法确认顺序，请用 scripts/export_ragchecker_input.py 生成输入"
            )
        doc_id = str(raw_doc_id).strip()
        if doc_id != context.doc_id:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 第 {position} 段 doc_id 对不上："
                f"输出为 {doc_id!r}，数据集为 {context.doc_id!r}"
            )
        doc_ids.append(doc_id)
    return doc_ids


def parse_checking_outputs(
    payload: Mapping[str, object],
    samples: Sequence[RAGSample],
    *,
    strict_single_claim: bool = False,
) -> list[ClaimLabelDistribution]:
    """解析 ``checking_outputs.json``，产出完整的 (claim, doc) 标签分布。

    Args:
        payload: 已读入内存的 RAGChecker 输出 JSON。
        samples: 当前 split 的样本，决定期望的完整网格。
        strict_single_claim: 为 ``True`` 时，抽出的子 claim 条数不等于 1 即报错。

    Returns:
        每个 (sample, claim, doc) 组合一条，顺序与 ``samples`` 一致。

    Raises:
        RAGCheckerOutputError: 结构不符合契约，或与样本对不上。
    """
    results = payload.get("results")
    if not isinstance(results, list):
        raise RAGCheckerOutputError("输出 JSON 顶层缺少 results 数组")

    by_query: dict[str, Mapping[str, object]] = {}
    for position, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise RAGCheckerOutputError(f"results[{position}] 不是 JSON 对象")
        raw_id = result.get("query_id")
        if raw_id is None:
            raise RAGCheckerOutputError(f"results[{position}] 缺少 query_id")
        query_id = str(raw_id).strip()
        if query_id in by_query:
            raise RAGCheckerOutputError(f"输出中 query_id 重复：{query_id!r}")
        by_query[query_id] = result

    expected = {sample.sample_id for sample in samples}
    missing = sorted(expected - set(by_query))
    if missing:
        raise RAGCheckerOutputError(
            f"输出缺少 {len(missing)} 条样本的结果，前几条："
            f"{missing[:5]}；缺失的判断不会被补成 neutral"
        )
    extra = sorted(set(by_query) - expected)
    if extra:
        raise RAGCheckerOutputError(
            f"输出包含 {len(extra)} 条不属于本 split 的 query_id，前几条：{extra[:5]}"
        )

    distributions: list[ClaimLabelDistribution] = []
    for sample in samples:
        result = by_query[sample.sample_id]
        if len(sample.claims) != 1:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 有 {len(sample.claims)} 条 claim；"
                "本摄取路径要求每个样本恰好一条 claim，否则无法与 RAGChecker "
                "自行抽取的 claim 对齐"
            )
        claim_id = sample.claims[0].claim_id

        retrieved = result.get("retrieved_context")
        if not isinstance(retrieved, list):
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 的结果缺少 retrieved_context"
            )
        doc_ids = _check_docs_align(sample, retrieved)

        matrix = result.get("retrieved2response")
        if matrix is None:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 的结果里 retrieved2response 为空；"
                "请确认运行 RAGChecker 时 --metrics 至少包含 faithfulness"
            )
        if not isinstance(matrix, list):
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 的 retrieved2response 不是数组"
            )
        if not matrix:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 没有抽出任何 claim"
                "（retrieved2response 为空数组）；"
                "这是 extractor 的失败，不会被当作 neutral 处理"
            )
        if strict_single_claim and len(matrix) != 1:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 抽出了 {len(matrix)} 条子 claim，"
                "但当前要求严格一一对齐（strict_single_claim=True）"
            )

        tallies: dict[str, dict[RAGCheckerLabel, int]] = {
            doc_id: {label: 0 for label in RAGCheckerLabel} for doc_id in doc_ids
        }
        for claim_index, row in enumerate(matrix):
            if not isinstance(row, list):
                raise RAGCheckerOutputError(
                    f"样本 {sample.sample_id!r} 的 "
                    f"retrieved2response[{claim_index}] 不是数组；"
                    "该字段形状必须是 [claim_num][doc_num]"
                )
            if len(row) != len(doc_ids):
                raise RAGCheckerOutputError(
                    f"样本 {sample.sample_id!r} 的 retrieved2response[{claim_index}] 有 "
                    f"{len(row)} 个标签，但检索文档有 {len(doc_ids)} 段；"
                    "形状必须是 [claim_num][doc_num]，请勿把两维调换"
                )
            for doc_index, raw_label in enumerate(row):
                label = _parse_label(
                    raw_label,
                    f"样本 {sample.sample_id!r} 的 "
                    f"retrieved2response[{claim_index}][{doc_index}]",
                )
                tallies[doc_ids[doc_index]][label] += 1

        for doc_id in doc_ids:
            counts = tallies[doc_id]
            distributions.append(
                ClaimLabelDistribution(
                    sample_id=sample.sample_id,
                    claim_id=claim_id,
                    doc_id=doc_id,
                    entailment=counts[RAGCheckerLabel.ENTAILMENT],
                    neutral=counts[RAGCheckerLabel.NEUTRAL],
                    contradiction=counts[RAGCheckerLabel.CONTRADICTION],
                )
            )
    return distributions


def parse_checking_output_probabilities(
    payload: Mapping[str, object],
    samples: Sequence[RAGSample],
    *,
    evaluator: str,
    evaluator_reliability: float = 1.0,
) -> list[RelationPrediction]:
    """读取连续概率字段，直接产出 :class:`RelationPrediction`。

    为什么要有这条路径
    ------------------
    离散标签会把诊断空间压成有限格点：checker 每篇文档只给 3 个标签之一、
    共 n 篇文档，而 Dempster 组合与 ``K_doc = 1 - prod(1 - K_i)`` 都与顺序无关，
    于是诊断结果**只取决于各标签的计数**，最多只有 ``C(n+2, 2)`` 种取值。
    n=5 时上限是 21 个点 —— 实测 80 条 claim 只落在 11 个点上，其中 42 条挤在
    同一个点，**任何阈值都分不开它们**。

    NLI 模型本身输出的就是三类 softmax，正好对应
    ``(p_support, p_unknown, p_refute)``；保留连续值就不会有这个格点问题。
    本函数因此**完全不经过标签映射表**。

    字段约定
    --------
    ``retrieved2response_probabilities``，形状 ``[claim_num][doc_num][3]``，
    每个三元组按 :data:`NLI_LABEL_ORDER` 排列（Entailment / Neutral /
    Contradiction）。多条子 claim 时按文档取平均 —— 与
    :func:`distribution_to_probabilities` 的凸组合口径一致。

    Args:
        payload: 已读入内存的输出 JSON。
        samples: 当前 split 的样本。
        evaluator: 写进每条输出的评估器名。
        evaluator_reliability: 评估器可靠性。

    Returns:
        每个 (sample, claim, doc) 组合一条。

    Raises:
        RAGCheckerOutputError: 缺少概率字段、形状不符，或概率不归一。
    """
    name = evaluator.strip()
    if not name:
        raise ValueError("evaluator 名称不能为空")

    # 先走一遍标签解析：样本覆盖、doc_id 对齐、形状检查全部复用同一套校验。
    distributions = parse_checking_outputs(payload, samples)
    by_key = {(d.sample_id, d.claim_id, d.doc_id): d for d in distributions}

    results = payload["results"]  # parse_checking_outputs 已确认存在且为 list
    by_query = {
        str(r.get("query_id", "")).strip(): r
        for r in results
        if isinstance(r, Mapping)
    }

    predictions: list[RelationPrediction] = []
    for sample in samples:
        result = by_query[sample.sample_id]
        claim_id = sample.claims[0].claim_id
        doc_ids = [chunk.doc_id for chunk in sample.contexts]

        matrix = result.get("retrieved2response_probabilities")
        if matrix is None:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 缺少 retrieved2response_probabilities；"
                "请用 scripts/run_refchecker_nli.py --emit-probabilities 重跑"
            )
        if not isinstance(matrix, list) or not matrix:
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 的 retrieved2response_probabilities 不是非空数组"
            )

        label_matrix = result.get("retrieved2response")
        if isinstance(label_matrix, list) and len(matrix) != len(label_matrix):
            raise RAGCheckerOutputError(
                f"样本 {sample.sample_id!r} 的概率矩阵有 {len(matrix)} 条子 claim，"
                f"标签矩阵有 {len(label_matrix)} 条；两者必须同源"
            )

        summed: list[list[float]] = [[0.0, 0.0, 0.0] for _ in doc_ids]
        for claim_index, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != len(doc_ids):
                raise RAGCheckerOutputError(
                    f"样本 {sample.sample_id!r} 的概率矩阵第 {claim_index} 行长度不是 "
                    f"{len(doc_ids)}；形状必须是 [claim_num][doc_num][3]"
                )
            for doc_index, triple in enumerate(row):
                if not isinstance(triple, list) or len(triple) != 3:
                    raise RAGCheckerOutputError(
                        f"样本 {sample.sample_id!r} 的 "
                        f"概率[{claim_index}][{doc_index}] 不是三元组"
                    )
                values = [float(x) for x in triple]
                if any(v < 0.0 or v > 1.0 for v in values):
                    raise RAGCheckerOutputError(
                        f"样本 {sample.sample_id!r} 的 "
                        f"概率[{claim_index}][{doc_index}] 超出 [0, 1]：{values}"
                    )
                total = sum(values)
                if abs(total - 1.0) > 1e-3:
                    raise RAGCheckerOutputError(
                        f"样本 {sample.sample_id!r} 的 "
                        f"概率[{claim_index}][{doc_index}] 之和为 {total!r}，不归一"
                    )
                for position in range(3):
                    summed[doc_index][position] += values[position]

        sub_claims = len(matrix)
        for doc_index, doc_id in enumerate(doc_ids):
            entail, neutral, contra = (v / sub_claims for v in summed[doc_index])
            scale = entail + neutral + contra
            # NLI 的类别顺序是 (Entailment, Neutral, Contradiction)，
            # 对应本项目的 (p_support, p_unknown, p_refute)：中间两项要换位。
            predictions.append(
                prediction_from_probabilities(
                    sample_id=sample.sample_id,
                    claim_id=claim_id,
                    doc_id=doc_id,
                    evaluator=name,
                    probabilities=(entail / scale, contra / scale, neutral / scale),
                    evaluator_reliability=evaluator_reliability,
                )
            )
        # 复用标签路径的覆盖性检查结果，确保两条路径看到的是同一批组合。
        for doc_id in doc_ids:
            if (sample.sample_id, claim_id, doc_id) not in by_key:
                raise RAGCheckerOutputError(
                    f"组合 ({sample.sample_id}, {claim_id}, {doc_id}) 只在概率里出现"
                )
    return predictions


def distributions_to_predictions(
    distributions: Iterable[ClaimLabelDistribution],
    mapping: LabelProbabilityMapping,
    *,
    evaluator: str,
    evaluator_reliability: float = 1.0,
) -> list[RelationPrediction]:
    """按映射表把标签分布转成 :class:`RelationPrediction`。"""
    name = evaluator.strip()
    if not name:
        raise ValueError("evaluator 名称不能为空")
    return [
        prediction_from_probabilities(
            sample_id=item.sample_id,
            claim_id=item.claim_id,
            doc_id=item.doc_id,
            evaluator=name,
            probabilities=distribution_to_probabilities(item, mapping),
            evaluator_reliability=evaluator_reliability,
        )
        for item in distributions
    ]
