"""样本与关系预测的完整性检查。

D-S pipeline 与 baseline 共用这一份检查，避免两套判定标准漂移 —— 若两条
链路对「什么算合法输入」理解不同，实验对比就失去了共同前提。

所有检查都在计算开始**之前**一次性完成：数据有问题时不会先算出一半结果
再失败。任何缺失都直接报错，不会被跳过、补全或用别的来源顶替。
"""

from __future__ import annotations

from rag_ds.schemas import RAGSample, RelationPrediction

__all__ = [
    "DuplicateRelationPredictionError",
    "InconsistentEvaluatorReliabilityError",
    "MissingRelationPredictionError",
    "NoClaimsError",
    "PipelineError",
    "PredictionKey",
    "ReferentialIntegrityError",
    "check_samples",
    "evaluator_reliability_for",
    "evaluators_for_claim",
    "index_predictions",
    "predictions_for_claim",
]

#: ``(sample_id, claim_id, doc_id, evaluator)``。
PredictionKey = tuple[str, str, str, str]


class PipelineError(Exception):
    """输入数据有问题时抛出的基类。

    与程序 bug 区分开：命令行入口只把这一族异常转成简洁提示与非零退出码，
    其他异常一律照常抛出，不吞。
    """


class NoClaimsError(PipelineError):
    """样本没有任何 claim。

    本阶段**不做**自动 claim 抽取，因此没有 claim 就没有可诊断的对象，
    直接报错而不是静默产出零条结果。
    """


class MissingRelationPredictionError(PipelineError):
    """某个评估器没有覆盖该 claim 的全部检索文档。

    缺失的预测不会被跳过，也不会用 ``p_unknown=1`` 自动补全，更不会拿另一个
    评估器的结果顶替 —— 那三种做法都会让「评估器没判」和「评估器判为不确定」
    在结果里无法区分。
    """


class DuplicateRelationPredictionError(PipelineError):
    """同一 ``(sample, claim, doc, evaluator)`` 出现多条关系预测。"""


class ReferentialIntegrityError(PipelineError):
    """关系预测引用了不存在的 sample / claim / doc。"""


class InconsistentEvaluatorReliabilityError(PipelineError):
    """同一 claim 下，同一评估器的不同文档记录了不同的 evaluator_reliability。

    评估器可靠性是评估器自身的属性，在一条 claim 内必须唯一；若各文档给出
    不同数值，就无从决定该用哪一个。
    """


def check_samples(samples: list[RAGSample]) -> None:
    """检查样本层面的唯一性与 claim 是否存在。

    Args:
        samples: 待检查的样本。

    Raises:
        PipelineError: ``sample_id`` 重复，或样本内 ID 重复。
        NoClaimsError: 某个样本没有 claim。
    """
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise PipelineError(f"sample_id 重复：{sample.sample_id!r}")
        seen.add(sample.sample_id)
        if not sample.claims:
            raise NoClaimsError(
                f"样本 {sample.sample_id!r} 没有任何 claim；"
                "本阶段不做自动 claim 抽取，请先补齐 claims"
            )
        # claim_id / doc_id 的样本内唯一性已由 RAGSample 校验器保证，
        # 这里再确认一次，让前置条件自成闭环。
        claim_ids = [c.claim_id for c in sample.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise PipelineError(f"样本 {sample.sample_id!r} 内 claim_id 重复")
        doc_ids = [c.doc_id for c in sample.contexts]
        if len(set(doc_ids)) != len(doc_ids):
            raise PipelineError(f"样本 {sample.sample_id!r} 内 doc_id 重复")


def index_predictions(
    samples: list[RAGSample], predictions: list[RelationPrediction]
) -> dict[PredictionKey, RelationPrediction]:
    """建立关系预测索引，并检查重复与引用完整性。

    Args:
        samples: 已通过 :func:`check_samples` 的样本。
        predictions: 关系评估器的输出。

    Returns:
        以 :data:`PredictionKey` 为键的索引。

    Raises:
        ReferentialIntegrityError: 预测引用了不存在的 sample / claim / doc。
        DuplicateRelationPredictionError: 同一查询键出现多条预测。
    """
    claims_by_sample = {s.sample_id: {c.claim_id for c in s.claims} for s in samples}
    docs_by_sample = {s.sample_id: {c.doc_id for c in s.contexts} for s in samples}

    table: dict[PredictionKey, RelationPrediction] = {}
    for prediction in predictions:
        if prediction.sample_id not in claims_by_sample:
            raise ReferentialIntegrityError(
                f"关系预测引用了不存在的 sample_id={prediction.sample_id!r}"
                f"（claim_id={prediction.claim_id!r}, doc_id={prediction.doc_id!r}）"
            )
        if prediction.claim_id not in claims_by_sample[prediction.sample_id]:
            raise ReferentialIntegrityError(
                f"关系预测引用了不存在的 claim_id={prediction.claim_id!r}"
                f"（sample_id={prediction.sample_id!r}）"
            )
        if prediction.doc_id not in docs_by_sample[prediction.sample_id]:
            raise ReferentialIntegrityError(
                f"关系预测引用了不存在的 doc_id={prediction.doc_id!r}"
                f"（sample_id={prediction.sample_id!r}, "
                f"claim_id={prediction.claim_id!r}）"
            )

        key: PredictionKey = (
            prediction.sample_id,
            prediction.claim_id,
            prediction.doc_id,
            prediction.evaluator,
        )
        if key in table:
            raise DuplicateRelationPredictionError(
                "同一查询键出现多条关系预测："
                f"sample_id={key[0]!r}, claim_id={key[1]!r}, "
                f"doc_id={key[2]!r}, evaluator={key[3]!r}"
            )
        table[key] = prediction

    return table


def evaluators_for_claim(
    table: dict[PredictionKey, RelationPrediction], sample_id: str, claim_id: str
) -> tuple[str, ...]:
    """列出对该 claim 有预测的评估器。

    返回**按名称排序**的顺序：处理顺序只取决于评估器集合本身，与预测文件的
    行序无关，结果可复现。
    """
    return tuple(
        sorted(
            {key[3] for key in table if key[0] == sample_id and key[1] == claim_id}
        )
    )


def predictions_for_claim(
    table: dict[PredictionKey, RelationPrediction],
    sample: RAGSample,
    claim_id: str,
    evaluator: str,
) -> list[RelationPrediction]:
    """按 ``sample.contexts`` 的原始顺序取出该评估器的全部预测。

    Raises:
        MissingRelationPredictionError: 该评估器未覆盖某篇文档。
    """
    collected: list[RelationPrediction] = []
    for context in sample.contexts:
        key: PredictionKey = (sample.sample_id, claim_id, context.doc_id, evaluator)
        prediction = table.get(key)
        if prediction is None:
            raise MissingRelationPredictionError(
                f"评估器 {evaluator!r} 缺少关系预测："
                f"sample_id={sample.sample_id!r}, claim_id={claim_id!r}, "
                f"doc_id={context.doc_id!r}。"
                "缺失的预测不会被跳过或自动补全，请补齐后重跑"
            )
        collected.append(prediction)
    return collected


def evaluator_reliability_for(
    predictions: list[RelationPrediction],
    sample_id: str,
    claim_id: str,
    evaluator: str,
) -> float:
    """取出该评估器的可靠性，并确认各文档记录一致。

    Raises:
        InconsistentEvaluatorReliabilityError: 各文档记录的可靠性不一致。
    """
    values = {p.evaluator_reliability for p in predictions}
    if len(values) > 1:
        raise InconsistentEvaluatorReliabilityError(
            f"评估器 {evaluator!r} 在 sample_id={sample_id!r}, "
            f"claim_id={claim_id!r} 的各文档预测中记录了不同的 "
            f"evaluator_reliability：{sorted(values)}"
        )
    return values.pop()
