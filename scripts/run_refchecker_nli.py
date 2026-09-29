"""用 RefChecker 自带的 NLI checker 跑关系判断，产出 checking_outputs.json。

**这个脚本要在独立的 `.venv-refchecker` 环境里运行，不是主 venv。**
refchecker 的依赖很重（torch / spacy / litellm / pytorch_lightning），刻意与
主环境分开，避免顶掉主环境里的 numpy 与 scikit-learn。

    .venv-refchecker\\Scripts\\python.exe scripts/run_refchecker_nli.py \
        --input outputs/ragchecker/test_checking_inputs.json \
        --output outputs/ragchecker/test_checking_outputs.json

为什么不用 ragchecker-cli
-------------------------
``ragchecker-cli`` 必须先用一个 **LLM extractor** 从 response 里抽 claim，
而本数据集的 claim 是给定的（每个样本恰好一条，``answer`` 就是 claim 原文）。
于是这里跳过抽取环节，直接把给定的 claim 交给 checker：

* 不需要任何 LLM 凭证，也不花钱，完全离线（模型下载后）；
* claim 对齐天然是 1:1，不会出现 extractor 把一句话拆成 N 条的错位风险。

判断环节与 RAGChecker 完全一致：同一个 ``checker.check(...)`` 调用，
``merge_psg=False``，产出形状为 ``[claim_num][doc_num]`` 的 ``retrieved2response``。

诚实边界
--------
输出里 ``response_claims`` 是**数据集给定的 claim**，不是模型抽取的结果；
``extractor`` 记为 ``None``。论文里必须写成「RefChecker NLI checker，
claim 由数据集给定、未经 LLM 抽取」，不能写成「RAGChecker 全流程」。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

#: RefChecker NLIChecker 的默认模型（见 refchecker/checker/nli_checker.py）。
DEFAULT_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
VALID_LABELS = {"Entailment", "Neutral", "Contradiction"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="用 RefChecker 的 NLI checker 产出 checking_outputs.json"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="export_ragchecker_input.py 的输出"
    )
    parser.add_argument("--output", type=Path, required=True, help="结果 JSON 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NLI 模型名")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto / cpu / cuda:0 等；auto 表示有 GPU 用 GPU，否则用 CPU",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="只跑前 N 条样本（冒烟测试）"
    )
    parser.add_argument(
        "--emit-probabilities",
        action="store_true",
        help=(
            "额外写出每个 (claim, document) 的三类 softmax 概率。"
            "RefChecker 在 nli_checker.py 里算完 softmax 又用 argmax 丢掉了，"
            "而这三类正好对应 (p_support, p_unknown, p_refute)；"
            "保留连续值可以避免把诊断空间压成有限格点"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


#: RefChecker 在 nli_checker.py 里**硬编码**的类别顺序。
#:
#: .. danger::
#:     ``NLIChecker._check`` 直接用 ``LABELS[argmax]`` 取标签，完全没有查过模型
#:     自己的 ``id2label``。而不同 NLI 模型的类别顺序并不一致，实测：
#:
#:     * ``ynie/roberta-large-...-nli``      ``[entailment, neutral, contradiction]``
#:     * ``microsoft/deberta-large-mnli``    ``[CONTRADICTION, NEUTRAL, ENTAILMENT]``
#:     * ``cross-encoder/nli-deberta-v3-base`` ``[contradiction, entailment, neutral]``
#:
#:     也就是说，把默认模型以外的模型交给 RefChecker 的 NLIChecker，
#:     entailment 与 contradiction 会被**静默对调**，不报错、数字照出。
#:     本脚本因此一律以模型 ``config.id2label`` 为准重新取标签。
REFCHECKER_ASSUMED_ORDER = ("entailment", "neutral", "contradiction")

#: 本项目输出文件里概率三元组的固定顺序。
CANONICAL_ORDER = ("entailment", "neutral", "contradiction")

#: 把各家写法归一到本项目的三个标签名。
_LABEL_ALIASES = {
    "entailment": "entailment",
    "entail": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradict": "contradiction",
}

#: 输出文件里使用的首字母大写形式，与 RefChecker 的写法保持一致。
_DISPLAY = {
    "entailment": "Entailment",
    "neutral": "Neutral",
    "contradiction": "Contradiction",
}


def _model_label_order(checker) -> tuple[str, ...]:
    """读取模型自己的 ``id2label``，归一成本项目的标签名。

    Raises:
        ValueError: 模型不是三分类，或标签名无法识别。
    """
    id2label = getattr(checker.model.config, "id2label", None)
    if not id2label or len(id2label) != 3:
        raise ValueError(
            f"模型的 id2label 不是三分类：{id2label!r}；无法确认类别顺序"
        )
    order: list[str] = []
    for index in range(3):
        raw = id2label.get(index, id2label.get(str(index)))
        if raw is None:
            raise ValueError(f"模型的 id2label 缺少下标 {index}：{id2label!r}")
        key = str(raw).strip().lower().replace("-", "_")
        canonical = _LABEL_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"无法识别的 NLI 标签 {raw!r}；已知写法：{sorted(_LABEL_ALIASES)}"
            )
        order.append(canonical)
    if len(set(order)) != 3:
        raise ValueError(f"模型的三个标签有重复：{order}")
    return tuple(order)


def _softmax_probabilities(
    checker, pairs: list[tuple[str, str]], batch_size: int
) -> list[tuple[float, float, float]]:
    """复用 checker 已加载的模型，算出每个 (reference, claim) 对的三类概率。

    刻意复用 ``checker.model`` / ``checker.tokenizer`` 而不是另外加载一份：
    同一份权重、同样的 tokenize 方式，才能保证概率与 ``check()`` 给出的标签
    严格对应（调用方会逐条断言 argmax 等于标签）。

    ``nli_checker.py`` 里的写法是 ``tokenizer(batch_references, batch_claims)``，
    即 reference 作前提、claim 作假设，这里必须保持同样的顺序。
    """
    import torch

    out: list[tuple[float, float, float]] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            inputs = checker.tokenizer(
                [reference for reference, _ in chunk],
                [claim for _, claim in chunk],
                max_length=512,
                truncation=True,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=True,
            )
            inputs = {k: v.to(checker.device) for k, v in inputs.items()}
            probabilities = checker.model(**inputs).logits.softmax(dim=-1).cpu()
            out.extend(
                (float(row[0]), float(row[1]), float(row[2])) for row in probabilities
            )
    return out


def _resolve_device(requested: str) -> int | str:
    """把 --device 解析成 NLIChecker 接受的取值。"""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return 0
    return "cpu"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _normalise_row(raw: object, doc_count: int, query_id: str) -> list[str]:
    """把 checker 返回的一行整理成 doc_count 个标签。"""
    if isinstance(raw, str):
        row = [raw]
    elif isinstance(raw, list):
        row = []
        for item in raw:
            if isinstance(item, list):
                # max_reference_segment_length=0 时每段文档不再切分，
                # 因此这里只应有一个标签。多于一个说明切分被启用了，
                # 静默取第一个会丢掉其余段的判断，宁可报错。
                if len(item) != 1:
                    raise ValueError(
                        f"{query_id}: 某段文档返回了 {len(item)} 个标签，"
                        "预期只有 1 个；请确认 max_reference_segment_length=0"
                    )
                row.append(item[0])
            else:
                row.append(item)
    else:
        raise ValueError(f"{query_id}: checker 返回了无法解析的结果 {raw!r}")

    if len(row) != doc_count:
        raise ValueError(
            f"{query_id}: checker 返回 {len(row)} 个标签，但检索文档有 {doc_count} 段；"
            "形状必须是 [claim_num][doc_num]"
        )
    for label in row:
        if label not in VALID_LABELS:
            raise ValueError(f"{query_id}: checker 返回了未知标签 {label!r}")
    return list(row)


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    if args.output.exists() and not args.overwrite:
        print(f"[输出文件已存在] {args.output}", file=sys.stderr)
        print("提示：加 --overwrite 可覆盖。", file=sys.stderr)
        return 1

    try:
        from refchecker.checker import NLIChecker
    except ImportError as error:
        print(f"[环境错误] 无法导入 refchecker：{error}", file=sys.stderr)
        print("本脚本必须在 .venv-refchecker 环境里运行。", file=sys.stderr)
        return 1

    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    results = payload["results"]
    if args.limit is not None:
        results = results[: args.limit]
    if not results:
        print("[输入错误] 没有任何样本", file=sys.stderr)
        return 1

    # claim 由数据集给定：每个样本恰好一条，response 就是 claim 原文。
    batch_claims = [[item["response"]] for item in results]
    batch_references = [
        [chunk["text"] for chunk in item["retrieved_context"]] for item in results
    ]
    batch_questions = [item["query"] for item in results]
    pair_count = sum(len(refs) for refs in batch_references)

    device = _resolve_device(args.device)
    print(f"模型   {args.model}")
    print(f"设备   {device}")
    print(f"样本   {len(results)} 条，判断 {pair_count} 个 (claim, document) 组合")
    print("首次运行会下载模型（roberta-large 量级，约 1.4 GB），请耐心等待……")

    load_started = time.perf_counter()
    checker = NLIChecker(model=args.model, device=device, batch_size=args.batch_size)
    load_elapsed = time.perf_counter() - load_started

    # 只计推理时间：首次运行的模型下载动辄十分钟，混进来会让速率失真。
    started = time.perf_counter()
    checking_results = checker.check(
        batch_claims=batch_claims,
        batch_references=batch_references,
        batch_questions=batch_questions,
        max_reference_segment_length=0,
        # 与 RAGChecker 计算 retrieved2response 时的调用保持一致：
        # 每段文档单独判断，不合并成一整段参考。
        merge_psg=False,
        is_joint=False,
    )
    elapsed = time.perf_counter() - started

    if len(checking_results) != len(results):
        print(
            f"[结果错位] checker 返回 {len(checking_results)} 条，输入 {len(results)} 条",
            file=sys.stderr,
        )
        return 1

    # 一律以模型自己的 id2label 为准重新取标签：RefChecker 硬编码的顺序
    # 只对它的默认模型成立，换模型会静默把 entailment 和 contradiction 对调。
    try:
        model_order = _model_label_order(checker)
    except ValueError as error:
        print(f"[模型标签顺序无法确认] {error}", file=sys.stderr)
        return 1
    order_matches = model_order == REFCHECKER_ASSUMED_ORDER
    print(f"模型类别顺序 {model_order}")
    if order_matches:
        print("  与 RefChecker 硬编码的顺序一致，其 check() 标签可直接采信。")
    else:
        print("  ⚠ 与 RefChecker 硬编码的顺序 " f"{REFCHECKER_ASSUMED_ORDER} 不一致！")
        print("  RefChecker 的 check() 标签在这个模型上是错的，已按模型 config 重算。")

    pairs = [
        (chunk["text"], item["response"])
        for item in results
        for chunk in item["retrieved_context"]
    ]
    probabilities = _softmax_probabilities(checker, pairs, args.batch_size)
    # 把概率重排成本项目的固定顺序，下游无需再关心模型内部顺序。
    index_of = {label: position for position, label in enumerate(model_order)}
    canonical = [
        tuple(triple[index_of[label]] for label in CANONICAL_ORDER)
        for triple in probabilities
    ]

    label_counts: dict[str, int] = {label: 0 for label in sorted(VALID_LABELS)}
    output_results = []
    cursor = 0
    for item, raw in zip(results, checking_results):
        doc_count = len(item["retrieved_context"])
        chunk_probs = canonical[cursor : cursor + doc_count]
        # 标签由概率按模型真实顺序导出，而不是采信 check() 的返回值。
        matrix = [
            [
                _DISPLAY[CANONICAL_ORDER[max(range(3), key=lambda i: triple[i])]]
                for triple in chunk_probs
            ]
        ]
        if order_matches:
            try:
                from_check = [
                    _normalise_row(row, doc_count, item["query_id"]) for row in raw
                ]
            except ValueError as error:
                print(f"[结果错位] {error}", file=sys.stderr)
                return 1
            if from_check[0] != matrix[0]:
                print(
                    f"[概率与标签不一致] {item['query_id']}："
                    f"check() 返回 {from_check[0]}，按概率导出为 {matrix[0]}",
                    file=sys.stderr,
                )
                return 1
        for row in matrix:
            for label in row:
                label_counts[label] += 1

        record = {
            **item,
            # claim 是数据集给定的，不是 extractor 抽的。
            "response_claims": [[item["response"]]],
            "retrieved2response": matrix,
        }
        if args.emit_probabilities:
            # 外层 claim、内层文档，与 retrieved2response 形状一致；
            # 每个三元组按 CANONICAL_ORDER 排列。
            record["retrieved2response_probabilities"] = [
                [list(triple) for triple in chunk_probs]
            ]
        cursor += doc_count
        output_results.append(record)

    if cursor != len(canonical):
        print(
            f"[结果错位] 概率条数 {len(canonical)} 与消费掉的 {cursor} 不符",
            file=sys.stderr,
        )
        return 1

    _atomic_write_json(
        args.output,
        {
            "results": output_results,
            "rag_ds_run_info": {
                "producer": "refchecker-nli",
                "checker_model": args.model,
                "extractor": None,
                "claims_source": "dataset_given",
                "device": str(device),
                "inference_seconds": round(elapsed, 1),
                "model_load_seconds": round(load_elapsed, 1),
                "has_probabilities": args.emit_probabilities,
                "probability_label_order": [_DISPLAY[x] for x in CANONICAL_ORDER],
                # 模型自报的类别顺序，以及它是否与 RefChecker 的硬编码一致。
                "model_label_order": list(model_order),
                "refchecker_order_matches": order_matches,
            },
        },
    )

    print()
    print(f"模型加载/下载 {load_elapsed:.1f} 秒")
    print(f"推理 {elapsed:.1f} 秒（{pair_count / max(elapsed, 1e-9):.1f} 个判断/秒）")
    print("标签分布：")
    for label, count in label_counts.items():
        print(f"  {label:<14}{count:>6}  ({count / pair_count:.1%})")
    print()
    print(f"输出 {args.output}")
    print()
    print("下一步（回到主 venv）：")
    print("  python scripts/calibrate_label_mapping.py --outputs <validation 输出>")
    print("  python scripts/ingest_ragchecker.py --split <split> --outputs <输出> \\")
    print(f"    --calibration <校准文件> --checker-name {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
