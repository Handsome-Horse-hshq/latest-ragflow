"""分块断点续跑 RefChecker NLI 推理：为 CPU 环境下的长时间运行设计。

``run_refchecker_nli.py`` 一次跑完整个输入文件；v2 全量有 616 条样本、
3080 个 (claim, document) 组合，CPU 上远超单次可调用的时限。本脚本把同一
套逻辑切成多轮：每轮在时间预算内处理若干样本，把已完成记录原子写入
``<output>.partial.json``；下一轮自动从断点继续，直到全部完成后才写出与
``run_refchecker_nli.py`` 完全同形的最终输出并删除断点文件。

用法（在 .venv-refchecker 环境里，需要 HF_HOME 指向模型缓存）::

    set HF_HOME=.model-cache
    .venv-refchecker\\Scripts\\python.exe scripts/run_refchecker_nli_chunked.py \
        --input outputs/ragchecker_v2/train_checking_inputs.json \
        --output outputs/ragchecker_v2/train_checking_outputs_probs.json \
        --emit-probabilities

退出码：0 = 全部完成；2 = 有进度但未完成（再调用一次继续）；1 = 出错。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from run_refchecker_nli import (
    CANONICAL_ORDER,
    DEFAULT_MODEL,
    REFCHECKER_ASSUMED_ORDER,
    _DISPLAY,
    _atomic_write_json,
    _model_label_order,
    _normalise_row,
    _resolve_device,
    _softmax_probabilities,
)

#: 退出码：有进度但输入未处理完。
EXIT_INCOMPLETE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--seconds",
        type=float,
        default=260.0,
        help="本轮推理的时间预算（不含模型加载），到点存盘退出",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=20,
        help="每个处理块的样本数；块与块之间检查时间预算并存盘",
    )
    parser.add_argument("--emit-probabilities", action="store_true")
    return parser.parse_args(argv)


def _partial_path(output: Path) -> Path:
    """断点文件路径。"""
    return output.with_name(output.name + ".partial.json")


def _load_partial(path: Path) -> dict:
    """读取断点；不存在则返回初始状态。"""
    if not path.exists():
        return {
            "done_records": [],
            "label_counts": {"Contradiction": 0, "Entailment": 0, "Neutral": 0},
            "inference_seconds": 0.0,
            "load_seconds": 0.0,
        }
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _process_chunk(
    checker,
    chunk: list[dict],
    args: argparse.Namespace,
    model_order: tuple[str, ...],
    order_matches: bool,
) -> tuple[list[dict], dict[str, int], float]:
    """处理一块样本，返回 (记录, 本块标签计数, 本块推理秒数)。"""
    batch_claims = [[item["response"]] for item in chunk]
    batch_references = [
        [ctx["text"] for ctx in item["retrieved_context"]] for item in chunk
    ]
    batch_questions = [item["query"] for item in chunk]

    started = time.perf_counter()
    checking_results = checker.check(
        batch_claims=batch_claims,
        batch_references=batch_references,
        batch_questions=batch_questions,
        max_reference_segment_length=0,
        merge_psg=False,
        is_joint=False,
    )
    pairs = [
        (ctx["text"], item["response"])
        for item in chunk
        for ctx in item["retrieved_context"]
    ]
    probabilities = _softmax_probabilities(checker, pairs, args.batch_size)
    elapsed = time.perf_counter() - started

    if len(checking_results) != len(chunk):
        raise RuntimeError(
            f"checker 返回 {len(checking_results)} 条，本块输入 {len(chunk)} 条"
        )

    index_of = {label: position for position, label in enumerate(model_order)}
    canonical = [
        tuple(triple[index_of[label]] for label in CANONICAL_ORDER)
        for triple in probabilities
    ]

    label_counts: dict[str, int] = {"Contradiction": 0, "Entailment": 0, "Neutral": 0}
    records = []
    cursor = 0
    for item, raw in zip(chunk, checking_results):
        doc_count = len(item["retrieved_context"])
        chunk_probs = canonical[cursor : cursor + doc_count]
        matrix = [
            [
                _DISPLAY[CANONICAL_ORDER[max(range(3), key=lambda i: triple[i])]]
                for triple in chunk_probs
            ]
        ]
        if order_matches:
            from_check = [_normalise_row(row, doc_count, item["query_id"]) for row in raw]
            if from_check[0] != matrix[0]:
                raise RuntimeError(
                    f"概率与标签不一致 {item['query_id']}："
                    f"check() 返回 {from_check[0]}，按概率导出为 {matrix[0]}"
                )
        for row in matrix:
            for label in row:
                label_counts[label] += 1

        record = {
            **item,
            "response_claims": [[item["response"]]],
            "retrieved2response": matrix,
        }
        if args.emit_probabilities:
            record["retrieved2response_probabilities"] = [
                [list(triple) for triple in chunk_probs]
            ]
        cursor += doc_count
        records.append(record)

    return records, label_counts, elapsed


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    if args.output.exists():
        print(f"[已完成] 最终输出已存在：{args.output}")
        return 0

    try:
        from refchecker.checker import NLIChecker
    except ImportError as error:
        print(f"[环境错误] 无法导入 refchecker：{error}", file=sys.stderr)
        return 1

    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    results = payload["results"]
    total = len(results)

    partial_file = _partial_path(args.output)
    state = _load_partial(partial_file)
    done = len(state["done_records"])
    if done >= total:
        print("[状态错误] 断点记录数已超过输入样本数，请删除断点文件重跑", file=sys.stderr)
        return 1

    device = _resolve_device(args.device)
    print(f"模型   {args.model}")
    print(f"设备   {device}")
    print(f"进度   {done}/{total} 条已完成，本轮从第 {done + 1} 条继续")

    load_started = time.perf_counter()
    checker = NLIChecker(model=args.model, device=device, batch_size=args.batch_size)
    state["load_seconds"] += time.perf_counter() - load_started

    try:
        model_order = _model_label_order(checker)
    except ValueError as error:
        print(f"[模型标签顺序无法确认] {error}", file=sys.stderr)
        return 1
    order_matches = model_order == REFCHECKER_ASSUMED_ORDER
    print(f"模型类别顺序 {model_order}（与 RefChecker 硬编码一致：{order_matches}）")

    budget_start = time.perf_counter()
    position = done
    while position < total:
        if time.perf_counter() - budget_start > args.seconds:
            break
        chunk = results[position : position + args.chunk_samples]
        records, chunk_counts, elapsed = _process_chunk(
            checker, chunk, args, model_order, order_matches
        )
        state["done_records"].extend(records)
        for label, count in chunk_counts.items():
            state["label_counts"][label] += count
        state["inference_seconds"] += elapsed
        position += len(chunk)
        _atomic_write_json(partial_file, state)
        rate = len(chunk) / max(elapsed, 1e-9)
        print(
            f"  已处理 {position}/{total} 条"
            f"（本块 {elapsed:.1f} 秒，{rate:.2f} 样本/秒）",
            flush=True,
        )

    if position < total:
        print(f"[未完成] 本轮结束于 {position}/{total}，断点已保存：{partial_file}")
        print("再调用一次本脚本即可继续。")
        return EXIT_INCOMPLETE

    pair_count = sum(state["label_counts"].values())
    _atomic_write_json(
        args.output,
        {
            "results": state["done_records"],
            "rag_ds_run_info": {
                "producer": "refchecker-nli",
                "checker_model": args.model,
                "extractor": None,
                "claims_source": "dataset_given",
                "device": str(device),
                "inference_seconds": round(state["inference_seconds"], 1),
                "model_load_seconds": round(state["load_seconds"], 1),
                "has_probabilities": args.emit_probabilities,
                "probability_label_order": [_DISPLAY[x] for x in CANONICAL_ORDER],
                "model_label_order": list(model_order),
                "refchecker_order_matches": order_matches,
                "chunked_runner": True,
            },
        },
    )
    partial_file.unlink(missing_ok=True)

    print()
    print(f"全部完成：{total} 条样本，{pair_count} 个判断")
    print(f"累计推理 {state['inference_seconds']:.1f} 秒")
    print("标签分布：")
    for label in sorted(state["label_counts"]):
        count = state["label_counts"][label]
        print(f"  {label:<14}{count:>6}  ({count / max(pair_count, 1):.1%})")
    print(f"输出 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
