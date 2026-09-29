"""flan-t5 评判器：与 NLI 模型机制异构的第三类评估器（生成式语言模型）。

为什么是它
----------
前两个评估器（ynie roberta-large、MoritzLaurer DeBERTa-v3）都是**判别式
NLI 分类头**，训练数据同源（MNLI/FEVER/ANLI 家族），错误模式高度相关——
双评估器实验里 K_eval 告警只占 3.7% 就是证据。flan-t5-base 是指令微调的
**生成式** T5，靠语言建模似然作答，机制与训练分布都不同，错误模式应更独立，
这才给 K_eval（评估器间冲突检测）与可靠性折扣提供真正的用武之地。

打分方式（不生成文本，用似然）
------------------------------
对每个 (claim, document) 组合，把文档当前提、claim 当假设拼成
``nli premise: ... hypothesis: ...``，分别计算三个候选标签
``entailment`` / ``neutral`` / ``contradiction`` 在解码器下的对数似然
（teacher forcing，取候选词全部 token 的对数概率之和），softmax 后得到
``(p_support, p_refute, p_unknown)``。这样输出天然归一，满足
``RelationPrediction`` 的约束，也避免解析自由文本的脆弱性。

分块断点续跑
------------
与 ``run_refchecker_nli_chunked.py`` 同一套模式：每轮在时间预算内处理若干
样本，进度写 ``<output>.partial.jsonl``；全部完成后原子改名成最终文件。
退出码：0 = 完成；2 = 有进度未完成（再调用继续）；1 = 出错。

用法（.venv-refchecker 环境，HF_HOME 指向模型缓存）::

    set HF_HOME=.model-cache&& set HF_HUB_OFFLINE=1&& \
    .venv-refchecker\\Scripts\\python.exe scripts/run_t5_judge_chunked.py \
        --samples data/processed/climate_fever_v2/train.jsonl \
        --output outputs/ragchecker_v2/train_relations_flan_t5_judge.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

#: 退出码：有进度但样本未处理完。
EXIT_INCOMPLETE = 2

EVALUATOR_NAME = "flan_t5_judge"
DEFAULT_MODEL = "google/flan-t5-base"
#: (候选标签文本, 对应概率字段)
CANDIDATES = (
    ("entailment", "p_support"),
    ("contradiction", "p_refute"),
    ("neutral", "p_unknown"),
)
#: 编码器输入最大 token 数，超出截断（文档侧优先截断）。
MAX_INPUT_TOKENS = 448


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--seconds", type=float, default=250.0,
        help="本轮推理时间预算（不含模型加载），到点存盘退出",
    )
    parser.add_argument(
        "--chunk-samples", type=int, default=10,
        help="每个处理块的样本数；块间检查预算并存盘",
    )
    return parser.parse_args(argv)


def _partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial.jsonl")


def _load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_candidates(model, tokenizer, premise: str, hypothesis: str) -> dict[str, float]:
    """返回三个候选标签 softmax 后的概率字典。"""
    import torch

    prompt = f"nli premise: {premise} hypothesis: {hypothesis}"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
    )
    logps: list[float] = []
    for label_text, _ in CANDIDATES:
        label_ids = tokenizer(label_text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        # teacher forcing：decoder 输入右移由模型内部完成，labels 提供目标
        outputs = model(**inputs, labels=label_ids)
        # loss 是逐 token 平均的负对数似然，乘回 token 数得到总和
        logps.append(-float(outputs.loss) * label_ids.shape[1])
    top = max(logps)
    exps = [math.exp(x - top) for x in logps]
    total = sum(exps)
    return {
        field: exps[i] / total for i, (_, field) in enumerate(CANDIDATES)
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples = _load_samples(args.samples)
    partial = _partial_path(args.output)

    done_ids: set[str] = set()
    records: list[dict] = []
    if partial.exists():
        with partial.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
                    done_ids.add(record["sample_id"])
    elif args.output.exists():
        print("最终输出已存在，无事可做。")
        return 0

    remaining = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"总样本 {len(samples)}，已完成 {len(done_ids)}，本轮待处理 {len(remaining)}")
    if not remaining:
        os.replace(partial, args.output)
        print("已全部完成，写出最终文件。")
        return 0

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()

    start = time.monotonic()
    import torch

    with torch.no_grad():
        for offset in range(0, len(remaining), args.chunk_samples):
            chunk = remaining[offset : offset + args.chunk_samples]
            for sample in chunk:
                for claim in sample.get("claims", []):
                    for context in sample.get("contexts", []):
                        probs = _score_candidates(
                            model, tokenizer, context["text"], claim["text"]
                        )
                        records.append(
                            {
                                "sample_id": sample["sample_id"],
                                "claim_id": claim["claim_id"],
                                "doc_id": context["doc_id"],
                                "evaluator": EVALUATOR_NAME,
                                "p_support": probs["p_support"],
                                "p_refute": probs["p_refute"],
                                "p_unknown": probs["p_unknown"],
                                "evaluator_reliability": 1.0,
                            }
                        )
                done_ids.add(sample["sample_id"])
            # 块间存盘（JSONL 直接整写，量小可接受）
            with partial.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            elapsed = time.monotonic() - start
            done_count = len(samples) - len(remaining) + min(
                offset + args.chunk_samples, len(remaining)
            )
            print(f"  进度 {done_count}/{len(samples)}，本轮已用 {elapsed:.0f}s")
            if elapsed > args.seconds:
                print("时间预算用完，存盘退出。")
                return EXIT_INCOMPLETE

    os.replace(partial, args.output)
    print(f"全部完成，写出 {args.output}（{len(records)} 条关系预测）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
