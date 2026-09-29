"""flan-t5 评判器：与 NLI 分类头机制异构的第三类评估器（生成式语言模型）。

为什么是它
----------
前两个评估器（ynie roberta-large、MoritzLaurer DeBERTa-v3）都是**判别式
NLI 分类头**，训练数据同源（MNLI/FEVER/ANLI 家族），错误模式高度相关——
两者的 Cohen's kappa = 0.623，双评估器实验里 K_eval 告警只占 3.7%。
Dempster 组合规则要求证据源相互独立，同源评估器会重复计数同一份证据，
这是双评估器反而不如单评估器的原因。flan-t5-base 是指令微调的**生成式**
T5，机制与训练分布都不同，用来检验「异构评估器能否带来独立证据」。

打分方式：FLAN 原生 MNLI 模板 + 选项首 token
--------------------------------------------
对每个 (claim, document) 组合，把文档当前提、claim 当假设，套 FLAN 训练时
用过的 MNLI 指令模板::

    Premise: {premise}
    Hypothesis: {hypothesis}
    Does the premise entail the hypothesis?

    OPTIONS:
    - yes
    - it is not possible to tell
    - no

只做一次前向：编码器读提示，解码器只喂起始符，取第一步输出在三个选项
**首 token**（``▁yes`` / ``▁it`` / ``▁no``，在 flan-t5 分词器下各为单个 token）
上的 logit，softmax 得到 ``(p_support, p_unknown, p_refute)``。

.. danger:: 早期版本的两个缺陷（其输出已作废）

    1. **长度偏置**：对候选词 ``entailment`` / ``neutral`` / ``contradiction``
       的各 token 对数概率**求和**。``entailment`` 在该分词器下是 4 个 token，
       另两个各 1 个，求和使 entailment 在结构上几乎不可能胜出；
    2. **提示格式**：``nli premise: ... hypothesis: ...`` 不是 FLAN 的原生模板，
       模型面对它默认回答 neutral。

    实测 validation 620 条：99.8% 判 unknown、0% 判 support，与人工 oracle 的
    Cohen's kappa = +0.002（随机水平）。只把求和改成求平均并不能修好——
    在 200 条子集上它会从「全判 neutral」翻成「全判 support」，kappa 仍为 0；
    换成原生模板后 kappa = +0.093，support / refute 与 oracle 的相关系数
    分别为 +0.21 / +0.27，与 roberta-large 同量级。
    作废输出保存在 ``outputs/ragchecker_v2/_invalid_flan_t5_sum_logprob/``。

截断：只截前提，永不截选项
--------------------------
提示超过 ``MAX_INPUT_TOKENS`` 时，只从**前提文本**的尾部截断，模板与选项
完整保留 —— 选项被截掉等于让模型在不知道选项的情况下作答。v2 数据上最长的
提示是 439 token，目前不会触发截断。

分块断点续跑
------------
每轮在时间预算内处理若干样本，进度写 ``<output>.partial.jsonl``；全部完成后
原子改名成最终文件。续跑时会校验已有记录的评估器名与打分版本，**与当前版本
不一致就拒绝续跑**，防止新旧两种打分混进同一个文件。

退出码：0 = 完成；2 = 有进度未完成（再调用继续）；1 = 出错。

用法（.venv-refchecker 环境，HF_HOME 指向模型缓存）::

    set HF_HOME=.model-cache&& set HF_HUB_OFFLINE=1&& \
    .venv-refchecker\\Scripts\\python.exe scripts/run_t5_judge_chunked.py \
        --samples data/processed/climate_fever_v2/train.jsonl \
        --output outputs/ragchecker_v2/train_relations_flan_t5_mnli.jsonl
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

#: 评估器名。刻意与作废的 ``flan_t5_judge`` 不同，两者绝不能混在一起。
EVALUATOR_NAME = "flan_t5_mnli"
#: 打分方式的版本号，写进每条记录与 run-info，续跑时用来校验。
SCORING_VERSION = "flan-mnli-template/first-token/v1"
DEFAULT_MODEL = "google/flan-t5-base"

#: FLAN 的 MNLI 指令模板。选项顺序与 FLAN 训练时一致。
PROMPT_HEAD = "Premise: {premise}\nHypothesis: {hypothesis}\n"
PROMPT_TAIL = (
    "Does the premise entail the hypothesis?\n\n"
    "OPTIONS:\n- yes\n- it is not possible to tell\n- no"
)

#: (选项首词, 对应概率字段)。每个首词在 flan-t5 分词器下都必须是单个 token。
OPTION_FIRST_WORDS = (
    ("yes", "p_support"),
    ("no", "p_refute"),
    ("it", "p_unknown"),
)

#: 编码器输入最大 token 数；超出时只截前提，模板与选项完整保留。
MAX_INPUT_TOKENS = 448


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="flan-t5 评判器（FLAN MNLI 模板）")
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
    parser.add_argument(
        "--batch-size", type=int, default=16, help="每次前向的 (claim, doc) 组合数"
    )
    return parser.parse_args(argv)


def _partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial.jsonl")


def _run_info_path(output: Path) -> Path:
    return output.with_name(output.name + ".run_info.json")


def _load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _option_token_ids(tokenizer) -> list[int]:
    """取三个选项首词的 token id，并确认各自恰好是单个 token。"""
    ids: list[int] = []
    for word, _ in OPTION_FIRST_WORDS:
        pieces = tokenizer(word, add_special_tokens=False).input_ids
        if len(pieces) != 1:
            raise ValueError(
                f"选项首词 {word!r} 在该分词器下是 {len(pieces)} 个 token；"
                "首 token 打分要求每个选项首词恰好是单个 token"
            )
        ids.append(pieces[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"三个选项首词的 token id 有重复：{ids}")
    return ids


def build_prompt(tokenizer, premise: str, hypothesis: str) -> str:
    """套 FLAN MNLI 模板；超长时只截前提，模板与选项完整保留。"""
    prompt = PROMPT_HEAD.format(premise=premise, hypothesis=hypothesis) + PROMPT_TAIL
    if len(tokenizer(prompt).input_ids) <= MAX_INPUT_TOKENS:
        return prompt

    fixed = PROMPT_HEAD.format(premise="", hypothesis=hypothesis) + PROMPT_TAIL
    budget = MAX_INPUT_TOKENS - len(tokenizer(fixed).input_ids)
    if budget <= 0:
        raise ValueError(
            "假设文本与模板本身已超过输入上限，无法在保留选项的前提下截断"
        )
    premise_ids = tokenizer(premise, add_special_tokens=False).input_ids[:budget]
    truncated = tokenizer.decode(premise_ids, skip_special_tokens=True)
    return PROMPT_HEAD.format(premise=truncated, hypothesis=hypothesis) + PROMPT_TAIL


def score_batch(model, tokenizer, option_ids, pairs) -> list[dict[str, float]]:
    """对一批 (premise, hypothesis) 做一次前向，返回每条的三个概率。"""
    import torch

    prompts = [build_prompt(tokenizer, p, h) for p, h in pairs]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    start = torch.full(
        (len(prompts), 1), model.config.decoder_start_token_id, dtype=torch.long
    )
    logits = model(**encoded, decoder_input_ids=start).logits[:, -1, :]
    picked = logits[:, option_ids].double()
    probabilities = torch.softmax(picked, dim=-1).tolist()
    return [
        {field: row[i] for i, (_, field) in enumerate(OPTION_FIRST_WORDS)}
        for row in probabilities
    ]


def _check_resumable(records: list[dict], output: Path) -> None:
    """已有进度必须来自同一个评估器与打分版本，否则拒绝续跑。"""
    foreign = {
        (r.get("evaluator"), r.get("scoring_version"))
        for r in records
        if r.get("evaluator") != EVALUATOR_NAME
        or r.get("scoring_version") != SCORING_VERSION
    }
    if foreign:
        raise ValueError(
            f"{output} 的已有进度来自别的评估器/打分版本 {sorted(foreign)}，"
            f"当前为 ({EVALUATOR_NAME!r}, {SCORING_VERSION!r})；"
            "新旧打分不能混进同一个文件，请先移走旧文件"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples = _load_samples(args.samples)
    partial = _partial_path(args.output)

    if args.output.exists():
        # 最终文件不带 scoring_version（RelationPrediction 禁止多余字段），
        # 版本信息在同名 run_info 里。
        info_path = _run_info_path(args.output)
        info = (
            json.loads(info_path.read_text(encoding="utf-8"))
            if info_path.exists()
            else {}
        )
        if (
            info.get("evaluator") != EVALUATOR_NAME
            or info.get("scoring_version") != SCORING_VERSION
        ):
            print(
                f"[拒绝] {args.output} 已存在，但其 run_info 记录的版本为 "
                f"({info.get('evaluator')!r}, {info.get('scoring_version')!r})，"
                f"与当前 ({EVALUATOR_NAME!r}, {SCORING_VERSION!r}) 不一致；"
                "请先移走旧文件",
                file=sys.stderr,
            )
            return 1
        print("最终输出已存在且版本一致，无事可做。")
        return 0

    done_ids: set[str] = set()
    records: list[dict] = []
    if partial.exists():
        records = _load_samples(partial)
        try:
            _check_resumable(records, partial)
        except ValueError as error:
            print(f"[拒绝续跑] {error}", file=sys.stderr)
            return 1
        done_ids = {r["sample_id"] for r in records}

    remaining = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"总样本 {len(samples)}，已完成 {len(done_ids)}，本轮待处理 {len(remaining)}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    option_ids = _option_token_ids(tokenizer)

    if remaining:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        model.eval()
        import torch

        start = time.monotonic()
        with torch.no_grad():
            for offset in range(0, len(remaining), args.chunk_samples):
                chunk = remaining[offset : offset + args.chunk_samples]
                keys, pairs = [], []
                for sample in chunk:
                    for claim in sample.get("claims", []):
                        for context in sample.get("contexts", []):
                            keys.append(
                                (sample["sample_id"], claim["claim_id"], context["doc_id"])
                            )
                            pairs.append((context["text"], claim["text"]))
                scored: list[dict[str, float]] = []
                for b in range(0, len(pairs), args.batch_size):
                    scored.extend(
                        score_batch(
                            model, tokenizer, option_ids, pairs[b : b + args.batch_size]
                        )
                    )
                for (sample_id, claim_id, doc_id), probs in zip(keys, scored):
                    records.append(
                        {
                            "sample_id": sample_id,
                            "claim_id": claim_id,
                            "doc_id": doc_id,
                            "evaluator": EVALUATOR_NAME,
                            "p_support": probs["p_support"],
                            "p_refute": probs["p_refute"],
                            "p_unknown": probs["p_unknown"],
                            "evaluator_reliability": 1.0,
                            "scoring_version": SCORING_VERSION,
                        }
                    )
                with partial.open("w", encoding="utf-8", newline="\n") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                elapsed = time.monotonic() - start
                done_count = len(samples) - len(remaining) + min(
                    offset + args.chunk_samples, len(remaining)
                )
                print(f"  进度 {done_count}/{len(samples)}，本轮已用 {elapsed:.0f}s")
                if elapsed > args.seconds and offset + args.chunk_samples < len(remaining):
                    print("时间预算用完，存盘退出。")
                    return EXIT_INCOMPLETE

    # RelationPrediction 不允许多余字段：最终文件去掉 scoring_version，
    # 版本信息转存到同名 run_info 文件里。
    final_records = [
        {k: v for k, v in r.items() if k != "scoring_version"} for r in records
    ]
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in final_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _run_info_path(args.output).write_text(
        json.dumps(
            {
                "producer": "flan-t5-judge",
                "evaluator": EVALUATOR_NAME,
                "scoring_version": SCORING_VERSION,
                "checker_model": args.model,
                "prompt_template": PROMPT_HEAD + PROMPT_TAIL,
                "option_first_words": [w for w, _ in OPTION_FIRST_WORDS],
                "option_token_ids": option_ids,
                "max_input_tokens": MAX_INPUT_TOKENS,
                "records": len(final_records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    partial.unlink(missing_ok=True)
    print(f"全部完成，写出 {args.output}（{len(final_records)} 条关系预测）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
