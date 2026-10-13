"""用大模型当关系评估器：逐个 (claim, 文档) 组合给出三元概率。

为什么要跑
----------
前面的实验反复指向同一件事：瓶颈在关系评估器。NLI 模型在 CLIMATE-FEVER 上
只有约 3.5% 的组合判成 Entailment；把关系概率换成人工投票后，同一批方法的
Macro-F1 从 0.35 升到 0.72–0.85。这个脚本把 NLI 换成大模型，其余链路（D-S、
Statistic Gate、conflict_aware、朴素聚合、嵌套交叉验证）一个字不改，用来回答：
评估器变强之后分数能涨多少、各方法之间的关系变不变。

与 NLI 路径保持一致的地方
------------------------
* **一次只看一篇文档。** 每个 (claim, 文档) 组合单独发一个请求，模型看不到同一
  条 claim 的其他文档 —— 文档之间是否矛盾留给下游的融合去发现，评估器不越权。
* **输入输出与 ``run_refchecker_nli.py`` 同形**：读同一份 ``checking_inputs.json``，
  写出带 ``retrieved2response_probabilities`` 的 ``checking_outputs``，可以直接
  交给 ``ingest_ragchecker.py --use-probabilities``。

必须写进论文的一点
------------------
这里的概率是模型**口头给出**的（让它直接写三个数），不是 softmax。它比 NLI 的
连续概率粗，常见取值会集中在少数几个整数百分比上；谱系里记为
``probability_source: verbalized``。

用法（密钥读自环境变量 ``DEEPSEEK_API_KEY`` 或项目根目录的 ``.env``，脚本不打印、
不记录它）::

    python scripts/run_llm_relations.py \
        --input outputs/ragchecker_v2/test_checking_inputs.json \
        --output outputs/ragchecker_v2/test_checking_outputs_deepseek_flash.json

先 ``--dry-run`` 看提示词与请求数。中断后原样重跑即可续跑：已判完的组合记在
``<output>.pairs.jsonl``。退出码：0 = 全部完成；2 = 还有组合没判成；1 = 出错。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_llm_judge import (
    DEEPSEEK_CONTENT_ATTEMPTS,
    DEEPSEEK_KEY_ENV,
    PROJECT_ROOT,
    FatalJudgeError,
    RetryableJudgeError,
    _secret,
    make_deepseek_post,
)

DEFAULT_MODEL = "deepseek-flash"
#: 概率三元组的顺序，与 NLI 路径一致。
LABEL_ORDER = ("Entailment", "Neutral", "Contradiction")
_KEYS = ("entailment", "neutral", "contradiction")
EXIT_INCOMPLETE = 2

SYSTEM_PROMPT = """\
You are judging the relation between ONE evidence document and a claim.

Use ONLY this document. Do not use your own background knowledge about whether \
the claim is true, and do not guess what other documents might say.

- entailment: the document, on its own, gives real evidence that the claim is true.
- contradiction: the document, on its own, gives real evidence that the claim is false.
- neutral: the document is off-topic, too vague, or does not settle the claim \
either way (including when it says the answer could not be found).

Answer with a JSON object of exactly this form and nothing else: \
{"entailment": 0.10, "neutral": 0.80, "contradiction": 0.10} - three \
probabilities between 0 and 1 that sum to 1, expressing how confident you are \
in each relation."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="大模型关系评估器（DeepSeek）")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8, help="并发请求数")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条样本（冒烟测试）")
    parser.add_argument(
        "--dry-run", action="store_true", help="打印提示词与请求数，不调用 API"
    )
    return parser.parse_args(argv)


def prompt_digest() -> str:
    """系统提示词的 SHA-256：改了提示词，旧结果就不能续用。"""
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def build_user_message(claim: str, document: str) -> str:
    """一个 (claim, 文档) 组合的用户消息。"""
    return f"Claim: {claim}\n\nDocument: {document}"


def parse_probabilities(text: str, where: str) -> list[float]:
    """把模型写的三个数解析成归一化的 ``[Entailment, Neutral, Contradiction]``。

    Raises:
        RetryableJudgeError: 不是约定的 JSON、有负数或非数字、或三者之和为 0。
    """
    try:
        payload = json.loads(text)
        values = [float(payload[key]) for key in _KEYS]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RetryableJudgeError(f"{where}：输出不是约定的 JSON：{text[:120]!r}") from error
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise RetryableJudgeError(f"{where}：概率超出 [0, 1]：{values}")
    total = sum(values)
    if total <= 0.0:
        raise RetryableJudgeError(f"{where}：三个概率之和为 0")
    # 模型写的数常常只是大致归一（例如 0.33 × 3）；按和归一化，不改变相对大小。
    return [value / total for value in values]


def judge_pair(post, query_id: str, doc_id: str, claim: str, document: str, args) -> dict:
    """判一个 (claim, 文档) 组合，返回一条可落盘的记录。"""
    where = f"{query_id}/{doc_id}"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(claim, document)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    last: RetryableJudgeError | None = None
    for _ in range(DEEPSEEK_CONTENT_ATTEMPTS):
        body = post(payload)
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RetryableJudgeError(
                f"{where}：finish_reason={choice.get('finish_reason')!r}，输出不完整；"
                "若是 length，请调大 --max-tokens"
            )
        content = (choice["message"].get("content") or "").strip()
        try:
            if not content:
                raise RetryableJudgeError(f"{where}：模型返回了空内容")
            probabilities = parse_probabilities(content, where)
        except RetryableJudgeError as error:
            last = error
            continue
        usage = body.get("usage") or {}
        return {
            "query_id": query_id,
            "doc_id": doc_id,
            "probabilities": probabilities,
            "requested_model": args.model,
            "served_model": body.get("model", args.model),
            "prompt_sha256": prompt_digest(),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    assert last is not None
    raise last


def pairs_path(output: Path) -> Path:
    """逐组合结果（也是断点）的路径。"""
    return output.with_name(output.name + ".pairs.jsonl")


def load_done(path: Path, model: str) -> dict[tuple[str, str], dict]:
    """读取已判完的组合；提示词或模型对不上时拒绝续跑。"""
    done: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return done
    digest = prompt_digest()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field, expected in (("prompt_sha256", digest), ("requested_model", model)):
                if record.get(field) != expected:
                    raise ValueError(
                        f"{path} 第 {line_number} 行的 {field}={record.get(field)!r} "
                        f"与本次运行的 {expected!r} 不一致；请换一个 --output"
                    )
            done[(record["query_id"], record["doc_id"])] = record
    return done


def assemble(items: list[dict], done: dict[tuple[str, str], dict], model: str) -> dict:
    """把逐组合结果拼成与 ``run_refchecker_nli.py`` 同形的输出。"""
    results = []
    for item in items:
        triples = [
            done[(item["query_id"], context["doc_id"])]["probabilities"]
            for context in item["retrieved_context"]
        ]
        labels = [
            LABEL_ORDER[max(range(3), key=lambda index: triple[index])] for triple in triples
        ]
        results.append(
            {
                **item,
                "response_claims": [[item["response"]]],
                "retrieved2response": [labels],
                "retrieved2response_probabilities": [triples],
            }
        )
    records = list(done.values())
    return {
        "results": results,
        "rag_ds_run_info": {
            "producer": "llm-relation-checker",
            "checker_model": model,
            "served_models": sorted({r["served_model"] for r in records}),
            "extractor": None,
            "claims_source": "dataset_given",
            "has_probabilities": True,
            "probability_label_order": list(LABEL_ORDER),
            # 模型口头给出的概率，不是 softmax。
            "probability_source": "verbalized",
            "prompt_sha256": prompt_digest(),
            "pairs": sum(len(item["retrieved_context"]) for item in items),
            "input_tokens": sum(r["input_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
        },
    }


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


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    try:
        items = json.loads(args.input.read_text(encoding="utf-8-sig"))["results"]
    except (OSError, ValueError, KeyError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1
    if args.limit is not None:
        items = items[: args.limit]

    progress = pairs_path(args.output)
    try:
        done = load_done(progress, args.model)
    except ValueError as error:
        print(f"[已有结果不可续用] {error}", file=sys.stderr)
        return 1

    pairs = [
        (item["query_id"], context["doc_id"], item["response"], context["text"])
        for item in items
        for context in item["retrieved_context"]
    ]
    todo = [pair for pair in pairs if (pair[0], pair[1]) not in done]
    print(f"{args.input.name}：{len(items)} 条 claim，{len(pairs)} 个组合，"
          f"已判 {len(pairs) - len(todo)} 个，待判 {len(todo)} 个")
    print(f"模型 {args.model}，提示词 {prompt_digest()[:12]}…")

    if args.dry_run:
        words = sum(len(build_user_message(c, d).split()) for _, _, c, d in todo)
        print(f"[dry-run] 不调用 API。待发请求 {len(todo)} 个，"
              f"用户消息合计约 {words} 个英文词（token 数大致是它的 1.3–1.5 倍）。")
        if todo:
            print("---- 系统提示词 ----")
            print(SYSTEM_PROMPT)
            print("---- 第一条用户消息 ----")
            print(build_user_message(todo[0][2], todo[0][3]))
        return 0

    if todo:
        api_key = _secret(DEEPSEEK_KEY_ENV)
        if api_key is None:
            print(f"[没有可用的凭证] 没有找到 {DEEPSEEK_KEY_ENV}。", file=sys.stderr)
            print(f"请在 {PROJECT_ROOT / '.env'} 里写一行：{DEEPSEEK_KEY_ENV}=你的密钥",
                  file=sys.stderr)
            return 1
        post = make_deepseek_post(api_key)
        progress.parent.mkdir(parents=True, exist_ok=True)
        failures = 0
        with (
            progress.open("a", encoding="utf-8", newline="\n") as handle,
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool,
        ):
            futures = {pool.submit(judge_pair, post, *pair, args): pair for pair in todo}
            for count, future in enumerate(as_completed(futures), start=1):
                pair = futures[future]
                try:
                    record = future.result()
                except FatalJudgeError as error:
                    print(f"[无法继续] {error}", file=sys.stderr)
                    pool.shutdown(cancel_futures=True)
                    return 1
                except RetryableJudgeError as error:
                    failures += 1
                    print(f"[这一个没判成，稍后重跑] {pair[0]}/{pair[1]}: {error}", file=sys.stderr)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                done[(record["query_id"], record["doc_id"])] = record
                if count % 100 == 0 or count == len(todo):
                    print(f"  已处理 {count}/{len(todo)}")
        if failures:
            print(f"有 {failures} 个组合没判成；原样重跑即可续跑。", file=sys.stderr)
            return EXIT_INCOMPLETE

    payload = assemble(items, done, args.model)
    _atomic_write_json(args.output, payload)
    info = payload["rag_ds_run_info"]
    counts = {label: 0 for label in LABEL_ORDER}
    for result in payload["results"]:
        for label in result["retrieved2response"][0]:
            counts[label] += 1
    print("关系评估完成。")
    print("  标签分布 " + "，".join(
        f"{label} {count}（{count / max(info['pairs'], 1):.1%}）" for label, count in counts.items()
    ))
    print(f"  token 用量：输入 {info['input_tokens']}，输出 {info['output_tokens']}")
    if args.limit is not None:
        print(f"  注意：这是 --limit {args.limit} 的子集，只能用于冒烟测试。")
    print(f"  输出 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
