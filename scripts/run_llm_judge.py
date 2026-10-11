"""LLM 直接判四类：不做任何证据融合的强对照。

把一条 claim 和它的全部证据文档一次性交给大模型，让它直接输出
``supported`` / ``refuted`` / ``insufficient`` / ``conflicting``。这是审稿人最
可能问的那个对照：「为什么不直接问大模型？」

它与本项目其余方法的不同之处
--------------------------
* **零样本，不调任何阈值**，所以不需要交叉验证：每条 claim 判一次，与嵌套
  交叉验证汇总出的逐条预测（每条 claim 也恰好一条）直接做配对自助检验。
* **只看给定的文档。** 提示词要求模型不使用自己的背景知识 —— 要比的是「从这批
  证据能得出什么」，不是模型记不记得这条 claim 的真假。
* 同一份提示词用于所有数据集，其 SHA-256 记在汇总文件里。

两个接口（``--provider``）
--------------------------
``deepseek``
    DeepSeek 的 chat completions 接口，JSON 输出模式，只用标准库发请求，不需要
    装任何包。密钥读自环境变量 ``DEEPSEEK_API_KEY``；没有时读项目根目录的
    ``.env`` 文件里同名的一行（``.env`` 已在 .gitignore 里）。
``anthropic``
    Anthropic SDK（主环境里先 ``python -m pip install anthropic``），结构化输出。
    凭证按 SDK 的默认方式解析（``ANTHROPIC_API_KEY`` 等）。

**密钥只在发请求时使用，不打印、不写进任何输出文件。**

用法::

    python scripts/run_llm_judge.py --provider deepseek \
        --manifest data/processed/climate_fever_v2/manifest.json \
        --out-dir outputs/llm_judge/climate_fever_v2/deepseek-v4-pro \
        --compare-with outputs/metrics/cross_validation_controls/cv_predictions.csv

先用 ``--dry-run`` 看提示词和请求数（不调用 API、不花钱），再用 ``--limit 5``
冒烟，最后跑全量。中断后原样重跑即可续跑：已有结果的 claim 会被跳过。

退出码：0 = 全部完成；2 = 有 claim 没判成（重跑一次继续）；1 = 出错。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rag_ds.data_io import load_samples
from rag_ds.dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    verify_artifact,
)
from rag_ds.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    GOLD_LABELS,
    UNDETERMINED_LABEL,
    bootstrap_interval,
    classification_report,
    paired_bootstrap,
)
from rag_ds.schemas import Claim, RAGSample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "deepseek": "deepseek-v4-pro"}
#: 判四类的结果在汇总表与配对比较里用的方法名。
METHOD_NAME = "llm_judge"
EXIT_INCOMPLETE = 2

SYSTEM_PROMPT = """\
You are assessing what a fixed set of evidence documents establishes about a claim.

Judge ONLY from the documents provided. Do not use your own background knowledge \
about whether the claim is true: if the documents do not settle it, the answer is \
not "supported" or "refuted", however confident you are from memory.

Choose exactly one label:

- supported: taken together, the documents support the claim, and none of them \
gives real evidence against it.
- refuted: taken together, the documents contradict the claim, and none of them \
gives real evidence for it.
- insufficient: the documents do not contain enough information to tell whether \
the claim is true or false (they are off-topic, too vague, or say the answer \
could not be found).
- conflicting: the documents disagree with each other about the claim - at least \
one gives real evidence for it and at least one gives real evidence against it.

Documents that say nothing relevant count as neither support nor refutation."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(GOLD_LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}

#: DeepSeek 的 JSON 模式不强制 schema，且要求提示词里出现 "json" 与格式示例。
JSON_INSTRUCTION = (
    "\n\nAnswer with a JSON object of exactly this form and nothing else: "
    '{"label": "supported"} - where the value is one of "supported", "refuted", '
    '"insufficient", "conflicting".'
)

#: 拒答回退：模型因安全策略拒绝时，由服务端按拒绝类别换一个模型重跑同一请求。
FALLBACK_BETA = "server-side-fallback-2026-07-01"

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"
#: DeepSeek 偶尔返回空内容或不合格的 JSON：同一条最多重问几次。
DEEPSEEK_CONTENT_ATTEMPTS = 3


class FatalJudgeError(RuntimeError):
    """整批都不该再发的错误：凭证无效、余额不足、请求本身不被接受。"""


class RetryableJudgeError(RuntimeError):
    """只影响这一条的错误：限流、服务端故障、网络中断、输出不完整。重跑可续。"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LLM 直接判四类（强对照）")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=tuple(DEFAULT_MODELS), default="anthropic")
    parser.add_argument("--model", default=None, help="默认取所选接口的默认模型")
    parser.add_argument(
        "--effort",
        default=None,
        choices=("low", "medium", "high", "xhigh", "max"),
        help="思考深度（仅 anthropic，默认 medium）；写进每条记录。更高更贵",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数")
    parser.add_argument(
        "--no-fallbacks",
        action="store_true",
        help="（仅 anthropic）不启用服务端拒答回退；代理或网关不支持这个 beta 时使用",
    )
    parser.add_argument("--limit", type=int, default=None, help="只判前 N 条（冒烟测试）")
    parser.add_argument(
        "--dry-run", action="store_true", help="打印提示词与请求数，不调用 API"
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="交叉验证的 cv_predictions.csv；给出时与其中每个方法做配对自助检验",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]
    if args.provider == "anthropic":
        args.effort = args.effort or "medium"
    elif args.effort is not None:
        parser.error("--effort 只适用于 --provider anthropic")
    else:
        args.effort = "default"
    return args


def system_prompt(provider: str = "anthropic") -> str:
    """实际发给模型的系统提示词。"""
    return SYSTEM_PROMPT + JSON_INSTRUCTION if provider == "deepseek" else SYSTEM_PROMPT


def build_user_message(sample: RAGSample, claim: Claim) -> str:
    """把一条 claim 与它的全部证据文档排成一条用户消息。"""
    lines = [f"Claim: {claim.text}", "", "Evidence documents:"]
    if not sample.contexts:
        lines.append("(no documents were retrieved)")
    for index, chunk in enumerate(sample.contexts, start=1):
        lines.append(f"[{index}] {chunk.text}")
    return "\n".join(lines)


def prompt_digest(provider: str = "anthropic") -> str:
    """系统提示词与输出格式的 SHA-256：改了提示词，旧结果就不能续用。"""
    payload = json.dumps(
        {"system": system_prompt(provider), "schema": OUTPUT_SCHEMA}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_done(path: Path, digest: str, model: str, effort: str) -> dict[tuple[str, str], dict]:
    """读取已有结果；提示词、模型或 effort 对不上时拒绝续跑。"""
    done: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field, expected in (
                ("prompt_sha256", digest),
                ("requested_model", model),
                ("effort", effort),
            ):
                if record.get(field) != expected:
                    raise ValueError(
                        f"{path} 第 {line_number} 行的 {field}={record.get(field)!r} "
                        f"与本次运行的 {expected!r} 不一致；新旧设置的结果不能混在"
                        "同一个文件里，请换一个 --out-dir"
                    )
            done[(record["sample_id"], record["claim_id"])] = record
    return done


def _base_record(sample: RAGSample, claim: Claim, args: argparse.Namespace) -> dict:
    return {
        "sample_id": sample.sample_id,
        "claim_id": claim.claim_id,
        "gold_label": sample.gold_state.value,  # type: ignore[union-attr]
        "requested_model": args.model,
        "effort": args.effort,
        "prompt_sha256": prompt_digest(args.provider),
    }


def _label_from_json(text: str, claim_id: str) -> str:
    """从模型的 JSON 输出里取出标签；格式或取值不对时报可重试的错误。"""
    try:
        label = json.loads(text)["label"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RetryableJudgeError(f"{claim_id}：输出不是约定的 JSON：{text[:120]!r}") from error
    if label not in GOLD_LABELS:
        raise RetryableJudgeError(f"{claim_id}：模型返回了未知标签 {label!r}")
    return label


def _classify_anthropic_error(error: Exception) -> Exception:
    """把 Anthropic SDK 的异常归成「整批停下」或「这一条重跑」。"""
    try:
        import anthropic
    except ImportError:
        return error
    if isinstance(
        error,
        (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ),
    ):
        # 凭证、模型名、beta 或参数不被接受：重试没有意义。
        return FatalJudgeError(f"{type(error).__name__}: {error.message}")
    if isinstance(error, anthropic.RateLimitError):
        return RetryableJudgeError(f"限流：{error.message}")
    if isinstance(error, anthropic.APIStatusError):
        kind = RetryableJudgeError if error.status_code >= 500 else FatalJudgeError
        return kind(f"API 错误 {error.status_code}: {error.message}")
    if isinstance(error, anthropic.APIConnectionError):
        return RetryableJudgeError(f"网络错误：{error}")
    return error


def _judge_anthropic(client, sample: RAGSample, claim: Claim, args: argparse.Namespace) -> dict:
    request = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "system": system_prompt("anthropic"),
        "messages": [{"role": "user", "content": build_user_message(sample, claim)}],
        "output_config": {
            "effort": args.effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }
    try:
        if args.no_fallbacks:
            response = client.messages.create(**request)
        else:
            response = client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **request
            )
    except Exception as error:
        classified = _classify_anthropic_error(error)
        if classified is error:
            raise
        raise classified from error

    record = {
        **_base_record(sample, claim, args),
        # 启用回退时，实际作答的可能是另一个模型；如实记下。
        "served_model": response.model,
        "stop_reason": response.stop_reason,
        "request_id": response._request_id,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    if response.stop_reason == "refusal":
        # 模型拒答：没有认领任何一类，与 D-S 的 undetermined 同样计为判错。
        record["predicted_label"] = UNDETERMINED_LABEL
        return record
    if response.stop_reason != "end_turn":
        raise RetryableJudgeError(
            f"{claim.claim_id}：stop_reason={response.stop_reason!r}，输出不完整；"
            "若是 max_tokens，请调大 --max-tokens"
        )
    text = next(block.text for block in response.content if block.type == "text")
    record["predicted_label"] = _label_from_json(text, claim.claim_id)
    return record


def make_deepseek_post(api_key: str, *, timeout: float = 300.0, max_retries: int = 5):
    """返回一个 ``post(payload) -> dict``：向 DeepSeek 发一次请求，自带退避重试。

    只用标准库。429 / 5xx / 网络错误按指数退避重试；401（密钥无效）、
    402（余额不足）、400 / 404 / 422（请求不被接受）直接报致命错误。
    """

    def post(payload: dict) -> dict:
        request = urllib.request.Request(
            DEEPSEEK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:300]
                if error.code in (429, 500, 502, 503, 504):
                    last = RetryableJudgeError(f"DeepSeek HTTP {error.code}: {detail}")
                else:
                    raise FatalJudgeError(f"DeepSeek HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last = RetryableJudgeError(f"DeepSeek 网络错误：{error}")
            except json.JSONDecodeError as error:
                last = RetryableJudgeError(f"DeepSeek 返回的不是 JSON：{error}")
            if attempt < max_retries:
                time.sleep(min(2.0**attempt, 30.0))
        assert last is not None
        raise last

    return post


def _judge_deepseek(post, sample: RAGSample, claim: Claim, args: argparse.Namespace) -> dict:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt("deepseek")},
            {"role": "user", "content": build_user_message(sample, claim)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    last: RetryableJudgeError | None = None
    for _ in range(DEEPSEEK_CONTENT_ATTEMPTS):
        body = post(payload)
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        record = {
            **_base_record(sample, claim, args),
            "served_model": body.get("model", args.model),
            "stop_reason": choice.get("finish_reason"),
            "request_id": body.get("id"),
            "input_tokens": usage.get("prompt_tokens", 0),
            # 思考模式下，输出 token 数包含推理过程。
            "output_tokens": usage.get("completion_tokens", 0),
        }
        if choice.get("finish_reason") == "content_filter":
            record["stop_reason"] = "refusal"
            record["predicted_label"] = UNDETERMINED_LABEL
            return record
        if choice.get("finish_reason") != "stop":
            raise RetryableJudgeError(
                f"{claim.claim_id}：finish_reason={choice.get('finish_reason')!r}，"
                "输出不完整；若是 length，请调大 --max-tokens"
            )
        content = (choice["message"].get("content") or "").strip()
        try:
            if not content:
                raise RetryableJudgeError(f"{claim.claim_id}：模型返回了空内容")
            record["predicted_label"] = _label_from_json(content, claim.claim_id)
        except RetryableJudgeError as error:
            last = error
            continue
        return record
    assert last is not None
    raise last


def judge_one(client, sample: RAGSample, claim: Claim, args: argparse.Namespace) -> dict:
    """判一条 claim，返回一条结果记录。

    ``client`` 对 anthropic 是 SDK 客户端，对 deepseek 是
    :func:`make_deepseek_post` 返回的函数。

    Raises:
        FatalJudgeError: 整批都不该再发。
        RetryableJudgeError: 这一条没判成，重跑可续。
    """
    if args.provider == "deepseek":
        return _judge_deepseek(client, sample, claim, args)
    return _judge_anthropic(client, sample, claim, args)


def _secret(name: str) -> str | None:
    """取一个密钥：先看环境变量，再看项目根目录 ``.env`` 里的同名行。"""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, separator, raw = line.partition("=")
            if separator and key.strip() == name:
                return raw.strip().strip("\"'") or None
    return None


def _make_client(args: argparse.Namespace):
    """按所选接口建好客户端；没有凭证时返回 ``None`` 并打印怎么配置。"""
    if args.provider == "deepseek":
        api_key = _secret(DEEPSEEK_KEY_ENV)
        if api_key is None:
            print(f"[没有可用的凭证] 没有找到 {DEEPSEEK_KEY_ENV}。", file=sys.stderr)
            print(f"请在 {PROJECT_ROOT / '.env'} 里写一行：{DEEPSEEK_KEY_ENV}=你的密钥",
                  file=sys.stderr)
            return None
        return make_deepseek_post(api_key)
    try:
        import anthropic
    except ImportError:
        print("[缺少依赖] 请先运行：python -m pip install anthropic", file=sys.stderr)
        return None
    try:
        return anthropic.Anthropic(max_retries=5)
    except anthropic.AnthropicError as error:
        print(f"[没有可用的凭证] {error}", file=sys.stderr)
        print("请设置环境变量 ANTHROPIC_API_KEY（或运行 ant auth login）。", file=sys.stderr)
        return None


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


def summarise(
    records: list[dict],
    compare_with: Path | None,
    resamples: int,
    seed: int,
) -> dict:
    """算分类指标、置信区间、token 用量，以及与交叉验证各方法的配对比较。"""
    ordered = sorted(records, key=lambda r: (r["sample_id"], r["claim_id"]))
    truth = [r["gold_label"] for r in ordered]
    predicted = [r["predicted_label"] for r in ordered]
    report = classification_report(truth, predicted, method=METHOD_NAME)
    interval = bootstrap_interval(
        truth, predicted, METHOD_NAME, labels=GOLD_LABELS, n_resamples=resamples, seed=seed
    )
    summary = {
        "claim_count": len(ordered),
        "report": json.loads(report.model_dump_json()),
        "interval": json.loads(interval.model_dump_json()),
        "predicted_label_counts": dict(Counter(predicted)),
        "refusals": sum(1 for r in ordered if r["stop_reason"] == "refusal"),
        "served_models": dict(Counter(r["served_model"] for r in ordered)),
        "input_tokens": sum(r["input_tokens"] for r in ordered),
        "output_tokens": sum(r["output_tokens"] for r in ordered),
    }
    if compare_with is None:
        return summary

    by_method: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    with compare_with.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_method[row["method"]][(row["sample_id"], row["claim_id"])] = row[
                "predicted_label"
            ]
    keys = [(r["sample_id"], r["claim_id"]) for r in ordered]
    comparisons = []
    for method, labels in sorted(by_method.items()):
        if set(labels) != set(keys):
            raise ValueError(
                f"{compare_with} 里 {method} 覆盖的 claim 与本次判的不一致，无法配对"
            )
        comparisons.append(
            json.loads(
                paired_bootstrap(
                    truth,
                    predicted,
                    [labels[key] for key in keys],
                    METHOD_NAME,
                    method,
                    labels=GOLD_LABELS,
                    n_resamples=resamples,
                    seed=seed,
                ).model_dump_json()
            )
        )
    summary["compared_with"] = str(compare_with)
    summary["comparisons"] = comparisons
    return summary


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = _parse_args(argv)
    predictions_path = args.out_dir / "judge_predictions.jsonl"
    summary_path = args.out_dir / "judge_summary.json"

    manifest_path = args.manifest.resolve()
    try:
        manifest = load_dataset_manifest(manifest_path)
        samples: list[RAGSample] = []
        for split in SPLITS:
            samples.extend(
                load_samples(
                    verify_artifact(manifest_path.parent, manifest.splits[split].samples)
                )
            )
    except (DatasetManifestError, ValueError, OSError) as error:
        print(f"[输入数据错误] {error}", file=sys.stderr)
        return 1

    pairs = sorted(
        ((sample, claim) for sample in samples for claim in sample.claims),
        key=lambda pair: (pair[0].sample_id, pair[1].claim_id),
    )
    missing_gold = [s.sample_id for s, _ in pairs if s.gold_state is None]
    if missing_gold:
        print(f"[输入数据错误] 样本缺少 gold_state：{missing_gold[:5]}", file=sys.stderr)
        return 1
    if args.limit is not None:
        pairs = pairs[: args.limit]

    digest = prompt_digest(args.provider)
    try:
        done = load_done(predictions_path, digest, args.model, args.effort)
    except ValueError as error:
        print(f"[已有结果不可续用] {error}", file=sys.stderr)
        return 1
    todo = [(s, c) for s, c in pairs if (s.sample_id, c.claim_id) not in done]

    print(f"数据集 {manifest.dataset_name}：共 {len(pairs)} 条 claim，"
          f"已有结果 {len(pairs) - len(todo)} 条，待判 {len(todo)} 条")
    print(f"接口 {args.provider}，模型 {args.model}，effort={args.effort}，提示词 {digest[:12]}…")

    if args.dry_run:
        words = sum(len(build_user_message(s, c).split()) for s, c in todo)
        print(f"[dry-run] 不调用 API。待发请求 {len(todo)} 个，"
              f"用户消息合计约 {words} 个英文词（token 数大致是它的 1.3–1.5 倍）。")
        if todo:
            print("---- 系统提示词 ----")
            print(system_prompt(args.provider))
            print("---- 第一条用户消息 ----")
            print(build_user_message(*todo[0]))
        return 0

    if todo:
        client = _make_client(args)
        if client is None:
            return 1
        args.out_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        with (
            predictions_path.open("a", encoding="utf-8", newline="\n") as handle,
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool,
        ):
            futures = {pool.submit(judge_one, client, s, c, args): c for s, c in todo}
            for count, future in enumerate(as_completed(futures), start=1):
                claim = futures[future]
                try:
                    record = future.result()
                except FatalJudgeError as error:
                    print(f"[无法继续] {error}", file=sys.stderr)
                    if args.provider == "anthropic" and not args.no_fallbacks:
                        print("若报错与 fallbacks / beta 有关，请加 --no-fallbacks 重跑。",
                              file=sys.stderr)
                    pool.shutdown(cancel_futures=True)
                    return 1
                except RetryableJudgeError as error:
                    failures += 1
                    print(f"[这一条没判成，稍后重跑] {claim.claim_id}: {error}", file=sys.stderr)
                    continue
                # 结果在主线程里逐条落盘：中断后已判完的不会丢。
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                done[(record["sample_id"], record["claim_id"])] = record
                if count % 25 == 0 or count == len(todo):
                    print(f"  已处理 {count}/{len(todo)}")
        if failures:
            print(f"有 {failures} 条没判成；原样重跑即可续跑。", file=sys.stderr)
            return EXIT_INCOMPLETE

    records = [done[(s.sample_id, c.claim_id)] for s, c in pairs]
    try:
        summary = summarise(records, args.compare_with, args.resamples, args.bootstrap_seed)
    except (ValueError, OSError) as error:
        print(f"[无法汇总] {error}", file=sys.stderr)
        return 1
    _atomic_write_json(
        summary_path,
        {
            "dataset_name": manifest.dataset_name,
            "method": METHOD_NAME,
            "provider": args.provider,
            "requested_model": args.model,
            "effort": args.effort,
            "fallbacks_enabled": args.provider == "anthropic" and not args.no_fallbacks,
            "prompt_sha256": digest,
            "is_subset": args.limit is not None,
            **summary,
        },
    )

    report, interval = summary["report"], summary["interval"]
    print()
    print(f"  {METHOD_NAME}: Accuracy {report['accuracy']:.4f}  Macro-F1 {report['macro_f1']:.4f}"
          f"  95% CI [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")
    print(f"  预测分布 {summary['predicted_label_counts']}；拒答 {summary['refusals']} 条")
    print(f"  实际作答的模型 {summary['served_models']}")
    print(f"  token 用量：输入 {summary['input_tokens']}，输出 {summary['output_tokens']}")
    for item in summary.get("comparisons", []):
        significant = item["ci_low"] > 0 or item["ci_high"] < 0
        print(f"  {METHOD_NAME} − {item['method_b']:<28}{item['observed_delta']:>+8.4f}"
              f"  [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]  p={item['p_value']:.3f}"
              f"  {'显著' if significant else '不显著'}")
    if args.limit is not None:
        print(f"  注意：这是 --limit {args.limit} 的子集，只能用于冒烟测试。")
    print(f"  输出 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
