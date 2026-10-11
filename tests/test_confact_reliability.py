"""CONFACT 数据构建与文档级可靠性实验的测试。

三件最容易悄悄出错的事：来源不明的 pickle 不能执行代码；网页切段与选段要
确定、可复现；「有无可靠性」两种条件下只有可靠性不同。
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pytest

from rag_ds.data_io import load_samples
from rag_ds.datasets.confact import (
    CREDIBILITY_RELIABILITY,
    UNRATED_RELIABILITY,
    ConfactDataError,
    build_confact_dataset,
    load_untrusted_pickle,
    lookup_credibility,
    select_passage,
    source_domain,
    split_passages,
)
from rag_ds.experiments.reliability_verdict import (
    VERDICT_METHODS,
    reliability_from_credibility,
    verdict_scores,
)
from rag_ds.schemas import Claim, ContextChunk, EvidenceState, RAGSample, RelationPrediction

MEDIA = {
    "trusted.org": {"media": "Trusted", "credibility": "high"},
    "tabloid.com": {"media": "Tabloid", "credibility": "low"},
    "middle.net": {"media": "Middle", "credibility": "medium"},
}


class _Exploit:
    """加载时会调用 os.system 的对象：典型的恶意 pickle。"""

    def __reduce__(self):
        import os

        return (os.system, ("echo should-never-run",))


class TestUntrustedPickle:
    def test_plain_containers_load(self, tmp_path: Path) -> None:
        path = tmp_path / "data.pkl.gz"
        path.write_bytes(gzip.compress(pickle.dumps([{"id": 1, "claim": "x", "n": 1.5}])))

        assert load_untrusted_pickle(path) == [{"id": 1, "claim": "x", "n": 1.5}]

    def test_pickle_that_imports_anything_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "evil.pkl"
        path.write_bytes(pickle.dumps([_Exploit()]))

        with pytest.raises(pickle.UnpicklingError, match="拒绝加载"):
            load_untrusted_pickle(path)


class TestDomains:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.Trusted.org/a/b?c=1", "trusted.org"),
            ("http://news.trusted.org/x", "news.trusted.org"),
            ("", ""),
        ],
    )
    def test_source_domain(self, url: str, expected: str) -> None:
        assert source_domain(url) == expected

    def test_lookup_falls_back_to_the_registered_domain(self) -> None:
        assert lookup_credibility("trusted.org", MEDIA) == "high"
        assert lookup_credibility("news.trusted.org", MEDIA) == "high"
        assert lookup_credibility("unknown.example", MEDIA) is None


class TestPassages:
    def test_windows_overlap_and_cover_the_text(self) -> None:
        text = " ".join(f"w{i}" for i in range(10))

        passages = split_passages(text, window=4, stride=2)

        assert passages == [
            "w0 w1 w2 w3",
            "w2 w3 w4 w5",
            "w4 w5 w6 w7",
            "w6 w7 w8 w9",
        ]

    def test_short_text_is_one_passage_and_empty_text_none(self) -> None:
        assert split_passages("a  b\nc", window=10, stride=5) == ["a b c"]
        assert split_passages("   ", window=10, stride=5) == []

    def test_long_documents_are_capped(self) -> None:
        text = " ".join(["x"] * 100)

        passages = split_passages(text, window=10, stride=10, max_words=30)

        assert len(passages) == 3

    def test_most_similar_passage_is_chosen_per_document(self) -> None:
        chosen = select_passage(
            "polar bear population growing",
            [
                ["cookie banner subscribe newsletter", "the polar bear population is growing"],
                ["polar bear numbers", "weather forecast for tomorrow"],
            ],
        )

        assert chosen == [1, 0]

    def test_no_overlap_falls_back_to_the_first_passage(self) -> None:
        assert select_passage("polar bear", [["alpha beta", "gamma delta"]]) == [0]


def _raw_claims() -> list[dict]:
    filler = "navigation menu subscribe " * 20
    return [
        {
            "id": 1,
            "claim": "The bridge was built in 1900.",
            "label": "Supported ",  # 原始数据里有带尾随空格的标签
            "question": "Was the bridge built in 1900?",
            "evidence_url": [
                {"evidence_id": "1_1", "original_link": "https://www.trusted.org/a",
                 "content": filler + "Records show the bridge was built in 1900 by the city."},
                {"evidence_id": "1_2", "original_link": "https://tabloid.com/b",
                 "content": "Some say the bridge was built much later, in 1950."},
                {"evidence_id": "1_3", "original_link": "https://nowhere.example/c", "content": "  "},
            ],
        },
        {
            "id": 2,
            "claim": "The lake is salty.",
            "label": "Refuted",
            "question": "",
            "evidence_url": [
                {"evidence_id": "2_1", "original_link": "https://blog.unrated.example/x",
                 "content": "The lake is a freshwater lake."},
            ],
        },
    ]


def _write_raw(directory: Path, claims: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "HumC.pkl.gz").write_bytes(gzip.compress(pickle.dumps(claims)))
    (directory / "mbfc_media_data.pkl").write_bytes(pickle.dumps(MEDIA))
    return directory


class TestBuild:
    def test_samples_reliability_and_provenance(self, tmp_path: Path) -> None:
        raw = _write_raw(tmp_path / "confact", _raw_claims())
        out = tmp_path / "out"

        meta = build_confact_dataset(raw, out, window=12, stride=6)
        first, second = load_samples(out / "samples.jsonl")

        assert first.sample_id == "confact-humc-0001"
        assert first.gold_state is EvidenceState.SUPPORTED
        assert second.gold_state is EvidenceState.REFUTED
        # 空文档被跳过，doc_id 仍然连续。
        assert [c.doc_id for c in first.contexts] == [
            "confact-humc-0001-d1",
            "confact-humc-0001-d2",
        ]
        assert [c.reliability for c in first.contexts] == [
            CREDIBILITY_RELIABILITY["high"],
            CREDIBILITY_RELIABILITY["low"],
        ]
        assert second.contexts[0].reliability == UNRATED_RELIABILITY
        # 选中的是讲到 claim 的那一段，而不是网页开头的导航文字。
        assert "built in 1900" in first.contexts[0].text
        assert second.question == "The lake is salty."  # 没有问题时用 claim 本身

        assert meta["claims"] == 2
        assert meta["label_counts"] == {"supported": 1, "refuted": 1}
        assert meta["documents"] == 3
        assert meta["documents_skipped_empty"] == 1
        assert meta["document_credibility_counts"] == {"high": 1, "low": 1, "unrated": 1}
        provenance = [
            json.loads(line)
            for line in (out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [c["credibility"] for c in provenance[0]["contexts"]] == ["high", "low"]
        assert provenance[1]["contexts"][0]["credibility"] is None

    def test_build_is_reproducible(self, tmp_path: Path) -> None:
        raw = _write_raw(tmp_path / "confact", _raw_claims())

        build_confact_dataset(raw, tmp_path / "a", window=12, stride=6)
        build_confact_dataset(raw, tmp_path / "b", window=12, stride=6)

        for name in ("samples.jsonl", "provenance.jsonl", "meta.json"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

    def test_unknown_label_is_rejected(self, tmp_path: Path) -> None:
        claims = _raw_claims()
        claims[0]["label"] = "Half True"
        raw = _write_raw(tmp_path / "confact", claims)

        with pytest.raises(ConfactDataError, match="标签未知"):
            build_confact_dataset(raw, tmp_path / "out")

    def test_malicious_raw_file_is_refused(self, tmp_path: Path) -> None:
        raw = _write_raw(tmp_path / "confact", _raw_claims())
        (raw / "HumC.pkl.gz").write_bytes(gzip.compress(pickle.dumps([_Exploit()])))

        with pytest.raises(pickle.UnpicklingError):
            build_confact_dataset(raw, tmp_path / "out")

    def test_existing_output_is_not_overwritten(self, tmp_path: Path) -> None:
        raw = _write_raw(tmp_path / "confact", _raw_claims())
        build_confact_dataset(raw, tmp_path / "out")

        with pytest.raises(FileExistsError):
            build_confact_dataset(raw, tmp_path / "out")


def _verdict_sample(gold: EvidenceState = EvidenceState.SUPPORTED) -> RAGSample:
    return RAGSample(
        sample_id="s1",
        question="问题？",
        answer="答案。",
        claims=(Claim(claim_id="s1-c1", text="答案。"),),
        contexts=(
            ContextChunk(doc_id="s1-good", text="可信来源。"),
            ContextChunk(doc_id="s1-bad1", text="不可信来源一。"),
            ContextChunk(doc_id="s1-bad2", text="不可信来源二。"),
        ),
        gold_state=gold,
    )


def _verdict_predictions() -> list[RelationPrediction]:
    """一篇可信文档支持，两篇不可信文档反驳。"""
    triples = {
        "s1-good": (0.80, 0.05, 0.15),
        "s1-bad1": (0.05, 0.70, 0.25),
        "s1-bad2": (0.05, 0.70, 0.25),
    }
    return [
        RelationPrediction(
            sample_id="s1", claim_id="s1-c1", doc_id=doc_id, evaluator="e1",
            p_support=s, p_refute=r, p_unknown=u,
        )
        for doc_id, (s, r, u) in triples.items()
    ]


class TestVerdictScores:
    CREDIBILITY = {"s1-good": "high", "s1-bad1": "low", "s1-bad2": None}

    def test_reliability_mapping(self) -> None:
        reliability = reliability_from_credibility(
            self.CREDIBILITY, {"high": 1.0, "medium": 0.6, "low": 0.2}, unrated=0.5
        )

        assert reliability == {"s1-good": 1.0, "s1-bad1": 0.2, "s1-bad2": 0.5}

    def test_three_methods_one_score_per_claim(self) -> None:
        scores = verdict_scores([_verdict_sample()], _verdict_predictions(), 1.0)

        assert [s.method for s in scores] == list(VERDICT_METHODS)
        assert all(s.gold_label == "supported" for s in scores)

    def test_without_reliability_the_two_refuting_documents_win(self) -> None:
        scores = {s.method: s for s in verdict_scores([_verdict_sample()], _verdict_predictions(), 1.0)}

        assert scores["ds"].predicted_label == "refuted"
        assert scores["weighted_average"].predicted_label == "refuted"
        assert scores["majority_vote"].predicted_label == "refuted"
        assert scores["weighted_average"].margin == pytest.approx((0.9 - 1.45) / 3)
        assert scores["majority_vote"].margin == pytest.approx(-1 / 3)

    def test_discounting_unreliable_documents_flips_the_verdict(self) -> None:
        reliability = {"s1-good": 1.0, "s1-bad1": 0.1, "s1-bad2": 0.1}

        scores = {
            s.method: s
            for s in verdict_scores([_verdict_sample()], _verdict_predictions(), reliability)
        }

        assert scores["ds"].predicted_label == "supported"
        assert scores["weighted_average"].predicted_label == "supported"
        # 多数票不看可靠性，结论不变。
        assert scores["majority_vote"].predicted_label == "refuted"
        assert scores["majority_vote"].margin == pytest.approx(-1 / 3)

    def test_fully_distrusted_evidence_is_undetermined(self) -> None:
        scores = {
            s.method: s for s in verdict_scores([_verdict_sample()], _verdict_predictions(), 0.0)
        }

        assert scores["ds"].predicted_label == "undetermined"
        assert scores["ds"].margin == 0.0
        assert scores["weighted_average"].predicted_label == "undetermined"

    def test_total_conflict_is_flagged_not_guessed(self) -> None:
        """一篇完全支持、一篇完全反驳：Dempster 归一化无定义，D-S 记为未定并标出来。"""
        sample = RAGSample(
            sample_id="s1",
            question="问题？",
            answer="答案。",
            claims=(Claim(claim_id="s1-c1", text="答案。"),),
            contexts=(
                ContextChunk(doc_id="s1-a", text="甲。"),
                ContextChunk(doc_id="s1-b", text="乙。"),
            ),
            gold_state=EvidenceState.REFUTED,
        )
        predictions = [
            RelationPrediction(
                sample_id="s1", claim_id="s1-c1", doc_id=doc_id, evaluator="e1",
                p_support=s, p_refute=r, p_unknown=0.0,
            )
            for doc_id, (s, r) in {"s1-a": (1.0, 0.0), "s1-b": (0.0, 1.0)}.items()
        ]

        scores = {s.method: s for s in verdict_scores([sample], predictions, 1.0)}

        assert scores["ds"].ds_undefined
        assert (scores["ds"].margin, scores["ds"].predicted_label) == (0.0, "undetermined")
        # 其余方法不受影响，也不带这个标记。
        assert not scores["weighted_average"].ds_undefined
        assert scores["weighted_average"].predicted_label == "undetermined"  # 0.5 对 0.5

    def test_four_class_gold_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="不是 supported / refuted"):
            verdict_scores(
                [_verdict_sample(EvidenceState.CONFLICTING)], _verdict_predictions(), 1.0
            )
