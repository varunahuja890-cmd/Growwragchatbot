"""Pipeline tests — fake store and fake LLM, no network (implementation.md §8)."""

from __future__ import annotations

import numpy as np
import pytest

from groww_faq.config import AppConfig, IngestionSettings, Paths
from groww_faq.query.pipeline import QueryPipeline
from groww_faq.registry import TIER_A, citation_by_slug
from groww_faq.utils import write_json

LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
MID_CAP_URL = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
FLEXI_URL = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
AMC_URL = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
DIM = 8


def make_cfg(tmp_path):
    cfg = AppConfig(
        ingestion=IngestionSettings(urls=[e.url for e in TIER_A]),
        paths=Paths(artifacts=tmp_path),
    )
    cfg.paths.ensure_all()
    write_json(
        cfg.paths.embeddings / "manifest.json",
        {
            "model_name": cfg.embedding.model_name,
            "dim": DIM,
            "normalize": True,
            "count": 1,
            "chunk_ids": ["c1"],
            "checksum": "test",
        },
    )
    return cfg


def fact_hit(chunk_id="c1", text=None, **meta_over):
    text = text or (
        "HDFC Mutual Fund | HDFC Large Cap Fund | Direct Growth | "
        "expense_ratio: 0.74%"
    )
    meta = {
        "amc": "HDFC Mutual Fund",
        "scheme": "HDFC Large Cap Fund",
        "scheme_slug": "hdfc-large-cap-fund-direct-growth",
        "plan": "Direct Growth",
        "page_type": "scheme",
        "chunk_type": "fact",
        "section": "Fund facts",
        "fact_key": "expense_ratio",
        "source_url": LARGE_CAP_URL,
        "fetched_at": "2026-11-11T00:00:00Z",
        "content_hash": "hash1",
    }
    meta.update(meta_over)
    return {"chunk_id": chunk_id, "text": text, "meta": meta}


class FakeStore:
    def __init__(self, hits, distance=0.15):
        self.hits = hits
        self.distance = distance
        self.calls = 0

    def count(self):
        return len(self.hits)

    def query(self, query_embeddings, n_results, where=None):
        self.calls += 1
        rows = [
            h
            for h in self.hits
            if not where or h["meta"]["scheme_slug"] == where.get("scheme_slug")
        ]
        rows = rows[:n_results]
        return {
            "ids": [h["chunk_id"] for h in rows],
            "documents": [h["text"] for h in rows],
            "metadatas": [h["meta"] for h in rows],
            "distances": [self.distance] * len(rows),
            "embeddings": [[1.0] * DIM for _ in rows],
        }


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=False):
        return np.ones((len(texts), DIM), dtype=np.float32)


class FakeLLM:
    def __init__(self, reply="The expense ratio is 0.74%."):
        self.reply = reply
        self.calls = 0

    def complete(self, system, user, temperature, max_tokens):
        self.calls += 1
        return self.reply

    def check_model(self):
        return "ok", ""


def make_pipeline(tmp_path, hits, llm=None, distance=0.15):
    cfg = make_cfg(tmp_path)
    store = FakeStore(hits, distance=distance)
    return (
        QueryPipeline(
            cfg,
            store=store,
            llm=llm or FakeLLM(),
            encoder=FakeEncoder(),
            trace_dir=tmp_path / "traces",
        ),
        store,
        cfg,
    )


def traces(tmp_path):
    return sorted(p.name for p in (tmp_path / "traces").glob("*.json"))


# ------------------------------------------------------------------ answer
def test_answer_path_with_citation(tmp_path):
    pipeline, store, _ = make_pipeline(tmp_path, [fact_hit()])
    answer = pipeline.answer_question(
        "What is the expense ratio of HDFC Large Cap?"
    )
    assert answer.path == "answer"
    assert answer.body == "The expense ratio is 0.74%."
    assert answer.citation_url == LARGE_CAP_URL
    assert answer.fetched_at == "2026-11-11T00:00:00Z"
    assert (tmp_path / "traces" / f"{answer.query_id}.json").exists()
    trace_files = traces(tmp_path)
    assert len(trace_files) == 1


# ------------------------------------------------------------------ pii
def test_pii_block_stores_nothing(tmp_path):
    pipeline, store, _ = make_pipeline(tmp_path, [fact_hit()])
    answer = pipeline.answer_question(
        "My PAN is ABCDE1234F, check my folio"
    )
    assert answer.path == "pii_block"
    assert answer.body == "I don't need or store personal details."
    assert store.calls == 0
    assert traces(tmp_path) == []  # no 05_retrieval file for a PII turn


# ---------------------------------------------------------------- refusals
def test_advice_refusal_no_llm_no_retrieval(tmp_path):
    llm = FakeLLM()
    pipeline, store, _ = make_pipeline(
        tmp_path, [fact_hit(scheme_slug="hdfc-mid-cap-fund-direct-growth")]
    )
    answer = pipeline.answer_question("Should I invest in HDFC Mid Cap?")
    assert answer.path == "refusal"
    assert answer.citation_url == MID_CAP_URL
    assert llm.calls == 0
    assert store.calls == 0


def test_comparison_refusal_keeps_related_url_on_trace(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, [fact_hit()])
    answer = pipeline.answer_question(
        "Which is better, Large Cap or Mid Cap?"
    )
    assert answer.path == "refusal"
    assert answer.citation_url == LARGE_CAP_URL  # one primary link
    assert answer.related_url == MID_CAP_URL  # stored, not rendered
    import json

    trace = json.loads(
        (tmp_path / "traces" / f"{answer.query_id}.json").read_text("utf-8")
    )
    assert trace["related_url"] == MID_CAP_URL


def test_portfolio_refusal_uses_educational_link(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, [fact_hit()])
    answer = pipeline.answer_question(
        "How much of my portfolio should be in ELSS?"
    )
    link = citation_by_slug("groww-mf-knowledge")
    assert answer.path == "refusal"
    assert answer.citation_url == link.url


# ----------------------------------------------------------------- clarify
def test_scheme_specific_without_scheme_clarifies(tmp_path):
    llm = FakeLLM()
    pipeline, store, _ = make_pipeline(tmp_path, [fact_hit()])
    answer = pipeline.answer_question("What is the expense ratio?")
    assert answer.path == "clarify"
    for name in (
        "HDFC Large Cap Fund",
        "HDFC Flexi Cap Fund",
        "HDFC ELSS Tax Saver Fund",
        "HDFC Mid Cap Fund",
    ):
        assert name in answer.body
    assert llm.calls == 0
    assert store.calls == 0


# -------------------------------------------------------------------- miss
def test_below_threshold_is_miss(tmp_path):
    pipeline, _, _ = make_pipeline(
        tmp_path,
        [fact_hit(scheme_slug="hdfc-mid-cap-fund-direct-growth")],
        distance=0.95,  # similarity 0.05 < 0.25
    )
    answer = pipeline.answer_question("What is the AUM of HDFC Mid Cap?")
    assert answer.path == "miss"
    assert answer.citation_url == MID_CAP_URL


def test_statement_miss_uses_tier_b_help_url(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, [fact_hit()], distance=0.95)
    answer = pipeline.answer_question("How do I download my statement?")
    link = citation_by_slug("groww-help-mf-statement")
    assert answer.path == "miss"
    assert answer.citation_url == link.url


# ---------------------------------------------------------------- redirect
def test_performance_redirects_without_cagr(tmp_path):
    llm = FakeLLM()
    pipeline, _, _ = make_pipeline(tmp_path, [fact_hit()])
    answer = pipeline.answer_question(
        "What were the 3-year returns of HDFC Flexi Cap?"
    )
    assert answer.path == "redirect"
    assert answer.citation_url == FLEXI_URL
    assert llm.calls == 0


# ---------------------------------------------------------------- template
def test_bad_model_text_falls_back_to_template(tmp_path):
    llm = FakeLLM(reply="You should buy this fund right now.")
    pipeline, _, _ = make_pipeline(tmp_path, [fact_hit()], llm=llm)
    answer = pipeline.answer_question(
        "What is the expense ratio of HDFC Large Cap?"
    )
    assert answer.path == "template"
    assert answer.body.startswith("HDFC Mutual Fund | HDFC Large Cap Fund")
    assert llm.calls == 2  # first attempt + one regenerate


def test_llm_failure_falls_back_to_template(tmp_path):
    class BrokenLLM:
        calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("provider down")

        def check_model(self):
            return "ok", ""

    pipeline, _, _ = make_pipeline(tmp_path, [fact_hit()], llm=BrokenLLM())
    answer = pipeline.answer_question(
        "What is the expense ratio of HDFC Large Cap?"
    )
    assert answer.path == "template"
    assert answer.citation_url == LARGE_CAP_URL


def test_citation_off_registry_is_substituted(tmp_path):
    rogue = fact_hit(source_url="https://evil.example.com/fake")
    pipeline, _, _ = make_pipeline(tmp_path, [rogue])
    answer = pipeline.answer_question(
        "What is the expense ratio of HDFC Large Cap?"
    )
    assert answer.citation_url in {
        e.url for e in TIER_A
    }  # allowlisted substitute
