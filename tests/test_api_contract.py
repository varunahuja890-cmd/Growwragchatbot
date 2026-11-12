"""Vercel function contract tests — the port must keep the pipeline's shapes.

The serverless entrypoint (``api/ask.py``) mirrors ``groww_faq.query.pipeline``
stage for stage (implementation.md §9.2). It is not an installed package, so it
is loaded by path. Nothing here touches the network or the ONNX model: only the
pure guardrail/link helpers are exercised.

Regression covered: ``_refusal_link`` once fell through to a 2-tuple while the
caller unpacks 3, so every refusal with no scheme (e.g. "I invested 50000,
will it double?") returned a 500 from ``/api/ask``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FUNCTION = ROOT / "api" / "ask.py"


@pytest.fixture(scope="module")
def ask():
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    spec = importlib.util.spec_from_file_location("vercel_api_ask", FUNCTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_refusal_link_always_returns_triple(ask):
    """Every branch of the refusal linker must be a 3-tuple (url, label, related)."""
    # No scheme named, not a portfolio question: the branch that used to crash.
    url, label, related = ask._refusal_link(
        "I invested 50000, will it double?", "ADVICE", [], None
    )
    assert url.startswith("https://groww.in/")
    assert label
    assert related == ""

    # Portfolio advice keeps the educational Tier B link, related empty.
    url, label, related = ask._refusal_link(
        "Should I put my whole portfolio into this fund?", "PORTFOLIO", [], None
    )
    assert url.startswith("https://groww.in/")
    assert related == ""

    # Scheme named wins over the fallback.
    url, label, related = ask._refusal_link(
        "Is HDFC Mid Cap good?", "ADVICE", ["hdfc-mid-cap-fund-direct-growth"], None
    )
    assert "hdfc-mid-cap-fund" in url


def test_refusal_question_answers_instead_of_raising(ask, monkeypatch):
    """End-to-end guardrail turn: the handler-facing dict, no retrieval, no LLM."""
    def boom(*_args, **_kwargs):  # pragma: no cover - must never be called
        raise AssertionError("refusal turns must not retrieve or call the LLM")

    monkeypatch.setattr(ask, "search", boom)
    monkeypatch.setattr(ask, "llm_complete", boom)

    result = ask.answer_question("I invested 50000, will it double?")
    assert result["path"] in {"refusal", "pii_block"}
    if result["path"] == "refusal":
        assert result["citation_url"].startswith("https://groww.in/")
        assert result["hits"] == []
