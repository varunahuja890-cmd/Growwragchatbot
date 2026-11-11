"""Typed data contracts shared by the ingestion and query pipelines."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class RegistryEntry(BaseModel):
    """A Tier A corpus page — the only pages that are ever fetched."""

    slug: str
    url: str
    page_type: str  # "amc" | "scheme"
    amc: str = "HDFC Mutual Fund"
    scheme: str = ""  # empty on the AMC overview page
    plan: str = ""


class CitationLink(BaseModel):
    """A Tier B URL: cited in answers/refusals, never fetched."""

    slug: str
    url: str
    label: str
    purpose: str


class FactPair(BaseModel):
    key: str  # canonical fact key (see ingestion.load.FACT_KEYS)
    label: str  # label as printed on the page
    value: str  # verbatim value from the page — never invented
    via: str = "dom"  # "dom" | "dom-section" | "bs4-prose" | "trafilatura-prose"
    section: str = ""  # heading of the section it was found in


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class Section(BaseModel):
    heading: str
    level: int = 2
    text_blocks: list[str] = Field(default_factory=list)
    key_value_pairs: list[FactPair] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)

    def full_text(self) -> str:
        return "\n".join(self.text_blocks)


class PageDocument(BaseModel):
    """Cleaned page — written to artifacts/01_clean/{slug}.json."""

    source_url: str
    page_title: str
    page_type: str  # "amc" | "scheme"
    slug: str
    amc: str
    scheme: str  # empty on the AMC page
    plan: str
    fetched_at: str  # UTC ISO-8601, from the raw fetch
    content_hash: str  # SHA-256 of the cleaned body
    sections: list[Section] = Field(default_factory=list)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    def cleaned_body_hash(self) -> str:
        payload = json.dumps(
            [s.model_dump() for s in self.sections],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """One retrievable unit — written to artifacts/02_chunks/{slug}.jsonl."""

    chunk_id: str
    text: str
    tokens: int
    amc: str
    scheme: str
    scheme_slug: str
    plan: str
    page_type: str
    chunk_type: str  # "fact" | "table_row" | "section"
    section: str
    fact_key: str = ""  # empty for section / table_row chunks
    source_url: str
    fetched_at: str
    content_hash: str  # SHA-256 of the chunk text — the embedding cache key

    def metadata(self) -> dict[str, str]:
        """Scalar metadata for the vector store (Chroma rejects nested objects)."""
        return {
            "amc": self.amc,
            "scheme": self.scheme,
            "scheme_slug": self.scheme_slug,
            "plan": self.plan,
            "page_type": self.page_type,
            "chunk_type": self.chunk_type,
            "section": self.section,
            "fact_key": self.fact_key,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "content_hash": self.content_hash,
        }


class Hit(BaseModel):
    """One retrieval hit — the trace unit for the demo panel."""

    chunk_id: str
    score: float  # cosine similarity = 1 - chroma distance
    text: str
    source_url: str = ""
    fetched_at: str = ""
    amc: str = ""
    scheme: str = ""
    scheme_slug: str = ""
    plan: str = ""
    page_type: str = ""
    chunk_type: str = ""
    section: str = ""
    fact_key: str = ""


class Answer(BaseModel):
    """One rendered turn. ``path`` is one of:
    answer | refusal | redirect | clarify | miss | template | pii_block."""

    body: str
    path: str
    citation_url: str = ""
    citation_label: str = ""
    related_url: str = ""  # comparison refusals only, never rendered as a second link
    fetched_at: str = ""
    query_id: str = ""
    intent: str = ""
    scheme_slug: str = ""


class RetrievalTrace(BaseModel):
    """artifacts/05_retrieval/{query_id}.json — written only after the PII
    guard passes (PII-blocked turns leave no trace)."""

    query_id: str
    created_at: str
    question: str
    intent: str
    scheme_slug: str = ""
    filter: dict[str, Any] = Field(default_factory=dict)
    path: str = ""
    hits: list[dict[str, Any]] = Field(default_factory=list)
    citation_url: str = ""
    citation_label: str = ""
    related_url: str = ""
    answer_body: str = ""
    llm: dict[str, Any] = Field(default_factory=dict)
