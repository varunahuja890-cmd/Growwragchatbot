"""Phase 4 — Chunking (strategy ``layout_hybrid``).

Reads ``artifacts/01_clean/{slug}.json`` and writes
``artifacts/02_chunks/{slug}.jsonl`` plus ``stats.json``.

| chunk type   | how it is cut                                        | size          |
|--------------|------------------------------------------------------|---------------|
| fact         | one per canonical key-value pair                     | 15–40 tokens  |
| table_row    | one per row, header cells included                   | 20–50 tokens  |
| section      | one per H2/H3, split at 400 tokens, 60-token overlap | 150–400 tokens|

Fact text is self-describing:
``{AMC} | {Scheme} | {Plan} | {fact_key}: {fact_value}``

Token counts use the embedding model's tokenizer (MiniLM).
``chunk_id`` is a hash of (source_url, chunk_type, section, fact_key, text)
so a rebuild with the same config produces the same ids. Chunks under
``min_tokens`` or over ``hard_max_tokens`` are dropped — except a canonical
fact that is naturally short (for example ``exit_load: Nil``). Identical
text on the same page is deduplicated.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from groww_faq.config import AppConfig
from groww_faq.ingestion.load import required_keys
from groww_faq.models import Chunk, PageDocument, Section, TableData
from groww_faq.registry import TIER_A, entry_by_slug
from groww_faq.utils import read_json, sha256_text, utc_now_iso, write_json, write_jsonl

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class MiniLMTokenCounter:
    """Token counting backed by the embedding model's tokenizer."""

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for chunking — token counts use the "
                f"MiniLM tokenizer ({model_name}). Install sentence-transformers."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class Chunker:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.chunk_cfg = cfg.chunking
        self.counter = MiniLMTokenCounter(cfg.embedding.model_name)
        self.dropped = {"too_short": 0, "too_long": 0, "duplicates": 0}

    # ------------------------------------------------------------ text forms
    def _entity_prefix(self, doc: PageDocument) -> str:
        return " | ".join(p for p in (doc.amc, doc.scheme, doc.plan) if p)

    def _fact_text(self, doc: PageDocument, key: str, value: str) -> str:
        return f"{self._entity_prefix(doc)} | {key}: {value}"

    def _table_row_text(self, doc: PageDocument, table: TableData, row: list[str]) -> str:
        prefix = self._entity_prefix(doc)
        if table.headers and len(table.headers) == len(row):
            cells = ", ".join(
                f"{h}: {v}" for h, v in zip(table.headers, row) if v
            )
        else:
            cells = " | ".join(v for v in row if v)
        return f"{prefix} | {cells}"

    def _section_text(self, section: Section) -> str:
        return f"{section.heading}\n{section.full_text()}".strip()

    # ------------------------------------------------------------- splitting
    def _split_section_text(self, text: str) -> list[str]:
        """Split at max_tokens with an overlap of overlap_tokens (sentences
        are never cut mid-word; a single over-long sentence is word-split)."""
        max_tokens = self.chunk_cfg.max_tokens
        overlap = self.chunk_cfg.overlap_tokens

        if self.counter.count(text) <= max_tokens:
            return [text]

        sentences = [s for s in (s.strip() for s in _SENTENCE_RE.split(text)) if s]
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            st = self.counter.count(sentence)
            if st > max_tokens:  # rare: one enormous sentence
                if current:
                    pieces.append(" ".join(current))
                    current, current_tokens = [], 0
                pieces.extend(self._hard_split(sentence))
                continue
            if current and current_tokens + st > max_tokens:
                pieces.append(" ".join(current))
                tail: list[str] = []
                tail_tokens = 0
                for prev in reversed(current):
                    pt = self.counter.count(prev)
                    if tail_tokens + pt > overlap:
                        break
                    tail.insert(0, prev)
                    tail_tokens += pt
                current, current_tokens = tail, tail_tokens
            current.append(sentence)
            current_tokens += st

        if current:
            pieces.append(" ".join(current))
        return pieces

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for word in words:
            wt = self.counter.count(word)
            if current and current_tokens + wt + 1 > self.chunk_cfg.max_tokens:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(word)
            current_tokens += wt + 1
        if current:
            pieces.append(" ".join(current))
        return pieces

    # -------------------------------------------------------------- chunking
    def _make_chunk(
        self,
        doc: PageDocument,
        chunk_type: str,
        section: str,
        fact_key: str,
        text: str,
    ) -> Chunk | None:
        tokens = self.counter.count(text)
        content_hash = sha256_text(text)
        chunk_id = sha256_text(
            f"{doc.source_url}|{chunk_type}|{section}|{fact_key}|{text}"
        )[:16]
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            tokens=tokens,
            amc=doc.amc,
            scheme=doc.scheme,
            scheme_slug=doc.slug,
            plan=doc.plan,
            page_type=doc.page_type,
            chunk_type=chunk_type,
            section=section,
            fact_key=fact_key,
            source_url=doc.source_url,
            fetched_at=doc.fetched_at,
            content_hash=content_hash,
        )

    def _keep(self, chunk: Chunk | None) -> Chunk | None:
        """Apply the size rules. Canonical facts are exempt from min_tokens."""
        if chunk is None:
            return None
        if chunk.tokens > self.chunk_cfg.hard_max_tokens:
            self.dropped["too_long"] += 1
            return None
        if (
            chunk.tokens < self.chunk_cfg.min_tokens
            and chunk.chunk_type != "fact"
        ):
            self.dropped["too_short"] += 1
            return None
        return chunk

    def chunk_document(self, doc: PageDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen_hashes: set[str] = set()

        def add(chunk: Chunk | None) -> None:
            if chunk is None:
                return
            if chunk.content_hash in seen_hashes:
                self.dropped["duplicates"] += 1
                return
            seen_hashes.add(chunk.content_hash)
            chunks.append(chunk)

        for section in doc.sections:
            if self.chunk_cfg.fact_chunks:
                for pair in section.key_value_pairs:
                    add(
                        self._keep(
                            self._make_chunk(
                                doc, "fact", section.heading, pair.key,
                                self._fact_text(doc, pair.key, pair.value),
                            )
                        )
                    )
            if self.chunk_cfg.table_rows:
                for table in section.tables:
                    for row in table.rows:
                        add(
                            self._keep(
                                self._make_chunk(
                                    doc, "table_row", section.heading, "",
                                    self._table_row_text(doc, table, row),
                                )
                            )
                        )
            if section.text_blocks:
                for piece in self._split_section_text(self._section_text(section)):
                    add(
                        self._keep(
                            self._make_chunk(doc, "section", section.heading, "", piece)
                        )
                    )
        return chunks

    # ----------------------------------------------------------------- stats
    def _stats(self, by_slug: dict[str, list[Chunk]]) -> dict[str, Any]:
        all_chunks = [c for chunks in by_slug.values() for c in chunks]
        by_type: dict[str, int] = {"fact": 0, "table_row": 0, "section": 0}
        histogram = {"0-50": 0, "50-100": 0, "100-200": 0, "200-400": 0, "400-800": 0, "800+": 0}

        for chunk in all_chunks:
            by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1
            if chunk.tokens <= 50:
                histogram["0-50"] += 1
            elif chunk.tokens <= 100:
                histogram["50-100"] += 1
            elif chunk.tokens <= 200:
                histogram["100-200"] += 1
            elif chunk.tokens <= 400:
                histogram["200-400"] += 1
            elif chunk.tokens <= 800:
                histogram["400-800"] += 1
            else:
                histogram["800+"] += 1

        fact_keys_by_slug = {
            slug: sorted({c.fact_key for c in chunks if c.fact_key})
            for slug, chunks in by_slug.items()
        }
        return {
            "generated_at": utc_now_iso(),
            "strategy": self.chunk_cfg.strategy,
            "total_chunks": len(all_chunks),
            "by_type": by_type,
            "by_slug": {
                slug: {
                    "total": len(chunks),
                    "fact": sum(1 for c in chunks if c.chunk_type == "fact"),
                    "table_row": sum(1 for c in chunks if c.chunk_type == "table_row"),
                    "section": sum(1 for c in chunks if c.chunk_type == "section"),
                }
                for slug, chunks in by_slug.items()
            },
            "token_histogram": histogram,
            "fact_keys_by_slug": fact_keys_by_slug,
            "dropped": dict(self.dropped),
        }


def chunk_all(cfg: AppConfig) -> dict[str, Any]:
    """Chunk every loaded page. Writes 02_chunks/{slug}.jsonl + stats.json."""
    chunks_dir = cfg.paths.chunks
    chunker = Chunker(cfg)
    by_slug: dict[str, list[Chunk]] = {}

    for entry in TIER_A:
        clean_path = cfg.paths.clean / f"{entry.slug}.json"
        if not clean_path.exists():
            logger.error("[%s] no clean document — run the load stage first", entry.slug)
            continue
        doc = PageDocument(**read_json(clean_path))
        by_slug[entry.slug] = chunker.chunk_document(doc)
        write_jsonl(
            chunks_dir / f"{entry.slug}.jsonl",
            [c.model_dump(mode="json") for c in by_slug[entry.slug]],
        )
        logger.info(
            "[%s] %d chunks (%d facts)",
            entry.slug,
            len(by_slug[entry.slug]),
            sum(1 for c in by_slug[entry.slug] if c.chunk_type == "fact"),
        )

    if not by_slug:
        raise RuntimeError(
            f"no clean documents found under {cfg.paths.clean} — "
            "run `python scripts/ingest.py --from load` first"
        )

    stats = chunker._stats(by_slug)

    # Exit gate: every fact the parse report marked found has a fact chunk.
    gate: dict[str, Any] = {}
    report_path = cfg.paths.clean / "parse_report.json"
    if report_path.exists():
        report = read_json(report_path)
        for page in report.get("pages", []):
            slug = page.get("slug")
            if slug not in by_slug:
                continue
            entry = entry_by_slug(slug)
            required = required_keys(entry) if entry is not None else []
            present = {c.fact_key for c in by_slug[slug] if c.fact_key}
            found_on_page = {
                k for k, v in page.get("facts", {}).items()
                if v.get("status") == "found"
            }
            gate[slug] = {
                "required": required,
                "present": sorted(present),
                "missing_fact_chunks": sorted(
                    (found_on_page | set(required)) - present
                ),
            }
    stats["fact_gate"] = gate
    write_json(chunks_dir / "stats.json", stats)
    return stats
