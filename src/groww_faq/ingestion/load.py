"""Phase 3 — Loading the document.

Reads the HTML snapshots saved by phase 2 from ``artifacts/01_raw/`` (no
network) and loads each page into a typed ``PageDocument`` written to
``artifacts/01_clean/{slug}.json``:

* BeautifulSoup (lxml) supplies the structure — H2/H3 sections, spec
  key-value grids and tables — after nav / footer / cookie banner /
  app-download noise is stripped;
* trafilatura produces the cleaned main text, used as the prose source of
  last resort for facts;
* fact labels are mapped onto the canonical fact keys;
* identity fields (amc, scheme, plan, page_type, slug) come from the
  registry, never from the page;
* ``content_hash`` is the SHA-256 of the cleaned body; ``fetched_at`` is
  the raw fetch timestamp.

A page with zero sections fails the step: no clean JSON is written and the
raw HTML is kept for audit. A missing fact is recorded as a ``gap`` in
``artifacts/01_clean/parse_report.json`` — a gap is never invented.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup, Comment

from groww_faq.config import AppConfig
from groww_faq.models import FactPair, PageDocument, RegistryEntry, Section, TableData
from groww_faq.registry import TIER_A
from groww_faq.utils import read_json, utc_now_iso, write_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical fact keys and how to recognise them on a page
# ---------------------------------------------------------------------------

FACT_KEYS = [
    "expense_ratio",
    "exit_load",
    "min_sip",
    "min_lumpsum",
    "lock_in",
    "riskometer",
    "benchmark",
    "category",
    "objective",
    "statement_download",
]

SCHEME_REQUIRED = (
    "expense_ratio",
    "exit_load",
    "min_sip",
    "min_lumpsum",
    "riskometer",
    "benchmark",
    "category",
    "objective",
)


def required_keys(entry: RegistryEntry) -> list[str]:
    """Facts the parse gate requires: scheme facts; lock-in only on ELSS."""
    if entry.page_type != "scheme":
        return []
    keys = list(SCHEME_REQUIRED)
    if "elss" in entry.slug:
        keys.append("lock_in")
    return keys


# Grid-style labels: "Expense ratio" ... "0.74%" (matched on normalized text).
LABEL_PATTERNS: dict[str, list[str]] = {
    "expense_ratio": [r"^expense ratios?$"],
    "exit_load": [r"^exit loads?$"],
    "min_sip": [
        r"^min(?:imum)?\.? for sip$",
        r"^min(?:imum)? sip investment$",
        r"^min\.? sip$",
    ],
    "min_lumpsum": [
        r"^min(?:imum)?\.? for 1st investment$",
        r"^min(?:imum)? lumpsum investment$",
        r"^min\.? lumpsum$",
    ],
    "lock_in": [r"^lock[- ]?in periods?$", r"^lock[- ]?in$"],
    "riskometer": [r"^riskometer$", r"^risk level$", r"^risk$"],
    "benchmark": [r"^benchmarks?$", r"^benchmark index$", r"^fund benchmark$"],
    "category": [r"^category$", r"^sub[- ]?category$", r"^fund category$"],
    "statement_download": [r"^statement download$", r"^download statement$"],
}

# A DOM value candidate must look like a real value, not a glossary sentence.
VALUE_HINTS: dict[str, str] = {
    "expense_ratio": r"^\d+(\.\d+)?\s*%",
    "exit_load": r"(?i)^(nil|exit load of|\d+(\.\d+)?\s*%)",
    "min_sip": r"(?i)^(₹|rs\.?)\s*[\d,]+",
    "min_lumpsum": r"(?i)^(₹|rs\.?)\s*[\d,]+",
    "lock_in": r"(?i)(\d+\s*years?|no\s+lock|nil)",
    "riskometer": r"(?i)^(very\s+high|high|moderately\s+high|moderate|moderately\s+low|low|very\s+low)(\s+risk)?$",
    "benchmark": r"(?i)(index|tri|nifty|bse|sensex|s&p|crisil)",
    "category": r"^[A-Za-z][A-Za-z /&-]{0,40}$",
    "statement_download": r"(?i)(download|statement)",
}

# Prose-stated facts, with one capture group.
PROSE_PATTERNS: dict[str, list[str]] = {
    "expense_ratio": [r"expense ratio\D{0,24}?(\d+(?:\.\d+)?\s*%)"],
    "exit_load": [r"(?i)\b(exit load of [^.;\n]{2,120})"],
    "min_sip": [
        r"(?i)minimum sip investment(?:\s+is\s+set\s+to|\s+is|\s+of|\s*:)?\s*(₹\s?[\d,]+)"
    ],
    "min_lumpsum": [
        r"(?i)minimum lumpsum investment(?:\s+is\s+set\s+to|\s+is|\s+of|\s*:)?\s*(₹\s?[\d,]+)"
    ],
    "lock_in": [
        r"(?i)lock[- ]?in(?:\s+period)?(?:\s+of|\s+is)?\s*:?\s*(\d+\s*years?)",
        r"(?i)(\d+\s*years?)\s+lock[- ]?in",
    ],
    "riskometer": [
        r"(?i)rated\s+(very\s+low|low|moderately\s+low|moderate|moderately\s+high|high|very\s+high)\s+risk"
    ],
    "benchmark": [
        r"(?i)fund\s+benchmark\s*:?\s*([A-Za-z0-9 &/]+?(?:Total\s+Return\s+Index|TRI|Index))"
    ],
    "category": [
        r"(?i)\bis\s+an?\s+([A-Za-z][A-Za-z /]{1,40}?)\s+Mutual\s+Fund\s+Scheme"
    ],
    "statement_download": [
        r"(?i)(how\s+to\s+download[^.?\n]{0,100}statement[^.?\n]{0,100})"
    ],
}

LABEL_RES = {k: [re.compile(p) for p in v] for k, v in LABEL_PATTERNS.items()}
VALUE_HINT_RES = {k: re.compile(p) for k, p in VALUE_HINTS.items()}
PROSE_RES = {k: [re.compile(p) for p in v] for k, v in PROSE_PATTERNS.items()}

# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

NOISE_SELECTORS = [
    "nav", "footer", "header", "aside", "script", "style", "noscript", "svg",
    "iframe", "form", "template", "button", "link", "meta",
    '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
    '[class*="cookie"]', '[id*="cookie"]', '[class*="consent"]',
    '[class*="banner"]', '[class*="modal"]', '[class*="popup"]',
    '[class*="sidebar"]', '[class*="navbar"]', '[class*="footer"]',
    '[class*="app-download"]', '[class*="download-app"]',
]

_JUNK_RES = [
    re.compile(r"^show more$", re.I),
    re.compile(r"^see all$", re.I),
    re.compile(r"^view details$", re.I),
    re.compile(r"^compare$", re.I),
    re.compile(r"^check past data$", re.I),
    re.compile(r"^was the answer helpful", re.I),
    re.compile(r"^home$", re.I),
]


def _clean_ws(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_junk_block(text: str) -> bool:
    return any(p.match(text) for p in _JUNK_RES)


def _make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # lxml missing or unhappy markup
        return BeautifulSoup(html, "html.parser")


def _strip_noise(soup: BeautifulSoup) -> None:
    """Drop nav, footer, cookie banners, app-download prompts, ad slots."""
    for selector in NOISE_SELECTORS:
        try:
            for el in soup.select(selector):
                el.decompose()
        except Exception:
            continue
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()


def _main_container(soup: BeautifulSoup):
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if main is not None:
        return main
    return soup.body or soup


# ---------------------------------------------------------------------------
# Structure extraction (sections, leaves, tables)
# ---------------------------------------------------------------------------

_BLOCK_TEXT_TAGS = {"p", "li", "dt", "dd", "blockquote", "figcaption", "caption", "pre"}
_LEAFISH_TAGS = {
    "div", "span", "a", "strong", "em", "b", "i", "u", "h4", "h5", "h6",
    "label", "small", "time", "td", "th", "code",
}


def _is_leafish(el) -> bool:
    """An element whose children are only text and <br> — a text leaf."""
    if el.name not in _LEAFISH_TAGS:
        return False
    for child in el.children:
        name = getattr(child, "name", None)
        if name not in (None, "br"):
            return False
    return True


def _parse_table(el) -> TableData | None:
    headers: list[str] = []
    rows: list[list[str]] = []
    thead = el.find("thead")
    if thead is not None:
        headers = [
            _clean_ws(c.get_text(" ", strip=True))
            for c in thead.find_all(["th", "td"])
        ]
    for tr in el.find_all("tr"):
        cells = [
            _clean_ws(c.get_text(" ", strip=True))
            for c in tr.find_all(["td", "th"])
        ]
        if not any(cells):
            continue
        has_th = tr.find("th") is not None
        has_td = tr.find("td") is not None
        if has_th and not has_td:  # header row
            if not headers:
                headers = cells
            continue
        if not has_th and not has_td:
            continue
        rows.append(cells)
    if not rows and not headers:
        return None
    return TableData(headers=[h for h in headers if h], rows=rows)


def _parse_aria_table(el) -> TableData | None:
    """CSS-grid tables marked up with role=table/row/cell."""
    headers: list[str] = []
    rows: list[list[str]] = []
    for row in el.select('[role="row"]'):
        cells = row.select('[role="cell"], [role="columnheader"], [role="rowheader"]')
        texts = [_clean_ws(c.get_text(" ", strip=True)) for c in cells]
        if not any(texts):
            continue
        if row.select('[role="columnheader"]') and not headers:
            headers = texts
            continue
        rows.append(texts)
    if not rows and not headers:
        return None
    return TableData(headers=headers, rows=rows)


def _extract_sections(container) -> tuple[list[Section], list[tuple[int, str]]]:
    """Walk the cleaned DOM: each H2/H3 starts a section; text leaves and
    tables are collected into the current section.

    Returns (sections, leaves) where leaves are (section_index, text) pairs
    in document order — the input to the label→value fact scan.
    """
    sections: list[Section] = [Section(heading="Overview", level=2)]
    leaves: list[tuple[int, str]] = []
    handled: set[int] = set()

    for el in container.find_all(True):
        if id(el) in handled:
            continue
        ancestor = el.parent
        skip = False
        while ancestor is not None:
            if id(ancestor) in handled:
                skip = True
                break
            ancestor = ancestor.parent
        if skip:
            continue

        name = el.name or ""

        if name in ("h2", "h3"):
            heading = _clean_ws(el.get_text(" ", strip=True))
            if heading:
                sections.append(Section(heading=heading, level=int(name[1])))
            handled.add(id(el))
            for sub in el.find_all(True):
                handled.add(id(sub))
            continue

        if name == "table" or (name == "div" and el.get("role") == "table"):
            table = _parse_table(el) if name == "table" else _parse_aria_table(el)
            if table is not None:
                sections[-1].tables.append(table)
            handled.add(id(el))
            for sub in el.find_all(True):
                handled.add(id(sub))
            continue

        if name not in _BLOCK_TEXT_TAGS and not _is_leafish(el):
            continue

        text = _clean_ws(el.get_text(" ", strip=True))
        if len(text) < 3:
            continue
        if not _is_junk_block(text):
            sections[-1].text_blocks.append(text)
            leaves.append((len(sections) - 1, text))
        handled.add(id(el))

    # Drop sections that collected nothing (an empty leading "Overview" or a
    # bare heading) and remap the leaf indices onto the kept sections, so
    # the fact scan never dereferences a stale index.
    kept: list[Section] = []
    remap: dict[int, int] = {}
    for idx, sec in enumerate(sections):
        if sec.text_blocks or sec.tables:
            remap[idx] = len(kept)
            kept.append(sec)
    leaves = [(remap[sec_idx], text) for sec_idx, text in leaves if sec_idx in remap]
    return kept, leaves


def _page_title(soup: BeautifulSoup, entry: RegistryEntry) -> str:
    title = ""
    if soup.title is not None and soup.title.string:
        title = _clean_ws(str(soup.title.string))
    if not title:
        h1 = soup.find("h1")
        if h1 is not None:
            title = _clean_ws(h1.get_text(" ", strip=True))
    if not title:
        title = entry.scheme or entry.amc
    return title[:200]


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------


def _match_label(text: str) -> str | None:
    normalized = _clean_ws(text).lower().rstrip(":")
    if not normalized or len(normalized) > 48:
        return None
    for key, patterns in LABEL_RES.items():
        for pattern in patterns:
            if pattern.match(normalized):
                return key
    return None


def _scan_dom_facts(
    sections: list[Section], leaves: list[tuple[int, str]]
) -> dict[str, FactPair]:
    """Label → value scan over adjacent leaves in the spec grids.

    A label only accepts a following leaf as its value when the value
    passes the key's value hint — this skips glossary definitions
    ("Exit load — a fee payable ...") and takes the real spec value
    ("Exit load of 1% if redeemed within 1 year").
    """
    facts: dict[str, FactPair] = {}
    for i, (sec_idx, text) in enumerate(leaves):
        key = _match_label(text)
        if key is None or key in facts:
            continue
        hint = VALUE_HINT_RES.get(key)
        for j in range(i + 1, min(i + 4, len(leaves))):
            _, candidate = leaves[j]
            if _match_label(candidate) is not None:
                break  # ran into the next label — no value here
            if hint is not None and not hint.search(candidate):
                continue
            value = _clean_ws(candidate)
            if not value or len(value) > 300:
                break
            facts[key] = FactPair(
                key=key,
                label=_clean_ws(text),
                value=value,
                via="dom",
                section=sections[sec_idx].heading,
            )
            break
    return facts


def _objective_from_sections(sections: list[Section]) -> FactPair | None:
    for sec in sections:
        if re.search(r"(?i)(investment\s+objective|^objective\s*$)", sec.heading):
            blocks = [b for b in sec.text_blocks if len(b) >= 30]
            if blocks:
                return FactPair(
                    key="objective",
                    label="Investment Objective",
                    value=blocks[0][:600],
                    via="dom-section",
                    section=sec.heading,
                )
    return None


def _prose_value(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        m = pattern.search(text)
        if m and m.groups():
            value = _clean_ws(m.group(1))
            if value:
                return value
    return None


def _extract_facts(
    sections: list[Section],
    leaves: list[tuple[int, str]],
    trafilatura_text: str | None,
) -> dict[str, FactPair]:
    """DOM grid scan first, then section prose, then trafilatura main text."""
    facts = _scan_dom_facts(sections, leaves)

    objective = _objective_from_sections(sections)
    if objective is not None:
        facts.setdefault("objective", objective)

    for sec in sections:
        text = sec.full_text()
        if not text:
            continue
        for key, patterns in PROSE_RES.items():
            if key in facts:
                continue
            value = _prose_value(patterns, text)
            if value:
                facts[key] = FactPair(
                    key=key,
                    label=key.replace("_", " ").title(),
                    value=value,
                    via="bs4-prose",
                    section=sec.heading,
                )

    if trafilatura_text:
        for key, patterns in PROSE_RES.items():
            if key in facts:
                continue
            value = _prose_value(patterns, trafilatura_text)
            if value:
                facts[key] = FactPair(
                    key=key,
                    label=key.replace("_", " ").title(),
                    value=value,
                    via="trafilatura-prose",
                    section="main text",
                )
    return facts


def _attach_facts(doc: PageDocument, facts: dict[str, FactPair]) -> None:
    for key in FACT_KEYS:
        fact = facts.get(key)
        if fact is None:
            continue
        target = None
        for sec in doc.sections:
            if sec.heading == fact.section:
                target = sec
                break
        if target is None:
            target = doc.sections[0]
        if all(kv.key != key for kv in target.key_value_pairs):
            target.key_value_pairs.append(fact)


# ---------------------------------------------------------------------------
# Boilerplate (disclaimers repeated on every page)
# ---------------------------------------------------------------------------


def _find_boilerplate(docs: list[PageDocument]) -> set[str]:
    """Text blocks of >= 40 chars that appear verbatim on >= 2 pages."""
    counts: dict[str, int] = {}
    for doc in docs:
        seen: set[str] = set()
        for sec in doc.sections:
            for block in sec.text_blocks:
                if len(block) >= 40:
                    seen.add(block)
        for block in seen:
            counts[block] = counts.get(block, 0) + 1
    return {b for b, n in counts.items() if n >= 2}


def _strip_boilerplate(doc: PageDocument, boilerplate: set[str]) -> int:
    removed = 0
    for sec in doc.sections:
        kept = []
        for block in sec.text_blocks:
            if len(block) >= 40 and block in boilerplate:
                removed += 1
            else:
                kept.append(block)
        sec.text_blocks = kept
    return removed


# ---------------------------------------------------------------------------
# One page / all pages
# ---------------------------------------------------------------------------


def _load_one(cfg: AppConfig, entry: RegistryEntry) -> tuple[PageDocument | None, dict[str, Any]]:
    raw_dir = cfg.paths.raw
    html = (raw_dir / f"{entry.slug}.html").read_text(encoding="utf-8", errors="replace")

    fetched_at = utc_now_iso()
    meta_path = raw_dir / f"{entry.slug}.meta.json"
    if meta_path.exists():
        try:
            fetched_at = read_json(meta_path).get("fetched_at") or fetched_at
        except Exception:
            pass

    trafilatura_text: str | None = None
    warnings: list[str] = []
    try:
        import trafilatura

        trafilatura_text = trafilatura.extract(
            html,
            include_comments=False,
            include_links=False,
            include_tables=False,
            favor_recall=True,
        )
    except Exception as exc:  # trafilatura missing or failed on this page
        warnings.append(f"trafilatura_unavailable:{type(exc).__name__}")
    trafilatura_chars = len(trafilatura_text or "")

    soup = _make_soup(html)
    _strip_noise(soup)
    container = _main_container(soup)
    sections, leaves = _extract_sections(container)  # empty sections dropped
    page_title = _page_title(soup, entry)

    if not sections:
        return None, {
            "slug": entry.slug,
            "source_url": entry.url,
            "ok": False,
            "reason": "zero_sections",
            "facts": {k: {"status": "gap"} for k in FACT_KEYS},
            "required_missing": required_keys(entry),
            "warnings": warnings,
        }

    facts = _extract_facts(sections, leaves, trafilatura_text)

    doc = PageDocument(
        source_url=entry.url,
        page_title=page_title,
        page_type=entry.page_type,
        slug=entry.slug,
        amc=entry.amc,
        scheme=entry.scheme,
        plan=entry.plan,
        fetched_at=fetched_at,
        content_hash="",
        sections=sections,
    )
    _attach_facts(doc, facts)

    fact_report: dict[str, Any] = {}
    for key in FACT_KEYS:
        fact = facts.get(key)
        if fact is not None:
            fact_report[key] = {
                "status": "found",
                "value": fact.value,
                "via": fact.via,
                "section": fact.section,
            }
        else:
            fact_report[key] = {"status": "gap"}

    row = {
        "slug": entry.slug,
        "source_url": entry.url,
        "ok": True,
        "page_title": page_title,
        "section_count": len(doc.sections),
        "text_block_count": sum(len(s.text_blocks) for s in doc.sections),
        "kv_pair_count": sum(len(s.key_value_pairs) for s in doc.sections),
        "table_count": sum(len(s.tables) for s in doc.sections),
        "trafilatura_chars": trafilatura_chars,
        "facts": fact_report,
        "required_missing": [k for k in required_keys(entry) if facts.get(k) is None],
        "warnings": warnings,
    }
    return doc, row


def load_all(cfg: AppConfig) -> dict[str, Any]:
    """Load every scraped page. Writes 01_clean/{slug}.json + parse_report.json."""
    clean_dir = cfg.paths.clean
    docs: list[PageDocument] = []
    rows: list[dict[str, Any]] = []

    for entry in TIER_A:
        if not (cfg.paths.raw / f"{entry.slug}.html").exists():
            rows.append({
                "slug": entry.slug,
                "source_url": entry.url,
                "ok": False,
                "reason": "raw_html_missing",
                "facts": {k: {"status": "gap"} for k in FACT_KEYS},
                "required_missing": required_keys(entry),
                "warnings": [],
            })
            logger.error("[%s] no raw HTML — run the fetch stage first", entry.slug)
            continue
        try:
            doc, row = _load_one(cfg, entry)
        except Exception as exc:
            logger.exception("[%s] load failed", entry.slug)
            doc, row = None, {
                "slug": entry.slug,
                "source_url": entry.url,
                "ok": False,
                "reason": f"parse_error:{type(exc).__name__}",
                "facts": {k: {"status": "gap"} for k in FACT_KEYS},
                "required_missing": required_keys(entry),
                "warnings": [],
            }
        if doc is not None:
            docs.append(doc)
        rows.append(row)

    # Second pass: drop disclaimers repeated across pages, then hash.
    boilerplate = _find_boilerplate(docs)
    for doc in docs:
        _strip_boilerplate(doc, boilerplate)
        doc.content_hash = doc.cleaned_body_hash()

    for doc in docs:
        write_json(clean_dir / f"{doc.slug}.json", doc.model_dump(mode="json"))
        logger.info(
            "[%s] loaded: %d sections, %d facts, hash=%s",
            doc.slug,
            doc.section_count,
            sum(len(s.key_value_pairs) for s in doc.sections),
            doc.content_hash[:12],
        )

    report = {
        "generated_at": utc_now_iso(),
        "pages": rows,
        "pages_ok": len(docs),
        "pages_failed": len(rows) - len(docs),
        "boilerplate_blocks_removed": sorted(boilerplate),
        "total_facts_found": sum(
            1
            for r in rows
            if r.get("ok")
            for k in FACT_KEYS
            if r["facts"].get(k, {}).get("status") == "found"
        ),
        "total_gaps": sum(
            1
            for r in rows
            if r.get("ok")
            for k in FACT_KEYS
            if r["facts"].get(k, {}).get("status") == "gap"
        ),
        "required_missing": {
            r["slug"]: r["required_missing"]
            for r in rows
            if r.get("required_missing")
        },
    }
    write_json(clean_dir / "parse_report.json", report)
    return report
