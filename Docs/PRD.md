# PRD — Groww MF Facts-Only FAQ Assistant

| Field | Value |
|---|---|
| Product | Groww Mutual Fund Facts-Only FAQ Assistant (RAG demo) |
| Version | v1.0 (draft for build) |
| Status | Ready for implementation |
| Owner | _(assign)_ |
| Build window | 5 phases, phase-gated (see §13) |
| Type | Internal demo / working prototype — not a production Groww surface |

---

## 1. Problem Statement

Retail investors comparing HDFC mutual fund schemes, and the support/content teams who answer the same questions repeatedly, need **fast, trustworthy, single-fact answers** about a scheme: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements.

Today this means opening a long fund page, scrolling past marketing copy and return tables, and hunting for one number. Answers found elsewhere are often unsourced, out of date, or bundled with advice.

**We will build a small, facts-only RAG assistant** over a fixed corpus of **five official Groww pages** (one AMC overview + four HDFC schemes). It answers only factual lookups, always shows **exactly one citation link**, and **politely refuses** anything opinionated, comparative, predictive, or portfolio-related.

### Why this is a RAG problem (and not a chatbot problem)
- The answer is a **verbatim fact** that must be traceable to a page — hallucination is the primary failure mode.
- The corpus is **tiny and fixed** (5 pages), so retrieval precision matters more than recall breadth.
- The guardrail (answer vs. refuse) is a **classification decision made before generation**.

---

## 2. Goals

| # | Goal | Measure |
|---|---|---|
| G1 | Answer single-fact MF queries correctly from the fixed corpus | ≥90% accuracy on the 20-question eval set |
| G2 | Every answer carries one working citation | 100% citation validity, 0 uncited answers |
| G3 | Refuse advice/opinion/portfolio queries without being unhelpful | ≥95% refusal precision on the adversarial set; refusal always paired with an educational link |
| G4 | Make every RAG stage inspectable for demo purposes | 6 stage artifact sets persisted to disk and viewable |
| G5 | Swap embedding model / vector DB / LLM without code changes | 100% of those knobs live in `config/config.yaml` + `.env` |

## 3. Non-Goals

- ❌ Live NAV, current price, or real-time data.
- ❌ Returns, CAGR, alpha/beta, or any **performance computation or comparison**.
- ❌ Fund recommendations, rankings, "should I buy/sell/hold", portfolio allocation, or tax advice.
- ❌ Scraping any link *inside* the five pages (no crawl, depth = 1, exactly 5 fetches).
- ❌ Third-party blogs, aggregators, YouTube, or screenshots as sources.
- ❌ User accounts, auth, chat history persistence, or any PII handling.
- ❌ Multi-AMC or multi-scheme coverage beyond the 5 listed pages.
- ❌ Production deployment, load testing, or SLA guarantees.

---

## 4. Users & Personas

| Persona | Need | Success looks like |
|---|---|---|
| **Retail comparer** | "What's the exit load on HDFC ELSS?" while comparing 3 schemes | One-line answer + link in < 5 s, no sales pitch |
| **Support agent** | Paste-ready factual reply with a defensible source | Answer + citation they can forward verbatim |
| **Content team** | Verify a fact before publishing | Citation resolves to the official page |
| **Evaluator (W1/W2/W3)** | See retrieval and prompting decisions | Stage artifacts + refusal examples are inspectable |

---

## 5. Scope — Corpus (exactly 5 pages, depth 1)

| # | Page type | Scheme / entity | URL |
|---|---|---|---|
| 1 | AMC overview | HDFC Mutual Fund | `https://groww.in/mutual-funds/amc/hdfc-mutual-funds` |
| 2 | Large Cap | HDFC Large Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| 3 | Flexi Cap | HDFC Flexi Cap Fund – Direct Plan Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| 4 | ELSS (tax saver) | HDFC ELSS Tax Saver Fund – Direct Plan Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| 5 | Mid Cap | HDFC Mid Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth` |

**Hard rule:** the scraper fetches these 5 URLs and nothing else. No following anchors, no "related funds", no sitemap.

### 5.1 Facts we must be able to answer (per scheme)
Expense ratio (direct plan) · Exit load · Minimum SIP / minimum lumpsum · ELSS lock-in (scheme 4 only) · Riskometer level · Benchmark index · Fund category & objective · How to download statement / capital-gains statement.

---

## 6. Product Principles

1. **Facts, not opinions.** If the answer requires judgement, we refuse.
2. **One question, one fact, one link.** No multi-fact essays.
3. **Cite or stay silent.** No citation → no answer.
4. **Say "I don't know."** Out-of-corpus facts get an honest miss + the official page link, never a guess.
5. **No PII, ever.** Detect and reject before it touches a log.

---

## 7. Functional Requirements

### FR-1 — Data Loading (Phase 1)
- Fetch each of the 5 URLs once with `requests`/`httpx`, browser-like `User-Agent`, 20 s timeout, 2 retries with backoff, 1 req/s politeness delay.
- Parse to clean text with `trafilatura` (fallback: BeautifulSoup + tag whitelist). **Strip** nav, footer, cookie banners, app-download prompts, disclaimers repeated on every page, and ad slots.
- **Preserve structure**: emit a JSON document per page with `sections[] = {heading, level, text_blocks[], key_value_pairs[]}`. Fund spec blocks (expense ratio, exit load, min SIP…) must survive as key-value pairs, not flattened prose.
- Store raw HTML for auditability alongside cleaned JSON.
- Record `source_url`, `page_title`, `fetched_at` (UTC ISO-8601), `content_hash`.

**Artifacts:** `artifacts/01_raw/{slug}.html`, `artifacts/01_clean/{slug}.json`

### FR-2 — Data Chunking (Phase 2)
Chunking strategy is **layout-aware hybrid**, chosen for this corpus (heterogeneous pages: prose "About the fund", spec key-value grids, tables):

| Chunk type | Rule | Typical size | Purpose |
|---|---|---|---|
| `section` | One chunk per H2/H3 section, split recursively at 400 tokens, 60-token overlap | 150–400 tok | Narrative answers (objective, statement download steps) |
| `fact` | **One chunk per key-value pair**, prefixed with entity context | 15–40 tok | Atomic precision on "Expense ratio of X?" |
| `table_row` | One chunk per row, header prepended | 20–50 tok | Spec grids that parse as tables |

- Fact chunk template: `"{AMC} | {Scheme} | {Plan} | {fact_key}: {fact_value}"` → makes the chunk self-describing even when retrieved alone.
- Reject chunks < 10 tokens or > 800 tokens; deduplicate by content hash.
- Every chunk carries metadata: `chunk_id, amc, scheme, scheme_slug, plan, page_type, chunk_type, section, fact_key, source_url, fetched_at`.
- Chunking parameters live in config, not code.

**Artifacts:** `artifacts/02_chunks/{slug}.jsonl` + `artifacts/02_chunks/stats.json` (counts by type, token histogram)

### FR-3 — Data Embedding (Phase 3)
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-d), `normalize_embeddings=True`.
- Batch 32, CPU by default (`device` configurable), deterministic.
- Cache embeddings keyed by `content_hash + model_name` so re-runs are free and model swaps are detectable.
- Emit an **embedding manifest** (model, dim, count, checksum) so retrieval can assert the index matches the query encoder.

**Artifacts:** `artifacts/03_embeddings/{slug}.npy`, `artifacts/03_embeddings/manifest.json`

### FR-4 — Vector Store (Phase 4)
- ChromaDB persistent client at `artifacts/04_chroma/`, one collection `groww_mf_facts`, distance `cosine`.
- Upsert with `chunk_id` as ID; metadata as above; store raw chunk text in the document field.
- Index build is idempotent and rebuildable from `02_chunks` alone.
- Collection name and persist path are config values.

### FR-5 — Retrieval (Phase 5)
- Embed the query with **the same model** as ingestion (assert via manifest).
- Retrieve `top_k=6` (cosine), apply `score_threshold` (default 0.25), optional MMR (`lambda=0.5`) to avoid 6 near-identical chunks.
- **Scheme resolution:** detect scheme/AMC in the query (alias map: `elss → hdfc-elss-tax-saver…`, `flexi → hdfc-equity-fund…`), then apply a metadata filter. If no scheme is named and the query is scheme-specific, ask a one-line clarifying question listing the 4 schemes.
- **Intent classification before retrieval** (W1): `FACTUAL` → retrieve · `ADVICE/OPINION/PREDICTION/COMPARISON/PORTFOLIO/PII/PERFORMANCE` → refuse without retrieval.
- Persist the retrieval trace (query, rewritten query, filters, top-k ids, scores, text) for the debug view.

**Artifacts:** `artifacts/05_retrieval/{query_id}.json`

### FR-6 — Answer Generation
- LLM via Groq (OpenAI-compatible chat endpoint), `temperature=0.0`, `max_tokens≈220`.
- **Grounded-only prompt**: answer strictly from provided context; no external knowledge; no arithmetic; no comparisons; no advice; if context lacks the fact, say so.
- **Answer contract** (enforced by prompt + post-validator):
  1. Body ≤ **3 sentences**.
  2. Exactly **one** citation URL, taken from retrieved chunk metadata (never generated by the LLM).
  3. Final line: `Last updated from sources: {fetched_at}`.
  4. No numbers that are not verbatim in the context.
  5. No advice verbs ("should", "recommend", "better", "avoid", "buy", "sell", "hold").
- **Post-validator** rejects: >3 sentences, zero or >1 URLs, URL not in the retrieved set, advice vocabulary → regenerate once → else fall back to a templated answer.
- If asked for returns/performance: no computation; return a one-line redirect to the official factsheet/AMC page with the citation.

### FR-7 — Refusal Policy
| Trigger | Example | Response shape |
|---|---|---|
| Advice / suitability | "Should I buy HDFC Flexi Cap?" | Refuse + educational link |
| Comparison / ranking | "Which is better, Large Cap or Mid Cap?" | Refuse + link to both scheme pages |
| Prediction | "Will this fund give 15% next year?" | Refuse + factsheet link |
| Portfolio | "How much of my portfolio should be in ELSS?" | Refuse + link |
| Performance numbers | "What's the 3-year return?" | No computation → factsheet link |
| PII in input | PAN / Aadhaar / account no. / OTP / email / phone | Reject before logging: "I don't need or store personal details." |

Refusal template (2 sentences + link): *"I can only share factual scheme details, so I can't advise whether this fund suits you. You can review the scheme's official page for its objective and risk level: {link}"*

### FR-8 — UI
- Streamlit (or FastAPI + single HTML page), one screen.
- **Welcome line**, **3 example question chips**, and a persistent footer note: **"Facts-only. No investment advice."**
- Every answer renders: body → single citation link (opens the official page) → `Last updated from sources:` line.
- A collapsible **"Show retrieval"** panel per answer (chunk ids, scores, source sections) for demoing the RAG.
- **PII guard on the input box**: pattern-match PAN/Aadhaar/account/OTP/email/phone; block submission, never persist the raw text.

### FR-9 — Stage Inspectability (demo requirement)
Every run writes the 6 artifact sets listed in FR-1…FR-6 plus `artifacts/run_manifest.json` (config snapshot, git-ish hash, timestamps). A `scripts/inspect.py` CLI prints any stage for a given chunk/query id.

### FR-10 — Configuration
All of FR-1…FR-6 parameters are config-driven; secrets only in `.env`.

---

## 8. Configuration Spec

`config/config.yaml` (defaults; env overrides via `pydantic-settings`):

```yaml
app:
  name: "Groww MF Facts-Only FAQ"
  disclaimer: "Facts-only. No investment advice."
  max_answer_sentences: 3

ingestion:
  urls: [ ... 5 urls from §5 ... ]
  user_agent: "Mozilla/5.0 (compatible; GrowwFaqBot/1.0)"
  timeout_s: 20
  retries: 2
  delay_s: 1.0
  respect_robots: true

chunking:
  strategy: "layout_hybrid"        # layout_hybrid | recursive | fixed
  max_tokens: 400
  overlap_tokens: 60
  min_tokens: 10
  fact_chunks: true
  table_rows: true

embedding:
  provider: "sentence-transformers"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32
  normalize: true

vectorstore:
  provider: "chromadb"
  persist_dir: "artifacts/04_chroma"
  collection: "groww_mf_facts"
  distance: "cosine"

retrieval:
  top_k: 6
  score_threshold: 0.25
  use_mmr: true
  mmr_lambda: 0.5
  scheme_filter: true

llm:
  provider: "groq"                 # groq | gemini | openai | anthropic | mistral
  model: "qwen/qwen3-32b"          # override with GROQ_MODEL in .env
  temperature: 0.0
  max_tokens: 220
  api_key_env: "GROQ_API_KEY"
```

`.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=qwen/qwen3-32b
```

> **Note on the requested model ID:** `qwen/qwen3.8-27b` does not resolve to a Groq model as written. The build must pin the exact ID from Groq's model list at implementation time and record it in `.env`; the PRD treats the ID as a config value, not a hard-coded string.

---

## 9. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Latency | p95 end-to-end < 6 s (retrieval < 300 ms on 5-page corpus) |
| Reproducibility | Same config → same chunk ids; rebuild from raw HTML in one command |
| Privacy | No PII in logs, artifacts, or Chroma metadata; input guard runs before any persistence |
| Compliance | Only the 5 specified URLs scraped; citations resolve to official Groww/AMC/SEBI/AMFI pages |
| Cost | Full re-index < 1 min CPU; per-query LLM cost negligible (< 1k tokens) |
| Portability | Runs on Windows/macOS/Linux, CPU-only, Python 3.11 |

---

## 10. Deliverables & Acceptance Criteria

| # | Deliverable | Acceptance criteria |
|---|---|---|
| D1 | Working prototype (local app + ≤3-min demo video if unhosted) | All 5 example queries answered with citation; 3 refusal examples shown |
| D2 | Source list `docs/sources.md` + `sources.csv` — **15–25 URLs** | See §10.1 for how this reconciles with the 5-page scrape rule |
| D3 | `README.md` | Setup steps, scope (AMC + 4 schemes), known limits, config reference |
| D4 | `docs/sample_qa.md` — 8 Q&A pairs | Each has question, answer, single link, refusal example included |
| D5 | Disclaimer snippet in UI | Exact string: "Facts-only. No investment advice." |
| D6 | Stage artifacts (FR-9) | All 6 stage folders populated and inspectable |

### 10.1 Resolving "5 pages scraped" vs "15–25 URLs used"
The corpus is **5 pages**. The source list is a **citation registry** with two tiers:

- **Tier A — corpus (5):** the pages in §5, actually scraped and chunked.
- **Tier B — reference/citation-only (10–20):** official destinations we *link to* but never scrape — HDFC MF scheme pages & factsheet PDFs, AMFI investor-corner/MFI pages, SEBI MF regulations & SCORES, Groww statement-download help. Used for performance redirects, refusal links, and "download statement" answers.

Total 15–25 URLs, all first-party (Groww / HDFC MF / SEBI / AMFI). **No third-party blogs.**

### 10.2 Sample Q&A targets (D4)
1. Expense ratio of HDFC Flexi Cap Direct? → fact + link
2. ELSS lock-in period? → 3 years + link
3. Minimum SIP for HDFC Mid Cap Direct? → fact + link
4. Exit load on HDFC Large Cap Direct? → fact + link
5. Riskometer level of HDFC ELSS Tax Saver? → fact + link
6. Benchmark of HDFC Flexi Cap? → fact + link
7. How do I download my capital-gains statement? → steps + link
8. Should I invest in HDFC Mid Cap? → **refusal** + educational link
9. What were the 3-year returns? → **redirect** to factsheet + link
10. "My PAN is ABCDE1234F, check my folio" → **PII refusal**, nothing stored

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Groww page markup changes / client-side rendering | Ingest returns empty sections | Snapshot raw HTML; detect zero-parse and fail loudly; Playwright fallback (still 5 URLs only) |
| Fact present only in an image or JS widget | Missed fact | Log unparsed key-value candidates in Phase 1 report |
| LLM invents a number | Trust failure | temp 0, grounded-only prompt, "verbatim numbers only" rule, post-validator |
| Citation drift (URL vs. answer) | Compliance failure | Citation is copied from chunk metadata, never generated |
| Over-refusal hurts usability | Demo looks broken | Adversarial + benign eval sets; refusal precision ≥95% while keeping factual accuracy ≥90% |
| Chunking splits a fact from its scheme name | Wrong-scheme answers | Entity-prefixed fact chunks + metadata filter |
| Groq model ID invalid | Blocked at Phase 5 | Validate model ID at startup, fail fast with a clear error |
| Terms-of-use concern on scraping | Legal | 5 public pages, 1 req/s, UA identification, cached snapshots, no redistribution of full text |

---

## 12. Open Questions

1. **Hosting:** local Streamlit only, or a public demo link? (Decides D1 video vs. link.)
2. **Statement-download answer:** the "how to download" content lives on Groww help pages, not the 5 corpus pages — confirm it may be answered from the AMC page section, or add one Tier-B link as the answer target.
3. **Exact Groq model ID** to pin (see §8 note).
4. **Fallback LLM provider** if Groq quota is exhausted mid-demo — Gemini/OpenAI/Anthropic/Mistral adapters are config-selectable; confirm which is the fallback.
5. **Eval set ownership:** who writes the 20-question gold set and the 10 adversarial prompts?

---

## 13. Phase Plan (phase-gated)

| Phase | Deliverable | Exit gate |
|---|---|---|
| **P1 — Data Loading** | 5 raw + 5 clean JSON, parse report | All 5 pages yield the required 7 facts (or documented gaps) |
| **P2 — Chunking** | `.jsonl` per page + stats | Fact chunks exist for expense ratio, exit load, min SIP, lock-in, riskometer, benchmark |
| **P3 — Embedding** | `.npy` + manifest | Dim 384, count matches chunks, checksums recorded |
| **P4 — Vector DB** | Persistent Chroma collection | Rebuild from chunks in one command; sample query returns correct scheme |
| **P5 — Retrieval + LLM + UI** | Working app, refusals, citations, artifacts | §10 acceptance table fully green |

Each phase must be demoable on its own (this is the "show each stage explicitly" requirement).

---

## 14. Success Metrics (demo)

- Factual accuracy on 20-question eval: **≥90%**
- Citation validity: **100%** (all links resolve to an allowlisted domain)
- Refusal precision on 10 adversarial prompts: **≥95%**, refusal always includes a link
- Answer-length compliance (≤3 sentences): **100%**
- PII leakage into artifacts: **0**
- Stages inspectable: **6/6**
