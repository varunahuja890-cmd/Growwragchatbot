# Groww MF Facts-Only FAQ Assistant

A facts-only RAG demo over a fixed corpus of **five official Groww pages**
(one HDFC AMC overview + four HDFC schemes). It answers single-fact
questions (expense ratio, exit load, minimum SIP, ELSS lock-in,
riskometer, benchmark) with exactly one citation, and refuses anything
opinionated, comparative, predictive, or portfolio-related.

Product documents: [`Docs/PRD.md`](Docs/PRD.md),
[`Docs/architecture.md`](Docs/architecture.md),
[`Docs/implementation.md`](Docs/implementation.md).

## Status

| Phase | What | Status |
|-------|------|--------|
| 1 | Directory + config + models + registry | ✅ |
| 2 | Scraping (`ingestion/fetch.py`) — the 5 shared links only | ✅ |
| 3 | Loading the document (`ingestion/load.py`) — raw HTML → clean JSON | ✅ |
| 4 | Chunking (`ingestion/chunk.py`) — layout_hybrid | ✅ |
| 5 | Embedding (`ingestion/embed.py`) — MiniLM + manifest | ✅ |
| 6 | Vector DB (`ingestion/index.py`, `vectorstore.py`) — Chroma + verification | ✅ |
| 7 | Retrieval + LLM + guardrails (`query/`, `llm/`) | ✅ |
| 8 | UI (`ui/app.py`, Streamlit) + guardrail tests (`tests/`) | ✅ |

## Setup

Python 3.11, CPU-only, Windows/macOS/Linux.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # then paste your GROQ_API_KEY into .env
```

`.env` holds `LLM_PROVIDER` (groq | openai | anthropic | gemini | mistral),
`GROQ_API_KEY`, and `GROQ_MODEL`. Without an API key the app still runs —
it shows a warning banner and answers fall back to the cited template.

## Run the pipeline (phases 2–6)

```bash
python scripts/ingest.py               # fetch → load → chunk → embed → index
python scripts/ingest.py --from load   # skip the network, restart at load
python scripts/ingest.py --only index  # rebuild the vector DB from artifacts
```

The index stage verifies itself before finishing (this is the "are all
chunks present?" check): the collection count equals the manifest count,
the stored id set equals the manifest `chunk_ids` (missing/extra ids are
listed in `artifacts/run_manifest.json`), a stored vector matches itself
with similarity ≥ 0.99, and a semantic sample query returns a chunk of the
filtered scheme. The stage fails loudly if any gate fails.

Every stage writes inspectable artifacts:

| Stage | Writes |
|-------|--------|
| fetch | `artifacts/01_raw/{slug}.html`, `{slug}.meta.json`, `fetch_report.json` |
| load | `artifacts/01_clean/{slug}.json`, `parse_report.json` |
| chunk | `artifacts/02_chunks/{slug}.jsonl`, `stats.json` |
| embed | `artifacts/03_embeddings/{slug}.npy`, `manifest.json`, `cache/…` |
| index | `artifacts/04_chroma/` (persistent Chroma), `artifacts/run_manifest.json` |

Inspect any stage (FR-9):

```bash
python scripts/inspect.py list
python scripts/inspect.py stage 01_clean --id hdfc-elss-tax-saver-fund-direct-plan-growth
python scripts/inspect.py stage 02_chunks --id hdfc-mid-cap-fund-direct-growth
python scripts/inspect.py stage 03_embeddings --id hdfc-mid-cap-fund-direct-growth
python scripts/inspect.py stage 04_chroma              # chunk-presence check + per-scheme counts
python scripts/inspect.py stage 05_retrieval            # list query ids
python scripts/inspect.py stage 05_retrieval --id <query_id>
```

## Run the app (phase 8)

```bash
streamlit run ui/app.py
```

One screen: three chips (Flexi Cap expense ratio, ELSS lock-in, Large Cap
exit load), a text box, the answer with exactly one link and a
last-updated line, a "Show retrieval" expander reading
`artifacts/05_retrieval/{query_id}.json`, and the footer
"Facts-only. No investment advice."

Startup checks: the manifest matches the configured encoder, the
collection is non-empty, and the configured LLM model id exists at the
provider — a bad model id (e.g. `qwen/qwen3.8-27b` on Groq) stops the app
with that id in the error. A missing/rejected API key shows a warning
banner and answers degrade to the cited template.

## Deploy on Vercel (serverless edition)

The same facts-only pipeline also runs as a static page + one Python
function — no torch, no Chroma, no persistent disk:

- `public/index.html` — Groww-styled UI, POSTs to `/api/ask`.
- `api/ask.py` — mirrors `query/pipeline.py` stage for stage; guardrail
  modules are imported from `src/groww_faq`, and the retrieval trace rides
  in the JSON response (the serverless filesystem is read-only).
- `api/data/` — int8 ONNX MiniLM (query-side, gated ≥ 0.985 mean cosine
  vs the torch encoder) + the torch document vectors and chunk metadata,
  so ranking matches the app's Chroma index.
- `vercel.json` + `api/requirements.txt` — static output from `public/`,
  light function deps only (the root `requirements.txt` is never
  installed into the function).

Deploy: import the repo at vercel.com (Framework Preset **Other**), or
`npx vercel --prod`. Set `GROQ_API_KEY` and `GROQ_MODEL`
(`openai/gpt-oss-120b`) as project environment variables — never commit
the key. Rebuild the index locally with
`python scripts/export_onnx.py` (see `Docs/implementation.md` §9).

## Tests (phase 8)

```bash
python -m pytest tests/ -q
```

PII guard, intent classifier, post-validator, and the full pipeline
(fake store + fake LLM, no network): refusals, clarifications, misses,
redirects, template fallbacks, citation allowlist, and "a PII turn
stores no trace".

## Scope and hard rules

- **Exactly five URLs are fetched** (Tier A in `src/groww_faq/registry.py`):
  no anchors, no "related funds", no internal links, no sitemaps — depth 1,
  one request per page, 1 req/s, robots.txt honored.
- Tier B URLs (HDFC MF, AMFI, SEBI, Groww help) are **cited only, never
  scraped** — see `Docs/sources.md` / `Docs/sources.csv` (15 URLs total).
- Facts are copied verbatim from pages. A missing fact is a documented
  `gap` in the parse report — never invented.
- `chunk_id` is a deterministic hash of (source_url, type, section, fact
  key, text), so the same config rebuilds the same ids.
- No PII is accepted anywhere: the guard runs before any log line, embed
  call, or artifact write, and PII turns leave no trace.
- Query pipeline order: PII → clarify → refusals → performance redirect →
  retrieve → grounded LLM → validate (≤ 3 sentences, no URL, no advice
  words, verbatim numbers; one retry) → template fallback → one citation
  from the registry allowlist.

## Layout

```
config/config.yaml          all non-secret knobs (PRD §8)
.env                        secrets (gitignored; see .env.example)
src/groww_faq/
  config.py                 yaml + env overrides, path anchors
  models.py                 PageDocument, Section, FactPair, Chunk, Hit, Answer, ...
  registry.py               Tier A (scraped) + Tier B (citation-only)
  utils.py                  hashing, timestamps, JSON/JSONL IO
  vectorstore.py            VectorStore seam — Chroma implementation
  ingestion/
    fetch.py                phase 2 — scraping (5 URLs only)
    load.py                 phase 3 — loading the document
    chunk.py                phase 4 — chunking (layout_hybrid)
    embed.py                phase 5 — embedding + manifest
    index.py                phase 6 — Chroma indexing + verification
  query/
    pii.py                  phase 7 — PII guard (first, store nothing)
    intent.py               phase 7 — rules-first classifier
    scheme.py               phase 7 — alias map → scheme_slug
    retrieve.py             phase 7 — filter, top-k, threshold, MMR
    prompt.py               phase 7 — grounded system prompt
    validate.py             phase 7 — post-validator + template
    pipeline.py             phase 7 — orchestrator + retrieval traces
  llm/factory.py            phase 7 — groq/openai/anthropic/gemini/mistral
ui/app.py                   phase 8 — Streamlit one-screen UI
tests/                      phase 8 — guardrail tests (no network)
scripts/ingest.py           pipeline orchestrator (5 stages)
scripts/export_onnx.py      Vercel index builder (ONNX int8, gated)
scripts/inspect.py          stage inspector
api/                        Vercel edition — ask.py + data/ index
public/index.html           Vercel edition — static frontend
vercel.json                 Vercel edition — routing + bundle excludes
artifacts/                  generated state (gitignored)
Docs/                       product docs + sources.md/sources.csv
```

> Note: the PRD's `docs/` folder and the existing `Docs/` product folder are
> the same directory on Windows (case-insensitive filesystem), so
> `sources.md` / `sources.csv` live in `Docs/`.

## Known limits

- Live NAV, returns/CAGR, and any performance computation are out of scope.
- The AMC overview page has no per-scheme spec grid — its facts show as
  `gap` in the parse report by design.
- Statement-download content is not on the five corpus pages; the answer
  path returns a miss with the Tier B help URL.
- If a Groww page is a JS shell, the fetch report marks it
  `needs_browser`; `ingestion/fetch.py:fetch_with_playwright` is the
  documented fallback (still the same URL, still no link following).

## Config reference

See `config/config.yaml` — every non-secret knob (URLs, chunk sizes,
embedding model, batch size, retrieval top-k/threshold/MMR, LLM provider)
is there; provider/model secrets come from `.env`. Changing
`embedding.model_name` requires a re-run of the embed + index stages
(cache is keyed by model + content hash; retrieval refuses to run on a
manifest/config model mismatch).
