# Implementation — Groww MF Facts-Only FAQ Assistant

| Field | Value |
|---|---|
| Aligns to | `Docs/architecture.md`, `Docs/PRD.md` v1.0 |
| Runtime | Python 3.11, CPU-only |

Build in this order. Each step reads the previous step’s files under `artifacts/`.

---

## 1. Creating the directory

```
config/config.yaml
.env                        # secrets, gitignored
src/groww_faq/
  config.py                 # load yaml, env overrides secrets
  models.py                 # PageDocument, Chunk, Answer
  registry.py               # 5 corpus URLs + citation-only links
  utils.py                  # slug/content_hash/utc_now helpers
  ingestion/
    fetch.py                # phase 2: raw HTML, no link following
    load.py                 # phase 3: HTML -> structured PageDocument
    chunk.py
    embed.py
    index.py
  query/
    pii.py
    intent.py
    scheme.py
    retrieve.py
    prompt.py
    validate.py
    pipeline.py
  llm/
    factory.py              # groq | gemini | openai | anthropic | mistral
ui/app.py
scripts/ingest.py
scripts/inspect.py
artifacts/                  # generated, gitignored
Docs/sources.md
Docs/sources.csv
tests/
```

On Windows the folder is `Docs/` (case-insensitive filesystem); the PRD's `docs/` paths resolve to the same directory.

Python packages: `httpx`, `trafilatura`, `beautifulsoup4`, `lxml`, `pyyaml`, `pydantic`, `pydantic-settings`, `numpy`, `sentence-transformers`, `chromadb`, `streamlit`, `pytest`.

`config/config.yaml` holds every non-secret knob (URLs, chunk sizes, embedding model, Chroma path, top-k, score threshold, LLM provider and model). `.env` holds `GROQ_API_KEY` and `GROQ_MODEL`. Startup checks that the yaml URL list is exactly the five registry URLs.

Corpus (Tier A — the only pages that are fetched):

| Slug | URL |
|---|---|
| `hdfc-mutual-funds` | `https://groww.in/mutual-funds/amc/hdfc-mutual-funds` |
| `hdfc-large-cap-fund-direct-growth` | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| `hdfc-equity-fund-direct-growth` | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| `hdfc-elss-tax-saver-fund-direct-plan-growth` | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| `hdfc-mid-cap-fund-direct-growth` | `https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth` |

Tier B links (factsheets, statement help, SEBI, AMFI) go in `Docs/sources.csv` and are cited only. They are never fetched. Combined list is 15 official URLs.

```bash
python scripts/ingest.py            # fetch → load → chunk → embed → index
streamlit run ui/app.py
python scripts/inspect.py stage 02_chunks --id <chunk_id>
```

`--from load|chunk|embed|index` restarts at that step and skips the network when HTML is already saved.

---

## 2. Scraping the documents

`ingestion/fetch.py`. This step only saves raw HTML. It never follows links, never crawls internal pages, and never touches Tier B URLs.

- GET each Tier A URL once with `httpx`, the configured User-Agent, a 20s timeout, and 2 retries with backoff.
- Wait 1 second between pages.
- Honor `robots.txt`. If a page redirects off the allowlist, record the failure and save nothing.
- Write `artifacts/01_raw/{slug}.html` and a small `{slug}.meta.json` (`source_url`, status, `fetched_at`).
- Write `artifacts/01_raw/fetch_report.json`: per-slug status, redirect target, `needs_browser` flag, total time.
- If the saved HTML has almost no visible text, mark `needs_browser` and stop. A Playwright fetch of that same URL is the fallback, still with no link following.

---

## 3. Loading the document

`ingestion/load.py`. This step turns saved HTML into structured JSON. It reads `artifacts/01_raw/` only — no network, so a re-load is a pure rebuild.

- Read each `{slug}.html` from `01_raw` and its meta JSON.
- Clean with BeautifulSoup (lxml): drop nav, footer, cookie banners, app-download prompts, and other noise containers. In a second pass, strip disclaimers of 40+ characters that repeat on two or more pages (cross-page boilerplate).
- Keep structure: each H2/H3 becomes a section with `text_blocks`, `key_value_pairs`, and `table_rows` (header cells kept with each row).
- Use trafilatura's cleaned main text as the prose fallback so paragraph sections keep readable text.
- Map labels onto canonical fact keys: `expense_ratio`, `exit_load`, `min_sip`, `min_lumpsum`, `lock_in`, `riskometer`, `benchmark`, `category`, `objective`, `statement_download`. Label-to-value scanning rejects glossary definitions (for example "Exit load — a fee payable…" is not a value) and keeps the real spec value ("Exit load of 1% if redeemed within 1 year").
- Identity fields (`scheme`, `plan`, `slug`, `page_type`) come from the registry, not from page scraping.
- `content_hash` is the SHA-256 of the cleaned body. `fetched_at` is UTC ISO-8601, carried over from the fetch meta.

Write `artifacts/01_clean/{slug}.json` and `artifacts/01_clean/parse_report.json`. The report marks each required fact `found` or `gap`. Lock-in is required only on the ELSS page. A page with zero sections fails the step. A missing fact is recorded as a gap and is not invented.

---

## 4. Chunking

`ingestion/chunk.py`. Strategy `layout_hybrid`. Token counts use the MiniLM tokenizer.

| Type | How it is cut | Size |
|---|---|---|
| `fact` | One chunk per canonical key-value pair | about 15–40 tokens |
| `table_row` | One chunk per row, header cells included | about 20–50 tokens |
| `section` | One chunk per H2/H3, split at 400 tokens with 60-token overlap | about 150–400 tokens |

Fact text is self-describing:

```
{AMC} | {Scheme} | {Plan} | {fact_key}: {fact_value}
```

Every chunk stores `chunk_id`, `amc`, `scheme`, `scheme_slug`, `plan`, `page_type`, `chunk_type`, `section`, `fact_key`, `source_url`, `fetched_at`, `content_hash`. `chunk_id` is a hash of the URL, type, section, fact key, and text, so a rebuild produces the same ids.

Drop chunks under 10 tokens or over 800 tokens, except a canonical fact that is naturally short (for example `exit_load: Nil`). Deduplicate identical text on the same page.

Write `artifacts/02_chunks/{slug}.jsonl` and `artifacts/02_chunks/stats.json` (counts by type, token histogram, fact keys present per slug).

Done when every fact the parse report marked `found` has a fact chunk.

---

## 5. Embedding

`ingestion/embed.py`.

- Model: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, normalized, CPU, batch size 32.
- The same model encodes documents and, later, queries.
- Cache each vector at `artifacts/03_embeddings/cache/{model}/{content_hash}.npy`. A model change uses a different cache folder.
- Write `artifacts/03_embeddings/{slug}.npy` with one row per jsonl line, in that line order.
- Write `artifacts/03_embeddings/manifest.json`: `model_name`, `dim`, `normalize`, `count`, `chunk_ids`, `checksum`.

Retrieval refuses to run if the live encoder’s name or dimension disagrees with the manifest.

Done when `dim` is 384 and `count` equals the number of chunk lines.

---

## 6. Storing data in the vector DB

`ingestion/index.py` using a `VectorStore` wrapper. v1 implements Chroma only.

- Persistent client at `artifacts/04_chroma/`.
- One collection, `groww_mf_facts`, cosine distance.
- Upsert id is `chunk_id`. The document is the chunk text. Metadata is the scalar fields from the chunk (empty string when a value is absent). Pass the precomputed vectors so Chroma does not embed again.
- Default rebuild deletes the collection and upserts from `02_chunks` plus `03_embeddings`, with no network.
- Write `artifacts/run_manifest.json`: timestamp, redacted config, embedding checksum, collection count. API keys appear as `set` or `missing`.

Chroma returns a distance. Convert with `similarity = 1 - distance`. Confirm the conversion by querying a stored vector against itself; similarity should be at least 0.99.

Done when a sample query for a scheme slug (`expense ratio` filtered by `scheme_slug`) returns a chunk of that scheme.

---

## 7. Retrieval logic and UI

`query/pipeline.py` is the only function the UI calls: `answer_question(question) -> Answer`.

```
question
  → PII check          (block; store nothing)
  → intent check       (refuse or redirect; no retrieval)
  → scheme resolve     (ask which scheme if a fact needs one)
  → embed + search     (same model, filter, top 6, threshold, MMR)
  → grounded LLM call  (temperature 0, no URL in the model text)
  → validate           (one retry, then a template from the top chunk)
  → attach one citation from chunk metadata
```

**Scheme filter.** Alias map: `elss` / `tax saver`, `flexi`, `large cap`, `mid cap`, and `hdfc mutual fund` for the AMC page. A scheme-specific question (expense ratio, exit load, SIP, lock-in, riskometer, benchmark) with no scheme returns one clarifying line that names the four schemes.

**Search.** Embed the question. Filter by `scheme_slug` when a scheme was named. Take 6 neighbors, drop similarity below 0.25, then MMR (`lambda` 0.5) so near-duplicate chunks do not fill the context. No hit above the threshold becomes “I don’t know” plus the official page link.

**Generation.** Provider comes from config (default Groq, OpenAI-compatible chat). The system prompt allows only the supplied context: no outside knowledge, no arithmetic, no comparisons, no advice, at most 3 sentences, no URL. The pipeline appends the citation. The URL is `source_url` of the top chunk, and it must be on the registry allowlist.

**Answer shape.** Body, then one link, then `Last updated from sources: {fetched_at}`. Paths stored on the answer: `answer`, `refusal`, `redirect`, `clarify`, `miss`, `template`, `pii_block`.

**Trace.** After the PII check passes, write `artifacts/05_retrieval/{query_id}.json` with the question, intent, filter, hits (id, score, section, fact key, text), path, and citation.

**UI** (`ui/app.py`, one Streamlit screen).

- Title and a one-line welcome.
- Three chips: Flexi Cap expense ratio, ELSS lock-in, Large Cap exit load.
- One text box. PII is rejected in the form before it is saved in session state.
- Answer, one link, last-updated line.
- Expander “Show retrieval” that reads the trace.
- Footer, exact text: `Facts-only. No investment advice.`

Startup of the app checks the manifest, a non-empty collection, and that the configured model id exists at the provider. A bad Groq id (including `qwen/qwen3.8-27b`) stops the app with that id in the error. The real id is `GROQ_MODEL` in `.env`.

`scripts/inspect.py stage <01_raw|01_clean|02_chunks|03_embeddings|04_chroma|05_retrieval> --id <id>` prints that stage for a demo.

---

## 8. Edge cases and guardrail tests

Guardrails run in `pii.py`, `intent.py`, and `validate.py`. Tests live in `tests/test_pii.py`, `tests/test_intent.py`, `tests/test_validate.py`, and `tests/test_pipeline.py`. Pipeline tests use a fake store and a fake LLM, so they do not call the network.

### PII — block before any log, embed, or file

| Input | Expected |
|---|---|
| `My PAN is ABCDE1234F, check my folio` | `pii_block`. Body: “I don't need or store personal details.” No `05_retrieval` file. The PAN does not appear under `artifacts/`. |
| Aadhaar `1234 5678 9012` | `pii_block` |
| An email address or a 10-digit mobile starting 6–9 | `pii_block` |
| `OTP 482193` | `pii_block` |
| `account number 123456789012` | `pii_block` |
| A normal fact question with no such pattern | Guard returns nothing; the pipeline continues |

### Intent — decide before retrieval

Checked in this order: advice, comparison, prediction, portfolio, performance, otherwise factual.

| Input | Expected |
|---|---|
| `Should I invest in HDFC Mid Cap?` | `refusal`. No LLM call. One link: the Mid Cap Groww page. |
| `Which is better, Large Cap or Mid Cap?` | `refusal`. One citation. The other scheme URL is stored on the trace as related. |
| `Will this fund give 15% next year?` | `refusal`. No predicted number in the body. |
| `How much of my portfolio should be in ELSS?` | `refusal` plus an educational link. |
| `What were the 3-year returns of HDFC Flexi Cap?` | `redirect` to that scheme’s factsheet link. No CAGR is computed. |
| Two scheme names in one question | Treated as comparison even if the wording is mild. |
| `What is the expense ratio?` | `clarify`. Lists the four schemes. No LLM call. |

Refusal copy is two sentences plus one link:

> I can only share factual scheme details, so I can't advise whether this fund suits you. You can review the scheme's official page for its objective and risk level: {link}

### Answer checks — after the model returns

The validator rejects more than 3 sentences, any URL in the model text, the words should / recommend / better / avoid / buy / sell / hold, and any number that is not copied from the retrieved text. One regenerate, then a template built from the top chunk.

| Case | Expected |
|---|---|
| Model returns one sentence and the expense ratio exactly as stored | `answer`. Citation equals that chunk’s `source_url`. |
| Model returns four sentences, then a short sentence on retry | `answer` |
| Model includes `https://` or “you should buy” on both tries | `template`. Still one allowlisted link. |
| Model writes `1.25%` when the context says `0.85%` | `invented_number`, then retry or template |
| All hits fall below similarity 0.25 | `miss`. “I don’t know” plus the official page. No LLM call. |
| Statement-download question and the five pages have no such section | `miss` with the Tier B help URL. That URL is not scraped. |
| Embedding manifest model name differs from config | Error asking for a re-index. No generated fact. |
| Chroma directory missing | “Index is not built. Run python scripts/ingest.py.” |
| LLM timeout twice, and a hit exists | `template` from the top chunk |
| Citation URL not in the registry | Rejected. The template or miss path substitutes an allowlisted URL. |

### Ingestion edge cases

| Case | Expected |
|---|---|
| One of the five URLs times out | The other four still save. Exit code 1. The report names the failed slug. |
| Page HTML has no sections | No clean JSON for that slug. Raw HTML kept. |
| Required fact absent on the page | `gap` in the parse report. No invented chunk. |
| Same page chunked twice | Same `chunk_id` values. |
| Re-embed with unchanged hashes | Cache used, encode call count stays 0 for those rows. |
| Index built twice | Collection count equals the chunk count, not double. |

### What “done” looks like

- Five factual questions (expense ratio, lock-in, minimum SIP, exit load, riskometer) each return one cited answer.
- The three refusal examples above behave as in the tables.
- Disclaimer on screen is exactly `Facts-only. No investment advice.`
- `inspect.py` can show all six artifact stages.
- A scan of `artifacts/` finds none of the PII strings used in the tests.
