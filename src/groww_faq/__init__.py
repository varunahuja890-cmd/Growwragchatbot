"""Groww MF Facts-Only FAQ assistant — RAG demo package.

Phases implemented (see Docs/implementation.md):
1. Directory + config + models + registry
2. Scraping  (ingestion.fetch  — the five Tier A URLs only)
3. Loading   (ingestion.load   — raw HTML -> PageDocument JSON)
4. Chunking  (ingestion.chunk  — layout_hybrid fact / table_row / section)
5. Embedding (ingestion.embed  — MiniLM vectors, cache, manifest)
6. Indexing  (ingestion.index  — Chroma upsert + presence verification)
7. Query     (query.pipeline   — PII -> intent -> retrieve -> LLM -> validate)
8. UI + tests (ui.app Streamlit; tests/ guardrails, no network)
"""
