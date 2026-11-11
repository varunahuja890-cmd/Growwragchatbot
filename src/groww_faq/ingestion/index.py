"""Phase 6 — Storing data in the vector DB (Chroma).

Reads ``artifacts/02_chunks/{slug}.jsonl`` plus ``artifacts/03_embeddings``
(no network) and upserts every chunk into one persistent Chroma collection:

- persistent client at ``artifacts/04_chroma/``;
- one collection, ``groww_mf_facts``, cosine distance;
- upsert id is ``chunk_id``; the document is the chunk text; metadata is the
  chunk's scalar fields (empty string when a value is absent);
- precomputed vectors are passed so Chroma never embeds again;
- the default rebuild deletes the collection first (idempotent — building
  twice yields the chunk count, not double).

Verification (this is how "are all the chunks present?" is answered):

1. ``collection.count()`` equals the manifest count and the jsonl line count;
2. the set of stored ids equals the set of manifest ``chunk_ids`` — any
   missing or extra id is listed in ``run_manifest.json``;
3. a stored vector queried against itself returns similarity >= 0.99
   (this confirms the distance -> ``similarity = 1 - distance`` conversion);
4. a semantic sample query ("expense ratio" filtered by a scheme slug)
   returns a chunk of that scheme.

Writes ``artifacts/run_manifest.json``: timestamp, redacted config,
embedding checksum, collection count, verification results. API keys appear
as ``set`` or ``missing`` only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from groww_faq.config import AppConfig, get_secrets
from groww_faq.registry import TIER_A
from groww_faq.utils import read_json, read_jsonl, utc_now_iso, write_json
from groww_faq.vectorstore import ChromaStore

logger = logging.getLogger(__name__)

# Chroma caps the size of one upsert batch; stay far below it.
UPSERT_BATCH = 512

# Metadata fields stored per chunk (scalars only — Chroma rejects nested).
META_FIELDS = (
    "amc", "scheme", "scheme_slug", "plan", "page_type", "chunk_type",
    "section", "fact_key", "source_url", "fetched_at", "content_hash",
)


def _load_chunks(cfg: AppConfig) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Manifest + chunk rows + vectors, all in registry order."""
    manifest_path = cfg.paths.embeddings / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"no embedding manifest at {manifest_path} — run "
            "`python scripts/ingest.py --from embed` first"
        )
    manifest = read_json(manifest_path)
    if manifest.get("model_name") != cfg.embedding.model_name:
        raise RuntimeError(
            f"embedding model mismatch: manifest has {manifest.get('model_name')!r} "
            f"but config wants {cfg.embedding.model_name!r} — re-run ingest"
        )

    rows: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    for entry in TIER_A:
        jsonl_path = cfg.paths.chunks / f"{entry.slug}.jsonl"
        npy_path = cfg.paths.embeddings / f"{entry.slug}.npy"
        if not jsonl_path.exists() or not npy_path.exists():
            logger.error("[%s] missing chunk/vectors — run the embed stage first", entry.slug)
            continue
        page_rows = read_jsonl(jsonl_path)
        vectors = np.load(npy_path)
        if vectors.shape[0] != len(page_rows):
            raise RuntimeError(
                f"{entry.slug}: {len(page_rows)} chunk rows but "
                f"{vectors.shape[0]} vectors — re-run the embed stage"
            )
        rows.extend(page_rows)
        matrices.append(vectors)

    if not rows:
        raise RuntimeError(
            f"no chunks found under {cfg.paths.chunks} — run "
            "`python scripts/ingest.py --from chunk` first"
        )
    vectors = np.vstack(matrices).astype(np.float32)
    if len(rows) != manifest.get("count"):
        raise RuntimeError(
            f"manifest count {manifest.get('count')} != loaded rows {len(rows)} "
            "— re-run the embed stage"
        )
    return manifest, rows, vectors


def _metadata(row: dict[str, Any]) -> dict[str, str]:
    return {field: str(row.get(field) or "") for field in META_FIELDS}


def _sample_query(cfg: AppConfig, store: ChromaStore) -> dict[str, Any]:
    """Done gate: 'expense ratio' filtered by scheme_slug returns that scheme."""
    slug = "hdfc-large-cap-fund-direct-growth"
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg.embedding.model_name, device=cfg.embedding.device)
        qvec = model.encode(
            ["expense ratio"], normalize_embeddings=cfg.embedding.normalize
        )
        qvec = np.asarray(qvec[0], dtype=np.float32).tolist()
        res = store.query([qvec], n_results=3, where={"scheme_slug": slug})
        if not res["ids"]:
            return {"question": "expense ratio", "filter": {"scheme_slug": slug},
                    "ok": False, "error": "no hits"}
        meta = res["metadatas"][0]
        top_doc = res["documents"][0]
        ok = (
            meta.get("scheme_slug") == slug
            and (meta.get("fact_key") == "expense_ratio"
                 or "expense ratio" in top_doc.lower())
        )
        return {
            "question": "expense ratio",
            "filter": {"scheme_slug": slug},
            "ok": ok,
            "top_hit": {
                "chunk_id": res["ids"][0],
                "scheme_slug": meta.get("scheme_slug", ""),
                "fact_key": meta.get("fact_key", ""),
                "text": top_doc[:160],
            },
        }
    except Exception as exc:  # encoder unavailable — fail the gate loudly
        logger.exception("sample query failed")
        return {"question": "expense ratio", "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def index_all(cfg: AppConfig) -> dict[str, Any]:
    """Build the collection and verify it. Raises on a failed verification."""
    manifest, rows, vectors = _load_chunks(cfg)
    cfg.paths.chroma.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(cfg.paths.chroma, cfg.vectorstore.collection)
    store.reset()  # idempotent rebuild — never double-insert

    total = len(rows)
    for start in range(0, total, UPSERT_BATCH):
        batch = rows[start:start + UPSERT_BATCH]
        store.upsert(
            ids=[r["chunk_id"] for r in batch],
            embeddings=vectors[start:start + len(batch)].tolist(),
            documents=[r["text"] for r in batch],
            metadatas=[_metadata(r) for r in batch],
        )
    logger.info("upserted %d chunks into %s", total, cfg.vectorstore.collection)

    # ---- gate 1+2: count and id-set equality ----------------------------
    count = store.count()
    stored_ids = set(store.all_ids())
    manifest_ids = set(manifest["chunk_ids"])
    missing = sorted(manifest_ids - stored_ids)
    extra = sorted(stored_ids - manifest_ids)
    all_present = (
        count == manifest["count"] == total
        and not missing
        and not extra
    )

    # ---- gate 3: self-similarity (confirms distance -> similarity) ------
    probe_id = rows[0]["chunk_id"]
    probe_vec = vectors[0].tolist()
    res = store.query([probe_vec], n_results=1)
    self_similarity = max(0.0, 1.0 - float(res["distances"][0]))
    self_sim_ok = res["ids"][0] == probe_id and self_similarity >= 0.99

    # ---- gate 4: semantic sample query ----------------------------------
    sample = _sample_query(cfg, store)

    secrets = get_secrets()
    run_manifest: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "config": cfg.model_dump(mode="json"),
        "secrets": secrets.redacted(),
        "embedding": {
            "model_name": manifest["model_name"],
            "dim": manifest["dim"],
            "normalize": manifest["normalize"],
            "checksum": manifest["checksum"],
        },
        "collection": {
            "name": cfg.vectorstore.collection,
            "distance": cfg.vectorstore.distance,
            "count": count,
            "chunk_count": total,
            "all_chunk_ids_present": all_present,
            "missing_ids": missing[:50],
            "missing_count": len(missing),
            "extra_count": len(extra),
        },
        "self_similarity_check": {
            "probe_chunk_id": probe_id,
            "similarity": round(self_similarity, 6),
            "passed": self_sim_ok,
        },
        "sample_query": sample,
        "ok": all_present and self_sim_ok and sample.get("ok", False),
    }
    write_json(cfg.paths.artifacts / "run_manifest.json", run_manifest)

    if not all_present:
        raise RuntimeError(
            f"index verification failed: {len(missing)} chunk ids missing "
            f"from the collection (see run_manifest.json)"
        )
    if not self_sim_ok:
        raise RuntimeError(
            "self-similarity check failed: "
            f"{self_similarity:.4f} < 0.99 for {probe_id}"
        )
    if not sample.get("ok", False):
        raise RuntimeError(
            "sample query gate failed: 'expense ratio' did not return a "
            f"matching chunk ({sample.get('error', sample.get('top_hit'))})"
        )
    return run_manifest
