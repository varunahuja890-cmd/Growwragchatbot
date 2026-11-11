"""Phase 5 — Embedding.

Reads ``artifacts/02_chunks/{slug}.jsonl`` (registry order) and writes:

- ``artifacts/03_embeddings/{slug}.npy`` — one row per jsonl line, in that
  line order;
- ``artifacts/03_embeddings/manifest.json`` — model_name, dim, normalize,
  count, chunk_ids, checksum;
- ``artifacts/03_embeddings/cache/{model}/{content_hash}.npy`` — per-chunk
  vectors keyed by content hash + model, so re-runs with unchanged chunks
  are free and a model change misses the cache on purpose.

Model: ``sentence-transformers/all-MiniLM-L6-v2`` (384-d), normalized, CPU,
batch 32 — the same model later encodes queries. Retrieval must refuse to
run when the live encoder's name or dimension disagrees with the manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np

from groww_faq.config import AppConfig
from groww_faq.registry import TIER_A
from groww_faq.utils import read_jsonl, utc_now_iso, write_json

logger = logging.getLogger(__name__)

# Models with a known dimension, used for the phase-5 done check.
EXPECTED_DIM = {"sentence-transformers/all-MiniLM-L6-v2": 384}


class Embedder:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.embed_cfg = cfg.embedding
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the embed stage "
                    f"({self.embed_cfg.model_name})."
                ) from exc
            self._model = SentenceTransformer(
                self.embed_cfg.model_name, device=self.embed_cfg.device
            )
        return self._model

    def embed_all(self) -> dict[str, Any]:
        paths = self.cfg.paths
        model_key = self.embed_cfg.model_name.replace("/", "__")
        cache_dir = paths.embedding_cache / model_key
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Gather chunks in deterministic registry order.
        entries: list[tuple[str, dict[str, Any]]] = []
        slug_rows: dict[str, list[int]] = {}
        for page in TIER_A:
            jsonl_path = paths.chunks / f"{page.slug}.jsonl"
            if not jsonl_path.exists():
                logger.error("[%s] no chunk file — run the chunk stage first", page.slug)
                continue
            rows = read_jsonl(jsonl_path)
            slug_rows[page.slug] = list(
                range(len(entries), len(entries) + len(rows))
            )
            entries.extend((page.slug, row) for row in rows)

        if not entries:
            raise RuntimeError(
                f"no chunk files found under {paths.chunks} — "
                "run `python scripts/ingest.py --from chunk` first"
            )

        # Cache pass: unchanged chunks are not re-encoded.
        vectors: list[np.ndarray | None] = [None] * len(entries)
        to_encode: list[int] = []
        for idx, (_, chunk) in enumerate(entries):
            cache_path = cache_dir / f"{chunk['content_hash']}.npy"
            if cache_path.exists():
                vectors[idx] = np.load(cache_path)
            else:
                to_encode.append(idx)

        if to_encode:
            model = self._ensure_model()
            texts = [entries[idx][1]["text"] for idx in to_encode]
            batch = model.encode(
                texts,
                batch_size=self.embed_cfg.batch_size,
                normalize_embeddings=self.embed_cfg.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for j, idx in enumerate(to_encode):
                vec = np.asarray(batch[j], dtype=np.float32)
                vectors[idx] = vec
                np.save(cache_dir / f"{entries[idx][1]['content_hash']}.npy", vec)

        # Per-slug .npy files, one row per jsonl line, in that line order.
        dim = int(vectors[0].shape[0]) if vectors else 0
        for slug, indices in slug_rows.items():
            if not indices:
                logger.warning(
                    "[%s] no chunks to embed — no .npy written", slug
                )
                continue
            arr = np.vstack([vectors[i] for i in indices])
            np.save(paths.embeddings / f"{slug}.npy", arr)
            logger.info(
                "[%s] embedded %d chunks (dim=%d)", slug, arr.shape[0], dim
            )

        matrix = np.vstack([v for v in vectors])  # type: ignore[arg-type]
        chunk_ids = [chunk["chunk_id"] for _, chunk in entries]

        hasher = hashlib.sha256()
        hasher.update(json.dumps(chunk_ids).encode("utf-8"))
        hasher.update(np.ascontiguousarray(matrix, dtype=np.float32).tobytes())

        expected = EXPECTED_DIM.get(self.embed_cfg.model_name)
        if expected is not None and dim != expected:
            raise ValueError(
                f"embedding dim mismatch: {self.embed_cfg.model_name} should "
                f"produce {expected} dims, got {dim}"
            )

        manifest = {
            "model_name": self.embed_cfg.model_name,
            "dim": dim,
            "normalize": self.embed_cfg.normalize,
            "count": len(entries),
            "chunk_ids": chunk_ids,
            "checksum": hasher.hexdigest(),
            "per_slug": {slug: len(idx) for slug, idx in slug_rows.items()},
            "encoded": len(to_encode),
            "cache_hits": len(entries) - len(to_encode),
            "generated_at": utc_now_iso(),
        }
        write_json(paths.embeddings / "manifest.json", manifest)

        if manifest["count"] != sum(len(idx) for idx in slug_rows.values()):
            raise ValueError("manifest count does not match chunk line count")

        logger.info(
            "manifest: model=%s dim=%d count=%d (encoded %d, cache hits %d)",
            self.embed_cfg.model_name, dim, manifest["count"],
            manifest["encoded"], manifest["cache_hits"],
        )
        return manifest


def embed_all(cfg: AppConfig) -> dict[str, Any]:
    return Embedder(cfg).embed_all()
