"""Phase 7 — retrieval.

Same-encoder assertion, metadata filter, top-k, score threshold, MMR
(architecture.md §6.4):

1. Load ``manifest.json`` and assert the encoder model name matches config.
2. Embed the query with that encoder (the same model as documents).
3. Filter by ``scheme_slug`` when a scheme resolved.
4. Request neighbors by cosine distance (3x candidates when MMR is on, so
   diversification can actually replace near-duplicates).
5. Convert distance to similarity (``1 - d``) *before* the threshold, so
   the config value means similarity.
6. MMR (``lambda`` 0.5) so six near-copies of one fact do not fill the
   context.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from groww_faq.config import AppConfig
from groww_faq.models import Hit
from groww_faq.utils import read_json

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, cfg: AppConfig, store: Any, encoder: Any = None) -> None:
        self.cfg = cfg
        self.rcfg = cfg.retrieval
        self.store = store
        self._encoder = encoder
        self.manifest = self._load_manifest(cfg)

    # ------------------------------------------------------------- manifest
    def _load_manifest(self, cfg: AppConfig) -> dict[str, Any]:
        path = cfg.paths.embeddings / "manifest.json"
        if not path.exists():
            raise RuntimeError(
                "Index is not built. Run python scripts/ingest.py"
            )
        manifest = read_json(path)
        if manifest.get("model_name") != cfg.embedding.model_name:
            raise RuntimeError(
                f"embedding model mismatch: manifest has {manifest.get('model_name')!r} "
                f"but config wants {cfg.embedding.model_name!r} — re-run "
                "python scripts/ingest.py to re-index"
            )
        return manifest

    # -------------------------------------------------------------- encoder
    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.cfg.embedding.model_name, device=self.cfg.embedding.device
            )
        return self._encoder

    def _embed_query(self, question: str) -> np.ndarray:
        encoder = self._ensure_encoder()
        vec = np.asarray(
            encoder.encode(
                [question], normalize_embeddings=self.cfg.embedding.normalize
            )[0],
            dtype=np.float32,
        )
        if vec.shape[0] != int(self.manifest["dim"]):
            raise RuntimeError(
                f"query embedding dim {vec.shape[0]} != manifest dim "
                f"{self.manifest['dim']} — re-run ingest"
            )
        return vec

    # ---------------------------------------------------------------- search
    def search(
        self, question: str, where: dict[str, Any] | None = None
    ) -> tuple[list[Hit], dict[str, Any]]:
        qvec = self._embed_query(question)
        n_results = (
            max(self.rcfg.top_k * 3, 12)
            if self.rcfg.use_mmr
            else self.rcfg.top_k
        )
        res = self.store.query(
            [qvec.tolist()], n_results=n_results, where=where
        )

        hits: list[Hit] = []
        embeddings: list[np.ndarray] = []
        for cid, doc, meta, dist, emb in zip(
            res["ids"], res["documents"], res["metadatas"],
            res["distances"], res["embeddings"],
        ):
            score = 1.0 - float(dist)
            score = max(0.0, min(1.0, score))  # similarity = 1 - distance
            if score < self.rcfg.score_threshold:
                continue
            meta = meta or {}
            hits.append(
                Hit(
                    chunk_id=cid,
                    score=round(score, 4),
                    text=doc,
                    source_url=meta.get("source_url", ""),
                    fetched_at=meta.get("fetched_at", ""),
                    amc=meta.get("amc", ""),
                    scheme=meta.get("scheme", ""),
                    scheme_slug=meta.get("scheme_slug", ""),
                    plan=meta.get("plan", ""),
                    page_type=meta.get("page_type", ""),
                    chunk_type=meta.get("chunk_type", ""),
                    section=meta.get("section", ""),
                    fact_key=meta.get("fact_key", ""),
                )
            )
            embeddings.append(np.asarray(emb, dtype=np.float32))

        if self.rcfg.use_mmr and len(hits) > self.rcfg.top_k:
            order = mmr_order(
                qvec, np.vstack(embeddings), self.rcfg.top_k, self.rcfg.mmr_lambda
            )
            hits = [hits[i] for i in order]
        else:
            hits = hits[: self.rcfg.top_k]

        info = {
            "where": where or {},
            "n_results": n_results,
            "top_k": self.rcfg.top_k,
            "score_threshold": self.rcfg.score_threshold,
            "use_mmr": self.rcfg.use_mmr,
            "mmr_lambda": self.rcfg.mmr_lambda,
        }
        return hits, info


def mmr_order(
    query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lam: float
) -> list[int]:
    """Greedy maximal-marginal-relevance selection.

    Vectors are normalized, so dot products are cosine similarities.
    score = lam * sim(query, doc) - (1 - lam) * max sim(doc, selected).
    """
    q_sim = cand_vecs @ query_vec
    sim = cand_vecs @ cand_vecs.T
    n = cand_vecs.shape[0]
    selected: list[int] = []
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_i, best_score = None, -1e9
        for i in remaining:
            redundancy = max(sim[i][j] for j in selected) if selected else 0.0
            score = lam * float(q_sim[i]) - (1.0 - lam) * float(redundancy)
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected
