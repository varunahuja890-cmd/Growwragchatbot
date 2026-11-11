"""VectorStore seam — v1 implements Chroma only (architecture.md §10).

The retriever and the indexer both talk to this thin wrapper, never to the
Chroma SDK directly, so a second provider can be added behind the same
protocol without touching the pipelines. All results are normalized to
flat lists (one query at a time).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """The interface both pipelines depend on."""

    def count(self) -> int: ...
    def all_ids(self) -> list[str]: ...
    def get_all(self) -> dict[str, list[Any]]: ...
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None: ...
    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, list[Any]]: ...
    def reset(self) -> None: ...


class ChromaStore:
    """Persistent Chroma client, one collection, cosine distance."""

    def __init__(self, persist_dir: Path, collection: str = "groww_mf_facts") -> None:
        import chromadb  # lazy: importing is heavy and tests use a fake store

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ api
    def count(self) -> int:
        return self._collection.count()

    def all_ids(self) -> list[str]:
        return list(self._collection.get()["ids"])

    def get_all(self) -> dict[str, list[Any]]:
        """Ids + metadata for every row (small corpus — inspection use)."""
        res = self._collection.get(include=["metadatas"])
        return {"ids": list(res["ids"]), "metadatas": list(res["metadatas"])}

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, list[Any]]:
        """Single-query search. Returns flat lists; cosine distance."""
        res = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        return {
            "ids": list(res["ids"][0]),
            "documents": list(res["documents"][0]),
            "metadatas": list(res["metadatas"][0]),
            "distances": list(res["distances"][0]),
            "embeddings": list(res["embeddings"][0]),
        }

    def reset(self) -> None:
        """Idempotent rebuild: delete the collection and recreate it empty."""
        name = self._collection.name
        try:
            self._client.delete_collection(name)
            logger.info("deleted collection %s (idempotent rebuild)", name)
        except Exception:
            pass  # first build — nothing to delete
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
