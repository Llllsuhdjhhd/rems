from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

# Chroma 向量库：事件 L1/摘要文本的语义检索入口（回忆服务使用）。距离度量 cosine。

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..config import REMSConfig

logger = logging.getLogger(__name__)


class _DeterministicHashEmbeddingFunction:
    """Pure-local deterministic embedding with zero external dependencies.

    Generates stable pseudo-vectors from text via SHA-256 expansion.
    It is not semantically rich like sentence-transformers, but it is
    network-free and robust for offline smoke tests / constrained envs.
    """

    is_legacy = False

    def __init__(self, dim: int = 128):
        self._dim = max(8, dim)

    def name(self) -> str:
        return "deterministic-hash"

    def __call__(self, input):  # noqa: A002
        return [self._embed_one(text) for text in input]

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, list):
            return [self._embed_one(str(x)) for x in input]
        return [self._embed_one(str(input))]

    def _embed_one(self, text: str) -> list[float]:
        seed = hashlib.sha256((text or "").encode("utf-8")).digest()
        values: list[float] = []
        counter = 0
        while len(values) < self._dim:
            block = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            for b in block:
                values.append((b / 255.0) * 2.0 - 1.0)  # [-1, 1]
                if len(values) >= self._dim:
                    break
            counter += 1

        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class VectorStore:
    """ChromaDB-backed vector index for event semantic retrieval.

    使用持久化 Chroma 集合 ``rems_events``，默认 SentenceTransformer 嵌入，距离为 cosine。
    ``add_event`` / ``search`` 为回忆服务提供「事件 L1 或摘要文本 → 向量 → 近邻事件 ID」能力（白皮书 4.4）。
    """

    def __init__(self, config: REMSConfig):
        provider = (config.embedding.provider or "local").lower()
        if provider == "hash":
            # Offline-first: never downloads model weights.
            self._ef = _DeterministicHashEmbeddingFunction(dim=128)
        else:
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=config.embedding.model_name,
            )
        self._client = chromadb.PersistentClient(path=config.storage.chromadb_path)
        self._collection = self._client.get_or_create_collection(
            name="rems_events",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._wp_collection = self._client.get_or_create_collection(
            name="rems_white_paintings",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    def add_event(
        self,
        event_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._collection.upsert(
            ids=[event_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        items: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        for i, eid in enumerate(ids):
            items.append({
                "event_id": eid,
                "document": docs[i] if i < len(docs) else "",
                "distance": dists[i] if i < len(dists) else 0.0,
                "metadata": metas[i] if i < len(metas) else {},
            })
        return items

    # ------------------------------------------------------------------
    def delete_event(self, event_id: str) -> None:
        try:
            self._collection.delete(ids=[event_id])
        except Exception:
            logger.warning("Failed to delete event %s from vector store", event_id)

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # White-Painting Vector Methods (Stream B)
    # ------------------------------------------------------------------
    def add_white_painting(
        self,
        role_id: str,
        event_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        wp_id = f"{role_id}::{event_id}"
        meta = metadata or {}
        meta["role_id"] = role_id
        meta["event_id"] = event_id
        self._wp_collection.upsert(
            ids=[wp_id],
            documents=[text],
            metadatas=[meta],
        )

    def search_white_paintings(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self._wp_collection.query(**kwargs)

        items: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        for i, wid in enumerate(ids):
            items.append({
                "wp_id": wid,
                "document": docs[i] if i < len(docs) else "",
                "distance": dists[i] if i < len(dists) else 0.0,
                "metadata": metas[i] if i < len(metas) else {},
            })
        return items

    def delete_white_painting(self, role_id: str, event_id: str) -> None:
        try:
            self._wp_collection.delete(ids=[f"{role_id}::{event_id}"])
        except Exception:
            logger.warning("Failed to delete WP %s::%s from vector store", role_id, event_id)
