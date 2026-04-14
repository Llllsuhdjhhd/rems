from __future__ import annotations

import logging
from typing import Any

# Chroma 向量库：事件 L1/摘要文本的语义检索入口（回忆服务使用）。距离度量 cosine。

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..config import REMSConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector index for event semantic retrieval.

    使用持久化 Chroma 集合 ``rems_events``，默认 SentenceTransformer 嵌入，距离为 cosine。
    ``add_event`` / ``search`` 为回忆服务提供「事件 L1 或摘要文本 → 向量 → 近邻事件 ID」能力（白皮书 4.4）。
    """

    def __init__(self, config: REMSConfig):
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding.model_name,
        )
        self._client = chromadb.PersistentClient(path=config.storage.chromadb_path)
        self._collection = self._client.get_or_create_collection(
            name="rems_events",
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
