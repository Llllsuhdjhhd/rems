from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, Any

# Chroma 向量库：事件 L1/摘要文本的语义检索入口（回忆服务使用）。距离度量 cosine。

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..config import REMSConfig

if TYPE_CHECKING:
    from ..observability import PerfMonitor

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

    def __init__(self, config: REMSConfig, perf_monitor: "PerfMonitor | None" = None):
        provider = (config.embedding.provider or "local").lower()
        if provider == "hash":
            # Offline-first: never downloads model weights.
            self._ef = _DeterministicHashEmbeddingFunction(dim=128)
        else:
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=config.embedding.model_name,
            )
        self._in_memory = not bool(config.storage.chromadb_path)
        # 可选 PerfMonitor：把 search() 与 search_white_paintings() 包一层 timer，
        # 让"RAG 耗时"参与 load_factor 计算，进而驱动遗忘加速 / 判重剪枝（白皮书 §2.3）。
        self._perf = perf_monitor
        self._memory_events: dict[str, dict[str, Any]] = {}
        self._memory_wp: dict[str, dict[str, Any]] = {}
        if self._in_memory:
            self._client = None
            self._collection = None
            self._wp_collection = None
            return

        if config.storage.chromadb_path:
            self._client = chromadb.PersistentClient(path=config.storage.chromadb_path)
        else:
            self._client = chromadb.Client()
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
    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Coerce metadata into Chroma-compatible primitives.

        Chroma 元数据值仅支持标量（``str/int/float/bool/None``）。
        我们对 list 做扁平化兜底：
            - ``role_ids: ["A", "B"]`` 仍保留为逗号字符串（人类可读字段）；
            - 对每个 role_id 同时写入 ``role_<id>: True`` 这种"白名单标志位"，
              使 70/30 分层检索可以通过 ``$or`` 等值匹配高效过滤（白皮书 §4.4）。
        其他 list / dict 值统一 ``str()`` 兜底，避免 upsert 抛 TypeError。
        """
        if not metadata:
            return {}
        out: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                out[key] = value
            elif isinstance(value, list):
                # role_ids 是常用列表字段：保留可读字符串
                if key == "role_ids":
                    out[key] = ",".join(str(v) for v in value)
                else:
                    out[key] = ",".join(str(v) for v in value)
            else:
                out[key] = str(value)
        return out

    def add_event(
        self,
        event_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = self._sanitize_metadata(metadata)
        if self._in_memory:
            self._memory_events[event_id] = {
                "document": text,
                "metadata": meta,
                "embedding": self._ef.embed_query(text)[0],
            }
            return
        self._collection.upsert(
            ids=[event_id],
            documents=[text],
            metadatas=[meta],
        )

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._perf is not None:
            with self._perf.timer("rag_search"):
                return self._search_inner(query, n_results, where)
        return self._search_inner(query, n_results, where)

    def _search_inner(
        self,
        query: str,
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if self._in_memory:
            return self._memory_search(self._memory_events, query, n_results, where, id_key="event_id")

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
        if self._in_memory:
            self._memory_events.pop(event_id, None)
            return
        try:
            self._collection.delete(ids=[event_id])
        except Exception:
            logger.warning("Failed to delete event %s from vector store", event_id)

    def count(self) -> int:
        if self._in_memory:
            return len(self._memory_events)
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
        meta = dict(metadata or {})
        meta["role_id"] = role_id
        meta["event_id"] = event_id
        meta = self._sanitize_metadata(meta)
        if self._in_memory:
            self._memory_wp[wp_id] = {
                "document": text,
                "metadata": meta,
                "embedding": self._ef.embed_query(text)[0],
            }
            return
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
        if self._perf is not None:
            with self._perf.timer("rag_search"):
                return self._search_white_paintings_inner(query, n_results, where)
        return self._search_white_paintings_inner(query, n_results, where)

    def _search_white_paintings_inner(
        self,
        query: str,
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if self._in_memory:
            return self._memory_search(self._memory_wp, query, n_results, where, id_key="wp_id")

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
        if self._in_memory:
            self._memory_wp.pop(f"{role_id}::{event_id}", None)
            return
        try:
            self._wp_collection.delete(ids=[f"{role_id}::{event_id}"])
        except Exception:
            logger.warning("Failed to delete WP %s::%s from vector store", role_id, event_id)

    def _memory_search(
        self,
        store: dict[str, dict[str, Any]],
        query: str,
        n_results: int,
        where: dict[str, Any] | None,
        *,
        id_key: str,
    ) -> list[dict[str, Any]]:
        query_embedding = self._ef.embed_query(query)[0]
        rows: list[dict[str, Any]] = []
        for item_id, row in store.items():
            metadata = row["metadata"]
            if where and not self._matches_where(metadata, where):
                continue
            distance = 1.0 - self._cosine(query_embedding, row["embedding"])
            rows.append({
                id_key: item_id,
                "document": row["document"],
                "distance": distance,
                "metadata": metadata,
            })
        rows.sort(key=lambda item: item["distance"])
        return rows[:n_results]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        denom = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
        return sum(x * y for x, y in zip(a, b)) / denom

    @classmethod
    def _matches_where(cls, metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        if "$and" in where:
            return all(cls._matches_where(metadata, clause) for clause in where["$and"])
        if "$or" in where:
            return any(cls._matches_where(metadata, clause) for clause in where["$or"])
        for key, expected in where.items():
            actual = metadata.get(key)
            if key == "status" and expected == "active" and actual is None:
                continue
            if isinstance(expected, dict):
                if "$eq" in expected and not (actual == expected["$eq"]):
                    return False
                if "$ne" in expected and not (actual != expected["$ne"]):
                    return False
                if "$gt" in expected and not (actual is not None and actual > expected["$gt"]):
                    return False
                if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                    return False
                if "$lt" in expected and not (actual is not None and actual < expected["$lt"]):
                    return False
                if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                    return False
                if "$in" in expected:
                    if actual not in expected["$in"]:
                        return False
                if "$nin" in expected:
                    if actual in expected["$nin"]:
                        return False
                if "$contains" in expected:
                    # Document-level operator in real Chroma; we approximate on metadata strings/lists.
                    needle = expected["$contains"]
                    if isinstance(actual, str):
                        if needle not in actual:
                            return False
                    elif isinstance(actual, list):
                        if needle not in actual:
                            return False
                    else:
                        return False
            elif actual != expected:
                return False
        return True
