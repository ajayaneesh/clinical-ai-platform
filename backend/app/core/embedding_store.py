"""Embedding stores: InMemoryEmbeddingStore (ephemeral, lost on restart) and
QdrantEmbeddingStore (persistent, indexed at scale).

Both sit behind the same EmbeddingStore Protocol so callers don't change when
switching backends — same pattern as LocalQueue -> Kafka.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from app.core.similarity import cosine_similarity
from app.models.embedding import Embedding

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Record


@dataclass
class StoredEmbedding:
    vector: Embedding
    model: str
    dimension: int
    filename: str | None = None
    label: str | None = None
    timestamp: str | None = None
    embedding_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SearchHit:
    embedding_id: str
    score: float
    model: str
    filename: str | None = None
    label: str | None = None
    timestamp: str | None = None


class EmbeddingStore(Protocol):
    def add(
        self,
        vector: Embedding,
        model: str,
        filename: str | None = None,
        label: str | None = None,
        timestamp: str | None = None,
    ) -> str: ...

    def get(self, embedding_id: str) -> StoredEmbedding: ...

    def all(self) -> list[StoredEmbedding]: ...

    def count(self) -> int: ...

    def memory_bytes(self) -> int: ...

    def search(
        self,
        query: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[SearchHit]: ...


class InMemoryEmbeddingStore(EmbeddingStore):
    def __init__(self) -> None:
        self._items: dict[str, StoredEmbedding] = {}

    def add(
        self,
        vector: Embedding,
        model: str,
        filename: str | None = None,
        label: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        item = StoredEmbedding(
            vector=vector,
            model=model,
            dimension=len(vector),
            filename=filename,
            label=label,
            timestamp=timestamp,
        )
        self._items[item.embedding_id] = item
        return item.embedding_id

    def get(self, embedding_id: str) -> StoredEmbedding:
        return self._items[embedding_id]

    def all(self) -> list[StoredEmbedding]:
        return list(self._items.values())

    def count(self) -> int:
        return len(self._items)

    def memory_bytes(self) -> int:
        # Rough estimate: floats are 8 bytes; total = n_vectors * dimension * 8.
        return sum(len(item.vector) * 8 for item in self._items.values())

    def search(
        self,
        query: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[SearchHit]:
        # Brute-force: cosine against every stored vector, then take the top_k.
        # O(n) — fine for an in-memory demo; Qdrant does this indexed at scale.
        candidates: list[StoredEmbedding] = list(self._items.values())
        if label is not None:
            candidates = [item for item in candidates if item.label == label]
        hits = [
            SearchHit(
                item.embedding_id,
                cosine_similarity(query, item.vector),
                item.model,
                filename=item.filename,
                label=item.label,
                timestamp=item.timestamp,
            )
            for item in candidates
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


class QdrantEmbeddingStore:
    """Persistent, indexed embedding store backed by Qdrant.

    Same collection holds vector + payload (filename, label, timestamp) per
    point, so metadata search/filtering is a single round trip. The collection
    is created lazily on the first add(), sized to that vector's dimension —
    every embedding model used against one collection must share a dimension.
    """

    def __init__(self, host: str, port: int, collection: str) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(host=host, port=port)
        self._collection = collection

    @classmethod
    def in_memory(cls, collection: str = "test") -> "QdrantEmbeddingStore":
        """Build against Qdrant's in-process backend (no server) — for tests."""
        from qdrant_client import QdrantClient

        store = cls.__new__(cls)
        store._client = QdrantClient(location=":memory:")
        store._collection = collection
        return store

    def _ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        # Indexed lookup instead of a linear payload scan when filtering by label.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="label",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def add(
        self,
        vector: Embedding,
        model: str,
        filename: str | None = None,
        label: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        from qdrant_client.models import PointStruct

        self._ensure_collection(len(vector))
        embedding_id = str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=embedding_id,
                    vector=vector,
                    payload={
                        # Qdrant normalizes stored vectors under COSINE distance;
                        # keep the original so get()/all() return exact values.
                        "vector": vector,
                        "model": model,
                        "dimension": len(vector),
                        "filename": filename,
                        "label": label,
                        "timestamp": timestamp,
                    },
                )
            ],
        )
        return embedding_id

    def get(self, embedding_id: str) -> StoredEmbedding:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[embedding_id],
        )
        if not points:
            raise KeyError(embedding_id)
        return _stored_embedding_from_point(points[0])

    def all(self) -> list[StoredEmbedding]:
        if not self._client.collection_exists(self._collection):
            return []
        items: list[StoredEmbedding] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                offset=offset,
            )
            items.extend(_stored_embedding_from_point(p) for p in points)
            if offset is None:
                break
        return items

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection).count

    def memory_bytes(self) -> int:
        from qdrant_client.models import VectorParams

        # Qdrant holds vectors out-of-process; approximate the same way the
        # in-memory store does (n_vectors * dimension * 8) using collection info.
        if not self._client.collection_exists(self._collection):
            return 0
        info = self._client.get_collection(self._collection)
        vectors_config = info.config.params.vectors
        dimension = (
            vectors_config.size if isinstance(vectors_config, VectorParams) else 0
        )
        return self.count() * dimension * 8

    def search(
        self,
        query: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[SearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self._collection):
            return []
        query_filter = None
        if label is not None:
            query_filter = Filter(
                must=[FieldCondition(key="label", match=MatchValue(value=label))]
            )
        results = self._client.query_points(
            collection_name=self._collection,
            query=query,
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [
            SearchHit(
                embedding_id=str(point.id),
                score=point.score,
                model=(point.payload or {}).get("model", ""),
                filename=(point.payload or {}).get("filename"),
                label=(point.payload or {}).get("label"),
                timestamp=(point.payload or {}).get("timestamp"),
            )
            for point in results
        ]


def _stored_embedding_from_point(point: Record) -> StoredEmbedding:
    payload = point.payload or {}
    vector = payload.get("vector")
    if not isinstance(vector, list):
        raise TypeError(f"expected a stored vector in payload, got {vector!r}")
    return StoredEmbedding(
        vector=vector,
        model=payload.get("model", ""),
        dimension=payload.get("dimension", len(vector)),
        filename=payload.get("filename"),
        label=payload.get("label"),
        timestamp=payload.get("timestamp"),
        embedding_id=str(point.id),
    )
