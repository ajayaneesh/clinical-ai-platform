"""In-memory store for embeddings — the local, ephemeral precursor to a vector
DB (Qdrant). Holds vectors for the process lifetime; lost on restart.

Behind a Protocol so a persistent/indexed store (QdrantStore) can swap in later
without changing callers — same pattern as LocalQueue -> Kafka.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from app.core.similarity import cosine_similarity
from app.models.embedding import Embedding


@dataclass
class StoredEmbedding:
    vector: Embedding
    model: str
    dimension: int
    embedding_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SearchHit:
    embedding_id: str
    score: float
    model: str


class EmbeddingStore(Protocol):
    def add(self, vector: Embedding, model: str) -> str: ...

    def get(self, embedding_id: str) -> StoredEmbedding: ...

    def all(self) -> list[StoredEmbedding]: ...

    def count(self) -> int: ...

    def memory_bytes(self) -> int: ...

    def search(self, query: Embedding, top_k: int) -> list[SearchHit]: ...


class InMemoryEmbeddingStore:
    def __init__(self) -> None:
        self._items: dict[str, StoredEmbedding] = {}

    def add(self, vector: Embedding, model: str) -> str:
        item = StoredEmbedding(vector=vector, model=model, dimension=len(vector))
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

    def search(self, query: Embedding, top_k: int) -> list[SearchHit]:
        # Brute-force: cosine against every stored vector, then take the top_k.
        # O(n) — fine for an in-memory demo; Qdrant does this indexed at scale.
        hits = [
            SearchHit(
                item.embedding_id, cosine_similarity(query, item.vector), item.model
            )
            for item in self._items.values()
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
