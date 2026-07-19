from app.core.embedding_store import InMemoryEmbeddingStore, QdrantEmbeddingStore


def test_add_returns_unique_ids():
    store = InMemoryEmbeddingStore()
    id1 = store.add([0.1, 0.2], "m")
    id2 = store.add([0.3, 0.4], "m")
    assert id1 != id2
    assert store.count() == 2


def test_get_returns_stored_embedding():
    store = InMemoryEmbeddingStore()
    eid = store.add([0.1, 0.2, 0.3], "biomedclip")
    stored = store.get(eid)
    assert stored.vector == [0.1, 0.2, 0.3]
    assert stored.model == "biomedclip"
    assert stored.dimension == 3
    assert stored.embedding_id == eid


def test_all_returns_everything():
    store = InMemoryEmbeddingStore()
    store.add([0.1], "m")
    store.add([0.2], "m")
    assert len(store.all()) == 2


def test_empty_store():
    store = InMemoryEmbeddingStore()
    assert store.count() == 0
    assert store.all() == []


def test_search_ranks_by_cosine():
    store = InMemoryEmbeddingStore()
    id_same = store.add([1.0, 0.0], "m")  # identical direction to query
    store.add([0.0, 1.0], "m")  # orthogonal
    id_opp = store.add([-1.0, 0.0], "m")  # opposite

    hits = store.search([1.0, 0.0], top_k=3)
    assert hits[0].embedding_id == id_same  # most similar first
    assert hits[-1].embedding_id == id_opp  # opposite last
    assert hits[0].score > hits[1].score > hits[2].score


def test_search_respects_top_k():
    store = InMemoryEmbeddingStore()
    for _ in range(10):
        store.add([1.0, 0.0], "m")
    assert len(store.search([1.0, 0.0], top_k=5)) == 5


def test_memory_bytes_scales_with_vectors():
    store = InMemoryEmbeddingStore()
    assert store.memory_bytes() == 0
    store.add([0.0] * 512, "m")
    assert store.memory_bytes() == 512 * 8


def test_add_stores_metadata():
    store = InMemoryEmbeddingStore()
    eid = store.add(
        [0.1, 0.2],
        "m",
        filename="scan.png",
        label="pneumonia",
        timestamp="2026-07-15T00:00:00+00:00",
    )
    stored = store.get(eid)
    assert stored.filename == "scan.png"
    assert stored.label == "pneumonia"
    assert stored.timestamp == "2026-07-15T00:00:00+00:00"


def test_search_filters_by_label():
    store = InMemoryEmbeddingStore()
    id_match = store.add([1.0, 0.0], "m", label="pneumonia")
    store.add([1.0, 0.0], "m", label="normal")

    hits = store.search([1.0, 0.0], top_k=5, label="pneumonia")
    assert [h.embedding_id for h in hits] == [id_match]


# --- QdrantEmbeddingStore ---------------------------------------------------
# Runs against Qdrant's real in-process backend (no server) via
# QdrantEmbeddingStore.in_memory(), so these exercise real Qdrant behavior
# rather than a mock.


def test_qdrant_add_and_get_roundtrips():
    store = QdrantEmbeddingStore.in_memory()
    eid = store.add(
        [0.1, 0.2, 0.3],
        "biomedclip",
        filename="scan.png",
        label="pneumonia",
        timestamp="2026-07-15T00:00:00+00:00",
    )
    stored = store.get(eid)
    assert stored.vector == [0.1, 0.2, 0.3]
    assert stored.model == "biomedclip"
    assert stored.dimension == 3
    assert stored.filename == "scan.png"
    assert stored.label == "pneumonia"
    assert stored.timestamp == "2026-07-15T00:00:00+00:00"
    assert stored.embedding_id == eid


def test_qdrant_empty_store():
    store = QdrantEmbeddingStore.in_memory()
    assert store.count() == 0
    assert store.all() == []
    assert store.memory_bytes() == 0
    assert store.search([1.0, 0.0], top_k=5) == []


def test_qdrant_count_and_all():
    store = QdrantEmbeddingStore.in_memory()
    store.add([0.1, 0.2], "m")
    store.add([0.3, 0.4], "m")
    assert store.count() == 2
    assert len(store.all()) == 2


def test_qdrant_search_ranks_by_similarity():
    store = QdrantEmbeddingStore.in_memory()
    id_same = store.add([1.0, 0.0], "m")
    store.add([0.0, 1.0], "m")
    id_opp = store.add([-1.0, 0.0], "m")

    hits = store.search([1.0, 0.0], top_k=3)
    assert hits[0].embedding_id == id_same
    assert hits[-1].embedding_id == id_opp
    assert hits[0].score > hits[1].score > hits[2].score


def test_qdrant_search_filters_by_label():
    store = QdrantEmbeddingStore.in_memory()
    id_match = store.add([1.0, 0.0], "m", label="pneumonia")
    store.add([1.0, 0.0], "m", label="normal")

    hits = store.search([1.0, 0.0], top_k=5, label="pneumonia")
    assert [h.embedding_id for h in hits] == [id_match]


def test_qdrant_memory_bytes_scales_with_vectors():
    store = QdrantEmbeddingStore.in_memory()
    store.add([0.0] * 512, "m")
    assert store.memory_bytes() == 512 * 8
