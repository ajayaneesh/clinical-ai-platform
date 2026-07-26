"""Retrieval: DenseRetriever (vector similarity), SparseRetriever (BM25 lexical
match), and HybridRetriever (fuses both via weighted sum: 0.7 * dense + 0.3 *
BM25, each min-max normalized first).

All three sit behind the same Retriever Protocol so RetrievalService can swap
strategies without callers changing — same pattern as EmbeddingStore.

CrossEncoderReranker sits downstream of any Retriever: Query -> Retriever ->
Top 20 -> Cross Encoder -> Top 5. It only reranks RetrievedDocuments that carry
text (SparseRetriever/HybridRetriever output), since a cross-encoder scores
(query, document text) pairs jointly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.embedding_store import EmbeddingStore
from app.models.embedding import Embedding

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedDocument:
    embedding_id: str
    score: float
    filename: str | None = None
    label: str | None = None
    timestamp: str | None = None
    text: str | None = None


@dataclass
class SparseDocument:
    """One entry in a SparseRetriever's corpus: raw text plus the same
    metadata EmbeddingStore attaches to a vector, so BM25 hits carry the same
    RetrievedDocument shape as dense hits."""

    embedding_id: str
    text: str
    filename: str | None = None
    label: str | None = None
    timestamp: str | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        query_vector: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[RetrievedDocument]: ...


class DenseRetriever:
    """Vector similarity search over an EmbeddingStore (dense embeddings)."""

    def __init__(self, store: EmbeddingStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: str,
        query_vector: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[RetrievedDocument]:
        hits = self._store.search(query_vector, top_k=top_k, label=label)
        return [
            RetrievedDocument(
                embedding_id=hit.embedding_id,
                score=hit.score,
                filename=hit.filename,
                label=hit.label,
                timestamp=hit.timestamp,
            )
            for hit in hits
        ]


class SparseRetriever:
    """Lexical BM25 search: Text Query -> BM25 -> Top K Documents."""

    def __init__(self, corpus: list[SparseDocument]) -> None:
        self._documents = corpus
        self._bm25 = BM25Okapi([_tokenize(doc.text) for doc in corpus])

    def retrieve(
        self,
        query: str,
        query_vector: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[RetrievedDocument]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._documents, scores), key=lambda pair: pair[1], reverse=True
        )
        if label is not None:
            ranked = [pair for pair in ranked if pair[0].label == label]
        return [
            RetrievedDocument(
                embedding_id=doc.embedding_id,
                score=score,
                filename=doc.filename,
                label=doc.label,
                timestamp=doc.timestamp,
                text=doc.text,
            )
            for doc, score in ranked[:top_k]
        ]


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        # All-tied scores (including a single result) carry no ranking signal;
        # treat them as full-strength so they don't get zeroed out below.
        return dict.fromkeys(scores, 1.0)
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in scores.items()}


class HybridRetriever:
    """Fuses DenseRetriever and SparseRetriever scores via simple weighted
    sum: 0.7 * normalized dense score + 0.3 * normalized BM25 score."""

    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever) -> None:
        self._dense = dense
        self._sparse = sparse

    def retrieve(
        self,
        query: str,
        query_vector: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[RetrievedDocument]:
        # Pull a wider candidate pool from each side than top_k: a document
        # ranked highly by one retriever but unseen by the other should still
        # be eligible for fusion, not silently dropped for lack of a BM25 (or
        # dense) score.
        pool_size = max(top_k * 5, top_k)
        dense_hits = self._dense.retrieve(query, query_vector, pool_size, label)
        sparse_hits = self._sparse.retrieve(query, query_vector, pool_size, label)

        dense_scores = _min_max_normalize(
            {hit.embedding_id: hit.score for hit in dense_hits}
        )
        sparse_scores = _min_max_normalize(
            {hit.embedding_id: hit.score for hit in sparse_hits}
        )

        documents: dict[str, RetrievedDocument] = {
            hit.embedding_id: hit for hit in [*dense_hits, *sparse_hits]
        }
        fused_scores = {
            doc_id: self.DENSE_WEIGHT * dense_scores.get(doc_id, 0.0)
            + self.SPARSE_WEIGHT * sparse_scores.get(doc_id, 0.0)
            for doc_id in documents
        }

        ranked_ids = sorted(
            fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True
        )
        results = []
        for doc_id in ranked_ids[:top_k]:
            doc = documents[doc_id]
            results.append(
                RetrievedDocument(
                    embedding_id=doc_id,
                    score=fused_scores[doc_id],
                    filename=doc.filename,
                    label=doc.label,
                    timestamp=doc.timestamp,
                    text=doc.text,
                )
            )
        return results


class CrossEncoderReranker:
    """Reranks a candidate pool by scoring (query, document text) pairs
    jointly through a Hugging Face cross-encoder — more accurate than
    cosine/BM25 similarity but too expensive to run over a whole corpus, so
    it's used only as a final stage: Retriever -> Top 20 -> Cross Encoder ->
    Top 5.

    Construct via build_cross_encoder_reranker() rather than directly.
    Documents with no text (e.g. pure DenseRetriever hits over image
    embeddings) are skipped: there is nothing to pair with the query.
    """

    def __init__(self, model: CrossEncoder, label: str) -> None:
        self._model = model
        self._label = label
        logger.info("reranker_model_loaded", extra={"model": label})

    def rerank(
        self, query: str, documents: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        candidates = [doc for doc in documents if doc.text is not None]
        if not candidates:
            return []

        pairs = [(query, doc.text) for doc in candidates if doc.text is not None]
        scores = self._model.predict(pairs, convert_to_numpy=True)  # type: ignore[arg-type]

        reranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            RetrievedDocument(
                embedding_id=doc.embedding_id,
                score=float(score),
                filename=doc.filename,
                label=doc.label,
                timestamp=doc.timestamp,
                text=doc.text,
            )
            for doc, score in reranked[:top_k]
        ]


def build_cross_encoder_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoderReranker:
    model = CrossEncoder(model_name, device=str(_select_device()))
    return CrossEncoderReranker(model, model_name)
