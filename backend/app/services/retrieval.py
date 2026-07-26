from app.core.retrieval import CrossEncoderReranker, Retriever, RetrievedDocument
from app.models.embedding import Embedding


class RetrievalService:
    """Runs a configured Retriever end to end, then reranks the candidate pool
    with a cross-encoder if one is configured: Retriever -> Top N -> Cross
    Encoder -> Top K.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker | None = None,
        rerank_pool_size: int = 20,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rerank_pool_size = rerank_pool_size

    def retrieve(
        self,
        query: str,
        query_vector: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[RetrievedDocument]:
        if self._reranker is None:
            return self._retriever.retrieve(query, query_vector, top_k, label)

        candidates = self._retriever.retrieve(
            query, query_vector, self._rerank_pool_size, label
        )
        return self._reranker.rerank(query, candidates, top_k)
