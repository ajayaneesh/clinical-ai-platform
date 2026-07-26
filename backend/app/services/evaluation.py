from collections.abc import Callable
from dataclasses import dataclass

from app.core.evaluation import (
    EvalQuery,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.core.retrieval import Retriever
from app.models.embedding import Embedding


@dataclass
class EvalReport:
    """Metrics averaged across every EvalQuery in a dataset."""

    num_queries: int
    recall_at_5: float
    recall_at_10: float
    hit_rate_at_10: float
    mrr: float
    ndcg_at_5: float


class EvaluationService:
    """Runs a Retriever against a labeled set of EvalQuery examples and
    reports Recall@5, Recall@10, Hit Rate@10, MRR, and NDCG@5.

    embed_query turns each EvalQuery's text into the vector the Retriever
    expects — SparseRetriever ignores it, DenseRetriever/HybridRetriever
    need it, so the caller supplies whichever embedding model is under test.
    """

    def __init__(
        self, retriever: Retriever, embed_query: Callable[[str], Embedding]
    ) -> None:
        self._retriever = retriever
        self._embed_query = embed_query

    def evaluate(self, eval_queries: list[EvalQuery], top_k: int = 10) -> EvalReport:
        if not eval_queries:
            return EvalReport(
                num_queries=0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                hit_rate_at_10=0.0,
                mrr=0.0,
                ndcg_at_5=0.0,
            )

        recall_5_scores = []
        recall_10_scores = []
        hit_rate_10_scores = []
        rr_scores = []
        ndcg_5_scores = []

        for eval_query in eval_queries:
            query_vector = self._embed_query(eval_query.query)
            results = self._retriever.retrieve(eval_query.query, query_vector, top_k)

            recall_5_scores.append(recall_at_k(results, eval_query, k=5))
            recall_10_scores.append(recall_at_k(results, eval_query, k=10))
            hit_rate_10_scores.append(hit_rate_at_k(results, eval_query, k=10))
            rr_scores.append(reciprocal_rank(results, eval_query))
            ndcg_5_scores.append(ndcg_at_k(results, eval_query, k=5))

        n = len(eval_queries)
        return EvalReport(
            num_queries=n,
            recall_at_5=sum(recall_5_scores) / n,
            recall_at_10=sum(recall_10_scores) / n,
            hit_rate_at_10=sum(hit_rate_10_scores) / n,
            mrr=sum(rr_scores) / n,
            ndcg_at_5=sum(ndcg_5_scores) / n,
        )
