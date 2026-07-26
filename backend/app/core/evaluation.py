"""Retrieval evaluation metrics: Recall@K, MRR, Hit Rate@K, NDCG@K.

Each metric scores one query's ranked results against its labeled ground
truth, then EvaluationService averages per-query scores across a labeled
EvalDataset to report a single number per metric.

No prebuilt trec_eval dependency: pytrec_eval failed to install (it fetches
a C source tarball from GitHub at build time, blocked in this environment),
so these are hand-rolled directly against RetrievedDocument/EvalQuery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.core.retrieval import RetrievedDocument


@dataclass
class EvalQuery:
    """One labeled example: a query plus its ground-truth relevant docs.

    relevance_grades maps embedding_id -> graded relevance (e.g. 0-3, higher
    is more relevant). Any embedding_id present here is treated as relevant
    for Recall/Hit Rate/MRR; the grade itself is only used by NDCG.
    """

    query: str
    relevance_grades: dict[str, int] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> set[str]:
        return set(self.relevance_grades)


def recall_at_k(
    results: list[RetrievedDocument], eval_query: EvalQuery, k: int
) -> float:
    relevant = eval_query.relevant_ids
    if not relevant:
        return 0.0
    retrieved_ids = {doc.embedding_id for doc in results[:k]}
    return len(retrieved_ids & relevant) / len(relevant)


def hit_rate_at_k(
    results: list[RetrievedDocument], eval_query: EvalQuery, k: int
) -> float:
    relevant = eval_query.relevant_ids
    if not relevant:
        return 0.0
    retrieved_ids = {doc.embedding_id for doc in results[:k]}
    return 1.0 if retrieved_ids & relevant else 0.0


def reciprocal_rank(results: list[RetrievedDocument], eval_query: EvalQuery) -> float:
    relevant = eval_query.relevant_ids
    for rank, doc in enumerate(results, start=1):
        if doc.embedding_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(gains: list[int], k: int) -> float:
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains[:k]))


def ndcg_at_k(results: list[RetrievedDocument], eval_query: EvalQuery, k: int) -> float:
    grades = eval_query.relevance_grades
    if not grades:
        return 0.0
    gains = [grades.get(doc.embedding_id, 0) for doc in results]
    dcg = _dcg_at_k(gains, k)
    ideal_gains = sorted(grades.values(), reverse=True)
    idcg = _dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
