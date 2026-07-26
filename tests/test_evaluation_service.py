from app.core.evaluation import EvalQuery
from app.core.retrieval import RetrievedDocument
from app.models.embedding import Embedding
from app.services.evaluation import EvaluationService


class FakeRetriever:
    """Returns a fixed ranking per query text, regardless of query_vector."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings

    def retrieve(
        self,
        query: str,
        query_vector: Embedding,
        top_k: int,
        label: str | None = None,
    ) -> list[RetrievedDocument]:
        ids = self._rankings.get(query, [])
        return [
            RetrievedDocument(embedding_id=doc_id, score=1.0 - i * 0.01)
            for i, doc_id in enumerate(ids[:top_k])
        ]


def _embed_query(query: str) -> Embedding:
    return [0.0]


def test_evaluate_empty_dataset_returns_zeroed_report():
    service = EvaluationService(FakeRetriever({}), _embed_query)
    report = service.evaluate([])
    assert report.num_queries == 0
    assert report.recall_at_5 == 0.0
    assert report.mrr == 0.0


def test_evaluate_perfect_retriever_scores_max_on_every_metric():
    retriever = FakeRetriever({"q1": ["a", "b"], "q2": ["c"]})
    service = EvaluationService(retriever, _embed_query)
    eval_queries = [
        EvalQuery(query="q1", relevance_grades={"a": 1, "b": 1}),
        EvalQuery(query="q2", relevance_grades={"c": 1}),
    ]
    report = service.evaluate(eval_queries, top_k=10)
    assert report.num_queries == 2
    assert report.recall_at_5 == 1.0
    assert report.recall_at_10 == 1.0
    assert report.hit_rate_at_10 == 1.0
    assert report.mrr == 1.0
    assert report.ndcg_at_5 == 1.0


def test_evaluate_averages_across_queries():
    # q1's relevant doc is retrieved first (RR=1); q2's is retrieved second (RR=0.5).
    retriever = FakeRetriever({"q1": ["a", "x"], "q2": ["y", "b"]})
    service = EvaluationService(retriever, _embed_query)
    eval_queries = [
        EvalQuery(query="q1", relevance_grades={"a": 1}),
        EvalQuery(query="q2", relevance_grades={"b": 1}),
    ]
    report = service.evaluate(eval_queries, top_k=10)
    assert report.mrr == (1.0 + 0.5) / 2


def test_evaluate_retriever_with_no_relevant_hits_scores_zero():
    retriever = FakeRetriever({"q1": ["x", "y"]})
    service = EvaluationService(retriever, _embed_query)
    eval_queries = [EvalQuery(query="q1", relevance_grades={"a": 1})]
    report = service.evaluate(eval_queries, top_k=10)
    assert report.recall_at_5 == 0.0
    assert report.mrr == 0.0
    assert report.ndcg_at_5 == 0.0
