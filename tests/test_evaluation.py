from app.core.evaluation import (
    EvalQuery,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.core.retrieval import RetrievedDocument


def _doc(embedding_id: str) -> RetrievedDocument:
    return RetrievedDocument(embedding_id=embedding_id, score=1.0)


def test_recall_at_k_counts_fraction_of_relevant_found():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 1, "b": 1, "c": 1, "d": 1})
    results = [_doc("a"), _doc("x"), _doc("b"), _doc("y"), _doc("z")]
    # top 5 contains a, b out of 4 relevant docs (a, b, c, d) -> 2/4
    assert recall_at_k(results, eval_query, k=5) == 0.5


def test_recall_at_k_no_relevant_docs_is_zero():
    eval_query = EvalQuery(query="q", relevance_grades={})
    assert recall_at_k([_doc("a")], eval_query, k=5) == 0.0


def test_recall_at_k_only_looks_within_cutoff():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 1})
    results = [_doc("x"), _doc("y"), _doc("a")]
    assert recall_at_k(results, eval_query, k=2) == 0.0
    assert recall_at_k(results, eval_query, k=3) == 1.0


def test_hit_rate_at_k_binary_any_hit():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 1, "b": 1})
    assert hit_rate_at_k([_doc("x"), _doc("a")], eval_query, k=10) == 1.0
    assert hit_rate_at_k([_doc("x"), _doc("y")], eval_query, k=10) == 0.0


def test_hit_rate_at_k_no_relevant_docs_is_zero():
    eval_query = EvalQuery(query="q", relevance_grades={})
    assert hit_rate_at_k([_doc("a")], eval_query, k=10) == 0.0


def test_reciprocal_rank_first_hit_position():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 1})
    assert reciprocal_rank([_doc("a"), _doc("x")], eval_query) == 1.0
    assert reciprocal_rank([_doc("x"), _doc("a")], eval_query) == 0.5
    assert reciprocal_rank([_doc("x"), _doc("y"), _doc("a")], eval_query) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 1})
    assert reciprocal_rank([_doc("x"), _doc("y")], eval_query) == 0.0


def test_ndcg_perfect_ranking_is_one():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 3, "b": 2, "c": 1})
    results = [_doc("a"), _doc("b"), _doc("c")]
    assert ndcg_at_k(results, eval_query, k=3) == 1.0


def test_ndcg_reversed_ranking_is_less_than_one():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 3, "b": 2, "c": 1})
    results = [_doc("c"), _doc("b"), _doc("a")]
    score = ndcg_at_k(results, eval_query, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_no_relevant_docs_is_zero():
    eval_query = EvalQuery(query="q", relevance_grades={})
    assert ndcg_at_k([_doc("a")], eval_query, k=5) == 0.0


def test_ndcg_irrelevant_results_score_zero_gain():
    eval_query = EvalQuery(query="q", relevance_grades={"a": 1})
    results = [_doc("x"), _doc("y")]
    assert ndcg_at_k(results, eval_query, k=5) == 0.0
