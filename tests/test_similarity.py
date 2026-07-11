import math

from app.core.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    find_similar,
)


def test_dot_product_basic():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0  # 4 + 10 + 18


def test_cosine_identical_direction_is_one():
    # Same direction, different magnitude -> cosine 1.0 (angle-only).
    assert cosine_similarity([1, 0], [5, 0]) == 1.0


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0


def test_cosine_ignores_magnitude_but_dot_does_not():
    # Direction identical -> cosine same; dot product scales with magnitude.
    a = [1, 0]
    assert cosine_similarity(a, [1, 0]) == cosine_similarity(a, [100, 0]) == 1.0
    assert dot_product(a, [1, 0]) == 1.0
    assert dot_product(a, [100, 0]) == 100.0  # dot grew with length


def test_euclidean_distance_basic():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0  # 3-4-5 triangle


def test_cosine_equals_dot_on_unit_vectors():
    # On L2-normalized vectors, cosine == dot product.
    inv = 1 / math.sqrt(2)
    a = [inv, inv]  # unit vector
    b = [1.0, 0.0]  # unit vector
    assert math.isclose(cosine_similarity(a, b), dot_product(a, b), abs_tol=1e-9)


def test_euclidean_relates_to_cosine_on_unit_vectors():
    # euclidean^2 == 2 * (1 - cosine) for unit vectors.
    inv = 1 / math.sqrt(2)
    a = [inv, inv]
    b = [1.0, 0.0]
    cos = cosine_similarity(a, b)
    euc = euclidean_distance(a, b)
    assert math.isclose(euc**2, 2 * (1 - cos), abs_tol=1e-9)


def test_find_similar_ranks_by_cosine():
    query = [1.0, 0.0]
    candidates = [
        [1.0, 0.0],  # 0: identical direction
        [0.0, 1.0],  # 1: orthogonal
        [0.9, 0.1],  # 2: close
        [-1.0, 0.0],  # 3: opposite
    ]
    ranked = find_similar(query, candidates, top_k=4)
    order = [i for i, _ in ranked]
    assert order[0] == 0  # most similar first
    assert order[-1] == 3  # opposite last
    # Similarities are sorted descending.
    sims = [s for _, s in ranked]
    assert sims == sorted(sims, reverse=True)


def test_zero_vector_cosine_is_zero_not_error():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
