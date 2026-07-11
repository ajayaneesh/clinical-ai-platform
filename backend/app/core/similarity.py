"""Vector similarity metrics — the three ways to compare embeddings.

Implemented from scratch with numpy so the math is visible:
- dot_product:       raw alignment (sensitive to magnitude)
- cosine_similarity: angle only (magnitude-invariant); == dot product on unit vectors
- euclidean_distance: straight-line distance (a DIS-similarity: smaller = closer)

See docs / your notes on how these relate: on L2-normalized vectors,
cosine == dot product, and L2_distance^2 = 2(1 - cosine).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = list[float]


def _arr(v: Vector) -> NDArray[np.float64]:
    return np.asarray(v, dtype=np.float64)


def dot_product(a: Vector, b: Vector) -> float:
    # Sum of elementwise products. Grows with vector length AND alignment.
    return float(np.dot(_arr(a), _arr(b)))


def cosine_similarity(a: Vector, b: Vector) -> float:
    # Dot product divided by both magnitudes -> the cosine of the angle.
    # Range [-1, 1]; measures direction only, not length.
    va, vb = _arr(a), _arr(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def euclidean_distance(a: Vector, b: Vector) -> float:
    # Straight-line (L2) distance. A DIS-similarity: 0 = identical, larger = farther.
    return float(np.linalg.norm(_arr(a) - _arr(b)))


def find_similar(
    query: Vector,
    candidates: list[Vector],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Rank candidates by cosine similarity to the query.

    Returns [(index, similarity), ...] sorted most-similar first, top_k long.
    Uses cosine so ranking depends on direction (meaning), not magnitude.
    """
    scores = [(i, cosine_similarity(query, c)) for i, c in enumerate(candidates)]
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:top_k]
