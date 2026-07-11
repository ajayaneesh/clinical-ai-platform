"""Explore embedding similarity on a folder of chest X-ray images.

Embeds every image in a directory with BiomedCLIP, then for a chosen query image
ranks the others by cosine similarity, dot product, and (inverse) Euclidean
distance — printed side by side so you can SEE how the three metrics differ.

Usage:
    CLINICAL_AI_ENABLE_EMBEDDINGS=true \
    uv run python scripts/find_similar_images.py --dir path/to/xrays [--query 0]

Images: any .png/.jpg in --dir. See the script's README note for where to get
sample chest X-rays. First run downloads BiomedCLIP to the HF cache.
"""

from __future__ import annotations

import argparse
import sys
from base64 import b64encode
from pathlib import Path

# Make the app package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.similarity import (  # noqa: E402
    dot_product,
    euclidean_distance,
    find_similar,
)


def _load_images(directory: Path) -> list[tuple[str, str]]:
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no images ({exts}) found in {directory}")
    return [(p.name, b64encode(p.read_bytes()).decode()) for p in files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find similar chest X-rays.")
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--query", type=int, default=0, help="index of query image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Build the real embedding model (loads BiomedCLIP once).
    from app.api.dependencies import build_embedding_service

    service = build_embedding_service()

    named = _load_images(args.dir)
    names = [n for n, _ in named]
    print(f"embedding {len(named)} images with BiomedCLIP...")
    embeddings = service.embed_batch([img for _, img in named])
    dim = len(embeddings[0])
    print(f"embedding dimension: {dim}\n")

    q = args.query
    query_vec = embeddings[q]
    print(f"query: [{q}] {names[q]}\n")

    # Rank by cosine (the recommended metric for meaning).
    ranked = find_similar(query_vec, embeddings, top_k=len(embeddings))
    ranked = [(i, s) for i, s in ranked if i != q][: args.top_k]

    # Show all three metrics side by side for the ranked results.
    print(f"{'img':<28} {'cosine':>8} {'dot':>8} {'euclid':>8}")
    print("-" * 56)
    for i, cos in ranked:
        dot = dot_product(query_vec, embeddings[i])
        euc = euclidean_distance(query_vec, embeddings[i])
        print(f"{names[i]:<28} {cos:>8.4f} {dot:>8.4f} {euc:>8.4f}")

    print(
        "\nNote: BiomedCLIP embeddings are L2-normalized, so cosine == dot, and "
        "smaller euclidean tracks higher cosine (euclidean^2 = 2(1 - cosine))."
    )


if __name__ == "__main__":
    main()
