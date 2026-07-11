from typing import Protocol

# An embedding is a fixed-length vector of floats (the latent representation).
Embedding = list[float]


class EmbeddingModel(Protocol):
    """Produces an embedding (latent vector) for an image."""

    @property
    def name(self) -> str:
        """Identifier of the model (its 'version'), e.g. 'biomedclip'."""
        ...

    def embed(self, image: str) -> Embedding: ...

    def embed_batch(self, images: list[str]) -> list[Embedding]: ...
