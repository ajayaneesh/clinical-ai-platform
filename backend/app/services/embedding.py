from app.models.embedding import Embedding, EmbeddingModel


class EmbeddingService:
    """Turns images into embeddings (latent vectors) via an EmbeddingModel."""

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model.name

    def embed(self, image: str) -> Embedding:
        return self._model.embed(image)

    def embed_batch(self, images: list[str]) -> list[Embedding]:
        return self._model.embed_batch(images)
