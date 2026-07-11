from app.models.embedding import Embedding
from app.services.embedding import EmbeddingService


class FakeEmbeddingModel:
    @property
    def name(self) -> str:
        return "fake"

    def embed(self, image: str) -> Embedding:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[str]) -> list[Embedding]:
        return [[0.1, 0.2, 0.3] for _ in images]


def test_service_delegates_embed():
    service = EmbeddingService(FakeEmbeddingModel())
    assert service.embed("img") == [0.1, 0.2, 0.3]


def test_service_delegates_embed_batch():
    service = EmbeddingService(FakeEmbeddingModel())
    result = service.embed_batch(["a", "b"])
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert len(result) == 2


def test_embedding_model_config_default():
    from app.core.config import Settings

    # Default is whichever model config.py ships; assert it's a known option.
    assert Settings().embedding_model in {"biomedclip", "laion-clip"}


def test_embedding_model_config_selectable(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("CLINICAL_AI_EMBEDDING_MODEL", "laion-clip")
    assert Settings().embedding_model == "laion-clip"


def test_unknown_embedding_model_raises(monkeypatch):
    # An unknown selector must fail loudly at build time, not silently.
    import pytest

    from app.api import dependencies
    from app.core.config import settings

    original = settings.embedding_model
    settings.embedding_model = "does-not-exist"
    try:
        with pytest.raises(ValueError):
            dependencies.build_embedding_service()
    finally:
        settings.embedding_model = original
