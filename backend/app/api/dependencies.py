from fastapi import Request

from app.core.config import settings
from app.core.embedding_store import EmbeddingStore, InMemoryEmbeddingStore
from app.core.queue import Queue
from app.models.inference import InferenceModel
from app.models.manager import ModelManager
from app.models.torch_model import TorchInferenceModel
from app.services.embedding import EmbeddingService
from app.services.image_processing import ImageProcessingService
from app.services.inference import InferenceService


def get_image_processing() -> ImageProcessingService:
    return ImageProcessingService(max_bytes=settings.max_image_bytes)


def _build_current_model() -> InferenceModel:
    """Construct the one model this deployment serves.

    With a verified CLINICAL_AI_MODEL_ID set, loads the real Hugging Face
    classifier; otherwise falls back to the placeholder TorchInferenceModel
    (fast, no download — used by tests and local dev without a model).
    """
    images = get_image_processing()
    if settings.model_id:
        # Imported lazily so tests / no-model runs don't pull in transformers.
        from app.models.hf_model import HuggingFaceInferenceModel

        return HuggingFaceInferenceModel(settings.model_id, images)
    return TorchInferenceModel(images)


def build_model_manager() -> ModelManager:
    """Composition root for models: build and register the current model.

    Today it registers exactly one model under a default name. Future models
    register additional entries here — nothing downstream changes.
    """
    default_name = settings.model_id or "default"
    manager = ModelManager(default_name=default_name)
    manager.register(default_name, _build_current_model())
    return manager


def get_inference_service() -> InferenceService:
    manager = build_model_manager()
    return InferenceService(manager.get())


def build_embedding_service() -> EmbeddingService:
    """Composition root for the embedding model. Loads the selected model ONCE;
    call this at startup, not per request. Lazy import so no-embedding runs /
    tests don't pull in open_clip."""
    from app.models.biomedclip_model import build_biomedclip, build_laion_clip

    images = get_image_processing()
    builders = {"biomedclip": build_biomedclip, "laion-clip": build_laion_clip}
    try:
        builder = builders[settings.embedding_model]
    except KeyError:
        raise ValueError(
            f"unknown embedding_model '{settings.embedding_model}'; "
            f"choose from {sorted(builders)}"
        )
    return EmbeddingService(builder(images))


def get_embedding_service(request: Request) -> EmbeddingService:
    from fastapi import HTTPException, status

    service = getattr(request.app.state, "embedding_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings are not enabled (set CLINICAL_AI_ENABLE_EMBEDDINGS=true).",
        )
    assert isinstance(service, EmbeddingService)
    return service


def build_embedding_store() -> EmbeddingStore:
    """Composition root for the embedding store: "memory" (default, ephemeral)
    or "qdrant" (persistent, indexed). Lazy import so memory-only runs / tests
    don't require qdrant-client to be reachable."""
    if settings.vector_store == "memory":
        return InMemoryEmbeddingStore()
    if settings.vector_store == "qdrant":
        from app.core.embedding_store import QdrantEmbeddingStore

        return QdrantEmbeddingStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
        )
    raise ValueError(
        f"unknown vector_store '{settings.vector_store}'; choose from ['memory', 'qdrant']"
    )


def get_embedding_store(request: Request) -> EmbeddingStore:
    store: EmbeddingStore = request.app.state.embedding_store
    return store


def get_queue(request: Request) -> Queue:
    queue: Queue = request.app.state.queue
    return queue
