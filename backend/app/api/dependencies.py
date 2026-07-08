from fastapi import Request

from app.core.config import settings
from app.core.queue import Queue
from app.models.inference import InferenceModel
from app.models.manager import ModelManager
from app.models.torch_model import TorchInferenceModel
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


def get_queue(request: Request) -> Queue:
    queue: Queue = request.app.state.queue
    return queue
