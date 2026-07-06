from fastapi import Request

from app.core.config import settings
from app.core.queue import Queue
from app.models.inference import InferenceModel
from app.models.torch_model import TorchInferenceModel
from app.services.inference import InferenceService


def get_model() -> InferenceModel:
    """Provide the concrete inference model.

    Composition root: the single place that picks the concrete model. With a
    verified CLINICAL_AI_MODEL_ID set, loads the real Hugging Face classifier;
    otherwise falls back to the placeholder TorchInferenceModel (fast, no
    download — used by tests and local dev without a model).
    """
    if settings.model_id:
        # Imported lazily so tests / no-model runs don't pull in transformers.
        from app.models.hf_model import HuggingFaceInferenceModel

        return HuggingFaceInferenceModel(settings.model_id)
    return TorchInferenceModel()


def get_inference_service() -> InferenceService:
    return InferenceService(get_model())


def get_queue(request: Request) -> Queue:
    queue: Queue = request.app.state.queue
    return queue
