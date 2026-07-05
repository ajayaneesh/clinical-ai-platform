from fastapi import Request

from app.core.queue import Queue
from app.models.inference import InferenceModel
from app.models.torch_model import TorchInferenceModel
from app.services.inference import InferenceService


def get_model() -> InferenceModel:
    """Provide the concrete inference model.

    This is the composition root: the single place that knows which concrete
    model to use. Swap the implementation here — nothing else changes. Tests
    override this via app.dependency_overrides.
    """
    return TorchInferenceModel()


def get_inference_service() -> InferenceService:
    return InferenceService(get_model())


def get_queue(request: Request) -> Queue:
    queue: Queue = request.app.state.queue
    return queue
