from fastapi import Request

from app.core.queue import Queue
from app.models.inference import DummyInferenceModel, InferenceModel
from app.services.inference import InferenceService


def get_model() -> InferenceModel:
    """Provide the concrete inference model.

    This is the composition root: the single place that knows which concrete
    model to use. Swap DummyInferenceModel for a real one here — nothing else
    changes. Tests override this via app.dependency_overrides.
    """
    return DummyInferenceModel()


def get_inference_service() -> InferenceService:
    return InferenceService(get_model())


def get_queue(request: Request) -> Queue:
    queue: Queue = request.app.state.queue
    return queue
