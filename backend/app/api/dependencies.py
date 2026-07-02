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
