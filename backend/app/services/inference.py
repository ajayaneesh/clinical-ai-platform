from app.models.inference import DummyInferenceModel, InferenceModel, InferenceResult


class InferenceService:
    def __init__(self, model: InferenceModel | None = None) -> None:
        self._model = model or DummyInferenceModel()

    def predict(self, text: str) -> InferenceResult:
        return self._model.predict(text)
