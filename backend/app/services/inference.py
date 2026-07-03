from app.models.inference import InferenceModel, InferenceResult


class InferenceService:
    def __init__(self, model: InferenceModel) -> None:
        self._model = model

    def predict(self, image: str) -> InferenceResult:
        return self._model.predict(image)
