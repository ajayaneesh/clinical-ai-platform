from typing import Protocol, TypedDict


class InvalidImageError(Exception):
    """Raised when the input cannot be decoded/opened as an image."""


class InferenceResult(TypedDict):
    prediction: str
    confidence: float


class InferenceModel(Protocol):
    def predict(self, image: str) -> InferenceResult: ...

    def predict_batch(self, images: list[str]) -> list[InferenceResult]: ...


class DummyInferenceModel:
    def predict(self, image: str) -> InferenceResult:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[str]) -> list[InferenceResult]:
        return [{"prediction": "normal", "confidence": 0.95} for _ in images]
