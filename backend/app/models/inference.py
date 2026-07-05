from typing import Protocol, TypedDict


class InvalidImageError(Exception):
    """Raised when the input cannot be decoded/opened as an image."""


class InferenceResult(TypedDict):
    prediction: str
    confidence: float


class InferenceModel(Protocol):
    def predict(self, image: str) -> InferenceResult: ...


class DummyInferenceModel:
    def predict(self, image: str) -> InferenceResult:
        return {"prediction": "normal", "confidence": 0.95}
