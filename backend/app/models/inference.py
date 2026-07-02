from typing import Protocol, TypedDict


class InferenceResult(TypedDict):
    prediction: str
    confidence: float


class InferenceModel(Protocol):
    def predict(self, text: str) -> InferenceResult: ...


class DummyInferenceModel:
    def predict(self, text: str) -> InferenceResult:
        return {"prediction": "normal", "confidence": 0.95}
