from pydantic import BaseModel


class RootResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str


class InferenceRequest(BaseModel):
    text: str


class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
