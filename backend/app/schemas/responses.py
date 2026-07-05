from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str


class InferenceResponse(BaseModel):
    prediction: str = Field(
        description="Predicted class label.",
        examples=["normal"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction, from 0 to 1.",
        examples=[0.95],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Image could not be decoded."])
