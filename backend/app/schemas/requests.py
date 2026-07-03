from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    image: str = Field(
        min_length=1,
        description="Base64-encoded image to classify.",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAUA..."],
    )
