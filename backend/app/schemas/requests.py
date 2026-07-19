from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    image: str = Field(
        min_length=1,
        description="Base64-encoded image to classify.",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAUA..."],
    )


class EmbedRequest(BaseModel):
    image: str = Field(
        min_length=1,
        description="Base64-encoded image to embed.",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAUA..."],
    )
    filename: str | None = Field(
        default=None,
        description="Original filename of the image, stored alongside the vector.",
        examples=["patient_042_chest_xray.png"],
    )
    label: str | None = Field(
        default=None,
        description="Diagnosis label to attach to this embedding, if known.",
        examples=["pneumonia"],
    )


class SearchRequest(BaseModel):
    image: str = Field(
        min_length=1,
        description="Base64-encoded query image.",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAUA..."],
    )
    label: str | None = Field(
        default=None,
        description="Restrict the search to stored embeddings with this label.",
        examples=["pneumonia"],
    )
