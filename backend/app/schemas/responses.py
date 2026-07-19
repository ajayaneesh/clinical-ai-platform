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


class EmbeddingResponse(BaseModel):
    embedding_id: str = Field(
        description="Id of the stored embedding (reference it later).",
        examples=["3f9a1c2e-..."],
    )
    model: str = Field(
        description="Embedding model version that produced the vector.",
        examples=["biomedclip"],
    )
    embedding: list[float] = Field(
        description="Latent vector (L2-normalized) representing the image.",
        examples=[[0.18, -0.43, 0.02]],
    )
    dimension: int = Field(
        description="Length of the embedding vector.",
        examples=[512],
    )
    inference_ms: float = Field(
        description="Time to compute the embedding, in milliseconds.",
        examples=[42.7],
    )
    filename: str | None = Field(
        default=None,
        description="Original filename stored alongside the vector, if provided.",
        examples=["patient_042_chest_xray.png"],
    )
    label: str | None = Field(
        default=None,
        description="Diagnosis label stored alongside the vector, if provided.",
        examples=["pneumonia"],
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp of when the embedding was stored.",
        examples=["2026-07-15T10:32:00+00:00"],
    )


class SearchHitResponse(BaseModel):
    embedding_id: str = Field(examples=["3f9a1c2e-..."])
    score: float = Field(
        description="Cosine similarity to the query (higher = more similar).",
        examples=[0.87],
    )
    model: str = Field(examples=["biomedclip"])
    filename: str | None = Field(default=None, examples=["patient_042_chest_xray.png"])
    label: str | None = Field(default=None, examples=["pneumonia"])
    timestamp: str | None = Field(default=None, examples=["2026-07-15T10:32:00+00:00"])


class SearchResponse(BaseModel):
    results: list[SearchHitResponse] = Field(
        description="Top-k most similar stored images, most similar first."
    )
    searched: int = Field(
        description="Number of stored embeddings compared against.",
        examples=[27],
    )
    embedding_ms: float = Field(
        description="Time to embed the query image, in milliseconds.",
        examples=[42.7],
    )
    search_ms: float = Field(
        description="Time for the similarity search over the store.",
        examples=[1.3],
    )
    store_memory_bytes: int = Field(
        description="Approximate memory held by the stored vectors.",
        examples=[110592],
    )
