from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLINICAL_AI_", env_file=".env")

    queue_timeout_seconds: float = 30.0
    # One batching worker (Option A) forms the largest batches for a single GPU.
    # Increase only for multiple GPUs (one worker each).
    worker_count: int = 1

    # Hugging Face model repo id for the classifier, e.g.
    # "lxyuan/vit-xray-pneumonia-classification". Verify it exists on
    # huggingface.co before use. Empty string = use the placeholder TorchModel.
    model_id: str = "microsoft/resnet-50"

    # Reject decoded images larger than this many bytes (validation guard).
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MiB

    # Batch inference: the worker collects up to max_batch_size jobs, or waits at
    # most max_batch_wait_ms for the first job's batch to fill, then runs them in
    # one forward pass. Bigger batch = higher GPU throughput; longer wait = more
    # latency. See docs/architecture/performance-baseline.md.
    max_batch_size: int = 8
    max_batch_wait_ms: int = 20

    # How often the background task samples process CPU/memory into gauges.
    resource_sample_interval_s: float = 5.0

    # Load an embedding model at startup so /embed works.
    enable_embeddings: bool = True

    # Which embedding model to use: "biomedclip" (medical, default) or
    # "laion-clip" (general-purpose CLIP). Switch to compare the two on X-rays.
    embedding_model: str = "laion-clip"

    # Which embedding store backs /embed and /search: "memory" (default, no
    # setup, lost on restart) or "qdrant" (persistent, indexed at scale).
    vector_store: str = "memory"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "clinical_embeddings"


settings = Settings()
