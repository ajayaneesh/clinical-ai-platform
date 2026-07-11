import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api import dependencies
from app.api.dependencies import get_inference_service
from app.api.routes import router
from app.core.config import settings
from app.core.embedding_store import InMemoryEmbeddingStore
from app.core.logging import configure_logging
from app.core.metrics import MODEL_COLD_START
from app.core.middleware import add_logging_middleware
from app.core.queue import LocalQueue
from app.core.resources import start_resource_sampler
from app.workers.inference_worker import start_workers

configure_logging()
logger = logging.getLogger("app.startup")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    queue = LocalQueue(timeout=settings.queue_timeout_seconds)
    app.state.queue = queue

    # Cold start: time how long loading the model takes (weights + move to
    # device). Measured once, at startup.
    t0 = time.perf_counter()
    service = get_inference_service()
    cold_start_s = time.perf_counter() - t0
    MODEL_COLD_START.set(cold_start_s)
    logger.info("model_cold_start", extra={"cold_start_s": round(cold_start_s, 3)})

    # Embedding model (BiomedCLIP): load once at startup, only if enabled.
    # /embed returns 503 if this was never set.
    app.state.embedding_service = None
    app.state.embedding_store = InMemoryEmbeddingStore()
    if settings.enable_embeddings:
        app.state.embedding_service = dependencies.build_embedding_service()
        logger.info("embedding_service_loaded")

    worker_tasks = start_workers(
        queue,
        service,
        max_batch_size=settings.max_batch_size,
        max_batch_wait=settings.max_batch_wait_ms / 1000,
        count=settings.worker_count,
    )
    sampler_task = start_resource_sampler(settings.resource_sample_interval_s)
    try:
        yield
    finally:
        for task in worker_tasks:
            task.cancel()
        sampler_task.cancel()


app = FastAPI(
    title="Clinical AI Platform",
    description="API for serving clinical AI inferences.",
    version="0.1.0",
    lifespan=lifespan,
)
add_logging_middleware(app)
app.include_router(router)
