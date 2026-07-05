from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.dependencies import get_inference_service
from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.middleware import add_logging_middleware
from app.core.queue import LocalQueue
from app.workers.inference_worker import start_workers

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    queue = LocalQueue(timeout=settings.queue_timeout_seconds)
    app.state.queue = queue
    worker_tasks = start_workers(
        queue, get_inference_service(), count=settings.worker_count
    )
    try:
        yield
    finally:
        for task in worker_tasks:
            task.cancel()


app = FastAPI(
    title="Clinical AI Platform",
    description="API for serving clinical AI inferences.",
    version="0.1.0",
    lifespan=lifespan,
)
add_logging_middleware(app)
app.include_router(router)
