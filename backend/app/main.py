from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.dependencies import get_inference_service
from app.api.routes import router
from app.core.logging import configure_logging
from app.core.middleware import add_logging_middleware
from app.core.queue import LocalQueue
from app.workers.inference_worker import start_worker

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    queue = LocalQueue()
    app.state.queue = queue
    worker_task = start_worker(queue, get_inference_service())
    try:
        yield
    finally:
        worker_task.cancel()


app = FastAPI(
    title="Clinical AI Platform",
    description="API for serving clinical AI inferences.",
    version="0.1.0",
    lifespan=lifespan,
)
add_logging_middleware(app)
app.include_router(router)
