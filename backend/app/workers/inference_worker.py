import asyncio
import logging
import time

from app.core.metrics import INFERENCE_LATENCY
from app.core.queue import Queue
from app.services.inference import InferenceService

logger = logging.getLogger("app.inference")


class InferenceWorker:
    def __init__(self, queue: Queue, service: InferenceService) -> None:
        self._queue = queue
        self._service = service

    async def run(self) -> None:
        """Consume jobs forever: pull, predict, publish the result back.

        A failing job must NOT kill the worker — we fail that one job and keep
        consuming, so one bad request can't take the service down.
        """
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            # predict() is synchronous and may be slow (a real model does CPU/GPU
            # work). Run it in a thread pool so it doesn't block the event loop.
            # This is the in-process stand-in for the eventual separate worker
            # process; when that lands, this offload moves out of the API entirely.
            start = time.perf_counter()
            try:
                result = await loop.run_in_executor(
                    None, self._service.predict, job.image
                )
            except Exception as exc:
                logger.warning(
                    "inference_failed",
                    extra={"job_id": job.job_id, "error": type(exc).__name__},
                )
                self._queue.fail(job.job_id, exc)
                continue
            elapsed = time.perf_counter() - start

            INFERENCE_LATENCY.observe(elapsed)
            logger.info(
                "inference",
                extra={
                    "job_id": job.job_id,
                    "inference_latency_ms": round(elapsed * 1000, 2),
                },
            )
            self._queue.complete(job.job_id, result)


def start_workers(
    queue: Queue, service: InferenceService, count: int = 1
) -> list[asyncio.Task[None]]:
    """Start `count` workers on the shared queue.

    A single worker awaits each job to completion before pulling the next, so
    throughput is capped at 1 / per-job-latency. Multiple workers pull
    concurrently, scaling throughput ~linearly with count (see
    docs/architecture/performance-baseline.md).
    """
    worker = InferenceWorker(queue, service)
    return [asyncio.create_task(worker.run()) for _ in range(count)]
