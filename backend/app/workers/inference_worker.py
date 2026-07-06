import asyncio
import logging
import time

from app.core.metrics import INFERENCE_LATENCY
from app.core.queue import Job, Queue
from app.services.inference import InferenceService

logger = logging.getLogger("app.inference")


class InferenceWorker:
    def __init__(
        self,
        queue: Queue,
        service: InferenceService,
        max_batch_size: int,
        max_batch_wait: float,
    ) -> None:
        self._queue = queue
        self._service = service
        self._max_batch_size = max_batch_size
        self._max_batch_wait = max_batch_wait

    async def run(self) -> None:
        """Consume jobs in batches forever: collect a batch, run ONE forward
        pass over it, publish each job's result back.

        A failing batch must NOT kill the worker, and one bad image must not fail
        the others — on a batch error we retry each job individually to isolate
        the culprit.
        """
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._queue.get_batch(
                self._max_batch_size, self._max_batch_wait
            )
            await self._process_batch(loop, batch)

    async def _process_batch(
        self, loop: asyncio.AbstractEventLoop, batch: list[Job]
    ) -> None:
        images = [job.image for job in batch]
        start = time.perf_counter()
        try:
            results = await loop.run_in_executor(
                None, self._service.predict_batch, images
            )
        except Exception:
            # One bad image poisoned the batch. Fall back to per-job so the good
            # ones still succeed and only the culprit(s) fail.
            await self._process_individually(loop, batch)
            return

        elapsed = time.perf_counter() - start
        INFERENCE_LATENCY.observe(elapsed)
        logger.info(
            "inference_batch",
            extra={
                "batch_size": len(batch),
                "batch_latency_ms": round(elapsed * 1000, 2),
            },
        )
        for job, result in zip(batch, results):
            self._queue.complete(job.job_id, result)

    async def _process_individually(
        self, loop: asyncio.AbstractEventLoop, batch: list[Job]
    ) -> None:
        for job in batch:
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
            self._queue.complete(job.job_id, result)


def start_workers(
    queue: Queue,
    service: InferenceService,
    max_batch_size: int,
    max_batch_wait: float,
    count: int = 1,
) -> list[asyncio.Task[None]]:
    """Start `count` batching workers on the shared queue.

    Option A: a single batching worker (count=1) is the default — one consumer
    forms the largest batches to feed the scarce GPU. Multiple GPUs would use one
    worker each. See docs/architecture/performance-baseline.md.
    """
    worker = InferenceWorker(queue, service, max_batch_size, max_batch_wait)
    return [asyncio.create_task(worker.run()) for _ in range(count)]
