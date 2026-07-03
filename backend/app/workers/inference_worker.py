import asyncio

from app.core.queue import Queue
from app.services.inference import InferenceService


class InferenceWorker:
    def __init__(self, queue: Queue, service: InferenceService) -> None:
        self._queue = queue
        self._service = service

    async def run(self) -> None:
        """Consume jobs forever: pull, predict, publish the result back."""
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            # predict() is synchronous and may be slow (a real model does CPU/GPU
            # work). Run it in a thread pool so it doesn't block the event loop.
            # This is the in-process stand-in for the eventual separate worker
            # process; when that lands, this offload moves out of the API entirely.
            result = await loop.run_in_executor(None, self._service.predict, job.image)
            self._queue.complete(job.job_id, result)


def start_worker(queue: Queue, service: InferenceService) -> asyncio.Task[None]:
    worker = InferenceWorker(queue, service)
    return asyncio.create_task(worker.run())
