import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from app.models.inference import InferenceResult

DEFAULT_TIMEOUT_SECONDS = 30.0


class QueueTimeout(Exception):
    """Raised when a submitted job is not completed within the timeout."""


@dataclass
class Job:
    image: str
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class Queue(Protocol):
    """Request/reply queue: submit a job, await its result.

    LocalQueue implements this with asyncio. A future KafkaQueue implements the
    same protocol — swap it in the composition root, nothing else changes.
    """

    async def submit(self, job: Job) -> InferenceResult: ...

    async def get(self) -> Job: ...

    async def get_batch(self, max_size: int, max_wait: float) -> list[Job]: ...

    def complete(self, job_id: str, result: InferenceResult) -> None: ...

    def fail(self, job_id: str, exc: Exception) -> None: ...


class LocalQueue:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        # job_id -> Future the submitting request awaits; the worker resolves it.
        self._results: dict[str, asyncio.Future[InferenceResult]] = {}
        self._timeout = timeout

    async def submit(self, job: Job) -> InferenceResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        self._results[job.job_id] = future
        await self._queue.put(job)
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            raise QueueTimeout(job.job_id)
        finally:
            self._results.pop(job.job_id, None)

    async def get(self) -> Job:
        return await self._queue.get()

    async def get_batch(self, max_size: int, max_wait: float) -> list[Job]:
        """Collect up to max_size jobs, waiting at most max_wait for the batch
        to fill after the first job arrives.

        Blocks for the first job (no busy-waiting when idle), then greedily takes
        whatever is already queued, and keeps waiting for stragglers until either
        max_size is reached or max_wait elapses.
        """
        loop = asyncio.get_running_loop()
        batch = [await self._queue.get()]  # block until at least one job
        deadline = loop.time() + max_wait

        while len(batch) < max_size:
            # Grab anything already waiting without blocking.
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break
        return batch

    def complete(self, job_id: str, result: InferenceResult) -> None:
        future = self._results.get(job_id)
        if future is not None and not future.done():
            future.set_result(result)

    def fail(self, job_id: str, exc: Exception) -> None:
        future = self._results.get(job_id)
        if future is not None and not future.done():
            future.set_exception(exc)
