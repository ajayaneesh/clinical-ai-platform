from base64 import b64encode
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (8, 8), (120, 30, 200)).save(buf, "PNG")
    return b64encode(buf.getvalue()).decode()


VALID_IMAGE = _png_b64()


def test_root_returns_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Clinical AI Platform"}


def test_health_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_dummy_prediction():
    # Context-manager form runs the lifespan, which starts the queue + worker.
    with TestClient(app) as c:
        response = c.post("/predict", json={"image": VALID_IMAGE})
    assert response.status_code == 200
    assert response.json() == {"prediction": "normal", "confidence": 0.95}


def test_predict_rejects_empty_image():
    # min_length=1 -> Pydantic validation fails -> 422 (automatic).
    response = client.post("/predict", json={"image": ""})
    assert response.status_code == 422


def test_predict_rejects_missing_image():
    # Required field absent -> 422 (automatic).
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_non_base64_image():
    # Passes Pydantic (non-empty string) but fails the base64 rule -> 400.
    # Validation happens before the queue, so no running worker is needed.
    response = client.post("/predict", json={"image": "not base64!!!"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Image could not be decoded as base64."}


def test_request_id_header_present():
    response = client.get("/health")
    assert "X-Request-ID" in response.headers


def test_predict_uses_injected_model():
    # Prove DI still works: override the model at the composition root, so the
    # worker (started in lifespan) uses the fake model. Result flows back
    # through the queue to the awaiting request.
    from app.api import dependencies
    from app.models.inference import (
        DummyInferenceModel,
        InferenceModel,
        InferenceResult,
    )

    class FakeModel:
        def predict(self, image: str) -> InferenceResult:
            return {"prediction": "critical", "confidence": 0.42}

    original: InferenceModel = DummyInferenceModel()
    dependencies.get_model = lambda: FakeModel()  # type: ignore[assignment]
    try:
        with TestClient(app) as c:
            response = c.post("/predict", json={"image": VALID_IMAGE})
        assert response.status_code == 200
        assert response.json() == {"prediction": "critical", "confidence": 0.42}
    finally:
        dependencies.get_model = lambda: original  # type: ignore[assignment]


def test_worker_offloads_prediction_and_does_not_block_loop():
    # The worker calls a SYNCHRONOUS predict() that may be slow. It must run that
    # via run_in_executor so it does NOT freeze the event loop. We prove this at
    # the loop level: while the worker processes a slow prediction, a concurrent
    # 50ms task must still finish in ~50ms — not wait out the full slow sleep.
    #
    # (An HTTP-level test can't show this reliably: httpx's in-memory ASGI
    # transport doesn't interleave requests the way a real socket server does,
    # so it can't reproduce genuine event-loop contention.)
    import asyncio
    import time

    from app.core.queue import Job, LocalQueue
    from app.models.inference import InferenceResult
    from app.services.inference import InferenceService
    from app.workers.inference_worker import start_workers

    SLOW = 1.0

    class SlowModel:
        def predict(self, image: str) -> InferenceResult:
            time.sleep(SLOW)  # blocking, synchronous
            return {"prediction": "normal", "confidence": 0.95}

    async def scenario() -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        queue = LocalQueue()
        worker_tasks = start_workers(queue, InferenceService(SlowModel()), count=1)
        try:
            start = loop.time()
            predict_task = asyncio.create_task(queue.submit(Job(image="x")))

            async def quick_tick() -> float:
                await asyncio.sleep(0.05)
                return loop.time() - start

            tick_time = await quick_tick()  # concurrent with the slow prediction
            await predict_task
            predict_time = loop.time() - start
            return tick_time, predict_time
        finally:
            for task in worker_tasks:
                task.cancel()

    tick_time, predict_time = asyncio.run(scenario())

    # Loop stayed free: the 50ms task finished promptly, not after SLOW seconds.
    assert tick_time < SLOW / 2, f"event loop was blocked ({tick_time:.2f}s)"
    # The prediction itself still took the full slow time.
    assert predict_time >= SLOW, (
        f"prediction should take >= {SLOW}s, got {predict_time:.2f}s"
    )


def test_submit_times_out_when_no_worker_completes_it():
    # No worker consumes the queue, so the job is never completed. submit() must
    # raise QueueTimeout after the configured timeout instead of hanging forever.
    import asyncio

    import pytest

    from app.core.queue import Job, LocalQueue, QueueTimeout

    async def scenario() -> None:
        queue = LocalQueue(timeout=0.1)  # short timeout for a fast test
        with pytest.raises(QueueTimeout):
            await queue.submit(Job(image="x"))

    asyncio.run(scenario())


def test_predict_returns_504_on_timeout():
    # End-to-end: with a worker that never finishes, /predict surfaces the
    # QueueTimeout as HTTP 504.
    import asyncio

    import httpx

    from app.core.queue import Job, LocalQueue

    async def scenario() -> httpx.Response:
        # A queue with a short timeout and NO worker started for it.
        app.state.queue = LocalQueue(timeout=0.1)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.post("/predict", json={"image": VALID_IMAGE})

    # Don't run lifespan (which would start a worker); set the queue manually.
    assert Job and LocalQueue  # referenced for clarity
    response = asyncio.run(scenario())
    assert response.status_code == 504
    assert response.json() == {"detail": "Prediction timed out."}


def test_queue_timeout_is_config_driven(monkeypatch):
    # The timeout comes from the CLINICAL_AI_QUEUE_TIMEOUT_SECONDS env var.
    from app.core.config import Settings

    monkeypatch.setenv("CLINICAL_AI_QUEUE_TIMEOUT_SECONDS", "5")
    assert Settings().queue_timeout_seconds == 5.0

    monkeypatch.delenv("CLINICAL_AI_QUEUE_TIMEOUT_SECONDS")
    assert Settings().queue_timeout_seconds == 30.0  # default when unset


def test_ready_when_queue_initialized():
    # Lifespan sets app.state.queue -> /ready returns 200.
    with TestClient(app) as c:
        response = c.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_without_queue():
    # No lifespan run (plain client) -> no queue -> not ready.
    app.state.queue = None
    response = client.get("/ready")
    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_text():
    # Exercise a request so counters are non-empty, then scrape /metrics.
    with TestClient(app) as c:
        c.post("/predict", json={"image": VALID_IMAGE})
        response = c.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "http_request_latency_seconds" in body
    assert "inference_latency_seconds" in body
    assert "http_requests_total" in body
