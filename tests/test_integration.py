"""Integration tests: exercise the full request pipeline (API -> queue -> worker
-> prediction) with the real lifespan running, plus invalid-input edge cases.

Unit-level validation tests live in test_routes.py; these focus on end-to-end
behavior and malformed requests.
"""

from base64 import b64encode
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (8, 8), (120, 30, 200)).save(buf, "PNG")
    return b64encode(buf.getvalue()).decode()


VALID_IMAGE = _png_b64()


@pytest.fixture
def running_app():
    # Context-manager form runs lifespan -> starts the queue + worker.
    # conftest's autouse fixture forces the fast placeholder model.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def decoding_app():
    # For tests that verify IMAGE-DECODE error handling: force TorchModel, which
    # actually decodes the image (the dummy accepts anything). Still offline/fast
    # (returns a fixed label). Overrides conftest's dummy for these tests only.
    from app.api import dependencies
    from app.models.torch_model import TorchInferenceModel

    original = dependencies.get_model
    dependencies.get_model = lambda: TorchInferenceModel()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        dependencies.get_model = original


def test_full_pipeline_returns_prediction(running_app):
    # API -> queue -> worker -> dummy model -> response, all wired up live.
    response = running_app.post("/predict", json={"image": VALID_IMAGE})
    assert response.status_code == 200
    body = response.json()
    assert body == {"prediction": "normal", "confidence": 0.95}


def test_pipeline_handles_many_sequential_requests(running_app):
    # The single worker must drain a burst of jobs without losing any.
    for _ in range(25):
        response = running_app.post("/predict", json={"image": VALID_IMAGE})
        assert response.status_code == 200


def test_ready_and_metrics_reflect_traffic(running_app):
    running_app.post("/predict", json={"image": VALID_IMAGE})
    assert running_app.get("/ready").json() == {"status": "ready"}
    metrics = running_app.get("/metrics").text
    assert "inference_latency_seconds_count" in metrics


# --- invalid / malformed requests -----------------------------------------


@pytest.mark.parametrize(
    "payload, expected_status",
    [
        ({}, 422),  # missing required field
        ({"image": ""}, 422),  # empty string violates min_length
        ({"image": None}, 422),  # null for a required str
        ({"image": 123}, 422),  # wrong type
        ({"image": ["a", "b"]}, 422),  # wrong type (list)
        ({"wrong_field": "x"}, 422),  # unknown field, image missing
        ({"image": "not base64!!!"}, 400),  # valid shape, bad content
        ({"image": "a"}, 400),  # non-empty but not valid base64
    ],
)
def test_predict_rejects_invalid_payloads(running_app, payload, expected_status):
    response = running_app.post("/predict", json=payload)
    assert response.status_code == expected_status


def test_predict_rejects_non_json_body(running_app):
    response = running_app.post(
        "/predict",
        content="this is not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_rejects_wrong_method(running_app):
    # /predict is POST-only; GET must be rejected.
    response = running_app.get("/predict")
    assert response.status_code == 405


def test_unknown_route_returns_404(running_app):
    response = running_app.get("/does-not-exist")
    assert response.status_code == 404


def test_valid_base64_but_not_an_image_returns_400(decoding_app):
    # Valid base64 that decodes to a truncated/invalid image (the reported bug):
    # must return 400, NOT hang until a 504 timeout.
    response = decoding_app.post(
        "/predict", json={"image": "iVBORw0KGgoAAAANSUhEUgAAAAUA"}
    )
    assert response.status_code == 400


def test_bad_image_does_not_kill_worker(decoding_app):
    # A malformed image must fail only THAT request; the worker keeps running and
    # the next valid request still succeeds.
    bad = decoding_app.post("/predict", json={"image": "iVBORw0KGgoAAAANSUhEUgAAAAUA"})
    assert bad.status_code == 400

    good = decoding_app.post("/predict", json={"image": VALID_IMAGE})
    assert good.status_code == 200
    assert good.json() == {"prediction": "normal", "confidence": 0.95}


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), (120, 30, 200)).save(buf, "PNG")
    return buf.getvalue()


def test_upload_returns_same_response_as_predict(running_app):
    # /predict/upload takes a real file and must return the same shape/values
    # as /predict for an equivalent image.
    response = running_app.post(
        "/predict/upload",
        files={"file": ("xray.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "normal", "confidence": 0.95}


def test_upload_empty_file_returns_400(running_app):
    response = running_app.post(
        "/predict/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_upload_non_image_returns_400(decoding_app):
    # A non-image file: pipeline must fail cleanly with 400, not hang or 500.
    response = decoding_app.post(
        "/predict/upload",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_missing_file_returns_422(running_app):
    # No file part at all -> FastAPI validation -> 422.
    response = running_app.post("/predict/upload")
    assert response.status_code == 422


def test_multiple_workers_increase_throughput():
    # Two workers processing a fixed-latency model must finish a batch of jobs
    # in roughly half the time a single worker would. Proves the worker-count
    # fix scales throughput (see docs/architecture/performance-baseline.md).
    import asyncio
    import time

    from app.core.queue import Job, LocalQueue
    from app.models.inference import InferenceResult
    from app.services.inference import InferenceService
    from app.workers.inference_worker import start_workers

    DELAY = 0.02
    JOBS = 8

    class FixedModel:
        def predict(self, image: str) -> InferenceResult:
            time.sleep(DELAY)
            return {"prediction": "normal", "confidence": 0.95}

    async def run_with(count: int) -> float:
        queue = LocalQueue(timeout=30)
        tasks = start_workers(queue, InferenceService(FixedModel()), count=count)
        try:
            loop = asyncio.get_running_loop()
            start = loop.time()
            await asyncio.gather(*(queue.submit(Job(image="x")) for _ in range(JOBS)))
            return loop.time() - start
        finally:
            for task in tasks:
                task.cancel()

    one = asyncio.run(run_with(1))
    two = asyncio.run(run_with(2))

    # 1 worker is serial (~JOBS * DELAY); 2 workers roughly halve it. Use a loose
    # bound to stay robust against scheduling jitter.
    assert two < one * 0.75, f"2 workers ({two:.3f}s) not faster than 1 ({one:.3f}s)"
