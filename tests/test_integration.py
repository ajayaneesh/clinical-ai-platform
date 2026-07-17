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
    from app.services.image_processing import ImageProcessingService

    original = dependencies._build_current_model
    images = ImageProcessingService(max_bytes=10 * 1024 * 1024)
    dependencies._build_current_model = lambda: TorchInferenceModel(images)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        dependencies._build_current_model = original


@pytest.fixture
def embedding_app():
    # Force embeddings ON with a FAKE embedding service (no BiomedCLIP download).
    from app.api import dependencies
    from app.core.config import settings
    from app.services.embedding import EmbeddingService

    class FakeEmbeddingModel:
        @property
        def name(self) -> str:
            return "fake"

        def embed(self, image: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        def embed_batch(self, images: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in images]

    original_flag = settings.enable_embeddings
    original_builder = dependencies.build_embedding_service
    settings.enable_embeddings = True
    dependencies.build_embedding_service = lambda: EmbeddingService(
        FakeEmbeddingModel()
    )
    try:
        with TestClient(app) as c:
            yield c
    finally:
        settings.enable_embeddings = original_flag
        dependencies.build_embedding_service = original_builder


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


def test_batching_processes_many_jobs_in_one_pass():
    # A single batching worker must coalesce concurrent jobs into ONE forward
    # pass: N jobs finish in ~one batch's time, NOT N * per-job time. Proves
    # batch inference (see docs/architecture/performance-baseline.md).
    import asyncio
    import time

    from app.core.queue import Job, LocalQueue
    from app.models.inference import InferenceResult
    from app.services.inference import InferenceService
    from app.workers.inference_worker import start_workers

    DELAY = 0.1  # time for ONE batch forward pass
    JOBS = 8

    class BatchModel:
        # Batch call costs one DELAY regardless of batch size (models the GPU
        # amortization). Per-item fallback would cost DELAY each.
        def predict(self, image: str) -> InferenceResult:
            return self.predict_batch([image])[0]

        def predict_batch(self, images: list[str]) -> list[InferenceResult]:
            time.sleep(DELAY)
            return [{"prediction": "normal", "confidence": 0.95} for _ in images]

    async def scenario() -> float:
        queue = LocalQueue(timeout=30)
        tasks = start_workers(
            queue,
            InferenceService(BatchModel()),
            max_batch_size=JOBS,
            max_batch_wait=0.05,
            count=1,
        )
        try:
            loop = asyncio.get_running_loop()
            start = loop.time()
            await asyncio.gather(*(queue.submit(Job(image="x")) for _ in range(JOBS)))
            return loop.time() - start
        finally:
            for task in tasks:
                task.cancel()

    elapsed = asyncio.run(scenario())

    # Batched: ~1 * DELAY. Un-batched (serial) would be ~JOBS * DELAY. Assert we
    # are far below the serial cost — the jobs coalesced into ~one batch.
    assert elapsed < JOBS * DELAY * 0.5, (
        f"jobs did not batch: {elapsed:.3f}s ~ serial {JOBS * DELAY:.3f}s"
    )


def test_embed_returns_vector(embedding_app):
    response = embedding_app.post("/embed", json={"image": VALID_IMAGE})
    assert response.status_code == 200
    body = response.json()
    assert body["embedding"] == [0.1, 0.2, 0.3]
    assert body["dimension"] == 3
    assert body["model"] == "fake"
    assert "embedding_id" in body
    assert isinstance(body["inference_ms"], (int, float))


def test_embed_stores_vector_in_memory(embedding_app):
    # Each /embed call stores the vector; two calls -> two stored, distinct ids.
    r1 = embedding_app.post("/embed", json={"image": VALID_IMAGE})
    r2 = embedding_app.post("/embed", json={"image": VALID_IMAGE})
    id1 = r1.json()["embedding_id"]
    id2 = r2.json()["embedding_id"]
    assert id1 != id2

    store = embedding_app.app.state.embedding_store
    assert store.count() == 2
    stored = store.get(id1)
    assert stored.vector == [0.1, 0.2, 0.3]
    assert stored.model == "fake"


def test_embed_rejects_invalid_base64(embedding_app):
    response = embedding_app.post("/embed", json={"image": "not base64!!!"})
    assert response.status_code == 400


def test_embed_returns_503_when_disabled(running_app):
    # Embeddings not enabled (default) -> /embed is unavailable.
    response = running_app.post("/embed", json={"image": VALID_IMAGE})
    assert response.status_code == 503


def test_embed_upload_returns_same_as_embed(embedding_app):
    # /embed/upload takes a real file and returns the same shape as /embed.
    response = embedding_app.post(
        "/embed/upload",
        files={"file": ("xray.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["embedding"] == [0.1, 0.2, 0.3]
    assert body["dimension"] == 3
    assert body["model"] == "fake"
    assert "embedding_id" in body


def test_embed_upload_empty_file_returns_400(embedding_app):
    response = embedding_app.post(
        "/embed/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_embed_upload_returns_503_when_disabled(running_app):
    response = running_app.post(
        "/embed/upload",
        files={"file": ("xray.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503


def test_search_returns_top_hits_and_measurements(embedding_app):
    # Store a few embeddings first, then search.
    for _ in range(3):
        embedding_app.post("/embed", json={"image": VALID_IMAGE})

    response = embedding_app.post("/search", json={"image": VALID_IMAGE})
    assert response.status_code == 200
    body = response.json()
    assert body["searched"] == 3
    assert len(body["results"]) == 3  # fewer than top_k=5 stored -> returns all
    # The three measurements the mini search engine reports:
    assert isinstance(body["embedding_ms"], (int, float))
    assert isinstance(body["search_ms"], (int, float))
    assert body["store_memory_bytes"] == 3 * 3 * 8  # 3 vectors x dim 3 x 8 bytes
    # Each hit has an id and a cosine score.
    for hit in body["results"]:
        assert "embedding_id" in hit
        assert -1.0 <= hit["score"] <= 1.0


def test_search_empty_store_returns_no_results(embedding_app):
    response = embedding_app.post("/search", json={"image": VALID_IMAGE})
    assert response.status_code == 200
    body = response.json()
    assert body["searched"] == 0
    assert body["results"] == []


def test_search_upload_works(embedding_app):
    embedding_app.post("/embed", json={"image": VALID_IMAGE})
    response = embedding_app.post(
        "/search/upload",
        files={"file": ("q.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["searched"] == 1


def test_search_rejects_invalid_base64(embedding_app):
    response = embedding_app.post("/search", json={"image": "not base64!!!"})
    assert response.status_code == 400


def test_search_returns_503_when_disabled(running_app):
    response = running_app.post("/search", json={"image": VALID_IMAGE})
    assert response.status_code == 503


def test_embed_stores_and_returns_metadata(embedding_app):
    response = embedding_app.post(
        "/embed",
        json={
            "image": VALID_IMAGE,
            "filename": "patient_042_chest_xray.png",
            "diagnosis_label": "pneumonia",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "patient_042_chest_xray.png"
    assert body["diagnosis_label"] == "pneumonia"
    assert body["timestamp"]  # server-generated, non-empty

    store = embedding_app.app.state.embedding_store
    stored = store.get(body["embedding_id"])
    assert stored.filename == "patient_042_chest_xray.png"
    assert stored.diagnosis_label == "pneumonia"
    assert stored.timestamp == body["timestamp"]


def test_embed_upload_defaults_filename_from_file(embedding_app):
    response = embedding_app.post(
        "/embed/upload",
        files={"file": ("xray.png", _png_bytes(), "image/png")},
        params={"diagnosis_label": "normal"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "xray.png"
    assert body["diagnosis_label"] == "normal"


def test_search_filters_by_diagnosis_label(embedding_app):
    embedding_app.post(
        "/embed", json={"image": VALID_IMAGE, "diagnosis_label": "pneumonia"}
    )
    embedding_app.post(
        "/embed", json={"image": VALID_IMAGE, "diagnosis_label": "normal"}
    )

    response = embedding_app.post(
        "/search", json={"image": VALID_IMAGE, "diagnosis_label": "pneumonia"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["searched"] == 2  # store still holds both
    assert len(body["results"]) == 1
    assert body["results"][0]["diagnosis_label"] == "pneumonia"
