"""Integration tests: exercise the full request pipeline (API -> queue -> worker
-> prediction) with the real lifespan running, plus invalid-input edge cases.

Unit-level validation tests live in test_routes.py; these focus on end-to-end
behavior and malformed requests.
"""

from base64 import b64encode

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_IMAGE = b64encode(b"fake image bytes").decode()


@pytest.fixture
def running_app():
    # Context-manager form runs lifespan -> starts the queue + worker.
    with TestClient(app) as c:
        yield c


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
