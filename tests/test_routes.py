from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Clinical AI Platform"}


def test_health_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_infer_returns_dummy_prediction():
    response = client.post("/infer", json={"text": "Patient labs are stable."})
    assert response.status_code == 200
    assert response.json() == {"prediction": "normal", "confidence": 0.95}


def test_request_id_header_present():
    response = client.get("/health")
    assert "X-Request-ID" in response.headers


def test_infer_uses_injected_model():
    # Prove dependency injection works: override the model with a fake one
    # and confirm the endpoint returns the fake's result, not the dummy's.
    from app.api.dependencies import get_inference_service
    from app.models.inference import InferenceResult
    from app.services.inference import InferenceService

    class FakeModel:
        def predict(self, text: str) -> InferenceResult:
            return {"prediction": "critical", "confidence": 0.42}

    app.dependency_overrides[get_inference_service] = lambda: InferenceService(FakeModel())
    try:
        response = client.post("/infer", json={"text": "anything"})
        assert response.status_code == 200
        assert response.json() == {"prediction": "critical", "confidence": 0.42}
    finally:
        app.dependency_overrides.clear()
