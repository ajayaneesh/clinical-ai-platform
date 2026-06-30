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


def test_request_id_header_present():
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
