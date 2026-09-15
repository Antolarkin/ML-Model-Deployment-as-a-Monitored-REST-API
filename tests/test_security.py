from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import _request_log
from app.main import app


def test_missing_api_key_returns_401():
    raw_client = TestClient(app)
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = raw_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_invalid_api_key_returns_401(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_unexpected_extra_field_is_forbidden(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "extra_field": "should fail",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_health_endpoint_remains_available_after_rate_limit(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    _request_log.clear()
    original_limit = settings.RATE_LIMIT_REQUESTS
    settings.RATE_LIMIT_REQUESTS = 1

    try:
        client.post("/api/v1/predict", json=payload)
        limited_response = client.post("/api/v1/predict", json=payload)
        health_response = client.get("/api/v1/health")

        assert limited_response.status_code == 429
        assert health_response.status_code == 200
    finally:
        settings.RATE_LIMIT_REQUESTS = original_limit
        _request_log.clear()


def test_rate_limit_exceeded_returns_429(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    headers = {"X-API-Key": settings.API_KEY}
    original_limit = settings.RATE_LIMIT_REQUESTS
    settings.RATE_LIMIT_REQUESTS = 2
    try:
        client.post("/api/v1/predict", json=payload, headers=headers)
        client.post("/api/v1/predict", json=payload, headers=headers)
        response = client.post("/api/v1/predict", json=payload, headers=headers)
        assert response.status_code == 429
    finally:
        settings.RATE_LIMIT_REQUESTS = original_limit
