import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = settings.API_KEY
RUN_CONTAINER_TESTS = os.getenv("RUN_CONTAINER_TESTS") == "1"
RESET_CONTAINER = os.getenv("RESET_CONTAINER", "1") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_CONTAINER_TESTS,
    reason="Set RUN_CONTAINER_TESTS=1 while the Docker Compose stack is running",
)

PREDICTION_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


@pytest.fixture(scope="module")
def container_client():
    if RESET_CONTAINER and API_BASE_URL.startswith(("http://127.0.0.1", "http://localhost")):
        subprocess.run(
            ["docker", "compose", "restart", "api"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        client.headers.update({"X-API-Key": API_KEY})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                health_response = client.get("/api/v1/health")
            except httpx.HTTPError:
                health_response = None

            if health_response is not None and health_response.status_code == 200:
                break

            time.sleep(0.25)
        else:
            raise RuntimeError("Container did not become ready within 30 seconds")

        yield client


def test_container_health_endpoint(container_client):
    response = container_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_container_v1_prediction_endpoint(container_client):
    response = container_client.post("/api/v1/predict", json=PREDICTION_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"setosa", "versicolor", "virginica"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"]) == {"setosa", "versicolor", "virginica"}


def test_container_v1_batch_endpoint(container_client):
    response = container_client.post(
        "/api/v1/predict-batch",
        json={"items": [PREDICTION_PAYLOAD, PREDICTION_PAYLOAD]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2
    assert all(result["prediction"] in {"setosa", "versicolor", "virginica"} for result in data["results"])


def test_container_metrics_endpoint(container_client):
    prediction_response = container_client.post("/api/v1/predict", json=PREDICTION_PAYLOAD)
    assert prediction_response.status_code == 200

    response = container_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text
    assert "ml_predictions_total" in response.text
    assert 'predicted_class="setosa"' in response.text
