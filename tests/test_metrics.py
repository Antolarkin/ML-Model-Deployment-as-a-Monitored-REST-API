from app.metrics import PREDICTION_SUCCESS_TOTAL

VALID_PREDICTION = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_metrics_endpoint_exposes_default_and_custom_metrics(client):
    PREDICTION_SUCCESS_TOTAL.clear()

    prediction_response = client.post("/api/v1/predict", json=VALID_PREDICTION)
    assert prediction_response.status_code == 200

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in metrics_response.text
    assert "http_request_duration_seconds" in metrics_response.text
    assert "ml_predictions_total" in metrics_response.text
    assert 'predicted_class="setosa"' in metrics_response.text


def test_custom_metric_counts_predictions_by_class(client):
    PREDICTION_SUCCESS_TOTAL.clear()

    assert client.post("/api/v1/predict", json=VALID_PREDICTION).status_code == 200
    assert client.post("/api/v2/predict", json=VALID_PREDICTION).status_code == 200
    batch_response = client.post(
        "/api/v1/predict-batch",
        json={"items": [VALID_PREDICTION, VALID_PREDICTION]},
    )
    assert batch_response.status_code == 200

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert 'ml_predictions_total{predicted_class="setosa"} 4.0' in metrics_response.text
