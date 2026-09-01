def test_predict_batch_with_valid_batch_returns_200(client):
    payload = {
        "items": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
            {
                "sepal_length": 6.2,
                "sepal_width": 3.4,
                "petal_length": 5.4,
                "petal_width": 2.3,
            },
        ]
    }
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "batch_size" in data
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2


def test_predict_batch_with_empty_batch_returns_422(client):
    payload = {"items": []}
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 422


def test_predict_batch_with_oversized_batch_returns_422(client):
    payload = {
        "items": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ]
        * 101
    }
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 422
