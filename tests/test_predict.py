def test_predict_with_valid_input_returns_200(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["prediction"] == "setosa"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_with_missing_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_with_invalid_value_returns_422(client):
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
