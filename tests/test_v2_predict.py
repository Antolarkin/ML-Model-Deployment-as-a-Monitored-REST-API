def test_v2_predict_returns_200(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v2/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence_score" in data
    assert "probabilities" in data
    assert "model_version" in data
    assert data["prediction"] == "setosa"
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert data["model_version"] == "2.0.0"


def test_v2_predict_with_missing_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }
    response = client.post("/api/v2/predict", json=payload)
    assert response.status_code == 422


def test_v2_predict_returns_different_shape_than_v1(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    v1_response = client.post("/api/v1/predict", json=payload)
    v2_response = client.post("/api/v2/predict", json=payload)

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    assert "confidence" in v1_data
    assert "confidence" not in v2_data
    assert "confidence_score" in v2_data
    assert "model_version" in v2_data
    assert "model_version" not in v1_data

    assert v1_data["prediction"] == v2_data["prediction"]
    assert v1_data["probabilities"] == v2_data["probabilities"]
