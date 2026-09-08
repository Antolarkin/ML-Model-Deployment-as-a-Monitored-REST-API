def test_feature_boundary_zero_rejected(client):
    payload = {
        "sepal_length": 0.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_feature_boundary_ten_accepted(client):
    payload = {
        "sepal_length": 10.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200


def test_negative_feature_rejected(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_extreme_large_value_rejected(client):
    payload = {
        "sepal_length": 999.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_empty_string_input_rejected(client):
    payload = {
        "sepal_length": "",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
