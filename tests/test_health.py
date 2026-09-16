def test_health_returns_200(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_root_endpoint_returns_alive_message(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "ML API is alive"}


def test_predict_returns_503_when_model_is_not_loaded(client):
    client.app.state.model = None
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}
