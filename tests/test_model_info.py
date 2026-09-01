def test_model_info_returns_expected_keys(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    expected_keys = {
        "model_type",
        "model_version",
        "training_date",
        "expected_features",
        "target_names",
        "parameters",
        "test_accuracy",
    }
    assert expected_keys.issubset(data.keys())
