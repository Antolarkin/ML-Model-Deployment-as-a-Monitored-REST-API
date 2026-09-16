import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import _request_log
from app.main import app
from app.metrics import PREDICTION_SUCCESS_TOTAL


class MockModel:
    def predict(self, X):
        return np.array([0] * len(X))

    def predict_proba(self, X):
        return np.array([[1.0, 0.0, 0.0]] * len(X))


@pytest.fixture(autouse=True)
def reset_runtime_state():
    original_rate_limit = settings.RATE_LIMIT_REQUESTS
    _request_log.clear()
    PREDICTION_SUCCESS_TOTAL.clear()
    yield
    settings.RATE_LIMIT_REQUESTS = original_rate_limit
    _request_log.clear()
    PREDICTION_SUCCESS_TOTAL.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        test_client.app.state.model = MockModel()
        test_client.app.state.target_names = ["setosa", "versicolor", "virginica"]
        test_client.app.state.model_info = {
            "model_type": "RandomForestClassifier",
            "model_version": "1.0.0",
            "training_date": "2026-08-31T10:18:57.799610Z",
            "expected_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target_names": ["setosa", "versicolor", "virginica"],
            "parameters": {"n_estimators": 100, "random_state": 42},
            "test_accuracy": 1.0,
        }
        test_client.headers["X-API-Key"] = settings.API_KEY
        yield test_client
