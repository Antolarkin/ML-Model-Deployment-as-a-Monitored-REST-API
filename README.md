# ML Model Deployment as a Monitored REST API

A complete FastAPI service for serving a scikit-learn Iris classifier. The project covers model training and serialization, validated and versioned API contracts, structured logging, environment-based configuration, automated tests, Docker Compose deployment, API-key security, Prometheus metrics, integration testing, and load testing.

## Project Overview

The service accepts four Iris flower measurements and returns the predicted species, class probabilities, and a request ID. Version 1 and version 2 are served side by side so a client can migrate without breaking existing integrations.

The model is a `RandomForestClassifier` inside a scikit-learn `Pipeline` with `StandardScaler`. It is trained on the built-in Iris dataset from scikit-learn.

## Features

- FastAPI application with automatic OpenAPI documentation
- Pydantic input and response validation
- Versioned `/api/v1` and `/api/v2` prediction contracts
- Single and batch inference
- Model metadata endpoint
- Startup model loading with fail-fast behavior
- API-key authentication and sliding-window rate limiting
- Explicit CORS allowlist
- Rotating structured application logs
- Prometheus request, latency, and prediction-class metrics
- Docker image and Docker Compose deployment
- Model artifacts generated during the Docker build
- Host-side model swapping through a bind mount
- Unit, security, metrics, and live-container integration tests
- Async HTTP load-testing utility
- GitHub Actions test and Docker-build workflow

## Architecture

```text
Client
  |
  | HTTP request
  v
Docker / Docker Compose
  |
  v
Uvicorn
  |
  v
FastAPI
  |
  +--> request logging middleware
  |
  +--> API-key dependency
  |
  +--> rate-limit dependency
  |
  +--> Pydantic validation
  |
  v
Feature conversion
  |
  v
In-memory scikit-learn model
  |
  +--> prediction response
  |
  +--> structured log event with request_id
  |
  +--> Prometheus metric update
  |
  v
JSON response to client
```

The model, target names, and metadata are loaded once during application lifespan startup. The Docker build trains a default model and stores a template inside the image. At container startup, `scripts/start_api.py` copies missing template files into the configured model directory. This makes a clean Compose checkout reproducible while preserving the `ml/saved_model/` bind mount for model swaps.

## API Contract

All prediction endpoints require:

```text
X-API-Key: dev-secret-key
```

The default key is for local development only. Set `API_KEY` to a private value before exposing the service.

### `GET /`

Returns a simple liveness message.

```bash
curl http://127.0.0.1:8000/
```

### `GET /docs`

Opens the interactive FastAPI documentation. The OpenAPI schema includes the `X-API-Key` security scheme.

```bash
curl http://127.0.0.1:8000/docs
```

### `GET /api/v1/health`

Returns the service and model-loading state.

```bash
curl -H "X-API-Key: dev-secret-key" \
  http://127.0.0.1:8000/api/v1/health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /api/v1/predict`

Request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Response:

```json
{
  "request_id": "5f7c6c0b-4d9f-4f8d-9c9a-3a6f0d9f2a11",
  "prediction": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

### `POST /api/v1/predict-batch`

Request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"items":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},{"sepal_length":6.2,"sepal_width":3.4,"petal_length":5.4,"petal_width":2.3}]}'
```

Response:

```json
{
  "batch_size": 2,
  "results": [
    {
      "request_id": "generated-request-id",
      "prediction": "setosa",
      "confidence": 1.0,
      "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
      }
    },
    {
      "request_id": "generated-request-id",
      "prediction": "virginica",
      "confidence": 0.98,
      "probabilities": {
        "setosa": 0.0,
        "versicolor": 0.02,
        "virginica": 0.98
      }
    }
  ]
}
```

The batch schema accepts 1 through `MAX_BATCH_SIZE` items. Invalid, empty, and oversized batches return `422`.

### `GET /api/v1/model-info`

Returns the metadata written by `ml/train.py`.

```bash
curl -H "X-API-Key: dev-secret-key" \
  http://127.0.0.1:8000/api/v1/model-info
```

### `POST /api/v2/predict`

Version 2 intentionally changes the response contract. It uses `confidence_score` and adds `model_version`.

```bash
curl -X POST http://127.0.0.1:8000/api/v2/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Response:

```json
{
  "request_id": "generated-request-id",
  "prediction": "setosa",
  "confidence_score": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "2.0.0"
}
```

### `GET /metrics`

Returns Prometheus text exposition data. This endpoint is intentionally available to a monitoring scraper.

```bash
curl http://127.0.0.1:8000/metrics
```

Relevant metrics include:

- `http_requests_total`
- `http_request_duration_seconds`
- `ml_predictions_total{predicted_class="..."}`

### HTTP Status Codes

| Status | Meaning |
|---:|---|
| 200 | Request succeeded |
| 401 | Missing or invalid API key |
| 404 | Unknown endpoint |
| 422 | Invalid JSON, missing fields, invalid feature values, or invalid batch size |
| 429 | Rate limit exceeded |
| 500 | Unexpected inference or application error |
| 503 | Model or model metadata is not loaded |

## Project Structure

```text
.
├── .dockerignore
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml
├── app/
│   ├── config.py
│   ├── dependencies.py
│   ├── features.py
│   ├── logging_config.py
│   ├── main.py
│   ├── metrics.py
│   ├── models/
│   │   └── schemas.py
│   └── routers/
│       ├── v1.py
│       └── v2.py
├── docker-compose.yml
├── Dockerfile
├── ml/
│   ├── saved_model/
│   │   ├── model_metadata.json
│   │   ├── model.joblib
│   │   └── target_names.joblib
│   ├── train.py
│   └── verify_model.py
├── pytest.ini
├── render.yaml
├── requirements.txt
├── scripts/
│   ├── load_test.py
│   └── start_api.py
├── TESTING.md
└── tests/
    ├── conftest.py
    ├── test_container_integration.py
    ├── test_edge_cases.py
    ├── test_health.py
    ├── test_metrics.py
    ├── test_model_info.py
    ├── test_predict.py
    ├── test_predict_batch.py
    ├── test_security.py
    └── test_v2_predict.py
```

The two `.joblib` files are generated artifacts and are ignored by Git. They are created by `ml/train.py` and by the Docker build.

## Local Setup

### Prerequisites

- Python 3.12 or newer
- Docker Desktop or Docker Engine with the Compose plugin

### Docker Compose

```bash
docker compose up --build
```

The service is available at `http://127.0.0.1:8000`.

Stop it with:

```bash
docker compose down
```

The first image build trains the default model. The bind mount remains available for replacing model files on the host. After replacing `model.joblib`, restart the service:

```bash
docker compose restart api
```

### Local Python Environment

```bash
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

Then install and run:

```powershell
python -m pip install -r requirements.txt
python ml/train.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On Linux or macOS, activate with `source venv/bin/activate`.

### Configuration

Copy `.env.example` to `.env` when a local value needs to differ from the defaults:

```bash
cp .env.example .env
```

Available settings:

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model artifact path |
| `TARGET_NAMES_PATH` | Target-name artifact path |
| `MODEL_METADATA_PATH` | Model metadata path |
| `API_TITLE` | OpenAPI title |
| `LOG_LEVEL` | Application log level |
| `MAX_BATCH_SIZE` | Maximum batch request size |
| `API_KEY` | API authentication key |
| `CORS_ORIGINS` | Allowed browser origins |
| `RATE_LIMIT_REQUESTS` | Requests allowed per window |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate-limit window duration |

`.env` is ignored by Git. Never commit a production API key.

## Testing

Run the unit, security, metrics, and edge-case suite:

```bash
pytest -q
```

Run the live-container integration checklist after starting Compose:

```bash
$env:RUN_CONTAINER_TESTS = "1"
pytest -q tests/test_container_integration.py
```

The integration fixture restarts a local Compose API service before the module so rate-limit state from an earlier run cannot make the test flaky. Set `RESET_CONTAINER=0` when testing an external service that should not be restarted.

Run the concurrent load test:

```powershell
python scripts/load_test.py --requests 100 --concurrency 20 --base-url http://127.0.0.1:8000 --timeout 10
```

The load-test output reports throughput, status codes, error rate, minimum, mean, p95, and maximum latency. Detailed results and the Task 19 bug fix are recorded in `TESTING.md`.

## Independent Extension

I independently added a GitHub Actions workflow at `.github/workflows/tests.yml`.

On every push and pull request, the workflow:

1. Checks out the repository.
2. Installs Python 3.12 and the direct dependencies.
3. Runs the complete pytest suite.
4. Builds the Docker image.

This extension turns the local verification process into a repeatable CI check and catches dependency, test, and container-build regressions before deployment.

## Deployment

### Reproducible Local Deployment

`docker compose up --build` is the canonical deployment command. The image contains a trained default model, and the startup initializer populates an empty host bind mount when needed.

### Render Blueprint

`render.yaml` defines a Docker-based Render web service with:

- Free compute plan
- Oregon region
- `/metrics` health check
- Generated `API_KEY` secret
- Automatic deployment after passing checks

To deploy manually:

1. Create a Render account and connect the GitHub repository.
2. Import the repository Blueprint from `render.yaml`.
3. Allow Render to generate the `API_KEY` environment variable.
4. Deploy the service.
5. Retrieve the generated API key from the Render dashboard and use it in prediction requests.

A public URL was not created automatically in this environment because no Render, Railway, Fly.io, or GitHub deployment credentials were available. The Compose command and Render Blueprint provide the reproducible deployment path without embedding credentials in the repository.

## What I Learned

This project showed me that serving a model is a system-design problem, not only a machine-learning problem. A useful API needs stable contracts, validation before inference, consistent request tracing, and clear failure behavior. Docker makes the runtime reproducible, but model artifacts and startup behavior still need explicit handling. Security controls such as API keys, CORS, and rate limits protect the service, while Prometheus metrics and load testing reveal behavior that unit tests cannot show. Versioning lets the API evolve without silently breaking existing clients, and automated tests make each change reviewable instead of relying on manual browser checks.

## References

- [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)
- [MLOps Project](https://github.com/MubahsirHassan/MLOps-Project)
- [Deploy BERT for Sentiment Analysis with FastAPI](https://github.com/curiousily/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI)
- [Render Blueprint specification](https://render.com/docs/blueprint-spec)

## License

Educational and portfolio project.
