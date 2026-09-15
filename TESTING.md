# Testing Report

## Scope

Task 19 validates the API service deployed through Docker Compose over real HTTP connections. It covers the running container, configuration, model loading, API-key authentication, logging, Prometheus metrics, and concurrent prediction traffic.

## Environment

- Date: 2026-09-15
- Runtime: Docker Compose with the project Dockerfile
- API address: `http://127.0.0.1:8000`
- Model: Iris Random Forest loaded from the bind-mounted `ml/saved_model/` directory
- Stack command:

```powershell
docker compose up --build
```

## Integration Checks

The integration suite uses `httpx.Client` against the running container. It is opt-in so the normal unit-test command does not fail when Compose is not running. For a local Compose target, the module fixture restarts the API service before the checks so rate-limit state from an earlier load run cannot make the suite flaky:

```powershell
$env:RUN_CONTAINER_TESTS = "1"
pytest -q tests/test_container_integration.py
```

Set `RESET_CONTAINER=0` when testing an external service that should not be restarted.

Result: **4 passed on repeated runs**

The checks verified:

- `GET /api/v1/health` returned `200` with `model_loaded: true`.
- `POST /api/v1/predict` returned a valid Iris prediction, confidence, and all three class probabilities.
- `POST /api/v1/predict-batch` returned `batch_size: 2` and two valid results.
- `GET /metrics` returned Prometheus text containing HTTP request/latency metrics and the custom `ml_predictions_total` metric.

## Load Test

Command:

```powershell
python scripts/load_test.py --requests 100 --concurrency 20 --base-url http://127.0.0.1:8000 --timeout 10
```

Result:

| Measurement | Result |
|---|---:|
| Total requests | 100 |
| Concurrency | 20 |
| Elapsed time | 3.311 seconds |
| Throughput | 30.2 requests/second |
| Successful requests | 100 |
| Failed requests | 0 |
| Error rate | 0.00% |
| Minimum latency | 137.97 ms |
| Mean latency | 582.52 ms |
| p95 latency | 915.74 ms |
| Maximum latency | 992.50 ms |
| Post-load health status | 200 |
| Post-load metrics status | 200 |

The container remained running throughout the test. Container logs showed successful prediction requests with `200` responses and no application exceptions. The metrics endpoint reported:

```text
ml_predictions_total{predicted_class="setosa"} 100.0
http_requests_total{handler="/api/v1/predict",method="POST",status="2xx"} 100.0
http_request_duration_seconds_count{handler="/api/v1/predict",method="POST"} 100.0
```

## Bug Found and Fixed

### Finding

The health endpoint used the same in-memory rate-limit budget as prediction endpoints. With `RATE_LIMIT_REQUESTS=1`, two prediction requests exhausted the budget and the following health request returned `429`. This makes a service appear unavailable to health checks while prediction traffic is saturated.

The regression was reproduced before the fix:

```text
POST /api/v1/predict -> 200
POST /api/v1/predict -> 429
GET  /api/v1/health -> 429
```

### Fix

`enforce_rate_limit` now returns before recording a request when the path is a health endpoint. API-key authentication remains enforced by the router-level `verify_api_key` dependency. Prediction, batch, model-info, and other protected traffic still consumes the rate-limit budget.

### Regression Coverage

`tests/test_security.py::test_health_endpoint_remains_available_after_rate_limit` reproduces the saturated-budget scenario and verifies that health remains available with `200` while prediction returns `429`.

Result after the fix: **1 passed**

## Final Verification

```powershell
pytest -q
python -m compileall -q app scripts tests
docker compose ps
```

Result:

- Unit and regression suite: **23 passed, 4 integration tests skipped by default**
- Container integration suite: **4 passed**
- Python compilation: passed
- Compose service: running and reachable on port 8000
- Load test: 100/100 successful requests

## Reference Practices Used

- Made-With-ML: separate repeatable test tooling and document test results.
- testdrivenio/fastapi-tdd-docker: keep Docker-based verification distinct from in-process tests and use repeatable commands.
