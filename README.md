# 🚀 ML Model Deployment as a Monitored REST API

> Take a trained machine-learning model and turn it into a **reliable, production-style REST API** — complete with validation, versioning, logging, testing, Docker deployment, security, and monitoring.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset & ML Problem](#-dataset--ml-problem)
- [API Contract](#-api-contract)
- [Architecture Flow](#-architecture-flow)
- [Tech Stack](#-tech-stack)
- [Planned Project Structure](#-planned-project-structure)
- [20-Task Roadmap](#-20-task-roadmap)
- [Checkpoints](#-checkpoints)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🎯 Project Overview

This is **not** just an ML model project — it is an **ML deployment and API engineering** project.

The ML model is only one component. The major learning areas are:

| Area | What You Learn |
|------|---------------|
| Python + FastAPI | Building REST APIs |
| Pydantic | Input validation |
| ML Integration | Loading & serving a trained model |
| Testing | Automated tests with pytest |
| Logging | Structured, traceable logs |
| Versioning | Backward-compatible API evolution |
| Security | API-key authentication + CORS |
| Docker | Containerized deployment |
| Monitoring | Prometheus metrics collection |
| Deployment | Cloud-hosted public API |

**The journey:**

```
model.predict() in a notebook  →  Production-style ML REST API
```

---

## 🧩 Problem Statement

A data scientist might train a model like this:

```python
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(prediction)
```

That works on **your computer**. But a website or mobile app **cannot access your Python notebook**.

Instead, they need to communicate over HTTP:

```
Website / App
      ↓
  HTTP Request (POST /api/v1/predict)
      ↓
  ML REST API (FastAPI + Uvicorn)
      ↓
  Trained ML Model
      ↓
  Prediction + Confidence
      ↓
  HTTP Response (JSON)
      ↓
Website / App
```

**This project builds that bridge.**

---

## 🌸 Dataset & ML Problem

### Dataset: Iris Flower Dataset (scikit-learn built-in)

| Property | Detail |
|----------|--------|
| **Source** | `sklearn.datasets.load_iris` |
| **Samples** | 150 |
| **Features** | 4 (all numeric) |
| **Classes** | 3 species |
| **Task** | Multi-class classification |
| **Download needed?** | ❌ No — built into scikit-learn |

### Why Iris?

- **Simple to explain:** "Give me 4 flower measurements, I tell you the species."
- **Zero setup:** Built into scikit-learn, no CSV download needed.
- **Supports probability:** We can return prediction + confidence.
- **Universally known:** Reviewers and interviewers instantly recognize it.
- **Focus stays on engineering:** The ML is intentionally simple — the API engineering is the point.

### The 4 Input Features

| Feature | Unit | Example |
|---------|------|---------|
| `sepal_length` | centimeters | 5.1 |
| `sepal_width` | centimeters | 3.5 |
| `petal_length` | centimeters | 1.4 |
| `petal_width` | centimeters | 0.2 |

### The 3 Output Classes

| Class ID | Species Name |
|----------|-------------|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

---

## 📝 API Contract

### In Plain English

> **The `/predict` endpoint accepts 4 numeric flower measurements (sepal length, sepal width, petal length, petal width) as a JSON object via HTTP POST. It validates that all 4 values are present and are positive numbers. If validation passes, it sends the data to the trained Iris classification model and returns a JSON response containing: the predicted species name, the confidence probability, the model version used, and a unique request ID for tracing. If validation fails, it returns a 422 error with a clear message explaining what went wrong.**

### Request

```
POST /api/v1/predict
Content-Type: application/json
```

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Successful Response (200 OK)

```json
{
  "prediction": "setosa",
  "prediction_id": 0,
  "confidence": 0.97,
  "model_version": "v1",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Validation Error Response (422 Unprocessable Entity)

```json
{
  "detail": [
    {
      "loc": ["body", "sepal_length"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

### Other Endpoints

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/health` | GET | Service health check | `{"status": "ok"}` |
| `/metrics` | GET | Prometheus metrics | Prometheus text format |
| `/api/v1/predict` | POST | Prediction (v1) | Prediction + confidence |
| `/api/v2/predict` | POST | Prediction (v2 — future) | Enhanced response |

### HTTP Status Codes Used

| Code | Meaning | When |
|------|---------|------|
| 200 | Success | Valid prediction returned |
| 400 | Bad Request | Malformed JSON |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Unknown endpoint |
| 422 | Validation Error | Pydantic rejects the input |
| 500 | Server Error | Unexpected internal failure |

---

## 🏗️ Architecture Flow

```
                 ┌─────────────────────┐
                 │       CLIENT        │
                 │  Website / App      │
                 └──────────┬──────────┘
                            │
                            │ HTTP POST
                            ▼
                 ┌─────────────────────┐
                 │   API Key Check     │
                 │   (Security Layer)  │
                 └──────────┬──────────┘
                            │
                       Valid key?
                      ┌─────┴─────┐
                     No          Yes
                      ↓            ↓
                    401      ┌──────────────────┐
                             │     FastAPI       │
                             │     Uvicorn       │
                             └──────────┬───────┘
                                        │
                                        ▼
                             ┌──────────────────┐
                             │     Pydantic      │
                             │  Input Validation │
                             └──────────┬───────┘
                                        │
                                   Valid data?
                                 ┌──────┴──────┐
                                No            Yes
                                 ↓              ↓
                               422    ┌──────────────────┐
                                      │    ML MODEL       │
                                      │  (scikit-learn)   │
                                      │  Loaded ONCE at   │
                                      │  startup          │
                                      └──────────┬───────┘
                                                 │
                                                 ▼
                                      Prediction + Confidence
                                                 │
                                                 ▼
                                      ┌──────────────────┐
                                      │  JSON RESPONSE    │
                                      │  + request_id     │
                                      │  + model_version  │
                                      └──────────┬───────┘
                                                 │
                                      ┌──────────┴──────────┐
                                      ▼                     ▼
                               Structured Logs          /metrics
                               (JSON format)                │
                                                            ▼
                                                       Prometheus
```

### Key Design Decisions

1. **Model loaded once at startup** — not per request. Loading a model is slow; keeping it in memory is fast.
2. **Validate before predict** — bad data never reaches the model. Pydantic catches it first.
3. **Request IDs on everything** — every request gets a unique ID so logs can be traced end-to-end.
4. **Versioned endpoints** — `/api/v1/predict` and `/api/v2/predict` can coexist. Old clients keep working when new versions ship.
5. **Structured JSON logs** — not `print()` statements. Searchable, filterable, production-ready.

---

## 🛠️ Tech Stack

| Technology | Role | Why |
|-----------|------|-----|
| **Python 3.10+** | Language | Industry standard for ML |
| **FastAPI** | Web framework | Fast, async, auto-docs, Pydantic integration |
| **Uvicorn** | ASGI server | Runs FastAPI, handles HTTP connections |
| **Pydantic** | Data validation | Validates inputs before they hit the model |
| **scikit-learn** | ML library | Train and serve the Iris classifier |
| **joblib** | Model serialization | Save/load trained model to disk |
| **pytest** | Testing | Automated test suite |
| **Docker** | Containerization | "Works on my machine" → works everywhere |
| **Docker Compose** | Multi-service orchestration | Run API + Prometheus together |
| **Prometheus** | Monitoring | Scrapes `/metrics` for operational data |
| **Locust** | Load testing | Simulate many concurrent requests |

---

## 📁 Planned Project Structure

```
ML-Model-Deployment-as-a-Monitored-REST-API/
│
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
├── Dockerfile                 ← Container definition
├── docker-compose.yml         ← Multi-service orchestration
├── .env.example               ← Environment variable template
├── .gitignore                 ← Files to exclude from git
│
├── app/                       ← FastAPI application
│   ├── __init__.py
│   ├── main.py                ← App entry point, startup events
│   ├── config.py              ← Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   └── predict.py     ← v1 prediction endpoint
│   │   └── v2/
│   │       ├── __init__.py
│   │       └── predict.py     ← v2 prediction endpoint
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         ← Pydantic input/output models
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction.py      ← ML model loading & prediction logic
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── logging.py         ← Structured logging middleware
│   └── core/
│       ├── __init__.py
│       └── security.py        ← API key authentication
│
├── ml/                        ← ML model training
│   ├── train.py               ← Train and save the model
│   └── saved_models/
│       └── iris_model_v1.joblib
│
├── tests/                     ← Automated tests
│   ├── __init__.py
│   ├── test_health.py
│   ├── test_predict_v1.py
│   ├── test_predict_v2.py
│   ├── test_validation.py
│   └── test_security.py
│
├── monitoring/                ← Prometheus configuration
│   └── prometheus.yml
│
├── load_tests/                ← Load testing scripts
│   └── locustfile.py
│
└── docs/                      ← Additional documentation
    └── API_DOCS.md
```

> **Inspired by:** [cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science) project structure best practices and [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) production ML scoping approach.

---

## 🗺️ 20-Task Roadmap

### Stage 1 — Foundation (Tasks 1–4)

| Task | Title | What You Build |
|------|-------|---------------|
| ✅ 1 | Understand & Plan | This README — dataset, contract, architecture |
| 2 | Project Setup | Folder structure, virtual environment, dependencies |
| 3 | Train & Save Model | Train Iris classifier, save with joblib |
| 4 | First FastAPI App | Basic `/health` endpoint running with Uvicorn |

### Stage 2 — Core API (Tasks 5–9)

| Task | Title | What You Build |
|------|-------|---------------|
| 5 | Load Model at Startup | Model loaded once, kept in memory |
| 6 | Pydantic Schemas | Input/output validation models |
| 7 | Prediction Endpoint | `POST /api/v1/predict` — **Checkpoint 1** |
| 8 | Error Handling | Graceful error responses, edge cases |
| 9 | Structured Logging | JSON logs with request_id, timestamps |

### Stage 3 — Features (Tasks 10–14)

| Task | Title | What You Build |
|------|-------|---------------|
| 10 | API Versioning | `/api/v2/predict` without breaking v1 — **Checkpoint 2** |
| 11 | Multiple Endpoints | `/health`, `/predict`, `/metrics` |
| 12 | Configuration | Environment-based config management |
| 13 | Automated Testing | pytest suite — correctness, failures, edge cases |
| 14 | Test Coverage | Security tests, validation tests |

### Stage 4 — Advanced (Tasks 15–17)

| Task | Title | What You Build |
|------|-------|---------------|
| 15 | Docker | Containerize the application |
| 16 | Docker Compose | API + Prometheus in one command |
| 17 | Security & CORS | API-key auth, CORS configuration |

### Stage 5 — Completion (Tasks 18–20)

| Task | Title | What You Build |
|------|-------|---------------|
| 18 | Prometheus Monitoring | `/metrics` endpoint, dashboard |
| 19 | Load Testing | Stress test with Locust |
| 20 | Deployment & Docs | Cloud deploy, final documentation, independent extension |

---

## 🏁 Checkpoints

### Checkpoint 1 — Task 7
> Core prediction API working: FastAPI + model loading + Pydantic validation. Send a POST request, get a prediction back.

### Checkpoint 2 — Task 14
> API versioning working: v1 and v2 coexist. v2 introduces meaningful changes without breaking v1.

### Final Checkpoint — Task 20
> Complete project delivered: API + validation + versioning + testing + Docker + security + monitoring + deployment + documentation + one independent extension.

---

## 🚦 Getting Started

> **Note:** This README is the Task 1 deliverable. Code begins in Task 2.

```bash
# Clone the repository
git clone https://github.com/Antolarkin/ML-Model-Deployment-as-a-Monitored-REST-API.git

# Navigate into the project
cd ML-Model-Deployment-as-a-Monitored-REST-API
```

Further setup instructions will be added as we complete each task.

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 👤 Author

**Antolarkin**

- GitHub: [@Antolarkin](https://github.com/Antolarkin)

---

<p align="center">
  <b>Built with ❤️ to learn ML deployment engineering</b>
</p>
