import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from app.logging_config import logger
from app.models.schemas import PredictionInput, PredictionOutput
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logging.getLogger("ml_api").setLevel(logging.INFO)

MODEL_PATH = Path("ml/saved_model/model.joblib")
TARGET_NAMES_PATH = Path("ml/saved_model/target_names.joblib")


class InferenceError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.target_names = None
    try:
        app.state.model = joblib.load(MODEL_PATH)
        app.state.target_names = joblib.load(TARGET_NAMES_PATH)
        logger.info("Model and target names loaded at startup")
    except Exception as exc:
        logger.error("Failed to load model at startup: %s", exc)
    yield
    logger.info("Application shutting down")


app = FastAPI(title="ML Model API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "Request failed | request_id=%s | method=%s | path=%s | error=%s",
            request_id,
            request.method,
            request.url.path,
            exc,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request completed | request_id=%s | method=%s | path=%s | status=%d | duration=%.2fms",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is alive"}


@app.get("/health")
def health() -> dict[str, str | bool]:
    model_loaded = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionOutput)
def predict(request: Request, payload: PredictionInput) -> dict:
    model = app.state.model
    target_names = app.state.target_names
    request_id = request.state.request_id

    if model is None or target_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_array = np.array([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]])

        prediction_idx = int(model.predict(feature_array)[0])
        prediction_name = target_names[prediction_idx]
        probabilities = model.predict_proba(feature_array)[0]
        class_probabilities = {name: float(prob) for name, prob in zip(target_names, probabilities)}
        confidence = float(max(probabilities))

        logger.info(
            "Prediction successful | request_id=%s | prediction=%s | confidence=%.4f",
            request_id,
            prediction_name,
            confidence,
        )

        return {
            "request_id": request_id,
            "prediction": prediction_name,
            "confidence": confidence,
            "probabilities": class_probabilities,
        }
    except InferenceError as exc:
        logger.error("Inference failed | request_id=%s | error=%s", request_id, exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    except Exception as exc:
        logger.error("Unexpected error during prediction | request_id=%s | error=%s", request_id, exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error("ValueError raised | request_id=%s | error=%s", request.state.request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Prediction failed"},
    )
