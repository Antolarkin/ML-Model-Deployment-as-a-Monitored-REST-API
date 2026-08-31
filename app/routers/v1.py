import logging
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.logging_config import logger
from app.models.schemas import PredictionInput, PredictionOutput

router = APIRouter(prefix="/api/v1")

MODEL_PATH = Path("ml/saved_model/model.joblib")
TARGET_NAMES_PATH = Path("ml/saved_model/target_names.joblib")


class InferenceError(Exception):
    pass


@router.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@router.post("/predict", response_model=PredictionOutput)
def predict(request: Request, payload: PredictionInput) -> dict:
    model = request.app.state.model
    target_names = request.app.state.target_names
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


# v2 planning: if we need to add extra fields to the predict response
# without breaking existing v1 clients, create a new router at prefix="/api/v2".
# Candidate additions: model_version, prediction_id, processing_time_ms.
# v1 contract remains untouched so existing clients continue to work.
