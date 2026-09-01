import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.logging_config import logger
from app.models.schemas import (
    PredictionBatchInput,
    PredictionBatchOutput,
    PredictionInput,
    PredictionOutput,
)

router = APIRouter(prefix="/api/v1")


class InferenceError(Exception):
    pass


def _build_feature_array(items: list[PredictionInput]) -> np.ndarray:
    """Convert validated inputs into a 2D array for scikit-learn batch inference."""
    return np.array([
        [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
        for item in items
    ])


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
        feature_array = _build_feature_array([payload])

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


@router.post("/predict-batch", response_model=PredictionBatchOutput)
def predict_batch(request: Request, payload: PredictionBatchInput) -> dict:
    model = request.app.state.model
    target_names = request.app.state.target_names
    request_id = request.state.request_id

    if model is None or target_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(payload.items) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(payload.items)} exceeds maximum allowed size of {settings.MAX_BATCH_SIZE}",
        )

    start_time = time.perf_counter()

    try:
        feature_array = _build_feature_array(payload.items)

# Run inference on the entire batch in one model call.
# This avoids repeated Python/model-call overhead and lets
# scikit-learn process the batch efficiently.

        predictions_idx = model.predict(feature_array)
        probabilities = model.predict_proba(feature_array)

        results = []
        for pred_idx, proba in zip(predictions_idx, probabilities):
            pred_name = target_names[int(pred_idx)]
            confidence = float(max(proba))
            class_probabilities = {name: float(p) for name, p in zip(target_names, proba)}
            item_request_id = str(uuid.uuid4())
            results.append({
                "request_id": item_request_id,
                "prediction": pred_name,
                "confidence": confidence,
                "probabilities": class_probabilities,
            })

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Batch prediction successful | request_id=%s | batch_size=%d | duration=%.2fms",
            request_id,
            len(payload.items),
            duration_ms,
        )

        return {
            "results": results,
            "batch_size": len(payload.items),
        }
    except Exception as exc:
        logger.error("Batch prediction failed | request_id=%s | error=%s", request_id, exc)
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc


@router.get("/model-info")
def model_info(request: Request) -> dict:
    model_info = getattr(request.app.state, "model_info", None)
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return model_info


# v2 planning: if we need to add extra fields to the predict response
# without breaking existing v1 clients, create a new router at prefix="/api/v2".
# Candidate additions: model_version, prediction_id, processing_time_ms.
# v1 contract remains untouched so existing clients continue to work.
