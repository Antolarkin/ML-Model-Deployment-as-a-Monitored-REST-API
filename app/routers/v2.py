from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.logging_config import logger
from app.models.schemas import PredictionInput, PredictionOutputV2
from app.routers.v1 import _build_feature_array

router = APIRouter(prefix="/api/v2")


@router.post("/predict", response_model=PredictionOutputV2)
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
        confidence_score = float(max(probabilities))

        logger.info(
            "v2 Prediction successful | request_id=%s | prediction=%s | confidence_score=%.4f",
            request_id,
            prediction_name,
            confidence_score,
        )

        return {
            "request_id": request_id,
            "prediction": prediction_name,
            "confidence_score": confidence_score,
            "probabilities": class_probabilities,
            "model_version": "2.0.0",
        }
    except Exception as exc:
        logger.error("v2 Prediction failed | request_id=%s | error=%s", request_id, exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
