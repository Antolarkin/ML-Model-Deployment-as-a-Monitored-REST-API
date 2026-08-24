import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from app.models.schemas import PredictionInput
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml/saved_model/model.joblib")
TARGET_NAMES_PATH = Path("ml/saved_model/target_names.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    app.state.target_names = joblib.load(TARGET_NAMES_PATH)
    logger.info("Model and target names loaded at startup")
    yield
    logger.info("Application shutting down")


app = FastAPI(title="ML Model API", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict[str, str | dict[str, float]]:
    model = app.state.model
    target_names = app.state.target_names

    feature_array = np.array([[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]])

    prediction_idx = model.predict(feature_array)[0]
    prediction_name = target_names[prediction_idx]
    probabilities = model.predict_proba(feature_array)[0]
    class_probabilities = {name: float(prob) for name, prob in zip(target_names, probabilities)}

    return {
        "prediction": prediction_name,
        "probabilities": class_probabilities,
    }
