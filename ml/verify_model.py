import logging
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml/saved_model/model.joblib")
TARGET_NAMES_PATH = Path("ml/saved_model/target_names.joblib")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not TARGET_NAMES_PATH.exists():
        raise FileNotFoundError(f"Target names not found at {TARGET_NAMES_PATH}")

    model = joblib.load(MODEL_PATH)
    target_names = joblib.load(TARGET_NAMES_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction_idx = model.predict(sample)[0]
    prediction_name = target_names[prediction_idx]
    probabilities = model.predict_proba(sample)[0]
    class_probabilities = {name: float(prob) for name, prob in zip(target_names, probabilities)}

    sample_str = f"[[{sample[0][0]} {sample[0][1]}, {sample[0][2]}, {sample[0][3]}]]"
    print(f"Prediction for sample {sample_str}")
    print(f"Prediction: {prediction_name}")
    print(f"Probabilities: {class_probabilities}")


if __name__ == "__main__":
    main()
