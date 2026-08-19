import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml/saved_model/model.joblib")
TARGET_NAMES_PATH = Path("ml/saved_model/target_names.joblib")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    iris = load_iris()
    return iris.data, iris.target, iris.target_names.tolist()


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]
    )
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")
    return accuracy


def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")


def save_target_names(target_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(target_names, output_path)
    logger.info(f"Target names saved to {output_path}")


def main() -> None:
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)
    save_target_names(target_names, TARGET_NAMES_PATH)


if __name__ == "__main__":
    main()
