import numpy as np

from app.models.schemas import PredictionInput


def build_feature_array(items: list[PredictionInput]) -> np.ndarray:
    return np.asarray(
        [
            [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
            for item in items
        ],
        dtype=float,
    )
