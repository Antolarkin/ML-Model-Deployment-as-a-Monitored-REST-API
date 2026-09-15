from prometheus_client import Counter

PREDICTION_SUCCESS_TOTAL = Counter(
    "ml_predictions_total",
    "Successful ML predictions by predicted class.",
    labelnames=("predicted_class",),
)


def record_successful_prediction(predicted_class: str) -> None:
    PREDICTION_SUCCESS_TOTAL.labels(predicted_class=predicted_class).inc()
