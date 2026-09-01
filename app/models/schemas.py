from pydantic import BaseModel, Field

from app.config import settings


class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm (positive, max 10)")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm (positive, max 10)")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm (positive, max 10)")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm (positive, max 10)")


class PredictionOutput(BaseModel):
    request_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class PredictionBatchInput(BaseModel):
    items: list[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_BATCH_SIZE,
        description=f"Batch of predictions (1-{settings.MAX_BATCH_SIZE} items)",
    )


class PredictionBatchOutput(BaseModel):
    results: list[PredictionOutput]
    batch_size: int
