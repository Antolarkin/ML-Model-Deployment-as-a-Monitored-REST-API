from pydantic import BaseModel, ConfigDict, Field

from app.config import settings


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm (positive, max 10)")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm (positive, max 10)")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm (positive, max 10)")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm (positive, max 10)")


class PredictionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]


class PredictionOutputV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    prediction: str
    confidence_score: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str


class PredictionBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_BATCH_SIZE,
        description=f"Batch of predictions (1-{settings.MAX_BATCH_SIZE} items)",
    )


class PredictionBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[PredictionOutput]
    batch_size: int
