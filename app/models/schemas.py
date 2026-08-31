from pydantic import BaseModel, Field


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
        max_length=100,
        description="Batch of predictions (1-100 items)",
    )


class PredictionBatchOutput(BaseModel):
    results: list[PredictionOutput]
    batch_size: int
