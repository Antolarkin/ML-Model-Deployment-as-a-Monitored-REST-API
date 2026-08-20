from fastapi import FastAPI

app = FastAPI(title="ML Model API", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict() -> dict[str, str]:
    return {"prediction": "hardcoded_result"}
