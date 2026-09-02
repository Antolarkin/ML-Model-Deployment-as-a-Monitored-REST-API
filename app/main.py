import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import joblib
from app.config import settings
from app.logging_config import logger
from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.target_names = None
    app.state.model_info = None
    try:
        app.state.model = joblib.load(settings.MODEL_PATH)
        app.state.target_names = joblib.load(settings.TARGET_NAMES_PATH)
        with open(settings.MODEL_METADATA_PATH) as f:
            app.state.model_info = json.load(f)
        logger.info("Model, target names, and metadata loaded at startup")
    except Exception as exc:
        logger.error("Failed to load model at startup: %s", exc)
    yield
    logger.info("Application shutting down")


app = FastAPI(title=settings.API_TITLE, version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "Request failed | request_id=%s | method=%s | path=%s | error=%s",
            request_id,
            request.method,
            request.url.path,
            exc,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request completed | request_id=%s | method=%s | path=%s | status=%d | duration=%.2fms",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is alive"}


app.include_router(v1_router)
app.include_router(v2_router)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error("ValueError raised | request_id=%s | error=%s", request.state.request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Prediction failed"},
    )
