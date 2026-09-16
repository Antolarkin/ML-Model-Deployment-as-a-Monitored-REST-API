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
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

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
        with settings.MODEL_METADATA_PATH.open(encoding="utf-8") as metadata_file:
            app.state.model_info = json.load(metadata_file)
        logger.info("Model, target names, and metadata loaded at startup")
    except Exception:
        logger.exception("Failed to load model at startup")
        raise
    yield


app = FastAPI(title=settings.API_TITLE, version="0.1.0", lifespan=lifespan)

instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app, include_in_schema=False)

_original_openapi = app.openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = _original_openapi()
    openapi_schema.setdefault("components", {})
    openapi_schema["components"].setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["APIKeyHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }
    openapi_schema["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
