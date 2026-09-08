from collections import defaultdict
from time import perf_counter

from fastapi import Depends, HTTPException, Request

from app.config import settings

_request_log: dict[str, list[float]] = defaultdict(list)


async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


async def enforce_rate_limit(request: Request):
    api_key = request.headers.get("X-API-Key", "anonymous")
    now = perf_counter()
    window = settings.RATE_LIMIT_WINDOW_SECONDS
    max_requests = settings.RATE_LIMIT_REQUESTS

    timestamps = _request_log[api_key]
    cutoff = now - window
    _request_log[api_key] = [t for t in timestamps if t > cutoff]

    if len(_request_log[api_key]) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {max_requests} requests per {window} seconds",
        )

    _request_log[api_key].append(now)
