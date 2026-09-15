import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass

import httpx

PREDICTION_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


@dataclass
class RequestResult:
    status_code: int | str
    latency_ms: float


async def send_prediction(client: httpx.AsyncClient) -> RequestResult:
    started_at = time.perf_counter()
    try:
        response = await client.post("/api/v1/predict", json=PREDICTION_PAYLOAD)
        return RequestResult(response.status_code, (time.perf_counter() - started_at) * 1000)
    except httpx.HTTPError:
        return RequestResult("error", (time.perf_counter() - started_at) * 1000)


def percentile(values: list[float], percent: float) -> float:
    ordered_values = sorted(values)
    index = max(0, math.ceil(len(ordered_values) * percent) - 1)
    return ordered_values[index]


def build_summary(
    results: list[RequestResult],
    concurrency: int,
    elapsed_seconds: float,
    health_status: int,
    metrics_status: int,
    metrics_body: str,
) -> dict:
    latencies = [result.latency_ms for result in results]
    status_counts = Counter(str(result.status_code) for result in results)
    failed_requests = sum(result.status_code != 200 for result in results)

    return {
        "requests": len(results),
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "requests_per_second": round(len(results) / elapsed_seconds, 2),
        "successful_requests": len(results) - failed_requests,
        "failed_requests": failed_requests,
        "error_rate_percent": round((failed_requests / len(results)) * 100, 2),
        "status_codes": dict(sorted(status_counts.items())),
        "latency_ms": {
            "min": round(min(latencies), 2),
            "mean": round(statistics.fmean(latencies), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "max": round(max(latencies), 2),
        },
        "post_load_health_status": health_status,
        "metrics_status": metrics_status,
        "custom_metric_present": "ml_predictions_total" in metrics_body,
    }


async def run_load_test(request_count: int, concurrency: int, base_url: str, timeout: float) -> dict:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    headers = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key")}
    started_at = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=headers,
        limits=limits,
        timeout=httpx.Timeout(timeout),
    ) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one() -> RequestResult:
            async with semaphore:
                return await send_prediction(client)

        results = await asyncio.gather(*(run_one() for _ in range(request_count)))
        elapsed_seconds = time.perf_counter() - started_at
        health_response = await client.get("/api/v1/health")
        metrics_response = await client.get("/metrics")

    return build_summary(
        results,
        concurrency,
        elapsed_seconds,
        health_response.status_code,
        metrics_response.status_code,
        metrics_response.text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a concurrent prediction load test")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("requests and concurrency must be positive")

    summary = await run_load_test(
        args.requests,
        args.concurrency,
        args.base_url,
        args.timeout,
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
