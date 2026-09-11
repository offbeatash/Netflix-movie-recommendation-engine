import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import (
    API_KEY,
    BASELINE_MODEL_PATH,
    CORS_ALLOW_ORIGINS,
    ENRICHED_MOVIES_PATH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    SVD_MODEL_PATH,
    TRAIN_DATA_PATH,
)
from src.inference.recommend import generate_genre_recommendations
from src.serving.monitoring import (
    COLD_START_REQUESTS,
    EMPTY_RECOMMENDATION_RESULTS,
    HTTP_ERRORS,
    HTTP_REQUESTS,
    HTTP_REQUEST_LATENCY,
    INFERENCE_ERRORS,
    PERSONALIZED_REQUESTS,
    RECOMMENDATION_LATENCY,
    RECOMMENDATION_REQUESTS,
    RECOMMENDATIONS_RETURNED,
    SERVICE_UPTIME_SECONDS,
    observe_metric,
)
from src.serving.security import RateLimiter, require_api_key

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Netflix Recommendation Engine",
    description="Classical ML movie recommendations with temporal offline evaluation.",
    version="2.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOW_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_SERVICE_START_TIME = time.perf_counter()
_RATE_LIMITER = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = Field(default=5, ge=1, le=100)


REQUIRED_RUNTIME_PATHS = (
    ("train_data", TRAIN_DATA_PATH),
    ("movie_metadata", ENRICHED_MOVIES_PATH),
    ("popularity_model", BASELINE_MODEL_PATH),
    ("svd_model", SVD_MODEL_PATH),
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        labels = (request.method, request.url.path, str(status_code))
        observe_metric(HTTP_REQUESTS.labels(*labels).inc)
        if status_code >= 400:
            observe_metric(HTTP_ERRORS.labels(*labels).inc)
        observe_metric(
            HTTP_REQUEST_LATENCY.labels(request.method, request.url.path).observe,
            elapsed,
        )


@app.get("/")
def root():
    return {
        "service": "Netflix Recommendation Engine",
        "version": "2.2.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
        "recommendations": "/recommend",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "recommendation-api"}


@app.get("/ready")
def readiness_check():
    missing = [
        name for name, path in REQUIRED_RUNTIME_PATHS if not Path(path).is_file()
    ]
    if missing:
        return Response(
            content='{"status":"not_ready","missing":['
            + ",".join(f'"{name}"' for name in missing)
            + "]}",
            status_code=503,
            media_type="application/json",
        )
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    uptime = time.perf_counter() - _SERVICE_START_TIME
    observe_metric(SERVICE_UPTIME_SECONDS.set, uptime)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest, http_request: Request):
    """Generate recommendations without blocking the FastAPI event loop."""
    require_api_key(http_request, API_KEY)
    client_key = http_request.client.host if http_request.client else "unknown"
    if not _RATE_LIMITER.allow(client_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    observe_metric(RECOMMENDATION_REQUESTS.inc)
    start_time = time.perf_counter()
    try:
        status_msg, results_df = await run_in_threadpool(
            generate_genre_recommendations,
            user_id=request.user_id,
            top_n=request.top_n,
        )
        count = len(results_df)
        observe_metric(RECOMMENDATIONS_RETURNED.inc, count)
        if count == 0:
            observe_metric(EMPTY_RECOMMENDATION_RESULTS.inc)
        if "not found" in status_msg.lower():
            observe_metric(COLD_START_REQUESTS.inc)
            request_type = "cold_start"
        else:
            observe_metric(PERSONALIZED_REQUESTS.inc)
            request_type = "personalized"
        observe_metric(
            RECOMMENDATION_LATENCY.labels(request_type=request_type).observe,
            time.perf_counter() - start_time,
        )
        return {
            "status_message": status_msg,
            "recommendations": results_df.to_dict(orient="records"),
        }
    except Exception as exc:
        error_type = "inference"
        if "missing" in str(exc).lower() or "not found" in str(exc).lower():
            error_type = "data_loading"
        elif "model" in str(exc).lower():
            error_type = "model_loading"
        observe_metric(INFERENCE_ERRORS.labels(error_type=error_type).inc)
        logger.exception("Recommendation request failed")
        raise HTTPException(
            status_code=500, detail="An internal inference error occurred."
        ) from exc
