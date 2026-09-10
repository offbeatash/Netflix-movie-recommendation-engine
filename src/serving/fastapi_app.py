import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import (
    BASELINE_MODEL_PATH,
    ENRICHED_MOVIES_PATH,
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

logger = logging.getLogger(__name__)

# Initialize API
app = FastAPI(
    title="Netflix Recommendation Engine",
    description=(
        "Production API for collaborative filtering and popularity-based "
        "movie recommendations."
    ),
    version="1.0.0",
)

# Track service start time for uptime metric
_SERVICE_START_TIME = time.perf_counter()


# Define request schema
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
    path = request.url.path
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        raise
    finally:
        elapsed = time.perf_counter() - start
        labels = (request.method, path, str(status_code))
        observe_metric(HTTP_REQUESTS.labels(*labels).inc)
        if status_code >= 400:
            observe_metric(HTTP_ERRORS.labels(*labels).inc)
        observe_metric(
            HTTP_REQUEST_LATENCY.labels(request.method, path).observe,
            elapsed,
        )


@app.get("/")
def root():
    """Basic API information endpoint."""
    return {
        "service": "Netflix Recommendation Engine",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
        "recommendations": "/recommend",
    }


@app.get("/health")
def health_check():
    """Liveness probe for container orchestration."""
    return {"status": "healthy", "service": "recommendation-api"}


@app.get("/ready")
def readiness_check():
    """Reports whether serving data and pre-trained artifacts are available."""
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
    # Update uptime gauge before serving metrics
    # Update uptime gauge before serving metrics
    uptime = time.perf_counter() - _SERVICE_START_TIME
    observe_metric(SERVICE_UPTIME_SECONDS.set, uptime)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    """Generates top-N movie recommendations per genre for a given user."""
    observe_metric(RECOMMENDATION_REQUESTS.inc)
    start_time = time.perf_counter()
    try:
        status_msg, results_df = generate_genre_recommendations(
            user_id=request.user_id, top_n=request.top_n
        )
        recommendation_count = len(results_df)
        observe_metric(RECOMMENDATIONS_RETURNED.inc, recommendation_count)
        if recommendation_count == 0:
            observe_metric(EMPTY_RECOMMENDATION_RESULTS.inc)
        if "not found" in status_msg.lower():
            observe_metric(COLD_START_REQUESTS.inc)
            request_type = "cold_start"
        else:
            observe_metric(PERSONALIZED_REQUESTS.inc)
            request_type = "personalized"

        # Record recommendation generation latency
        observe_metric(
            RECOMMENDATION_LATENCY.labels(request_type=request_type).observe,
            time.perf_counter() - start_time,
        )

        return {
            "status_message": status_msg,
            "recommendations": results_df.to_dict(orient="records"),
        }
    except Exception as e:
        # Track inference errors
        error_type = "inference"
        if "not found" in str(e).lower() or "missing" in str(e).lower():
            error_type = "data_loading"
        elif "model" in str(e).lower():
            error_type = "model_loading"
        observe_metric(INFERENCE_ERRORS.labels(error_type=error_type).inc)
        logger.exception("Recommendation request failed")
        raise HTTPException(
            status_code=500,
            detail="An internal inference error occurred.",
        )
