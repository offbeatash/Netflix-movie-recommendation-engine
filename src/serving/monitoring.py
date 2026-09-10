from prometheus_client import Counter, Gauge, Histogram

# HTTP Metrics
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the recommendation API.",
    ("method", "path", "status"),
)
HTTP_ERRORS = Counter(
    "http_errors_total",
    "Total HTTP responses with a client or server error status.",
    ("method", "path", "status"),
)
HTTP_REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "path"),
)

# Recommendation Request Metrics
RECOMMENDATION_REQUESTS = Counter(
    "recommendation_requests_total",
    "Total recommendation requests.",
)
COLD_START_REQUESTS = Counter(
    "cold_start_requests_total",
    "Recommendation requests for users absent from the training data.",
)
PERSONALIZED_REQUESTS = Counter(
    "personalized_recommendation_requests_total",
    "Recommendation requests served with personalized predictions.",
)
EMPTY_RECOMMENDATION_RESULTS = Counter(
    "empty_recommendation_results_total",
    "Recommendation requests that returned no recommendations.",
)
RECOMMENDATIONS_RETURNED = Counter(
    "recommendations_returned_total",
    "Total number of recommendation rows returned.",
)

# Recommendation Latency Metrics
RECOMMENDATION_LATENCY = Histogram(
    "recommendation_generation_duration_seconds",
    "Time spent generating recommendations.",
    ("request_type",),  # personalized or cold_start
)

# Error Metrics by Category
INFERENCE_ERRORS = Counter(
    "inference_errors_total",
    "Total errors during recommendation inference.",
    ("error_type",),  # e.g., "model_loading", "data_loading", "prediction"
)
MODEL_LOADING_ERRORS = Counter(
    "model_loading_errors_total",
    "Total errors during model loading.",
    ("model_type",),  # e.g., "popularity", "als", "svd", "ensemble"
)
DATA_LOADING_ERRORS = Counter(
    "data_loading_errors_total",
    "Total errors during data loading.",
    ("data_type",),  # e.g., "train", "val", "test", "movies"
)

# Model/Service Health Metrics
MODEL_LOAD_STATUS = Gauge(
    "model_load_status",
    "Status of model loading (1 = loaded, 0 = failed, -1 = not attempted).",
    ("model_type",),  # e.g., "popularity", "svd", "ensemble"
)
DATA_LOAD_STATUS = Gauge(
    "data_load_status",
    "Status of data loading (1 = loaded, 0 = failed, -1 = not attempted).",
    ("data_type",),  # e.g., "train", "movies"
)
SERVICE_UPTIME_SECONDS = Gauge(
    "service_uptime_seconds",
    "Seconds since service started.",
)
MODEL_CACHE_INITIALIZATION_SECONDS = Gauge(
    "model_cache_initialization_seconds",
    "Time spent initializing the inference data and model cache.",
)


def observe_metric(operation, *args, **kwargs):
    """Keep telemetry failures from affecting API behavior."""
    try:
        operation(*args, **kwargs)
    except Exception:
        pass
