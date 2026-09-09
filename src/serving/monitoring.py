from prometheus_client import Counter, Gauge, Histogram

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
