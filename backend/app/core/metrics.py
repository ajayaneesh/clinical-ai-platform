from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "endpoint", "status_code"),
)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    labelnames=("method", "endpoint", "status_code"),
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds (measured around the model call).",
)
