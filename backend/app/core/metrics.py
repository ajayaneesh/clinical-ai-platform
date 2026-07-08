from prometheus_client import Counter, Gauge, Histogram

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
    "End-to-end batch latency in seconds (observed by the worker around the "
    "whole predict_batch call, including thread-pool handoff).",
)

PREPROCESS_LATENCY = Histogram(
    "preprocess_latency_seconds",
    "Image load + preprocessing latency in seconds (per batch, observed in the "
    "model before the forward pass).",
)

FORWARD_PASS_LATENCY = Histogram(
    "forward_pass_latency_seconds",
    "Model forward-pass latency in seconds (per batch, the tensor computation "
    "only, excluding preprocessing).",
)

# Cold start: how long the model took to load at startup. Set once per process.
MODEL_COLD_START = Gauge(
    "model_cold_start_seconds",
    "Time to load the model at startup (weights + move to device).",
)

# Process resource usage, sampled periodically by a background task.
PROCESS_CPU_PERCENT = Gauge(
    "process_cpu_percent",
    "Process CPU utilization percent (across all cores; can exceed 100).",
)

PROCESS_MEMORY_BYTES = Gauge(
    "process_memory_bytes",
    "Process resident set size (RSS) in bytes.",
)
