import logging
import time
import uuid

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger("app.access")


def _record_metrics(
    method: str, endpoint: str, status_code: int, seconds: float
) -> None:
    labels = (method, endpoint, str(status_code))
    REQUEST_LATENCY.labels(*labels).observe(seconds)
    REQUEST_COUNT.labels(*labels).inc()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured log line per request.

    Runs BEFORE the endpoint (capture start time, mint a request id) and
    AFTER it (now status code and latency are known) — see the request/
    response bracket below.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id  # available to route handlers
        start = time.perf_counter()

        try:
            # --- hand off to the endpoint ---
            response = await call_next(request)
            # --- back from the endpoint: status & latency are now known ---
        except Exception:
            # The endpoint raised. Log it with the same fields (status 500),
            # then re-raise so FastAPI's error handling still produces the
            # response — we observe the failure, we don't swallow it.
            elapsed = time.perf_counter() - start
            latency_ms = round(elapsed * 1000, 2)
            _record_metrics(request.method, request.url.path, 500, elapsed)
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "status_code": 500,
                    "latency_ms": latency_ms,
                },
            )
            raise

        elapsed = time.perf_counter() - start
        latency_ms = round(elapsed * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        _record_metrics(request.method, request.url.path, response.status_code, elapsed)

        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response


def add_logging_middleware(app: FastAPI) -> None:
    app.add_middleware(LoggingMiddleware)
