import logging
import time
import uuid

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("app.access")


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
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
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

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

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
