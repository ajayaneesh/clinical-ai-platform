import binascii
from base64 import b64decode

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.dependencies import get_queue
from app.core.queue import Job, Queue, QueueTimeout
from app.models.inference import InvalidImageError
from app.schemas.requests import InferenceRequest
from app.schemas.responses import (
    ErrorResponse,
    HealthResponse,
    InferenceResponse,
    ReadyResponse,
    RootResponse,
)

router = APIRouter()


@router.get("/")
async def root() -> RootResponse:
    return RootResponse(message="Clinical AI Platform")


@router.get("/health", summary="Liveness: is the process alive?")
async def health() -> HealthResponse:
    # Liveness: if this returns at all, the event loop is responsive.
    return HealthResponse(status="healthy")


@router.get(
    "/ready",
    summary="Readiness: can we serve traffic?",
    responses={503: {"model": ErrorResponse, "description": "Not ready to serve."}},
)
async def ready(request: Request) -> ReadyResponse:
    # Readiness: we can only process /predict if the queue (and its worker) is up.
    if getattr(request.app.state, "queue", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference queue is not initialized.",
        )
    return ReadyResponse(status="ready")


@router.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
    summary="Classify an image",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid base64 or undecodable image.",
        },
        504: {"model": ErrorResponse, "description": "Prediction timed out."},
    },
)
async def predict(
    request: InferenceRequest,
    queue: Queue = Depends(get_queue),
) -> InferenceResponse:
    try:
        b64decode(request.image, validate=True)
    except binascii.Error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image could not be decoded as base64.",
        )

    try:
        result = await queue.submit(Job(image=request.image))
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input is valid base64 but not a decodable image.",
        )
    except QueueTimeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Prediction timed out.",
        )
    return InferenceResponse(**result)
