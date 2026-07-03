import binascii
from base64 import b64decode

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_queue
from app.core.queue import Job, Queue, QueueTimeout
from app.schemas.requests import InferenceRequest
from app.schemas.responses import (
    ErrorResponse,
    HealthResponse,
    InferenceResponse,
    RootResponse,
)

router = APIRouter()


@router.get("/")
async def root() -> RootResponse:
    return RootResponse(message="Clinical AI Platform")


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
    summary="Classify an image",
    responses={
        400: {"model": ErrorResponse, "description": "Image is not valid base64."},
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
    except QueueTimeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Prediction timed out.",
        )
    return InferenceResponse(**result)
