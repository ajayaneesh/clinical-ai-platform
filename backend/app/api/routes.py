import binascii
from base64 import b64decode, b64encode

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
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

_PREDICT_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "Invalid or undecodable image."},
    504: {"model": ErrorResponse, "description": "Prediction timed out."},
}


async def _run_inference(queue: Queue, image_b64: str) -> InferenceResponse:
    """Submit a base64 image through the queue and map failures to HTTP errors.

    Shared by /predict (base64 JSON) and /predict/upload (file) so both behave
    identically.
    """
    try:
        result = await queue.submit(Job(image=image_b64))
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input is not a decodable image.",
        )
    except QueueTimeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Prediction timed out.",
        )
    return InferenceResponse(**result)


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
    summary="Classify an image (base64 JSON)",
    responses=_PREDICT_ERROR_RESPONSES,
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
    return await _run_inference(queue, request.image)


@router.post(
    "/predict/upload",
    status_code=status.HTTP_200_OK,
    summary="Classify an uploaded image file",
    responses=_PREDICT_ERROR_RESPONSES,
)
async def predict_upload(
    file: UploadFile = File(...),
    queue: Queue = Depends(get_queue),
) -> InferenceResponse:
    # Read the uploaded bytes, then base64-encode so the same pipeline (which
    # expects a base64 string) and the same response model are reused.
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    image_b64 = b64encode(contents).decode()
    return await _run_inference(queue, image_b64)
