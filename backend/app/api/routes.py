import asyncio
import binascii
from base64 import b64decode, b64encode
from datetime import datetime, timezone

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

from app.api.dependencies import (
    get_embedding_service,
    get_embedding_store,
    get_queue,
)
from app.core.embedding_store import EmbeddingStore
from app.core.queue import Job, Queue, QueueTimeout
from app.models.inference import InvalidImageError
from app.schemas.requests import EmbedRequest, InferenceRequest, SearchRequest
from app.schemas.responses import (
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    InferenceResponse,
    ReadyResponse,
    RootResponse,
    SearchHitResponse,
    SearchResponse,
)
from app.services.embedding import EmbeddingService

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


async def _run_embedding(
    service: EmbeddingService,
    store: EmbeddingStore,
    image_b64: str,
    filename: str | None,
    diagnosis_label: str | None,
) -> EmbeddingResponse:
    """Embed a base64 image, store the vector + metadata, return id + timing.

    Shared by /embed (base64 JSON) and /embed/upload (file) so both behave
    identically.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        # Blocking forward pass -> offload to a thread so the event loop stays free.
        vector = await loop.run_in_executor(None, service.embed, image_b64)
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input is not a decodable image.",
        )
    inference_ms = round((loop.time() - start) * 1000, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    embedding_id = store.add(
        vector,
        service.model_name,
        filename=filename,
        diagnosis_label=diagnosis_label,
        timestamp=timestamp,
    )
    return EmbeddingResponse(
        embedding_id=embedding_id,
        model=service.model_name,
        embedding=vector,
        dimension=len(vector),
        inference_ms=inference_ms,
        filename=filename,
        diagnosis_label=diagnosis_label,
        timestamp=timestamp,
    )


@router.post(
    "/embed",
    status_code=status.HTTP_200_OK,
    summary="Generate an embedding for an image (base64 JSON)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or undecodable image."},
    },
)
async def embed(
    request: EmbedRequest,
    service: EmbeddingService = Depends(get_embedding_service),
    store: EmbeddingStore = Depends(get_embedding_store),
) -> EmbeddingResponse:
    try:
        b64decode(request.image, validate=True)
    except binascii.Error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image could not be decoded as base64.",
        )
    return await _run_embedding(
        service, store, request.image, request.filename, request.diagnosis_label
    )


@router.post(
    "/embed/upload",
    status_code=status.HTTP_200_OK,
    summary="Generate an embedding for an uploaded image file",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or undecodable image."},
    },
)
async def embed_upload(
    file: UploadFile = File(...),
    diagnosis_label: str | None = None,
    service: EmbeddingService = Depends(get_embedding_service),
    store: EmbeddingStore = Depends(get_embedding_store),
) -> EmbeddingResponse:
    # Read the uploaded bytes, base64-encode, reuse the same embedding path.
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    return await _run_embedding(
        service,
        store,
        b64encode(contents).decode(),
        file.filename,
        diagnosis_label,
    )


async def _run_search(
    service: EmbeddingService,
    store: EmbeddingStore,
    image_b64: str,
    top_k: int,
    diagnosis_label: str | None = None,
) -> SearchResponse:
    """Embed a query image, search the store by cosine similarity, and report
    the top-k hits plus timing/memory measurements."""
    loop = asyncio.get_running_loop()

    # 1. Embedding generation time.
    t0 = loop.time()
    try:
        query_vec = await loop.run_in_executor(None, service.embed, image_b64)
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input is not a decodable image.",
        )
    embedding_ms = round((loop.time() - t0) * 1000, 2)

    # 2. Similarity search latency (cosine is cheap CPU work; kept on the loop).
    t1 = loop.time()
    hits = store.search(query_vec, top_k=top_k, diagnosis_label=diagnosis_label)
    search_ms = round((loop.time() - t1) * 1000, 2)

    return SearchResponse(
        results=[
            SearchHitResponse(
                embedding_id=h.embedding_id,
                score=h.score,
                model=h.model,
                filename=h.filename,
                diagnosis_label=h.diagnosis_label,
                timestamp=h.timestamp,
            )
            for h in hits
        ],
        searched=store.count(),
        embedding_ms=embedding_ms,
        search_ms=search_ms,
        store_memory_bytes=store.memory_bytes(),
    )


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Find the top-k stored images most similar to a query image (base64)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or undecodable image."},
    },
)
async def search(
    request: SearchRequest,
    service: EmbeddingService = Depends(get_embedding_service),
    store: EmbeddingStore = Depends(get_embedding_store),
) -> SearchResponse:
    try:
        b64decode(request.image, validate=True)
    except binascii.Error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image could not be decoded as base64.",
        )
    return await _run_search(
        service, store, request.image, top_k=5, diagnosis_label=request.diagnosis_label
    )


@router.post(
    "/search/upload",
    status_code=status.HTTP_200_OK,
    summary="Find the top-k stored images most similar to an uploaded query image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or undecodable image."},
    },
)
async def search_upload(
    file: UploadFile = File(...),
    diagnosis_label: str | None = None,
    service: EmbeddingService = Depends(get_embedding_service),
    store: EmbeddingStore = Depends(get_embedding_store),
) -> SearchResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    return await _run_search(
        service,
        store,
        b64encode(contents).decode(),
        top_k=5,
        diagnosis_label=diagnosis_label,
    )
