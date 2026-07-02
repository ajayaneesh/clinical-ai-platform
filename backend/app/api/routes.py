from fastapi import APIRouter, Depends

from app.api.dependencies import get_inference_service
from app.schemas.responses import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    RootResponse,
)
from app.services.inference import InferenceService

router = APIRouter()


@router.get("/")
async def root() -> RootResponse:
    return RootResponse(message="Clinical AI Platform")


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.post("/infer")
async def infer(
    request: InferenceRequest,
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    result = service.predict(request.text)
    return InferenceResponse(**result)
