from fastapi import APIRouter

from app.schemas.responses import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    RootResponse,
)
from app.services.inference import InferenceService

router = APIRouter()
inference_service = InferenceService()


@router.get("/")
async def root() -> RootResponse:
    return RootResponse(message="Clinical AI Platform")


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.post("/infer")
async def infer(request: InferenceRequest) -> InferenceResponse:
    result = inference_service.predict(request.text)
    return InferenceResponse(**result)
