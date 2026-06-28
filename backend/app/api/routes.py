from fastapi import APIRouter

from app.schemas.responses import HealthResponse, RootResponse

router = APIRouter()


@router.get("/")
async def root() -> RootResponse:
    return RootResponse(message="Clinical AI Platform")


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="healthy:)")
