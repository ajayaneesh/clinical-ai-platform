from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Clinical AI Platform",
    description="API for serving clinical AI inferences.",
    version="0.1.0",
)
app.include_router(router)
