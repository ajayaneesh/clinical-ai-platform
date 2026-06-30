from fastapi import FastAPI

from app.api.routes import router
from app.core.logging import configure_logging
from app.core.middleware import add_logging_middleware

configure_logging()

app = FastAPI(
    title="Clinical AI Platform",
    description="API for serving clinical AI inferences.",
    version="0.1.0",
)
add_logging_middleware(app)
app.include_router(router)
