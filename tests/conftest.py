"""Shared test fixtures.

Force the fast placeholder model for the entire suite so tests never trigger a
real Hugging Face download — regardless of CLINICAL_AI_MODEL_ID / .env. The model
is built by lifespan via dependencies._build_current_model (through the model
manager), which runs before FastAPI dependency_overrides apply, so we patch the
module attribute directly.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_dummy_model():
    from app.api import dependencies
    from app.core.config import settings
    from app.models.inference import DummyInferenceModel

    original = dependencies._build_current_model
    original_embeddings = settings.enable_embeddings
    dependencies._build_current_model = lambda: DummyInferenceModel()
    # Never load BiomedCLIP by default — tests must not hit the Hub. The
    # embedding_app fixture opts in explicitly with a fake service.
    settings.enable_embeddings = False
    try:
        yield
    finally:
        dependencies._build_current_model = original
        settings.enable_embeddings = original_embeddings
