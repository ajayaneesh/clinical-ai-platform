"""Shared test fixtures.

Force the fast placeholder model for the entire suite so tests never trigger a
real Hugging Face download — regardless of CLINICAL_AI_MODEL_ID / .env. The model
is built by lifespan via dependencies.get_model, which runs before FastAPI
dependency_overrides apply, so we patch the module attribute directly.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_dummy_model():
    from app.api import dependencies
    from app.models.inference import DummyInferenceModel

    original = dependencies.get_model
    dependencies.get_model = lambda: DummyInferenceModel()
    try:
        yield
    finally:
        dependencies.get_model = original
