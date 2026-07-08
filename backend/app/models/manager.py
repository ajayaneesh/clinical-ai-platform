"""Registry of loaded inference models.

Today this holds exactly one model (the current/default one). The point is the
*structure*: models are registered by name and looked up through one owner, so
future models slot in by registering more entries — no change to the queue,
worker, service, or API. It is NOT dynamic model loading or per-request
switching yet; it just prepares that seam.

Models are constructed once (respecting the load-once lifecycle) and handed to
the manager; the manager never reloads on lookup.
"""

from __future__ import annotations

from app.models.inference import InferenceModel


class ModelManager:
    def __init__(self, default_name: str) -> None:
        self._models: dict[str, InferenceModel] = {}
        self._default_name = default_name

    def register(self, name: str, model: InferenceModel) -> None:
        self._models[name] = model

    def get(self, name: str | None = None) -> InferenceModel:
        """Return a registered model by name, or the default if name is None."""
        key = name or self._default_name
        try:
            return self._models[key]
        except KeyError:
            raise KeyError(
                f"model '{key}' is not registered; available: {sorted(self._models)}"
            )

    @property
    def default_name(self) -> str:
        return self._default_name

    def names(self) -> list[str]:
        return sorted(self._models)
