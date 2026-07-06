"""A real Hugging Face image classifier for chest X-rays.

Loads the model weights and image processor ONCE at construction (call it from
startup, never per request). Uses the model's own preprocessing and class labels
so predictions match how the model was trained.

Pick a verified model id from huggingface.co (Task: Image Classification) and set
it via CLINICAL_AI_MODEL_ID. These community models are for demo/learning only —
NOT clinically validated.
"""

from __future__ import annotations

import io
import logging
from base64 import b64decode

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.models.inference import InferenceResult, InvalidImageError

logger = logging.getLogger("app.model")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HuggingFaceInferenceModel:
    def __init__(self, model_id: str) -> None:
        # Model lifecycle: download + load weights and the matching image
        # processor ONCE, here. Reused for every predict() call.
        self._device = _select_device()
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()
        # The model ships its own class names — use them, never hardcode indices.
        self._id2label = self._model.config.id2label
        logger.info(
            "model_loaded",
            extra={"model_id": model_id, "device": str(self._device)},
        )

    def predict(self, image: str) -> InferenceResult:
        try:
            pixels = Image.open(io.BytesIO(b64decode(image))).convert("RGB")
        except Exception as exc:
            raise InvalidImageError(str(exc)) from exc

        # The processor applies the EXACT preprocessing the model was trained
        # with (resize, crop, normalization) — do not hand-roll it.
        inputs = self._processor(images=pixels, return_tensors="pt").to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        idx = int(probs.argmax(dim=1).item())
        return {
            "prediction": self._id2label[idx],
            "confidence": float(probs[0, idx].item()),
        }
