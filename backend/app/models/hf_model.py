"""A real Hugging Face image classifier for chest X-rays.

Loads the model weights and image processor ONCE at construction (call it from
startup, never per request). Uses the model's own preprocessing and class labels
so predictions match how the model was trained.

Pick a verified model id from huggingface.co (Task: Image Classification) and set
it via CLINICAL_AI_MODEL_ID. These community models are for demo/learning only —
NOT clinically validated.
"""

from __future__ import annotations

import logging
import time

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.core.metrics import FORWARD_PASS_LATENCY, PREPROCESS_LATENCY
from app.models.inference import InferenceResult
from app.services.image_processing import ImageProcessingService

logger = logging.getLogger("app.model")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HuggingFaceInferenceModel:
    def __init__(self, model_id: str, image_processing: ImageProcessingService) -> None:
        # Model lifecycle: download + load weights and the matching image
        # processor ONCE, here. Reused for every predict() call.
        self._images = image_processing
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
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[str]) -> list[InferenceResult]:
        # Load + validate each image, then run the processor (resize/crop/norm).
        t0 = time.perf_counter()
        pixels = [self._images.load(img) for img in images]
        inputs = self._processor(images=pixels, return_tensors="pt").to(self._device)
        preprocess_s = time.perf_counter() - t0
        PREPROCESS_LATENCY.observe(preprocess_s)

        t1 = time.perf_counter()
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        forward_s = time.perf_counter() - t1
        FORWARD_PASS_LATENCY.observe(forward_s)

        logger.info(
            "predict_batch",
            extra={
                "batch_size": len(images),
                "preprocess_ms": round(preprocess_s * 1000, 2),
                "inference_ms": round(forward_s * 1000, 2),
            },
        )

        # One row of probs per input image; map each to its label.
        results: list[InferenceResult] = []
        for row in probs:
            idx = int(row.argmax().item())
            results.append(
                {
                    "prediction": self._id2label[idx],
                    "confidence": float(row[idx].item()),
                }
            )
        return results
