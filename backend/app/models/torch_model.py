"""A real PyTorch inference model that still returns a fixed (fake) label.

The full inference machinery is genuine — device selection, base64->PIL->tensor
preprocessing, a forward pass under torch.no_grad on the selected device — so the
pipeline exercises real PyTorch work. Only the final label mapping is a
placeholder; swap the marked section for a trained model + real class labels.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision import transforms

from app.models.inference import InferenceResult
from app.services.image_processing import ImageProcessingService

logger = logging.getLogger("app.model")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchInferenceModel:
    def __init__(self, image_processing: ImageProcessingService) -> None:
        # Model lifecycle: build + move to device ONCE, at construction — never
        # per request. eval() disables training-only layers (dropout/batchnorm).
        self._images = image_processing
        self._device = _select_device()
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 2),  # 2 logits: placeholder classifier
        ).to(self._device)
        self._model.eval()

        # Standard preprocessing: resize -> tensor. Real models add normalization.
        self._preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        logger.info("model_loaded", extra={"device": str(self._device)})

    def predict(self, image: str) -> InferenceResult:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[str]) -> list[InferenceResult]:
        # 1. Load + validate + preprocess each image, then STACK into one tensor
        #    of shape [N, 3, 224, 224]. A failure here raises InvalidImageError;
        #    the worker validates per-image first, so inputs reaching here are
        #    already valid and the batch stays intact.
        tensors = [self._preprocess(self._images.load(img)) for img in images]
        batch = torch.stack(tensors).to(self._device)

        # 2. ONE forward pass over the whole batch (no_grad: inference only).
        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)

        # --- PLACEHOLDER label mapping -----------------------------------
        # The batched forward pass above is real; the labels below are fake.
        # Replace with real class mapping, e.g. per row:
        #   idx = int(probs[i].argmax()); CLASSES[idx], float(probs[i, idx])
        _ = probs  # computed for realism; not yet used for the labels
        return [{"prediction": "normal", "confidence": 0.95} for _ in images]
