"""A real PyTorch inference model that still returns a fixed (fake) label.

The full inference machinery is genuine — device selection, base64->PIL->tensor
preprocessing, a forward pass under torch.no_grad on the selected device — so the
pipeline exercises real PyTorch work. Only the final label mapping is a
placeholder; swap the marked section for a trained model + real class labels.
"""

from __future__ import annotations

import io
import logging
from base64 import b64decode

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from app.models.inference import InferenceResult, InvalidImageError

logger = logging.getLogger("app.model")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchInferenceModel:
    def __init__(self) -> None:
        # Model lifecycle: build + move to device ONCE, at construction — never
        # per request. eval() disables training-only layers (dropout/batchnorm).
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
        # 1. Decode the base64 image into a PIL image (RGB).
        try:
            pixels = Image.open(io.BytesIO(b64decode(image))).convert("RGB")
        except Exception as exc:
            raise InvalidImageError(str(exc)) from exc

        # 2. Preprocess -> tensor, add batch dim, move to device.
        tensor = self._preprocess(pixels).unsqueeze(0).to(self._device)

        # 3. Forward pass (no_grad: inference only, no autograd overhead).
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)

        # --- PLACEHOLDER label mapping -----------------------------------
        # The forward pass above is real; the label below is fake. Replace with
        # a trained model and a real class list, e.g.:
        #   idx = int(probs.argmax(dim=1))
        #   return {"prediction": CLASSES[idx], "confidence": float(probs[0, idx])}
        _ = probs  # computed for realism; not yet used for the label
        return {"prediction": "normal", "confidence": 0.95}
