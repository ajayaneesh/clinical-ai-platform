"""Image preprocessing service: the model-agnostic front of the pipeline.

Responsibilities (everything up to a validated PIL image):
  - load:     base64 string -> raw bytes
  - validate: reject empty / oversized / non-image input
  - convert:  bytes -> RGB PIL.Image

It deliberately stops at PIL. The model-specific step (Hugging Face
AutoImageProcessor -> tensors) stays in the model, since the processor must
match the model it feeds. This keeps the service reusable across any model.
"""

from __future__ import annotations

import io
from base64 import b64decode

from PIL import Image

from app.models.inference import InvalidImageError


class ImageProcessingService:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes

    def load(self, image_b64: str) -> Image.Image:
        raw = self._decode(image_b64)
        self._validate(raw)
        return self._to_pil(raw)

    def _decode(self, image_b64: str) -> bytes:
        try:
            return b64decode(image_b64, validate=True)
        except Exception as exc:
            raise InvalidImageError(f"input is not valid base64: {exc}") from exc

    def _validate(self, raw: bytes) -> None:
        if not raw:
            raise InvalidImageError("image is empty")
        if len(raw) > self._max_bytes:
            raise InvalidImageError(
                f"image is {len(raw)} bytes, exceeds limit of {self._max_bytes}"
            )

    def _to_pil(self, raw: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(raw))
            image.verify()  # detect truncated/corrupt files before use
            # verify() leaves the image unusable; reopen to actually load pixels.
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except InvalidImageError:
            raise
        except Exception as exc:
            raise InvalidImageError(f"not a decodable image: {exc}") from exc
