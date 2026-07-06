from base64 import b64encode
from io import BytesIO

import pytest
from PIL import Image

from app.models.inference import InvalidImageError
from app.services.image_processing import ImageProcessingService


def _png_b64(size: tuple[int, int] = (8, 8)) -> str:
    buf = BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, "PNG")
    return b64encode(buf.getvalue()).decode()


@pytest.fixture
def service() -> ImageProcessingService:
    return ImageProcessingService(max_bytes=1024)


def test_loads_valid_image_to_rgb_pil(service):
    img = service.load(_png_b64())
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_rejects_invalid_base64(service):
    with pytest.raises(InvalidImageError):
        service.load("not base64!!!")


def test_rejects_empty_input(service):
    with pytest.raises(InvalidImageError):
        service.load(b64encode(b"").decode())


def test_rejects_valid_base64_that_is_not_an_image(service):
    with pytest.raises(InvalidImageError):
        service.load(b64encode(b"totally not an image").decode())


def test_rejects_oversized_image(service):
    # Bytes exceeding the 1024-byte limit are rejected before decode. Use random
    # noise so it doesn't compress below the cap the way a flat-color PNG would.
    big = b64encode(b"\x00\xff" * 2000).decode()  # ~4000 bytes > 1024
    with pytest.raises(InvalidImageError):
        service.load(big)


def test_converts_grayscale_to_rgb(service):
    buf = BytesIO()
    Image.new("L", (8, 8), 128).save(buf, "PNG")  # single-channel grayscale
    img = service.load(b64encode(buf.getvalue()).decode())
    assert img.mode == "RGB"
