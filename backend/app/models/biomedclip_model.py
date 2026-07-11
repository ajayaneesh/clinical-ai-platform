"""OpenCLIP embedding models — the vision encoder of a CLIP-family model.

A CLIP model has a ViT vision encoder and a text encoder trained (contrastively)
into ONE shared embedding space. We use only the vision encoder (image -> latent
vector) to produce embeddings.

Two variants are supported, both loaded via open_clip:
  - BiomedCLIP (medical): loaded by an "hf-hub:" reference.
  - LAION CLIP ViT-B/32 (general): loaded by (architecture, pretrained-tag).

Weights download from the Hugging Face Hub on first use.
"""

from __future__ import annotations

import logging

import open_clip
import torch

from app.models.embedding import Embedding
from app.services.image_processing import ImageProcessingService

logger = logging.getLogger("app.embedding")

BIOMEDCLIP_HF_HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LAION_CLIP_ARCH = "ViT-B-32"
LAION_CLIP_PRETRAINED = "laion2b_s34b_b79k"


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class OpenClipEmbeddingModel:
    """Wraps a loaded open_clip model + its preprocess transform.

    Construct via the factories below rather than directly, so callers don't
    need to know each model's loading convention.
    """

    def __init__(
        self,
        model: object,
        preprocess: object,
        images: ImageProcessingService,
        label: str,
    ) -> None:
        self._images = images
        self._device = _select_device()
        self._model = model.to(self._device)  # type: ignore[attr-defined]
        self._model.eval()
        self._preprocess = preprocess
        self._label = label
        logger.info(
            "embedding_model_loaded",
            extra={"model": label, "device": str(self._device)},
        )

    @property
    def name(self) -> str:
        return self._label

    def embed(self, image: str) -> Embedding:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[str]) -> list[Embedding]:
        # Load + validate (model-agnostic), apply the model's own preprocess,
        # stack into one [N, 3, H, W] tensor.
        pixels = [self._images.load(img) for img in images]
        batch = torch.stack([self._preprocess(p) for p in pixels]).to(self._device)  # type: ignore[operator]

        with torch.no_grad():
            features = self._model.encode_image(batch)
            # L2-normalize -> unit sphere: cosine == dot product, comparable magnitudes.
            features = features / features.norm(dim=-1, keepdim=True)

        return [row.tolist() for row in features.cpu()]


def build_biomedclip(images: ImageProcessingService) -> OpenClipEmbeddingModel:
    # BiomedCLIP is loaded by an hf-hub reference (bundles its own preprocess).
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_HF_HUB)
    return OpenClipEmbeddingModel(model, preprocess, images, "biomedclip")


def build_laion_clip(images: ImageProcessingService) -> OpenClipEmbeddingModel:
    # LAION CLIP is loaded by (architecture, pretrained-tag). create_model_and_
    # transforms returns train+val transforms; we use the val (preprocess) one.
    model, _, preprocess = open_clip.create_model_and_transforms(
        LAION_CLIP_ARCH, pretrained=LAION_CLIP_PRETRAINED
    )
    return OpenClipEmbeddingModel(model, preprocess, images, "laion-clip")
