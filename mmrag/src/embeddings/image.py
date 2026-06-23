"""Image embedder using open_clip visual encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ImageEmbedder:
    """Encode images into dense vectors via open_clip.

    Args:
        model_name: open_clip architecture name (e.g. ``"ViT-B-32"``).
        pretrained: Pretrained weights tag (e.g. ``"openai"``).
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        raise NotImplementedError

    def embed_image(self, file_path: str | Path) -> np.ndarray:
        """Encode a single image file into a 1-D embedding vector.

        Args:
            file_path: Path to the image file.

        Returns:
            L2-normalised float32 vector of shape ``(D,)``.
        """
        raise NotImplementedError

    def embed_images(self, file_paths: list[str | Path]) -> np.ndarray:
        """Encode a batch of images.

        Args:
            file_paths: List of paths to image files.

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        raise NotImplementedError

    def embed_text_for_image_retrieval(self, text: str) -> np.ndarray:
        """Encode a text query into the *image* embedding space via CLIP text encoder.

        Enables cross-modal retrieval (query: text → results: images).

        Args:
            text: Natural-language query string.

        Returns:
            L2-normalised float32 vector of shape ``(D,)``.
        """
        raise NotImplementedError
