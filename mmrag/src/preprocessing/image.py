"""Image preprocessing: resize, normalize, and convert to tensors for CLIP."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# ImageNet normalisation constants (used by most torchvision/CLIP backbones)
_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image(
    file_path: str | Path,
    target_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """Load an image file and apply standard preprocessing (resize + normalize).

    Args:
        file_path: Path to the image file.
        target_size: ``(width, height)`` to resize to.
        normalize: Whether to apply ImageNet mean/std normalisation.

    Returns:
        Float32 numpy array of shape ``(C, H, W)``.
    """
    from PIL import Image  # noqa: PLC0415

    image = Image.open(file_path).convert("RGB").resize(target_size, Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, C)

    if normalize:
        array = (array - _IMAGENET_MEAN) / _IMAGENET_STD

    return np.transpose(array, (2, 0, 1))  # (C, H, W)


@lru_cache(maxsize=4)
def _get_open_clip_preprocess(model_name: str):
    import open_clip  # noqa: PLC0415

    _, _, preprocess = open_clip.create_model_and_transforms(model_name)
    return preprocess


def open_clip_transform(image: Any, model_name: str = "ViT-B-32") -> Any:
    """Return the preprocessing transform expected by a given open_clip model.

    Args:
        image: PIL Image or path.
        model_name: open_clip model identifier.

    Returns:
        Preprocessed tensor ready for the encoder.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(image, (str, Path)):
        image = PILImage.open(image).convert("RGB")

    preprocess = _get_open_clip_preprocess(model_name)
    return preprocess(image)
