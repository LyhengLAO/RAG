"""BLIP image captioner: image path → generated caption string.

The generated caption is meant to *enrich* an existing caption (from the
ingestion manifest), not replace it.  The pipeline concatenates:

    final_text = f"{original_caption}. {generated_caption}"

Cache
-----
Each image file gets a sidecar ``<image_path>.blip.txt`` next to it on first
run.  Subsequent calls read the sidecar and make zero model inferences.
Delete the sidecar to force re-captioning.

Offline mode
------------
After the BLIP weights have been downloaded once (to ``HF_HOME``), set
``TRANSFORMERS_OFFLINE=1`` to prevent any further network access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BlipCaptioner:
    """Generate image captions with ``Salesforce/blip-image-captioning-base``.

    Args:
        model_name: HuggingFace model identifier.
        device:     ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        max_length: Maximum token length for the generated caption.
        batch_size: Number of images to encode in a single forward pass.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None,
        max_length: int = 64,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self.max_length = max_length
        self.batch_size = batch_size
        self._processor: Any = None
        self._model: Any = None

    # ── lazy model loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: PLC0415

        logger.info("BlipCaptioner: loading %s on %s …", self.model_name, self.device)
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("BlipCaptioner: model ready")

    # ── cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_path(image_path: Path) -> Path:
        return image_path.with_suffix(image_path.suffix + ".blip.txt")

    @staticmethod
    def _read_cache(image_path: Path) -> str | None:
        cp = BlipCaptioner._cache_path(image_path)
        if cp.exists():
            return cp.read_text(encoding="utf-8").strip()
        return None

    @staticmethod
    def _write_cache(image_path: Path, caption: str) -> None:
        BlipCaptioner._cache_path(image_path).write_text(caption, encoding="utf-8")

    # ── public API ────────────────────────────────────────────────────────────

    def caption(self, image_path: str | Path) -> str:
        """Generate a caption for a single image, using the sidecar cache when available.

        Args:
            image_path: Path to a JPEG / PNG / WEBP image file.

        Returns:
            Generated caption string.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        cached = self._read_cache(path)
        if cached is not None:
            logger.debug("BlipCaptioner: cache hit %s", path.name)
            return cached

        self._load()
        result = self._infer_batch([path])[0]
        self._write_cache(path, result)
        return result

    def caption_batch(
        self,
        image_paths: list[str | Path],
        show_progress: bool = True,
    ) -> list[str]:
        """Generate captions for a list of images with batching and a progress bar.

        Args:
            image_paths:   Paths to image files.
            show_progress: Display a tqdm progress bar over batches.

        Returns:
            Caption strings in the same order as *image_paths*.
        """
        paths = [Path(p).resolve() for p in image_paths]
        results: dict[int, str] = {}

        # Separate cached from uncached
        uncached_indices: list[int] = []
        for i, p in enumerate(paths):
            cached = self._read_cache(p)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [results[i] for i in range(len(paths))]

        self._load()

        try:
            from tqdm import tqdm  # noqa: PLC0415

            _tqdm = tqdm
        except ImportError:
            _tqdm = None  # type: ignore[assignment]

        # Process in batches
        batch_iter = range(0, len(uncached_indices), self.batch_size)
        if show_progress and _tqdm is not None:
            batch_iter = _tqdm(
                list(batch_iter),
                desc="BLIP captioning",
                unit="batch",
            )

        for batch_start in batch_iter:
            batch_indices = uncached_indices[batch_start : batch_start + self.batch_size]
            batch_paths = [paths[i] for i in batch_indices]
            try:
                captions = self._infer_batch(batch_paths)
            except Exception as exc:
                logger.warning("BlipCaptioner: batch failed (%s) — inserting empty captions", exc)
                captions = [""] * len(batch_paths)

            for i, (orig_idx, caption) in enumerate(zip(batch_indices, captions)):
                results[orig_idx] = caption
                if caption:
                    self._write_cache(paths[orig_idx], caption)

        return [results.get(i, "") for i in range(len(paths))]

    def enrich_text(self, original_caption: str, image_path: str | Path) -> str:
        """Concatenate *original_caption* with the BLIP-generated caption.

        This is the function the pipeline calls to build the enriched ``text``
        field stored in ChromaDB alongside the CLIP native embedding.

        Args:
            original_caption: Caption from the ingestion manifest (e.g. Flickr30k).
            image_path:       Path to the image file.

        Returns:
            ``"{original_caption}. {generated_caption}"`` with deduplication of
            identical content.
        """
        generated = self.caption(image_path)
        if not generated or generated.lower() == original_caption.lower():
            return original_caption
        return f"{original_caption}. {generated}"

    # ── internal inference ────────────────────────────────────────────────────

    def _infer_batch(self, paths: list[Path]) -> list[str]:
        """Run BLIP forward pass on a batch of image paths.

        Args:
            paths: Image file paths (must already exist).

        Returns:
            List of raw caption strings, one per image.
        """
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        images = []
        valid_mask: list[bool] = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_mask.append(True)
            except Exception as exc:
                logger.warning("BlipCaptioner: could not open %s — %s", p.name, exc)
                valid_mask.append(False)

        if not any(valid_mask):
            return [""] * len(paths)

        valid_images = [img for img, ok in zip(images, valid_mask) if ok]
        inputs = self._processor(images=valid_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_length=self.max_length)

        raw_captions = self._processor.batch_decode(out_ids, skip_special_tokens=True)

        # Re-insert empty strings for failed images
        captions: list[str] = []
        cap_iter = iter(raw_captions)
        for ok in valid_mask:
            captions.append(next(cap_iter).strip() if ok else "")
        return captions
