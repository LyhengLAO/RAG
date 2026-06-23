"""open-clip image+text embedder for cross-modal retrieval.

Both image and text encoders project into the *same* 512-dim CLIP space so
a text query can retrieve images and vice-versa — no alignment step needed.

Architecture decision (see session notes)
-----------------------------------------
This embedder feeds the dedicated ``image_clip`` ChromaDB collection.
The text collection uses :class:`TextEmbedder`; never mix the two.

Default model: ViT-B-32 / OpenAI pretrained weights (512-dim, Apache-2.0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415
        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except (ImportError, AttributeError):
        return "cpu"


class ClipEmbedder:
    """Encode images AND text into the CLIP embedding space via open_clip.

    Args:
        model_name:  open_clip architecture name (e.g. ``"ViT-B-32"``).
        pretrained:  Weight tag (e.g. ``"openai"`` or ``"laion2b_s34b_b79k"``).
        device:      ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        batch_size:  Items processed per forward pass.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    # ── lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        logger.info(
            "ClipEmbedder: loading %s / %s on %s", self.model_name, self.pretrained, self.device
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model.to(self.device).eval()
        logger.info("ClipEmbedder: ready, dim=%d", self.embedding_dim)

    # ── image encoding ────────────────────────────────────────────────────────

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Encode a single image file.

        Args:
            image_path: Path to a JPEG / PNG / WEBP image.

        Returns:
            L2-normalised float32 vector of shape ``(D,)``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return self.embed_images([path])[0]

    def embed_images(
        self,
        image_paths: list[str | Path],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a batch of image files.

        Skips and logs corrupted images; the corresponding row is a zero vector.

        Args:
            image_paths:   Paths to image files.
            show_progress: Display a tqdm progress bar over batches.

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._load()

        try:
            from tqdm import tqdm  # noqa: PLC0415
            _wrap = tqdm if show_progress else (lambda x, **kw: x)
        except ImportError:
            _wrap = lambda x, **kw: x  # noqa: E731

        all_vecs: list[np.ndarray] = []
        paths = [Path(p).resolve() for p in image_paths]

        for batch_start in _wrap(
            range(0, len(paths), self.batch_size),
            desc="CLIP image",
            unit="batch",
        ):
            batch_paths = paths[batch_start : batch_start + self.batch_size]
            tensors: list[Any] = []
            valid: list[bool] = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self._preprocess(img))
                    valid.append(True)
                except Exception as exc:
                    logger.warning("ClipEmbedder: could not load %s — %s", p.name, exc)
                    valid.append(False)

            if not any(valid):
                for _ in batch_paths:
                    all_vecs.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            good_tensors = [t for t, ok in zip(tensors, valid) if ok]
            batch_tensor = torch.stack(good_tensors).to(self.device)

            with torch.no_grad():
                feats = self._model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            feat_iter = iter(feats.cpu().numpy().astype(np.float32))
            for ok in valid:
                if ok:
                    all_vecs.append(next(feat_iter))
                else:
                    all_vecs.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return np.stack(all_vecs) if all_vecs else np.empty((0, self.embedding_dim), dtype=np.float32)

    # ── text encoding (cross-modal queries) ───────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text query into the CLIP *image* space for cross-modal retrieval.

        Args:
            text: Natural-language query.

        Returns:
            L2-normalised float32 vector of shape ``(D,)``.
        """
        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a batch of text strings in the CLIP space.

        Args:
            texts:         List of query strings.
            show_progress: Display a tqdm progress bar over batches.

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        import torch  # noqa: PLC0415

        self._load()

        try:
            from tqdm import tqdm  # noqa: PLC0415
            _wrap = tqdm if show_progress else (lambda x, **kw: x)
        except ImportError:
            _wrap = lambda x, **kw: x  # noqa: E731

        all_vecs: list[np.ndarray] = []
        for batch_start in _wrap(
            range(0, len(texts), self.batch_size),
            desc="CLIP text",
            unit="batch",
        ):
            batch = texts[batch_start : batch_start + self.batch_size]
            tokens = self._tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_vecs.append(feats.cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs) if all_vecs else np.empty((0, self.embedding_dim), dtype=np.float32)

    # ── metadata ──────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimensionality (512 for ViT-B-32)."""
        self._load()
        import torch  # noqa: PLC0415
        dummy = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            return int(self._model.encode_image(dummy).shape[-1])

    @property
    def model_id(self) -> str:
        """Short identifier used in stable Chroma IDs."""
        return f"clip-{self.model_name.lower().replace('/', '-')}"
