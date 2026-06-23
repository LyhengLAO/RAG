"""CLAP (Contrastive Language-Audio Pretraining) embedder for audio+text retrieval.

Uses ``laion/clap-htsat-fused`` via HuggingFace transformers.  Both audio and
text are projected into the *same* 512-dim space, enabling cross-modal
audio-to-text and text-to-audio retrieval.

Architecture note
-----------------
CLAP is kept as a standalone component; in the current multimodal strategy it
is NOT merged with the text collection.  Use it only if you create a dedicated
``audio_clap`` ChromaDB collection.  Audio transcripts (Whisper) go into the
``text`` collection instead for simpler maintenance.

Sampling rate
-------------
CLAP expects 48 kHz mono audio.  Files at other rates are resampled with
librosa before inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CLAP_SAMPLE_RATE = 48_000  # model requirement


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415
        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except (ImportError, AttributeError):
        return "cpu"


class ClapEmbedder:
    """Audio + text embedder using laion/clap-htsat-fused.

    Args:
        model_name: HuggingFace CLAP model identifier.
        device:     ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        batch_size: Files per forward pass.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-fused",
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None

    # ── lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        logger.info("ClapEmbedder: loading %s on %s …", self.model_name, self.device)
        self._processor = ClapProcessor.from_pretrained(self.model_name)
        self._model = ClapModel.from_pretrained(self.model_name)
        self._model.to(self.device).eval()
        logger.info("ClapEmbedder: ready")

    # ── internal audio loader ─────────────────────────────────────────────────

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        """Load and resample an audio file to 48 kHz mono float32."""
        import librosa  # noqa: PLC0415
        audio, _ = librosa.load(str(path), sr=_CLAP_SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)

    # ── audio encoding ────────────────────────────────────────────────────────

    def embed_audio(self, audio_path: str | Path) -> np.ndarray:
        """Encode a single audio file.

        Args:
            audio_path: Path to a WAV / MP3 / FLAC / M4A file.

        Returns:
            L2-normalised float32 vector of shape ``(D,)``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return self.embed_audios([path])[0]

    def embed_audios(
        self,
        audio_paths: list[str | Path],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a batch of audio files.

        Skips and logs files that cannot be loaded; the corresponding row is
        a zero vector.

        Args:
            audio_paths:   Paths to audio files.
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

        paths = [Path(p).resolve() for p in audio_paths]
        all_vecs: list[np.ndarray] = []

        for batch_start in _wrap(
            range(0, len(paths), self.batch_size),
            desc="CLAP audio",
            unit="batch",
        ):
            batch_paths = paths[batch_start : batch_start + self.batch_size]
            arrays: list[np.ndarray] = []
            valid: list[bool] = []

            for p in batch_paths:
                try:
                    arrays.append(self._load_audio(p))
                    valid.append(True)
                except Exception as exc:
                    logger.warning("ClapEmbedder: could not load %s — %s", p.name, exc)
                    valid.append(False)

            if not any(valid):
                for _ in batch_paths:
                    all_vecs.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            good_arrays = [a for a, ok in zip(arrays, valid) if ok]
            inputs = self._processor(
                audios=good_arrays,
                sampling_rate=_CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                feats = self._model.get_audio_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            feat_np = feats.cpu().numpy().astype(np.float32)
            feat_iter = iter(feat_np)
            for ok in valid:
                if ok:
                    all_vecs.append(next(feat_iter))
                else:
                    all_vecs.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return np.stack(all_vecs) if all_vecs else np.empty((0, self.embedding_dim), dtype=np.float32)

    # ── text encoding (cross-modal queries) ───────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text query into the CLAP *audio* space for cross-modal retrieval.

        Args:
            text: Natural-language description of the target audio.

        Returns:
            L2-normalised float32 vector of shape ``(D,)``.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings into the CLAP audio space.

        Args:
            texts: List of query strings.

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        import torch  # noqa: PLC0415

        self._load()
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy().astype(np.float32)

    # ── metadata ──────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimensionality (512 for clap-htsat-fused)."""
        return 512

    @property
    def model_id(self) -> str:
        """Short identifier used in stable Chroma IDs."""
        return self.model_name.split("/")[-1]
