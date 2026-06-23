"""Audio embedder: transcribe with Whisper, then embed transcript as text."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings.text import TextEmbedder


class AudioEmbedder:
    """Embed audio files via transcription (faster-whisper) + text encoding.

    Args:
        text_embedder: A :class:`TextEmbedder` instance used after transcription.
        whisper_model_size: Whisper model variant.
        device: ``"cpu"`` or ``"cuda"``.
        language: ISO 639-1 language code, or ``None`` for auto-detection.
    """

    def __init__(
        self,
        text_embedder: TextEmbedder | None = None,
        whisper_model_size: str = "base",
        device: str = "cpu",
        language: str | None = None,
    ) -> None:
        raise NotImplementedError

    def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file to a plain string.

        Args:
            file_path: Path to the audio file.

        Returns:
            Full transcript as a single string.
        """
        raise NotImplementedError

    def embed_audio(self, file_path: str | Path) -> np.ndarray:
        """Transcribe then embed a single audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Embedding vector as float32 array of shape ``(D,)``.
        """
        raise NotImplementedError

    def embed_audios(self, file_paths: list[str | Path]) -> np.ndarray:
        """Transcribe and embed a batch of audio files.

        Args:
            file_paths: List of paths to audio files.

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        raise NotImplementedError
