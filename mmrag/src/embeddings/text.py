"""Text embedder wrapping sentence-transformers, compatible with LangChain Embeddings API."""

from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings


class TextEmbedder(Embeddings):
    """Sentence-transformers based text embedder.

    Args:
        model_name: HuggingFace model identifier.
        device: ``"cpu"`` or ``"cuda"``.
        batch_size: Number of sentences to encode at once.
        normalize_embeddings: L2-normalise output vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of strings to encode.

        Returns:
            List of embedding vectors (as plain Python lists of floats).
        """
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query string.

        Returns:
            Embedding vector as a plain Python list of floats.
        """
        raise NotImplementedError

    def embed_numpy(self, texts: list[str]) -> np.ndarray:
        """Return embeddings as a ``(N, D)`` float32 numpy array."""
        raise NotImplementedError
