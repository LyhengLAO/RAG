"""Sentence-transformers text embedder, LangChain Embeddings-compatible.

Default model: ``BAAI/bge-small-en-v1.5`` (384-dim, strong RAG performance,
Apache-2.0).  Drop-in compatible with all sentence-transformers models.

BGE-specific note
-----------------
BGE models perform best when queries are prefixed with an instruction string.
The correct instruction is applied automatically based on model name; pass
``query_instruction=""`` to disable or ``query_instruction="..."`` to override.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Asymmetric-search query instructions for known BGE variants
_BGE_INSTRUCTIONS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5":  "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en":      "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en":       "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en":      "Represent this sentence for searching relevant passages: ",
}


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415
        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except (ImportError, AttributeError):
        return "cpu"


class TextEmbedder(Embeddings):
    """Sentence-transformers text embedder, L2-normalised by default.

    Implements :class:`langchain_core.embeddings.Embeddings` so it can be
    plugged directly into any LangChain retriever or vector store.

    Args:
        model_name:        HuggingFace model identifier.
        device:            ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        batch_size:        Sentences encoded per forward pass.
        normalize:         L2-normalise output vectors (recommended for cosine retrieval).
        query_instruction: Prefix prepended to query strings only.
                           Defaults to the BGE instruction when relevant, ``""`` otherwise.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        query_instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_instruction = (
            query_instruction
            if query_instruction is not None
            else _BGE_INSTRUCTIONS.get(model_name, "")
        )
        self._model: Any = None  # lazy-loaded

    # ── lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("TextEmbedder: loading %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("TextEmbedder: ready, dim=%d", self.embedding_dim)

    # ── LangChain Embeddings interface ────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document passages (no query instruction prepended).

        Args:
            texts: Strings to encode.

        Returns:
            List of float lists, shape ``(N, D)``.
        """
        return self.embed_numpy(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string with the BGE instruction if applicable.

        Args:
            text: Query string.

        Returns:
            Float list of shape ``(D,)``.
        """
        return self.embed_numpy([self.query_instruction + text])[0].tolist()

    # ── numpy helpers ─────────────────────────────────────────────────────────

    def embed_numpy(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* and return a ``(N, D)`` float32 numpy array.

        Args:
            texts: Strings to encode.

        Returns:
            Float32 array of shape ``(len(texts), embedding_dim)``.
        """
        self._load()
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_query_numpy(self, text: str) -> np.ndarray:
        """Encode a single query as a float32 vector of shape ``(D,)``.

        Args:
            text: Query string (instruction prepended automatically).

        Returns:
            Float32 array of shape ``(embedding_dim,)``.
        """
        return self.embed_numpy([self.query_instruction + text])[0]

    # ── metadata ──────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_id(self) -> str:
        """Short identifier used in stable Chroma IDs."""
        return self.model_name.split("/")[-1]
