"""Dense (ANN) retriever backed by the ChromaDB vector store."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from langchain_core.documents import Document

from src.indexing.vector_store import VectorStoreIndex

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieve documents by dense vector similarity (cosine).

    The :class:`VectorStoreIndex` stores pre-computed embeddings; this class
    adds the *query embedding* step so callers only need to pass a plain string.

    Args:
        index:           Built :class:`VectorStoreIndex`.
        embedder:        Any object with ``embed_query_numpy(query: str) -> np.ndarray``.
                         Typically a :class:`~src.embeddings.text_embedder.TextEmbedder`
                         or :class:`~src.embeddings.clip_embedder.ClipEmbedder`.
        top_k:           Default number of results to return.
        score_threshold: Minimum similarity score (0 = no filtering).
        modality:        Chroma collection to search (default ``"text"``).
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        embedder: Any,
        top_k: int = 5,
        score_threshold: float = 0.0,
        modality: str = "text",
    ) -> None:
        self._index = index
        self._embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.modality = modality

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Embed *query* and retrieve the most similar documents.

        Args:
            query:  Natural-language query string.
            k:      Override default ``top_k``.
            filter: Optional ChromaDB ``where`` clause.

        Returns:
            ``[(Document, cosine_score), …]`` sorted by descending score.
        """
        final_k = k if k is not None else self.top_k
        threshold = self.score_threshold if self.score_threshold > 0.0 else None

        vec: np.ndarray = self._embedder.embed_query_numpy(query)
        return self._index.similarity_search(
            query_embedding=vec,
            modality=self.modality,
            k=final_k,
            filter=filter,
            score_threshold=threshold,
        )
