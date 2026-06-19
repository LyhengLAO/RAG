"""Sparse (BM25) retriever — thin wrapper around BM25Index."""

from __future__ import annotations

from langchain_core.documents import Document

from src.indexing.bm25 import BM25Index


class SparseRetriever:
    """Retrieve documents via BM25 keyword matching.

    Wraps a pre-built :class:`~src.indexing.bm25.BM25Index`.  For a
    self-contained retriever that builds its own index from a document list,
    use :class:`~src.retrieval.sparse_bm25.BM25Retriever` directly.

    Args:
        index: A built :class:`BM25Index`.
        top_k: Default number of results to return.
    """

    def __init__(self, index: BM25Index, top_k: int = 5) -> None:
        self._index = index
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Retrieve the highest BM25-scoring documents for *query*.

        Args:
            query: Natural-language query string.
            k:     Override default ``top_k``.

        Returns:
            ``[(Document, bm25_score), …]`` sorted by descending score.
        """
        return self._index.search(query, k=k if k is not None else self.top_k)
