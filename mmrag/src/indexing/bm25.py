"""BM25 sparse index backed by rank-bm25.

Delegates all index logic to :class:`~src.retrieval.sparse_bm25.BM25Retriever`
so the core BM25 implementation lives in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.documents import Document


class BM25Index:
    """In-memory BM25 index with optional disk persistence via pickle.

    Args:
        tokenizer: ``Callable[[str], list[str]]``.  Defaults to lowercase
                   whitespace split.  Must be consistent between build and load.
    """

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        # Import lazily so that BM25Index can be imported without rank_bm25 installed
        from src.retrieval.sparse_bm25 import BM25Retriever  # noqa: PLC0415
        self._inner = BM25Retriever([], tokenizer=tokenizer)

    def build(self, documents: list[Document]) -> None:
        """Build the BM25 index from *documents*.

        Args:
            documents: Documents whose ``page_content`` will be indexed.
        """
        self._inner.rebuild(documents)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Return the *k* highest-scoring documents for *query*.

        Args:
            query: Natural-language query string.
            k:     Number of results to return.

        Returns:
            ``[(Document, bm25_score), …]`` sorted by descending score.
        """
        return self._inner.retrieve(query, k=k)

    def save(self, path: str | Path) -> None:
        """Serialize the BM25 index to disk (pickle).

        Args:
            path: Destination file path.
        """
        self._inner.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> "BM25Index":
        """Deserialize a BM25 index from disk.

        Args:
            path:      Path to a previously saved pickle file.
            tokenizer: Must match the tokenizer used at build time.

        Returns:
            Loaded :class:`BM25Index` instance.
        """
        from src.retrieval.sparse_bm25 import BM25Retriever  # noqa: PLC0415
        inst = cls.__new__(cls)
        inst._inner = BM25Retriever.load(path, tokenizer=tokenizer)
        return inst
