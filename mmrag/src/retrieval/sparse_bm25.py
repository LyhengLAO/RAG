"""BM25 sparse retriever built on rank-bm25 (BM25Okapi).

Operates on the same text corpus as the sentence-transformer dense index — text
passages, image captions, and audio transcripts are all indexed as plain text.
The index is held in memory after build; use :meth:`save` / :meth:`load` (pickle)
for fast restarts.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _whitespace_tokenizer(text: str) -> list[str]:
    """Lowercase + whitespace split — no external NLP library needed."""
    return text.lower().split()


class BM25Retriever:
    """Lexical retriever using BM25Okapi from the *rank-bm25* library.

    The index is built once from a list of :class:`langchain_core.documents.Document`
    objects.  Call :meth:`rebuild` to replace the corpus without creating a new
    instance.

    Args:
        documents:  Initial corpus.  May be empty — call :meth:`rebuild` later.
        tokenizer:  ``Callable[[str], list[str]]``.
                    Defaults to :func:`_whitespace_tokenizer` (lowercase split).
        k1:         BM25Okapi term-frequency saturation parameter (default 1.5).
        b:          BM25Okapi length-normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        documents: list[Document],
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenizer: Callable[[str], list[str]] = tokenizer or _whitespace_tokenizer
        self._k1 = k1
        self._b = b
        self._docs: list[Document] = []
        self._bm25: Any = None

        if documents:
            self.rebuild(documents)

    # ── index management ──────────────────────────────────────────────────────

    def rebuild(self, documents: list[Document]) -> None:
        """(Re)build the BM25 index from *documents*.

        Replaces any previously indexed corpus.

        Args:
            documents: Corpus to index.
        """
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        self._docs = list(documents)
        tokenised = [self._tokenizer(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(tokenised, k1=self._k1, b=self._b)
        logger.info("BM25Retriever: indexed %d documents", len(self._docs))

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Return the top-k documents by BM25 score for *query*.

        Only documents with score > 0 are returned (exact-keyword filter);
        results are sorted by descending score.

        Args:
            query: Plain-text query string (tokenised internally).
            k:     Maximum number of results.

        Returns:
            ``[(Document, bm25_score), …]`` sorted by descending score.

        Raises:
            RuntimeError: If :meth:`rebuild` has never been called.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25Retriever: index not built. Pass documents to __init__ or call rebuild()."
            )

        tokens = self._tokenizer(query)
        scores: np.ndarray = self._bm25.get_scores(tokens)

        # argsort is ascending → reverse for descending
        order = np.argsort(scores)[::-1]
        results: list[tuple[Document, float]] = []
        for idx in order:
            if len(results) >= k:
                break
            score = float(scores[idx])
            if score > 0.0:
                results.append((self._docs[int(idx)], score))

        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the index to a pickle file.

        Args:
            path: Destination path (e.g. ``chroma_db/bm25.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"docs": self._docs, "bm25": self._bm25, "k1": self._k1, "b": self._b}
        with path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25Retriever: saved %d docs to %s", len(self._docs), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> "BM25Retriever":
        """Deserialise a previously :meth:`save`-d index.

        Args:
            path:      Path to the pickle file.
            tokenizer: Must match the tokenizer used at build time.

        Returns:
            Fully initialised :class:`BM25Retriever`.
        """
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        inst = cls.__new__(cls)
        inst._tokenizer = tokenizer or _whitespace_tokenizer
        inst._docs = state["docs"]
        inst._bm25 = state["bm25"]
        inst._k1 = state.get("k1", 1.5)
        inst._b = state.get("b", 0.75)
        logger.info("BM25Retriever: loaded %d docs from %s", len(inst._docs), path)
        return inst

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def corpus_size(self) -> int:
        """Number of documents currently indexed."""
        return len(self._docs)
