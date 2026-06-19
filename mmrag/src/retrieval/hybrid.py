"""Hybrid retriever: Reciprocal Rank Fusion of dense and sparse results.

Why RRF instead of a weighted score sum
----------------------------------------
Dense (cosine) scores and BM25 scores live on completely different numerical
scales and have different statistical distributions:

- Cosine similarities cluster near 0.9–1.0 for good results in a well-trained
  embedding space, with very little spread between ranks.
- BM25 scores scale with term frequency, document length, and IDF; a single
  rare keyword match can produce a score orders of magnitude higher than typical
  results.

A fixed linear combination  ``α * dense + β * bm25``  is fragile because:
1. **Scale mismatch**: calibrating α and β requires corpus-specific tuning and
   breaks when the corpus size or domain shifts.
2. **Outlier dominance**: one extreme BM25 score swamps the dense contribution
   regardless of α.  Similarly, min-max normalisation is query-dependent.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) solves both
problems by converting each retriever's raw scores to **ranks** before merging:

    rrf_score(doc) = Σ_{r ∈ retrievers}  1 / (k + rank_r(doc))

Ranks are dimensionless integers — scale-free by construction.  The smoothing
constant ``k`` (default 60) caps the maximum per-retriever contribution
``1/(k+1)`` so that no single top-1 hit dominates the fusion.  The value 60
was validated empirically across many TREC benchmarks and generalises well
without per-corpus tuning.

Practical note: you can widen or narrow the effective retrieval window by
changing ``top_k_per_retriever`` (more candidates → better recall before fusion)
without touching retriever-specific hyperparameters.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── duck-typed retriever protocols ────────────────────────────────────────────


@runtime_checkable
class _StringRetriever(Protocol):
    """Anything with ``retrieve(query: str, k: int) -> list[tuple[Document, float]]``."""

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        ...


@runtime_checkable
class _EmbeddingRetriever(Protocol):
    """VectorStoreIndex or mock with ``similarity_search(embedding, modality, k)``."""

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        modality: str,
        k: int,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        ...


# ── helpers ───────────────────────────────────────────────────────────────────


def _doc_key(doc: Document) -> str:
    """Stable identity key for a Document (chunk_id if present, else content hash)."""
    key = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id")
    if key:
        return str(key)
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Apply Reciprocal Rank Fusion over any number of ranked document lists.

    Documents are matched by the ``chunk_id`` metadata field (falls back to a
    SHA-256 prefix of the page content when ``chunk_id`` is absent).

    Args:
        ranked_lists: Each inner list is a ``(Document, score)`` sequence in
                      descending relevance order (rank 1 = index 0).
        k:            Smoothing constant.  Larger *k* → more uniform influence
                      across ranks; smaller *k* → higher reward for top-1 hits.

    Returns:
        Merged ``[(Document, rrf_score), …]`` sorted by descending RRF score.
        Documents that appear in multiple lists accumulate higher scores.
    """
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank_1based, (doc, _original_score) in enumerate(ranked, start=1):
            key = _doc_key(doc)
            if key not in doc_store:
                doc_store[key] = doc
                rrf_scores[key] = 0.0
            rrf_scores[key] += 1.0 / (k + rank_1based)

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [(doc_store[key], rrf_scores[key]) for key in sorted_keys]


# ── HybridRetriever ───────────────────────────────────────────────────────────


class HybridRetriever:
    """Fuse dense-vector and BM25 results using Reciprocal Rank Fusion (RRF).

    Accepts two retriever styles interchangeably:

    * **Dense**: a :class:`~src.retrieval.dense.DenseRetriever` whose
      ``retrieve(query, k)`` method embeds the query internally.
    * **Sparse**: a :class:`~src.retrieval.sparse_bm25.BM25Retriever` or any
      object with ``retrieve(query: str, k: int)`` returning
      ``[(Document, float)]``.

    The duck-typed design means mock retrievers work without any real model.

    Args:
        dense:                Retriever with ``retrieve(query, k)`` (dense side).
        sparse:               Retriever with ``retrieve(query, k)`` (sparse side).
        rrf_k:                RRF smoothing constant (default 60, see module docstring).
        top_k_per_retriever:  How many candidates each sub-retriever should fetch
                              before fusion.  Should be ≥ final ``k`` to ensure
                              recall.  Default is 20.
    """

    def __init__(
        self,
        dense: Any,    # DenseRetriever or mock — duck-typed
        sparse: Any,   # BM25Retriever or mock — duck-typed
        rrf_k: int = 60,
        top_k_per_retriever: int = 20,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._top_k_per_retriever = top_k_per_retriever

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Retrieve and RRF-fuse candidates from both dense and sparse retrievers.

        Each sub-retriever fetches ``max(k, top_k_per_retriever)`` candidates
        independently.  RRF is applied across the two ranked lists, then the
        top-*k* merged results are returned.

        Args:
            query: Natural-language query string.
            k:     Maximum number of results to return.

        Returns:
            ``[(Document, rrf_score), …]`` sorted by descending RRF score.
            RRF scores are not calibrated probabilities; they reflect rank-based
            fusion quality only.
        """
        n = max(k, self._top_k_per_retriever)

        dense_hits  = self._dense.retrieve(query, k=n)
        sparse_hits = self._sparse.retrieve(query, k=n)

        logger.debug(
            "HybridRetriever: dense=%d, sparse=%d hits before RRF",
            len(dense_hits),
            len(sparse_hits),
        )

        fused = self._reciprocal_rank_fusion([dense_hits, sparse_hits], k=self._rrf_k)
        top_k = fused[:k]

        logger.debug("HybridRetriever: %d docs after RRF (k=%d)", len(top_k), k)
        return top_k

    # ── static method kept for API compatibility with the original stub ────────

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[Document, float]]],
        k: int = 60,
    ) -> list[tuple[Document, float]]:
        """Thin delegator to the module-level :func:`reciprocal_rank_fusion`.

        Kept as a ``@staticmethod`` for backwards compatibility with callers that
        used the original stub's method name.

        Args:
            ranked_lists: Multiple ``(Document, score)`` ranked sequences.
            k:            RRF smoothing constant.

        Returns:
            Merged ranked list; see :func:`reciprocal_rank_fusion`.
        """
        return reciprocal_rank_fusion(ranked_lists, k=k)
