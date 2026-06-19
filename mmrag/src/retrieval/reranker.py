"""Cross-encoder reranker using sentence-transformers.

Usage pattern (over-retrieve then rerank)
-----------------------------------------
::

    # Stage 1 — over-retrieve (k_retrieval > k_final)
    candidates = hybrid.retrieve(query, k=top_k_retrieval)   # e.g. 20 candidates

    # Stage 2 — rerank to top_k_final
    final = reranker.rerank(query, candidates, k=top_k_final) # e.g. 5 results

The cross-encoder jointly encodes each ``(query, passage)`` pair, allowing it to
model fine-grained token-level interactions that bi-encoder (dense) models miss.
This extra expressiveness comes at the cost of O(N) forward passes — which is
why reranking is applied only to the small candidate set, not the full corpus.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415
        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except (ImportError, AttributeError):
        return "cpu"


class CrossEncoderReranker:
    """Re-score retrieval candidates with a cross-encoder model.

    The model jointly encodes every ``(query, passage)`` pair and produces a
    relevance logit.  Results are then sorted by that logit and truncated to *k*.

    Args:
        model_name:  HuggingFace cross-encoder model identifier.
                     Default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
                     (fast, strong on passage ranking, Apache-2.0 license).
        top_k:       Default number of results to keep after reranking.
                     Overridden by the ``k`` argument of :meth:`rerank`.
        device:      ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        batch_size:  ``(query, passage)`` pairs scored per forward pass.
                     Increase for GPU; reduce for CPU-only inference.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self._model: Any = None  # lazy-loaded on first rerank() call

    # ── lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        logger.info(
            "CrossEncoderReranker: loading %s on %s", self.model_name, self.device
        )
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("CrossEncoderReranker: ready")

    # ── public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Re-score *candidates* and return the top-k by cross-encoder score.

        The first-stage retrieval score is discarded; the returned score is the
        raw cross-encoder logit (higher = more relevant).

        Args:
            query:      The original user question.
            candidates: First-stage retrieval results ``[(Document, score), …]``.
                        Typically 10–50 candidates (over-retrieved).
            k:          How many results to return.  Defaults to :attr:`top_k`.

        Returns:
            Top-k ``[(Document, cross_encoder_score), …]`` sorted by descending
            cross-encoder score.  Returns ``[]`` when *candidates* is empty.
        """
        if not candidates:
            return []

        final_k = k if k is not None else self.top_k
        self._load()

        pairs = [(query, doc.page_content) for doc, _ in candidates]
        raw_scores: np.ndarray = self._model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(
            zip(candidates, raw_scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )

        result = [(doc, float(score)) for (doc, _first_stage), score in ranked[:final_k]]
        logger.debug(
            "CrossEncoderReranker: %d → %d candidates (k=%d)",
            len(candidates),
            len(result),
            final_k,
        )
        return result
