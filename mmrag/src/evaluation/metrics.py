"""Backward-compatible facade over the evaluation metric modules.

The real implementations live in dedicated modules so they can be imported and
tested independently of heavy optional dependencies:

* :mod:`src.evaluation.retrieval_metrics`  — Hit@k, MRR, Recall@k, nDCG@k
  (pure numpy, no extra deps).
* :mod:`src.evaluation.ragas_eval`         — RAGAS suite (lazy-imports
  ``ragas`` / ``datasets`` only when actually called).

This module re-exports the stable public names used elsewhere in the codebase
and keeps the original private ``_precision_at_k`` / ``_mrr`` / … helpers so
existing imports continue to work.  No metric is a placeholder.
"""

from __future__ import annotations

from typing import Sequence

from src.evaluation.retrieval_metrics import (
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    compute_retrieval_metrics,
)

__all__ = [
    "compute_ragas_metrics",
    "compute_retrieval_metrics",
    "_precision_at_k",
    "_recall_at_k",
    "_mrr",
    "_ndcg_at_k",
]


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Evaluate generated answers with RAGAS (local Ollama judge).

    Thin wrapper around :func:`src.evaluation.ragas_eval.compute_ragas` that
    returns only the aggregate mean per metric.  ``ragas``/``datasets`` are
    imported lazily inside that function, so importing this module is cheap.

    Args:
        questions: Input questions.
        answers: Generated answers (one per question).
        contexts: Retrieved context strings per question.
        ground_truths: Reference answers per question.
        metrics: RAGAS metric names. Defaults to the full supported set.

    Returns:
        Dict mapping metric name to its mean score. Failed metrics are ``NaN``
        (never fabricated).
    """
    from src.evaluation.ragas_eval import compute_ragas  # noqa: PLC0415

    result = compute_ragas(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        metric_names=list(metrics) if metrics is not None else None,
    )
    return result["overall"]
