"""Retrieval-quality metrics computed from ``relevant_doc_ids``.

Implements the standard IR cut-off metrics on a **document-id** basis:

    Hit@k        1.0 if at least one relevant doc appears in the top-k, else 0.0
    Recall@k     fraction of the relevant set retrieved within the top-k
    Precision@k  fraction of the top-k that is relevant
    MRR          reciprocal rank of the first relevant doc (0 if none)
    nDCG@k       normalised discounted cumulative gain (binary relevance)

Document granularity
--------------------
Retrieval returns *chunks*, but ground truth is expressed as *document* ids
(``text_0001``, ``image_0042`` …).  The caller therefore passes the ordered
list of ``doc_id`` values behind the retrieved chunks.  Because several chunks
of the same document may be retrieved, the list is de-duplicated **preserving
first-occurrence order** before any metric is computed — so a document's rank is
the rank of its best (earliest) chunk.  This is the conventional way to score
document retrieval from a chunk-level retriever and avoids a single long
document inflating precision by occupying many top-k slots.

Purity
------
Only :mod:`numpy` is imported.  No model, network, or heavy dependency is
touched, so these functions are deterministic and unit-testable in isolation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


# ── helpers ────────────────────────────────────────────────────────────────────


def dedup_preserve_order(items: Iterable[str]) -> list[str]:
    """Return *items* with duplicates removed, keeping first-occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ── per-query primitives ───────────────────────────────────────────────────────


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc is in the top-*k* retrieved, else 0.0."""
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-*k* retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    n_rel = sum(1 for doc in topk if doc in relevant)
    return n_rel / float(k)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set retrieved within the top-*k*."""
    if not relevant or k <= 0:
        return 0.0
    topk = set(retrieved[:k])
    n_found = len(topk & relevant)
    return n_found / float(len(relevant))


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant doc (1-based); 0.0 if none found."""
    if not relevant:
        return 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """nDCG@k with binary relevance.

    DCG  = Σ_{i=1..k} rel_i / log2(i + 1)
    IDCG = Σ_{i=1..min(|relevant|, k)} 1 / log2(i + 1)
    nDCG = DCG / IDCG   (0.0 when IDCG == 0)
    """
    if not relevant or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ── corpus-level aggregation ────────────────────────────────────────────────────


def compute_retrieval_metrics(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k_values: Sequence[int] | None = None,
) -> dict[str, float]:
    """Aggregate retrieval metrics over a set of queries (macro-average).

    Args:
        retrieved_ids: For each query, the ordered list of retrieved ``doc_id``
            values (chunk-level duplicates are removed internally).
        relevant_ids: For each query, the ground-truth relevant ``doc_id`` list.
        k_values: Cut-off values for ``@k`` metrics. Defaults to ``(1, 3, 5, 10)``.

    Returns:
        Dict mapping metric name to its mean score across queries, e.g.
        ``{"hit@1", "hit@3", ..., "recall@5", "precision@5", "ndcg@10", "mrr",
        "n_queries"}``.  Queries with an empty relevant set are skipped (and
        reported via ``"n_queries"``) so they cannot silently deflate the means.

    Raises:
        ValueError: If the two input lists differ in length.
    """
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError(
            f"retrieved_ids ({len(retrieved_ids)}) and relevant_ids "
            f"({len(relevant_ids)}) must have the same length"
        )

    ks = tuple(k_values) if k_values is not None else DEFAULT_K_VALUES

    hit: dict[int, list[float]] = {k: [] for k in ks}
    rec: dict[int, list[float]] = {k: [] for k in ks}
    prec: dict[int, list[float]] = {k: [] for k in ks}
    ndcg: dict[int, list[float]] = {k: [] for k in ks}
    rr: list[float] = []

    n_scored = 0
    for retrieved, relevant_list in zip(retrieved_ids, relevant_ids):
        relevant = set(relevant_list)
        if not relevant:
            # No ground truth → metric undefined for this query; skip it.
            continue
        n_scored += 1
        retrieved = dedup_preserve_order(retrieved)
        for k in ks:
            hit[k].append(hit_at_k(retrieved, relevant, k))
            rec[k].append(recall_at_k(retrieved, relevant, k))
            prec[k].append(precision_at_k(retrieved, relevant, k))
            ndcg[k].append(ndcg_at_k(retrieved, relevant, k))
        rr.append(mrr(retrieved, relevant))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = _mean(hit[k])
        out[f"recall@{k}"] = _mean(rec[k])
        out[f"precision@{k}"] = _mean(prec[k])
        out[f"ndcg@{k}"] = _mean(ndcg[k])
    out["mrr"] = _mean(rr)
    out["n_queries"] = float(n_scored)
    return out


# ── private aliases kept for backward-compat with the original metrics stub ─────

_precision_at_k = precision_at_k
_recall_at_k = recall_at_k
_mrr = mrr
_ndcg_at_k = ndcg_at_k
