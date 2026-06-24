"""System-level metrics: latency percentiles, token counts, per-modality split.

These describe the *cost* of a pipeline rather than its answer quality:

    latency_ms   wall-clock time of ``pipeline.query()`` (p50 / p95 / mean)
    tokens       prompt + completion tokens reported by the LLM provider

Both are aggregated **overall** and **broken down per modality** (text / image /
audio) so the comparison can show, for example, that audio questions are slower
or cheaper than text ones.

NaN policy
----------
A per-query value of ``NaN`` means "could not be measured" (e.g. the LLM was
unavailable so the provider returned no token counts).  All aggregation uses
``nan``-aware reductions and additionally reports how many samples were valid,
so a missing measurement is never silently treated as a zero.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

# A per-query record is expected to expose at least these keys.
LATENCY_KEY = "latency_ms"
MODALITY_KEY = "modality"
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _clean(values: Iterable[float]) -> np.ndarray:
    """Return a float array with NaN/inf entries removed."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def percentiles(values: Iterable[float]) -> dict[str, float]:
    """Summarise a sample with p50, p95, mean, min, max and valid count.

    Non-finite values (NaN / inf) are dropped before computing statistics. If no
    finite value remains, every statistic is ``NaN`` and ``n`` is ``0``.

    Args:
        values: Iterable of numeric samples.

    Returns:
        ``{"p50", "p95", "mean", "min", "max", "n"}``.
    """
    arr = _clean(values)
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return {"p50": nan, "p95": nan, "mean": nan, "min": nan, "max": nan, "n": 0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": n,
    }


def _token_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-token-type percentile summary across *records*."""
    out: dict[str, dict[str, float]] = {}
    for key in TOKEN_KEYS:
        vals = [r.get(key, float("nan")) for r in records]
        out[key] = percentiles(vals)
    return out


def aggregate_system_metrics(
    per_query: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate latency and token usage overall and per modality.

    Args:
        per_query: One dict per evaluated question, each containing at least
            ``latency_ms`` and ``modality`` and, when available, the token-count
            keys ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
            (use ``NaN`` when a value could not be measured).

    Returns:
        ``{"overall": {...}, "per_modality": {modality: {...}}}`` where each
        inner block has ``{"latency_ms": <percentiles>, "tokens": {...},
        "n_queries": int}``.
    """

    def _block(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "latency_ms": percentiles(r.get(LATENCY_KEY, float("nan")) for r in records),
            "tokens": _token_summary(records),
            "n_queries": len(records),
        }

    overall = _block(per_query)

    modalities = sorted(
        {str(r.get(MODALITY_KEY, "unknown")) for r in per_query}
    )
    per_modality = {
        m: _block([r for r in per_query if str(r.get(MODALITY_KEY, "unknown")) == m])
        for m in modalities
    }

    return {"overall": overall, "per_modality": per_modality}


def token_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """Compute the token usage of a single query from cumulative LLM counters.

    Given two snapshots of :attr:`src.generation.llm.LLMClient.total_usage`
    (taken immediately before and after one ``pipeline.query()`` call), return
    the per-query ``prompt`` / ``completion`` / ``total`` token deltas.

    If any LLM call during the window failed to report usage (``missing``
    increased) the counts are returned as ``NaN`` — never as a fabricated 0.

    Args:
        before: ``total_usage`` snapshot before the query.
        after: ``total_usage`` snapshot after the query.

    Returns:
        ``{"prompt_tokens", "completion_tokens", "total_tokens"}``.
    """
    nan = float("nan")
    d_calls = after.get("calls", 0) - before.get("calls", 0)
    d_missing = after.get("missing", 0) - before.get("missing", 0)

    if d_calls <= 0 or d_missing > 0:
        # No successful, usage-reporting LLM call in this window.
        return {"prompt_tokens": nan, "completion_tokens": nan, "total_tokens": nan}

    d_prompt = after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0)
    d_completion = after.get("completion_tokens", 0) - before.get("completion_tokens", 0)
    return {
        "prompt_tokens": float(d_prompt),
        "completion_tokens": float(d_completion),
        "total_tokens": float(d_prompt + d_completion),
    }
