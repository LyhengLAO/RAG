"""RAGAS evaluation with a fully local judge (Ollama LLM + local embeddings).

Metrics
-------
* ``faithfulness``        — is every claim in the answer supported by the context?
* ``answer_relevancy``    — does the answer address the question? (uses embeddings)
* ``context_precision``   — are the retrieved contexts ranked by usefulness?
* ``context_recall``      — did retrieval surface everything the ground truth needs?
* ``answer_correctness``  — does the answer match the ground truth? (available
  because the eval set ships ground-truth answers)

Judge model
-----------
All LLM-based metrics use a **local Ollama** model (the same backend the
pipelines generate with). ``answer_relevancy`` and ``answer_correctness`` also
need an embedding model; a local sentence-transformers model is used so the
whole evaluation runs offline with no API keys.

No fabricated numbers
---------------------
``ragas``/``datasets``/``langchain`` are imported lazily so this module is cheap
to import. If the import fails, Ollama is unreachable, or a metric raises, the
affected scores are returned as ``NaN`` and the error is logged — a metric is
never replaced by a made-up value. RAGAS itself also emits per-row ``NaN`` for
rows it cannot score (``raise_exceptions=False``); those propagate through the
nan-aware aggregation below.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def _nan_result(metric_names: list[str], n_rows: int, error: str) -> dict[str, Any]:
    """Build an all-NaN result block (used when evaluation cannot run)."""
    nan = float("nan")
    return {
        "overall": {m: nan for m in metric_names},
        "per_row": [{m: nan for m in metric_names} for _ in range(n_rows)],
        "valid_counts": {m: 0 for m in metric_names},
        "n_rows": n_rows,
        "error": error,
    }


def _build_judge(judge_model: str | None, embedding_model: str) -> tuple[Any, Any]:
    """Construct (llm, embeddings) LangChain objects for the local judge.

    Tries the modern ``langchain_ollama`` / ``langchain_huggingface`` packages
    first and falls back to ``langchain_community`` for older installs.

    Raises:
        ImportError: If no compatible LangChain integration is available.
    """
    model = judge_model or settings.ollama_model
    base_url = settings.ollama_host

    # ── chat LLM (Ollama) ──────────────────────────────────────────────────────
    llm = None
    try:
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)
    except Exception:  # noqa: BLE001 - fall back to community integration
        from langchain_community.chat_models import ChatOllama  # noqa: PLC0415

        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)

    # ── embeddings (local sentence-transformers) ───────────────────────────────
    emb = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415

        emb = HuggingFaceEmbeddings(model_name=embedding_model)
    except Exception:  # noqa: BLE001
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415

        emb = HuggingFaceEmbeddings(model_name=embedding_model)

    return llm, emb


def _resolve_metrics(metric_names: list[str]) -> list[Any]:
    """Map metric name strings to RAGAS metric objects."""
    import ragas.metrics as rm  # noqa: PLC0415

    available = {
        "faithfulness": getattr(rm, "faithfulness", None),
        "answer_relevancy": getattr(rm, "answer_relevancy", None),
        "context_precision": getattr(rm, "context_precision", None),
        "context_recall": getattr(rm, "context_recall", None),
        "answer_correctness": getattr(rm, "answer_correctness", None),
    }
    resolved: list[Any] = []
    for name in metric_names:
        metric = available.get(name)
        if metric is None:
            logger.warning("ragas_eval: metric %r not available in this ragas version — skipped", name)
            continue
        resolved.append(metric)
    return resolved


def compute_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    metric_names: list[str] | None = None,
    judge_model: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Run the RAGAS suite with a local Ollama judge.

    Args:
        questions: One question per row.
        answers: Generated answer per row.
        contexts: Retrieved context strings per row (list[str] each).
        ground_truths: Reference answer per row.
        metric_names: Subset of :data:`DEFAULT_METRICS`. Defaults to all.
        judge_model: Ollama model name for the judge. Defaults to
            ``settings.ollama_model``.
        embedding_model: Local sentence-transformers model for embedding-based
            metrics.

    Returns:
        ``{"overall": {metric: mean|NaN}, "per_row": [{metric: val|NaN}, ...],
        "valid_counts": {metric: int}, "n_rows": int, "error": str|None}``.
        ``overall`` is the nan-aware mean over rows; ``valid_counts`` reports how
        many rows produced a finite score for each metric.
    """
    names = list(metric_names) if metric_names is not None else list(DEFAULT_METRICS)
    n_rows = len(questions)

    if not (len(answers) == len(contexts) == len(ground_truths) == n_rows):
        raise ValueError("questions, answers, contexts, ground_truths must have equal length")
    if n_rows == 0:
        return {"overall": {m: float("nan") for m in names}, "per_row": [],
                "valid_counts": {m: 0 for m in names}, "n_rows": 0, "error": None}

    # ── lazy imports — degrade to all-NaN if unavailable ───────────────────────
    try:
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.error("ragas_eval: ragas/datasets import failed (%s) — all metrics NaN", exc)
        return _nan_result(names, n_rows, f"import_error: {exc}")

    try:
        metrics = _resolve_metrics(names)
        if not metrics:
            return _nan_result(names, n_rows, "no_resolvable_metrics")
        llm, emb = _build_judge(judge_model, embedding_model)
    except Exception as exc:  # noqa: BLE001
        logger.error("ragas_eval: judge/metric setup failed (%s) — all metrics NaN", exc)
        return _nan_result(names, n_rows, f"setup_error: {exc}")

    # RAGAS 0.1.x dataset schema: question / answer / contexts / ground_truth.
    dataset = Dataset.from_dict(
        {
            "question": list(questions),
            "answer": list(answers),
            "contexts": [list(c) for c in contexts],
            "ground_truth": list(ground_truths),
        }
    )

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=emb,
            raise_exceptions=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ragas_eval: evaluate() failed (%s) — all metrics NaN", exc)
        return _nan_result(names, n_rows, f"evaluate_error: {exc}")

    return _summarise(result, names, n_rows)


def _summarise(result: Any, names: list[str], n_rows: int) -> dict[str, Any]:
    """Convert a RAGAS Result into the per-row + overall summary dict."""
    try:
        df = result.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.error("ragas_eval: could not read RAGAS result (%s) — all metrics NaN", exc)
        return _nan_result(names, n_rows, f"result_parse_error: {exc}")

    per_row: list[dict[str, float]] = []
    for _, row in df.iterrows():
        per_row.append(
            {m: (float(row[m]) if m in df.columns else float("nan")) for m in names}
        )

    overall: dict[str, float] = {}
    valid_counts: dict[str, int] = {}
    for m in names:
        vals = [r[m] for r in per_row if isinstance(r[m], float) and not math.isnan(r[m])]
        valid_counts[m] = len(vals)
        overall[m] = float(sum(vals) / len(vals)) if vals else float("nan")
        if not vals:
            logger.warning("ragas_eval: metric %r produced no finite score across %d rows", m, n_rows)

    return {
        "overall": overall,
        "per_row": per_row,
        "valid_counts": valid_counts,
        "n_rows": n_rows,
        "error": None,
    }


__all__ = ["compute_ragas", "DEFAULT_METRICS"]
