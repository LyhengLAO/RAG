"""Evaluation runner: execute one ablation config over the full eval set.

For a given :class:`~src.evaluation.ablation.RunConfig` the runner:

1. Builds the configurable :class:`~src.pipelines.optimized.OptimizedPipeline`.
2. Ensures the index for that chunking strategy exists (built once, reused by
   every config sharing the strategy).
3. Runs every eval question, capturing per-query latency, provider-reported
   token usage, retrieved ``doc_id`` ranking, and the answer/contexts.
4. Computes — **overall and per modality** —
     * retrieval metrics (Hit@k, MRR, Recall@k, nDCG@k) from ``relevant_doc_ids``,
     * RAGAS quality metrics (local Ollama judge),
     * system metrics (latency p50/p95, token counts).
5. Writes ``results/runs/{name}.json`` (aggregated) and ``…_raw.jsonl`` (per-row
   answers + contexts for audit).

Honesty contract: every number is measured. When a measurement is impossible
(LLM down → no answer / no tokens; a RAGAS metric raises) the value is ``NaN``
and the reason is logged — nothing is fabricated.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.ablation import BaseSettings, RunConfig, build_pipeline_config
from src.evaluation.retrieval_metrics import compute_retrieval_metrics, dedup_preserve_order
from src.evaluation.system_metrics import aggregate_system_metrics, token_delta

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


# ── eval-set IO ─────────────────────────────────────────────────────────────────


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """Load the JSONL eval set produced by ``scripts/build_eval_set.py``.

    Each record must contain ``question``, ``ground_truth_answer``,
    ``relevant_doc_ids`` (list) and ``modality``.

    Args:
        path: Path to ``data/eval/eval_set.jsonl``.

    Returns:
        List of eval records.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid records are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Eval set not found at {path}. Run scripts/build_eval_set.py first."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("eval set line %d skipped — %s", lineno, exc)
                continue
            if not rec.get("question") or not rec.get("relevant_doc_ids"):
                logger.warning("eval set line %d missing question/relevant_doc_ids — skipped", lineno)
                continue
            records.append(rec)

    if not records:
        raise ValueError(f"No valid eval records in {path}")
    return records


# ── indexing ────────────────────────────────────────────────────────────────────


def _ensure_index(pipeline: Any, data_dir: str, *, force: bool) -> None:
    """Build the index for the pipeline's chunking strategy if needed."""
    try:
        n = pipeline.vector_index.count("text")
    except Exception:  # noqa: BLE001 - collection may not exist yet
        n = 0
    if force or n == 0:
        logger.info("Index for collection %r empty/forced — building …",
                    pipeline.vector_index.collection_prefix)
        pipeline.index(data_dir)
    else:
        logger.info("Reusing existing index (%d chunks) for collection %r",
                    n, pipeline.vector_index.collection_prefix)


# ── aggregation helpers ─────────────────────────────────────────────────────────


def _group_by_modality(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Return {modality: [row indices]} preserving sorted modality order."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(str(r.get("modality", "unknown")), []).append(i)
    return dict(sorted(groups.items()))


def _aggregate_retrieval(
    per_query: list[dict[str, Any]],
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Retrieval metrics overall and per modality."""

    def _compute(rows: list[dict[str, Any]]) -> dict[str, float]:
        retrieved = [r["retrieved_doc_ids"] for r in rows]
        relevant = [r["relevant_doc_ids"] for r in rows]
        return compute_retrieval_metrics(retrieved, relevant, k_values=k_values)

    overall = _compute(per_query)
    groups = _group_by_modality(per_query)
    per_modality = {m: _compute([per_query[i] for i in idx]) for m, idx in groups.items()}
    return {"overall": overall, "per_modality": per_modality}


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _aggregate_ragas(
    per_row_scores: list[dict[str, float]],
    per_query: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, Any]:
    """Aggregate RAGAS per-row scores overall and per modality (nan-aware)."""

    def _block(rows: list[int]) -> dict[str, float]:
        return {
            m: _nanmean([per_row_scores[i][m] for i in rows])
            for m in metric_names
        }

    overall = _block(list(range(len(per_row_scores))))
    groups = _group_by_modality(per_query)
    per_modality = {m: _block(idx) for m, idx in groups.items()}
    return {"overall": overall, "per_modality": per_modality}


# ── core ─────────────────────────────────────────────────────────────────────────


def run_config_eval(
    rc: RunConfig,
    eval_records: list[dict[str, Any]],
    base: BaseSettings | None = None,
    *,
    do_ragas: bool = True,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    results_dir: str | Path = "results",
    force_index: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """Evaluate a single ablation config over the whole eval set.

    Args:
        rc: The ablation cell to evaluate.
        eval_records: Output of :func:`load_eval_set`.
        base: Held-constant settings (defaults to :class:`BaseSettings`).
        do_ragas: Whether to run the (slow) RAGAS quality metrics.
        k_values: Cut-offs for retrieval ``@k`` metrics.
        results_dir: Where to write ``runs/{name}.json`` and the raw JSONL.
        force_index: Rebuild the index even if the collection is non-empty.
        save: Persist result files to disk.

    Returns:
        The aggregated result dict for this config.
    """
    base = base or BaseSettings()
    from src.pipelines.optimized import OptimizedPipeline  # noqa: PLC0415 - lazy heavy import

    cfg = build_pipeline_config(rc, base)
    pipeline = OptimizedPipeline(cfg)
    _ensure_index(pipeline, base.data_dir, force=force_index)

    # Ensure the LLM client exists so we can snapshot its cumulative token usage.
    llm = pipeline.llm

    per_query: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    logger.info("[%s] evaluating %d questions …", rc.name, len(eval_records))
    for i, rec in enumerate(eval_records, 1):
        question = rec["question"]
        modality = str(rec.get("modality", "unknown"))
        relevant = list(rec.get("relevant_doc_ids", []))

        usage_before = dict(llm.total_usage)
        try:
            result = pipeline.query(question)
        except Exception as exc:  # noqa: BLE001 - never abort the whole run
            logger.error("[%s] query %d failed: %s", rc.name, i, exc)
            result = {"answer": f"Error: {exc}", "retrieved_contexts": [], "sources": [],
                      "latency_ms": float("nan")}
        usage_after = dict(llm.total_usage)

        answer = result.get("answer", "")
        generation_ok = not answer.startswith("Error:")
        retrieved_doc_ids = dedup_preserve_order(
            str(s.get("doc_id", "")) for s in result.get("sources", [])
        )
        tokens = token_delta(usage_before, usage_after)
        latency = result.get("latency_ms", float("nan"))
        if not generation_ok:
            # No trustworthy answer ⇒ token attribution is meaningless.
            tokens = {k: float("nan") for k in tokens}

        per_query.append({
            "id": rec.get("id", f"q{i:04d}"),
            "modality": modality,
            "latency_ms": float(latency) if latency is not None else float("nan"),
            "retrieved_doc_ids": retrieved_doc_ids,
            "relevant_doc_ids": relevant,
            "generation_ok": generation_ok,
            **tokens,
        })
        raw_rows.append({
            "id": rec.get("id", f"q{i:04d}"),
            "modality": modality,
            "question": question,
            "answer": answer,
            "ground_truth_answer": rec.get("ground_truth_answer", ""),
            "retrieved_contexts": result.get("retrieved_contexts", []),
            "retrieved_doc_ids": retrieved_doc_ids,
            "relevant_doc_ids": relevant,
            "generation_ok": generation_ok,
        })
        if i % 10 == 0:
            logger.info("[%s] %d/%d done", rc.name, i, len(eval_records))

    # ── retrieval metrics ──────────────────────────────────────────────────────
    retrieval = _aggregate_retrieval(per_query, k_values)

    # ── system metrics ─────────────────────────────────────────────────────────
    system = aggregate_system_metrics(per_query)

    # ── RAGAS quality metrics ──────────────────────────────────────────────────
    n_ok = sum(1 for r in per_query if r["generation_ok"])
    n_failed = len(per_query) - n_ok
    ragas_block: dict[str, Any] = {"overall": {}, "per_modality": {}, "valid_counts": {},
                                   "error": None, "skipped": not do_ragas}
    if do_ragas:
        ragas_block = _run_ragas(eval_records, raw_rows, per_query, base)

    result_doc = {
        "config": {"name": rc.name, "label": rc.label, **rc.axes},
        "base_settings": asdict(base),
        "n_eval": len(eval_records),
        "k_values": list(k_values),
        "generation": {"n_ok": n_ok, "n_failed": n_failed},
        "retrieval": retrieval,
        "ragas": ragas_block,
        "system": system,
        "per_query": per_query,
    }

    if save:
        _save(result_doc, raw_rows, rc.name, results_dir)
    return result_doc


def _run_ragas(
    eval_records: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    per_query: list[dict[str, Any]],
    base: BaseSettings,
) -> dict[str, Any]:
    """Run RAGAS over successfully-generated rows and aggregate by modality."""
    from src.evaluation.ragas_eval import DEFAULT_METRICS, compute_ragas  # noqa: PLC0415

    metric_names = list(DEFAULT_METRICS)
    nan = float("nan")
    # Full per-row scores, NaN for rows where generation failed (not judged).
    full_scores: list[dict[str, float]] = [{m: nan for m in metric_names} for _ in raw_rows]

    ok_idx = [i for i, r in enumerate(raw_rows) if r["generation_ok"]]
    if not ok_idx:
        logger.warning("RAGAS: no successfully-generated answers — all metrics NaN")
        return {
            "overall": {m: nan for m in metric_names},
            "per_modality": {},
            "valid_counts": {m: 0 for m in metric_names},
            "error": "no_successful_generations",
            "skipped": False,
        }

    res = compute_ragas(
        questions=[raw_rows[i]["question"] for i in ok_idx],
        answers=[raw_rows[i]["answer"] for i in ok_idx],
        contexts=[raw_rows[i]["retrieved_contexts"] for i in ok_idx],
        ground_truths=[raw_rows[i]["ground_truth_answer"] for i in ok_idx],
        embedding_model=base.embedding_model,
    )
    for local_i, global_i in enumerate(ok_idx):
        full_scores[global_i] = res["per_row"][local_i]

    agg = _aggregate_ragas(full_scores, per_query, metric_names)
    return {
        "overall": agg["overall"],
        "per_modality": agg["per_modality"],
        "valid_counts": res.get("valid_counts", {}),
        "error": res.get("error"),
        "skipped": False,
    }


# ── persistence ─────────────────────────────────────────────────────────────────


def _save(result_doc: dict[str, Any], raw_rows: list[dict[str, Any]],
          name: str, results_dir: str | Path) -> None:
    runs_dir = Path(results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    out = runs_dir / f"{name}.json"
    out.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    raw = runs_dir / f"{name}_raw.jsonl"
    with raw.open("w", encoding="utf-8") as fh:
        for row in raw_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("[%s] saved metrics → %s  (+ raw → %s)", name, out, raw)


# ── CLI: run a single config and write results/metrics.json ─────────────────────


def main() -> None:
    """``python -m src.evaluation.runner --name optimized`` (single config)."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run one ablation config over the eval set.")
    parser.add_argument("--name", default="optimized",
                        help="Config name from the default ablation matrix, or one of baseline/optimized.")
    parser.add_argument("--eval-path", default="data/eval/eval_set.jsonl")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS quality metrics.")
    parser.add_argument("--force-index", action="store_true", help="Rebuild the index.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")

    from src.evaluation.ablation import ablation_matrix  # noqa: PLC0415

    matrix = {rc.name: rc for rc in ablation_matrix(full=True)}
    matrix.update({rc.name: rc for rc in ablation_matrix(full=False)})
    if args.name not in matrix:
        raise SystemExit(f"Unknown config {args.name!r}. Choices: {sorted(matrix)}")

    records = load_eval_set(args.eval_path)
    result = run_config_eval(
        matrix[args.name], records,
        do_ragas=not args.no_ragas,
        results_dir=args.results_dir,
        force_index=args.force_index,
    )
    # Single-config convenience: also write results/metrics.json.
    metrics_path = Path(args.results_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({result["config"]["name"]: result}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
