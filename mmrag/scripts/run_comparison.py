"""Run the full ablation matrix and produce results/metrics.json + figures.

Executes every config in the ablation matrix over the eval set, writes the
aggregated ``results/metrics.json`` ( ``{config_name: result_doc}`` ), renders
comparison figures into ``results/figures/``, and prints a compact comparison
table (including the marginal delta of each single-axis variant vs the baseline).

Usage
-----
    python scripts/run_comparison.py                     # 5-config default matrix
    python scripts/run_comparison.py --full              # all 8 cells
    python scripts/run_comparison.py --no-ragas          # retrieval+system only (fast)
    python scripts/run_comparison.py --embedding-model BAAI/bge-small-en-v1.5

Prereqs: ``data/eval/eval_set.jsonl`` (scripts/build_eval_set.py) and, for the
generation/RAGAS metrics, a running Ollama. Metrics are always real; anything
unmeasurable is reported as NaN.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.config import load_pipeline_config
from src.evaluation.ablation import BaseSettings, ablation_matrix
from src.evaluation.figures import generate_all_figures
from src.evaluation.retrieval_metrics import compute_retrieval_metrics, dedup_preserve_order
from src.evaluation.runner import load_eval_set, run_config_eval
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.optimized import OptimizedPipeline

app = typer.Typer(add_completion=False, help="Run the RAG ablation matrix and build the report.")
logger = logging.getLogger(__name__)

# Columns shown in the stdout summary table.
_TABLE_METRICS = [
    ("retrieval", "hit@5"),
    ("retrieval", "recall@5"),
    ("retrieval", "ndcg@10"),
    ("retrieval", "mrr"),
    ("ragas", "faithfulness"),
    ("ragas", "answer_correctness"),
]


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  NaN"
    return f"{float(v):.3f}"


def _get(result: dict[str, Any], block: str, metric: str) -> float:
    return result.get(block, {}).get("overall", {}).get(metric, float("nan"))


def _print_table(metrics: dict[str, dict[str, Any]]) -> None:
    headers = ["config"] + [m for _, m in _TABLE_METRICS] + ["lat_p50", "lat_p95", "tok_mean"]
    widths = [max(len(h), 14) for h in headers]
    typer.echo("\n" + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    typer.echo("  ".join("-" * w for w in widths))
    for name, res in metrics.items():
        row = [name.ljust(widths[0])]
        for (block, metric), w in zip(_TABLE_METRICS, widths[1:]):
            row.append(_fmt(_get(res, block, metric)).ljust(w))
        lat = res.get("system", {}).get("overall", {}).get("latency_ms", {})
        tok = res.get("system", {}).get("overall", {}).get("tokens", {}).get("total_tokens", {})
        row.append(_fmt(lat.get("p50")).ljust(widths[-3]))
        row.append(_fmt(lat.get("p95")).ljust(widths[-2]))
        row.append(_fmt(tok.get("mean")).ljust(widths[-1]))
        typer.echo("  ".join(row))


def _print_deltas(metrics: dict[str, dict[str, Any]]) -> None:
    if "baseline" not in metrics:
        return
    base = metrics["baseline"]
    typer.echo("\nMarginal contribution vs baseline (recall@5 / faithfulness):")
    for name, res in metrics.items():
        if name == "baseline":
            continue
        d_recall = _get(res, "retrieval", "recall@5") - _get(base, "retrieval", "recall@5")
        d_faith = _get(res, "ragas", "faithfulness") - _get(base, "ragas", "faithfulness")
        typer.echo(f"  {name:<16} Δrecall@5={d_recall:+.3f}   Δfaithfulness={d_faith:+.3f}")


# ── retrieval-only helpers (no Ollama needed) ─────────────────────────────────


def _retrieve_baseline_only(pipeline: BaselinePipeline, question: str) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    query_vec = pipeline.embedder.embed_query_numpy(question)
    top_k = pipeline._retrieval["top_k"]
    threshold = pipeline._retrieval.get("score_threshold", 0.0)
    hits = pipeline.vector_index.similarity_search(
        query_embedding=query_vec,
        modality="text",
        k=top_k,
        score_threshold=threshold if threshold > 0.0 else None,
    )
    return [doc.metadata for doc, _ in hits], round((time.perf_counter() - t0) * 1_000, 1)


def _retrieve_optimized_only(pipeline: OptimizedPipeline, question: str) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    candidates = pipeline._retrieve(question)
    if pipeline._rerank_cfg.get("enabled", True):
        top_k_final = pipeline._rerank_cfg.get("top_k_final", 5)
        final_hits = pipeline.reranker.rerank(question, candidates, k=top_k_final)
    else:
        top_k_final = pipeline._rerank_cfg.get("top_k_final", 5)
        final_hits = candidates[:top_k_final]
    return [doc.metadata for doc, _ in final_hits], round((time.perf_counter() - t0) * 1_000, 1)


def _ensure_index_fast(pipeline: Any, data_dir: str, label: str, force: bool) -> None:
    try:
        n = pipeline.vector_index.count("text")
    except Exception:
        n = 0
    bm25_missing = isinstance(pipeline, OptimizedPipeline) and pipeline._bm25 is None
    if force or n == 0 or bm25_missing:
        reason = "forced" if force else ("empty" if n == 0 else "BM25 missing")
        typer.echo(f"  [{label}] Building index ({reason}) ...")
        pipeline.index(data_dir)
        n = pipeline.vector_index.count("text")
        if isinstance(pipeline, OptimizedPipeline):
            bm25_n = pipeline._bm25.corpus_size if pipeline._bm25 else 0
            typer.echo(f"  [{label}] {n} chunks in Chroma, {bm25_n} docs in BM25")
        else:
            typer.echo(f"  [{label}] {n} chunks in Chroma")
    else:
        typer.echo(f"  [{label}] Reusing existing index ({n} chunks)")


def _eval_retrieval_only(pipeline: Any, retrieve_fn: Any, records: list[dict], label: str) -> dict:
    retrieved_all: list[list[str]] = []
    relevant_all: list[list[str]] = []
    latencies: list[float] = []

    for i, rec in enumerate(records, 1):
        try:
            sources, lat = retrieve_fn(pipeline, rec["question"])
            retrieved = dedup_preserve_order(str(s.get("doc_id", "")) for s in sources)
            latencies.append(lat)
        except Exception as exc:
            logger.warning("[%s] query %d failed: %s", label, i, exc)
            retrieved = []
            latencies.append(float("nan"))

        retrieved_all.append(retrieved)
        relevant_all.append(list(rec.get("relevant_doc_ids", [])))

        if i % 5 == 0 or i == len(records):
            typer.echo(f"  [{label}] {i}/{len(records)} done")

    metrics = compute_retrieval_metrics(retrieved_all, relevant_all, k_values=(1, 3, 5, 10))
    finite = sorted(v for v in latencies if not math.isnan(v))
    if finite:
        metrics["latency_p50"] = float(finite[len(finite) // 2])
        metrics["latency_p95"] = float(finite[int(len(finite) * 0.95)])
        metrics["latency_mean"] = float(sum(finite) / len(finite))
    else:
        metrics["latency_p50"] = metrics["latency_p95"] = metrics["latency_mean"] = float("nan")
    return metrics


def _md_val(v: float, is_lat: bool = False) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:.0f}" if is_lat else f"{v:.3f}"


def _md_delta(b: float, o: float, is_lat: bool = False) -> str:
    if math.isnan(b) or math.isnan(o):
        return "—"
    d = o - b
    return f"{d:+.0f}" if is_lat else f"{d:+.3f}"


def _run_pipeline_eval(pipeline: Any, records: list[dict], label: str) -> tuple[dict, list[dict]]:
    """Run full pipeline.query() for each question — returns retrieval metrics + raw rows for RAGAS."""
    retrieved_all: list[list[str]] = []
    relevant_all: list[list[str]] = []
    latencies: list[float] = []
    raw_rows: list[dict] = []

    for i, rec in enumerate(records, 1):
        try:
            result = pipeline.query(rec["question"])
            sources  = result.get("sources", [])
            retrieved = dedup_preserve_order(str(s.get("doc_id", "")) for s in sources)
            lat = float(result.get("latency_ms", float("nan")))
            answer   = result.get("answer", "")
            contexts = result.get("retrieved_contexts", [])
            generation_ok = not answer.startswith("Error:")
        except Exception as exc:
            logger.warning("[%s] query %d failed: %s", label, i, exc)
            retrieved = []
            lat = float("nan")
            answer = f"Error: {exc}"
            contexts = []
            generation_ok = False

        retrieved_all.append(retrieved)
        relevant_all.append(list(rec.get("relevant_doc_ids", [])))
        latencies.append(lat)
        raw_rows.append({
            "question": rec["question"],
            "answer": answer,
            "retrieved_contexts": contexts,
            "ground_truth_answer": rec.get("ground_truth_answer", ""),
            "generation_ok": generation_ok,
        })

        if i % 5 == 0 or i == len(records):
            typer.echo(f"  [{label}] {i}/{len(records)} done")

    metrics = compute_retrieval_metrics(retrieved_all, relevant_all, k_values=(1, 3, 5, 10))
    finite = sorted(v for v in latencies if not math.isnan(v))
    if finite:
        metrics["latency_p50"] = float(finite[len(finite) // 2])
        metrics["latency_p95"] = float(finite[int(len(finite) * 0.95)])
        metrics["latency_mean"] = float(sum(finite) / len(finite))
    else:
        metrics["latency_p50"] = metrics["latency_p95"] = metrics["latency_mean"] = float("nan")
    return metrics, raw_rows


def _compute_ragas_simple(raw_rows: list[dict], embedding_model: str) -> dict[str, float]:
    """Aggregate RAGAS metrics over successfully-generated rows (nan-aware)."""
    from src.evaluation.ragas_eval import DEFAULT_METRICS, compute_ragas  # noqa: PLC0415

    ok_rows = [r for r in raw_rows if r.get("generation_ok")]
    if not ok_rows:
        logger.warning("RAGAS: aucune réponse générée avec succès — métriques NaN")
        return {m: float("nan") for m in DEFAULT_METRICS}
    try:
        res = compute_ragas(
            questions=[r["question"] for r in ok_rows],
            answers=[r["answer"] for r in ok_rows],
            contexts=[r["retrieved_contexts"] for r in ok_rows],
            ground_truths=[r["ground_truth_answer"] for r in ok_rows],
            embedding_model=embedding_model,
        )
        return res.get("overall", {m: float("nan") for m in DEFAULT_METRICS})
    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s — all metrics NaN", exc)
        return {m: float("nan") for m in DEFAULT_METRICS}


def _build_comparison_md(
    b: dict,
    o: dict,
    n: int,
    ragas_b: Optional[dict] = None,
    ragas_o: Optional[dict] = None,
) -> str:
    retrieval_rows = [
        ("Hit@1",            "hit@1",       False),
        ("Hit@3",            "hit@3",       False),
        ("Hit@5",            "hit@5",       False),
        ("Recall@5",         "recall@5",    False),
        ("Recall@10",        "recall@10",   False),
        ("nDCG@10",          "ndcg@10",     False),
        ("MRR",              "mrr",         False),
        ("Latency p50 (ms)", "latency_p50", True),
        ("Latency p95 (ms)", "latency_p95", True),
    ]
    ragas_rows = [
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevancy",   "answer_relevancy"),
        ("Context Recall",     "context_recall"),
        ("Answer Correctness", "answer_correctness"),
    ]
    section = "Retrieval + Quality (RAGAS)" if ragas_b is not None else "Retrieval"
    lines = [
        f"## Pipeline Comparison — {section} ({n} questions)\n",
        "| Metric | Baseline | Optimized | Δ |",
        "|--------|:--------:|:---------:|:---:|",
    ]
    for lbl, key, is_lat in retrieval_rows:
        bv = b.get(key, float("nan"))
        ov = o.get(key, float("nan"))
        lines.append(f"| {lbl} | {_md_val(bv, is_lat)} | {_md_val(ov, is_lat)} | {_md_delta(bv, ov, is_lat)} |")

    if ragas_b is not None and ragas_o is not None:
        lines.append("| **— RAGAS quality —** | | | |")
        for lbl, key in ragas_rows:
            bv = ragas_b.get(key, float("nan"))
            ov = ragas_o.get(key, float("nan"))
            lines.append(f"| {lbl} | {_md_val(bv)} | {_md_val(ov)} | {_md_delta(bv, ov)} |")

    lines += [
        "",
        "**Baseline** : recursive chunking · dense-only retrieval · no reranking  ",
        "**Optimized**: semantic chunking · BM25 + dense hybrid (RRF) · CrossEncoder reranking  ",
        "*Embedding models — baseline: `all-MiniLM-L6-v2` · optimized: `BAAI/bge-small-en-v1.5`*",
    ]
    if ragas_b is not None:
        lines.append("*RAGAS judge: local Ollama · embeddings: `all-MiniLM-L6-v2`*")
    return "\n".join(lines)


def _run_simple_compare(
    eval_path: Path,
    data_dir: str,
    results_dir: Path,
    force: bool,
    n: Optional[int],
    do_ragas: bool = True,
) -> None:
    """Baseline vs Optimized — 2 configs only (no full ablation matrix).

    With do_ragas=True (default): also runs LLM generation and RAGAS quality metrics
    (faithfulness, answer_relevancy, context_recall, answer_correctness).
    Needs Ollama; if Ollama is down RAGAS scores are NaN but retrieval metrics still work.

    With do_ragas=False (--no-ragas): retrieval-only, no LLM, no Ollama needed.
    """
    records = load_eval_set(eval_path)
    records = [r for r in records if r.get("relevant_doc_ids")]
    if n:
        records = records[:n]
    typer.echo(f"Loaded {len(records)} eval questions")

    baseline  = BaselinePipeline(load_pipeline_config("configs/baseline.yaml"))
    optimized = OptimizedPipeline(load_pipeline_config("configs/optimized.yaml"))

    # Pre-warm ALL models for BOTH pipelines before any ChromaDB access.
    # On Windows, loading a sentence-transformers model after ChromaDB has
    # initialised its native SQLite/BLAS libs causes a segfault; loading
    # all models first avoids the conflict entirely.
    typer.echo("  Pre-loading embedding models ...")
    _ = baseline.embedder
    chunker_b = baseline.chunker
    if hasattr(chunker_b, "_get_embedder"):
        chunker_b._get_embedder()
    _ = optimized.embedder
    chunker_o = optimized.chunker
    if hasattr(chunker_o, "_get_embedder"):
        chunker_o._get_embedder()
    typer.echo("  Models ready.")

    typer.echo("\n-- Indexes " + "-" * 50)
    _ensure_index_fast(baseline,  data_dir, "baseline",  force)
    _ensure_index_fast(optimized, data_dir, "optimized", force)

    if do_ragas:
        embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        typer.echo("\n-- Baseline (retrieval + generation) " + "-" * 24)
        base_m, base_rows = _run_pipeline_eval(baseline, records, "baseline")
        typer.echo("\n-- Optimized (retrieval + generation) " + "-" * 23)
        opt_m, opt_rows   = _run_pipeline_eval(optimized, records, "optimized")

        typer.echo("\n-- RAGAS baseline " + "-" * 42)
        ragas_b = _compute_ragas_simple(base_rows, embed_model)
        typer.echo("\n-- RAGAS optimized " + "-" * 41)
        ragas_o = _compute_ragas_simple(opt_rows, embed_model)
    else:
        typer.echo("\n-- Baseline retrieval (no LLM) " + "-" * 30)
        base_m = _eval_retrieval_only(baseline, _retrieve_baseline_only, records, "baseline")
        typer.echo("\n-- Optimized retrieval (no LLM) " + "-" * 29)
        opt_m  = _eval_retrieval_only(optimized, _retrieve_optimized_only, records, "optimized")
        ragas_b = ragas_o = None
        base_rows = opt_rows = []

    results_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "n_questions": len(records),
        "modality":    "text",
        "do_ragas":    do_ragas,
        "baseline":    {"retrieval": base_m, "ragas": ragas_b},
        "optimized":   {"retrieval": opt_m,  "ragas": ragas_o},
    }
    json_path = results_dir / "quick_compare.json"
    json_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _build_comparison_md(base_m, opt_m, len(records), ragas_b, ragas_o)
    md_path = results_dir / "comparison.md"
    md_path.write_text(md + "\n", encoding="utf-8")

    # stdout summary
    w = 20
    hdr = f"{'Metric':<{w}}{'Baseline':>10}{'Optimized':>12}{'Delta':>10}"
    typer.echo("\n-- Retrieval " + "-" * (len(hdr) - 13))
    typer.echo(hdr)
    typer.echo("-" * len(hdr))
    for key, lbl in [("hit@5", "Hit@5"), ("recall@5", "Recall@5"), ("ndcg@10", "nDCG@10"), ("mrr", "MRR")]:
        bv = base_m.get(key, float("nan"))
        ov = opt_m.get(key, float("nan"))
        typer.echo(f"{lbl:<{w}}{_md_val(bv):>10}{_md_val(ov):>12}{_md_delta(bv, ov):>10}")

    if ragas_b is not None and ragas_o is not None:
        typer.echo("\n-- RAGAS " + "-" * (len(hdr) - 9))
        typer.echo(hdr)
        typer.echo("-" * len(hdr))
        for key, lbl in [
            ("faithfulness",       "Faithfulness"),
            ("answer_relevancy",   "Answer Relevancy"),
            ("context_recall",     "Context Recall"),
            ("answer_correctness", "Answer Correctness"),
        ]:
            bv = ragas_b.get(key, float("nan"))
            ov = ragas_o.get(key, float("nan"))
            typer.echo(f"{lbl:<{w}}{_md_val(bv):>10}{_md_val(ov):>12}{_md_delta(bv, ov):>10}")

    typer.echo(f"\nOK  {json_path}")
    typer.echo(f"OK  {md_path}  <- copier dans le README")


@app.command()
def main(
    eval_path: Path = typer.Option(Path("data/eval/eval_set.jsonl"), help="Eval set JSONL."),
    results_dir: Path = typer.Option(Path("results"), help="Output directory."),
    simple: bool = typer.Option(
        False, "--simple",
        help="Mode simplifié : baseline vs optimized seulement (pas la matrice complète). "
             "Construit les index auto, évalue retrieval + RAGAS, sauvegarde comparison.md.",
    ),
    data_dir: str = typer.Option("data", "--data-dir", help="Répertoire de données (pour --simple)."),
    n: Optional[int] = typer.Option(None, "--n", help="Limite le nombre de questions (pour --simple)."),
    full: bool = typer.Option(False, "--full", help="Run all 8 cells (default: 5-config matrix)."),
    no_ragas: bool = typer.Option(False, "--no-ragas", help="Skip RAGAS quality metrics."),
    force_index: bool = typer.Option(False, "--force-index", help="Rebuild indexes."),
    embedding_model: str = typer.Option(
        BaseSettings.embedding_model, help="Embedding model held constant across the matrix."
    ),
    judge_model: str = typer.Option(None, help="Ollama generation/judge model (default from .env)."),
    no_figures: bool = typer.Option(False, "--no-figures", help="Skip figure generation."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run the matrix, write metrics.json + figures, print the comparison table.

    Mode rapide (recommandé pour le README) :
        python scripts/run_comparison.py --simple            # retrieval + RAGAS
        python scripts/run_comparison.py --simple --no-ragas # retrieval only, sans Ollama

    Mode complet (matrice d'ablation 5 configs) :
        python scripts/run_comparison.py                     # avec RAGAS
        python scripts/run_comparison.py --full              # 8 configs
        python scripts/run_comparison.py --no-ragas          # sans RAGAS
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if simple:
        _run_simple_compare(
            eval_path=eval_path,
            data_dir=data_dir,
            results_dir=results_dir,
            force=force_index,
            n=n,
            do_ragas=not no_ragas,
        )
        return

    records = load_eval_set(eval_path)
    typer.echo(f"Loaded {len(records)} eval questions from {eval_path}")

    overrides: dict[str, Any] = {"embedding_model": embedding_model}
    if judge_model:
        overrides["gen_model"] = judge_model
    base = BaseSettings(**overrides)

    configs = ablation_matrix(full=full)
    typer.echo(f"Running {len(configs)} configs: {[c.name for c in configs]}")

    metrics: dict[str, dict[str, Any]] = {}
    for rc in configs:
        typer.echo(f"\n== {rc.name}  [{rc.label}] ==")
        try:
            result = run_config_eval(
                rc, records, base=base,
                do_ragas=not no_ragas,
                results_dir=results_dir,
                force_index=force_index,
                save=True,
            )
            metrics[rc.name] = result
        except Exception as exc:  # noqa: BLE001 - record the failure, keep going
            logger.exception("config %s failed", rc.name)
            metrics[rc.name] = {"config": {"name": rc.name, "label": rc.label, **rc.axes},
                                "error": str(exc)}

    # ── aggregate report ───────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"\nOK  metrics -> {metrics_path}")

    if not no_figures:
        figs = generate_all_figures(metrics, results_dir / "figures")
        typer.echo(f"OK  {len(figs)} figures -> {results_dir / 'figures'}")

    _print_table(metrics)
    _print_deltas(metrics)


if __name__ == "__main__":
    app()
