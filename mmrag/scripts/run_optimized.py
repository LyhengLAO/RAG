"""CLI for the optimized RAG pipeline.

Usage
-----
Build the vector + BM25 index from the preprocessed manifest::

    python scripts/run_optimized.py index

Ask a single question (index must already be built)::

    python scripts/run_optimized.py ask "Who wrote Hamlet?"

Run a batch of questions from a JSONL file::

    python scripts/run_optimized.py batch questions.jsonl --output results_optimized.jsonl

Compare optimized vs baseline on the same question::

    python scripts/run_optimized.py ask "..." --compare

Ablation: run with recursive chunking instead of semantic::

    python scripts/run_optimized.py index --strategy recursive
    python scripts/run_optimized.py ask "..."

Ablation: disable reranking at query time::

    python scripts/run_optimized.py ask "..." --no-rerank

Enable query transformation::

    python scripts/run_optimized.py ask "..." --query-transform multi_query
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="run-optimized",
    help="Optimized RAG pipeline: semantic chunking, hybrid retrieval, reranking.",
    add_completion=False,
)

# ── shared options ────────────────────────────────────────────────────────────

_CONFIG_OPT = typer.Option(
    Path("configs/optimized.yaml"),
    "--config",
    "-c",
    help="Path to pipeline YAML config file.",
    show_default=True,
)

_LOG_LEVEL_OPT = typer.Option(
    "INFO",
    "--log-level",
    "-l",
    help="Logging level.",
    show_default=True,
)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _patch_config(cfg: dict, **overrides: object) -> dict:
    """Return a shallow-patched copy of *cfg* for ablation runs."""
    import copy
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


# ── index command ─────────────────────────────────────────────────────────────


@app.command()
def index(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        "-d",
        help="Base data directory.",
        show_default=True,
    ),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Chunking strategy override: 'semantic' | 'recursive' (ablation).",
    ),
    config_path: Path = _CONFIG_OPT,
    log_level: str = _LOG_LEVEL_OPT,
) -> None:
    """Build the dense (ChromaDB) and sparse (BM25) indexes from the manifest.

    Safe to re-run: Chroma upserts are idempotent; BM25 is rebuilt and saved.
    Run ``scripts/build_dataset.py`` first to create the manifest.
    """
    _setup_logging(log_level)
    logger = logging.getLogger("run_optimized.index")

    from src.config import load_pipeline_config
    from src.pipelines.optimized import OptimizedPipeline

    cfg = load_pipeline_config(config_path)
    if strategy:
        cfg = _patch_config(cfg, **{"chunking.strategy": strategy})
        logger.info("Chunking strategy overridden to: %s", strategy)

    pipeline = OptimizedPipeline(cfg)
    logger.info("Starting index build from %s", data_dir / "processed" / "manifest.jsonl")

    try:
        pipeline.index(data_dir)
        n = pipeline.vector_index.count("text")
        typer.echo(f"Index built. Text collection: {n} chunks. BM25: {pipeline._bm25.corpus_size} docs.")
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        logger.exception("Index build failed")
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


# ── ask command ───────────────────────────────────────────────────────────────


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question."),
    config_path: Path = _CONFIG_OPT,
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Print full JSON result."
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Disable cross-encoder reranking (ablation)."
    ),
    query_transform: Optional[str] = typer.Option(
        None,
        "--query-transform",
        "-qt",
        help="Enable query transformation: 'multi_query' | 'hyde'.",
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Also run the baseline pipeline and show both answers side-by-side.",
    ),
    log_level: str = _LOG_LEVEL_OPT,
) -> None:
    """Answer a question with the optimized pipeline.

    The index must already be built (run the ``index`` command first).
    Ollama must be running: ``ollama serve``.
    """
    _setup_logging(log_level)

    from src.config import load_pipeline_config
    from src.pipelines.optimized import OptimizedPipeline

    cfg = load_pipeline_config(config_path)

    # Apply ablation overrides
    if no_rerank:
        cfg = _patch_config(cfg, **{"rerank.enabled": False})
    if query_transform:
        cfg = _patch_config(
            cfg,
            **{"query_transform.enabled": True, "query_transform.mode": query_transform},
        )

    pipeline = OptimizedPipeline(cfg)
    result = pipeline.query(question)

    if compare:
        _compare_with_baseline(question, result, log_level)
        return

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _pretty_print(question, result, label="Optimized")


def _pretty_print(question: str, result: dict, label: str = "Pipeline") -> None:
    typer.echo(f"\n[{label}]")
    typer.echo(f"Question : {question}")
    typer.echo(f"Answer   : {result['answer']}")
    typer.echo(f"Latency  : {result['latency_ms']} ms")
    if result.get("sources"):
        typer.echo(f"\nSources ({len(result['sources'])}):")
        for i, meta in enumerate(result["sources"], 1):
            doc_id   = meta.get("doc_id", "?")
            source   = meta.get("source", "?")
            modality = meta.get("modality", "?")
            typer.echo(f"  [{i}] {doc_id}  ({modality})  — {source}")


def _compare_with_baseline(question: str, opt_result: dict, log_level: str) -> None:
    """Run baseline pipeline and display both results side-by-side."""
    from src.config import load_pipeline_config as _lpc
    from src.pipelines.baseline import BaselinePipeline

    try:
        base_cfg      = _lpc(Path("configs/baseline.yaml"))
        base_pipeline = BaselinePipeline(base_cfg)
        base_result   = base_pipeline.query(question)
    except Exception as exc:
        typer.echo(f"[Baseline] failed: {exc}", err=True)
        base_result = {"answer": "N/A", "latency_ms": 0.0, "sources": []}

    _pretty_print(question, base_result, label="Baseline")
    typer.echo("")
    _pretty_print(question, opt_result,  label="Optimized")

    diff_ms = opt_result["latency_ms"] - base_result["latency_ms"]
    typer.echo(f"\nLatency delta: {diff_ms:+.1f} ms (optimized vs baseline)")


# ── batch command ─────────────────────────────────────────────────────────────


@app.command()
def batch(
    questions_file: Path = typer.Argument(
        ...,
        help="JSONL file: one question per line (plain string or {\"question\": ...}).",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSONL results here. Defaults to stdout."
    ),
    config_path: Path = _CONFIG_OPT,
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking (ablation)."),
    query_transform: Optional[str] = typer.Option(
        None, "--query-transform", "-qt", help="'multi_query' | 'hyde'."
    ),
    log_level: str = _LOG_LEVEL_OPT,
) -> None:
    """Run a batch of questions and write JSONL results.

    Input (one per line)::

        "What is Paris?"
        {"question": "Who wrote Hamlet?"}
    """
    _setup_logging(log_level)
    logger = logging.getLogger("run_optimized.batch")

    if not questions_file.exists():
        typer.echo(f"Error: file not found: {questions_file}", err=True)
        raise typer.Exit(code=1)

    questions: list[str] = []
    with questions_file.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, str):
                    questions.append(parsed)
                elif isinstance(parsed, dict):
                    questions.append(parsed["question"])
                else:
                    logger.warning("Line %d: unexpected type — skipping", lineno)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Line %d parse error — skipping: %s", lineno, exc)

    if not questions:
        typer.echo("No valid questions found.", err=True)
        raise typer.Exit(code=1)

    from src.config import load_pipeline_config
    from src.pipelines.optimized import OptimizedPipeline

    cfg = load_pipeline_config(config_path)
    if no_rerank:
        cfg = _patch_config(cfg, **{"rerank.enabled": False})
    if query_transform:
        cfg = _patch_config(
            cfg,
            **{"query_transform.enabled": True, "query_transform.mode": query_transform},
        )

    pipeline = OptimizedPipeline(cfg)
    logger.info("Running batch of %d questions …", len(questions))
    results = pipeline.run_batch(questions)

    out_lines = [
        json.dumps({"question": q, **r}, ensure_ascii=False)
        for q, r in zip(questions, results)
    ]

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        typer.echo(f"Results written to {output_file}  ({len(results)} items)")
    else:
        for line in out_lines:
            typer.echo(line)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
