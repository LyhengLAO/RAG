"""CLI for the baseline RAG pipeline.

Usage
-----
Build the vector index from the preprocessed manifest::

    python scripts/run_baseline.py index

Ask a single question (index must already be built)::

    python scripts/run_baseline.py ask "Who wrote Hamlet?"

Run a batch of questions from a JSONL file (one question per line)::

    python scripts/run_baseline.py batch questions.jsonl --output results.jsonl

Run end-to-end (index then ask)::

    python scripts/run_baseline.py index && python scripts/run_baseline.py ask "..."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(
    name="run-baseline",
    help="Baseline RAG pipeline: index documents or answer questions.",
    add_completion=False,
)

# ── shared options ────────────────────────────────────────────────────────────

_CONFIG_OPT = typer.Option(
    Path("configs/baseline.yaml"),
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


# ── index command ─────────────────────────────────────────────────────────────


@app.command()
def index(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        "-d",
        help="Base data directory (manifest expected at <data-dir>/processed/manifest.jsonl).",
        show_default=True,
    ),
    config_path: Path = _CONFIG_OPT,
    log_level: str = _LOG_LEVEL_OPT,
) -> None:
    """Build the ChromaDB vector index from the preprocessed manifest.

    Run ``scripts/build_dataset.py`` first to create the manifest.
    This command is idempotent — safe to run again after adding new documents.
    """
    _setup_logging(log_level)
    logger = logging.getLogger("run_baseline.index")

    from src.config import load_pipeline_config
    from src.pipelines.baseline import BaselinePipeline

    cfg = load_pipeline_config(config_path)
    pipeline = BaselinePipeline(cfg)

    logger.info("Starting index build from %s", data_dir / "processed" / "manifest.jsonl")
    try:
        pipeline.index(data_dir)
        n = pipeline.vector_index.count("text")
        typer.echo(f"Index built successfully. Text collection: {n} chunks.")
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
        False,
        "--json",
        "-j",
        help="Print full JSON result instead of human-readable output.",
    ),
    log_level: str = _LOG_LEVEL_OPT,
) -> None:
    """Answer a single question using the baseline RAG pipeline.

    The vector index must already be built (run the ``index`` command first).
    Ollama must be running: ``ollama serve`` and the model must be available.
    """
    _setup_logging(log_level)

    from src.config import load_pipeline_config
    from src.pipelines.baseline import BaselinePipeline

    cfg = load_pipeline_config(config_path)
    pipeline = BaselinePipeline(cfg)

    result = pipeline.query(question)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"\nQuestion : {question}")
        typer.echo(f"Answer   : {result['answer']}")
        typer.echo(f"Latency  : {result['latency_ms']} ms")
        if result["sources"]:
            typer.echo(f"\nSources ({len(result['sources'])}):")
            for i, meta in enumerate(result["sources"], 1):
                doc_id  = meta.get("doc_id", "?")
                source  = meta.get("source", "?")
                modality = meta.get("modality", "?")
                typer.echo(f"  [{i}] {doc_id}  ({modality})  — {source}")


# ── batch command ─────────────────────────────────────────────────────────────


@app.command()
def batch(
    questions_file: Path = typer.Argument(
        ...,
        help="JSONL file with one question per line (plain string or {\"question\": ...} object).",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write results to this JSONL file.  Defaults to stdout.",
    ),
    config_path: Path = _CONFIG_OPT,
    log_level: str = _LOG_LEVEL_OPT,
) -> None:
    """Run a batch of questions from a JSONL file and write results.

    Input format (one per line)::

        "What is Paris?"
        {"question": "Who wrote Hamlet?"}

    Output is JSONL with one result dict per line.
    """
    _setup_logging(log_level)
    logger = logging.getLogger("run_baseline.batch")

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
                    logger.warning("Line %d: unexpected type %s — skipping", lineno, type(parsed))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Line %d parse error — skipping: %s", lineno, exc)

    if not questions:
        typer.echo("No valid questions found in input file.", err=True)
        raise typer.Exit(code=1)

    from src.config import load_pipeline_config
    from src.pipelines.baseline import BaselinePipeline

    cfg = load_pipeline_config(config_path)
    pipeline = BaselinePipeline(cfg)

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
