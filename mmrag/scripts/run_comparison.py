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
from pathlib import Path
from typing import Any

import typer

from src.evaluation.ablation import BaseSettings, ablation_matrix
from src.evaluation.figures import generate_all_figures
from src.evaluation.runner import load_eval_set, run_config_eval

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


@app.command()
def main(
    eval_path: Path = typer.Option(Path("data/eval/eval_set.jsonl"), help="Eval set JSONL."),
    results_dir: Path = typer.Option(Path("results"), help="Output directory."),
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
    """Run the matrix, write metrics.json + figures, print the comparison table."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

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
        typer.echo(f"\n══ {rc.name}  [{rc.label}] ══")
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
    typer.echo(f"\n✓  metrics → {metrics_path}")

    if not no_figures:
        figs = generate_all_figures(metrics, results_dir / "figures")
        typer.echo(f"✓  {len(figs)} figures → {results_dir/'figures'}")

    _print_table(metrics)
    _print_deltas(metrics)


if __name__ == "__main__":
    app()
