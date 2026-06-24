"""Benchmark facade: baseline vs optimized, delegating to the ablation runner.

This is a thin compatibility layer over the real evaluation engine
(:mod:`src.evaluation.runner` + :mod:`src.evaluation.ablation` +
:mod:`src.evaluation.figures`).  It exists so older call-sites that expect a
``Benchmark`` object keep working; new code should call
:func:`src.evaluation.runner.run_config_eval` or use
``scripts/run_comparison.py`` directly.

Every method computes real metrics — there are no placeholders.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import typer

from src.evaluation.ablation import BASELINE, OPTIMIZED, BaseSettings, RunConfig
from src.evaluation.runner import load_eval_set, run_config_eval

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

_CONFIGS: dict[str, RunConfig] = {"baseline": BASELINE, "optimized": OPTIMIZED}


class Benchmark:
    """Run the baseline and/or optimized endpoints on an eval set and report.

    Args:
        eval_dataset_path: Path to the eval JSONL (see scripts/build_eval_set.py).
        output_dir: Directory for metrics JSON and figures.
        base: Held-constant settings shared by both endpoints.
        do_ragas: Whether to compute the RAGAS quality metrics.
    """

    def __init__(
        self,
        eval_dataset_path: str | Path,
        output_dir: str | Path = "results",
        base: BaseSettings | None = None,
        do_ragas: bool = True,
    ) -> None:
        self.eval_dataset_path = Path(eval_dataset_path)
        self.output_dir = Path(output_dir)
        self.base = base or BaseSettings()
        self.do_ragas = do_ragas
        self._records = load_eval_set(self.eval_dataset_path)

    def run(self, pipeline: Literal["baseline", "optimized", "both"] = "both") -> dict[str, Any]:
        """Evaluate the requested pipeline(s) and return ``{name: result_doc}``."""
        names = ["baseline", "optimized"] if pipeline == "both" else [pipeline]
        out: dict[str, Any] = {}
        for name in names:
            out[name] = run_config_eval(
                _CONFIGS[name], self._records, base=self.base,
                do_ragas=self.do_ragas, results_dir=self.output_dir, save=True,
            )
        return out

    def compare(self) -> dict[str, Any]:
        """Run both endpoints and add a ``delta`` block (optimized − baseline)."""
        results = self.run("both")
        base_r = results["baseline"].get("retrieval", {}).get("overall", {})
        opt_r = results["optimized"].get("retrieval", {}).get("overall", {})
        delta = {
            k: float(opt_r[k]) - float(base_r[k])
            for k in base_r
            if k in opt_r and k != "n_queries"
        }
        results["delta"] = {"retrieval": delta}
        return results

    def save_report(self, results: dict[str, Any], path: str | Path | None = None) -> Path:
        """Serialise *results* to JSON and return the path."""
        path = Path(path) if path else self.output_dir / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def plot(self, results: dict[str, Any], output_dir: str | Path | None = None) -> None:
        """Generate comparison figures from *results*."""
        from src.evaluation.figures import generate_all_figures  # noqa: PLC0415

        metrics = {k: v for k, v in results.items() if k != "delta"}
        generate_all_figures(metrics, Path(output_dir or self.output_dir) / "figures")


@app.command()
def main(
    mode: str = typer.Option("compare", help="baseline | optimized | compare"),
    eval_path: Path = typer.Option(Path("data/eval/eval_set.jsonl")),
    output_dir: Path = typer.Option(Path("results")),
    no_ragas: bool = typer.Option(False, "--no-ragas"),
) -> None:
    """CLI entry point: ``python -m src.evaluation.benchmark --mode compare``."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    bench = Benchmark(eval_path, output_dir=output_dir, do_ragas=not no_ragas)
    results = bench.compare() if mode == "compare" else bench.run(mode)  # type: ignore[arg-type]
    out = bench.save_report(results)
    bench.plot(results)
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()
