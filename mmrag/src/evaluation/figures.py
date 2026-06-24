"""Comparison figures for the ablation matrix (matplotlib, headless ``Agg``).

Consumes the aggregated ``results/metrics.json`` ( ``{config_name: result_doc}`` )
and writes PNG bar charts to ``results/figures/``:

* ``retrieval_comparison.png`` — one grouped-bar panel per retrieval metric,
  bars = configs.
* ``ragas_comparison.png``     — one panel per RAGAS metric, bars = configs.
* ``system_comparison.png``    — latency p50/p95 and mean total tokens per config.
* ``per_modality_<metric>.png``— a chosen metric per modality, bars grouped by
  config (the "ventilation par modalité").

NaN values (an unmeasurable metric) are drawn as a zero-height bar annotated
"NaN" so a missing measurement is visible, never silently shown as 0.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# Retrieval / RAGAS metrics shown by default (only those present are plotted).
RETRIEVAL_METRICS = ["hit@1", "hit@5", "recall@5", "recall@10", "ndcg@10", "mrr"]
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def _is_nan(x: Any) -> bool:
    return not isinstance(x, (int, float)) or (isinstance(x, float) and math.isnan(x))


def _bar_panel(
    ax: Any,
    labels: list[str],
    values: list[float],
    title: str,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> None:
    """Draw one bar panel; NaN bars become height-0 annotated 'NaN'."""
    heights = [0.0 if _is_nan(v) else float(v) for v in values]
    colors = ["#cccccc" if _is_nan(v) else "#3b76af" for v in values]
    x = np.arange(len(labels))
    bars = ax.bar(x, heights, color=colors)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    for rect, v in zip(bars, values):
        label = "NaN" if _is_nan(v) else f"{v:.2f}"
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                label, ha="center", va="bottom", fontsize=7)


def _config_names(metrics: dict[str, Any]) -> list[str]:
    return list(metrics.keys())


def _grid(n: int) -> tuple[int, int]:
    cols = min(3, n) if n else 1
    rows = math.ceil(n / cols) if n else 1
    return rows, cols


def _panel_figure(
    metrics: dict[str, Any],
    metric_names: list[str],
    value_fn: Callable[[dict[str, Any], str], float],
    suptitle: str,
    out_path: Path,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> Path | None:
    """Render one panel per metric (bars = configs). value_fn returns NaN when
    a metric is absent, so every requested metric still gets a (possibly NaN)
    panel."""
    configs = _config_names(metrics)
    if not configs or not metric_names:
        return None

    rows, cols = _grid(len(metric_names))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)
    for idx, metric in enumerate(metric_names):
        ax = axes[idx // cols][idx % cols]
        values = [value_fn(metrics[c], metric) for c in configs]
        _bar_panel(ax, configs, values, metric, ylim=ylim)
    # hide unused axes
    for j in range(len(metric_names), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("figure → %s", out_path)
    return out_path


# ── value extractors ────────────────────────────────────────────────────────────


def _retrieval_value(result: dict[str, Any], metric: str) -> float:
    return result.get("retrieval", {}).get("overall", {}).get(metric, float("nan"))


def _ragas_value(result: dict[str, Any], metric: str) -> float:
    return result.get("ragas", {}).get("overall", {}).get(metric, float("nan"))


# ── public API ──────────────────────────────────────────────────────────────────


def plot_retrieval_comparison(metrics: dict[str, Any], out_dir: Path) -> Path | None:
    return _panel_figure(metrics, RETRIEVAL_METRICS, _retrieval_value,
                         "Retrieval metrics by config", out_dir / "retrieval_comparison.png")


def plot_ragas_comparison(metrics: dict[str, Any], out_dir: Path) -> Path | None:
    return _panel_figure(metrics, RAGAS_METRICS, _ragas_value,
                         "RAGAS quality metrics by config", out_dir / "ragas_comparison.png")


def plot_system_comparison(metrics: dict[str, Any], out_dir: Path) -> Path | None:
    """Latency p50/p95 (ms) and mean total tokens per config."""
    configs = _config_names(metrics)
    if not configs:
        return None
    p50 = [metrics[c].get("system", {}).get("overall", {}).get("latency_ms", {}).get("p50", float("nan"))
           for c in configs]
    p95 = [metrics[c].get("system", {}).get("overall", {}).get("latency_ms", {}).get("p95", float("nan"))
           for c in configs]
    tokens = [metrics[c].get("system", {}).get("overall", {}).get("tokens", {}).get("total_tokens", {}).get("mean", float("nan"))
              for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    x = np.arange(len(configs))
    width = 0.38
    ax0 = axes[0][0]
    ax0.bar(x - width / 2, [0 if _is_nan(v) else v for v in p50], width, label="p50", color="#3b76af")
    ax0.bar(x + width / 2, [0 if _is_nan(v) else v for v in p95], width, label="p95", color="#e07b39")
    ax0.set_title("Latency (ms)")
    ax0.set_xticks(x); ax0.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax0.legend(fontsize=8)
    _bar_panel(axes[0][1], configs, tokens, "Mean total tokens / query", ylim=None)
    fig.suptitle("System metrics by config", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "system_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("figure → %s", out)
    return out


def plot_per_modality(
    metrics: dict[str, Any],
    out_dir: Path,
    block: str = "retrieval",
    metric: str = "recall@5",
) -> Path | None:
    """Per-modality breakdown of one metric, grouped bars by config.

    Args:
        metrics: Aggregated results.
        out_dir: Figure output directory.
        block: ``"retrieval"`` / ``"ragas"`` (uses ``per_modality`` block) or
            ``"latency"`` (system latency p50 per modality).
        metric: Metric key within the chosen block (ignored for ``"latency"``).
    """
    configs = _config_names(metrics)
    if not configs:
        return None

    modalities = sorted({
        m
        for c in configs
        for m in metrics[c].get(_block_key(block), {}).get("per_modality", {}).keys()
    })
    if not modalities:
        return None

    def value(cfg: str, modality: str) -> float:
        pm = metrics[cfg].get(_block_key(block), {}).get("per_modality", {}).get(modality, {})
        if block == "latency":
            return pm.get("latency_ms", {}).get("p50", float("nan"))
        return pm.get(metric, float("nan"))

    fig, ax = plt.subplots(figsize=(2.2 * len(modalities) + 3, 4.5))
    x = np.arange(len(modalities))
    n = len(configs)
    width = 0.8 / max(n, 1)
    palette = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
    for ci, cfg in enumerate(configs):
        vals = [value(cfg, m) for m in modalities]
        heights = [0.0 if _is_nan(v) else float(v) for v in vals]
        ax.bar(x + ci * width - 0.4 + width / 2, heights, width, label=cfg, color=palette[ci])

    title_metric = "latency p50 (ms)" if block == "latency" else f"{block}: {metric}"
    ax.set_title(f"Per-modality {title_metric}", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(modalities)
    if block != "latency":
        ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    safe_metric = metric.replace("@", "").replace("/", "_") if block != "latency" else "latency_p50"
    out = out_dir / f"per_modality_{block}_{safe_metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("figure → %s", out)
    return out


def _block_key(block: str) -> str:
    return "system" if block == "latency" else block


def generate_all_figures(metrics: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Generate the full figure set; returns the list of written paths."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    for fn in (plot_retrieval_comparison, plot_ragas_comparison, plot_system_comparison):
        try:
            p = fn(metrics, out_dir)
            if p:
                written.append(p)
        except Exception as exc:  # noqa: BLE001 - one bad chart shouldn't kill the rest
            logger.error("figure %s failed: %s", fn.__name__, exc)

    per_modality_specs = [
        ("retrieval", "recall@5"),
        ("retrieval", "hit@5"),
        ("ragas", "faithfulness"),
        ("ragas", "answer_correctness"),
        ("latency", ""),
    ]
    for block, metric in per_modality_specs:
        try:
            p = plot_per_modality(metrics, out_dir, block=block, metric=metric)
            if p:
                written.append(p)
        except Exception as exc:  # noqa: BLE001
            logger.error("per-modality figure (%s/%s) failed: %s", block, metric, exc)
    return written


__all__ = [
    "generate_all_figures",
    "plot_retrieval_comparison",
    "plot_ragas_comparison",
    "plot_system_comparison",
    "plot_per_modality",
]
