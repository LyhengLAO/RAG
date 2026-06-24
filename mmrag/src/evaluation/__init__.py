"""Evaluation sub-package: retrieval metrics, RAGAS, system metrics, runner.

Only the dependency-free symbols are imported eagerly.  Anything that pulls in
heavy optional deps (``ragas``, ``datasets``, ``chromadb``, the pipelines …) is
exposed lazily through :pep:`562` module ``__getattr__`` so that, e.g.,
``from src.evaluation import compute_retrieval_metrics`` works without those
packages installed.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.system_metrics import aggregate_system_metrics, percentiles

__all__ = [
    "compute_retrieval_metrics",
    "aggregate_system_metrics",
    "percentiles",
    "compute_ragas",
    "compute_ragas_metrics",
    "run_config_eval",
    "RunConfig",
    "BaseSettings",
    "ablation_matrix",
    "build_pipeline_config",
]

_LAZY: dict[str, tuple[str, str]] = {
    "compute_ragas": ("src.evaluation.ragas_eval", "compute_ragas"),
    "compute_ragas_metrics": ("src.evaluation.metrics", "compute_ragas_metrics"),
    "run_config_eval": ("src.evaluation.runner", "run_config_eval"),
    "RunConfig": ("src.evaluation.ablation", "RunConfig"),
    "BaseSettings": ("src.evaluation.ablation", "BaseSettings"),
    "ablation_matrix": ("src.evaluation.ablation", "ablation_matrix"),
    "build_pipeline_config": ("src.evaluation.ablation", "build_pipeline_config"),
}


def __getattr__(name: str) -> Any:  # noqa: D401 - PEP 562 lazy loader
    if name in _LAZY:
        import importlib  # noqa: PLC0415

        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
