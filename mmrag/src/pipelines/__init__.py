"""Pipelines sub-package: end-to-end baseline and optimized RAG pipelines."""

from src.pipelines.baseline import BaselinePipeline
from src.pipelines.optimized import OptimizedPipeline

__all__ = ["BaselinePipeline", "OptimizedPipeline"]
