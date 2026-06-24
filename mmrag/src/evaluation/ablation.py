"""Ablation matrix definition and pipeline-config construction.

Three binary axes are isolated for the comparison study:

    Axis A  chunking   ∈ {recursive, semantic}
    Axis B  retrieval  ∈ {dense, hybrid}
    Axis C  rerank     ∈ {off, on}

    Baseline   = (recursive, dense,  off)
    Optimized  = (semantic,  hybrid, on)

To measure the *marginal* contribution of each optimisation, the default matrix
also contains the three single-axis variants that flip exactly one axis away
from the baseline.  Everything else — embedding model, generation model and
decoding params, top-k, prompt — is held **constant** across all runs so any
metric delta is attributable solely to the flipped axis.  (This is stricter
than the repo's two stand-alone YAMLs, which also differ in embedding model;
for a controlled ablation we deliberately remove that confound.)

A single configurable engine — :class:`~src.pipelines.optimized.OptimizedPipeline`
— realises every cell of the matrix via ``retrieval.strategy`` (dense/hybrid),
``chunking.strategy`` (recursive/semantic) and ``rerank.enabled`` (on/off), so
all cells share one code path.

Index sharing
-------------
Only the chunking axis changes what is stored in the index; retrieval and rerank
are query-time choices.  Each chunking strategy therefore gets its own Chroma
collection and BM25 file (``{prefix}_{chunking}``), built once and reused by
every config with that chunking strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

Chunking = Literal["recursive", "semantic"]
Retrieval = Literal["dense", "hybrid"]


# ── held-constant settings shared by every cell of the matrix ───────────────────


@dataclass(frozen=True)
class BaseSettings:
    """Everything held constant across the ablation (the experimental control).

    Attributes:
        embedding_model: Text embedder used for indexing, dense retrieval and the
            semantic chunker's breakpoint detection. Constant ⇒ the retrieval
            axis is not confounded by an embedding-model change.
        chunk_size / chunk_overlap / breakpoint_percentile: Chunker params.
        top_k_retrieval: Candidates fetched before reranking (over-retrieval).
        top_k_final: Results returned to the generator and scored by retrieval
            metrics. Set ≥ the largest ``@k`` you want to report.
        rerank_model / rerank_batch_size: Cross-encoder settings.
        gen_provider / gen_model / gen_temperature / gen_max_tokens / system_prompt:
            Generation config (identical to the repo baseline/optimized YAMLs).
        collection_prefix: Chroma/BM25 namespace; per-chunking suffix is appended.
        data_dir: Base data directory holding ``processed/manifest.jsonl``.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    breakpoint_percentile: float = 25.0

    top_k_retrieval: int = 30
    top_k_final: int = 10

    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_batch_size: int = 32

    gen_provider: str = "ollama"
    gen_model: str = "llama3.2"
    gen_temperature: float = 0.1
    gen_max_tokens: int = 512
    system_prompt: str = (
        "You are a helpful assistant. Answer the question using only the provided "
        'context. If the answer is not in the context, say "I don\'t know."'
    )

    collection_prefix: str = "mmrag_ablation"
    data_dir: str = "data"


# ── one cell of the matrix ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunConfig:
    """A single ablation cell: the three axis values plus a stable name.

    Attributes:
        name: Short identifier used for filenames and figure labels.
        chunking: Axis A value.
        retrieval: Axis B value.
        rerank: Axis C value (True = reranker on).
        label: Human-readable description (auto-derived if empty).
    """

    name: str
    chunking: Chunking
    retrieval: Retrieval
    rerank: bool
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(
                self,
                "label",
                f"{self.chunking}/{self.retrieval}/rerank={'on' if self.rerank else 'off'}",
            )

    @property
    def axes(self) -> dict[str, Any]:
        return {"chunking": self.chunking, "retrieval": self.retrieval, "rerank": self.rerank}


# ── canonical endpoints ─────────────────────────────────────────────────────────

BASELINE = RunConfig("baseline", "recursive", "dense", False, label="baseline (recursive/dense/off)")
OPTIMIZED = RunConfig("optimized", "semantic", "hybrid", True, label="optimized (semantic/hybrid/on)")


def ablation_matrix(full: bool = False) -> list[RunConfig]:
    """Return the list of configs to run.

    Args:
        full: If ``False`` (default) return the baseline, the optimized config,
            and the three single-axis variants (each flipping exactly one axis
            from the baseline) — five configs total, enough to attribute the
            marginal effect of every axis. If ``True``, return all 2³ = 8 cells.

    Returns:
        Ordered list of :class:`RunConfig`.
    """
    if full:
        configs: list[RunConfig] = []
        for chunk, retr, rr in product(("recursive", "semantic"), ("dense", "hybrid"), (False, True)):
            if (chunk, retr, rr) == ("recursive", "dense", False):
                configs.append(BASELINE)
            elif (chunk, retr, rr) == ("semantic", "hybrid", True):
                configs.append(OPTIMIZED)
            else:
                name = f"{chunk[:3]}_{retr[:3]}_rr{'on' if rr else 'off'}"
                configs.append(RunConfig(name, chunk, retr, rr))  # type: ignore[arg-type]
        return configs

    # Default: baseline + single-axis flips + optimized.
    return [
        BASELINE,
        RunConfig("chunking_only", "semantic", "dense", False, label="+ semantic chunking"),
        RunConfig("retrieval_only", "recursive", "hybrid", False, label="+ hybrid retrieval"),
        RunConfig("rerank_only", "recursive", "dense", True, label="+ cross-encoder rerank"),
        OPTIMIZED,
    ]


# ── config construction ─────────────────────────────────────────────────────────


def _collection_name(base: BaseSettings, chunking: str) -> str:
    return f"{base.collection_prefix}_{chunking}"


def _bm25_path(base: BaseSettings, chunking: str) -> str:
    return f"chroma_db/bm25_{base.collection_prefix}_{chunking}.pkl"


def build_pipeline_config(rc: RunConfig, base: BaseSettings | None = None) -> dict[str, Any]:
    """Translate an ablation cell into an :class:`OptimizedPipeline` config dict.

    Args:
        rc: The ablation cell (axis values).
        base: Held-constant settings. Defaults to :class:`BaseSettings`.

    Returns:
        A config dict consumable by ``OptimizedPipeline(config=...)``.
    """
    base = base or BaseSettings()
    return {
        "pipeline": f"ablation:{rc.name}",
        "chunking": {
            "strategy": rc.chunking,
            "chunk_size": base.chunk_size,
            "chunk_overlap": base.chunk_overlap if rc.chunking == "recursive" else 0,
            "breakpoint_percentile": base.breakpoint_percentile,
            "breakpoint_threshold": None,
            "embedding_model": base.embedding_model,
        },
        "embeddings": {
            "text_model": base.embedding_model,
            "image_model": None,
            "audio_model": None,
        },
        "indexing": {
            "backend": "chromadb",
            "collection": _collection_name(base, rc.chunking),
        },
        "retrieval": {
            "strategy": rc.retrieval,
            "top_k_retrieval": base.top_k_retrieval,
            "score_threshold": 0.0,
            "bm25_index_path": _bm25_path(base, rc.chunking),
        },
        "rerank": {
            "enabled": rc.rerank,
            "model": base.rerank_model,
            "top_k_final": base.top_k_final,
            "batch_size": base.rerank_batch_size,
        },
        "query_transform": {"enabled": False, "mode": "multi_query", "n_queries": 3},
        "generation": {
            "provider": base.gen_provider,
            "model": base.gen_model,
            "temperature": base.gen_temperature,
            "max_tokens": base.gen_max_tokens,
            "system_prompt": base.system_prompt,
        },
    }


def configs_by_chunking(configs: list[RunConfig]) -> dict[str, list[RunConfig]]:
    """Group configs by chunking strategy (so each index is built once)."""
    groups: dict[str, list[RunConfig]] = {}
    for rc in configs:
        groups.setdefault(rc.chunking, []).append(rc)
    return groups


__all__ = [
    "BaseSettings",
    "RunConfig",
    "BASELINE",
    "OPTIMIZED",
    "ablation_matrix",
    "build_pipeline_config",
    "configs_by_chunking",
]
