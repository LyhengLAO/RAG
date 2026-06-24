"""FastAPI serving layer: decouples the frontend from the RAG core.

Endpoints
---------
POST /query   — run a question through the baseline or optimized pipeline
GET  /metrics — serve results/metrics.json for the dashboard
GET  /health  — readiness probe (Ollama reachable, ChromaDB accessible, pipelines loaded)

Usage
-----
    uvicorn src.serving.api:app --reload --port 8000

Both pipeline instances are loaded **once** at startup via the ASGI lifespan
and held in module-level state for the lifetime of the process.  A 503 is
returned for any request that requires a pipeline that failed to load.
"""

from __future__ import annotations

import importlib
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

# ── Pydantic schemas ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    pipeline: Literal["baseline", "optimized"] = Field(
        "optimized", description="Which pipeline variant to use"
    )


class SourceInfo(BaseModel):
    doc_id: str | None = None
    modality: str | None = None
    source: str | None = None
    license: str | None = None

    model_config = {"extra": "allow"}  # pass through any extra metadata fields


class QueryResponse(BaseModel):
    answer: str
    retrieved_contexts: list[str]
    sources: list[SourceInfo]
    latency_ms: float
    pipeline: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    pipelines_loaded: dict[str, bool]
    ollama_up: bool
    chroma_accessible: bool
    ollama_model: str


# ── Application state ──────────────────────────────────────────────────────────


class _AppState:
    """Pipeline singletons loaded once at startup, held for process lifetime."""

    baseline: Any | None = None
    optimized: Any | None = None
    load_errors: dict[str, str] = {}


_state = _AppState()

_PIPELINE_CLASSES: dict[str, tuple[str, str]] = {
    "baseline":  ("src.pipelines.baseline",  "BaselinePipeline"),
    "optimized": ("src.pipelines.optimized", "OptimizedPipeline"),
}


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load both pipeline instances once; keep them for the process lifetime."""
    for name, (mod_path, cls_name) in _PIPELINE_CLASSES.items():
        try:
            mod = importlib.import_module(mod_path)
            pipeline = getattr(mod, cls_name)()
            setattr(_state, name, pipeline)
            logger.info("Pipeline %r loaded", name)
        except Exception as exc:
            logger.error("Pipeline %r failed to load: %s", name, exc)
            _state.load_errors[name] = str(exc)
    yield
    # Nothing to tear down — models stay alive until process exit.


# ── App ────────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="MultiModal RAG API",
    description="Baseline vs Optimized RAG pipeline comparison",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: open to all origins so any Streamlit instance (default: :8501) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health probes ──────────────────────────────────────────────────────────────


def _ollama_up() -> bool:
    """Return True if Ollama responds to a tags listing within 3 s."""
    try:
        r = _requests.get(
            f"{settings.ollama_host.rstrip('/')}/api/tags",
            timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False


def _chroma_accessible() -> bool:
    """Return True if ChromaDB can list its collections at the configured path."""
    try:
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        client.list_collections()
        return True
    except Exception:
        return False


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe — returns 200 always; ``status`` field indicates severity."""
    ollama = _ollama_up()
    chroma = _chroma_accessible()
    loaded = {name: getattr(_state, name) is not None for name in _PIPELINE_CLASSES}
    all_ok = ollama and chroma and all(loaded.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        pipelines_loaded=loaded,
        ollama_up=ollama,
        chroma_accessible=chroma,
        ollama_model=settings.ollama_model,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run *question* through the selected pipeline and return the full result."""
    pipeline = getattr(_state, req.pipeline)
    if pipeline is None:
        err = _state.load_errors.get(req.pipeline, "pipeline not loaded")
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline '{req.pipeline}' is unavailable: {err}",
        )

    try:
        result: dict[str, Any] = pipeline.query(req.question)
    except RuntimeError as exc:
        # LLM/Ollama connectivity issues — already logged inside pipeline
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in pipeline %r", req.pipeline)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceInfo(**s) if isinstance(s, dict) else SourceInfo()
        for s in result.get("sources", [])
    ]
    return QueryResponse(
        answer=result["answer"],
        retrieved_contexts=result.get("retrieved_contexts", []),
        sources=sources,
        latency_ms=float(result.get("latency_ms", 0.0)),
        pipeline=req.pipeline,
    )


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Return the content of results/metrics.json, or 404 if not yet generated."""
    path = settings.results_dir / "metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found — run scripts/run_comparison.py first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))
