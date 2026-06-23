"""Baseline RAG pipeline: recursive chunking, dense-only retrieval, no reranking.

This is the *deliberately naive* reference point for the comparison study.
All multimodal content (text, image captions, audio transcripts) is projected
into a single text embedding space using sentence-transformers.  Retrieval is
a single cosine similarity search against that collection — no hybrid, no
rerank, no cross-modal CLIP/CLAP lookup.

Pipeline config is driven by ``configs/baseline.yaml``.  The pipeline must
remain intentionally simple: adding optimisations here would invalidate the
comparison with the optimised pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import get_baseline_config, settings
from src.generation.llm import LLMClient
from src.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.indexing.vector_store import VectorStoreIndex
from src.ingestion.schema import RawDocument
from src.preprocessing.chunking import RecursiveChunker

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_manifest(manifest_path: Path) -> list[RawDocument]:
    """Read JSONL manifest and return all RawDocuments."""
    docs: list[RawDocument] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(RawDocument.from_json(line))
            except Exception as exc:
                logger.warning("manifest line %d parse error — skipping: %s", lineno, exc)
    return docs


def _raw_to_langchain(doc: RawDocument) -> Document:
    """Convert a RawDocument to a LangChain Document for chunking.

    All modalities use the ``text`` field as page content:
    - text  : the passage itself
    - image : the original dataset caption
    - audio : the Whisper transcript
    """
    return Document(
        page_content=doc.text,
        metadata={
            "doc_id":   doc.id,
            "modality": doc.modality,
            "source":   doc.source,
            "license":  doc.license,
        },
    )


# ── pipeline class ────────────────────────────────────────────────────────────


class BaselinePipeline:
    """End-to-end baseline RAG pipeline driven by ``configs/baseline.yaml``.

    Component chain
    ---------------
    index():
        manifest → :class:`RawDocument` list
        → :class:`RecursiveChunker`
        → :class:`~src.embeddings.text_embedder.TextEmbedder`
        → :class:`VectorStoreIndex` (text collection)

    query():
        question string
        → embed_query_numpy()
        → :meth:`VectorStoreIndex.similarity_search` (text collection)
        → :func:`~src.generation.prompts.build_rag_prompt`
        → :meth:`LLMClient.generate`
        → ``{"answer", "retrieved_contexts", "sources", "latency_ms"}``

    Args:
        config: Pipeline configuration dict.  If *None*, loaded from
                ``configs/baseline.yaml``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._cfg: dict[str, Any] = config or get_baseline_config()
        self._chunking  = self._cfg["chunking"]
        self._embed_cfg = self._cfg["embeddings"]
        self._retrieval = self._cfg["retrieval"]
        self._gen_cfg   = self._cfg["generation"]
        self._idx_cfg   = self._cfg["indexing"]

        # Lazy-initialised components
        self._chunker:  RecursiveChunker | None = None
        self._embedder: Any | None = None          # TextEmbedder
        self._vector_index: VectorStoreIndex | None = None
        self._llm: LLMClient | None = None

    # ── lazy component accessors ──────────────────────────────────────────────

    @property
    def chunker(self) -> RecursiveChunker:
        if self._chunker is None:
            self._chunker = RecursiveChunker(
                chunk_size=self._chunking["chunk_size"],
                chunk_overlap=self._chunking["chunk_overlap"],
            )
        return self._chunker

    @property
    def embedder(self) -> Any:  # TextEmbedder
        if self._embedder is None:
            from src.embeddings.text_embedder import TextEmbedder  # noqa: PLC0415
            self._embedder = TextEmbedder(
                model_name=self._embed_cfg["text_model"],
            )
        return self._embedder

    @property
    def vector_index(self) -> VectorStoreIndex:
        if self._vector_index is None:
            self._vector_index = VectorStoreIndex(
                persist_dir=settings.chroma_persist_dir,
                collection_prefix=self._idx_cfg["collection"],
            )
        return self._vector_index

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=self._gen_cfg["provider"],
                model=self._gen_cfg["model"],
                temperature=self._gen_cfg["temperature"],
                max_tokens=self._gen_cfg["max_tokens"],
            )
        return self._llm

    # ── indexing ──────────────────────────────────────────────────────────────

    def index(self, data_dir: str | Path = "data") -> None:
        """Ingest, chunk, embed, and index all documents in the manifest.

        Reads ``{data_dir}/processed/manifest.jsonl``.  All modalities are
        indexed into the single ``text`` collection using sentence-transformer
        embeddings — captions and transcripts are treated as plain text.

        The operation is idempotent: stable chunk IDs mean re-running with the
        same manifest is safe and produces no duplicates in ChromaDB.

        Args:
            data_dir: Base data directory (default ``"data"``).

        Raises:
            FileNotFoundError: If the manifest does not exist.
        """
        data_dir = Path(data_dir)
        manifest_path = data_dir / "processed" / "manifest.jsonl"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run scripts/build_dataset.py first."
            )

        # ── load documents ────────────────────────────────────────────────────
        raw_docs = _load_manifest(manifest_path)
        logger.info("Loaded %d raw documents from manifest", len(raw_docs))

        # Filter out items with no usable text
        valid: list[RawDocument] = [d for d in raw_docs if d.text.strip()]
        skipped = len(raw_docs) - len(valid)
        if skipped:
            logger.warning("%d documents skipped (empty text field)", skipped)

        if not valid:
            logger.error("No documents with text content found — aborting index()")
            return

        # ── convert to LangChain Documents ───────────────────────────────────
        lc_docs = [_raw_to_langchain(d) for d in valid]

        # ── chunk ─────────────────────────────────────────────────────────────
        logger.info(
            "Chunking %d documents (strategy=recursive, size=%d, overlap=%d)",
            len(lc_docs),
            self._chunking["chunk_size"],
            self._chunking["chunk_overlap"],
        )
        chunks = self.chunker.chunk(lc_docs)
        logger.info("Produced %d chunks", len(chunks))

        if not chunks:
            logger.error("Chunking produced 0 chunks — aborting index()")
            return

        # ── embed ─────────────────────────────────────────────────────────────
        logger.info(
            "Embedding %d chunks with %s …", len(chunks), self._embed_cfg["text_model"]
        )
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_numpy(texts)   # (N, D) float32
        logger.info("Embedding complete, shape=%s", embeddings.shape)

        # ── upsert ────────────────────────────────────────────────────────────
        ids = self.vector_index.upsert(
            chunks=chunks,
            embeddings=embeddings,
            modality="text",
            embedder_name=self.embedder.model_id,
        )
        logger.info(
            "Indexed %d chunks into collection %r",
            len(ids),
            f"{self._idx_cfg['collection']}_text",
        )

    # ── querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """Run a single question through retrieval → generation.

        Args:
            question: Natural-language question.

        Returns:
            Dict with keys:
            - ``"answer"``             : generated string
            - ``"retrieved_contexts"`` : list of chunk text strings (top-k)
            - ``"sources"``            : list of chunk metadata dicts
            - ``"latency_ms"``         : wall-clock time in milliseconds
        """
        t0 = time.perf_counter()

        # 1. Embed query
        query_vec = self.embedder.embed_query_numpy(question)

        # 2. Dense retrieval from text collection
        top_k = self._retrieval["top_k"]
        threshold = self._retrieval.get("score_threshold", 0.0)
        hits = self.vector_index.similarity_search(
            query_embedding=query_vec,
            modality="text",
            k=top_k,
            score_threshold=threshold if threshold > 0.0 else None,
        )

        context_docs = [doc for doc, _ in hits]
        logger.debug("Retrieved %d chunks for question: %r", len(context_docs), question[:80])

        # 3. Build prompt
        system_prompt: str = self._gen_cfg.get("system_prompt", RAG_SYSTEM_PROMPT)
        prompt = build_rag_prompt(
            question=question,
            context_docs=context_docs,
            max_context_chars=4_000,
        )

        # 4. Generate — degrade gracefully if Ollama is unavailable
        try:
            answer_text = self.llm.generate(prompt, system_prompt=system_prompt)
        except RuntimeError as exc:
            logger.error("LLM generation failed: %s", exc)
            answer_text = (
                "Error: LLM unavailable. "
                "Ensure Ollama is running (`ollama serve`) and the model is pulled."
            )

        latency_ms = round((time.perf_counter() - t0) * 1_000, 1)

        return {
            "answer": answer_text,
            "retrieved_contexts": [doc.page_content for doc in context_docs],
            "sources": [doc.metadata for doc in context_docs],
            "latency_ms": latency_ms,
        }

    def run_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        """Run :meth:`query` for each question and return all results.

        Each question is processed independently; failures on one question do
        not abort the remaining ones.

        Args:
            questions: List of question strings.

        Returns:
            List of result dicts in the same order as *questions*.
        """
        results: list[dict[str, Any]] = []
        for i, q in enumerate(questions, 1):
            logger.info("Batch query %d/%d: %r", i, len(questions), q[:60])
            try:
                results.append(self.query(q))
            except Exception as exc:
                logger.error("query %d failed: %s", i, exc)
                results.append(
                    {
                        "answer": f"Error: {exc}",
                        "retrieved_contexts": [],
                        "sources": [],
                        "latency_ms": 0.0,
                    }
                )
        return results


# ── module-level convenience function ────────────────────────────────────────

_singleton: BaselinePipeline | None = None


def answer(query: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run *query* through the baseline pipeline and return a result dict.

    Initialises a singleton :class:`BaselinePipeline` on first call; reuses it
    on subsequent calls (model stays in memory).  Assumes the vector index has
    already been built via :meth:`BaselinePipeline.index`.

    Args:
        query:  Natural-language question.
        config: Optional config dict override.  Only applied on first call.

    Returns:
        ``{"answer", "retrieved_contexts", "sources", "latency_ms"}``
    """
    global _singleton
    if _singleton is None:
        _singleton = BaselinePipeline(config)
    return _singleton.query(query)


# ── __main__ entry point ─────────────────────────────────────────────────────


def __main__() -> None:  # noqa: N807
    """Minimal CLI: ``python -m src.pipelines.baseline 'your question'``."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.pipelines.baseline 'your question'\n"
            "       (use scripts/run_baseline.py for full CLI options)",
            file=sys.stderr,
        )
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    import json as _json
    result = answer(question)
    print(_json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    __main__()
