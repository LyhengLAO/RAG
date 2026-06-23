"""Optimized RAG pipeline: semantic chunking, hybrid retrieval, CrossEncoder reranking.

Comparison contract
-------------------
The generation layer (model, temperature, max_tokens, system_prompt) is
**identical** to the baseline.  Only chunking and retrieval change, so every
difference in answer quality is attributable to the retrieval improvements.

Output format is bit-for-bit identical to :class:`~src.pipelines.baseline.BaselinePipeline`
so evaluation scripts can compare results without any adapter code.

Ablation axes (all controlled by ``configs/optimized.yaml``)
------------------------------------------------------------
1. ``chunking.strategy``           : ``"semantic"`` vs ``"recursive"``
2. ``rerank.enabled``              : ``true`` vs ``false``
3. ``query_transform.enabled``     : ``false`` vs ``true``
   ``query_transform.mode``        : ``"multi_query"`` vs ``"hyde"``

Optional query transformation
------------------------------
Two strategies are supported, both activated by ``query_transform.enabled: true``:

**multi_query**
  The LLM generates *N* alternative phrasings of the user question.  Hybrid
  retrieval runs independently for each phrasing, then all result lists are
  fused via a second round of RRF.  Increases recall at the cost of *N* extra
  LLM calls before the main generation.

**HyDE** (Hypothetical Document Embeddings, Gao et al. 2022)
  The LLM writes a short passage that *would answer* the question (the
  "hypothetical document").  That passage is embedded and used for dense
  retrieval instead of the raw query vector.  Rationale: the hypothetical doc
  lies in the same latent space as real answers, making cosine similarity more
  reliable than query-to-passage similarity for difficult questions.  BM25 still
  uses the original question (keywords are from the question, not the hypo-doc).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import get_optimized_config, settings
from src.generation.llm import LLMClient
from src.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.indexing.vector_store import VectorStoreIndex
from src.ingestion.schema import RawDocument
from src.preprocessing.chunking import RecursiveChunker, SemanticChunker
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse_bm25 import BM25Retriever

logger = logging.getLogger(__name__)


# ── helpers (shared with baseline) ───────────────────────────────────────────


def _load_manifest(manifest_path: Path) -> list[RawDocument]:
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


class OptimizedPipeline:
    """End-to-end optimized RAG pipeline driven by ``configs/optimized.yaml``.

    Component chain
    ---------------
    index():
        manifest → :class:`RawDocument` list
        → :class:`SemanticChunker` (or :class:`RecursiveChunker` for ablation)
        → :class:`~src.embeddings.text_embedder.TextEmbedder`
        → :class:`VectorStoreIndex` (text collection)  +  :class:`BM25Retriever`

    query():
        question string
        → [optional query transformation: multi_query | hyde]
        → :class:`HybridRetriever` (BM25 + DenseRetriever via RRF)  k=top_k_retrieval
        → :class:`CrossEncoderReranker`                              k=top_k_final
        → :func:`~src.generation.prompts.build_rag_prompt`
        → :meth:`LLMClient.generate`
        → ``{"answer", "retrieved_contexts", "sources", "latency_ms"}``

    Args:
        config: Pipeline configuration dict.  If *None*, loaded from
                ``configs/optimized.yaml``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._cfg: dict[str, Any] = config or get_optimized_config()
        self._chunking    = self._cfg["chunking"]
        self._embed_cfg   = self._cfg["embeddings"]
        self._retrieval   = self._cfg["retrieval"]
        self._rerank_cfg  = self._cfg["rerank"]
        self._qt_cfg      = self._cfg.get("query_transform", {"enabled": False})
        self._gen_cfg     = self._cfg["generation"]
        self._idx_cfg     = self._cfg["indexing"]

        # Lazy components
        self._chunker:        RecursiveChunker | SemanticChunker | None = None
        self._embedder:       Any | None = None
        self._vector_index:   VectorStoreIndex | None = None
        self._reranker:       CrossEncoderReranker | None = None
        self._llm:            LLMClient | None = None
        self._bm25:           BM25Retriever | None = None
        self._hybrid_r:       HybridRetriever | None = None  # invalidated on index()
        self._dense_r:        DenseRetriever | None = None

        # Try to load a previously saved BM25 index (enables query without re-indexing)
        bm25_path = Path(self._retrieval.get("bm25_index_path", "chroma_db/bm25_optimized.pkl"))
        if bm25_path.exists():
            try:
                self._bm25 = BM25Retriever.load(bm25_path)
                logger.info(
                    "Loaded BM25 index from %s (%d docs)", bm25_path, self._bm25.corpus_size
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load BM25 from %s: %s — will rebuild on index()", bm25_path, exc
                )

    # ── lazy component accessors ──────────────────────────────────────────────

    @property
    def chunker(self) -> RecursiveChunker | SemanticChunker:
        if self._chunker is None:
            strategy = self._chunking.get("strategy", "semantic")
            chunk_size = self._chunking["chunk_size"]

            if strategy == "semantic":
                threshold     = self._chunking.get("breakpoint_threshold") or None
                percentile    = self._chunking.get("breakpoint_percentile", 25.0)
                embed_model   = self._chunking.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                )
                self._chunker = SemanticChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=self._chunking.get("chunk_overlap", 0),
                    embedding_model=embed_model,
                    breakpoint_threshold=threshold,
                    breakpoint_percentile=float(percentile),
                )
            elif strategy == "recursive":
                self._chunker = RecursiveChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=self._chunking.get("chunk_overlap", 64),
                )
            else:
                raise ValueError(
                    f"Unknown chunking strategy {strategy!r}. Use 'semantic' or 'recursive'."
                )
        return self._chunker

    @property
    def embedder(self) -> Any:  # TextEmbedder
        if self._embedder is None:
            from src.embeddings.text_embedder import TextEmbedder  # noqa: PLC0415
            self._embedder = TextEmbedder(model_name=self._embed_cfg["text_model"])
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
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            rc = self._rerank_cfg
            self._reranker = CrossEncoderReranker(
                model_name=rc.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_k=rc.get("top_k_final", 5),
                batch_size=rc.get("batch_size", 32),
            )
        return self._reranker

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            g = self._gen_cfg
            self._llm = LLMClient(
                provider=g["provider"],
                model=g["model"],
                temperature=g["temperature"],
                max_tokens=g["max_tokens"],
            )
        return self._llm

    @property
    def dense_retriever(self) -> DenseRetriever:
        if self._dense_r is None:
            self._dense_r = DenseRetriever(
                index=self.vector_index,
                embedder=self.embedder,
                top_k=self._retrieval["top_k_retrieval"],
                score_threshold=self._retrieval.get("score_threshold", 0.0),
                modality="text",
            )
        return self._dense_r

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_r is None:
            if self._bm25 is None:
                raise RuntimeError(
                    "BM25 index is not ready. "
                    "Run index() first, or ensure the BM25 pickle exists at "
                    f"{self._retrieval.get('bm25_index_path', 'chroma_db/bm25_optimized.pkl')}"
                )
            dense_r = DenseRetriever(
                index=self.vector_index,
                embedder=self.embedder,
                top_k=self._retrieval["top_k_retrieval"],
                score_threshold=self._retrieval.get("score_threshold", 0.0),
                modality="text",
            )
            self._hybrid_r = HybridRetriever(
                dense=dense_r,
                sparse=self._bm25,
                rrf_k=60,
                top_k_per_retriever=self._retrieval["top_k_retrieval"],
            )
        return self._hybrid_r

    # ── indexing ──────────────────────────────────────────────────────────────

    def index(self, data_dir: str | Path = "data") -> None:
        """Ingest, chunk, embed, and index all documents in the manifest.

        Reads ``{data_dir}/processed/manifest.jsonl``.  Builds both the dense
        Chroma index (text collection) and the BM25 sparse index from the same
        chunk corpus.

        The Chroma upsert is idempotent (stable IDs); the BM25 index is
        rebuilt and re-saved on every call.

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

        # ── load ──────────────────────────────────────────────────────────────
        raw_docs = _load_manifest(manifest_path)
        logger.info("Loaded %d raw documents from manifest", len(raw_docs))

        valid = [d for d in raw_docs if d.text.strip()]
        if not valid:
            logger.error("No documents with text content — aborting index()")
            return

        lc_docs = [_raw_to_langchain(d) for d in valid]

        # ── chunk ─────────────────────────────────────────────────────────────
        strategy = self._chunking.get("strategy", "semantic")
        logger.info(
            "Chunking %d documents (strategy=%s, chunk_size=%d)",
            len(lc_docs),
            strategy,
            self._chunking["chunk_size"],
        )
        chunks = self.chunker.chunk(lc_docs)
        logger.info("Produced %d chunks", len(chunks))

        if not chunks:
            logger.error("Chunking produced 0 chunks — aborting index()")
            return

        # ── dense embed + Chroma upsert ───────────────────────────────────────
        logger.info("Embedding %d chunks with %s …", len(chunks), self._embed_cfg["text_model"])
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_numpy(texts)
        logger.info("Embedding complete, shape=%s", embeddings.shape)

        ids = self.vector_index.upsert(
            chunks=chunks,
            embeddings=embeddings,
            modality="text",
            embedder_name=self.embedder.model_id,
        )
        logger.info("Indexed %d chunks into Chroma collection %r_text", len(ids), self._idx_cfg["collection"])

        # ── BM25 index build + save ───────────────────────────────────────────
        logger.info("Building BM25 index from %d chunks …", len(chunks))
        self._bm25 = BM25Retriever(chunks)
        self._hybrid_r = None  # invalidate cached HybridRetriever (new BM25)

        bm25_path = Path(self._retrieval.get("bm25_index_path", "chroma_db/bm25_optimized.pkl"))
        try:
            self._bm25.save(bm25_path)
            logger.info("BM25 index saved to %s", bm25_path)
        except Exception as exc:
            logger.warning("Could not save BM25 index: %s", exc)

    # ── query transformation helpers ──────────────────────────────────────────

    def _expand_multi_query(self, question: str, n: int) -> list[str]:
        """Ask the LLM to generate *n* alternative phrasings of *question*."""
        prompt = (
            f"Generate {n} alternative phrasings of the question below. "
            "Return ONLY the questions — one per line, no numbering, no preamble.\n\n"
            f"Question: {question}"
        )
        try:
            response = self.llm.generate(prompt, system_prompt=None)
            alt = [q.strip() for q in response.strip().split("\n") if q.strip()]
            queries = [question] + alt[:n]
            logger.debug("Multi-query: %d queries generated for %r", len(queries), question[:50])
            return queries
        except RuntimeError as exc:
            logger.warning("Multi-query expansion failed: %s — using original query", exc)
            return [question]

    def _generate_hyde_doc(self, question: str) -> str:
        """Generate a hypothetical answer passage for *question* (HyDE)."""
        instruction = self._qt_cfg.get(
            "hyde_instruction",
            "Write a short passage that directly answers the following question:\n",
        )
        prompt = f"{instruction}\n{question}"
        try:
            hypo = self.llm.generate(prompt, system_prompt=None)
            logger.debug("HyDE doc: %r", hypo[:80])
            return hypo
        except RuntimeError as exc:
            logger.warning("HyDE generation failed: %s — using original query", exc)
            return question

    # ── retrieval with optional query transformation ───────────────────────────

    def _retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Run retrieval with the configured strategy and query transformation.

        The ``retrieval.strategy`` axis selects the candidate generator:

        * ``"hybrid"`` (default) — BM25 + dense fused with RRF.
        * ``"dense"``            — dense vector search only (no BM25 needed).

        Query transformation (``multi_query`` / ``hyde``) is layered on top of
        hybrid retrieval and is independent of these three ablation axes.
        """
        top_k = self._retrieval["top_k_retrieval"]
        strategy = self._retrieval.get("strategy", "hybrid")

        # Dense-only retrieval (ablation axis B = "dense"). Skips BM25 entirely.
        if strategy == "dense" and not self._qt_cfg.get("enabled", False):
            return self.dense_retriever.retrieve(question, k=top_k)

        qt_mode = "none"
        if self._qt_cfg.get("enabled", False):
            qt_mode = self._qt_cfg.get("mode", "none")

        if qt_mode == "multi_query":
            n = self._qt_cfg.get("n_queries", 3)
            queries = self._expand_multi_query(question, n=n)
            ranked_lists = [self.hybrid_retriever.retrieve(q, k=top_k) for q in queries]
            return reciprocal_rank_fusion(ranked_lists, k=60)[:top_k]

        if qt_mode == "hyde":
            hypo_text = self._generate_hyde_doc(question)
            # Dense uses hypothetical-doc embedding; BM25 uses the original question
            hypo_vec   = self.embedder.embed_query_numpy(hypo_text)
            dense_hits = self.vector_index.similarity_search(hypo_vec, modality="text", k=top_k)
            bm25_hits  = self._bm25.retrieve(question, k=top_k)
            return reciprocal_rank_fusion([dense_hits, bm25_hits], k=60)[:top_k]

        # Default: standard hybrid retrieval (no transformation)
        return self.hybrid_retriever.retrieve(question, k=top_k)

    # ── querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """Run a single question through the full optimized pipeline.

        Output format is identical to
        :meth:`~src.pipelines.baseline.BaselinePipeline.query` for direct
        comparison:

        .. code-block:: python

            {
                "answer":             str,
                "retrieved_contexts": list[str],   # top-k_final chunk texts
                "sources":            list[dict],  # chunk metadata
                "latency_ms":         float,
            }

        Args:
            question: Natural-language question.

        Returns:
            Result dict (same schema as baseline).
        """
        t0 = time.perf_counter()

        # 1. Hybrid retrieval (+ optional query transformation)
        candidates = self._retrieve(question)
        logger.debug(
            "Retrieved %d candidates for %r", len(candidates), question[:60]
        )

        # 2. Cross-encoder reranking
        if self._rerank_cfg.get("enabled", True):
            top_k_final = self._rerank_cfg.get("top_k_final", 5)
            final_hits  = self.reranker.rerank(question, candidates, k=top_k_final)
        else:
            top_k_final = self._rerank_cfg.get("top_k_final", 5)
            final_hits  = candidates[:top_k_final]

        context_docs = [doc for doc, _ in final_hits]

        # 3. Build prompt (identical to baseline)
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

        Failures on individual questions are caught and recorded rather than
        propagated, so a bad query does not abort the batch.

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

_singleton: OptimizedPipeline | None = None


def answer(query: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run *query* through the optimized pipeline and return a result dict.

    Same semantics as :func:`~src.pipelines.baseline.answer`: initialises a
    singleton :class:`OptimizedPipeline` on first call and reuses it thereafter.
    Requires the index to have been built via :meth:`OptimizedPipeline.index`.

    Args:
        query:  Natural-language question.
        config: Optional config dict override.  Only applied on first call.

    Returns:
        ``{"answer", "retrieved_contexts", "sources", "latency_ms"}``
    """
    global _singleton
    if _singleton is None:
        _singleton = OptimizedPipeline(config)
    return _singleton.query(query)


# ── __main__ entry point ─────────────────────────────────────────────────────


def __main__() -> None:  # noqa: N807
    """Minimal CLI: ``python -m src.pipelines.optimized 'your question'``."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.pipelines.optimized 'your question'\n"
            "       (use scripts/run_optimized.py for full CLI options)",
            file=sys.stderr,
        )
        sys.exit(1)

    import json as _json
    result = answer(" ".join(sys.argv[1:]))
    print(_json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    __main__()
