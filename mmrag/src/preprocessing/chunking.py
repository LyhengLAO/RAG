"""Text chunking strategies with a common Chunker interface.

Three implementations selectable by config key
-----------------------------------------------
``fixed``     → FixedChunker      : sliding window of exactly *chunk_size* chars.
``recursive`` → RecursiveChunker  : LangChain RecursiveCharacterTextSplitter.
``semantic``  → SemanticChunker   : home-built cosine-similarity breakpoint splitter
                                    (no langchain_experimental dependency).

Every output Document carries these metadata keys in addition to whatever the
source Document already had:

    doc_id      str   stable identifier of the source document
    chunk_id    str   "{doc_id}::text::{index:04d}"
    modality    str   forwarded from source metadata (default "text")
    char_start  int   byte offset of chunk start in source text
    char_end    int   byte offset of chunk end in source text
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── device helper ─────────────────────────────────────────────────────────────


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415

        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except (ImportError, AttributeError):
        return "cpu"


# ── sentence splitting ────────────────────────────────────────────────────────


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into (sentence_text, char_start, char_end) triples.

    Uses simple punctuation heuristics; no external NLP library needed.
    """
    results: list[tuple[str, int, int]] = []
    parts = re.split(r"(?<=[.!?])\s+", text)
    pos = 0
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        idx = text.find(part_stripped, pos)
        if idx == -1:
            idx = pos
        end = idx + len(part_stripped)
        results.append((part_stripped, idx, end))
        pos = end

    if not results and text.strip():
        s = text.strip()
        idx = text.find(s)
        results.append((s, idx, idx + len(s)))

    return results


# ── embedder protocol (for dependency injection in tests) ────────────────────


@runtime_checkable
class _Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        ...


# ── abstract base ────────────────────────────────────────────────────────────


class Chunker(ABC):
    """Common interface for all chunking strategies."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split *documents* into chunks and return the flat list.

        Args:
            documents: Source documents; each must have ``page_content``.

        Returns:
            Flat list of chunk Documents with enriched metadata.
        """

    # ── shared metadata helper ────────────────────────────────────────────────

    @staticmethod
    def _make_metadata(
        source_doc: Document,
        chunk_idx: int,
        char_start: int,
        char_end: int,
    ) -> dict[str, Any]:
        src = source_doc.metadata
        doc_id: str = src.get("doc_id") or src.get("id") or "unknown"
        modality: str = src.get("modality", "text")
        chunk_id = f"{doc_id}::{modality}::{chunk_idx:04d}"
        return {
            **src,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "modality": modality,
            "char_start": char_start,
            "char_end": char_end,
        }


# ── strategy enum ─────────────────────────────────────────────────────────────


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# ── FixedChunker ──────────────────────────────────────────────────────────────


class FixedChunker(Chunker):
    """Sliding-window character splitter with exact size and overlap.

    Args:
        chunk_size:    Number of characters per chunk.
        chunk_overlap: Number of trailing characters shared with the next chunk.
                       Must be strictly less than *chunk_size*.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Document]:
        results: list[Document] = []
        for doc in documents:
            text = doc.page_content
            if not text.strip():
                logger.debug("FixedChunker: skipping empty document")
                continue
            chunk_idx = 0
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                if chunk_text.strip():
                    meta = self._make_metadata(doc, chunk_idx, start, end)
                    results.append(Document(page_content=chunk_text, metadata=meta))
                    chunk_idx += 1
                if end >= len(text):
                    break
                start += self._step
        return results


# ── RecursiveChunker ─────────────────────────────────────────────────────────


class RecursiveChunker(Chunker):
    """LangChain RecursiveCharacterTextSplitter with char-span tracking.

    Respects natural text separators (paragraphs → sentences → words → chars)
    before falling back to hard character cuts.

    Args:
        chunk_size:    Soft character limit per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Document]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]  # noqa: PLC0415

        # add_start_index injects metadata["start_index"] with the char offset
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        results: list[Document] = []
        for doc in documents:
            if not doc.page_content.strip():
                continue
            raw_chunks = splitter.split_documents([doc])
            for chunk_idx, raw_chunk in enumerate(raw_chunks):
                char_start: int = raw_chunk.metadata.get("start_index", 0)
                char_end: int = char_start + len(raw_chunk.page_content)
                meta = self._make_metadata(doc, chunk_idx, char_start, char_end)
                results.append(Document(page_content=raw_chunk.page_content, metadata=meta))
        return results


# ── SemanticChunker ───────────────────────────────────────────────────────────


class SemanticChunker(Chunker):
    """Split at cosine-similarity breakpoints between consecutive sentences.

    Algorithm
    ---------
    1. Split document into sentences.
    2. Embed all sentences in one batch (sentence-transformers).
    3. Compute cosine similarity between each pair of adjacent sentences.
    4. Mark a breakpoint wherever similarity falls below the threshold.
       The threshold is either fixed (``breakpoint_threshold``) or computed
       as the *breakpoint_percentile*-th percentile of all pairwise similarities.
    5. Group sentences between breakpoints into chunks.

    If a resulting chunk still exceeds *max_chunk_chars*, it is recursively
    split with a ``RecursiveChunker`` to enforce a loose upper bound.

    Args:
        chunk_size:           Soft character limit; chunks exceeding 2× this are
                              recursively split. Defaults to 512.
        embedding_model:      sentence-transformers model name.
        breakpoint_threshold: Fixed cosine-similarity threshold.  Mutually
                              exclusive with *breakpoint_percentile*.
        breakpoint_percentile: Use the Nth-percentile of pairwise similarities
                               as the threshold (default 25 = split at the most
                               dissimilar 25 % of sentence transitions).
        device:               ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        _embedder:            Inject a custom embedder (used in tests to avoid
                              model downloads; must expose ``.encode()``).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_threshold: float | None = None,
        breakpoint_percentile: float = 25.0,
        device: str | None = None,
        _embedder: _Embedder | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.breakpoint_percentile = breakpoint_percentile
        self.device = device or _auto_device()
        self._injected_embedder = _embedder
        self._model: Any = None  # lazy-loaded

    # ── lazy model ───────────────────────────────────────────────────────────

    def _get_embedder(self) -> _Embedder:
        if self._injected_embedder is not None:
            return self._injected_embedder
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("SemanticChunker: loading %s on %s", self.embedding_model, self.device)
            self._model = SentenceTransformer(self.embedding_model, device=self.device)
        return self._model  # type: ignore[return-value]

    # ── chunking ─────────────────────────────────────────────────────────────

    def chunk(self, documents: list[Document]) -> list[Document]:
        results: list[Document] = []
        for doc in documents:
            results.extend(self._split_one(doc))
        return results

    def _split_one(self, doc: Document) -> list[Document]:
        text = doc.page_content
        sentence_data = _sentence_spans(text)

        if not sentence_data:
            return []

        if len(sentence_data) == 1:
            s, start, end = sentence_data[0]
            meta = self._make_metadata(doc, 0, start, end)
            return [Document(page_content=s, metadata=meta)]

        sentences = [s for s, _, _ in sentence_data]
        embedder = self._get_embedder()
        embeddings: np.ndarray = embedder.encode(sentences, show_progress_bar=False)

        # Normalise rows for cosine via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        sims: list[float] = [
            float(normed[i] @ normed[i + 1]) for i in range(len(normed) - 1)
        ]

        # Compute threshold
        if self.breakpoint_threshold is not None:
            threshold = self.breakpoint_threshold
        else:
            threshold = float(np.percentile(sims, self.breakpoint_percentile))

        # Build split boundaries
        boundaries = [0] + [i + 1 for i, s in enumerate(sims) if s < threshold] + [len(sentence_data)]

        chunks: list[Document] = []
        for chunk_idx, (b_start, b_end) in enumerate(zip(boundaries, boundaries[1:])):
            group = sentence_data[b_start:b_end]
            chunk_text = " ".join(sent for sent, _, _ in group)
            char_start = group[0][1]
            char_end = group[-1][2]

            # Recursively split oversized semantic chunks
            if len(chunk_text) > 2 * self.chunk_size:
                sub_doc = Document(page_content=chunk_text, metadata=doc.metadata)
                sub_chunks = RecursiveChunker(
                    chunk_size=self.chunk_size, chunk_overlap=0
                ).chunk([sub_doc])
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    # Re-anchor char offsets to the original document
                    offset = char_start + sub_chunk.metadata.get("char_start", 0)
                    sub_end = char_start + sub_chunk.metadata.get("char_end", len(chunk_text))
                    meta = self._make_metadata(doc, chunk_idx * 1000 + sub_idx, offset, sub_end)
                    chunks.append(Document(page_content=sub_chunk.page_content, metadata=meta))
            else:
                meta = self._make_metadata(doc, chunk_idx, char_start, char_end)
                chunks.append(Document(page_content=chunk_text, metadata=meta))

        return chunks


# ── factory ───────────────────────────────────────────────────────────────────


def get_chunker(
    strategy: ChunkingStrategy | str = ChunkingStrategy.FIXED,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs: Any,
) -> Chunker:
    """Instantiate the right :class:`Chunker` for *strategy*.

    Args:
        strategy:      One of ``"fixed"``, ``"recursive"``, ``"semantic"``.
        chunk_size:    Passed to the chunker constructor.
        chunk_overlap: Passed to the chunker constructor.
        **kwargs:      Extra keyword arguments forwarded to the chunker.

    Returns:
        A ready-to-use :class:`Chunker` instance.

    Raises:
        ValueError: If *strategy* is not recognised.
    """
    key = ChunkingStrategy(strategy) if not isinstance(strategy, ChunkingStrategy) else strategy
    if key == ChunkingStrategy.FIXED:
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if key == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if key == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


# ── convenience wrapper (backward compat with pipeline stubs) ─────────────────


def chunk_documents(
    documents: list[Document],
    strategy: ChunkingStrategy | str = ChunkingStrategy.FIXED,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs: Any,
) -> list[Document]:
    """Split *documents* with *strategy* and return the flat chunk list.

    Thin wrapper around :func:`get_chunker` for one-liner use in pipelines.

    Args:
        documents:     Input documents to split.
        strategy:      Chunking strategy key.
        chunk_size:    Target chunk size in characters.
        chunk_overlap: Character overlap between consecutive chunks.
        **kwargs:      Forwarded to the underlying :class:`Chunker`.

    Returns:
        Flat list of chunk :class:`~langchain_core.documents.Document` objects.
    """
    return get_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs).chunk(
        documents
    )
