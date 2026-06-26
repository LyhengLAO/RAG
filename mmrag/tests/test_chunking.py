"""Tests for src.preprocessing.chunking.

All tests run without network or model downloads:
- FixedChunker and RecursiveChunker only need LangChain (installed dep).
- SemanticChunker tests inject a deterministic mock embedder.

Run fast subset (skip RecursiveChunker which needs langchain_text_splitters):
    pytest tests/test_chunking.py -k "not Recursive"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from langchain_core.documents import Document

from src.preprocessing.chunking import (
    ChunkingStrategy,
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    chunk_documents,
    get_chunker,
)

# ── fixtures and helpers ─────────────────────────────────────────────────────

REQUIRED_META = {"doc_id", "chunk_id", "modality", "char_start", "char_end"}


def _doc(text: str, doc_id: str = "doc_0001", modality: str = "text") -> Document:
    return Document(page_content=text, metadata={"doc_id": doc_id, "modality": modality})


def _lorem(n_chars: int = 1000) -> str:
    """Return a deterministic ASCII text of approximately n_chars."""
    word = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    return (word * (n_chars // len(word) + 1))[:n_chars]


# ── mock embedder for SemanticChunker ────────────────────────────────────────


class _TopicEmbedder:
    """Return [1, 0] for cooking-topic sentences and [0, 1] for space-topic ones.

    The cosine similarity between vectors from different topics is 0 (orthogonal),
    guaranteeing a semantic breakpoint at the topic boundary.
    """

    COOKING_KEYWORDS = {"food", "cook", "recipe", "kitchen", "flavour", "ingredients", "dish", "eat"}
    SPACE_KEYWORDS = {"space", "rocket", "orbit", "planet", "nasa", "astronaut", "galaxy", "star"}

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for sent in sentences:
            words = set(sent.lower().split())
            if words & self.COOKING_KEYWORDS:
                vecs.append(np.array([1.0, 0.0], dtype=np.float32))
            elif words & self.SPACE_KEYWORDS:
                vecs.append(np.array([0.0, 1.0], dtype=np.float32))
            else:
                vecs.append(np.array([0.7, 0.3], dtype=np.float32))  # neutral
        return np.array(vecs, dtype=np.float32)


# Two-topic test text with clear embedding gap in the middle.
COOKING_BLOCK = (
    "To make a great dish, fresh ingredients are essential. "
    "Cook the vegetables in a hot pan with olive oil. "
    "A well-balanced recipe balances flavour and nutrition. "
    "The kitchen is the heart of every home."
)
SPACE_BLOCK = (
    "NASA launched a rocket towards the outer planet last week. "
    "Astronauts living in orbit experience zero gravity every day. "
    "The galaxy contains billions of star systems and black holes. "
    "Space exploration pushes the boundaries of human knowledge."
)
TWO_TOPIC_TEXT = COOKING_BLOCK + " " + SPACE_BLOCK


# ── FixedChunker ─────────────────────────────────────────────────────────────


class TestFixedChunker:
    def test_exact_chunk_count_no_overlap(self) -> None:
        text = "x" * 1000
        chunks = FixedChunker(chunk_size=200, chunk_overlap=0).chunk([_doc(text)])
        assert len(chunks) == 5

    def test_chunk_count_with_overlap(self) -> None:
        # step = 200 - 50 = 150; ceil(1000 / 150) = 7
        text = "x" * 1000
        chunks = FixedChunker(chunk_size=200, chunk_overlap=50).chunk([_doc(text)])
        assert len(chunks) == 7

    def test_overlap_content_is_shared(self) -> None:
        text = "abcdefghij" * 20  # 200 chars, known content
        overlap = 20
        chunks = FixedChunker(chunk_size=50, chunk_overlap=overlap).chunk([_doc(text)])
        assert len(chunks) >= 2
        tail_of_first = chunks[0].page_content[-overlap:]
        head_of_second = chunks[1].page_content[:overlap]
        assert tail_of_first == head_of_second

    def test_last_chunk_is_tail_of_text(self) -> None:
        text = _lorem(300)
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk([_doc(text)])
        assert text.endswith(chunks[-1].page_content)

    def test_all_chunks_together_cover_text(self) -> None:
        text = _lorem(500)
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk([_doc(text)])
        reconstructed = "".join(c.page_content for c in chunks)
        assert reconstructed == text

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            FixedChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_document_returns_no_chunks(self) -> None:
        chunks = FixedChunker().chunk([_doc("   ")])
        assert chunks == []

    def test_multiple_documents_independent(self) -> None:
        docs = [_doc("a" * 300, doc_id=f"doc_{i:04d}") for i in range(3)]
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk(docs)
        ids = {c.metadata["doc_id"] for c in chunks}
        assert ids == {"doc_0000", "doc_0001", "doc_0002"}


# ── RecursiveChunker ─────────────────────────────────────────────────────────


class TestRecursiveChunker:
    def test_no_chunk_exceeds_chunk_size(self) -> None:
        text = _lorem(2000)
        chunks = RecursiveChunker(chunk_size=200, chunk_overlap=20).chunk([_doc(text)])
        oversized = [c for c in chunks if len(c.page_content) > 250]  # 25 % tolerance
        assert not oversized, f"{len(oversized)} chunks exceeded size limit"

    def test_at_least_one_chunk_produced(self) -> None:
        chunks = RecursiveChunker(chunk_size=200, chunk_overlap=20).chunk([_doc(_lorem(1000))])
        assert len(chunks) >= 1

    def test_empty_document_returns_no_chunks(self) -> None:
        chunks = RecursiveChunker().chunk([_doc("")])
        assert chunks == []

    def test_char_start_is_non_negative(self) -> None:
        chunks = RecursiveChunker(chunk_size=100, chunk_overlap=10).chunk([_doc(_lorem(500))])
        assert all(c.metadata["char_start"] >= 0 for c in chunks)

    def test_char_end_greater_than_char_start(self) -> None:
        chunks = RecursiveChunker(chunk_size=100, chunk_overlap=10).chunk([_doc(_lorem(500))])
        assert all(c.metadata["char_end"] > c.metadata["char_start"] for c in chunks)


# ── SemanticChunker (mock embedder) ──────────────────────────────────────────


class TestSemanticChunker:
    @pytest.fixture()
    def chunker(self) -> SemanticChunker:
        return SemanticChunker(
            chunk_size=2000,  # large so no recursive fallback
            breakpoint_threshold=0.3,  # cosine < 0.3 → split (orthogonal topics give 0.0)
            _embedder=_TopicEmbedder(),
        )

    def test_splits_at_topic_boundary(self, chunker: SemanticChunker) -> None:
        chunks = chunker.chunk([_doc(TWO_TOPIC_TEXT)])
        # Must produce at least 2 chunks (one per topic)
        assert len(chunks) >= 2

    def test_cooking_and_space_in_different_chunks(self, chunker: SemanticChunker) -> None:
        chunks = chunker.chunk([_doc(TWO_TOPIC_TEXT)])
        cooking_chunks = [c for c in chunks if "cook" in c.page_content.lower() or "food" in c.page_content.lower()]
        space_chunks = [c for c in chunks if "space" in c.page_content.lower() or "rocket" in c.page_content.lower()]
        assert cooking_chunks, "No cooking chunk found"
        assert space_chunks, "No space chunk found"
        # They should not be the same chunk
        assert cooking_chunks[0].page_content != space_chunks[0].page_content

    def test_single_sentence_returns_one_chunk(self, chunker: SemanticChunker) -> None:
        chunks = chunker.chunk([_doc("This is a single sentence.")])
        assert len(chunks) == 1

    def test_empty_document_returns_no_chunks(self, chunker: SemanticChunker) -> None:
        chunks = chunker.chunk([_doc("")])
        assert chunks == []


# ── Semantic vs Recursive boundary comparison ─────────────────────────────────


def test_semantic_vs_recursive_different_split_positions() -> None:
    """Core regression: semantic and recursive chunkers must split TWO_TOPIC_TEXT differently.

    The recursive chunker splits purely by character count and will NOT align
    with the topic boundary.  The semantic chunker, guided by embedding
    dissimilarity, should split *at* the topic transition.
    """
    # Use a chunk_size that forces at least 2 recursive splits but doesn't
    # align with the two-topic boundary.
    chunk_size = len(COOKING_BLOCK) + 30  # crossing the topic boundary

    recursive_chunks = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=0).chunk(
        [_doc(TWO_TOPIC_TEXT)]
    )
    semantic_chunks = SemanticChunker(
        chunk_size=9999,
        breakpoint_threshold=0.3,
        _embedder=_TopicEmbedder(),
    ).chunk([_doc(TWO_TOPIC_TEXT)])

    recursive_starts = {c.metadata["char_start"] for c in recursive_chunks}
    semantic_starts = {c.metadata["char_start"] for c in semantic_chunks}

    # They must differ somewhere
    assert recursive_starts != semantic_starts, (
        "Recursive and semantic chunkers produced identical split positions — "
        "the semantic chunker is not using embedding information."
    )

    # Semantic should produce exactly 2 chunks (one per topic)
    assert len(semantic_chunks) == 2, (
        f"Expected 2 semantic chunks (one per topic), got {len(semantic_chunks)}"
    )


# ── Metadata completeness ─────────────────────────────────────────────────────


class TestMetadata:
    @pytest.mark.parametrize(
        "chunker",
        [
            FixedChunker(chunk_size=100, chunk_overlap=10),
            RecursiveChunker(chunk_size=100, chunk_overlap=10),
            SemanticChunker(
                chunk_size=9999,
                breakpoint_threshold=0.3,
                _embedder=_TopicEmbedder(),
            ),
        ],
        ids=["fixed", "recursive", "semantic"],
    )
    def test_required_fields_present(self, chunker: object) -> None:
        chunks = chunker.chunk([_doc(_lorem(400))])  # type: ignore[union-attr]
        assert chunks, "No chunks produced"
        for chunk in chunks:
            missing = REQUIRED_META - set(chunk.metadata.keys())
            assert not missing, f"Chunk missing metadata fields: {missing}"

    def test_chunk_ids_unique_within_document(self) -> None:
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk([_doc(_lorem(500))])
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids within document"

    def test_doc_id_propagated_correctly(self) -> None:
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk(
            [_doc(_lorem(300), doc_id="my_doc")]
        )
        assert all(c.metadata["doc_id"] == "my_doc" for c in chunks)

    def test_modality_propagated(self) -> None:
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk(
            [_doc(_lorem(300), modality="audio")]
        )
        assert all(c.metadata["modality"] == "audio" for c in chunks)

    def test_char_span_bounds_for_fixed(self) -> None:
        text = _lorem(300)
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk([_doc(text)])
        for chunk in chunks:
            s, e = chunk.metadata["char_start"], chunk.metadata["char_end"]
            assert 0 <= s < e <= len(text), f"Invalid span [{s}, {e}] for text len {len(text)}"

    def test_fixed_chunk_content_matches_span(self) -> None:
        text = _lorem(300)
        chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk([_doc(text)])
        for chunk in chunks:
            s, e = chunk.metadata["char_start"], chunk.metadata["char_end"]
            assert chunk.page_content == text[s:e]


# ── get_chunker factory ───────────────────────────────────────────────────────


class TestGetChunker:
    def test_fixed_strategy(self) -> None:
        assert isinstance(get_chunker("fixed"), FixedChunker)

    def test_recursive_strategy(self) -> None:
        assert isinstance(get_chunker("recursive"), RecursiveChunker)

    def test_semantic_strategy(self) -> None:
        assert isinstance(get_chunker("semantic"), SemanticChunker)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            get_chunker("unknown_xyz")

    def test_chunk_documents_wrapper(self) -> None:
        chunks = chunk_documents([_doc(_lorem(300))], strategy="fixed", chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 3
