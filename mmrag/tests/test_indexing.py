"""Tests for src.indexing.vector_store — round-trip upsert→query + idempotency.

All tests use ``chromadb.EphemeralClient()`` (in-memory, no disk writes) and
pre-computed numpy embeddings (no model downloads).

The key insight for deterministic nearest-neighbour tests: use *orthogonal*
one-hot vectors.  Querying with vector[i] must return chunk[i] as the unique
top hit because all other cosine similarities are exactly 0.

Run:
    pytest tests/test_indexing.py -v
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from langchain_core.documents import Document

from src.indexing.vector_store import VectorStoreIndex, _stable_id, _to_chroma_meta

# ── helpers ───────────────────────────────────────────────────────────────────

N_CHUNKS = 5
EMBED_DIM = N_CHUNKS  # one-hot dim == number of chunks → exactly orthogonal
EMBEDDER  = "test-embedder"
MODALITY  = "text"


def _ortho_embeddings(n: int = N_CHUNKS) -> np.ndarray:
    """Return an (n, n) identity matrix: each row is orthogonal to all others."""
    return np.eye(n, dtype=np.float32)


def _make_chunks(n: int = N_CHUNKS) -> list[Document]:
    return [
        Document(
            page_content=f"Content of chunk {i}",
            metadata={
                "doc_id":   f"doc_{i:04d}",
                "chunk_id": f"doc_{i:04d}::text::0000",
                "modality": MODALITY,
                "source":   "test/dataset",
                "license":  "CC-BY-4.0",
                "char_start": i * 100,
                "char_end":   i * 100 + 80,
            },
        )
        for i in range(n)
    ]


@pytest.fixture()
def index() -> VectorStoreIndex:
    """Ephemeral VectorStoreIndex backed by an in-memory Chroma client."""
    import chromadb  # noqa: PLC0415
    return VectorStoreIndex(_client=chromadb.EphemeralClient())


@pytest.fixture()
def populated_index(index: VectorStoreIndex) -> tuple[VectorStoreIndex, list[Document], np.ndarray]:
    """Index with N_CHUNKS orthogonal chunks already upserted."""
    chunks = _make_chunks()
    embeddings = _ortho_embeddings()
    index.upsert(chunks, embeddings, modality=MODALITY, embedder_name=EMBEDDER)
    return index, chunks, embeddings


# ── stable ID ─────────────────────────────────────────────────────────────────


class TestStableId:
    def test_deterministic(self) -> None:
        assert _stable_id("a", "b", "c") == _stable_id("a", "b", "c")

    def test_length_is_16(self) -> None:
        assert len(_stable_id("doc_0001", "doc_0001::text::0000", "bge-small")) == 16

    def test_different_inputs_produce_different_ids(self) -> None:
        id1 = _stable_id("doc_0001", "chunk_0", "emb")
        id2 = _stable_id("doc_0002", "chunk_0", "emb")
        id3 = _stable_id("doc_0001", "chunk_0", "other-emb")
        assert id1 != id2
        assert id1 != id3

    def test_hex_characters_only(self) -> None:
        sid = _stable_id("x", "y", "z")
        assert all(c in "0123456789abcdef" for c in sid)


# ── metadata sanitisation ─────────────────────────────────────────────────────


class TestToChromaMeta:
    def test_scalars_pass_through(self) -> None:
        meta = {"a": "str", "b": 1, "c": 1.5, "d": True}
        assert _to_chroma_meta(meta) == meta

    def test_list_is_json_encoded(self) -> None:
        result = _to_chroma_meta({"captions": ["cap1", "cap2"]})
        assert isinstance(result["captions"], str)
        assert "cap1" in result["captions"]

    def test_none_becomes_empty_string(self) -> None:
        assert _to_chroma_meta({"x": None})["x"] == ""

    def test_non_scalar_dict_is_dropped(self) -> None:
        result = _to_chroma_meta({"nested": {"a": 1}})
        assert "nested" not in result

    def test_empty_dict_stays_empty(self) -> None:
        assert _to_chroma_meta({}) == {}


# ── upsert ────────────────────────────────────────────────────────────────────


class TestUpsert:
    def test_count_increases_after_upsert(self, index: VectorStoreIndex) -> None:
        chunks = _make_chunks(3)
        index.upsert(chunks, _ortho_embeddings(3)[:3], MODALITY, EMBEDDER)
        assert index.count(MODALITY) == 3

    def test_upsert_returns_stable_ids(self, index: VectorStoreIndex) -> None:
        chunks = _make_chunks(2)
        ids = index.upsert(chunks, _ortho_embeddings(2), MODALITY, EMBEDDER)
        assert len(ids) == 2
        expected = [
            _stable_id(chunks[i].metadata["doc_id"], chunks[i].metadata["chunk_id"], EMBEDDER)
            for i in range(2)
        ]
        assert ids == expected

    def test_upsert_is_idempotent(self, index: VectorStoreIndex) -> None:
        """Upserting the same chunks twice must not duplicate entries."""
        chunks = _make_chunks()
        embs = _ortho_embeddings()
        index.upsert(chunks, embs, MODALITY, EMBEDDER)
        index.upsert(chunks, embs, MODALITY, EMBEDDER)
        assert index.count(MODALITY) == N_CHUNKS  # still N, not 2×N

    def test_upsert_empty_list_is_noop(self, index: VectorStoreIndex) -> None:
        ids = index.upsert([], np.empty((0, EMBED_DIM), dtype=np.float32), MODALITY, EMBEDDER)
        assert ids == []
        assert index.count(MODALITY) == 0

    def test_mismatched_lengths_raise(self, index: VectorStoreIndex) -> None:
        chunks = _make_chunks(3)
        with pytest.raises(ValueError, match="same length"):
            index.upsert(chunks, _ortho_embeddings(2), MODALITY, EMBEDDER)

    def test_different_modalities_are_isolated(self, index: VectorStoreIndex) -> None:
        chunks = _make_chunks(2)
        index.upsert(chunks, np.eye(2, dtype=np.float32), "text", EMBEDDER)
        index.upsert(chunks, np.eye(2, dtype=np.float32), "image_clip", EMBEDDER)
        assert index.count("text") == 2
        assert index.count("image_clip") == 2

    def test_invalid_modality_raises(self, index: VectorStoreIndex) -> None:
        with pytest.raises(ValueError, match="Unknown modality"):
            index.upsert(_make_chunks(1), np.eye(1, dtype=np.float32), "banana", EMBEDDER)


# ── round-trip: upsert → similarity_search ────────────────────────────────────


class TestRoundTrip:
    def test_query_returns_exact_match(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        """Querying with vector[i] must return chunk[i] as the top result."""
        idx, chunks, embs = populated_index
        for i in range(N_CHUNKS):
            results = idx.similarity_search(embs[i], modality=MODALITY, k=1)
            assert len(results) == 1
            doc, score = results[0]
            assert doc.page_content == chunks[i].page_content, (
                f"Expected chunk[{i}] but got: {doc.page_content!r}"
            )
            assert pytest.approx(score, abs=1e-4) == 1.0

    def test_top_k_results_ordered_by_score(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, chunks, embs = populated_index
        results = idx.similarity_search(embs[0], modality=MODALITY, k=N_CHUNKS)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by descending score"

    def test_k_limits_results(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, chunks, embs = populated_index
        results = idx.similarity_search(embs[0], modality=MODALITY, k=2)
        assert len(results) == 2

    def test_score_threshold_filters(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        """With orthogonal vectors only the exact match scores 1.0; all others ≈0.
        A threshold of 0.9 should keep only the exact match."""
        idx, chunks, embs = populated_index
        results = idx.similarity_search(embs[0], modality=MODALITY, k=N_CHUNKS, score_threshold=0.9)
        assert len(results) == 1
        assert results[0][1] > 0.9

    def test_empty_collection_returns_empty(self, index: VectorStoreIndex) -> None:
        results = index.similarity_search(
            np.ones(EMBED_DIM, dtype=np.float32), modality=MODALITY, k=5
        )
        assert results == []

    def test_result_score_is_float(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, _, embs = populated_index
        _, score = idx.similarity_search(embs[0], modality=MODALITY, k=1)[0]
        assert isinstance(score, float)


# ── metadata preservation ─────────────────────────────────────────────────────


class TestMetadata:
    def test_required_fields_survive_round_trip(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, _, embs = populated_index
        doc, _ = idx.similarity_search(embs[0], modality=MODALITY, k=1)[0]
        for field in ("doc_id", "chunk_id", "modality", "source", "license"):
            assert field in doc.metadata, f"Field {field!r} missing from retrieved metadata"

    def test_scalar_metadata_values_unchanged(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, chunks, embs = populated_index
        doc, _ = idx.similarity_search(embs[0], modality=MODALITY, k=1)[0]
        assert doc.metadata["doc_id"]   == chunks[0].metadata["doc_id"]
        assert doc.metadata["chunk_id"] == chunks[0].metadata["chunk_id"]
        assert doc.metadata["source"]   == "test/dataset"
        assert doc.metadata["license"]  == "CC-BY-4.0"

    def test_integer_metadata_preserved(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, chunks, embs = populated_index
        doc, _ = idx.similarity_search(embs[0], modality=MODALITY, k=1)[0]
        assert doc.metadata["char_start"] == chunks[0].metadata["char_start"]

    def test_embedding_model_field_injected(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, _, embs = populated_index
        doc, _ = idx.similarity_search(embs[0], modality=MODALITY, k=1)[0]
        assert doc.metadata.get("embedding_model") == EMBEDDER

    def test_page_content_preserved(
        self, populated_index: tuple[VectorStoreIndex, list[Document], np.ndarray]
    ) -> None:
        idx, chunks, embs = populated_index
        for i, emb in enumerate(embs):
            doc, _ = idx.similarity_search(emb, modality=MODALITY, k=1)[0]
            assert doc.page_content == chunks[i].page_content


# ── admin / utilities ─────────────────────────────────────────────────────────


class TestAdmin:
    def test_count_zero_before_upsert(self, index: VectorStoreIndex) -> None:
        assert index.count(MODALITY) == 0

    def test_delete_collection_resets_count(self, index: VectorStoreIndex) -> None:
        index.upsert(_make_chunks(3), _ortho_embeddings(3)[:3], MODALITY, EMBEDDER)
        assert index.count(MODALITY) == 3
        index.delete_collection(MODALITY)
        assert index.count(MODALITY) == 0

    def test_list_modalities_after_upsert(self, index: VectorStoreIndex) -> None:
        index.upsert(_make_chunks(2), np.eye(2, dtype=np.float32), "text", EMBEDDER)
        index.upsert(_make_chunks(2), np.eye(2, dtype=np.float32), "image_clip", EMBEDDER)
        modalities = index.list_modalities()
        assert "text" in modalities
        assert "image_clip" in modalities

    def test_persist_is_safe_to_call(self, index: VectorStoreIndex) -> None:
        index.persist()  # should not raise

    def test_get_collection_unknown_raises(self, index: VectorStoreIndex) -> None:
        with pytest.raises(ValueError, match="Unknown modality"):
            index.get_collection("video")
