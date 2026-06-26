"""Tests for src.retrieval: BM25Retriever, HybridRetriever, CrossEncoderReranker.

Design
------
No model downloads, no disk I/O, no ChromaDB instance required.

- BM25Retriever tests: pure Python + rank_bm25 (lightweight, no GPU).
- HybridRetriever tests: real BM25 on a tiny corpus + a **mock** dense retriever
  that returns a fixed result regardless of the query.  This isolates the RRF
  fusion logic from embedding quality.
- CrossEncoderReranker tests: the mock CrossEncoder model is injected directly
  onto ``reranker._model`` — no HTTP call, no file download.

Two-source invariant
--------------------
The key correctness property of hybrid retrieval:

    A document findable ONLY by exact keyword match (BM25 path)
    AND a document findable ONLY by semantic similarity (dense path)
    must BOTH appear in the hybrid output after RRF fusion.

Tests ``test_finds_keyword_only_doc_not_in_dense`` and
``test_finds_semantic_doc_not_in_bm25`` verify each direction independently;
``test_retrieves_both_keyword_and_semantic_doc`` asserts both at once.
"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse_bm25 import BM25Retriever


# ── shared helpers ────────────────────────────────────────────────────────────


def _doc(content: str, doc_id: str, modality: str = "text") -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id":   doc_id,
            "chunk_id": f"{doc_id}::text::0000",
            "modality": modality,
            "source":   "test/corpus",
            "license":  "CC0-1.0",
        },
    )


# ── two-source corpus ─────────────────────────────────────────────────────────
#
# DOC_KEYWORD  — contains the word "Paris" explicitly.
#                BM25 ranks it high for the query "Paris sightseeing".
#                The mock dense retriever does NOT return it.
#
# DOC_SEMANTIC — describes France's capital without using the word "Paris".
#                BM25 score is ZERO for the query (keyword absent).
#                The mock dense retriever always returns it as top hit.
#
# DOC_NOISE    — unrelated topic; neither retriever should prefer it.

DOC_KEYWORD  = _doc(
    "Paris exposition universelle 1889. Gustave Eiffel. "
    "Boulangerie croissant bistrot café parisien.",
    "doc_keyword",
)
DOC_SEMANTIC = _doc(
    "The capital city of France lies along the river Seine in Western Europe. "
    "A major hub of art, culture, and fashion.",
    "doc_semantic",
)
DOC_NOISE = _doc(
    "Gradient descent optimises neural network weights by computing the partial "
    "derivative of the loss with respect to each parameter.",
    "doc_noise",
)

CORPUS = [DOC_KEYWORD, DOC_SEMANTIC, DOC_NOISE]

# Query that BM25 handles well (exact keyword) but a frozen dense embedding misses
KEYWORD_QUERY = "Paris sightseeing tours"


class _MockDenseRetriever:
    """Always returns [(DOC_SEMANTIC, 0.95), (DOC_NOISE, 0.30)] regardless of query.

    DOC_KEYWORD is intentionally absent — it can only be found via BM25.
    """

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        results = [(DOC_SEMANTIC, 0.95), (DOC_NOISE, 0.30)]
        return results[:k]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def bm25() -> BM25Retriever:
    return BM25Retriever(CORPUS)


@pytest.fixture()
def hybrid(bm25: BM25Retriever) -> HybridRetriever:
    return HybridRetriever(
        dense=_MockDenseRetriever(),
        sparse=bm25,
        rrf_k=60,
        top_k_per_retriever=10,
    )


# ── BM25Retriever ─────────────────────────────────────────────────────────────


class TestBM25Retriever:
    def test_keyword_doc_ranked_first_for_exact_match(self, bm25: BM25Retriever) -> None:
        results = bm25.retrieve("Paris", k=3)
        assert results, "Expected non-empty results"
        top_doc, _ = results[0]
        assert top_doc.metadata["doc_id"] == "doc_keyword"

    def test_semantic_doc_excluded_for_paris_query(self, bm25: BM25Retriever) -> None:
        """The word 'Paris' does not appear in DOC_SEMANTIC — BM25 score must be zero."""
        results = bm25.retrieve("Paris", k=len(CORPUS))
        returned_ids = {doc.metadata["doc_id"] for doc, _ in results}
        assert "doc_semantic" not in returned_ids, (
            "doc_semantic should score 0 for 'Paris' query and be excluded"
        )

    def test_only_positive_scores_returned(self, bm25: BM25Retriever) -> None:
        results = bm25.retrieve("Paris", k=10)
        assert all(score > 0.0 for _, score in results)

    def test_results_sorted_descending(self, bm25: BM25Retriever) -> None:
        results = bm25.retrieve("Paris capital France", k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_raises(self) -> None:
        retriever = BM25Retriever([])
        with pytest.raises(RuntimeError, match="index not built"):
            retriever.retrieve("anything")

    def test_k_limits_results(self, bm25: BM25Retriever) -> None:
        results = bm25.retrieve("Paris", k=1)
        assert len(results) <= 1

    def test_corpus_size(self, bm25: BM25Retriever) -> None:
        assert bm25.corpus_size == len(CORPUS)

    def test_rebuild_replaces_corpus(self, bm25: BM25Retriever) -> None:
        new_doc = _doc("A story about mountains and glaciers.", "doc_mountain")
        bm25.rebuild([new_doc])
        assert bm25.corpus_size == 1
        results = bm25.retrieve("mountains", k=1)
        assert results[0][0].metadata["doc_id"] == "doc_mountain"

    def test_save_and_load_roundtrip(
        self, bm25: BM25Retriever, tmp_path: pytest.TempPathFactory
    ) -> None:
        path = tmp_path / "bm25_index.pkl"
        bm25.save(path)
        loaded = BM25Retriever.load(path)
        assert loaded.corpus_size == bm25.corpus_size
        orig_ids    = [d.metadata["doc_id"] for d, _ in bm25.retrieve("Paris", k=3)]
        loaded_ids  = [d.metadata["doc_id"] for d, _ in loaded.retrieve("Paris", k=3)]
        assert orig_ids == loaded_ids


# ── RRF (module-level function) ───────────────────────────────────────────────


class TestRRF:
    def _ranked(self, items: list[tuple[str, float]]) -> list[tuple[Document, float]]:
        return [(_doc(f"Content {did}", did), score) for did, score in items]

    def test_single_list_preserves_order(self) -> None:
        ranked = self._ranked([("a", 1.0), ("b", 0.8)])
        result = reciprocal_rank_fusion([ranked], k=60)
        assert [d.metadata["doc_id"] for d, _ in result] == ["a", "b"]

    def test_union_of_two_disjoint_lists(self) -> None:
        list_a = self._ranked([("a", 1.0), ("b", 0.5)])
        list_b = self._ranked([("c", 1.0), ("d", 0.5)])
        result = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert {d.metadata["doc_id"] for d, _ in result} == {"a", "b", "c", "d"}

    def test_shared_doc_accumulates_higher_score(self) -> None:
        """A doc ranked #1 in both lists must outscore docs in only one list."""
        shared  = _doc("Shared content", "shared")
        only_a  = _doc("Only in A",      "only_a")
        only_b  = _doc("Only in B",      "only_b")

        result = reciprocal_rank_fusion(
            [[(shared, 1.0), (only_a, 0.8)],
             [(shared, 1.0), (only_b, 0.8)]],
            k=60,
        )
        assert result[0][0].metadata["doc_id"] == "shared"

    def test_rrf_scores_are_positive(self) -> None:
        ranked = self._ranked([("x", 0.9), ("y", 0.1)])
        result = reciprocal_rank_fusion([ranked], k=60)
        assert all(score > 0 for _, score in result)

    def test_empty_lists_return_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_deduplication_by_chunk_id(self) -> None:
        """The same chunk_id appearing in both lists must produce exactly one entry."""
        doc = _doc("Duplicate content", "dup")
        result = reciprocal_rank_fusion([[(doc, 1.0)], [(doc, 0.5)]], k=60)
        assert len(result) == 1

    def test_smaller_k_gives_wider_score_spread(self) -> None:
        """k=1 gives rank-1 much higher weight than k=1000."""
        ranked = self._ranked([("a", 1.0), ("b", 0.5)])
        result_low  = reciprocal_rank_fusion([ranked], k=1)
        result_high = reciprocal_rank_fusion([ranked], k=1000)
        spread_low  = result_low[0][1]  - result_low[-1][1]
        spread_high = result_high[0][1] - result_high[-1][1]
        assert spread_low > spread_high


# ── HybridRetriever ───────────────────────────────────────────────────────────


class TestHybridRetriever:

    # ── two-source invariant ──────────────────────────────────────────────────

    def test_finds_keyword_only_doc_not_in_dense(self, hybrid: HybridRetriever) -> None:
        """Core: DOC_KEYWORD is absent from dense results.

        Hybrid retrieval must surface it via the BM25 path after RRF fusion.
        Validates that sparse retrieval adds recall beyond what dense provides.
        """
        # Precondition: verify the mock dense does NOT contain doc_keyword
        dense_hits = hybrid._dense.retrieve(KEYWORD_QUERY, k=20)
        dense_ids  = {d.metadata["doc_id"] for d, _ in dense_hits}
        assert "doc_keyword" not in dense_ids, (
            "Test setup error: mock dense must not return doc_keyword"
        )

        results    = hybrid.retrieve(KEYWORD_QUERY, k=3)
        result_ids = {d.metadata["doc_id"] for d, _ in results}
        assert "doc_keyword" in result_ids, (
            "Hybrid must retrieve doc_keyword via BM25 even though dense misses it"
        )

    def test_finds_semantic_doc_not_in_bm25(self, hybrid: HybridRetriever) -> None:
        """Core: DOC_SEMANTIC scores zero in BM25 for KEYWORD_QUERY.

        Hybrid retrieval must surface it via the dense path after RRF fusion.
        Validates that dense retrieval adds recall beyond what BM25 provides.
        """
        # Precondition: verify BM25 does NOT return doc_semantic
        bm25_hits = hybrid._sparse.retrieve(KEYWORD_QUERY, k=20)
        bm25_ids  = {d.metadata["doc_id"] for d, _ in bm25_hits}
        assert "doc_semantic" not in bm25_ids, (
            "Test setup error: BM25 must not return doc_semantic for 'Paris' query"
        )

        results    = hybrid.retrieve(KEYWORD_QUERY, k=3)
        result_ids = {d.metadata["doc_id"] for d, _ in results}
        assert "doc_semantic" in result_ids, (
            "Hybrid must retrieve doc_semantic via dense even though BM25 misses it"
        )

    def test_retrieves_both_keyword_and_semantic_doc(self, hybrid: HybridRetriever) -> None:
        """End-to-end: hybrid finds BOTH sources in a single call."""
        results    = hybrid.retrieve(KEYWORD_QUERY, k=3)
        result_ids = {d.metadata["doc_id"] for d, _ in results}
        assert "doc_keyword" in result_ids, "BM25-only keyword doc must appear"
        assert "doc_semantic" in result_ids, "Dense-only semantic doc must appear"

    # ── ordering & basic properties ───────────────────────────────────────────

    def test_results_sorted_by_descending_rrf_score(self, hybrid: HybridRetriever) -> None:
        results = hybrid.retrieve(KEYWORD_QUERY, k=3)
        scores  = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_output_length(self, hybrid: HybridRetriever) -> None:
        for k in (1, 2, 3):
            assert len(hybrid.retrieve(KEYWORD_QUERY, k=k)) <= k

    def test_all_rrf_scores_positive(self, hybrid: HybridRetriever) -> None:
        results = hybrid.retrieve(KEYWORD_QUERY, k=5)
        assert all(score > 0 for _, score in results)

    def test_no_duplicate_docs_in_output(self, hybrid: HybridRetriever) -> None:
        results    = hybrid.retrieve(KEYWORD_QUERY, k=5)
        chunk_ids  = [doc.metadata["chunk_id"] for doc, _ in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks in hybrid output"

    def test_empty_sparse_falls_back_to_dense_only(self) -> None:
        """When BM25 finds nothing, hybrid output equals dense-only ranking."""
        bm25_empty = BM25Retriever([_doc("Unrelated topic xyz123.", "unrelated")])
        h = HybridRetriever(dense=_MockDenseRetriever(), sparse=bm25_empty)
        results    = h.retrieve("Paris", k=3)
        result_ids = {d.metadata["doc_id"] for d, _ in results}
        assert "doc_semantic" in result_ids

    def test_rrf_k_constant_controls_score_spread(self) -> None:
        """Smaller rrf_k → top rank is more strongly rewarded."""
        low_k_hybrid  = HybridRetriever(
            dense=_MockDenseRetriever(), sparse=BM25Retriever(CORPUS), rrf_k=1
        )
        high_k_hybrid = HybridRetriever(
            dense=_MockDenseRetriever(), sparse=BM25Retriever(CORPUS), rrf_k=10_000
        )
        res_low  = low_k_hybrid.retrieve(KEYWORD_QUERY, k=3)
        res_high = high_k_hybrid.retrieve(KEYWORD_QUERY, k=3)
        if len(res_low) >= 2 and len(res_high) >= 2:
            spread_low  = res_low[0][1]  - res_low[-1][1]
            spread_high = res_high[0][1] - res_high[-1][1]
            assert spread_low > spread_high, "Low rrf_k must produce wider score spread"


# ── CrossEncoderReranker ──────────────────────────────────────────────────────


class _MockCrossEncoder:
    """Assigns score = N - i (first input pair gets highest score = N)."""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        n = len(pairs)
        return np.array([float(n - i) for i in range(n)])


def _make_reranker() -> CrossEncoderReranker:
    """Return a CrossEncoderReranker with the mock model pre-injected."""
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r.top_k     = 5
    r.device    = "cpu"
    r.batch_size = 32
    r._model    = _MockCrossEncoder()
    return r


class TestCrossEncoderReranker:
    def test_empty_candidates_returns_empty(self) -> None:
        assert _make_reranker().rerank("query", []) == []

    def test_output_length_limited_by_k(self) -> None:
        candidates = [(_doc(f"Content {i}", f"doc_{i}"), float(i)) for i in range(6)]
        result     = _make_reranker().rerank("query", candidates, k=3)
        assert len(result) == 3

    def test_reranker_overrides_first_stage_order(self) -> None:
        """First-stage scores are ascending; mock CE reverses them.

        doc_0 has the LOWEST first-stage score (0.1) but the HIGHEST CE score
        because the mock assigns score = N - index (index 0 → highest).
        After reranking doc_0 must be first and doc_2 must be dropped.
        """
        docs       = [_doc(f"Text {i}", f"doc_{i}") for i in range(3)]
        candidates = [(docs[0], 0.1), (docs[1], 0.5), (docs[2], 0.9)]

        result = _make_reranker().rerank("query", candidates, k=2)

        assert len(result) == 2
        assert result[0][0].metadata["doc_id"] == "doc_0", (
            "doc_0 must be promoted despite lowest first-stage score"
        )
        result_ids = [doc.metadata["doc_id"] for doc, _ in result]
        assert "doc_2" not in result_ids, (
            "doc_2 must be demoted despite highest first-stage score"
        )

    def test_returned_score_is_cross_encoder_score(self) -> None:
        """Scores in output must come from CE, not from the first-stage retriever."""
        candidates = [(_doc("text", "d0"), 0.99)]
        _, score   = _make_reranker().rerank("query", candidates, k=1)[0]
        # Mock CE: n=1, i=0 → score = 1.0
        assert pytest.approx(score, abs=1e-6) == 1.0
        assert score != 0.99, "Score must not be the first-stage score"

    def test_k_defaults_to_top_k_attribute(self) -> None:
        r           = _make_reranker()
        r.top_k     = 2
        candidates  = [(_doc(f"T{i}", f"d{i}"), 1.0) for i in range(5)]
        result      = r.rerank("query", candidates)          # no k= argument
        assert len(result) == 2

    def test_results_sorted_descending(self) -> None:
        candidates = [(_doc(f"T{i}", f"d{i}"), 0.0) for i in range(5)]
        result     = _make_reranker().rerank("q", candidates, k=5)
        scores     = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_returned_scores_are_float(self) -> None:
        candidates = [(_doc("txt", "d"), 0.5)]
        _, score   = _make_reranker().rerank("q", candidates, k=1)[0]
        assert isinstance(score, float)

    def test_batch_size_does_not_affect_result(self) -> None:
        """batch_size is a performance knob only — results must be identical."""
        candidates = [(_doc(f"doc {i}", f"doc_{i}"), float(i)) for i in range(10)]
        r = _make_reranker()

        r.batch_size = 10
        ids_full  = [d.metadata["doc_id"] for d, _ in r.rerank("q", candidates, k=5)]
        r.batch_size = 3
        ids_small = [d.metadata["doc_id"] for d, _ in r.rerank("q", candidates, k=5)]

        assert ids_full == ids_small
