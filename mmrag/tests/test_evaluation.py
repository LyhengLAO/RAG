"""Tests for src.evaluation — retrieval metrics, system metrics, eval-set logic.

These cover the dependency-free core of the evaluation harness (no ragas,
chromadb, or LLM required), so they run in CI without heavy models.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.eval_dataset import (
    answer_is_grounded,
    parse_qa_response,
    question_leaks_verbatim,
    select_eval_docs,
    validate_qa,
)
from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    dedup_preserve_order,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.system_metrics import (
    aggregate_system_metrics,
    percentiles,
    token_delta,
)


class TestRetrievalMetrics:
    def test_perfect_retrieval_precision_is_one(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_no_relevant_doc_precision_is_zero(self) -> None:
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_partial_precision(self) -> None:
        assert precision_at_k(["a", "x"], {"a"}, 2) == pytest.approx(0.5)

    def test_recall(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == pytest.approx(0.5)
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_hit_at_k(self) -> None:
        assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0
        assert hit_at_k(["a", "b"], {"b"}, 1) == 0.0

    def test_mrr_first_hit_at_rank_one(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_ndcg_binary(self) -> None:
        # one relevant doc retrieved at rank 2: dcg = 1/log2(3), idcg = 1
        assert ndcg_at_k(["x", "a", "y"], {"a"}, 3) == pytest.approx(1 / math.log2(3))
        # perfect ordering → 1.0
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_dedup_preserves_order(self) -> None:
        assert dedup_preserve_order(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_compute_retrieval_metrics_returns_all_keys(self) -> None:
        m = compute_retrieval_metrics([["a", "b"], ["c"]], [["a"], ["c", "d"]], k_values=[1, 5])
        for key in ("hit@1", "hit@5", "recall@5", "precision@5", "ndcg@5", "mrr", "n_queries"):
            assert key in m

    def test_chunk_dedup_and_aggregation(self) -> None:
        m = compute_retrieval_metrics([["d1", "d1", "d2"], ["d3", "d4"]], [["d1"], ["d4", "d5"]])
        assert m["hit@1"] == pytest.approx(0.5)
        assert m["mrr"] == pytest.approx(0.75)
        assert m["recall@5"] == pytest.approx(0.75)
        assert m["n_queries"] == 2.0

    def test_empty_relevant_query_skipped(self) -> None:
        m = compute_retrieval_metrics([["a"], ["b"]], [["a"], []])
        assert m["n_queries"] == 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_retrieval_metrics([["a"]], [["a"], ["b"]])


class TestSystemMetrics:
    def test_percentiles_ignore_nan(self) -> None:
        p = percentiles([10, 20, 30, float("nan"), 40])
        assert p["n"] == 4
        assert p["p50"] == 25.0

    def test_percentiles_all_nan(self) -> None:
        p = percentiles([float("nan"), float("nan")])
        assert p["n"] == 0
        assert math.isnan(p["p50"])

    def test_aggregate_per_modality(self) -> None:
        agg = aggregate_system_metrics([
            {"latency_ms": 100, "modality": "text", "prompt_tokens": 50,
             "completion_tokens": 10, "total_tokens": 60},
            {"latency_ms": 200, "modality": "audio", "prompt_tokens": float("nan"),
             "completion_tokens": float("nan"), "total_tokens": float("nan")},
        ])
        assert agg["overall"]["n_queries"] == 2
        assert set(agg["per_modality"]) == {"text", "audio"}
        assert agg["per_modality"]["text"]["latency_ms"]["p50"] == 100.0
        assert agg["per_modality"]["audio"]["tokens"]["total_tokens"]["n"] == 0

    def test_token_delta_valid(self) -> None:
        before = {"calls": 0, "missing": 0, "prompt_tokens": 0, "completion_tokens": 0}
        after = {"calls": 1, "missing": 0, "prompt_tokens": 50, "completion_tokens": 10}
        assert token_delta(before, after)["total_tokens"] == 60.0

    def test_token_delta_missing_is_nan(self) -> None:
        before = {"calls": 0, "missing": 0, "prompt_tokens": 0, "completion_tokens": 0}
        after = {"calls": 1, "missing": 1, "prompt_tokens": 0, "completion_tokens": 0}
        assert math.isnan(token_delta(before, after)["total_tokens"])


class TestEvalDatasetLogic:
    def test_split_is_deterministic_and_disjoint(self) -> None:
        docs = (
            [{"id": f"text_{i:04d}", "modality": "text"} for i in range(10)]
            + [{"id": f"audio_{i:04d}", "modality": "audio"} for i in range(5)]
        )
        s1 = select_eval_docs(docs, {"text": 3, "audio": 2}, seed=42)
        s2 = select_eval_docs(docs, {"text": 3, "audio": 2}, seed=42)
        assert s1.eval_doc_ids == s2.eval_doc_ids
        assert s1.counts == {"text": 3, "audio": 2}
        assert set(s1.eval_doc_ids).isdisjoint(set(s1.index_doc_ids))
        assert len(s1.index_doc_ids) == 10

    def test_parse_strict_json(self) -> None:
        out = parse_qa_response('noise {"question": "Who?", "answer": "Bob"} tail')
        assert out == {"question": "Who?", "answer": "Bob"}

    def test_parse_heuristic_and_failure(self) -> None:
        out = parse_qa_response("question: What color?\nanswer: blue")
        assert out and out["answer"] == "blue"
        assert parse_qa_response("garbage") is None

    def test_grounding(self) -> None:
        src = "The Eiffel Tower is located in Paris and was completed in 1889."
        assert answer_is_grounded("Paris", src)
        assert not answer_is_grounded("Tokyo skyline at night festival", src)

    def test_verbatim_leak_guard(self) -> None:
        src = "The Eiffel Tower is located in Paris and was completed in 1889."
        assert question_leaks_verbatim(
            "Tell me: the Eiffel Tower is located in Paris and was completed", src, max_span_words=8
        )
        assert not question_leaks_verbatim("Where is the Eiffel Tower located?", src, max_span_words=8)

    def test_validate_accepts_good_pair(self) -> None:
        src = "The Eiffel Tower is located in Paris and was completed in 1889."
        ok, reasons = validate_qa("Where is the Eiffel Tower located?", "Paris", src)
        assert ok, reasons

    def test_validate_rejects_ungrounded(self) -> None:
        src = "The Eiffel Tower is located in Paris."
        ok, reasons = validate_qa("Where is it?", "Atlantis underwater city", src)
        assert not ok
        assert "answer_not_grounded" in reasons
