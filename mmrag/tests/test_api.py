"""Tests for src.serving.api — uses FastAPI TestClient, no real pipelines.

The lifespan is exercised through the ``with TestClient(app)`` context manager,
but the pipeline constructors are patched before startup so no heavy models
or ChromaDB are needed.  Ollama/Chroma probes are patched per-test.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.serving.api import _state, app

# ── Shared fixture data ────────────────────────────────────────────────────────

_MOCK_RESULT = {
    "answer": "Paris",
    "retrieved_contexts": ["The Eiffel Tower is located in Paris."],
    "sources": [
        {
            "doc_id": "doc_001",
            "modality": "text",
            "source": "wiki.org/Eiffel_Tower",
            "license": "CC-BY-SA",
        }
    ],
    "latency_ms": 42.0,
}


# ── Module-scoped client fixture ───────────────────────────────────────────────
#
# Patches both pipeline constructors so the lifespan stores mock instances
# instead of trying to load real models.  Scope is "module" so startup runs
# once per test file, matching production behaviour (load once, reuse).


@pytest.fixture(scope="module")
def client():
    mock_b = MagicMock()
    mock_b.query.return_value = _MOCK_RESULT
    mock_o = MagicMock()
    mock_o.query.return_value = _MOCK_RESULT

    import src.pipelines.baseline as bl_mod
    import src.pipelines.optimized as op_mod

    with (
        patch.object(bl_mod, "BaselinePipeline", return_value=mock_b),
        patch.object(op_mod, "OptimizedPipeline", return_value=mock_o),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok_when_all_services_up(self, client: TestClient) -> None:
        with (
            patch("src.serving.api._ollama_up", return_value=True),
            patch("src.serving.api._chroma_accessible", return_value=True),
        ):
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["ollama_up"] is True
        assert body["chroma_accessible"] is True
        assert body["pipelines_loaded"] == {"baseline": True, "optimized": True}
        assert "ollama_model" in body

    def test_health_degraded_when_ollama_down(self, client: TestClient) -> None:
        with (
            patch("src.serving.api._ollama_up", return_value=False),
            patch("src.serving.api._chroma_accessible", return_value=True),
        ):
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["ollama_up"] is False

    def test_health_degraded_when_chroma_down(self, client: TestClient) -> None:
        with (
            patch("src.serving.api._ollama_up", return_value=True),
            patch("src.serving.api._chroma_accessible", return_value=False),
        ):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        assert r.json()["chroma_accessible"] is False

    def test_health_degraded_when_pipeline_not_loaded(self, client: TestClient) -> None:
        prev = _state.baseline
        _state.baseline = None
        try:
            with (
                patch("src.serving.api._ollama_up", return_value=True),
                patch("src.serving.api._chroma_accessible", return_value=True),
            ):
                r = client.get("/health")
            body = r.json()
            assert body["status"] == "degraded"
            assert body["pipelines_loaded"]["baseline"] is False
            assert body["pipelines_loaded"]["optimized"] is True
        finally:
            _state.baseline = prev


# ── POST /query ────────────────────────────────────────────────────────────────


class TestQuery:
    def test_query_baseline_returns_typed_response(self, client: TestClient) -> None:
        r = client.post(
            "/query",
            json={"question": "Where is the Eiffel Tower?", "pipeline": "baseline"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Paris"
        assert body["pipeline"] == "baseline"
        assert body["retrieved_contexts"] == ["The Eiffel Tower is located in Paris."]
        assert body["latency_ms"] == 42.0
        src = body["sources"][0]
        assert src["doc_id"] == "doc_001"
        assert src["modality"] == "text"

    def test_query_optimized_returns_typed_response(self, client: TestClient) -> None:
        r = client.post(
            "/query",
            json={"question": "When was it built?", "pipeline": "optimized"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline"] == "optimized"
        assert body["answer"] == "Paris"

    def test_query_calls_correct_pipeline_instance(self, client: TestClient) -> None:
        # Each pipeline mock is distinct — verify the right one is called.
        _state.baseline.query.reset_mock()
        _state.optimized.query.reset_mock()

        client.post("/query", json={"question": "test q", "pipeline": "baseline"})
        _state.baseline.query.assert_called_once_with("test q")
        _state.optimized.query.assert_not_called()

        _state.baseline.query.reset_mock()
        client.post("/query", json={"question": "test q", "pipeline": "optimized"})
        _state.optimized.query.assert_called_once_with("test q")
        _state.baseline.query.assert_not_called()

    def test_query_503_when_pipeline_not_loaded(self, client: TestClient) -> None:
        prev = _state.optimized
        _state.optimized = None
        try:
            r = client.post("/query", json={"question": "x", "pipeline": "optimized"})
            assert r.status_code == 503
            assert "optimized" in r.json()["detail"]
        finally:
            _state.optimized = prev

    def test_query_503_on_llm_runtime_error(self, client: TestClient) -> None:
        prev_side = _state.baseline.query.side_effect
        _state.baseline.query.side_effect = RuntimeError("Ollama is down")
        try:
            r = client.post("/query", json={"question": "x", "pipeline": "baseline"})
            assert r.status_code == 503
            assert "Ollama" in r.json()["detail"]
        finally:
            _state.baseline.query.side_effect = prev_side

    def test_query_422_on_empty_question(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "", "pipeline": "baseline"})
        assert r.status_code == 422

    def test_query_422_on_unknown_pipeline(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "x", "pipeline": "turbo"})
        assert r.status_code == 422

    def test_query_default_pipeline_is_optimized(self, client: TestClient) -> None:
        _state.optimized.query.reset_mock()
        r = client.post("/query", json={"question": "default pipeline?"})
        assert r.status_code == 200
        assert r.json()["pipeline"] == "optimized"
        _state.optimized.query.assert_called_once()


# ── GET /metrics ───────────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_returns_json_content(
        self, client: TestClient, tmp_path: pytest.TempPathFactory
    ) -> None:
        data = {"baseline": {"retrieval": {"hit@5": 0.6}}, "optimized": {"retrieval": {"hit@5": 0.8}}}
        (tmp_path / "metrics.json").write_text(json.dumps(data), encoding="utf-8")

        with patch("src.serving.api.settings") as ms:
            ms.results_dir = tmp_path
            r = client.get("/metrics")

        assert r.status_code == 200
        body = r.json()
        assert body["baseline"]["retrieval"]["hit@5"] == pytest.approx(0.6)
        assert body["optimized"]["retrieval"]["hit@5"] == pytest.approx(0.8)

    def test_metrics_404_when_file_absent(
        self, client: TestClient, tmp_path: pytest.TempPathFactory
    ) -> None:
        with patch("src.serving.api.settings") as ms:
            ms.results_dir = tmp_path  # empty directory — no metrics.json
            r = client.get("/metrics")

        assert r.status_code == 404
        assert "metrics.json" in r.json()["detail"]
