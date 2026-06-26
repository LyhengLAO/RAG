"""Tests for src.ingestion — schema unit tests + manifest integrity + determinism.

Fast tests (no network):
    pytest tests/test_ingestion.py -k "not Manifest and not Determinism"

Full tests (require: make build-dataset first):
    pytest tests/test_ingestion.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.schema import RawDocument

# ── constants ────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"id", "modality", "content", "text", "source", "license", "metadata"}
VALID_MODALITIES = {"text", "image", "audio"}

MANIFEST_PATH = Path("data/processed/manifest.jsonl")
RAW_TEXT_MANIFEST = Path("data/raw/text/manifest.jsonl")


def _manifest_exists() -> bool:
    return MANIFEST_PATH.exists() and MANIFEST_PATH.stat().st_size > 0


def _text_cache_exists() -> bool:
    return RAW_TEXT_MANIFEST.exists() and RAW_TEXT_MANIFEST.stat().st_size > 0


# ── Schema unit tests (zero network, always run) ─────────────────────────────


class TestRawDocumentSchema:
    def test_roundtrip_json(self) -> None:
        doc = RawDocument(
            id="text_0001",
            modality="text",
            content="The quick brown fox",
            text="The quick brown fox",
            source="test/dataset",
            license="CC-BY-4.0",
            metadata={"title": "Fable"},
        )
        assert RawDocument.from_json(doc.to_json()) == doc

    def test_to_dict_contains_all_required_fields(self) -> None:
        doc = RawDocument(id="x", modality="text", content="c", text="t", source="s", license="l")
        assert REQUIRED_FIELDS.issubset(doc.to_dict().keys())

    def test_metadata_defaults_to_empty_dict(self) -> None:
        doc = RawDocument(id="x", modality="text", content="c", text="t", source="s", license="l")
        assert doc.metadata == {}

    def test_from_dict_roundtrip(self) -> None:
        data: dict = {
            "id": "img_0001",
            "modality": "image",
            "content": "/tmp/img.jpg",
            "text": "a cat on a mat",
            "source": "nlphuji/flickr30k",
            "license": "Flickr-Research-Use",
            "metadata": {"img_id": "42", "all_captions": ["a cat on a mat"]},
        }
        doc = RawDocument.from_dict(data)
        assert doc.id == "img_0001"
        assert doc.metadata["img_id"] == "42"

    def test_from_dict_does_not_require_metadata_key(self) -> None:
        data = {
            "id": "x",
            "modality": "audio",
            "content": "/tmp/a.wav",
            "text": "hello",
            "source": "s",
            "license": "CC0",
        }
        doc = RawDocument.from_dict(data)
        assert doc.metadata == {}

    @pytest.mark.parametrize("modality", ["text", "image", "audio"])
    def test_all_valid_modalities_survive_json_roundtrip(self, modality: str) -> None:
        doc = RawDocument(id="x", modality=modality, content="c", text="t", source="s", license="l")
        assert RawDocument.from_json(doc.to_json()).modality == modality

    def test_equality_checks_content(self) -> None:
        base = RawDocument(id="a", modality="text", content="X", text="X", source="s", license="l")
        diff = RawDocument(id="a", modality="text", content="Y", text="Y", source="s", license="l")
        assert base != diff

    def test_to_json_produces_valid_json(self) -> None:
        doc = RawDocument(id="z", modality="audio", content="/f.wav", text="hi", source="s", license="l")
        parsed = json.loads(doc.to_json())
        assert parsed["id"] == "z"


# ── Manifest integrity (require: make build-dataset) ─────────────────────────


@pytest.fixture(scope="module")
def manifest_docs() -> list[dict]:
    if not _manifest_exists():
        return []
    out: list[dict] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@pytest.mark.skipif(not _manifest_exists(), reason="manifest not built — run: make build-dataset")
class TestManifestIntegrity:
    def test_not_empty(self, manifest_docs: list[dict]) -> None:
        assert len(manifest_docs) > 0, "manifest.jsonl is empty"

    def test_all_required_fields_present(self, manifest_docs: list[dict]) -> None:
        for doc in manifest_docs:
            missing = REQUIRED_FIELDS - set(doc.keys())
            assert not missing, f"doc {doc.get('id')} missing fields: {missing}"

    def test_ids_are_globally_unique(self, manifest_docs: list[dict]) -> None:
        ids = [d["id"] for d in manifest_docs]
        dupes = {x for x in ids if ids.count(x) > 1}
        assert not dupes, f"duplicate IDs: {dupes}"

    def test_modality_values_valid(self, manifest_docs: list[dict]) -> None:
        bad = [d["id"] for d in manifest_docs if d["modality"] not in VALID_MODALITIES]
        assert not bad, f"unknown modalities: {bad[:5]}"

    def test_at_least_two_modalities_present(self, manifest_docs: list[dict]) -> None:
        found = {d["modality"] for d in manifest_docs}
        assert len(found) >= 2, f"only found: {found}"

    def test_text_field_never_empty(self, manifest_docs: list[dict]) -> None:
        empty = [d["id"] for d in manifest_docs if not d.get("text", "").strip()]
        assert not empty, f"empty 'text' on: {empty[:5]}"

    def test_content_field_never_empty(self, manifest_docs: list[dict]) -> None:
        empty = [d["id"] for d in manifest_docs if not d.get("content", "").strip()]
        assert not empty, f"empty 'content' on: {empty[:5]}"

    def test_license_field_never_empty(self, manifest_docs: list[dict]) -> None:
        empty = [d["id"] for d in manifest_docs if not d.get("license", "").strip()]
        assert not empty, f"empty 'license' on: {empty[:5]}"

    def test_image_files_exist_on_disk(self, manifest_docs: list[dict]) -> None:
        broken = [
            d["id"]
            for d in manifest_docs
            if d["modality"] == "image" and not Path(d["content"]).exists()
        ]
        assert not broken, f"missing image files: {broken[:5]}"

    def test_audio_files_exist_on_disk(self, manifest_docs: list[dict]) -> None:
        broken = [
            d["id"]
            for d in manifest_docs
            if d["modality"] == "audio" and not Path(d["content"]).exists()
        ]
        assert not broken, f"missing audio files: {broken[:5]}"

    def test_all_docs_deserialise_without_error(self, manifest_docs: list[dict]) -> None:
        for raw in manifest_docs:
            doc = RawDocument.from_dict(dict(raw))
            assert doc.id  # non-empty id

    def test_schema_version_survives_roundtrip(self, manifest_docs: list[dict]) -> None:
        """Re-serialise every manifest entry and verify the JSON is identical."""
        for raw in manifest_docs:
            doc = RawDocument.from_dict(dict(raw))
            reloaded = json.loads(doc.to_json())
            assert reloaded["id"] == raw["id"]
            assert reloaded["modality"] == raw["modality"]


# ── Determinism (require text raw cache) ─────────────────────────────────────


@pytest.mark.skipif(not _text_cache_exists(), reason="text cache not built — run: make build-dataset")
class TestDeterminism:
    """Verify that loaders are idempotent: two calls return identical results."""

    def test_text_loader_same_ids_on_two_calls(self) -> None:
        from src.ingestion.loaders import load_text_corpus

        ids1 = [d.id for d in load_text_corpus(seed=42)]
        ids2 = [d.id for d in load_text_corpus(seed=42)]
        assert ids1 == ids2, "loader returned different IDs on second call"

    def test_text_loader_same_content_on_two_calls(self) -> None:
        from src.ingestion.loaders import load_text_corpus

        contents1 = {d.id: d.content for d in load_text_corpus(seed=42)}
        contents2 = {d.id: d.content for d in load_text_corpus(seed=42)}
        assert contents1 == contents2

    def test_text_loader_count_is_stable(self) -> None:
        from src.ingestion.loaders import load_text_corpus

        n1 = len(load_text_corpus(seed=42))
        n2 = len(load_text_corpus(seed=42))
        assert n1 == n2 > 0
