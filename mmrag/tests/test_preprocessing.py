"""Tests for src.preprocessing (chunking, audio, image)."""

import numpy as np
import pytest
from langchain_core.documents import Document

from src.preprocessing.chunking import chunk_documents, ChunkingStrategy
from src.preprocessing.audio import extract_audio_features, normalize_audio
from src.preprocessing.image import preprocess_image


@pytest.fixture()
def sample_docs() -> list[Document]:
    return [
        Document(page_content="word " * 200, metadata={"source": "doc1.pdf"}),
        Document(page_content="sentence. " * 50, metadata={"source": "doc2.pdf"}),
    ]


class TestChunking:
    def test_fixed_produces_correct_chunk_count(self, sample_docs: list[Document]) -> None:
        raise NotImplementedError

    def test_chunks_preserve_source_metadata(self, sample_docs: list[Document]) -> None:
        raise NotImplementedError

    def test_unknown_strategy_raises(self, sample_docs: list[Document]) -> None:
        raise NotImplementedError


class TestAudioPreprocessing:
    def test_normalize_audio_range(self) -> None:
        waveform = np.random.randn(16_000).astype(np.float32) * 5
        result = normalize_audio(waveform)
        raise NotImplementedError  # assert abs(result).max() <= 1.0

    def test_extract_features_keys(self, tmp_path: object) -> None:
        raise NotImplementedError


class TestImagePreprocessing:
    def test_preprocess_returns_correct_shape(self, tmp_path: object) -> None:
        raise NotImplementedError
