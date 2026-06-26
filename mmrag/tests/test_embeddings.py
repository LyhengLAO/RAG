"""Tests for src.embeddings (text, image, audio)."""

import numpy as np
import pytest

from src.embeddings.text import TextEmbedder
from src.embeddings.image import ImageEmbedder
from src.embeddings.audio import AudioEmbedder


class TestTextEmbedder:
    def test_embed_documents_shape(self) -> None:
        raise NotImplementedError

    def test_embed_query_is_1d(self) -> None:
        raise NotImplementedError

    def test_embeddings_are_normalized(self) -> None:
        raise NotImplementedError

    def test_empty_input_raises(self) -> None:
        raise NotImplementedError


class TestImageEmbedder:
    def test_embed_image_shape(self, tmp_path: object) -> None:
        raise NotImplementedError

    def test_text_query_embedding_compatible_with_image(self) -> None:
        raise NotImplementedError


class TestAudioEmbedder:
    def test_transcribe_returns_string(self, tmp_path: object) -> None:
        raise NotImplementedError

    def test_embed_audio_shape(self, tmp_path: object) -> None:
        raise NotImplementedError
