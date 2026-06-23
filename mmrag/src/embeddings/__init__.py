"""Embeddings sub-package: text, image (CLIP), and audio (CLAP) encoders."""

from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.clip_embedder import ClipEmbedder
from src.embeddings.clap_embedder import ClapEmbedder

__all__ = ["TextEmbedder", "ClipEmbedder", "ClapEmbedder"]
