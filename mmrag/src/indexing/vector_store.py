"""Multi-collection ChromaDB vector store for multimodal RAG.

Design
------
One Chroma collection per embedding space, enforced by the modality key:

    modality="text"        → collection ``{prefix}_text``
                             (sentence-transformer embeddings of text chunks,
                              image captions, and audio transcripts)

    modality="image_clip"  → collection ``{prefix}_image_clip``
                             (open-clip ViT-B-32 image embeddings — 512-dim)

    modality="audio_clap"  → collection ``{prefix}_audio_clap``
                             (CLAP audio embeddings — 512-dim, optional)

Stable IDs
----------
Every document is assigned a deterministic 16-char hex ID:

    id = sha256(f"{doc_id}|{chunk_id}|{embedder_name}")[:16]

This makes upserts idempotent: re-indexing the same chunk with the same
embedder is a no-op in ChromaDB.

Chroma metadata constraints
---------------------------
Only scalar values (str / int / float / bool) are stored.  List/dict metadata
fields are serialised to JSON strings.  Non-serialisable values are dropped
with a warning.

Embeddings
----------
The caller is responsible for computing embeddings (TextEmbedder, ClipEmbedder,
etc.) and passing them as a ``(N, D)`` numpy array.  The VectorStoreIndex never
embeds text itself — it is a pure storage and retrieval layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supported modality keys → Chroma HNSW distance metric
_MODALITY_METRIC: dict[str, str] = {
    "text":       "cosine",
    "image_clip": "cosine",
    "audio_clap": "cosine",
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _stable_id(doc_id: str, chunk_id: str, embedder_name: str) -> str:
    """Compute a 16-char hex ID that is stable across re-indexing runs.

    Args:
        doc_id:       Source document identifier.
        chunk_id:     Chunk identifier within the document.
        embedder_name: Short name of the embedding model.

    Returns:
        16-character hexadecimal string.
    """
    raw = f"{doc_id}|{chunk_id}|{embedder_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _to_chroma_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Strip or encode non-scalar metadata values for ChromaDB.

    ChromaDB only accepts str / int / float / bool.  Lists are JSON-encoded;
    dicts and other types are dropped with a debug log.

    Args:
        meta: Arbitrary metadata dict.

    Returns:
        ChromaDB-compatible dict of scalar values.
    """
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            try:
                clean[k] = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                logger.debug("_to_chroma_meta: skipping unserializable list field %r", k)
        elif v is None:
            clean[k] = ""  # Chroma does not accept None
        else:
            logger.debug("_to_chroma_meta: dropping non-scalar field %r (type %s)", k, type(v).__name__)
    return clean


# ── main class ────────────────────────────────────────────────────────────────


class VectorStoreIndex:
    """Persistent multi-collection ChromaDB index.

    Args:
        persist_dir:       Directory where ChromaDB stores its data files.
                           Ignored when *_client* is provided (used in tests).
        collection_prefix: Prefix for all collection names (default ``"mmrag"``).
        _client:           Inject a custom Chroma client (e.g. an ephemeral
                           client for tests).  If None, a persistent client is
                           created at *persist_dir*.
    """

    def __init__(
        self,
        persist_dir: str | Path = "chroma_db",
        collection_prefix: str = "mmrag",
        _client: Any | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_prefix = collection_prefix

        if _client is not None:
            self._client = _client
        else:
            import chromadb  # noqa: PLC0415
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Cache of open collections: modality → chromadb.Collection
        self._collections: dict[str, Any] = {}

    # ── collection management ─────────────────────────────────────────────────

    def _collection_name(self, modality: str) -> str:
        return f"{self.collection_prefix}_{modality}"

    def get_collection(self, modality: str) -> Any:
        """Return (and lazily create) the Chroma collection for *modality*.

        Args:
            modality: One of ``"text"``, ``"image_clip"``, ``"audio_clap"``.

        Returns:
            A :class:`chromadb.Collection` object.

        Raises:
            ValueError: If *modality* is not recognised.
        """
        if modality not in _MODALITY_METRIC:
            raise ValueError(
                f"Unknown modality {modality!r}. "
                f"Supported: {sorted(_MODALITY_METRIC)}"
            )
        if modality not in self._collections:
            name = self._collection_name(modality)
            metric = _MODALITY_METRIC[modality]
            self._collections[modality] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": metric},
            )
            logger.debug("VectorStoreIndex: collection %r opened", name)
        return self._collections[modality]

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks: list[Document],
        embeddings: np.ndarray,
        modality: str,
        embedder_name: str = "default",
    ) -> list[str]:
        """Insert or update chunks with pre-computed embeddings (idempotent).

        The same ``(doc_id, chunk_id, embedder_name)`` triple always produces
        the same Chroma ID, so calling upsert twice with identical inputs is a
        safe no-op.

        Args:
            chunks:        Chunk Documents; each must have ``doc_id`` and
                           ``chunk_id`` in ``metadata``.
            embeddings:    Float32 array of shape ``(N, embedding_dim)``.
            modality:      Collection key (``"text"``, ``"image_clip"``, …).
            embedder_name: Short model name included in the stable ID hash.

        Returns:
            List of stable 16-char hex IDs that were upserted, in order.

        Raises:
            ValueError: If ``len(chunks) != len(embeddings)``.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length"
            )
        if not chunks:
            return []

        collection = self.get_collection(modality)

        # Build parallel lists; deduplicate by stable ID (keep last occurrence)
        # so that ChromaDB never receives a batch with duplicate IDs.
        dedup: dict[str, tuple] = {}
        for chunk, vec in zip(chunks, embeddings):
            meta = chunk.metadata
            doc_id  = str(meta.get("doc_id",  meta.get("id", "unknown")))
            chunk_id = str(meta.get("chunk_id", doc_id))

            stable = _stable_id(doc_id, chunk_id, embedder_name)

            # Build Chroma-compatible metadata (always include key fields)
            chroma_meta = _to_chroma_meta(meta)
            chroma_meta.setdefault("doc_id",   doc_id)
            chroma_meta.setdefault("chunk_id", chunk_id)
            chroma_meta.setdefault("modality", modality)
            chroma_meta["embedding_model"] = embedder_name

            dedup[stable] = (vec.tolist(), chunk.page_content, chroma_meta)

        if len(dedup) < len(chunks):
            logger.warning(
                "VectorStoreIndex.upsert: deduplicated %d → %d chunks (duplicate chunk_ids)",
                len(chunks), len(dedup),
            )

        ids        = list(dedup.keys())
        embed_list = [v[0] for v in dedup.values()]
        documents  = [v[1] for v in dedup.values()]
        metadatas  = [v[2] for v in dedup.values()]

        collection.upsert(
            ids=ids,
            embeddings=embed_list,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "VectorStoreIndex: upserted %d docs into collection %r",
            len(ids),
            self._collection_name(modality),
        )
        return ids

    # ── read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        modality: str = "text",
        k: int = 5,
        filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the k nearest chunks for *query_embedding*.

        For cosine distance (default), the returned score is:
            ``score = 1.0 - chroma_distance``
        so a perfect match gives ``score = 1.0`` and orthogonal gives ``score = 0.0``.

        Args:
            query_embedding: Float32 vector of shape ``(D,)`` or ``(1, D)``.
            modality:        Collection to search.
            k:               Maximum number of results.
            filter:          ChromaDB ``where`` clause (metadata equality filters).
            score_threshold: If set, discard results with ``score < threshold``.

        Returns:
            ``(Document, score)`` list sorted by descending score.
        """
        collection = self.get_collection(modality)
        n_stored = collection.count()
        if n_stored == 0:
            return []

        actual_k = min(k, n_stored)
        vec = query_embedding.flatten().tolist()

        results = collection.query(
            query_embeddings=[vec],
            n_results=actual_k,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        pairs: list[tuple[Document, float]] = []
        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = float(1.0 - dist)
            if score_threshold is not None and score < score_threshold:
                continue
            pairs.append((Document(page_content=doc_text, metadata=meta), score))

        return pairs

    # ── collection stats / admin ──────────────────────────────────────────────

    def count(self, modality: str) -> int:
        """Number of documents in the *modality* collection.

        Args:
            modality: Collection key.

        Returns:
            Integer count (0 if the collection is empty or does not exist yet).
        """
        try:
            return self.get_collection(modality).count()
        except Exception:
            return 0

    def delete_collection(self, modality: str) -> None:
        """Permanently delete a Chroma collection.

        Args:
            modality: Collection key to delete.
        """
        name = self._collection_name(modality)
        try:
            self._client.delete_collection(name)
            self._collections.pop(modality, None)
            logger.info("VectorStoreIndex: deleted collection %r", name)
        except Exception as exc:
            logger.warning("VectorStoreIndex: could not delete %r — %s", name, exc)

    def persist(self) -> None:
        """Flush pending writes to disk (no-op in ChromaDB 0.5+, kept for API compat)."""
        # PersistentClient auto-persists; EphemeralClient has no disk state.
        pass

    def list_modalities(self) -> list[str]:
        """Return the modality keys for which a collection currently exists.

        Returns:
            Sorted list of modality strings (e.g. ``["image_clip", "text"]``).
        """
        existing_names = {c.name for c in self._client.list_collections()}
        prefix = f"{self.collection_prefix}_"
        return sorted(
            name[len(prefix):]
            for name in existing_names
            if name.startswith(prefix)
        )
