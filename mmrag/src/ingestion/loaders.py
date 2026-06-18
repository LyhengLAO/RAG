"""HuggingFace dataset loaders for text, image, and audio modalities.

Each loader is idempotent: on first call it downloads from HuggingFace and writes a
per-modality manifest.jsonl under ``cache_dir``; on subsequent calls it reads that file
and makes zero network requests.

Sources
-------
* Text  : rajpurkar/squad (Wikipedia contexts) — CC-BY-SA-4.0
* Image : nlphuji/flickr30k (test split)       — Flickr Research-Use
* Audio : openslr/librispeech_asr clean        — CC-BY-4.0  (streamed to avoid bulk DL)
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Callable, TypeVar

from src.ingestion.schema import RawDocument

logger = logging.getLogger(__name__)
T = TypeVar("T")

# ── internal helpers ─────────────────────────────────────────────────────────


def _retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 1.0, label: str = "") -> T:
    """Call *fn* up to *attempts* times with exponential backoff on failure."""
    tag = f" [{label}]" if label else ""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                logger.error("Failed after %d attempts%s: %s", attempts, tag, exc)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d%s failed, retrying in %.0fs — %s", attempt, attempts, tag, delay, exc
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def _read_manifest(path: Path) -> list[RawDocument]:
    docs: list[RawDocument] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                docs.append(RawDocument.from_json(raw))
            except Exception as exc:
                logger.warning("manifest %s line %d skipped — %s", path.name, lineno, exc)
    return docs


def _write_manifest(docs: list[RawDocument], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.to_json() + "\n")


# ── public loaders ───────────────────────────────────────────────────────────


def load_text_corpus(
    n_samples: int = 500,
    seed: int = 42,
    cache_dir: Path | str = Path("data/raw/text"),
    hf_cache_dir: Path | str | None = None,
) -> list[RawDocument]:
    """Load unique Wikipedia passages from SQuAD (train split).

    SQuAD questions share passages, so we deduplicate by context text before
    sampling to avoid repetition.

    Args:
        n_samples:    Maximum number of unique passages to keep (first build only).
        seed:         Random seed for reproducible shuffle (first build only).
        cache_dir:    Directory for the per-modality manifest and any raw files.
        hf_cache_dir: Override the HuggingFace dataset cache root (``HF_HOME``).

    Returns:
        List of :class:`~mmrag.ingestion.schema.RawDocument` with ``modality="text"``.
    """
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.jsonl"

    if manifest.exists():
        docs = _read_manifest(manifest)
        logger.info("text: %d docs from cache (%s)", len(docs), manifest)
        return docs

    try:
        import datasets as hf  # noqa: PLC0415
    except ImportError:
        logger.error("text: 'datasets' not installed — pip install datasets")
        return []

    logger.info("text: downloading rajpurkar/squad …")
    try:
        ds = _retry(
            lambda: hf.load_dataset(
                "rajpurkar/squad",
                split="train",
                cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
            ),
            label="squad",
        )
    except Exception as exc:
        logger.error("text: download failed (%s) — returning []", exc)
        return []

    # Deduplicate contexts (many questions share the same Wikipedia passage)
    seen: dict[str, str] = {}  # context → title
    for row in ds:
        ctx: str = row["context"].strip()
        if ctx and ctx not in seen:
            seen[ctx] = row.get("title", "")

    unique = list(seen.items())
    random.Random(seed).shuffle(unique)
    selected = unique[:n_samples]

    docs = [
        RawDocument(
            id=f"text_{i:04d}",
            modality="text",
            content=ctx,
            text=ctx,
            source="rajpurkar/squad",
            license="CC-BY-SA-4.0",
            metadata={"title": title},
        )
        for i, (ctx, title) in enumerate(selected)
    ]

    _write_manifest(docs, manifest)
    logger.info("text: %d docs saved → %s", len(docs), manifest)
    return docs


def load_image_captions(
    n_samples: int = 500,
    seed: int = 42,
    cache_dir: Path | str = Path("data/raw/images"),
    hf_cache_dir: Path | str | None = None,
) -> list[RawDocument]:
    """Load image + caption pairs from Flickr30k (test split, 1 000 images).

    Each image is saved as a JPEG under *cache_dir*. The first caption is used
    as the ``text`` field; all five captions are kept in ``metadata``.

    Args:
        n_samples:    Max images to keep (capped at available count).
        seed:         Shuffle seed (first build only).
        cache_dir:    Directory where JPEG files and manifest are saved.
        hf_cache_dir: Override HuggingFace cache root.

    Returns:
        List of :class:`~mmrag.ingestion.schema.RawDocument` with ``modality="image"``.
    """
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.jsonl"

    if manifest.exists():
        docs = _read_manifest(manifest)
        logger.info("image: %d docs from cache (%s)", len(docs), manifest)
        return docs

    try:
        import datasets as hf  # noqa: PLC0415
    except ImportError:
        logger.error("image: 'datasets' not installed")
        return []

    logger.info("image: downloading nlphuji/flickr30k …")
    try:
        ds = _retry(
            lambda: hf.load_dataset(
                "nlphuji/flickr30k",
                split="test",
                cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
                trust_remote_code=True,
            ),
            label="flickr30k",
        )
    except Exception as exc:
        logger.error("image: download failed (%s) — returning []", exc)
        return []

    n_avail = len(ds)
    indices = list(range(n_avail))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[: min(n_samples, n_avail)]

    docs: list[RawDocument] = []
    for i, idx in enumerate(selected_indices):
        try:
            row = ds[idx]
            pil_img = row["image"]

            raw_caps = row.get("caption", row.get("captions", []))
            if isinstance(raw_caps, str):
                all_caps: list[str] = [raw_caps]
            else:
                all_caps = [c for c in raw_caps if isinstance(c, str) and c.strip()]

            if not all_caps:
                logger.warning("image idx=%d: no caption — skipped", idx)
                continue

            img_path = cache_dir / f"image_{i:04d}.jpg"
            if not img_path.exists():
                pil_img.convert("RGB").save(img_path, format="JPEG", quality=85)

            docs.append(
                RawDocument(
                    id=f"image_{i:04d}",
                    modality="image",
                    content=str(img_path),
                    text=all_caps[0],
                    source="nlphuji/flickr30k",
                    license="Flickr-Research-Use",
                    metadata={
                        "img_id": str(row.get("img_id", idx)),
                        "filename": str(row.get("filename", "")),
                        "all_captions": all_caps,
                    },
                )
            )
        except Exception as exc:
            logger.warning("image idx=%d: skipped — %s", idx, exc)

    _write_manifest(docs, manifest)
    logger.info("image: %d docs saved → %s", len(docs), manifest)
    return docs


def load_audio_clips(
    n_samples: int = 300,
    seed: int = 42,
    cache_dir: Path | str = Path("data/raw/audio"),
    hf_cache_dir: Path | str | None = None,
) -> list[RawDocument]:
    """Load audio clips from LibriSpeech ASR (clean / test.clean split).

    Uses HuggingFace *streaming* mode so only the first *n_samples* clips are
    fetched, avoiding a full-split download (~360 MB). Clips are saved as WAV.

    Args:
        n_samples:    Number of clips to collect.
        seed:         Shuffle-buffer seed (first build only).
        cache_dir:    Directory where WAV files and manifest are saved.
        hf_cache_dir: Override HuggingFace cache root.

    Returns:
        List of :class:`~mmrag.ingestion.schema.RawDocument` with ``modality="audio"``.
    """
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.jsonl"

    if manifest.exists():
        docs = _read_manifest(manifest)
        logger.info("audio: %d docs from cache (%s)", len(docs), manifest)
        return docs

    try:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        import datasets as hf  # noqa: PLC0415
    except ImportError as exc:
        logger.error("audio: missing dependency (%s) — pip install librosa soundfile datasets", exc)
        return []

    logger.info("audio: streaming openslr/librispeech_asr clean/test.clean …")
    try:
        stream = _retry(
            lambda: hf.load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split="test.clean",
                streaming=True,
                cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
                trust_remote_code=True,
            ),
            label="librispeech",
        )
    except Exception as exc:
        logger.error("audio: download failed (%s) — returning []", exc)
        return []

    # Shuffle a window of 2×n_samples before iterating
    stream = stream.shuffle(seed=seed, buffer_size=min(2 * n_samples, 2_000))

    docs: list[RawDocument] = []
    for stream_idx, row in enumerate(stream):
        if len(docs) >= n_samples:
            break
        try:
            audio_col = row["audio"]
            array = np.array(audio_col["array"], dtype=np.float32)
            sr = int(audio_col["sampling_rate"])
            transcript: str = row.get("text", "").strip()

            if not transcript:
                logger.warning("audio stream[%d]: empty transcript — skipped", stream_idx)
                continue
            if array.size == 0:
                logger.warning("audio stream[%d]: empty array — skipped", stream_idx)
                continue

            doc_idx = len(docs)
            wav_path = cache_dir / f"audio_{doc_idx:04d}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), array, sr)

            docs.append(
                RawDocument(
                    id=f"audio_{doc_idx:04d}",
                    modality="audio",
                    content=str(wav_path),
                    text=transcript,
                    source="openslr/librispeech_asr",
                    license="CC-BY-4.0",
                    metadata={
                        "speaker_id": str(row.get("speaker_id", "")),
                        "chapter_id": str(row.get("chapter_id", "")),
                        "sample_rate": sr,
                        "duration_s": round(float(len(array)) / sr, 2),
                    },
                )
            )
        except Exception as exc:
            logger.warning("audio stream[%d]: skipped — %s", stream_idx, exc)

    _write_manifest(docs, manifest)
    logger.info("audio: %d docs saved → %s", len(docs), manifest)
    return docs
