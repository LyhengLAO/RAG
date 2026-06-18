"""faster-whisper audio transcriber: audio path → transcript + timestamped segments.

Cache
-----
Each audio file gets a sidecar ``<audio_path>.whisper.json`` on first run.
Subsequent calls read the sidecar and make zero model inferences.
Delete the sidecar to force re-transcription.

Offline mode
------------
After weights are downloaded to ``HF_HOME``, set ``TRANSFORMERS_OFFLINE=1``
(or ``HF_DATASETS_OFFLINE=1``).  faster-whisper uses CTranslate2 which caches
models under ``HF_HOME/hub/``.

Model sizes vs. speed (CPU int8)
---------------------------------
tiny   : ~39 M params, very fast, lower accuracy
base   : ~74 M params, fast, decent accuracy        ← recommended default
small  : ~244 M params, moderate, good accuracy
medium : ~769 M params, slow, high accuracy
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


# ── result types ─────────────────────────────────────────────────────────────


@dataclass
class TranscriptSegment:
    """One timed segment of a transcript."""

    text: str
    start: float  # seconds
    end: float  # seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptResult:
    """Full transcription result for one audio file."""

    text: str  # full concatenated transcript
    language: str  # detected ISO 639-1 code (e.g. "en")
    language_probability: float
    duration_s: float
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptResult":
        segs = [TranscriptSegment(**s) for s in d.get("segments", [])]
        return cls(
            text=d["text"],
            language=d["language"],
            language_probability=d.get("language_probability", 1.0),
            duration_s=d.get("duration_s", 0.0),
            segments=segs,
        )


# ── transcriber ───────────────────────────────────────────────────────────────


class WhisperTranscriber:
    """Transcribe audio files with faster-whisper.

    Args:
        model_size:   Whisper model variant (``"tiny"`` / ``"base"`` / ``"small"``
                      / ``"medium"`` / ``"large-v3"``).
        device:       ``"cpu"`` or ``"cuda"`` (auto-detected if None).
        language:     ISO 639-1 language code, or ``None`` for auto-detection.
        beam_size:    Beam search width.  Higher = better accuracy, slower.
        batch_size:   Number of audio files to transcribe concurrently (using
                      threads; each file is still serialised through the model).
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
        batch_size: int = 1,
    ) -> None:
        self.model_size = model_size
        self.device = device or _auto_device()
        self.language = language
        self.beam_size = beam_size
        self.batch_size = batch_size
        self._model: Any = None

    # ── lazy model loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # noqa: PLC0415

        ct = _compute_type(self.device)
        logger.info(
            "WhisperTranscriber: loading %s on %s (compute_type=%s) …",
            self.model_size,
            self.device,
            ct,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=ct,
        )
        logger.info("WhisperTranscriber: model ready")

    # ── cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_path(audio_path: Path) -> Path:
        return audio_path.with_suffix(audio_path.suffix + ".whisper.json")

    @staticmethod
    def _read_cache(audio_path: Path) -> TranscriptResult | None:
        cp = WhisperTranscriber._cache_path(audio_path)
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            return TranscriptResult.from_dict(data)
        except Exception as exc:
            logger.warning("WhisperTranscriber: corrupt cache for %s (%s) — will re-transcribe", audio_path.name, exc)
            return None

    @staticmethod
    def _write_cache(audio_path: Path, result: TranscriptResult) -> None:
        cache_path = WhisperTranscriber._cache_path(audio_path)
        cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ── public API ────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribe a single audio file, returning text + timestamped segments.

        Uses the sidecar cache when available; runs inference otherwise.

        Args:
            audio_path: Path to a WAV / MP3 / FLAC / M4A file.

        Returns:
            :class:`TranscriptResult` with full text and per-segment timestamps.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        cached = self._read_cache(path)
        if cached is not None:
            logger.debug("WhisperTranscriber: cache hit %s", path.name)
            return cached

        self._load()
        result = self._infer(path)
        self._write_cache(path, result)
        return result

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
        show_progress: bool = True,
    ) -> list[TranscriptResult]:
        """Transcribe multiple audio files with a progress bar.

        Each file is processed independently; the batch loop is sequential
        (faster-whisper is already multi-threaded internally).

        Args:
            audio_paths:   Paths to audio files.
            show_progress: Display a tqdm progress bar.

        Returns:
            :class:`TranscriptResult` list in the same order as *audio_paths*.
        """
        paths = [Path(p).resolve() for p in audio_paths]

        try:
            from tqdm import tqdm  # noqa: PLC0415

            _wrap = tqdm if show_progress else (lambda x, **kw: x)
        except ImportError:
            _wrap = lambda x, **kw: x  # noqa: E731

        results: list[TranscriptResult] = []
        for path in _wrap(paths, desc="Whisper transcription", unit="file"):
            try:
                results.append(self.transcribe(path))
            except Exception as exc:
                logger.warning("WhisperTranscriber: failed on %s (%s) — inserting empty result", path.name, exc)
                results.append(
                    TranscriptResult(
                        text="",
                        language="unknown",
                        language_probability=0.0,
                        duration_s=0.0,
                        segments=[],
                    )
                )
        return results

    # ── internal inference ────────────────────────────────────────────────────

    def _infer(self, path: Path) -> TranscriptResult:
        """Run faster-whisper on *path* and return a :class:`TranscriptResult`.

        Args:
            path: Resolved path to the audio file.

        Returns:
            Full :class:`TranscriptResult`.
        """
        segments_gen, info = self._model.transcribe(
            str(path),
            beam_size=self.beam_size,
            language=self.language,
        )

        segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        for seg in segments_gen:
            text = seg.text.strip()
            segments.append(TranscriptSegment(text=text, start=seg.start, end=seg.end))
            full_text_parts.append(text)

        full_text = " ".join(full_text_parts)
        return TranscriptResult(
            text=full_text,
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration_s=round(info.duration, 2),
            segments=segments,
        )
