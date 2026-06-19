"""Low-level parsers for each modality: PDF, audio, image."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.config import settings
from src.ingestion.schema import RawDocument


def parse_pdf(file_path: str | Path) -> list[RawDocument]:
    """Extract text from a PDF file page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        One ``RawDocument`` (``modality="text"``) per non-empty page, with
        ``metadata["page"]`` (0-indexed) and ``metadata["n_pages"]``. Blank
        pages (e.g. scanned images with no extractable text layer) are skipped.
    """
    from pypdf import PdfReader  # noqa: PLC0415

    file_path = Path(file_path)
    reader = PdfReader(str(file_path))
    n_pages = len(reader.pages)

    docs: list[RawDocument] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            RawDocument(
                id=f"{file_path.stem}_p{i:04d}",
                modality="text",
                content=text,
                text=text,
                source=str(file_path),
                license="user-upload",
                metadata={"page": i, "n_pages": n_pages},
            )
        )
    return docs


@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str, device: str):
    from faster_whisper import WhisperModel  # noqa: PLC0415

    return WhisperModel(model_size, device=device)


def parse_audio(
    file_path: str | Path,
    model_size: str | None = None,
    device: str | None = None,
    language: str | None = None,
) -> list[RawDocument]:
    """Transcribe an audio file with faster-whisper and return RawDocuments by segment.

    Args:
        file_path: Path to the audio file (wav, mp3, m4a, …).
        model_size: Whisper model variant (tiny/base/small/medium/large-v3).
            Defaults to ``settings.whisper_model_size``.
        device: Compute device (``"cpu"`` or ``"cuda"``). Defaults to
            ``settings.whisper_device``.
        language: ISO 639-1 language code, or ``None`` for auto-detection.

    Returns:
        One ``RawDocument`` (``modality="audio"``) per non-empty transcript
        segment, with ``metadata["start"]`` / ``["end"]`` in seconds.
    """
    file_path = Path(file_path)
    model_size = model_size or settings.whisper_model_size
    device = device or settings.whisper_device

    model = _load_whisper_model(model_size, device)
    segments, _info = model.transcribe(str(file_path), language=language)

    docs: list[RawDocument] = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        docs.append(
            RawDocument(
                id=f"{file_path.stem}_s{i:04d}",
                modality="audio",
                content=str(file_path),
                text=text,
                source=str(file_path),
                license="user-upload",
                metadata={"start": round(seg.start, 2), "end": round(seg.end, 2)},
            )
        )
    return docs


@lru_cache(maxsize=2)
def _load_blip_model(model_name: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: PLC0415

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def parse_image(
    file_path: str | Path,
    caption_model: str | None = None,
) -> list[RawDocument]:
    """Extract a text description from an image via BLIP captioning.

    Args:
        file_path: Path to the image file (jpg, png, webp, …).
        caption_model: Optional HuggingFace model id override
            (default: ``Salesforce/blip-image-captioning-base``).

    Returns:
        Single-element list with the generated caption as ``text``
        (``modality="image"``).
    """
    from PIL import Image  # noqa: PLC0415

    file_path = Path(file_path)
    model_name = caption_model or "Salesforce/blip-image-captioning-base"
    processor, model = _load_blip_model(model_name)

    image = Image.open(file_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    return [
        RawDocument(
            id=file_path.stem,
            modality="image",
            content=str(file_path),
            text=caption,
            source=str(file_path),
            license="user-upload",
            metadata={"caption_model": model_name, "width": image.width, "height": image.height},
        )
    ]
