"""Canonical data type shared across all ingestion loaders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Modality = Literal["text", "image", "audio"]


@dataclass
class RawDocument:
    """Normalised representation of one ingested item, independent of modality.

    Fields
    ------
    id:       Unique stable identifier — ``text_0001``, ``image_0042``, ``audio_0007``.
    modality: Source media type.
    content:  For ``text``: the UTF-8 text itself.
              For ``image``/``audio``: absolute path to the saved file on disk.
    text:     Always a UTF-8 string — same as *content* for text modality,
              caption for images, transcript for audio.
    source:   HuggingFace dataset slug (e.g. ``rajpurkar/squad``).
    license:  SPDX identifier or free-form string (e.g. ``CC-BY-4.0``).
    metadata: Arbitrary scalar key/value pairs (no nested dicts/lists in Chroma).
    """

    id: str
    modality: Modality
    content: str
    text: str
    source: str
    license: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawDocument":
        d = dict(data)
        metadata = d.pop("metadata", {})
        return cls(**d, metadata=metadata)

    @classmethod
    def from_json(cls, line: str) -> "RawDocument":
        return cls.from_dict(json.loads(line))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RawDocument):
            return NotImplemented
        return self.to_dict() == other.to_dict()
