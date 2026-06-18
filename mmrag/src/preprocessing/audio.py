"""Audio preprocessing: mel-spectrogram, MFCCs, and feature tensor extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_audio_features(
    file_path: str | Path,
    sample_rate: int = 16_000,
    n_mels: int = 128,
    n_mfcc: int = 40,
) -> dict[str, np.ndarray]:
    """Load an audio file and extract mel-spectrogram and MFCC features with librosa.

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate for resampling.
        n_mels: Number of mel filterbanks.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Dict with keys ``"mel_spectrogram"`` and ``"mfcc"``, each a ``numpy.ndarray``.
    """
    import librosa  # noqa: PLC0415

    waveform, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)
    waveform = normalize_audio(waveform)

    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)

    return {"mel_spectrogram": mel_db, "mfcc": mfcc}


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize a waveform to [-1, 1].

    Args:
        waveform: 1-D float array.

    Returns:
        Normalized array of the same shape.
    """
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform
    return waveform / peak
