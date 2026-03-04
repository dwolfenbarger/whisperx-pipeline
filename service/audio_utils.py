"""Audio normalization and low-level audio helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


def ffmpeg_available() -> bool:
    """Return True if ffmpeg executable is available."""
    return shutil.which("ffmpeg") is not None


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with explicit error capture."""
    command = ["ffmpeg", "-y", *args]
    return subprocess.run(command, check=True, text=True, capture_output=True)


def normalize_audio(input_path: Path, output_path: Path, loudnorm: bool = False) -> Path:
    """Normalize incoming media into mono 16k WAV suitable for speech tasks."""
    filters = ["highpass=f=80", "lowpass=f=7600"]
    if loudnorm:
        filters.append("loudnorm")

    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            ",".join(filters),
            str(output_path),
        ]
    )
    return output_path


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read float32 mono waveform from file."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def duration_seconds(path: Path) -> float:
    """Get media duration from decoded waveform."""
    audio, sr = read_audio(path)
    return float(len(audio) / sr)


def chunk_intervals(total_seconds: float, chunk_seconds: int, overlap_seconds: int = 2) -> Iterable[tuple[float, float]]:
    """Yield chunk intervals with overlap for long-form processing."""
    if total_seconds <= chunk_seconds:
        yield (0.0, total_seconds)
        return

    start = 0.0
    while start < total_seconds:
        end = min(total_seconds, start + chunk_seconds)
        yield (start, end)
        start = max(end - overlap_seconds, start + 1)


def slice_audio(path: Path, start_s: float, end_s: float, out_path: Path) -> Path:
    """Write audio slice using ffmpeg for memory-safe chunking."""
    run_ffmpeg(
        [
            "-ss",
            str(start_s),
            "-to",
            str(end_s),
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ]
    )
    return out_path
