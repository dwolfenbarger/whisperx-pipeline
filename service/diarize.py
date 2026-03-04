"""pyannote diarization integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from service.config import settings

logger = logging.getLogger(__name__)


class Diarizer:
    """Lazy-loaded pyannote diarization pipeline."""

    def __init__(self) -> None:
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
        if settings.device == "cuda":
            import torch

            self._pipeline.to(torch.device("cuda"))

    def diarize(self, wav_path: Path) -> list[dict[str, Any]]:
        self._load()
        diarization = self._pipeline(str(wav_path), num_speakers=settings.max_speakers)
        segments: list[dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )
        return segments
