"""Whisper/WhisperX transcription integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from service.config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """Lazy-loaded WhisperX pipeline."""

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import whisperx

        logger.info("Loading WhisperX model=%s device=%s", settings.model_name, settings.device)
        self._model = whisperx.load_model(
            settings.model_name,
            device=settings.device,
            compute_type=settings.compute_type,
        )

    def transcribe(self, wav_path: Path, language: str | None = None) -> dict[str, Any]:
        self._load_model()
        import whisperx

        audio = whisperx.load_audio(str(wav_path))
        result = self._model.transcribe(audio, language=language, batch_size=16)
        return result

    def align(self, transcript: dict[str, Any], wav_path: Path) -> dict[str, Any]:
        import whisperx

        language_code = transcript.get("language", "en")
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=settings.device)
        audio = whisperx.load_audio(str(wav_path))
        aligned = whisperx.align(transcript["segments"], model_a, metadata, audio, settings.device, return_char_alignments=False)
        transcript["aligned_segments"] = aligned.get("segments", [])
        transcript["word_segments"] = aligned.get("word_segments", [])
        return transcript
