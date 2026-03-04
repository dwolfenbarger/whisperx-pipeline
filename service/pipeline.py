"""End-to-end transcription pipeline orchestration."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable

from service.align_merge import merge_alignment
from service.artifacts import write_json, write_rttm, write_srt, write_vtt
from service.audio_utils import normalize_audio, slice_audio
from service.config import settings
from service.diarize import Diarizer
from service.speaker_id import SpeakerIdentifier
from service.transcribe import Transcriber

logger = logging.getLogger(__name__)
ProgressCb = Callable[[str, int], None]


class TranscriptionPipeline:
    """Orchestrates normalization, transcription, alignment, diarization and speaker naming."""

    def __init__(self) -> None:
        self.transcriber = Transcriber()
        self.diarizer = Diarizer()
        self.speaker_id = SpeakerIdentifier()

    def run(
        self,
        source_file: Path,
        job_dir: Path,
        namespace: str,
        language: str | None,
        progress: ProgressCb,
    ) -> dict:
        job_dir.mkdir(parents=True, exist_ok=True)
        normalized = job_dir / "normalized.wav"

        progress("normalizing_audio", 10)
        normalize_audio(source_file, normalized, loudnorm=True)

        progress("transcribing", 35)
        transcript = self.transcriber.transcribe(normalized, language=language)

        progress("aligning", 55)
        transcript = self.transcriber.align(transcript, normalized)

        progress("diarizing", 70)
        diarization_turns = self.diarizer.diarize(normalized)

        progress("matching_speakers", 82)
        speaker_map, speaker_scores = self._match_speakers(diarization_turns, normalized, namespace, job_dir)

        progress("merging", 90)
        merged = merge_alignment(transcript, diarization_turns, speaker_labels=speaker_map)

        payload = {
            "job_id": job_dir.name,
            "namespace": namespace,
            "language": transcript.get("language"),
            "speaker_matches": speaker_scores,
            "segments": merged["merged_segments"],
            "suggested_name_hints": merged.get("suggested_name_hints", {}),
            "raw": {
                "transcript": transcript,
                "diarization": diarization_turns,
            },
        }

        progress("writing_artifacts", 96)
        result_json = job_dir / "result.json"
        write_json(result_json, payload)
        write_srt(job_dir / "result.srt", payload["segments"])
        write_vtt(job_dir / "result.vtt", payload["segments"])
        write_rttm(job_dir / "result.rttm", diarization_turns, file_id=job_dir.name)

        progress("complete", 100)
        return payload

    def _match_speakers(
        self,
        diarization_turns: list[dict],
        normalized_wav: Path,
        namespace: str,
        job_dir: Path,
    ) -> tuple[dict[str, str], dict[str, dict]]:
        by_speaker: dict[str, list[dict]] = {}
        for turn in diarization_turns:
            by_speaker.setdefault(turn["speaker"], []).append(turn)

        speaker_map: dict[str, str] = {}
        score_map: dict[str, dict] = {}

        for cluster, turns in by_speaker.items():
            turns_sorted = sorted(turns, key=lambda t: t["end"] - t["start"], reverse=True)
            best_turn = turns_sorted[0]
            sample_path = job_dir / f"cluster_{cluster}_{uuid.uuid4().hex[:8]}.wav"
            slice_audio(normalized_wav, best_turn["start"], best_turn["end"], sample_path)

            emb = self.speaker_id.build_cluster_embedding(sample_path)
            matched_name, score = self.speaker_id.match(emb, namespace=namespace)
            if matched_name:
                speaker_map[cluster] = matched_name
            score_map[cluster] = {
                "matched_name": matched_name,
                "score": score,
                "threshold": settings.speaker_match_threshold,
            }

        return speaker_map, score_map
