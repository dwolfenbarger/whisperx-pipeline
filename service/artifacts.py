"""Artifact serialization utilities for transcript outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _timestamp_srt(seconds: float) -> str:
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _timestamp_vtt(seconds: float) -> str:
    return _timestamp_srt(seconds).replace(",", ".")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_timestamp_srt(seg['start'])} --> {_timestamp_srt(seg['end'])}")
        lines.append(f"[{seg['speaker']}] {seg.get('text', '').strip()}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_timestamp_vtt(seg['start'])} --> {_timestamp_vtt(seg['end'])}")
        lines.append(f"<{seg['speaker']}>{seg.get('text', '').strip()}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rttm(path: Path, diarization_turns: list[dict[str, Any]], file_id: str) -> None:
    lines = []
    for turn in diarization_turns:
        start = turn["start"]
        dur = max(0.0, turn["end"] - turn["start"])
        speaker = turn["speaker"]
        lines.append(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>")
    path.write_text("\n".join(lines), encoding="utf-8")
