"""Merge aligned words/segments with diarization speaker turns."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

NAME_HINT_RE = re.compile(r"\b(?:hey\s+)?([A-Z][a-z]{1,20})(?:,|\s+can\s+you|\s+could\s+you)")


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def pick_speaker(start: float, end: float, turns: list[dict[str, Any]]) -> str:
    best = (0.0, "SPEAKER_00")
    for turn in turns:
        ov = overlap(start, end, turn["start"], turn["end"])
        if ov > best[0]:
            best = (ov, turn["speaker"])
    return best[1]


def merge_alignment(
    transcript: dict[str, Any], diarization_turns: list[dict[str, Any]], speaker_labels: dict[str, str] | None = None
) -> dict[str, Any]:
    """Attach speaker labels to aligned segments and words."""
    speaker_labels = speaker_labels or {}
    merged_segments = []
    hinted_names: dict[str, list[str]] = defaultdict(list)

    for segment in transcript.get("aligned_segments") or transcript.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        diarized_speaker = pick_speaker(start, end, diarization_turns)
        resolved_speaker = speaker_labels.get(diarized_speaker, diarized_speaker)
        text = segment.get("text", "")

        hints = [m.group(1) for m in NAME_HINT_RE.finditer(text)]
        for hint in hints:
            hinted_names[diarized_speaker].append(hint)

        words = []
        for word in segment.get("words", []):
            w_start = float(word.get("start", start))
            w_end = float(word.get("end", w_start))
            word_speaker = pick_speaker(w_start, w_end, diarization_turns)
            words.append({**word, "speaker": speaker_labels.get(word_speaker, word_speaker)})

        merged_segments.append(
            {
                **segment,
                "speaker": resolved_speaker,
                "speaker_cluster": diarized_speaker,
                "suggested_name_hints": sorted(set(hints)),
                "words": words,
            }
        )

    transcript["merged_segments"] = merged_segments
    transcript["suggested_name_hints"] = {k: sorted(set(v)) for k, v in hinted_names.items()}
    return transcript
