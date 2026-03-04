"""Enrollment and speaker matching using voice embeddings."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from service.audio_utils import duration_seconds, normalize_audio
from service.config import settings

logger = logging.getLogger(__name__)


SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class EnrollmentRecord:
    speaker_id: str
    name: str
    namespace: str
    embeddings: list[list[float]]


class SpeakerIdentifier:
    """Extract and match speaker embeddings against enrollment store."""

    def __init__(self) -> None:
        self._embedder = None
        self._backend = None

    @property
    def backend(self) -> str:
        if self._backend is None:
            self._load_embedder()
        return self._backend or "unknown"

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

            model = Model.from_pretrained("pyannote/embedding", use_auth_token=settings.hf_token)
            self._embedder = PretrainedSpeakerEmbedding(model, device=settings.device)
            self._backend = "pyannote"
            logger.info("Loaded pyannote speaker embedding model")
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("pyannote embedding unavailable: %s", exc)

        from speechbrain.inference.speaker import EncoderClassifier

        self._embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(settings.temp_dir / "speechbrain-ecapa"),
        )
        self._backend = "speechbrain"
        logger.info("Loaded speechbrain ECAPA embedding model")

    def _safe_key(self, name: str) -> str:
        return SAFE_NAME_PATTERN.sub("_", name.strip().lower()).strip("_") or "speaker"

    def _namespace_path(self, namespace: str) -> Path:
        path = settings.embedding_dir / self._safe_key(namespace)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _record_path(self, namespace: str, name: str) -> Path:
        return self._namespace_path(namespace) / f"{self._safe_key(name)}.json"

    def _extract_embedding(self, wav_path: Path) -> np.ndarray:
        self._load_embedder()
        if self._backend == "pyannote":
            import torch

            waveform, sr = self._read_tensor(wav_path)
            with torch.no_grad():
                emb = self._embedder(waveform[None], sr)
            return np.asarray(emb).reshape(-1)

        import torch

        signal, sr = self._read_tensor(wav_path)
        with torch.no_grad():
            emb = self._embedder.encode_batch(signal[None])
        return emb.squeeze().cpu().numpy().reshape(-1)

    def _read_tensor(self, wav_path: Path):
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.to(torch.float32), sr

    def enroll(self, name: str, namespace: str, input_media: Path) -> dict:
        """Create or update a speaker enrollment with one additional sample."""
        namespace = namespace or settings.default_namespace
        record_path = self._record_path(namespace, name)
        normalized = settings.temp_dir / f"enroll_{self._safe_key(name)}.wav"
        normalize_audio(input_media, normalized, loudnorm=True)

        seconds = duration_seconds(normalized)
        if seconds < settings.enroll_min_seconds or seconds > settings.enroll_max_seconds:
            raise ValueError(f"Enrollment audio must be {settings.enroll_min_seconds}-{settings.enroll_max_seconds}s")

        embedding = self._extract_embedding(normalized).tolist()
        if record_path.exists():
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "speaker_id": f"{self._safe_key(namespace)}:{self._safe_key(name)}",
                "name": name,
                "namespace": namespace,
                "embeddings": [],
                "backend": self.backend,
            }

        payload["embeddings"].append(embedding)
        record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "speaker_id": payload["speaker_id"],
            "name": payload["name"],
            "namespace": payload["namespace"],
            "samples": len(payload["embeddings"]),
            "backend": payload.get("backend", self.backend),
            "record_path": str(record_path),
        }

    def load_enrollments(self, namespace: str) -> list[EnrollmentRecord]:
        records: list[EnrollmentRecord] = []
        ns_dir = self._namespace_path(namespace)
        for path in ns_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                EnrollmentRecord(
                    speaker_id=data["speaker_id"],
                    name=data["name"],
                    namespace=data["namespace"],
                    embeddings=data.get("embeddings", []),
                )
            )
        return records

    def build_cluster_embedding(self, wav_path: Path) -> np.ndarray:
        return self._extract_embedding(wav_path)

    def match(self, cluster_embedding: np.ndarray, namespace: str) -> tuple[str | None, float]:
        enrollments = self.load_enrollments(namespace)
        if not enrollments:
            return None, 0.0

        best_name = None
        best_score = -1.0
        for record in enrollments:
            vectors = [np.asarray(v, dtype=np.float32) for v in record.embeddings]
            if not vectors:
                continue
            centroid = np.mean(np.vstack(vectors), axis=0)
            score = cosine_similarity(cluster_embedding, centroid)
            if score > best_score:
                best_score = score
                best_name = record.name

        if best_score >= settings.speaker_match_threshold:
            return best_name, best_score
        return None, best_score


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
