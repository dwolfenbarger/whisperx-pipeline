"""Application configuration for WhisperX transcription service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "whisperx-gpu-service"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 17860

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    model_name: str = Field(default="large-v3", alias="MODEL_NAME")
    device: str = Field(default="cuda", alias="DEVICE")
    compute_type: str = Field(default="float16", alias="COMPUTE_TYPE")

    output_dir: Path = Field(default=Path("service/outputs"), alias="OUTPUT_DIR")
    embedding_dir: Path = Field(default=Path("service/embeddings"), alias="EMBEDDINGS_DIR")
    temp_dir: Path = Field(default=Path("service/tmp"), alias="TEMP_DIR")
    default_namespace: str = Field(default="default", alias="NAMESPACE")

    speaker_match_threshold: float = Field(default=0.75, alias="SPEAKER_MATCH_THRESHOLD")
    max_speakers: int | None = Field(default=None, alias="MAX_SPEAKERS")

    gpu_workers: int = Field(default=1, alias="GPU_WORKERS")
    event_poll_interval_seconds: float = 0.5

    chunk_seconds: int = Field(default=900, alias="CHUNK_SECONDS")
    enroll_min_seconds: int = 10
    enroll_max_seconds: int = 60


settings = Settings()


def ensure_directories() -> None:
    """Create runtime directories if they do not exist."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.embedding_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
