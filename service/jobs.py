"""Async in-process job queue, persistence, and event streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from service.config import settings
from service.pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)


@dataclass
class JobState:
    id: str
    filename: str
    namespace: str
    language: str | None
    status: str = "queued"
    progress: int = 0
    message: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    result_path: str | None = None
    error: str | None = None


class JobManager:
    """Single-worker GPU queue manager with persisted job index."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()
        self.jobs: dict[str, JobState] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.lock = asyncio.Lock()
        self.pipeline = TranscriptionPipeline()
        self.worker_task: asyncio.Task | None = None
        self.index_file = settings.output_dir / "jobs.json"
        self._load_index()

    def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker_loop())

    async def enqueue(self, source_file: Path, filename: str, namespace: str, language: str | None) -> str:
        job_id = uuid.uuid4().hex
        state = JobState(id=job_id, filename=filename, namespace=namespace, language=language)
        async with self.lock:
            self.jobs[job_id] = state
            self.events[job_id] = [{"message": "queued", "progress": 0}]
            self._persist_index()

        await self.queue.put((job_id, source_file))
        return job_id

    def get(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    def result(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if not job or not job.result_path:
            return None
        path = Path(job.result_path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    async def stream_events(self, job_id: str) -> AsyncGenerator[dict[str, str], None]:
        cursor = 0
        while True:
            job = self.jobs.get(job_id)
            if job is None:
                yield {"event": "error", "data": json.dumps({"message": "job_not_found"})}
                break

            pending = self.events.get(job_id, [])
            while cursor < len(pending):
                event = pending[cursor]
                cursor += 1
                yield {"event": "progress", "data": json.dumps(event)}

            if job.status in {"completed", "failed"} and cursor >= len(pending):
                break
            await asyncio.sleep(settings.event_poll_interval_seconds)

    async def _worker_loop(self) -> None:
        while True:
            job_id, source_file = await self.queue.get()
            state = self.jobs[job_id]
            job_dir = settings.output_dir / job_id
            try:
                self._update(job_id, "running", 1, "started")
                result = await asyncio.to_thread(
                    self.pipeline.run,
                    source_file,
                    job_dir,
                    state.namespace,
                    state.language,
                    lambda msg, pct: self._update(job_id, "running", pct, msg),
                )
                _ = result
                self._update(job_id, "completed", 100, "completed", result_path=str(job_dir / "result.json"))
            except Exception as exc:
                tb = traceback.format_exc() if settings.debug else ""
                logger.exception("Job %s failed", job_id)
                self._update(job_id, "failed", 100, "failed", error=f"{exc}\n{tb}")
            finally:
                self.queue.task_done()

    def _update(self, job_id: str, status: str, progress: int, message: str, **kwargs: Any) -> None:
        state = self.jobs[job_id]
        state.status = status
        state.progress = progress
        state.message = message
        state.updated_at = datetime.now(UTC).isoformat()
        for key, value in kwargs.items():
            setattr(state, key, value)
        self.events.setdefault(job_id, []).append({"message": message, "progress": progress, "status": status})
        self._persist_index()

    def _persist_index(self) -> None:
        payload = {job_id: asdict(job) for job_id, job in self.jobs.items()}
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_index(self) -> None:
        if not self.index_file.exists():
            return
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
            for job_id, item in data.items():
                self.jobs[job_id] = JobState(**item)
                self.events[job_id] = []
        except Exception:
            logger.warning("Could not load jobs index, starting fresh")
