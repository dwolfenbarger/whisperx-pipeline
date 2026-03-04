"""FastAPI entrypoint for GPU transcription microservice."""

from __future__ import annotations

import json
import logging
import platform
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from service.config import ensure_directories, settings
from service.jobs import JobManager
from service.speaker_id import SpeakerIdentifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX GPU Transcription Service", version="1.0.0")
job_manager = JobManager()
speaker_identifier = SpeakerIdentifier()


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize worker and runtime directories."""
    ensure_directories()
    job_manager.start()


@app.get("/health", summary="Service and dependency health")
def health() -> dict:
    """Report process, ffmpeg and CUDA health metadata."""
    import torch

    cuda_ok = torch.cuda.is_available()
    return {
        "ok": True,
        "python": platform.python_version(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "torch": torch.__version__,
        "cuda_available": cuda_ok,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else None,
        "device": settings.device,
        "model_name": settings.model_name,
    }


@app.post("/speakers/enroll", summary="Enroll a known speaker for automatic naming")
async def enroll_speaker(
    name: str = Form(...),
    namespace: str = Form(default=settings.default_namespace),
    file: UploadFile = File(...),
) -> dict:
    """Enroll a speaker from a 10-60 second sample for future matching."""
    tmp_path = settings.temp_dir / f"enroll_upload_{file.filename}"
    try:
        tmp_path.write_bytes(await file.read())
        record = speaker_identifier.enroll(name=name, namespace=namespace, input_media=tmp_path)
        return {"ok": True, **record}
    except Exception as exc:
        detail = str(exc)
        if settings.debug:
            detail = f"{detail}\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=detail) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/transcribe", summary="Queue a transcription job")
async def transcribe(
    file: UploadFile = File(...),
    namespace: str = Form(default=settings.default_namespace),
    language: str | None = Form(default=None),
) -> dict:
    """Submit media for asynchronous transcription and speaker attribution."""
    source = settings.temp_dir / f"upload_{file.filename}"
    source.write_bytes(await file.read())
    job_id = await job_manager.enqueue(source, file.filename, namespace, language)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", summary="Get job status")
def job_status(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    payload = job.__dict__.copy()
    payload["artifacts"] = {
        "json": f"/jobs/{job_id}/artifact/result.json",
        "srt": f"/jobs/{job_id}/artifact/result.srt",
        "vtt": f"/jobs/{job_id}/artifact/result.vtt",
        "rttm": f"/jobs/{job_id}/artifact/result.rttm",
    }
    return payload


@app.get("/jobs/{job_id}/events", summary="Stream job progress as SSE")
async def job_events(job_id: str) -> StreamingResponse:
    async def event_gen():
        async for event in job_manager.stream_events(job_id):
            yield f"event: {event['event']}\ndata: {event['data']}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/result", summary="Get final JSON result")
def job_result(job_id: str):
    payload = job_manager.result(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="result_not_ready")
    return JSONResponse(content=payload)


@app.get("/jobs/{job_id}/artifact/{name}", summary="Download generated artifact")
def download_artifact(job_id: str, name: str):
    allowed = {"result.json", "result.srt", "result.vtt", "result.rttm", "normalized.wav"}
    if name not in allowed:
        raise HTTPException(status_code=400, detail="artifact_not_allowed")
    path = settings.output_dir / job_id / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact_not_found")
    return FileResponse(path=path)
