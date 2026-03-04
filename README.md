# WhisperX GPU Transcription Microservice

Production-oriented FastAPI microservice for asynchronous meeting transcription with speaker attribution and enrollment-based automatic speaker naming.

## Windows PowerShell setup (Python 3.12 + GPU)

```powershell
# 1) Create virtual environment (required launcher command)
py -3.12 -m venv whisperx-env

# 2) Activate venv
.\whisperx-env\Scripts\Activate.ps1

# 3) Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4) Install PyTorch nightly w/ CUDA 12.8+ (Blackwell / sm_120 path)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4b) Fallback if nightly cu128 wheel is unavailable
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 5) Install service dependencies
pip install -r requirements.txt

# 6) Verify CUDA visibility and GPU name
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# 7) Set auth token and optional config
$env:HF_TOKEN="<your_huggingface_token>"
$env:MODEL_NAME="large-v3"
$env:DEVICE="cuda"
$env:OUTPUT_DIR="service/outputs"
$env:NAMESPACE="default"
$env:SPEAKER_MATCH_THRESHOLD="0.75"

# 8) Run service on required bind address/port
python -m uvicorn service.app:app --host 0.0.0.0 --port 17860
```

## API usage examples

```bash
# Enroll speaker (10-60s audio sample)
curl -X POST "http://localhost:17860/speakers/enroll" \
  -F "name=Sarah Chen" \
  -F "namespace=acme-team" \
  -F "file=@./samples/sarah_voice.wav"

# Submit transcription job
curl -X POST "http://localhost:17860/transcribe" \
  -F "namespace=acme-team" \
  -F "file=@./samples/meeting.mp4"

# Poll job status
curl "http://localhost:17860/jobs/<job_id>"

# Stream SSE progress events
curl -N "http://localhost:17860/jobs/<job_id>/events"

# Fetch final JSON
curl "http://localhost:17860/jobs/<job_id>/result"

# Fetch artifacts
curl -OJ "http://localhost:17860/jobs/<job_id>/artifact/result.srt"
curl -OJ "http://localhost:17860/jobs/<job_id>/artifact/result.vtt"
curl -OJ "http://localhost:17860/jobs/<job_id>/artifact/result.rttm"
```

## Notes
- Automatic speaker naming is embedding/enrollment based only.
- If there are no enrollments or no threshold match, labels stay as `SPEAKER_XX`.
- Name hints from transcript text are returned separately as `suggested_name_hints` for manual review only.
