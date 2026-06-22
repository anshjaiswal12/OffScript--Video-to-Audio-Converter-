import asyncio
import json
import logging
import logging.config
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from batch import build_transcripts_zip, create_batch, get_batch, record_batch_result
from processing import (
    ALLOWED_EXTENSIONS,
    AudioExtractionError,
    extract_audio,
    is_supported_extension,
    safe_unlink,
    sweep_temp_audio,
)
from streaming import hub
from transcription import TranscriptionError, maybe_unload_model, transcribe_audio

# ---------------------------------------------------------------------------
# Logging setup — configure once at import time
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "transcription": {"level": "DEBUG"},
        "processing": {"level": "DEBUG"},
        "streaming": {"level": "DEBUG"},
        "batch": {"level": "DEBUG"},
    },
})

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

for _dir in (STATIC_DIR, UPLOADS_DIR, OUTPUTS_DIR):
    _dir.mkdir(exist_ok=True)

JOB_TTL_SECONDS = 3600
MAINTENANCE_INTERVAL_SECONDS = 120

# 4 GiB upload limit — large files should use the local-path endpoint instead
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024

_jobs: dict[str, dict[str, str | float]] = {}
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")


class BatchUploadRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    directory: str | None = None


class BatchDownloadRequest(BaseModel):
    files: list[str] = Field(default_factory=list)


async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


def _register_job(audio_path: str, filename: str) -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "audio_path": audio_path,
        "filename": filename,
        "created_at": time.time(),
    }
    return job_id


def _take_job(job_id: str) -> dict[str, str | float] | None:
    return _jobs.pop(job_id, None)


def _cleanup_stale_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    stale_ids = [
        job_id
        for job_id, job in _jobs.items()
        if float(job["created_at"]) < cutoff
    ]
    for job_id in stale_ids:
        job = _jobs.pop(job_id, None)
        if job:
            safe_unlink(str(job["audio_path"]))
    if stale_ids:
        logger.info("Cleaned up %d stale job(s).", len(stale_ids))


async def maintenance_loop() -> None:
    sweep_temp_audio()
    while True:
        await asyncio.sleep(MAINTENANCE_INTERVAL_SECONDS)
        _cleanup_stale_jobs()
        sweep_temp_audio()
        hub.cleanup_stale()
        await run_blocking(maybe_unload_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OffScript starting up …")
    sweep_temp_audio()
    task = asyncio.create_task(maintenance_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    _cleanup_stale_jobs()
    sweep_temp_audio()
    await run_blocking(maybe_unload_model, 0)
    _executor.shutdown(wait=False, cancel_futures=True)
    logger.info("OffScript shut down cleanly.")


app = FastAPI(title="OffScript", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ext_error(filename: str) -> JSONResponse:
    ext = Path(filename).suffix.lower() or "(none)"
    return JSONResponse(
        status_code=415,
        content={
            "status": "error",
            "detail": (
                f"Unsupported file type '{ext}'. "
                f"Accepted extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(html)
    return {"message": "Frontend not found"}


@app.get("/api/status")
async def status():
    return {"status": "idle", "active_jobs": len(_jobs)}


@app.post("/api/stream/open")
async def open_stream():
    stream_id = hub.open(asyncio.get_running_loop())
    return {"stream_id": stream_id}


@app.get("/api/stream-progress")
async def stream_progress(stream_id: str):
    if not hub.exists(stream_id):
        return JSONResponse(status_code=404, content={"detail": "Unknown stream_id"})

    async def event_source():
        async for payload in hub.listen(stream_id):
            yield {"event": "progress", "data": json.dumps(payload, ensure_ascii=False)}

    return EventSourceResponse(event_source())


@app.post("/api/upload-batch")
async def upload_batch(body: BatchUploadRequest):
    if not body.paths and not body.directory:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Provide paths and/or directory"},
        )

    batch = create_batch(paths=body.paths, directory=body.directory)
    if batch["total"] == 0:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "No files found for batch queue"},
        )

    return {
        "status": "queued",
        "batch_id": batch["batch_id"],
        "total": batch["total"],
        "queue": batch["queue"],
    }


@app.get("/api/batch/{batch_id}")
async def batch_status(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        return JSONResponse(status_code=404, content={"detail": "Batch not found"})
    return {
        "batch_id": batch["batch_id"],
        "status": batch["status"],
        "total": batch["total"],
        "completed": batch["completed"],
        "failed": batch["failed"],
    }


@app.post("/api/extract")
async def extract_step(
    file: UploadFile | None = File(None),
    path: str | None = Form(None),
):
    temp_video: Path | None = None
    filename: str | None = None
    audio_path: str | None = None

    try:
        if path:
            video_path = Path(path).expanduser().resolve()
            filename = video_path.name
            if not video_path.is_file():
                raise AudioExtractionError(f"Video file not found: {path}")
            if not is_supported_extension(filename):
                return _ext_error(filename)
        elif file and file.filename:
            filename = Path(file.filename).name
            if not is_supported_extension(filename):
                return _ext_error(filename)
            # Check content-length header for early size rejection
            content_length = getattr(file, "size", None)
            if content_length is not None and content_length > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "status": "error",
                        "detail": f"File too large ({content_length // (1024**3)} GiB). Maximum is 4 GiB.",
                    },
                )
            temp_video = UPLOADS_DIR / f"{uuid.uuid4().hex}_{filename}"
            with temp_video.open("wb") as buffer:
                await run_blocking(shutil.copyfileobj, file.file, buffer)
            # Post-write size check
            if temp_video.stat().st_size > MAX_UPLOAD_BYTES:
                raise AudioExtractionError(
                    f"Uploaded file exceeds the 4 GiB limit."
                )
            video_path = temp_video
        else:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "Provide a file upload or local path"},
            )

        audio_path = await run_blocking(extract_audio, str(video_path))
        job_id = _register_job(audio_path, filename)
        logger.info("Extracted audio for '%s' → job %s", filename, job_id)
        return {"status": "extracted", "job_id": job_id, "filename": filename}

    except AudioExtractionError as exc:
        safe_unlink(audio_path)
        logger.warning("Audio extraction failed for '%s': %s", filename, exc)
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": str(exc)},
        )
    finally:
        safe_unlink(temp_video)


@app.post("/api/transcribe-step")
async def transcribe_step(
    job_id: str = Form(...),
    stream_id: str | None = Form(None),
    keep_stream: bool = Form(False),
    file_index: int = Form(0),
    file_count: int = Form(1),
    language: str | None = Form(None),
    batch_id: str | None = Form(None),
):
    job = _take_job(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "detail": "Job not found or expired"},
        )

    audio_path = str(job["audio_path"])
    filename = str(job["filename"])
    transcript_path: str | None = None

    def on_event(payload: dict) -> None:
        if stream_id:
            hub.emit(stream_id, payload)

    try:
        stem = Path(filename).stem
        txt_path = OUTPUTS_DIR / f"{stem}_{uuid.uuid4().hex[:8]}.txt"
        transcript_path = await run_blocking(
            transcribe_audio,
            audio_path,
            str(txt_path),
            filename,
            on_event if stream_id else None,
            file_index,
            file_count,
            language,
        )
    except TranscriptionError as exc:
        if batch_id:
            record_batch_result(batch_id, filename, success=False, error=str(exc))
        if stream_id:
            hub.emit(
                stream_id,
                {
                    "file_name": filename,
                    "percent": 0,
                    "live_text": "",
                    "status": "error",
                    "error": str(exc),
                },
            )
            if not keep_stream:
                hub.close(stream_id)
        logger.warning("Transcription error for '%s': %s", filename, exc)
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": str(exc)},
        )
    finally:
        safe_unlink(audio_path)

    name = Path(transcript_path).name
    download_url = f"/api/transcript/{name}"

    if batch_id:
        record_batch_result(batch_id, filename, success=True, transcript_path=transcript_path)

    if stream_id:
        hub.emit(
            stream_id,
            {
                "file_name": filename,
                "percent": 100,
                "live_text": "",
                "status": "complete",
                "download_url": download_url,
            },
        )
        if not keep_stream:
            hub.close(stream_id)

    logger.info("Transcription complete: '%s' → '%s'", filename, name)
    return {
        "status": "complete",
        "filename": filename,
        "transcript_path": transcript_path,
        "download_url": download_url,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """One-shot: upload → extract → transcribe in a single request (no SSE progress)."""
    filename = Path(file.filename or "upload").name
    if not is_supported_extension(filename):
        return _ext_error(filename)

    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{filename}"
    audio_path: str | None = None
    transcript_path: str | None = None

    try:
        with dest.open("wb") as buffer:
            await run_blocking(shutil.copyfileobj, file.file, buffer)

        if dest.stat().st_size > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"status": "error", "detail": "File exceeds the 4 GiB upload limit."},
            )

        audio_path = await run_blocking(extract_audio, str(dest))

        stem = Path(file.filename or "transcript").stem
        txt_path = OUTPUTS_DIR / f"{stem}_{uuid.uuid4().hex[:8]}.txt"
        transcript_path = await run_blocking(
            transcribe_audio,
            audio_path,
            str(txt_path),
            file.filename or "upload",
            None,
            0,
            1,
            language,
        )
    except AudioExtractionError as exc:
        logger.warning("Extraction failed (one-shot) for '%s': %s", filename, exc)
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": str(exc)},
        )
    except TranscriptionError as exc:
        logger.warning("Transcription failed (one-shot) for '%s': %s", filename, exc)
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": str(exc)},
        )
    finally:
        safe_unlink(dest)
        safe_unlink(audio_path)

    name = Path(transcript_path).name
    return {
        "status": "complete",
        "filename": file.filename,
        "transcript_path": transcript_path,
        "download_url": f"/api/transcript/{name}",
    }


@app.get("/api/transcript/{name}")
async def download_transcript(name: str):
    # Sanitise: strip any path components to prevent directory traversal
    safe_name = Path(name).name
    path = OUTPUTS_DIR / safe_name
    if not path.is_file():
        return JSONResponse(status_code=404, content={"detail": "Transcript not found"})
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.post("/api/download-batch")
async def download_batch(body: BatchDownloadRequest):
    if not body.files:
        return JSONResponse(status_code=400, content={"detail": "No transcript files provided"})

    payload = await run_blocking(build_transcripts_zip, body.files, OUTPUTS_DIR)
    if not payload:
        return JSONResponse(status_code=404, content={"detail": "No transcripts found on disk"})

    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="offscript_transcripts.zip"'},
    )
