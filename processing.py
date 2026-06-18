import uuid
from pathlib import Path

import ffmpeg

BASE_DIR = Path(__file__).parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from a video file."""


def safe_unlink(path: str | Path | None) -> None:
    """Delete a temp file if it exists; ignore permission or race errors."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def sweep_temp_audio(max_age_seconds: int = 3600) -> int:
    """Remove orphaned WAV files older than max_age_seconds."""
    import time

    removed = 0
    now = time.time()
    for wav in TEMP_AUDIO_DIR.glob("*.wav"):
        try:
            if now - wav.stat().st_mtime >= max_age_seconds:
                wav.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def _decode_ffmpeg_error(err: ffmpeg.Error) -> str:
    if err.stderr:
        return err.stderr.decode("utf-8", errors="replace").strip()
    return str(err)


def _probe_video(path: Path) -> dict:
    try:
        return ffmpeg.probe(str(path))
    except ffmpeg.Error as e:
        raise AudioExtractionError(
            "Corrupted or unreadable video file."
        ) from e


def _has_audio_stream(probe: dict) -> bool:
    return any(
        stream.get("codec_type") == "audio"
        for stream in probe.get("streams", [])
    )


def extract_audio(video_path: str) -> str:
    """Extract mono 16 kHz PCM WAV from any video with an audio track."""
    src = Path(video_path).expanduser().resolve()

    if not src.is_file():
        raise AudioExtractionError(f"Video file not found: {video_path}")

    probe = _probe_video(src)
    if not _has_audio_stream(probe):
        raise AudioExtractionError("No audio track found in this file.")

    out = TEMP_AUDIO_DIR / f"{uuid.uuid4().hex}.wav"

    try:
        (
            ffmpeg
            .input(str(src))
            .output(
                str(out),
                acodec="pcm_s16le",
                ac=1,
                ar=16000,
                vn=None,
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        safe_unlink(out)
        stderr = _decode_ffmpeg_error(e)
        if any(
            token in stderr.lower()
            for token in ("invalid data", "moov atom not found", "could not find codec")
        ):
            raise AudioExtractionError("Corrupted or unreadable video file.") from e
        raise AudioExtractionError(
            f"Audio extraction failed: {stderr or 'unknown ffmpeg error'}"
        ) from e

    if not out.is_file() or out.stat().st_size == 0:
        safe_unlink(out)
        raise AudioExtractionError("Extraction produced an empty audio file.")

    return str(out)
