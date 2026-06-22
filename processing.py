import logging
import time
import uuid
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Supported media file extensions (checked on upload to give early feedback)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    # Video
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv",
    ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".f4v", ".rm", ".rmvb",
    # Audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus",
    ".aiff", ".aif", ".au", ".ra",
})


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from a video file."""


def safe_unlink(path: str | Path | None) -> None:
    """Delete a temp file if it exists; silently ignore permission or race errors."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def is_supported_extension(filename: str) -> bool:
    """Return True if the file extension is in the supported media list."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def sweep_temp_audio(max_age_seconds: int = 3600) -> int:
    """Remove orphaned WAV files older than *max_age_seconds*. Returns count removed."""
    removed = 0
    now = time.time()
    for wav in TEMP_AUDIO_DIR.glob("*.wav"):
        try:
            if now - wav.stat().st_mtime >= max_age_seconds:
                wav.unlink()
                removed += 1
                logger.debug("Swept orphaned temp file: %s", wav.name)
        except OSError:
            pass
    if removed:
        logger.info("Swept %d orphaned temp audio file(s).", removed)
    return removed


def _decode_ffmpeg_error(err: ffmpeg.Error) -> str:
    if err.stderr:
        return err.stderr.decode("utf-8", errors="replace").strip()
    return str(err)


def _probe_video(path: Path) -> dict:
    try:
        return ffmpeg.probe(str(path))
    except ffmpeg.Error as e:
        raise AudioExtractionError("Corrupted or unreadable video file.") from e


def _has_audio_stream(probe: dict) -> bool:
    return any(
        stream.get("codec_type") == "audio"
        for stream in probe.get("streams", [])
    )


# FFmpeg stderr substrings that indicate a corrupt / unreadable input
_CORRUPTION_TOKENS = (
    "invalid data",
    "moov atom not found",
    "could not find codec",
    "invalid pts",
    "no such file or directory",
    "end of file",
    "error while decoding",
    "invalid packet size",
    "broken pipe",
)


def extract_audio(video_path: str) -> str:
    """Extract a mono 16 kHz PCM WAV from any video/audio file with an audio track.

    Raises AudioExtractionError for all failure modes with a human-readable message.
    """
    src = Path(video_path).expanduser().resolve()

    if not src.is_file():
        raise AudioExtractionError(f"Video file not found: {video_path}")

    if src.stat().st_size == 0:
        raise AudioExtractionError("The file is empty (0 bytes).")

    logger.debug("Probing '%s' …", src.name)
    probe = _probe_video(src)
    if not _has_audio_stream(probe):
        raise AudioExtractionError(
            "No audio track found in this file. "
            "Make sure the video contains audio before transcribing."
        )

    out = TEMP_AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    logger.debug("Extracting audio from '%s' → '%s'", src.name, out.name)

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
        stderr_lower = stderr.lower()
        if any(token in stderr_lower for token in _CORRUPTION_TOKENS):
            raise AudioExtractionError(
                "Corrupted or unreadable video file. "
                "Try re-downloading or re-encoding the file."
            ) from e
        raise AudioExtractionError(
            f"Audio extraction failed: {stderr or 'unknown FFmpeg error'}"
        ) from e

    if not out.is_file() or out.stat().st_size == 0:
        safe_unlink(out)
        raise AudioExtractionError(
            "Extraction produced an empty audio file. "
            "The video may have a corrupt audio track."
        )

    logger.info("Audio extracted: '%s' (%.1f KB)", out.name, out.stat().st_size / 1024)
    return str(out)
