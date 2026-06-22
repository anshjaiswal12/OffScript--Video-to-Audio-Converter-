import logging
import time
import uuid
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Files longer than this (seconds) are split before transcription to avoid OOM.
# 45 min is a safe upper limit for ~5 GB RAM machines with the Whisper base model.
CHUNK_THRESHOLD_SECONDS: int = 45 * 60   # 45 minutes
CHUNK_DURATION_SECONDS:  int = 30 * 60   # each chunk is 30 minutes

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


def get_audio_duration(wav_path: str) -> float:
    """Return the duration of a WAV file in seconds (0.0 on failure)."""
    try:
        info = ffmpeg.probe(wav_path)
        return float(info.get("format", {}).get("duration", 0) or 0)
    except Exception:
        return 0.0


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
    Returns the path to the extracted WAV file.
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

    size_kb = out.stat().st_size / 1024
    duration = get_audio_duration(str(out))
    logger.info(
        "Audio extracted: '%s' (%.1f KB, %.1f min)",
        out.name, size_kb, duration / 60,
    )
    return str(out)


def split_audio_into_chunks(
    wav_path: str,
    chunk_duration: int = CHUNK_DURATION_SECONDS,
) -> list[str]:
    """Split a WAV file into sequential fixed-duration chunks using FFmpeg.

    Each chunk is a new temp WAV file.  The original file is NOT deleted here —
    the caller owns that lifecycle.  Returns a list of chunk paths in order.

    If the file is shorter than *chunk_duration*, returns ``[wav_path]`` (no split).
    """
    total_duration = get_audio_duration(wav_path)
    if total_duration <= 0:
        logger.warning("Could not determine duration of '%s' — will not split.", wav_path)
        return [wav_path]

    if total_duration <= chunk_duration:
        return [wav_path]

    n_chunks = int(total_duration / chunk_duration) + (1 if total_duration % chunk_duration else 0)
    logger.info(
        "Long audio detected (%.1f min) — splitting into %d × %d-min chunks.",
        total_duration / 60, n_chunks, chunk_duration // 60,
    )

    stem = Path(wav_path).stem
    chunks: list[str] = []

    for i in range(n_chunks):
        start = i * chunk_duration
        # Last chunk: let ffmpeg figure out the remaining length naturally
        out = TEMP_AUDIO_DIR / f"{stem}_chunk{i:03d}_{uuid.uuid4().hex[:6]}.wav"
        try:
            (
                ffmpeg
                .input(wav_path, ss=start, t=chunk_duration)
                .output(
                    str(out),
                    acodec="pcm_s16le",
                    ac=1,
                    ar=16000,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            # Clean up any chunks already created and propagate
            for c in chunks:
                safe_unlink(c)
            raise AudioExtractionError(
                f"Failed to split audio at chunk {i}: {_decode_ffmpeg_error(e)}"
            ) from e

        if not out.is_file() or out.stat().st_size == 0:
            for c in chunks:
                safe_unlink(c)
            raise AudioExtractionError(f"Chunk {i} was empty after split.")

        chunks.append(str(out))
        logger.debug(
            "  Chunk %d/%d: start=%.0f s, file=%s (%.1f KB)",
            i + 1, n_chunks, start, out.name, out.stat().st_size / 1024,
        )

    return chunks
