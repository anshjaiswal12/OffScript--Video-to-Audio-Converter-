import gc
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).parent
MODEL_CACHE_DIR = BASE_DIR / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MODEL_IDLE_SECONDS = int(os.getenv("WHISPER_IDLE_SECONDS", "300"))

_model: WhisperModel | None = None
_last_used: float = 0.0


class TranscriptionError(Exception):
    """Raised when local speech-to-text fails."""


def touch_model() -> None:
    global _last_used
    _last_used = time.monotonic()


def unload_model() -> bool:
    global _model
    if _model is None:
        return False
    del _model
    _model = None
    gc.collect()
    return True


def maybe_unload_model(idle_seconds: float | None = None) -> bool:
    if _model is None:
        return False
    threshold = idle_seconds if idle_seconds is not None else MODEL_IDLE_SECONDS
    if time.monotonic() - _last_used >= threshold:
        return unload_model()
    return False


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            download_root=str(MODEL_CACHE_DIR),
        )
    touch_model()
    return _model


def _progress_payload(
    file_name: str,
    segment_end: float,
    duration: float,
    live_text: str,
    file_index: int = 0,
    file_count: int = 1,
) -> dict:
    if duration > 0:
        file_pct = min(100, int((segment_end / duration) * 100))
    else:
        file_pct = 0
    if file_count > 1:
        overall = int(((file_index + (file_pct / 100)) / file_count) * 100)
    else:
        overall = file_pct
    return {
        "file_name": file_name,
        "percent": min(100, overall),
        "live_text": live_text,
    }


def _iter_segments(
    wav_path: str,
    file_name: str,
    on_event: Callable[[dict], None] | None = None,
    file_index: int = 0,
    file_count: int = 1,
) -> tuple[list[str], float]:
    model = _load_model()
    segments, info = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    duration = float(info.duration or 0)
    paragraphs: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            paragraphs.append(text)
        if on_event:
            on_event(
                _progress_payload(
                    file_name=file_name,
                    segment_end=float(segment.end),
                    duration=duration,
                    live_text=text,
                    file_index=file_index,
                    file_count=file_count,
                )
            )

    touch_model()
    return paragraphs, duration


def transcribe_audio(
    wav_path: str,
    output_txt_path: str,
    file_name: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    file_index: int = 0,
    file_count: int = 1,
) -> str:
    wav = Path(wav_path).expanduser().resolve()
    if not wav.is_file():
        raise TranscriptionError(f"Audio file not found: {wav_path}")

    out = Path(output_txt_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    label = file_name or wav.name

    try:
        paragraphs, duration = _iter_segments(
            str(wav),
            label,
            on_event=on_event,
            file_index=file_index,
            file_count=file_count,
        )
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    if not paragraphs:
        raise TranscriptionError("No speech detected in the audio.")

    if on_event:
        on_event(
            _progress_payload(
                file_name=label,
                segment_end=duration or 1.0,
                duration=duration or 1.0,
                live_text=paragraphs[-1],
                file_index=file_index,
                file_count=file_count,
            )
        )

    out.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    return str(out)


def progress_event_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
