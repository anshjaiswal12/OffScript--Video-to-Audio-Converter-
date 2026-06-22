import gc
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel
from indic_transliteration import sanscript

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
MODEL_CACHE_DIR = BASE_DIR / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MODEL_IDLE_SECONDS = int(os.getenv("WHISPER_IDLE_SECONDS", "300"))

# Files longer than this trigger chunked transcription (imported from processing)
from processing import (
    CHUNK_THRESHOLD_SECONDS,
    CHUNK_DURATION_SECONDS,
    get_audio_duration,
    safe_unlink,
    split_audio_into_chunks,
)

_model: WhisperModel | None = None
_last_used: float = 0.0
_model_lock = threading.RLock()
_active_transcriptions: int = 0


class TranscriptionError(Exception):
    """Raised when local speech-to-text fails."""


# Devanagari Unicode block
DEV_RE = re.compile(r"[\u0900-\u097F]")
_DEV_CHUNK_RE = re.compile(r"[\u0900-\u097F]+")

# Whisper supported language codes (ISO 639-1 subset)
VALID_WHISPER_LANGUAGES: frozenset[str] = frozenset({
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
    "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
    "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
    "yi", "yo", "zh", "yue",
})

ENGLISH_MAPPING = {
    "sarvar": "server",
    "kanekt": "connect",
    "detaabes": "database",
    "vidiyo": "video",
    "koding": "coding",
    "python": "python",
    "api": "api",
    "yuai": "ui",
    "yuyeks": "ux",
    "intarnet": "internet",
    "kompyutar": "computer",
    "softaveyar": "software",
    "kod": "code",
    "yutyub": "youtube",
    "chhanal": "channel",
    "deta": "data",
    "yujar": "user",
    "vebsait": "website",
    "link": "link",
    "projek": "project",
    "projekt": "project",
    "app": "app",
    "ep": "app",
    "apligation": "application",
    "aplike-shan": "application",
    "aplikeshn": "application",
    "aplikeshan": "application",
    "faail": "file",
    "fail": "file",
    "imel": "email",
    "masej": "message",
    "kriyat": "create",
    "daunlod": "download",
    "aplod": "upload",
    "shiyar": "share",
    "kament": "comment",
    "laik": "like",
    "sabskraib": "subscribe",
    "batan": "button",
    "klik": "click",
    "skrin": "screen",
    "mobail": "mobile",
    "fon": "phone",
    "leptop": "laptop",
}


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

def validate_language(language: str | None) -> str | None:
    """Return the language code if valid for Whisper, else raise TranscriptionError."""
    if language is None:
        return None
    code = language.strip().lower()
    if not code:
        return None
    if code not in VALID_WHISPER_LANGUAGES:
        raise TranscriptionError(
            f"Unsupported language code: '{language}'. "
            f"Use a valid ISO 639-1 code (e.g. 'en', 'hi') or leave blank for auto-detect."
        )
    return code


# ---------------------------------------------------------------------------
# Hinglish romanisation
# ---------------------------------------------------------------------------

def _transliterate_devanagari_only(text: str) -> str:
    """Transliterate only Devanagari runs via ITRANS; Latin/numeric portions stay intact."""
    parts: list[str] = []
    last = 0
    for m in _DEV_CHUNK_RE.finditer(text):
        parts.append(text[last : m.start()])
        parts.append(sanscript.transliterate(m.group(), sanscript.DEVANAGARI, sanscript.ITRANS))
        last = m.end()
    parts.append(text[last:])
    return "".join(parts)


def to_natural_roman(text: str) -> str:
    """Convert Devanagari-containing text to a natural Hinglish Roman spelling.

    Wraps the conversion so any unexpected exception returns the original text
    rather than crashing the whole transcription — important for very long files
    where one bad segment should not abort everything.
    """
    if not text:
        return text
    try:
        return _to_natural_roman_impl(text)
    except Exception as exc:
        logger.warning("Romanisation failed for segment (returning original): %s", exc)
        return text


def _to_natural_roman_impl(text: str) -> str:
    itrans_text = _transliterate_devanagari_only(text)

    # Clean ITRANS nasalisation markers
    itrans_text = itrans_text.replace(".N", "n")
    itrans_text = itrans_text.replace(".n", "n")
    itrans_text = itrans_text.replace("~N", "n")
    itrans_text = itrans_text.replace("~n", "n")
    itrans_text = itrans_text.replace("M", "n")

    words = []
    tokens = re.split(r"(\s+|[.,!?;:|।])", itrans_text)

    for token in tokens:
        if not token or re.match(r"^\s+$", token) or re.match(r"^[.,!?;:|।]$", token):
            words.append("." if token in ("।", "|") else token)
            continue

        word = token

        # Schwa deletion: inherent 'a' at word-end is silent in Hindi
        if word.endswith("a") and len(word) > 2:
            word = word[:-1]

        word = re.sub(r"([b-df-hj-np-tv-z])a([tT][eA])", r"\1\2", word)

        if word.startswith("A"):
            word = "aa" + word[1:]
        else:
            word = word.replace("A", "a")

        word = word.replace("I", "i")
        word = word.replace("U", "u")
        word = word.replace("T", "t")
        word = word.replace("D", "d")
        word = word.replace("N", "n")
        word = word.lower()

        if word.startswith("v"):
            if word == "video" or word.startswith("vide"):
                word = "video"
            elif word.startswith("vaala") or word.startswith("vala"):
                word = "wala"
            elif word == "vaha":
                word = "woh"

        if word in ("ham", "hama"):
            word = "hum"
        elif word in ("men", "me"):
            word = "mein"
        elif word == "kara":
            word = "kar"
        elif word == "hai":
            word = "hai"
        elif word == "hain":
            word = "hain"
        elif word in ("hu", "hun", "hoon"):
            word = "hoon"
        elif word == "aur":
            word = "aur"
        elif word == "ya":
            word = "ya"
        elif word == "vaha":
            word = "woh"
        elif word == "yaha":
            word = "yeh"

        word = ENGLISH_MAPPING.get(word, word)
        words.append(word)

    return "".join(words)


# ---------------------------------------------------------------------------
# Model lifecycle (thread-safe)
# ---------------------------------------------------------------------------

def touch_model() -> None:
    global _last_used
    _last_used = time.monotonic()


def unload_model() -> bool:
    """Forcibly unload the model. Caller must hold _model_lock."""
    global _model
    if _model is None:
        return False
    del _model
    _model = None
    gc.collect()
    logger.info("Whisper model unloaded.")
    return True


def maybe_unload_model(idle_seconds: float | None = None) -> bool:
    """Unload the model if it has been idle long enough and no transcription is active."""
    global _active_transcriptions
    with _model_lock:
        if _model is None:
            return False
        if _active_transcriptions > 0:
            logger.debug("Skipping model unload — %d active transcription(s).", _active_transcriptions)
            return False
        threshold = idle_seconds if idle_seconds is not None else MODEL_IDLE_SECONDS
        if time.monotonic() - _last_used >= threshold:
            return unload_model()
        return False


def _load_model() -> WhisperModel:
    """Load (or reuse) the Whisper model. Thread-safe."""
    global _model
    with _model_lock:
        if _model is None:
            logger.info("Loading Whisper model '%s' (CPU, int8) …", WHISPER_MODEL)
            try:
                _model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODEL_CACHE_DIR),
                )
                logger.info("Whisper model loaded successfully.")
            except Exception as exc:
                raise TranscriptionError(
                    f"Failed to load Whisper model '{WHISPER_MODEL}': {exc}. "
                    "Make sure the model is downloaded to the models/ directory."
                ) from exc
        touch_model()
        return _model


def _acquire_model() -> WhisperModel:
    global _active_transcriptions
    with _model_lock:
        model = _load_model()
        _active_transcriptions += 1
        return model


def _release_model() -> None:
    global _active_transcriptions
    with _model_lock:
        _active_transcriptions = max(0, _active_transcriptions - 1)
        touch_model()


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress_payload(
    file_name: str,
    elapsed_seconds: float,
    total_seconds: float,
    live_text: str,
    live_text_hinglish: str | None = None,
    file_index: int = 0,
    file_count: int = 1,
) -> dict:
    """Build a progress payload.

    *elapsed_seconds* and *total_seconds* are relative to the FULL file duration
    (not just the current chunk), so progress stays monotonically increasing.
    """
    if total_seconds > 0:
        file_pct = min(100, int((elapsed_seconds / total_seconds) * 100))
    else:
        file_pct = 0
    if file_count > 1:
        overall = int(((file_index + file_pct / 100) / file_count) * 100)
    else:
        overall = file_pct
    return {
        "file_name": file_name,
        "percent": min(100, overall),
        "live_text": live_text,
        "live_text_hinglish": live_text_hinglish,
    }


# ---------------------------------------------------------------------------
# Single-chunk transcription
# ---------------------------------------------------------------------------

def _transcribe_chunk(
    wav_path: str,
    file_name: str,
    language: str | None,
    on_event: Callable[[dict], None] | None,
    # Offset of this chunk within the full file (for progress calculation)
    chunk_offset_seconds: float,
    total_file_seconds: float,
    file_index: int,
    file_count: int,
) -> tuple[list[str], list[str]]:
    """Transcribe one WAV chunk and return (paragraphs, originals).

    Progress events are emitted with timestamps relative to the full file so the
    progress bar advances smoothly across all chunks.
    """
    model = _acquire_model()
    try:
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=5,
            vad_filter=True,
            language=language,
        )
        chunk_duration = float(info.duration or 0)
        paragraphs: list[str] = []
        original_paragraphs: list[str] = []

        logger.debug(
            "Transcribing chunk (offset=%.0f s, dur=%.0f s, lang=%s).",
            chunk_offset_seconds, chunk_duration, language or "auto",
        )

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            original_paragraphs.append(text)
            hinglish_text: str | None = None
            if DEV_RE.search(text):
                hinglish_text = to_natural_roman(text)
                text = hinglish_text
            paragraphs.append(text)

            if on_event:
                # segment.end is relative to this chunk; add offset for full-file progress
                full_elapsed = chunk_offset_seconds + float(segment.end)
                on_event(
                    _progress_payload(
                        file_name=file_name,
                        elapsed_seconds=full_elapsed,
                        total_seconds=total_file_seconds,
                        live_text=original_paragraphs[-1],
                        live_text_hinglish=hinglish_text,
                        file_index=file_index,
                        file_count=file_count,
                    )
                )
    finally:
        _release_model()

    return paragraphs, original_paragraphs


# ---------------------------------------------------------------------------
# Public transcription entry point
# ---------------------------------------------------------------------------

def transcribe_audio(
    wav_path: str,
    output_txt_path: str,
    file_name: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    file_index: int = 0,
    file_count: int = 1,
    language: str | None = None,
) -> str:
    """Transcribe *wav_path* to a text file at *output_txt_path*.

    For files longer than CHUNK_THRESHOLD_SECONDS the audio is automatically
    split into CHUNK_DURATION_SECONDS segments before transcription.  Each chunk
    is processed independently and deleted immediately after use, keeping peak
    memory consumption constant regardless of video length.

    Returns the path to the written transcript file.
    """
    wav = Path(wav_path).expanduser().resolve()
    if not wav.is_file():
        raise TranscriptionError(f"Audio file not found: {wav_path}")

    out = Path(output_txt_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    label = file_name or wav.name

    language = validate_language(language)

    total_duration = get_audio_duration(str(wav))
    if total_duration <= 0:
        # Fallback: let Whisper figure it out from the first chunk
        total_duration = 1.0

    # ── Decide: single pass or chunked ────────────────────────────────────────
    if total_duration <= CHUNK_THRESHOLD_SECONDS:
        chunks = [str(wav)]
        owns_chunks = False      # do NOT delete the original
        logger.info("Transcribing '%s' in a single pass (%.1f min).", label, total_duration / 60)
    else:
        logger.info(
            "File '%s' is %.1f min — splitting into %d-min chunks for memory safety.",
            label, total_duration / 60, CHUNK_DURATION_SECONDS // 60,
        )
        try:
            chunks = split_audio_into_chunks(str(wav), CHUNK_DURATION_SECONDS)
        except Exception as exc:
            raise TranscriptionError(f"Failed to split audio for chunked processing: {exc}") from exc
        owns_chunks = True       # we created these temp files; delete after use

    # ── Process chunks ─────────────────────────────────────────────────────────
    all_paragraphs: list[str] = []
    chunk_offset = 0.0

    for chunk_idx, chunk_path in enumerate(chunks):
        chunk_label = (
            f"{label} [part {chunk_idx + 1}/{len(chunks)}]"
            if len(chunks) > 1
            else label
        )
        logger.info("Processing %s …", chunk_label)

        try:
            paras, _originals = _transcribe_chunk(
                chunk_path,
                label,                # always use original name for UI matching
                language=language,
                on_event=on_event,
                chunk_offset_seconds=chunk_offset,
                total_file_seconds=total_duration,
                file_index=file_index,
                file_count=file_count,
            )
        except Exception as exc:
            # Clean up remaining chunks before re-raising
            if owns_chunks:
                for remaining in chunks[chunk_idx:]:
                    safe_unlink(remaining)
            raise TranscriptionError(
                f"Transcription failed at chunk {chunk_idx + 1}/{len(chunks)} "
                f"(~{int(chunk_offset // 60)} min into file): {exc}"
            ) from exc
        finally:
            # Delete this chunk immediately to free disk and memory pressure
            if owns_chunks:
                safe_unlink(chunk_path)
            # Force GC between chunks to return memory to OS
            gc.collect()

        all_paragraphs.extend(paras)
        chunk_offset += CHUNK_DURATION_SECONDS

    if not all_paragraphs:
        raise TranscriptionError("No speech detected in the audio.")

    out.write_text("\n\n".join(all_paragraphs) + "\n", encoding="utf-8")
    logger.info(
        "Transcript written to '%s' (%d paragraph(s), %.1f min).",
        out, len(all_paragraphs), total_duration / 60,
    )
    return str(out)


def progress_event_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
