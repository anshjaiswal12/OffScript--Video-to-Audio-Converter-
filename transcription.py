import gc
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel
from indic_transliteration import sanscript

BASE_DIR = Path(__file__).parent
MODEL_CACHE_DIR = BASE_DIR / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MODEL_IDLE_SECONDS = int(os.getenv("WHISPER_IDLE_SECONDS", "300"))

_model: WhisperModel | None = None
_last_used: float = 0.0


class TranscriptionError(Exception):
    """Raised when local speech-to-text fails."""


DEV_RE = re.compile(r"[\u0900-\u097F]")

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
    "laptoP": "laptop",
    "leptop": "laptop",
}


def to_natural_roman(text: str) -> str:
    if not text:
        return text

    # 1. Transliterate Devanagari to ITRANS
    itrans_text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)

    # Pre-clean ITRANS nasalization before tokenizing so dots in .N are not split as punctuation
    itrans_text = itrans_text.replace(".N", "n")
    itrans_text = itrans_text.replace(".n", "n")
    itrans_text = itrans_text.replace("~N", "n")
    itrans_text = itrans_text.replace("~n", "n")
    itrans_text = itrans_text.replace("M", "n")

    # 2. Process word by word
    words = []
    tokens = re.split(r"(\s+|[.,!?;:|।])", itrans_text)

    for token in tokens:
        if not token or re.match(r"^\s+$", token) or re.match(r"^[.,!?;:|।]$", token):
            if token in ("।", "|"):
                words.append(".")
            else:
                words.append(token)
            continue

        word = token

        # Schwa deletion: inherent 'a' at the end of a word is silent in Hindi
        if word.endswith("a") and len(word) > 2:
            word = word[:-1]

        # Middle schwa deletion: e.g. "karate" -> "karte"
        word = re.sub(r"([b-df-hj-np-tv-z])a([tT][eA])", r"\1\2", word)

        # Replace standard ITRANS characters with natural Roman equivalents
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

        # Map common pronoun / filler words
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
    live_text_hinglish: str | None = None,
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
        "live_text_hinglish": live_text_hinglish,
    }


def _iter_segments(
    wav_path: str,
    file_name: str,
    on_event: Callable[[dict], None] | None = None,
    file_index: int = 0,
    file_count: int = 1,
    language: str | None = None,
) -> tuple[list[str], list[str], float]:
    model = _load_model()
    segments, info = model.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,
        language=language,
    )
    duration = float(info.duration or 0)
    paragraphs: list[str] = []
    original_paragraphs: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            original_paragraphs.append(text)
            hinglish_text = None
            if DEV_RE.search(text):
                hinglish_text = to_natural_roman(text)
                text = hinglish_text
            paragraphs.append(text)
            if on_event:
                on_event(
                    _progress_payload(
                        file_name=file_name,
                        segment_end=float(segment.end),
                        duration=duration,
                        live_text=original_paragraphs[-1],
                        live_text_hinglish=hinglish_text,
                        file_index=file_index,
                        file_count=file_count,
                    )
                )

    touch_model()
    return paragraphs, original_paragraphs, duration


def transcribe_audio(
    wav_path: str,
    output_txt_path: str,
    file_name: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    file_index: int = 0,
    file_count: int = 1,
    language: str | None = None,
) -> str:
    wav = Path(wav_path).expanduser().resolve()
    if not wav.is_file():
        raise TranscriptionError(f"Audio file not found: {wav_path}")

    out = Path(output_txt_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    label = file_name or wav.name

    try:
        paragraphs, original_paragraphs, duration = _iter_segments(
            str(wav),
            label,
            on_event=on_event,
            file_index=file_index,
            file_count=file_count,
            language=language,
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
                live_text=original_paragraphs[-1] if original_paragraphs else paragraphs[-1] if paragraphs else "",
                live_text_hinglish=paragraphs[-1] if (paragraphs and original_paragraphs and paragraphs[-1] != original_paragraphs[-1]) else None,
                file_index=file_index,
                file_count=file_count,
            )
        )

    out.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    return str(out)


def progress_event_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
