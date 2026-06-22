#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

REQUIRED_DIRS = ("static", "temp_audio", "outputs", "uploads")
REQUIRED_MODULES = ("fastapi", "uvicorn", "ffmpeg", "faster_whisper", "indic_transliteration")


def fail(msg: str) -> None:
    print(f"[OffScript] ERROR: {msg}")
    sys.exit(1)


def check_dirs() -> None:
    for name in REQUIRED_DIRS:
        path = ROOT / name
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OffScript] OK  ./{name}/")


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        fail(
            "ffmpeg not found on PATH.\n"
            "Install it first (see FFMPEG_INSTALL.md), then rerun this script."
        )
    print("[OffScript] OK  ffmpeg")


def check_modules() -> None:
    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        fail(f"Missing Python packages: {', '.join(missing)}\nRun: pip install -r requirements.txt")
    print(f"[OffScript] OK  modules ({', '.join(REQUIRED_MODULES)})")


def open_browser_delayed() -> None:
    time.sleep(1.5)
    print(f"[OffScript] Opening {URL}")
    webbrowser.open(URL)


def main() -> None:
    print("[OffScript] Pre-flight checks…")
    check_dirs()
    check_ffmpeg()
    check_modules()
    print("[OffScript] All checks passed. Starting server…")
    threading.Thread(target=open_browser_delayed, daemon=True).start()
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
