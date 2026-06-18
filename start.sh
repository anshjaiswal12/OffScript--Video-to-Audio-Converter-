#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://${HOST}:${PORT}"

if [[ ! -d ".venv" ]]; then
  echo "Virtual environment not found. Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck source=/dev/null
source ".venv/bin/activate"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg not found on PATH. Install it first (see FFMPEG_INSTALL.md)."
fi

if command -v brave >/dev/null 2>&1; then
  (sleep 1 && brave "$URL" >/dev/null 2>&1 &) || true
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 1 && xdg-open "$URL" >/dev/null 2>&1 &) || true
fi

echo "Starting Offline Video Transcriber at $URL"
echo "Press Ctrl+C to stop."
exec uvicorn main:app --host "$HOST" --port "$PORT" --reload
