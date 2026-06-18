#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://${HOST}:${PORT}"
APP_NAME="OffScript"

log() { printf '[%s] %s\n' "$APP_NAME" "$*"; }

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    log "ERROR: Python 3 is required but not found."
    exit 1
  fi
}

ensure_env() {
  local py
  py="$(pick_python)"

  if [[ ! -d ".venv" ]]; then
    log "Creating virtual environment..."
    "$py" -m venv .venv
  fi

  # shellcheck source=/dev/null
  source ".venv/bin/activate"

  log "Checking dependencies..."
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt
}

stop_existing_server() {
  local pattern="${ROOT}/.venv/bin/python"
  if pgrep -f "${pattern}.*uvicorn main:app" >/dev/null 2>&1; then
    log "Stopping existing OffScript server..."
    pkill -f "${pattern}.*uvicorn main:app" 2>/dev/null || true
    sleep 1
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 0.5
  fi
}

wait_for_server() {
  local attempt
  for attempt in $(seq 1 60); do
    if curl -sf --connect-timeout 1 "${URL}/api/status" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

open_browser() {
  local target="$1"
  local browser=""

  for candidate in brave brave-browser google-chrome-stable google-chrome chromium firefox; do
    if command -v "$candidate" >/dev/null 2>&1; then
      browser="$candidate"
      break
    fi
  done

  if [[ -n "$browser" ]]; then
    log "Opening ${target} in ${browser}..."
    "$browser" "$target" >/dev/null 2>&1 &
    return 0
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    log "Opening ${target} with xdg-open..."
    xdg-open "$target" >/dev/null 2>&1 &
    return 0
  fi

  log "Could not auto-open a browser. Visit ${target} manually."
  return 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if ! command -v ffmpeg >/dev/null 2>&1; then
  log "WARNING: ffmpeg not found. Install it first (see FFMPEG_INSTALL.md)."
fi

ensure_env
stop_existing_server

log "Starting server at ${URL}"
uvicorn main:app --host "$HOST" --port "$PORT" --reload &
SERVER_PID=$!

if wait_for_server; then
  open_browser "$URL" || true
  log "Ready. Press Ctrl+C to stop."
else
  log "ERROR: Server failed to start on ${URL}"
  exit 1
fi

wait "$SERVER_PID"
