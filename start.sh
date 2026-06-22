#!/usr/bin/env bash
# OffScript — start.sh
# Usage: ./start.sh
#   Opens OffScript in your browser. Run from any terminal; no arguments needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://${HOST}:${PORT}"
APP_NAME="OffScript"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$APP_NAME" "$*"; }
ok()   { printf '\033[1;32m[%s]\033[0m %s\n' "$APP_NAME" "$*"; }
warn() { printf '\033[1;33m[%s]\033[0m WARNING: %s\n' "$APP_NAME" "$*"; }
err()  { printf '\033[1;31m[%s]\033[0m ERROR: %s\n' "$APP_NAME" "$*" >&2; }

# ── 1. Python ──────────────────────────────────────────────────────────────────
pick_python() {
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"; return
    fi
  done
  err "Python 3 is required but not found. Install it and re-run."
  exit 1
}

# ── 2. Virtual environment ────────────────────────────────────────────────────
ensure_venv() {
  local py
  py="$(pick_python)"

  # Validate existing venv by asking its Python binary for sys.prefix.
  # This is reliable across all Python versions and venv formats (including
  # Python 3.14's cygpath-style activate scripts on Linux).
  if [[ -x ".venv/bin/python" ]]; then
    local prefix
    prefix="$(".venv/bin/python" -c "import sys; print(sys.prefix)" 2>/dev/null || true)"
    if [[ "$prefix" == "${ROOT}/.venv" ]]; then
      # shellcheck source=/dev/null
      source ".venv/bin/activate"
      ok "Using existing virtual environment."
      return 0
    else
      warn "venv is stale (prefix='${prefix}') — re-creating..."
      rm -rf .venv
    fi
  fi

  log "Creating virtual environment..."
  "$py" -m venv .venv
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
  ok "Virtual environment created."
}

# ── 3. Dependencies (cached: only reinstall when requirements.txt changes) ────
REQS_HASH_FILE=".venv/.requirements_hash"

_hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    # Fallback: combine size + mtime (good enough for cache busting)
    stat -c "%s%Y" "$1" 2>/dev/null || wc -c < "$1"
  fi
}

ensure_deps() {
  local current_hash cached_hash
  current_hash="$(_hash_file requirements.txt 2>/dev/null || echo "nocache")"
  cached_hash="$(cat "$REQS_HASH_FILE" 2>/dev/null || echo "")"

  if [[ -n "$current_hash" && "$current_hash" == "$cached_hash" ]]; then
    ok "Dependencies up to date — skipping install."
    return 0
  fi

  log "Installing / updating dependencies (this runs once per requirements change)..."
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt
  printf '%s' "$current_hash" > "$REQS_HASH_FILE"
  ok "Dependencies installed."
}

# ── 4. Tailwind CSS (cached locally for offline use) ──────────────────────────
ensure_tailwind() {
  local target="static/tailwind-cdn.js"
  if [[ -f "$target" && -s "$target" ]]; then
    ok "Tailwind CSS: local cache found."
    return 0
  fi
  log "Tailwind CSS not cached — downloading once for offline use..."
  local ok_dl=0
  if command -v curl >/dev/null 2>&1; then
    curl -sfL --max-time 30 "https://cdn.tailwindcss.com" -o "$target" 2>/dev/null && ok_dl=1
  fi
  if [[ $ok_dl -eq 0 ]] && command -v wget >/dev/null 2>&1; then
    wget -qO "$target" --timeout=30 "https://cdn.tailwindcss.com" 2>/dev/null && ok_dl=1
  fi
  if [[ $ok_dl -eq 1 ]]; then
    ok "Tailwind CSS saved — all future runs are fully offline."
  else
    warn "Could not download Tailwind CSS (no internet?). UI will use fallback styling."
    warn "Re-run once with internet, or place the CDN script at: ${target}"
  fi
}

# ── 5. Required directories ────────────────────────────────────────────────────
ensure_dirs() {
  for d in uploads temp_audio outputs models; do
    mkdir -p "$d"
    touch "$d/.gitkeep" 2>/dev/null || true
  done
}

# ── 6. FFmpeg check ────────────────────────────────────────────────────────────
check_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
  else
    warn "ffmpeg not found. Audio extraction will fail."
    warn "Install it: sudo pacman -S ffmpeg   (or see FFMPEG_INSTALL.md)"
  fi
}

# ── 7. Stop a previous instance using a PID file ──────────────────────────────
# A PID file is the only reliable way to find our previous process regardless
# of how it was started (nohup, direct, etc.) or what pkill pattern it matches.
PID_FILE="${ROOT}/.offscript.pid"

stop_existing_server() {
  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      log "Stopping previous OffScript instance (PID $old_pid)..."
      kill "$old_pid" 2>/dev/null || true
      # Wait up to 5 s for it to exit
      local i
      for i in $(seq 1 10); do
        kill -0 "$old_pid" 2>/dev/null || break
        sleep 0.5
      done
      # Force-kill if still alive
      kill -9 "$old_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi

  # Fallback: kill anything else that grabbed our port
  local port_pid
  port_pid="$(ss -tlnp 2>/dev/null | awk -v p=":${PORT} " '$0 ~ p {match($0,/pid=([0-9]+)/,a); print a[1]}')"
  if [[ -n "$port_pid" && "$port_pid" != "0" ]]; then
    log "Port ${PORT} still held by PID ${port_pid} — killing..."
    kill "$port_pid" 2>/dev/null || true
    sleep 0.5
  fi
}

# ── 8. Wait for server to accept connections ───────────────────────────────────
wait_for_server() {
  local i
  for i in $(seq 1 80); do
    # Try curl first, fall back to a pure-Python check
    if command -v curl >/dev/null 2>&1; then
      curl -sf --connect-timeout 1 "${URL}/api/status" >/dev/null 2>&1 && return 0
    else
      python - <<'PYEOF' 2>/dev/null && return 0
import urllib.request, sys
try:
    urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=1)
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
    fi
    sleep 0.25
  done
  return 1
}

# ── 9. Open browser ────────────────────────────────────────────────────────────
open_browser() {
  local target="$1"
  # Ordered preference: Brave → Chrome → Chromium → Firefox → xdg-open
  for candidate in brave brave-browser google-chrome-stable google-chrome chromium-browser chromium firefox; do
    if command -v "$candidate" >/dev/null 2>&1; then
      log "Opening $target in $candidate..."
      "$candidate" "$target" >/dev/null 2>&1 &
      return 0
    fi
  done
  if command -v xdg-open >/dev/null 2>&1; then
    log "Opening $target with xdg-open..."
    xdg-open "$target" >/dev/null 2>&1 &
    return 0
  fi
  warn "Could not auto-open a browser. Visit $target manually."
}

# ── Cleanup on exit / Ctrl+C ──────────────────────────────────────────────────
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Shutting down server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  log "Goodbye."
}
trap cleanup EXIT INT TERM

# ══════════════════════════════════════════════════════════════════════════════
printf '\n\033[1;34m━━━  OffScript — Offline Video Transcription  ━━━\033[0m\n\n'

ensure_venv
ensure_deps
ensure_dirs
check_ffmpeg
ensure_tailwind
stop_existing_server

log "Starting server at $URL ..."
# Note: --reload is intentionally omitted — it double-loads the Whisper model.
python -m uvicorn main:app --host "$HOST" --port "$PORT" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"   # record PID so next run can stop us cleanly

if wait_for_server; then
  open_browser "$URL"
  printf '\n'
  ok "OffScript is running at \033[4m${URL}\033[0m"
  ok "Press Ctrl+C to stop."
  printf '\n'
else
  err "Server did not start within 20 s. Check the output above for errors."
  exit 1
fi

wait "$SERVER_PID"
