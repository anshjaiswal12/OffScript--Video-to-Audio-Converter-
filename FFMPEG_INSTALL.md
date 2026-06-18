# FFmpeg Installation Guide

`ffmpeg-python` is only a wrapper. You must install the **native FFmpeg binary** on your system before audio extraction will work.

Verify after install:

```bash
ffmpeg -version
```

---

## Linux (your OS: Arch)

```bash
sudo pacman -S ffmpeg
```

Other distros:

| Distro | Command |
|--------|---------|
| Debian / Ubuntu | `sudo apt update && sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| openSUSE | `sudo zypper install ffmpeg` |

---

## macOS

```bash
brew install ffmpeg
```

Requires [Homebrew](https://brew.sh/). Verify with `ffmpeg -version`.

---

## Windows

**Option A — winget (recommended):**

```powershell
winget install Gyan.FFmpeg
```

**Option B — manual:**

1. Download a build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) (e.g. gyan.dev full build).
2. Extract the archive (e.g. to `C:\ffmpeg`).
3. Add `C:\ffmpeg\bin` to your system **PATH**.
4. Open a new terminal and run `ffmpeg -version`.

---

## Troubleshooting

- **`ffmpeg: command not found`** — binary not on PATH; reinstall or fix PATH, then restart the terminal.
- **Permission denied (Linux)** — use `sudo` for package install, not for running `ffmpeg` on your own files.
