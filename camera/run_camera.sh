#!/usr/bin/env bash
# One command per camera laptop (macOS / Linux):
#   ./camera/run_camera.sh --server ws://<director-ip>:8000 --cam B --code <join-code>
# Creates a local venv next to this script on first run, installs 2 packages, streams.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
VPY="$HERE/.venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "[cue-cam] creating venv in $HERE/.venv ..."
  "$PY" -m venv "$HERE/.venv" || { echo "[cue-cam] venv creation failed. On Debian/Ubuntu: sudo apt install python3-venv"; exit 1; }
fi
# Gate on the packages, not on the interpreter: an interrupted pip run must retry next time.
if ! "$VPY" -c 'import cv2, websockets' >/dev/null 2>&1; then
  echo "[cue-cam] installing opencv-python + websockets ..."
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -r "$HERE/requirements-camera.txt" || { echo "[cue-cam] pip install failed (no internet?). Retry when online."; exit 1; }
fi
exec "$VPY" "$HERE/stream_camera.py" "$@"
