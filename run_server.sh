#!/usr/bin/env bash
# Director MacBook: ./run_server.sh        (first run: creates .venv, installs deps, downloads face models)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  echo "[cue] creating .venv ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv -q 2>/dev/null || uv venv .venv -q
    uv pip install -q --python .venv/bin/python -r requirements-server.txt
  else
    PY="${PYTHON:-python3}"
    "$PY" -m venv .venv
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -q -r requirements-server.txt
  fi
fi
mkdir -p models data/people
dl() { # name url min_bytes  -- download to a temp file, keep it only when complete
  if [ ! -s "models/$1" ] || [ "$(wc -c < "models/$1" | tr -d ' ')" -lt "$3" ]; then
    echo "[cue] downloading $1 ..."
    rm -f "models/$1.part"
    if curl -sL --fail --max-time 600 -o "models/$1.part" "$2" && [ "$(wc -c < "models/$1.part" | tr -d ' ')" -ge "$3" ]; then
      mv "models/$1.part" "models/$1"
    else
      rm -f "models/$1.part"
      echo "[cue] download of $1 failed or was cut short (no internet?). Face identity will be OFF until this succeeds; rerun ./run_server.sh when online."
    fi
  fi
}
dl face_detection_yunet_2023mar.onnx "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" 200000
dl face_recognition_sface_2021dec.onnx "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" 38000000
[ -f .env ] || cp .env.example .env
# Local LLM (Ollama): make sure the server is up and the model is present, so the first clause is not a cold miss.
LLM_URL="$(sed -n 's/^CUE_LLM_BASE_URL=//p' .env | tr -d '[:space:]"' | head -1)"
LLM_MODEL="$(sed -n 's/^CUE_LLM_MODEL=//p' .env | tr -d '[:space:]"' | head -1)"
if [ -n "${CUE_LLM_BASE_URL:-$LLM_URL}" ] && echo "${CUE_LLM_BASE_URL:-$LLM_URL}" | grep -q 11434; then
  if ! curl -sf --max-time 2 "${CUE_LLM_BASE_URL:-$LLM_URL}/models" >/dev/null 2>&1; then
    if command -v ollama >/dev/null 2>&1; then
      echo "[cue] starting ollama serve ..."
      (nohup ollama serve >/dev/null 2>&1 &)
      for i in $(seq 1 20); do curl -sf --max-time 1 "${CUE_LLM_BASE_URL:-$LLM_URL}/models" >/dev/null 2>&1 && break; sleep 0.5; done
    else
      echo "[cue] CUE_LLM_BASE_URL points at Ollama but ollama is not installed: brew install ollama && ollama pull ${CUE_LLM_MODEL:-$LLM_MODEL}"
    fi
  fi
  if command -v ollama >/dev/null 2>&1 && [ -n "${CUE_LLM_MODEL:-$LLM_MODEL}" ] && ! ollama list 2>/dev/null | grep -q "^${CUE_LLM_MODEL:-$LLM_MODEL}"; then
    echo "[cue] pulling model ${CUE_LLM_MODEL:-$LLM_MODEL} ..."
    ollama pull "${CUE_LLM_MODEL:-$LLM_MODEL}" || echo "[cue] model pull failed; the interpreter will run rules-only until it exists"
  fi
fi
# LAN address for the camera line: the default route's interface first (not always en0/en1, e.g. USB Ethernet).
DEF_IF="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}' || true)"
IP="$(ipconfig getifaddr "${DEF_IF:-en0}" 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)"
# Port: shell env wins, then .env, then 8000. Exported so the app advertises the port it really binds.
if [ -z "${CUE_PORT:-}" ] && [ -f .env ]; then
  CUE_PORT="$(sed -n 's/^CUE_PORT=//p' .env | tr -d '[:space:]"' | head -1)"
fi
PORT="${CUE_PORT:-8000}"
export CUE_PORT="$PORT"
if OWNER=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1) && [ -n "$OWNER" ]; then
  echo "[cue] port $PORT is already in use by pid $OWNER ($(ps -o command= -p "$OWNER" | cut -c1-80))."
  echo "[cue] stop it (kill $OWNER) or run: CUE_PORT=8001 ./run_server.sh"
  exit 1
fi
echo
echo "  CUE director  : http://localhost:$PORT/        (this MacBook)"
echo "  Setup page    : http://localhost:$PORT/setup"
echo "  Camera laptops: ./camera/run_camera.sh --server ws://$IP:$PORT --cam B --code <join code from /setup>"
echo
exec .venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port "$PORT" --log-level info --ws-max-size 8388608 "$@"
