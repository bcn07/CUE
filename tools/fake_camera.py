#!/usr/bin/env python3
"""Test camera without a laptop: streams an image, a video file, or a synthetic
pattern to the director as camera X. Same protocol as camera/stream_camera.py.

  python tools/fake_camera.py --cam B --image tests/fixtures/obama.jpg
  python tools/fake_camera.py --cam A --pattern
  python tools/fake_camera.py --cam C --video some.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np
from websockets.sync.client import connect


def _fetch_code(ws_base: str) -> str:
    import urllib.request
    http = ws_base.rstrip("/").replace("wss://", "https://").replace("ws://", "http://")
    try:
        return json.load(urllib.request.urlopen(f"{http}/api/join-code", timeout=5))["code"]
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://127.0.0.1:8000")
    ap.add_argument("--cam", default="B")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--pattern", action="store_true")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = forever)")
    ap.add_argument("--jitter", action="store_true", help="shift the image a little each frame")
    ap.add_argument("--code", default="", help="join code (default: fetched from the director's /api/join-code, local only)")
    a = ap.parse_args()

    src = None
    base = None
    if a.image:
        base = cv2.imread(a.image)
        if base is None:
            print("cannot read", a.image, file=sys.stderr)
            return 1
        h, w = base.shape[:2]
        s = min(a.width / w, a.height / h)
        img = cv2.resize(base, (int(w * s), int(h * s)))
        canvas = np.zeros((a.height, a.width, 3), np.uint8)
        y0, x0 = (a.height - img.shape[0]) // 2, (a.width - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        base = canvas
    elif a.video:
        src = cv2.VideoCapture(a.video)
        if not src.isOpened():
            print("cannot open", a.video, file=sys.stderr)
            return 1
    else:
        a.pattern = True

    code = a.code or _fetch_code(a.server)
    url = f"{a.server.rstrip('/')}/ingest?cam={a.cam.upper()}&label=Fake%20{a.cam.upper()}&code={code}"
    t_end = time.time() + a.seconds if a.seconds else None
    n = 0
    with connect(url, max_size=None) as ws:
        ws.send(json.dumps({"type": "hello", "cam": a.cam.upper(), "label": f"Fake {a.cam.upper()}"}))
        while t_end is None or time.time() < t_end:
            if a.pattern:
                frame = np.full((a.height, a.width, 3), 70, np.uint8)   # a lit room, not a black frame
                x = int((time.time() * 120) % a.width)
                cv2.rectangle(frame, (x, 40), (min(a.width, x + 80), 200), (0, 200, 255), -1)
                cv2.putText(frame, f"FAKE {a.cam.upper()} {time.strftime('%H:%M:%S')}", (20, a.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif src is not None:
                ok, frame = src.read()
                if not ok:
                    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (a.width, a.height))
            else:
                frame = base
                if a.jitter:
                    dx, dy = int(6 * np.sin(n / 7)), int(4 * np.cos(n / 5))
                    frame = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            ws.send(buf.tobytes())
            n += 1
            time.sleep(1.0 / a.fps)
    print(f"sent {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
