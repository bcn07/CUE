#!/usr/bin/env python3
"""CUE camera laptop script: webcam -> JPEG frames -> ws://<director>:8000/ingest

    python stream_camera.py --server ws://<director-ip>:8000 --cam B

Runs on macOS / Windows / Linux with Python 3.8+ and two packages:
    pip install opencv-python websockets
The camera id you pass is created on the director on first connect; set its role
and label on the director's /setup page (a --label here is only used the first time).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

try:
    import cv2
    import numpy as np
    from websockets.sync.client import connect
except ImportError as e:  # pragma: no cover
    print("missing dependency: %s\n  pip install opencv-python websockets" % e, file=sys.stderr)
    sys.exit(2)


def say(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def backend():
    if os.name == "nt":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def open_capture(index, width, height, fps):
    cap = cv2.VideoCapture(index, backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, max(fps, 15))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def fit(frame, W, H):
    """Letterbox into WxH, never stretch (a 4:3 webcam must not squash faces)."""
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    s = min(W / float(w), H / float(h))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if (nw, nh) == (W, H):
        return resized
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def list_devices(max_index=6):
    say("probing camera indices 0..%d (macOS may ask for Camera permission)" % (max_index - 1))
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend())
        if not cap.isOpened():
            cap.release()
            continue
        ok = False
        for _ in range(10):  # the first grab after open is often empty on AVFoundation
            ok, _f = cap.read()
            if ok:
                break
            time.sleep(0.1)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        say("  [%d] %s %dx%d" % (i, "OK " if ok else "opens but no frame", int(w), int(h)))
        cap.release()


class SendWatchdog(threading.Thread):
    """If a send blocks for too long (link stalled, peer asleep), close the socket so the
    blocked sendall raises and the outer loop reconnects instead of hanging forever."""

    def __init__(self, stall_s=5.0):
        super().__init__(daemon=True)
        self.stall_s = stall_s
        self.ws = None
        self.send_start = None
        self.lock = threading.Lock()
        self.fired = 0

    def begin(self, ws):
        with self.lock:
            self.ws, self.send_start = ws, time.monotonic()

    def end(self):
        with self.lock:
            self.send_start = None

    def run(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                stalled = self.send_start is not None and time.monotonic() - self.send_start > self.stall_s
                ws = self.ws
            if stalled and ws is not None:
                self.fired += 1
                say("link stalled for %.0fs, forcing reconnect" % self.stall_s, file=sys.stderr)
                try:
                    ws.socket.close()
                except Exception:
                    pass
                with self.lock:
                    self.send_start = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("CUE_SERVER", ""), help="ws://<director-ip>:8000")
    ap.add_argument("--cam", default=os.environ.get("CUE_CAM", "B"), help="camera id: A, B, C ...")
    ap.add_argument("--code", default=os.environ.get("CUE_CODE", ""), help="join code shown on the director's /setup page (required)")
    ap.add_argument("--label", default="", help="display label used when the camera is first created, e.g. 'Stage left'")
    ap.add_argument("--role", default="", choices=["", "wide", "host", "guest"], help="initial role (setup page can change it)")
    ap.add_argument("--device", type=int, default=int(os.environ.get("CUE_DEVICE", "0")), help="webcam index (see --list-devices)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality 40-90")
    ap.add_argument("--preview", action="store_true", help="show a local preview window")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()

    if a.list_devices:
        list_devices()
        return 0
    if not a.server:
        ap.error("--server ws://<director-ip>:8000 is required (or set CUE_SERVER)")
    if not (0 < a.fps <= 60):
        ap.error("--fps must be between 1 and 60")
    if not (1 <= a.quality <= 100):
        ap.error("--quality must be between 1 and 100")
    if a.width < 160 or a.height < 90:
        ap.error("--width/--height too small")
    server = a.server.rstrip("/")
    if not server.startswith(("ws://", "wss://")):
        server = "ws://" + server
    if ":" not in server.split("//", 1)[1]:
        server += ":8000"
    cam_id = a.cam.strip().upper()
    from urllib.parse import quote
    if not a.code:
        ap.error("--code <join code> is required (it is printed on the director's /setup page)")
    url = "%s/ingest?cam=%s&code=%s" % (server, quote(cam_id), quote(a.code))
    if a.label:
        url += "&label=" + quote(a.label)
    if a.role:
        url += "&role=" + a.role

    cap = open_capture(a.device, a.width, a.height, int(a.fps))
    if cap is None:
        say("cannot open camera index %d. Try --list-devices. On macOS grant Camera permission to Terminal." % a.device, file=sys.stderr)
        return 1
    say("camera %d open: %dx%d -> sending %dx%d @ %g fps q%d (letterboxed, never stretched)" % (
        a.device, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), a.width, a.height, a.fps, a.quality))
    say("streaming as camera %s to %s" % (cam_id, url))

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(a.quality)]
    frame_interval = 1.0 / a.fps
    backoff = 1.0
    watchdog = SendWatchdog()
    watchdog.start()
    try:
        while True:
            try:
                with connect(url, max_size=None, open_timeout=5, close_timeout=1) as ws:
                    hello = {"type": "hello", "cam": cam_id, "width": a.width, "height": a.height, "fps": a.fps}
                    if a.label:
                        hello["label"] = a.label
                    ws.send(json.dumps(hello))
                    try:
                        first = ws.recv(timeout=3)
                        say("director:", first)
                        if '"error"' in str(first) and "join code" in str(first):
                            say("wrong join code: check the director's /setup page", file=sys.stderr)
                            return 2
                    except Exception:
                        pass
                    backoff = 1.0
                    sent = 0
                    sent_bytes = 0
                    t_stat = time.time()
                    next_t = time.perf_counter()
                    fails = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            fails += 1
                            if fails > 30:
                                say("camera stopped delivering frames; reopening", file=sys.stderr)
                                cap.release()
                                time.sleep(0.5)
                                cap = open_capture(a.device, a.width, a.height, int(a.fps)) or cap
                                fails = 0
                            time.sleep(0.02)
                            continue
                        fails = 0
                        frame = fit(frame, a.width, a.height)
                        ok, buf = cv2.imencode(".jpg", frame, encode_params)
                        if not ok:
                            continue
                        data = buf.tobytes()
                        watchdog.begin(ws)
                        ws.send(data)
                        watchdog.end()
                        sent += 1
                        sent_bytes += len(data)
                        if a.preview:
                            cv2.imshow("CUE camera %s" % cam_id, frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                return 0
                        now = time.time()
                        if now - t_stat >= 2.0:
                            say("  %5.1f fps  %6.0f kb/s  %5.1f KB/frame" % (sent / (now - t_stat), sent_bytes * 8 / (now - t_stat) / 1000, len(data) / 1024.0))
                            sent, sent_bytes, t_stat = 0, 0, now
                        next_t += frame_interval
                        delay = next_t - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            next_t = time.perf_counter()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                watchdog.end()
                say("connection problem: %s: %s -> retry in %.0fs" % (type(e).__name__, e, backoff), file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
    except KeyboardInterrupt:
        say("\nbye")
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    sys.exit(main())
