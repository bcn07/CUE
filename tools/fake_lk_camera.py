#!/usr/bin/env python3
"""Publish a synthetic camera into the director's LiveKit room, exactly like a phone would.

    .venv/bin/python tools/fake_lk_camera.py --server http://127.0.0.1:8000 --cam D --code 123456 --seconds 20
    .venv/bin/python tools/fake_lk_camera.py ... --image path/to/face.jpg   # a still photo instead of the test pattern

It asks the director for a publisher token over the same /api/livekit/token route the phone page
uses (join code required), joins the room, publishes a moving test pattern at 15 fps and prints
the control messages (ack, ptz, standby, replaced) the director sends back. Use it to prove the
LiveKit path end to end without a phone, and to rehearse standby/approve with two instances.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import cv2
import httpx
import numpy as np


def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit into w x h without stretching: a squashed face is not detected (a portrait photo in 16:9)."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    out = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def pattern(w: int, h: int, i: int, cam: str, img: np.ndarray | None) -> np.ndarray:
    if img is not None:
        frame = img.copy()
    else:
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:] = (40 + (i * 3) % 60, 60, 90)
        x = int((i * 7) % (w - 120))
        cv2.rectangle(frame, (x, h // 3), (x + 120, h // 3 + 120), (30, 200, 240), -1)
    cv2.putText(frame, f"CUE fake camera {cam}  frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--cam", default="D")
    ap.add_argument("--code", required=True)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--image", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    from livekit import rtc

    w, h = (int(x) for x in a.size.lower().split("x"))
    img = None
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            print(f"cannot read {a.image}", file=sys.stderr)
            return 1
        img = letterbox(img, w, h)
    r = httpx.get(f"{a.server}/api/livekit/token", params={"cam": a.cam, "code": a.code, "label": a.label or f"fake {a.cam}"}, timeout=10)
    if r.status_code != 200:
        print(f"token refused: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        return 2
    tok = r.json()
    print(f"token ok: identity {tok['identity']} room {tok['room']} -> {tok['url']}")

    room = rtc.Room()
    got: list[dict] = []

    @room.on("data_received")
    def _on_data(pkt: rtc.DataPacket) -> None:
        try:
            msg = json.loads(pkt.data.decode())
        except Exception:
            return
        got.append(msg)
        print(f"  <- director: {msg}")

    t0 = time.monotonic()
    await room.connect(tok["url"], tok["token"], rtc.RoomOptions(auto_subscribe=False))
    print(f"connected in {time.monotonic() - t0:.2f}s as {room.local_participant.identity}")
    source = rtc.VideoSource(w, h)
    track = rtc.LocalVideoTrack.create_video_track(f"cam-{a.cam}", source)
    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_CAMERA
    opts.simulcast = False
    await room.local_participant.publish_track(track, opts)
    print("publishing; Ctrl-C to stop")
    i = 0
    period = 1.0 / a.fps
    end = time.monotonic() + a.seconds
    try:
        while time.monotonic() < end:
            bgr = pattern(w, h, i, a.cam.upper(), img)
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            source.capture_frame(rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, rgba.tobytes()))
            i += 1
            await asyncio.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        await room.disconnect()
    print(f"sent {i} frames in {a.seconds:.0f}s; control messages received: {[m.get('type') for m in got]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
