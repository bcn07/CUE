"""End-to-end without laptops, mic or API keys: real server process, fake cameras
streaming real face photos, typed clauses through the real pipeline.
Verifies: ingest -> identity -> rules interpreter -> director -> program switch,
min-shot hold + pending re-evaluation, non-NOW never cuts, dead-camera failover."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (FX / "obama.jpg").exists() or not (MODELS / "face_recognition_sface_2021dec.onnx").exists()
    or (MODELS / "face_recognition_sface_2021dec.onnx").stat().st_size < 10_000,
    reason="fixture photos or face models missing",
)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeCam(threading.Thread):
    def __init__(self, port: int, cam: str, image: Path | None, fps: float = 10):
        super().__init__(daemon=True)
        self.url = f"ws://127.0.0.1:{port}/ingest?cam={cam}&label=Fake{cam}"
        self.cam = cam
        self.fps = fps
        self.stop_ev = threading.Event()
        import cv2
        import numpy as np
        if image:
            img = cv2.imread(str(image))
            h, w = img.shape[:2]
            s = min(640 / w, 360 / h)
            img = cv2.resize(img, (int(w * s), int(h * s)))
            canvas = np.zeros((360, 640, 3), np.uint8)
            canvas[(360 - img.shape[0]) // 2:(360 - img.shape[0]) // 2 + img.shape[0], (640 - img.shape[1]) // 2:(640 - img.shape[1]) // 2 + img.shape[1]] = img
        else:
            canvas = np.full((360, 640, 3), 40, np.uint8)
            cv2.putText(canvas, "WIDE", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.jpeg = buf.tobytes()
        self.sent = 0

    def run(self):
        from websockets.sync.client import connect
        while not self.stop_ev.is_set():
            try:
                with connect(self.url, max_size=None) as ws:
                    ws.send(json.dumps({"type": "hello", "cam": self.cam}))
                    while not self.stop_ev.is_set():
                        ws.send(self.jpeg)
                        self.sent += 1
                        time.sleep(1 / self.fps)
            except Exception:
                if not self.stop_ev.is_set():
                    time.sleep(0.3)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = free_port()
    data = tmp_path_factory.mktemp("data")
    env = {**os.environ, "CUE_PORT": str(port), "CUE_DATA_DIR": str(data), "OPENAI_API_KEY": "", "DEEPGRAM_API_KEY": "", "CUE_LLM_BASE_URL": "",
           "CUE_IDENT_INTERVAL_S": "0.1", "CUE_MIN_SHOT_S": "2.5", "CUE_CAMERA_STALE_S": "1.5"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            if httpx.get(base + "/api/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        pytest.fail("server did not start:\n" + out[-4000:])
    yield base, port
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def wait_for(pred, timeout=10.0, step=0.15):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = pred()
        if last:
            return last
        time.sleep(step)
    return last


def test_full_pipeline_rules_only(server):
    base, port = server
    c = httpx.Client(base_url=base, timeout=20)

    # 1. setup: two people with photos (multipart), roles, cameras
    r = c.post("/api/people", data={"name": "Barack Obama", "aliases": "Barack, Mr Obama", "role": "guest of honour", "is_host": "false"},
               files=[("photos", ("obama.jpg", (FX / "obama.jpg").read_bytes(), "image/jpeg"))])
    assert r.status_code == 200 and r.json()["faces_enrolled"] == 1, r.text
    assert r.json()["id"] == "barack"
    r = c.post("/api/people", data={"name": "Joe Biden", "aliases": "Joe", "role": "host", "is_host": "true"},
               files=[("photos", ("biden.jpg", (FX / "biden.jpg").read_bytes(), "image/jpeg"))])
    assert r.status_code == 200 and r.json()["faces_enrolled"] == 1
    r = c.post("/api/script", json={"text": "Run of show: Joe hosts. Segment 2: Barack Obama speaks."})
    assert r.status_code == 200
    r = c.post("/api/cameras", json={"cameras": {"A": {"label": "Wide", "role": "wide"}, "B": {"label": "Guest close-up", "role": "guest"},
                                                  "C": {"label": "Host cam", "role": "host"}}})
    assert r.status_code == 200
    th = c.get("/api/people/barack/thumb")
    assert th.status_code == 200 and th.headers["content-type"] == "image/jpeg" and th.content[:2] == b"\xff\xd8"
    assert c.get("/api/people/nobody/thumb").status_code == 404
    st = c.get("/api/state").json()
    assert st["host_id"] == "joe" and st["services"]["identity"]["gallery_size"] == 2
    assert st["services"]["llm"]["configured"] is False  # rules-only run

    # 2. cameras: A wide (no face), B shows Obama (a different photo than enrolled), C shows Biden
    cams = {"A": FakeCam(port, "A", None), "B": FakeCam(port, "B", FX / "obama2.jpg"), "C": FakeCam(port, "C", FX / "biden.jpg")}
    for cam in cams.values():
        cam.start()
    try:
        st = wait_for(lambda: (lambda s: s if all(s["cameras"][k]["healthy"] for k in "ABC") else None)(c.get("/api/state").json()), 10)
        assert st, "cameras never became healthy"
        # first healthy camera -> wide is taken automatically
        st = wait_for(lambda: (lambda s: s if s["program"]["camera_id"] == "A" else None)(c.get("/api/state").json()), 5)
        assert st and st["program"]["camera_id"] == "A"
        t_wide = time.time()
        # identity confirms who is on which camera
        st = wait_for(lambda: (lambda s: s if "barack" in s["cameras"]["B"]["identity"]["present"] and "joe" in s["cameras"]["C"]["identity"]["present"] else None)(c.get("/api/state").json()), 12)
        assert st, c.get("/api/state").json()["cameras"]
        assert "barack" not in st["cameras"]["C"]["identity"]["present"]
        assert st["cameras"]["B"]["identity"]["faces"][0]["person_id"] == "barack"

        # 3. host names the guest -> SHOW -> program goes to the camera with the confirmed face
        time.sleep(max(0.0, 2.6 - (time.time() - t_wide)))  # the automatic wide take also owns a minimum shot
        r = c.post("/api/say", json={"text": "Please welcome Barack Obama!"}).json()
        assert r["last_cue"]["action"] == "SHOW" and r["last_cue"]["target_ids"] == ["barack"] and r["last_cue"]["source"] == "rules", r
        assert r["last_decision"]["action"] == "TAKE" and r["last_decision"]["camera_id"] == "B" and r["last_decision"]["evidence"] == "identity", r
        assert r["program"] == "B"
        t_cut = time.time()

        # 4. within the min shot, a new utterance is held, kept pending, then executed when the shot may end
        r = c.post("/api/say", json={"text": "Thank you Barack."}).json()
        assert r["last_cue"]["action"] == "HOST" and r["last_decision"]["action"] == "STAY" and "min shot" in r["last_decision"]["reason"]
        assert r["program"] == "B"
        st = wait_for(lambda: (lambda s: s if s["program"]["camera_id"] == "C" else None)(c.get("/api/state").json()), 4)
        assert st and st["program"]["camera_id"] == "C", c.get("/api/state").json()["events"][-6:]
        assert 2.0 <= time.time() - t_cut <= 4.0
        assert st["last_decision"]["evidence"] == "identity" and st["last_decision"]["target"] == "joe"

        # 5. non-NOW never cuts
        time.sleep(2.6)
        r = c.post("/api/say", json={"text": "Barack joins us again later this evening."}).json()
        assert r["last_cue"]["action"] == "HOLD" and r["last_cue"]["temporal_intent"] == "FUTURE"
        assert r["program"] == "C"
        r = c.post("/api/say", json={"text": "Who is Barack anyway?"}).json()
        assert r["last_cue"]["action"] == "HOLD" and r["program"] == "C"

        # 6. correction inside one utterance: "welcome Joe. Actually Barack." -> cut to host cam, then ONE fast re-cut to B
        assert c.post("/api/control", json={"action": "take", "camera_id": "A"}).json()["program"] == "A"
        time.sleep(2.6)
        r = c.post("/api/say", json={"text": "Please welcome Joe. Actually Barack, come on up."}).json()
        # a pending cue must be dropped when the host defers: SHOW held by min shot, then FUTURE
        
        assert r["program"] == "B", r
        st = c.get("/api/state").json()
        cuts = [e for e in st["events"] if e["kind"] == "cut"]
        assert cuts[-2]["text"].startswith("TAKE C") and cuts[-1]["text"].startswith("TAKE B") and "correction" in cuts[-1]["text"], cuts[-3:]

        # 7. manual controls
        assert c.post("/api/control", json={"action": "hold"}).json()["mode"] == "HOLD"
        r = c.post("/api/say", json={"text": "Over to Joe."}).json()
        assert r["program"] == "B" and "HOLD" in r["last_decision"]["reason"]
        assert c.post("/api/control", json={"action": "auto"}).json()["mode"] == "AUTO"
        assert c.post("/api/control", json={"action": "take", "camera_id": "A"}).json()["program"] == "A"

        # 8. latency panel has numbers
        L = c.get("/api/state").json()["latency"]
        assert L["clause_to_cut_ms"]["n"] >= 2 and L["interpret_ms"]["p95"] < 50 and L["identity_ms"]["last"] < 200

        # 9. operator labels win over what the laptop announces (FakeCam sends label=FakeX)
        assert c.get("/api/state").json()["cameras"]["B"]["label"] == "Guest close-up"

        # 10. a reconnecting camera: the newer connection owns the id exclusively; the old socket is closed by
        #     the server and its closing must not mark the new stream disconnected
        old_sent = cams["C"].sent
        c2 = FakeCam(port, "C", FX / "biden.jpg")
        c2.start()
        time.sleep(0.6)
        cams["C"].stop_ev.set()           # the replaced client would otherwise reconnect and take the id back
        cams["C"] = c2
        time.sleep(2.0)
        st = c.get("/api/state").json()
        assert st["cameras"]["C"]["connected"] and st["cameras"]["C"]["healthy"], st["cameras"]["C"]
        assert any("replaces the previous" in e["text"] for e in st["events"] if e["kind"] == "camera")

        # 10b. a deferral drops a cue that was held by the minimum shot (A is live; cut to C starts a fresh shot)
        assert c.post("/api/control", json={"action": "take", "camera_id": "C"}).json()["program"] == "C"
        time.sleep(0.2)
        r = c.post("/api/say", json={"text": "Over to Barack."}).json()
        assert r["last_decision"]["action"] == "STAY" and "min shot" in r["last_decision"]["reason"], r
        assert c.get("/api/state").json()["pending_cue"] is not None
        r = c.post("/api/say", json={"text": "Actually, Barack joins us later."}).json()
        assert r["last_cue"]["temporal_intent"] == "FUTURE"
        st = c.get("/api/state").json()
        assert st["pending_cue"] is None, st["events"][-4:]
        time.sleep(3.0)
        assert c.get("/api/state").json()["program"]["camera_id"] == "C"

        # 11. recording: program output + cut log (+ audio when a mic exists), files served back
        r = c.post("/api/record", json={"action": "start"}).json()
        assert r["ok"] and c.get("/api/state").json()["recording"]["active"]
        time.sleep(2.6)
        c.post("/api/control", json={"action": "take", "camera_id": "C"})
        c.post("/api/say", json={"text": "Barack, what do you think about the demo?"})
        time.sleep(1.0)
        r = c.post("/api/record", json={"action": "stop"}).json()["recording"]
        assert r["status"] == "done" and r["frames"] >= 30 and r["cuts"] >= 1, r
        assert "h264" in r.get("video_codec", "") or "cv2" in r.get("video_codec", "")
        name = Path(r["dir"]).name
        assert c.get(f"/recordings/{name}/{r['final']}").status_code == 200
        assert c.get(f"/recordings/{name}/cuts.jsonl").text.strip()
        tlines = [json.loads(l) for l in c.get(f"/recordings/{name}/transcript.jsonl").text.splitlines() if l.strip()]
        assert tlines and all(t["source"] == "typed" for t in tlines), tlines   # typed test input is never labelled as the mic
        assert any(x["name"] == name for x in c.get("/api/recordings").json()["recordings"])
        assert not c.get("/api/state").json()["recording"]["active"]

        # 12. no path traversal through the person id
        assert c.delete("/api/people/..").status_code == 404
        assert c.delete("/api/people/%2e%2e").status_code == 404
        assert len(c.get("/api/people").json()["people"]) == 2

        # 13. the live camera dies -> failover to a healthy camera within ~2 s
        c.post("/api/control", json={"action": "take", "camera_id": "B"})
        time.sleep(0.3)
        cams["B"].stop_ev.set()
        st = wait_for(lambda: (lambda s: s if s["program"]["camera_id"] != "B" else None)(c.get("/api/state").json()), 6)
        assert st and st["program"]["camera_id"] == "A", st["program"]
    finally:
        for cam in cams.values():
            cam.stop_ev.set()


def test_ui_websocket_receives_frames_and_state(server):
    import asyncio

    import websockets

    base, port = server
    cam = FakeCam(port, "Z", None, fps=10)
    cam.start()

    async def go():
        async with websockets.connect(f"ws://127.0.0.1:{port}/ui", max_size=None) as ws:
            got_state = got_frame = False
            deadline = time.time() + 8
            while time.time() < deadline and not (got_state and got_frame):
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, bytes):
                    n = msg[0]
                    cam_id = msg[1:1 + n].decode()
                    if cam_id == "Z" and msg[1 + n:1 + n + 2] == b"\xff\xd8":
                        got_frame = True
                else:
                    j = json.loads(msg)
                    if j.get("type") == "state" and "Z" in j["cameras"]:
                        got_state = True
            assert got_state and got_frame
            await ws.send(json.dumps({"type": "say", "text": "Nothing to see here."}))
            j = None
            for _ in range(20):
                m = await asyncio.wait_for(ws.recv(), timeout=5)
                if not isinstance(m, bytes):
                    j = json.loads(m)
                    if j.get("last_cue") and j["last_cue"]["clause_text"] == "Nothing to see here.":
                        break
            assert j and j["last_cue"]["action"] == "HOLD"

    try:
        asyncio.run(go())
    finally:
        cam.stop_ev.set()
