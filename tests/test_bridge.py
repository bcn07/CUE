"""TeamControlBridge against a fake team control API with the real revision semantics."""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Header, HTTPException

from server.bridge import TeamControlBridge, parse_camera_map

SECRET = "s3cr3t-producer"


def make_fake_team_api():
    app = FastAPI()
    st = {"mode": "ASSIST", "rev": 0, "seq": 0, "live": None, "takes": [], "modes": [], "bindings": {"CAM-GUEST": 3, "CAM-HOST": 1}}

    def auth(x_cue_producer_secret: str | None):
        if x_cue_producer_secret != SECRET:
            raise HTTPException(401, "Invalid producer credential")

    def snap():
        return {"eventId": "ev", "controlGeneration": "g1", "mode": st["mode"], "modeRevision": st["rev"], "decisionSequence": st["seq"],
                "liveCameraId": st["live"], "liveStreamEpoch": None, "pendingDecisionId": None}

    @app.get("/api/v1/events/{ev}/control-state")
    def control_state(ev: str, x_cue_producer_secret: str | None = Header(default=None)):
        auth(x_cue_producer_secret)
        return snap()

    @app.get("/api/v1/events/{ev}/bindings")
    def bindings(ev: str, x_cue_producer_secret: str | None = Header(default=None)):
        auth(x_cue_producer_secret)
        return [{"cameraId": k, "streamEpoch": v} for k, v in st["bindings"].items()]

    @app.post("/api/v1/events/{ev}/take")
    def take(ev: str, body: dict, x_cue_producer_secret: str | None = Header(default=None)):
        auth(x_cue_producer_secret)
        if body["expectedRevision"] != st["rev"]:
            raise HTTPException(409, "revision")
        st["rev"] += 1
        st["seq"] += 1
        st["takes"].append(body)
        return {"state": snap(), "renderCommand": {"decisionId": "d", "eventId": ev, "controlGeneration": "g1", "decisionSequence": st["seq"],
                                                   "modeRevision": st["rev"], "target": "CAMERA", "cameraId": body["cameraId"],
                                                   "streamEpoch": body["streamEpoch"], "reasonCode": body["reasonCode"], "createdAtMs": 0, "expiresAtMs": 0}}

    @app.post("/api/v1/events/{ev}/mode")
    def mode(ev: str, body: dict, x_cue_producer_secret: str | None = Header(default=None)):
        auth(x_cue_producer_secret)
        if body["expectedRevision"] != st["rev"]:
            raise HTTPException(409, "revision")
        st["rev"] += 1
        st["mode"] = body["mode"]
        st["modes"].append(body)
        return {"state": snap(), "renderCommand": None}

    return app, st


@pytest.fixture()
def team_api():
    app, st = make_fake_team_api()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", st
    server.should_exit = True
    th.join(timeout=3)


def test_take_mode_and_conflict_recovery(team_api):
    base, st = team_api

    async def go():
        b = TeamControlBridge(base, "ev", SECRET, parse_camera_map("A=CAM-WIDE,B=CAM-GUEST,C=CAM-HOST"))
        assert (await b.check())["mode"] == "ASSIST"
        res = await b.mirror_take("B", "identity", "fresh identity")
        assert res["renderCommand"]["cameraId"] == "CAM-GUEST" and res["renderCommand"]["streamEpoch"] == 3   # epoch from bindings
        assert st["takes"][-1]["reasonCode"] == "CUE_IDENTITY" and st["takes"][-1]["idempotencyKey"].startswith("cue-")
        # unmapped director camera: skipped, never sent
        assert await b.mirror_take("Z", "identity", "x") is None and b.stats["skipped"] == 1
        # HOLD / AUTO
        assert (await b.mirror_mode(True))["state"]["mode"] == "MANUAL_HOLD"
        assert (await b.mirror_mode(False))["state"]["mode"] == "ASSIST"
        assert (await b.mirror_mode(False))["mode"] == "ASSIST"   # already there: no write
        assert len(st["modes"]) == 2
        # explicit AUTO resume mode when the user decides CUE cuts on air
        b_auto = TeamControlBridge(base, "ev", SECRET, parse_camera_map("B=CAM-GUEST"), resume_mode="auto")
        assert (await b_auto.mirror_mode(False))["state"]["mode"] == "AUTO"
        assert (await b.mirror_mode(False))["state"]["mode"] == "ASSIST"
        await b_auto.close()
        # reason codes never impersonate the operator
        assert all(not t["reasonCode"].upper().startswith("MANUAL") for t in st["takes"])
        # someone else bumps the revision between our read and write: first POST 409s, retry succeeds
        real_state = b.state

        raced = {"done": False}

        async def racing_state():
            s = await real_state()
            if not raced["done"]:
                raced["done"] = True
                st["rev"] += 1  # a competing producer acted after we read, once
            return s
        b.state = racing_state  # type: ignore[method-assign]
        res = await b.mirror_take("C", "static", "static mapping")
        b.state = real_state  # type: ignore[method-assign]
        assert res is not None and res["renderCommand"]["cameraId"] == "CAM-HOST" and res["renderCommand"]["streamEpoch"] == 1
        assert b.stats["takes"] == 2 and b.stats["errors"] == 0
        # wrong secret: reported, never raises into the director
        bad = TeamControlBridge(base, "ev", "wrong", parse_camera_map("B=CAM-GUEST"))
        assert await bad.mirror_take("B", "identity", "x") is None and bad.stats["errors"] >= 1 and "401" in bad.stats["last_error"]
        await b.close(); await bad.close()

    asyncio.run(go())


def test_parse_camera_map():
    assert parse_camera_map(" a=cam-wide , B=CAM-GUEST,C=nope,=x") == {"A": "CAM-WIDE", "B": "CAM-GUEST"}
