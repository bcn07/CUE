"""Desk UI adapter: serve hemadassani/cue-desk-ui and speak its contract on top of the one-shot.

Additive only. It registers two routes, `GET /desk` (the page) and `WS /ws` (the
desk contract), and subscribes to the same state and frame fan-out the /ui page
uses. The /ui protocol is untouched. Like /ui, neither route exists through the
public tunnel host.

Desk contract (from the cue-desk-ui README): server -> desk `source`, `directing`,
`caption_status`, `deepgram_config`, `roster`, `mode`, `latency`,
`caption_provisional`, `caption_final`, `deepgram_result`, `decision`
(DecisionRecord), `camera_state`, `prepare`, `stand_down`; desk -> server
`manual_take`, `set_mode`, `accept_suggestion`, `skip_suggestion`, `rate`.
Frames go to the desk as the same binary the /ui page gets, with the camera id
in the header rewritten to the desk's CAM-HOST / CAM-GUEST / CAM-WIDE.

Honest limits: the one-shot has AUTO and HOLD only, so the desk's "assist" maps
to HOLD and the desk is told so; the one-shot measures no Deepgram first-word
latency, so the desk's latency row stays empty rather than showing a number
that means something else.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

log = logging.getLogger("cue.desk")

ROLE_TO_DESK = {"host": "CAM-HOST", "guest": "CAM-GUEST", "wide": "CAM-WIDE"}
DESK_CAMS = ("CAM-HOST", "CAM-GUEST", "CAM-WIDE")
MAX_QUEUED_TEXT = 200


# ------------------------------------------------------------------ pure


def camera_map(cameras: dict[str, dict]) -> dict[str, str]:
    """One-shot camera id -> desk camera id, by role. First camera per role wins."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for cid, cam in cameras.items():
        desk = ROLE_TO_DESK.get(str(cam.get("role", "")).lower())
        if desk and desk not in used:
            out[cid] = desk
            used.add(desk)
    return out


def supports_assist(snapshot: dict | None) -> bool:
    """The director exposes ASSIST once its snapshot carries the `assist` flag."""
    return bool(snapshot) and "assist" in snapshot


def desk_mode(mode: str | None, assist: bool = False) -> str:
    if assist:
        return "assist"
    return "manual" if str(mode or "").upper() == "HOLD" else "auto"


def snapshot_mode(snapshot: dict | None) -> str:
    snapshot = snapshot or {}
    return desk_mode(snapshot.get("mode"), bool(snapshot.get("assist")))


def suggestion_record(snapshot: dict, cmap: dict[str, str]) -> dict | None:
    """The director's current suggestion as a DecisionRecord without `accepted`,
    which is what makes the desk show its suggestion card in assist mode."""
    sug = snapshot.get("suggestion")
    if not sug:
        return None
    raw_cam = sug.get("camera_id")
    desk_cam = cmap.get(raw_cam, raw_cam) if raw_cam else None
    if not desk_cam:
        return None
    cue = snapshot.get("last_cue") or {} if sug.get("origin") == "cue" else {}
    codes = [str(c) for c in (sug.get("reason_codes") or [])]
    return {
        "decision_seq": sug.get("state_version") or snapshot.get("decision_ver"),
        "action": "TAKE",
        "camera_id": desk_cam,
        "reason": "suggest: " + (", ".join(codes) if codes else str(sug.get("origin") or "director")),
        "plain_reason": str(sug.get("explanation") or "").strip() or f"CUE suggests the {desk_cam}.",
        "transcript_span": {"text": cue.get("clause_text") or cue.get("evidence_text") or ""},
        "cue_summary": {
            "target_guest_ids": list(cue.get("target_ids") or []),
            "temporal_intent": cue.get("temporal_intent"),
            "scope": cue.get("scope"),
        },
        "latencies_ms": {},
        "confidence": sug.get("confidence"),
    }


def _suggestion_key(snapshot: dict | None) -> tuple | None:
    sug = (snapshot or {}).get("suggestion")
    if not sug:
        return None
    return (sug.get("camera_id"), sug.get("state_version"), sug.get("at_wall"))


def plain_reason(action: str, reason: str, evidence: str, desk_cam: str | None) -> str:
    names = {"CAM-HOST": "host camera", "CAM-GUEST": "guest camera", "CAM-WIDE": "wide shot"}
    where = names.get(desk_cam or "", "that camera")
    r = (reason or "").lower()
    if action == "SLATE":
        return "No usable camera, so the safe picture."
    if action == "TAKE" and r.startswith("manual"):
        return f"You took the {where}."
    if action == "STAY":
        if "hold" in r:
            return "Automatic cuts are paused."
        if "future" in r or "later" in r:
            return "That is for later, not now."
        if "negat" in r or "not " in r:
            return "The host said not to."
        return reason[:1].upper() + reason[1:] if reason else "Stayed put."
    if evidence == "identity":
        return f"The person named is in the {where}."
    if evidence == "host-role":
        return "Back to the host."
    if evidence == "wide":
        return "More than one person, so the wide shot."
    return reason[:1].upper() + reason[1:] if reason else f"Cut to the {where}."


def decision_record(snapshot: dict, cmap: dict[str, str]) -> dict | None:
    d = snapshot.get("last_decision")
    if not d:
        return None
    action = str(d.get("action") or "STAY").upper()
    reason = str(d.get("reason") or "")
    evidence = str(d.get("evidence") or "")
    raw_cam = d.get("camera_id")
    desk_cam = cmap.get(raw_cam, raw_cam) if raw_cam else None
    # Only an operator TAKE is "manual" to the desk. The one-shot also prefixes its held-slate
    # reason with "manual", and the desk reads a leading "manual" as an operator take.
    manual = action == "TAKE" and reason.lower().startswith("manual")
    if not manual and reason.lower().startswith("manual"):
        reason = "while held: " + reason
    cue = {} if manual else (snapshot.get("last_cue") or {})
    lat = snapshot.get("latency") or {}
    decide = (lat.get("decide_ms") or {}).get("last")
    cue_ms = cue.get("latency_ms")
    cue_decide = None
    if not manual and (cue_ms is not None or decide is not None):
        cue_decide = round(float(cue_ms or 0) + float(decide or 0), 1)
    return {
        "decision_seq": snapshot.get("decision_ver"),
        "action": action,
        "camera_id": desk_cam,
        "reason": reason,
        "plain_reason": plain_reason(action, reason, evidence, desk_cam),
        "transcript_span": {"text": cue.get("clause_text") or cue.get("evidence_text") or ""},
        "cue_summary": {
            "target_guest_ids": list(cue.get("target_ids") or []),
            "temporal_intent": cue.get("temporal_intent"),
            "scope": cue.get("scope"),
        },
        "latencies_ms": {"cue_decide_ms": cue_decide},
    }


def camera_states(snapshot: dict, cmap: dict[str, str]) -> dict[str, dict]:
    """Desk camera id -> {ready, note} from the one-shot's camera health."""
    out: dict[str, dict] = {}
    for cid, cam in (snapshot.get("cameras") or {}).items():
        desk = cmap.get(cid)
        if not desk:
            continue
        connected = bool(cam.get("connected"))
        healthy = bool(cam.get("healthy"))
        if connected and healthy:
            note = f"{cam.get('fps', 0)} fps"
        elif connected:
            note = "Stalled"
        else:
            note = "Not connected"
        out[desk] = {"ready": connected and healthy, "note": note}
    return out


def _deepgram(snapshot: dict) -> dict:
    return ((snapshot.get("services") or {}).get("deepgram")) or {}


def _caption_connected(snapshot: dict) -> bool:
    dg = _deepgram(snapshot)
    if not dg.get("configured"):
        return False
    return bool(dg.get("connected") or dg.get("open") or dg.get("sender_alive"))


def pending_camera(snapshot: dict, cmap: dict[str, str]) -> str | None:
    """Which desk camera a pending cue would take, if that can be told from the snapshot."""
    cue = snapshot.get("pending_cue")
    if not cue:
        return None
    cams = snapshot.get("cameras") or {}
    action = str(cue.get("action") or "").upper()
    if action == "WIDE":
        return next((cmap[c] for c, cam in cams.items() if cam.get("role") == "wide" and c in cmap), None)
    if action == "HOST":
        return next((cmap[c] for c, cam in cams.items() if cam.get("role") == "host" and c in cmap), None)
    targets = list(cue.get("target_ids") or [])
    for cid, cam in cams.items():
        if cam.get("fixed_person") and cam["fixed_person"] in targets and cid in cmap:
            return cmap[cid]
    return None


def on_connect_messages(snapshot: dict, cmap: dict[str, str]) -> list[dict]:
    dg = _deepgram(snapshot)
    roster = snapshot.get("roster") or []
    keyterms: list[str] = []
    for person in roster:
        for name in [person.get("name")] + list(person.get("aliases") or []):
            if name and name not in keyterms:
                keyterms.append(name)
    msgs: list[dict] = [
        {"type": "source", "source": "mic" if dg.get("configured") else "fixture"},
        {"type": "directing", "enabled": snapshot_mode(snapshot) == "auto"},
        {
            "type": "caption_status",
            "connected": _caption_connected(snapshot),
            "label": f"Deepgram {dg.get('model')}" if dg.get("configured") else "No Deepgram key: typed lines only",
        },
        {"type": "deepgram_config", "model": dg.get("model"), "keyterms": keyterms},
        {
            "type": "roster",
            "guests": [
                {"name": p.get("name"), "role": "Host" if p.get("is_host") else (p.get("role") or "Guest")}
                for p in roster
            ],
        },
        {"type": "mode", "mode": snapshot_mode(snapshot)},
    ]
    for desk, info in camera_states(snapshot, cmap).items():
        msgs.append({"type": "camera_state", "camera": desk, **info})
    record = decision_record(snapshot, cmap)
    if record:
        msgs.append({"type": "decision", "record": record})
    sug = suggestion_record(snapshot, cmap)
    if sug:
        msgs.append({"type": "prepare", "camera": sug["camera_id"]})
        msgs.append({"type": "decision", "record": sug})
    return msgs


def diff_messages(prev: dict | None, cur: dict, cmap: dict[str, str]) -> list[dict]:
    """What changed between two snapshots, as desk messages. `prev` None means only live values."""
    msgs: list[dict] = []
    prev = prev or {}

    if snapshot_mode(prev) != snapshot_mode(cur):
        msgs.append({"type": "mode", "mode": snapshot_mode(cur)})
        msgs.append({"type": "directing", "enabled": snapshot_mode(cur) == "auto"})

    interim = (cur.get("interim") or {}).get("text") or ""
    if interim and interim != ((prev.get("interim") or {}).get("text") or ""):
        msgs.append({"type": "caption_provisional", "text": interim})

    seen_final = {t.get("utterance_id") for t in (prev.get("transcript") or []) if t.get("final")}
    for t in cur.get("transcript") or []:
        if t.get("final") and t.get("utterance_id") not in seen_final and t.get("text"):
            msgs.append({"type": "caption_final", "text": t["text"]})
            msgs.append({"type": "deepgram_result", "text": t["text"], "is_final": True, "speech_final": True})

    if cur.get("decision_ver") != prev.get("decision_ver"):
        record = decision_record(cur, cmap)
        if record:
            msgs.append({"type": "decision", "record": record})

    before = camera_states(prev, camera_map(prev.get("cameras") or {})) if prev else {}
    for desk, info in camera_states(cur, cmap).items():
        if before.get(desk) != info:
            msgs.append({"type": "camera_state", "camera": desk, **info})

    had = bool(prev.get("pending_cue"))
    has = bool(cur.get("pending_cue"))
    if has and not had:
        cam = pending_camera(cur, cmap)
        if cam:
            msgs.append({"type": "prepare", "camera": cam})
    elif had and not has:
        msgs.append({"type": "stand_down"})

    if _suggestion_key(prev) != _suggestion_key(cur):
        sug = suggestion_record(cur, cmap)
        if sug:
            msgs.append({"type": "prepare", "camera": sug["camera_id"]})
            msgs.append({"type": "decision", "record": sug})
        elif _suggestion_key(prev) is not None:
            msgs.append({"type": "stand_down"})

    if prev and _caption_connected(prev) != _caption_connected(cur):
        dg = _deepgram(cur)
        msgs.append({"type": "caption_status", "connected": _caption_connected(cur), "label": f"Deepgram {dg.get('model')}"})

    level = _deepgram(cur).get("mic_level")
    if level is not None:
        msgs.append({"type": "mic_level", "level": float(level)})
    return msgs


def rewrite_frame(payload: bytes, cmap: dict[str, str]) -> bytes | None:
    """[u8 idlen][id][jpeg] with the one-shot id swapped for the desk id; None if unmapped."""
    if not payload:
        return None
    n = payload[0]
    cid = payload[1 : 1 + n].decode("utf-8", errors="replace")
    desk = cmap.get(cid)
    if not desk:
        return None
    did = desk.encode()
    return bytes([len(did)]) + did + payload[1 + n :]


def desk_to_control(
    msg: dict, cmap: dict[str, str], assist_supported: bool = False
) -> tuple[str, str | None] | None:
    """Desk message -> (action, camera_id) for Show.control, or None when nothing to do.

    With a director that has ASSIST, the desk's assist, accept and skip map straight
    through; without it, assist falls back to HOLD and accept/skip do nothing."""
    t = msg.get("type")
    inverse = {desk: cid for cid, desk in cmap.items()}
    if t == "manual_take":
        cam = inverse.get(str(msg.get("camera") or ""))
        if not cam:
            raise HTTPException(400, f"no camera is bound to {msg.get('camera')}")
        return ("take", cam)
    if t == "set_mode":
        mode = str(msg.get("mode") or "").lower()
        if mode == "auto":
            return ("auto", None)
        if mode == "assist":
            return ("assist", None) if assist_supported else ("hold", None)
        if mode in ("manual", "hold"):
            return ("hold", None)
        raise HTTPException(400, f"unknown mode {mode}")
    if t == "accept_suggestion":
        return ("accept", None) if assist_supported else None
    if t == "skip_suggestion":
        return ("skip", None) if assist_supported else None
    return None


# ------------------------------------------------------------------ live


class DeskClient:
    """One desk browser: same shape as UIClient so Show fans state and frames to it."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.frames: dict[str, bytes] = {}
        self.texts: deque[str] = deque(maxlen=MAX_QUEUED_TEXT)
        self.wake = asyncio.Event()
        self.alive = True
        self.cmap: dict[str, str] = {}
        self.prev: dict | None = None

    # Show.relay_frame calls this with the /ui binary payload.
    def offer(self, cam_id: str, payload: bytes) -> None:
        self.frames[cam_id] = payload
        self.wake.set()

    # Show's state loop calls this with the /ui snapshot JSON.
    def offer_state(self, payload: str) -> None:
        try:
            cur = json.loads(payload)
        except json.JSONDecodeError:
            return
        self.cmap = camera_map(cur.get("cameras") or {})
        for msg in diff_messages(self.prev, cur, self.cmap):
            self.texts.append(json.dumps(msg))
        self.prev = cur
        self.wake.set()

    def queue(self, msg: dict) -> None:
        self.texts.append(json.dumps(msg))
        self.wake.set()

    async def sender(self) -> None:
        try:
            while self.alive:
                await self.wake.wait()
                self.wake.clear()
                while self.texts:
                    await self.ws.send_text(self.texts.popleft())
                frames, self.frames = self.frames, {}
                for cam_id, payload in frames.items():
                    out = rewrite_frame(payload, self.cmap)
                    if out is not None:
                        await self.ws.send_bytes(out)
        except Exception as e:  # noqa: BLE001 -- a dead desk socket only ends this client
            log.debug("desk sender ended: %s", e)
            self.alive = False


def install(app: FastAPI, show: Any, static_dir: Path, is_public_host: Any) -> None:
    """Register /desk and /ws. Called once from app.py."""

    @app.get("/desk")
    async def desk_page(request: Request) -> Response:
        if is_public_host(request.headers.get("host", "")):
            return Response("not found", status_code=404)
        return FileResponse(str(Path(static_dir) / "desk.html"))

    @app.websocket("/ws")
    async def ws_desk(ws: WebSocket) -> None:
        if is_public_host(ws.headers.get("host", "")):
            await ws.close(code=4404, reason="not found")
            return
        await ws.accept()
        client = DeskClient(ws)
        snapshot = show.snapshot()
        client.cmap = camera_map(snapshot.get("cameras") or {})
        client.prev = snapshot
        for msg in on_connect_messages(snapshot, client.cmap):
            client.queue(msg)
        show.ui_clients.add(client)
        sender = asyncio.create_task(client.sender())
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                t = msg.get("type")
                try:
                    assist_ok = supports_assist(client.prev)
                    if t in ("manual_take", "set_mode", "accept_suggestion", "skip_suggestion"):
                        mapped = desk_to_control(msg, client.cmap, assist_ok)
                        if mapped:
                            show.control(*mapped)
                        if t == "set_mode" and str(msg.get("mode")).lower() == "assist" and not assist_ok:
                            client.queue({"type": "mode", "mode": "manual"})
                            client.queue({"type": "error", "error": "This director has AUTO and HOLD only; assist is treated as take-over"})
                        elif mapped is None and t in ("accept_suggestion", "skip_suggestion"):
                            show.log_event("desk", f"{t} for decision {msg.get('decision_seq')}: no suggestions in this director")
                    elif t == "rate":
                        show.log_event("rate", f"decision {msg.get('decision_seq')} rated {'right' if msg.get('right') else 'wrong' if msg.get('right') is False else 'cleared'}")
                except HTTPException as e:
                    client.queue({"type": "error", "error": e.detail})
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("desk ws closed: %s", e)
        finally:
            client.alive = False
            client.wake.set()
            sender.cancel()
            show.ui_clients.discard(client)
