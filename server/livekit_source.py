"""LiveKit video transport for the one-shot director.

Phones publish WebRTC video to a LiveKit Cloud room. The director joins the same room as a
subscribe-only participant, decodes every remote video track and re-encodes it as JPEG at a
bounded rate into the same Camera path the /ingest WebSocket uses, so identity, observers,
the recorder and the UI fan-out do not change. Control back to the phone (PTZ, standby, ack,
replaced) travels as LiveKit data messages on topic "cue".

Ownership per camera id matches /ingest: one live publisher; a second one waits in STANDBY
until the director approves the replacement (POST /api/cameras/{id}/approve) or the live
publisher is gone. A LiveKit-owned camera has `owner_ws is None` and `client` starting with
"livekit:"; while a phone waits, `pending_ws` holds its participant identity as a marker so
the existing approve route and snapshot work unchanged.

Secrets never leave the process: tokens are minted here and handed to the phone page over
the join-code gated /api/livekit/token route.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from datetime import timedelta
from typing import Any, Awaitable, Callable

import cv2
import numpy as np

log = logging.getLogger("cue.livekit")

CONTROL_TOPIC = "cue"
DIRECTOR_IDENTITY = f"director-{secrets.token_hex(3)}"   # unique per process: a second director (tests, a rehearsal) must not kick the live one
IDENTITY_PREFIX = "cam-"


# ------------------------------------------------------------------ tokens
def mint_token(api_key: str, api_secret: str, room: str, identity: str, *, name: str = "", metadata: dict | None = None,
               publish: bool, subscribe: bool, ttl_h: float = 8.0) -> str:
    from livekit import api

    grants = api.VideoGrants(room_join=True, room=room, can_publish=publish, can_subscribe=subscribe,
                             can_publish_data=True, can_publish_sources=["camera"] if publish else [])
    tok = api.AccessToken(api_key, api_secret).with_identity(identity).with_grants(grants).with_ttl(timedelta(hours=ttl_h))
    if name:
        tok = tok.with_name(name)
    if metadata:
        tok = tok.with_metadata(json.dumps(metadata, separators=(",", ":")))
    return tok.to_jwt()


def publisher_identity(cam_id: str) -> str:
    return f"{IDENTITY_PREFIX}{cam_id.upper()}-{secrets.token_hex(3)}"


def cam_id_for(identity: str, metadata: str | None) -> str | None:
    """Camera id of a remote participant: metadata {"cam": "B"} first, else identity cam-B-xxxxxx."""
    if metadata:
        try:
            cid = str(json.loads(metadata).get("cam") or "").strip().upper()[:12]
            if cid:
                return cid
        except (ValueError, AttributeError):
            pass
    if identity.startswith(IDENTITY_PREFIX):
        rest = identity[len(IDENTITY_PREFIX):]
        cid = rest.split("-", 1)[0].strip().upper()[:12]
        return cid or None
    return None


def encode_bgra(width: int, height: int, data: bytes | memoryview, quality: int = 70, max_width: int = 960) -> bytes | None:
    """Packed BGRA frame -> JPEG bytes (downscaled to max_width). None on a malformed buffer."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size < width * height * 4:
            return None
        bgr = arr[: width * height * 4].reshape(height, width, 4)[:, :, :3]
        if width > max_width:
            h = max(1, round(height * max_width / width))
            bgr = cv2.resize(bgr, (max_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return buf.tobytes() if ok else None
    except Exception as e:  # never let a bad frame kill the pump
        log.debug("encode failed: %s", e)
        return None


# ------------------------------------------------------------------ source
class LiveKitSource:
    """Joins the room, owns the per-camera publisher slots, feeds JPEGs to `on_frame(cam_id, jpeg)`.

    `get_camera(cam_id, label)` returns the director's Camera object (created if new). The object
    only needs the attributes the /ingest path uses: id, label, connected, owner, owner_ws,
    pending_ws, pending_label, replace_approved, client, ptz, healthy(now, stale_s).
    """

    def __init__(self, url: str, api_key: str, api_secret: str, room: str, *, get_camera: Callable[[str, str], Any],
                 on_frame: Callable[[str, bytes], None], log_event: Callable[[str, str], None], mark_dirty: Callable[[], None],
                 camera_stale_s: float = 2.0, fps: float = 10.0, jpeg_quality: int = 70, max_width: int = 960) -> None:
        self.url, self.api_key, self.api_secret, self.room_name = url, api_key, api_secret, room
        self.get_camera, self.on_frame, self.log_event, self.mark_dirty = get_camera, on_frame, log_event, mark_dirty
        self.camera_stale_s, self.fps, self.jpeg_quality, self.max_width = camera_stale_s, fps, jpeg_quality, max_width
        self.paused: dict[str, float] = {}   # cam id -> when its phone muted the video track
        self.room: Any = None
        self.stats: dict = {"configured": True, "room": room, "connected": False, "participants": 0, "frames": 0,
                            "reconnects": 0, "last_error": "", "publishers": {}, "state": "idle", "reconciles": 0, "forced_reconnects": 0}
        self._api: Any = None                        # server API client for the liveness cross-check
        self._missed: dict[str, int] = {}            # identity seen publishing by the server but unseen here
        self.live_identity: dict[str, str] = {}      # cam id -> identity that owns it
        self.live_token: dict[str, object] = {}      # cam id -> ownership token (matches Camera.owner)
        self.pending: dict[str, str] = {}            # cam id -> identity waiting for approval
        self._pumps: dict[str, asyncio.Task] = {}    # identity -> frame pump task
        self._watchers: dict[str, asyncio.Task] = {}  # identity -> standby watcher
        self._closing = False
        self._disconnected: asyncio.Event | None = None
        self.send: Callable[[str, dict], Awaitable[bool]] = self._send_data  # overridable for tests

    # ------------------------------------------------------------ tokens for the phone page
    def publisher_token(self, cam_id: str, label: str = "") -> dict:
        identity = publisher_identity(cam_id)
        token = mint_token(self.api_key, self.api_secret, self.room_name, identity, name=label or f"Camera {cam_id.upper()}",
                           metadata={"cam": cam_id.upper(), "label": label[:40]}, publish=True, subscribe=False)
        return {"url": self.url, "token": token, "identity": identity, "room": self.room_name, "cam": cam_id.upper()}

    def director_token(self) -> str:
        return mint_token(self.api_key, self.api_secret, self.room_name, DIRECTOR_IDENTITY, name="CUE director",
                          publish=False, subscribe=True, ttl_h=48)

    # ------------------------------------------------------------ room lifecycle
    async def run(self) -> None:
        from livekit import rtc

        backoff = 2.0
        while not self._closing:
            self._disconnected = asyncio.Event()
            room = rtc.Room()
            self.room = room

            @room.on("track_subscribed")
            def _on_track(track, publication, participant):
                if getattr(track, "kind", None) == rtc.TrackKind.KIND_VIDEO:
                    self._track_arrived(participant.identity, participant.metadata, track)

            @room.on("track_unsubscribed")
            def _on_untrack(track, publication, participant):
                self._publisher_gone(participant.identity, "track ended")

            @room.on("participant_disconnected")
            def _on_leave(participant):
                self._publisher_gone(participant.identity, "left the room")

            # A phone that locks its screen or sends Safari to the background keeps its place in the
            # room but stops sending video; the camera then shows "no frames". Say why on the pages.
            @room.on("track_muted")
            def _on_muted(participant, publication):
                cid = cam_id_for(participant.identity, participant.metadata)
                if cid and getattr(publication, "kind", None) == rtc.TrackKind.KIND_VIDEO:
                    self.paused[cid] = time.monotonic()
                    self.log_event("camera", f"{cid}: the phone paused its camera (screen locked or browser in the background); wake it and reopen the page")
                    self.mark_dirty()

            @room.on("track_unmuted")
            def _on_unmuted(participant, publication):
                cid = cam_id_for(participant.identity, participant.metadata)
                if cid and cid in self.paused:
                    self.paused.pop(cid, None)
                    self.log_event("camera", f"{cid}: the phone resumed its camera")
                    self.mark_dirty()

            @room.on("disconnected")
            def _on_disc(*args):
                if self._disconnected is not None:
                    self._disconnected.set()

            @room.on("reconnected")
            def _on_reconn(*args):
                self.stats["reconnects"] += 1
                self.log_event("livekit", "room connection restored")

            try:
                await room.connect(self.url, self.director_token(), rtc.RoomOptions(auto_subscribe=True))
                self.stats.update(connected=True, last_error="")
                backoff = 2.0
                self.log_event("livekit", f"joined room {self.room_name} ({self._host()})")
                log.info("LiveKit: joined room %s at %s", self.room_name, self._host())
                self.mark_dirty()
                last_ok = last_reconcile = time.monotonic()
                connected_state = getattr(rtc.ConnectionState, "CONN_CONNECTED", 1)
                while not self._closing and not self._disconnected.is_set():
                    try:
                        await asyncio.wait_for(self._disconnected.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        pass
                    now = time.monotonic()
                    self.stats["participants"] = len(getattr(room, "remote_participants", {}) or {})
                    state = getattr(room, "connection_state", None)
                    self.stats["state"] = str(state)
                    if state == connected_state:
                        last_ok = now
                    elif now - last_ok > 10.0:
                        # The SDK's "resume" after a network change can hang without ever reporting a disconnect.
                        self.stats["forced_reconnects"] += 1
                        self.log_event("livekit", f"room session stuck in {state} for 10 s; reconnecting")
                        break
                    if now - last_reconcile >= 8.0:
                        last_reconcile = now
                        if await self._reconcile():
                            self.stats["forced_reconnects"] += 1
                            self.log_event("livekit", "publisher visible to the LiveKit server but not here; reconnecting")
                            break
                if not self._closing:
                    self.stats["connected"] = False
                    self.log_event("livekit", "room connection lost; reconnecting")
            except Exception as e:
                self.stats.update(connected=False, last_error=f"{type(e).__name__}: {e}"[:200])
                log.warning("LiveKit connect failed (%s); retry in %.0fs", self.stats["last_error"], backoff)
            finally:
                for cid in list(self.live_identity):
                    self._release(cid, "room connection lost")
                for t in list(self._pumps.values()) + list(self._watchers.values()):
                    t.cancel()
                self._pumps.clear()
                self._watchers.clear()
                self.pending.clear()
                try:
                    await room.disconnect()
                except Exception:
                    pass
                self.mark_dirty()
            if not self._closing:
                await asyncio.sleep(backoff)
                backoff = min(15.0, backoff * 1.5)

    async def close(self) -> None:
        self._closing = True
        if self._api is not None:
            try:
                await self._api.aclose()
            except Exception:
                pass
        if self._disconnected is not None:
            self._disconnected.set()
        for t in list(self._pumps.values()) + list(self._watchers.values()):
            t.cancel()
        if self.room is not None:
            try:
                await self.room.disconnect()
            except Exception:
                pass

    def _host(self) -> str:
        return self.url.split("://", 1)[-1].split("/", 1)[0]

    # ------------------------------------------------------------ publisher slots
    def _track_arrived(self, identity: str, metadata: str | None, track: Any) -> None:
        cid = cam_id_for(identity, metadata)
        if cid is None:
            log.info("LiveKit: ignoring participant %s (no camera id)", identity)
            return
        label = ""
        try:
            label = str(json.loads(metadata or "{}").get("label") or "")[:40]
        except ValueError:
            pass
        cam = self.get_camera(cid, label)
        now = time.monotonic()
        live = cam.owner_ws is not None or (cid in self.live_identity)
        live = live and cam.connected and cam.healthy(now, self.camera_stale_s)
        if identity in self._pumps:
            self._pumps[identity].cancel()
        if live and self.live_identity.get(cid) != identity:
            if self.pending.get(cid) not in (None, identity):
                asyncio.ensure_future(self.send(identity, {"type": "error", "error": "another publisher is already waiting for this camera id"}))
                return
            self.pending[cid] = identity
            cam.pending_ws = identity
            cam.pending_label = f"livekit {label or identity}".strip()
            cam.replace_approved = False
            self.log_event("camera", f"{cid}: a second publisher is waiting for approval ({cam.pending_label})")
            asyncio.ensure_future(self.send(identity, {"type": "standby", "cam": cid,
                                                       "reason": "camera id in use; waiting for the director to approve the replacement"}))
            self._watchers[identity] = asyncio.create_task(self._standby_watch(cid, identity, track), name=f"lk-standby-{cid}")
            self.mark_dirty()
            return
        self._go_live(cid, identity, track, label)

    def _go_live(self, cid: str, identity: str, track: Any, label: str) -> None:
        cam = self.get_camera(cid, label)
        old = self.live_identity.get(cid)
        if old and old != identity:
            asyncio.ensure_future(self.send(old, {"type": "replaced", "cam": cid, "reason": "replaced by an approved publisher for this camera id"}))
            t = self._pumps.pop(old, None)
            if t:
                t.cancel()
        if cam.owner_ws is not None:  # a WebSocket laptop is live on this id; the phone replaces it (approved or stale)
            try:
                asyncio.ensure_future(cam.owner_ws.close(code=4000, reason="replaced by an approved publisher for this camera id"))
            except Exception:
                pass
            cam.owner_ws = None
        token = object()
        cam.owner = token
        cam.connected = True
        cam.client = f"livekit:{identity}"
        if cam.pending_ws == identity:
            cam.pending_ws = None
            cam.pending_label = ""
        cam.replace_approved = False
        self.live_identity[cid] = identity
        self.live_token[cid] = token
        self.pending.pop(cid, None)
        self.stats["publishers"][cid] = identity
        self.log_event("camera", f"{cid} connected via LiveKit ({label or identity})")
        asyncio.ensure_future(self.send(identity, {"type": "ack", "cam": cid, "role": getattr(cam, "role", "")}))
        asyncio.ensure_future(self.send(identity, {"type": "ptz", "cam": cid, **cam.ptz}))
        self._pumps[identity] = asyncio.create_task(self._pump(cam, cid, identity, token, track), name=f"lk-pump-{cid}")
        self.mark_dirty()

    async def _standby_watch(self, cid: str, identity: str, track: Any) -> None:
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                if self.pending.get(cid) != identity:
                    return
                cam = self.get_camera(cid, "")
                live = cam.connected and cam.healthy(time.monotonic(), self.camera_stale_s)
                if cam.replace_approved or not live:
                    self.log_event("camera", f"{cid}: replacement {'approved' if cam.replace_approved else 'takes over (live publisher gone)'}")
                    self._go_live(cid, identity, track, "")
                    return
            await self.send(identity, {"type": "error", "error": "standby timed out"})
        finally:
            if self.pending.get(cid) == identity:
                self.pending.pop(cid, None)
                cam = self.get_camera(cid, "")
                if cam.pending_ws == identity:
                    cam.pending_ws = None
                    cam.pending_label = ""
                self.mark_dirty()
            self._watchers.pop(identity, None)

    def _publisher_gone(self, identity: str, why: str) -> None:
        t = self._pumps.pop(identity, None)
        if t:
            t.cancel()
        w = self._watchers.pop(identity, None)
        if w:
            w.cancel()
        for cid, ident in list(self.pending.items()):
            if ident == identity:
                self.pending.pop(cid, None)
                cam = self.get_camera(cid, "")
                if cam.pending_ws == identity:
                    cam.pending_ws = None
                    cam.pending_label = ""
                self.log_event("camera", f"{cid}: waiting publisher {why}")
        for cid, ident in list(self.live_identity.items()):
            if ident == identity:
                self._release(cid, why)
        self.mark_dirty()

    def _release(self, cid: str, why: str) -> None:
        identity = self.live_identity.pop(cid, None)
        token = self.live_token.pop(cid, None)
        self.stats["publishers"].pop(cid, None)
        cam = self.get_camera(cid, "")
        if token is not None and cam.owner is token:   # a newer owner (ws or lk) keeps the camera
            cam.connected = False
            cam.client = ""
            self.log_event("camera", f"{cid} disconnected ({why})")
        elif identity:
            self.log_event("camera", f"{cid}: stale LiveKit publisher closed, newer stream keeps running")

    # ------------------------------------------------------------ frames and control
    async def _pump(self, cam: Any, cid: str, identity: str, token: object, track: Any) -> None:
        from livekit import rtc

        loop = asyncio.get_running_loop()
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.BGRA)
        min_dt = 1.0 / max(1.0, self.fps)
        last = 0.0
        try:
            async for ev in stream:
                if cam.owner is not token:
                    break
                now = time.monotonic()
                if now - last < min_dt:
                    continue
                last = now
                frame = getattr(ev, "frame", ev)
                jpeg = await loop.run_in_executor(None, encode_bgra, frame.width, frame.height, frame.data, self.jpeg_quality, self.max_width)
                if jpeg and cam.owner is token:
                    self.stats["frames"] += 1
                    self.on_frame(cid, jpeg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("LiveKit pump %s (%s) ended: %s", cid, identity, e)
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    async def _reconcile(self) -> bool:
        """Cross-check our publisher slots with the LiveKit server. Releases publishers the server no
        longer lists; returns True when the server has been showing a publisher we never received for
        two checks in a row (a zombie session), so the caller reconnects."""
        try:
            from livekit import api

            if self._api is None:
                self._api = api.LiveKitAPI(self.url.replace("wss://", "https://").replace("ws://", "http://"), self.api_key, self.api_secret)
            res = await asyncio.wait_for(self._api.room.list_participants(api.ListParticipantsRequest(room=self.room_name)), timeout=6.0)
        except Exception as e:
            log.debug("LiveKit reconcile skipped: %s", e)
            return False
        self.stats["reconciles"] += 1
        publishing = {p.identity for p in res.participants if p.tracks and not p.identity.startswith("director")}
        present = {p.identity for p in res.participants}
        for cid, ident in list(self.live_identity.items()):
            if ident not in present:
                self._publisher_gone(ident, "gone (server reconcile)")
        for cid, ident in list(self.pending.items()):
            if ident not in present:
                self._publisher_gone(ident, "gone (server reconcile)")
        known = set(self.live_identity.values()) | set(self.pending.values()) | set(self._pumps)
        zombie = False
        for ident in publishing:
            if ident in known:
                self._missed.pop(ident, None)
                continue
            self._missed[ident] = self._missed.get(ident, 0) + 1
            if self._missed[ident] >= 2:
                zombie = True
        for ident in list(self._missed):
            if ident not in publishing:
                self._missed.pop(ident, None)
        if zombie:
            self._missed.clear()
        return zombie

    async def send_ptz(self, cid: str, target: dict) -> bool:
        identity = self.live_identity.get(cid.upper())
        if not identity:
            return False
        return await self.send(identity, {"type": "ptz", "cam": cid.upper(), **target})

    async def _send_data(self, identity: str, msg: dict) -> bool:
        if self.room is None or not self.stats.get("connected"):
            return False
        try:
            await self.room.local_participant.publish_data(json.dumps(msg), reliable=True, destination_identities=[identity], topic=CONTROL_TOPIC)
            return True
        except Exception as e:
            log.debug("LiveKit data send to %s failed: %s", identity, e)
            return False
