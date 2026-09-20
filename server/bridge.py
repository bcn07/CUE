"""Bridge to the team's control plane (apps/api in the CUE repo).

The one-shot stays the brain: speech -> meaning -> deterministic director. Every program
change it makes is mirrored to the team backend as a producer take
(`POST /api/v1/events/{event}/take`), and HOLD / AUTO are mirrored to `/mode`. The team's
React GUI then executes the render command on its LiveKit video. Nothing here blocks the
director: calls are fire-and-forget tasks with a short timeout and a single retry when the
team's mode revision moved under us.

Camera ids differ: the one-shot uses A/B/C, the team uses CAM-WIDE / CAM-GUEST / CAM-HOST.
`CUE_TEAM_CAMERA_MAP=A=CAM-WIDE,B=CAM-GUEST,C=CAM-HOST` maps one to the other.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

import httpx

log = logging.getLogger("cue.bridge")

TEAM_CAMERAS = ("CAM-HOST", "CAM-GUEST", "CAM-WIDE")
_RE_REASON = r"^[A-Za-z0-9._:-]+$"


def parse_camera_map(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip().upper(), v.strip().upper()
            if k and v in TEAM_CAMERAS:
                out[k] = v
    return out


class TeamControlBridge:
    def __init__(self, api_base: str, event_id: str, producer_secret: str, camera_map: dict[str, str],
                 timeout_s: float = 2.5, resume_mode: str = "ASSIST"):
        # How the team compositor treats our takes (apps/web/src/compositor/controlAdapter.ts): reason codes
        # CUE_* are POLICY origin. In ASSIST they are shown as suggestions and the operator presses TAKE;
        # in AUTO they execute on air (2.5 s minimum shot). We never spoof MANUAL: automation must not look
        # like an operator in the team's timeline.
        self.resume_mode = "AUTO" if str(resume_mode).upper() == "AUTO" else "ASSIST"
        self.api_base = api_base.rstrip("/")
        self.event_id = event_id
        self.camera_map = camera_map
        self._client = httpx.AsyncClient(base_url=self.api_base, timeout=timeout_s,
                                         headers={"X-Cue-Producer-Secret": producer_secret})
        self._lock = asyncio.Lock()
        self.stats = {"takes": 0, "modes": 0, "errors": 0, "skipped": 0, "last_error": "", "last_ok_wall": None,
                      "team_mode": None, "team_live": None, "team_revision": None, "reachable": None}

    @property
    def _ev(self) -> str:
        return f"/api/v1/events/{self.event_id}"

    # ----------------------------------------------------------------- reads
    async def state(self) -> dict:
        r = await self._client.get(f"{self._ev}/control-state")
        r.raise_for_status()
        s = r.json()
        self.stats.update(team_mode=s.get("mode"), team_live=s.get("liveCameraId", s.get("live_camera_id")),
                          team_revision=s.get("modeRevision", s.get("mode_revision")), reachable=True)
        return s

    async def bindings(self) -> dict[str, int]:
        r = await self._client.get(f"{self._ev}/bindings")
        r.raise_for_status()
        out = {}
        for b in r.json():
            cid = b.get("cameraId", b.get("camera_id"))
            ep = b.get("streamEpoch", b.get("stream_epoch"))
            if cid and ep:
                out[cid] = int(ep)
        return out

    async def check(self) -> dict:
        try:
            s = await self.state()
            self.stats["last_ok_wall"] = time.time()
            return s
        except Exception as e:
            self.stats["reachable"] = False
            self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
            return {}

    # ---------------------------------------------------------------- writes
    @staticmethod
    def _revision(s: dict) -> int:
        return int(s.get("modeRevision", s.get("mode_revision", 0)) or 0)

    async def mirror_take(self, oneshot_camera: str, evidence: str, reason: str) -> dict | None:
        team_cam = self.camera_map.get(oneshot_camera.upper())
        if team_cam is None:
            self.stats["skipped"] += 1
            return None
        reason_code = ("CUE_" + (evidence or "take").upper().replace(" ", "_"))[:64]
        import re
        if not re.match(_RE_REASON, reason_code):
            reason_code = "CUE_TAKE"
        async with self._lock:
            for attempt in range(2):
                try:
                    s = await self.state()
                    if s.get("mode") == "ENDED":
                        self.stats["skipped"] += 1
                        return None
                    epochs = await self.bindings()
                    epoch = epochs.get(team_cam) or s.get("liveStreamEpoch") or s.get("live_stream_epoch") or 1
                    body = {"cameraId": team_cam, "streamEpoch": int(epoch), "expectedRevision": self._revision(s),
                            "idempotencyKey": f"oneshot-{uuid.uuid4().hex}", "reasonCode": reason_code}
                    r = await self._client.post(f"{self._ev}/take", json=body)
                    if r.status_code == 409 and attempt == 0:
                        continue  # revision moved (someone else took): re-read and retry once
                    r.raise_for_status()
                    self.stats["takes"] += 1
                    self.stats["last_ok_wall"] = time.time()
                    self.stats["reachable"] = True
                    res = r.json()
                    st = res.get("state", {})
                    self.stats.update(team_mode=st.get("mode"), team_live=st.get("liveCameraId"), team_revision=st.get("modeRevision"))
                    return res
                except Exception as e:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
                    if attempt == 1:
                        log.warning("team take failed: %s", self.stats["last_error"])
                        return None
        return None

    async def mirror_mode(self, hold: bool) -> dict | None:
        """One-shot HOLD -> team MANUAL_HOLD. One-shot AUTO -> team `resume_mode`:
        ASSIST (default) = our decisions appear as suggestions for the operator; AUTO = they cut on air."""
        mode = "MANUAL_HOLD" if hold else self.resume_mode
        async with self._lock:
            for attempt in range(2):
                try:
                    s = await self.state()
                    if s.get("mode") == mode:
                        return s
                    body = {"mode": mode, "expectedRevision": self._revision(s), "idempotencyKey": f"oneshot-{uuid.uuid4().hex}"}
                    r = await self._client.post(f"{self._ev}/mode", json=body)
                    if r.status_code == 409 and attempt == 0:
                        continue
                    r.raise_for_status()
                    self.stats["modes"] += 1
                    self.stats["last_ok_wall"] = time.time()
                    st = r.json().get("state", {})
                    self.stats.update(team_mode=st.get("mode"), team_revision=st.get("modeRevision"))
                    return r.json()
                except Exception as e:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
                    if attempt == 1:
                        log.warning("team mode change failed: %s", self.stats["last_error"])
                        return None
        return None

    async def close(self) -> None:
        await self._client.aclose()
