"""CUE one-shot director server.

Phones/laptops push JPEG frames over WebSocket (/ingest). The MacBook mic is
transcribed by Deepgram (speech.py). Each finished clause becomes a meaning
(semantics.py: rules fast path + LLM). A deterministic director (director.py)
decides SHOW / WIDE / HOST / HOLD using fresh face identity (identity.py) per
camera. The browser UI (/) shows the program output, multiview, and the judge
panel. Correctly deciding NOT to cut is a feature.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
import socket
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import director as D
from .assembler import Clause, Interim, UtteranceEnd
from .config import Settings, load_settings
from .identity import (FaceEngine, Gallery, IdentityWorker, PresenceTracker, enroll_people, ensure_decodable,
                       models_present, reference_thumbnail)
from .bridge import TeamControlBridge, parse_camera_map
from .recorder import Recorder, list_recordings
from .semantics import Action, Cue, LLMParser, Person, Roster, Semantics

APP_VERSION = "0.2.0"
log = logging.getLogger("cue.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_CAMERAS = {
    "A": {"label": "Camera A (wide)", "role": "wide", "fixed_person": None},
    "B": {"label": "Camera B", "role": "host", "fixed_person": None},
    "C": {"label": "Camera C", "role": "guest", "fixed_person": None},
}


_LAN_IP: dict[str, str] = {}


def lan_ip() -> str:
    if "ip" in _LAN_IP:
        return _LAN_IP["ip"]
    _LAN_IP["ip"] = _lan_ip_uncached()
    return _LAN_IP["ip"]


def _lan_ip_uncached() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no packet is sent for UDP connect
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def slug(s: str, fallback: str | None = "person") -> str | None:
    """ASCII slug; returns `fallback` (None -> None) when nothing usable is left, so '..' or
    '李' never silently become a real id."""
    s = re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")
    return s or fallback


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((p / 100) * (len(ys) - 1)))))
    return ys[k]


def stat(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "last": None, "p50": None, "p95": None}
    return {"n": len(xs), "last": round(xs[-1], 1), "p50": round(statistics.median(xs), 1), "p95": round(pct(xs, 95) or 0, 1)}


# ------------------------------------------------------------------ cameras
@dataclass
class Camera:
    id: str
    label: str
    role: str = "guest"
    fixed_person: str | None = None
    connected: bool = False
    seq: int = 0
    latest: bytes | None = None
    last_frame_mono: float = 0.0
    frames: int = 0
    bytes: int = 0
    fps: float = 0.0
    kbps: float = 0.0
    _win_t: float = 0.0
    _win_frames: int = 0
    _win_bytes: int = 0
    client: str = ""
    owner: object | None = None   # token of the connection that currently streams this camera
    owner_ws: Any = None          # its WebSocket, closed when a newer connection takes over

    def push(self, jpeg: bytes, now: float) -> None:
        self.seq += 1
        self.latest = jpeg
        self.last_frame_mono = now
        self.frames += 1
        self.bytes += len(jpeg)
        self._win_frames += 1
        self._win_bytes += len(jpeg)
        if self._win_t == 0.0:
            self._win_t = now
        elif now - self._win_t >= 1.0:
            dt = now - self._win_t
            self.fps = self._win_frames / dt
            self.kbps = self._win_bytes * 8 / 1000 / dt
            self._win_t, self._win_frames, self._win_bytes = now, 0, 0

    def healthy(self, now: float, stale_s: float) -> bool:
        return self.connected and self.latest is not None and (now - self.last_frame_mono) < stale_s


class UIClient:
    """One browser. Frames and state are coalesced per client and sent by its own task, so a
    stalled phone on bad Wi-Fi only stalls itself, never the other viewers."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.frames: dict[str, bytes] = {}   # latest frame per camera, coalesced
        self.state: str | None = None        # latest state JSON, coalesced
        self.wake = asyncio.Event()
        self.alive = True

    def offer(self, cam_id: str, payload: bytes) -> None:
        self.frames[cam_id] = payload
        self.wake.set()

    def offer_state(self, payload: str) -> None:
        self.state = payload
        self.wake.set()

    async def sender(self) -> None:
        try:
            while self.alive:
                await self.wake.wait()
                self.wake.clear()
                state, self.state = self.state, None
                if state is not None:
                    await self.ws.send_text(state)
                items = list(self.frames.items())
                self.frames.clear()
                for _, payload in items:
                    await self.ws.send_bytes(payload)
        except Exception:
            self.alive = False


# ---------------------------------------------------------------------- show
class Show:
    """All mutable runtime state, owned by the asyncio loop thread."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.loop: asyncio.AbstractEventLoop | None = None
        self.cameras: dict[str, Camera] = {}
        self.trackers: dict[str, PresenceTracker] = {}
        self.ui_clients: set[UIClient] = set()
        self.state = D.State()
        self.tun = D.Tunables(min_shot_s=settings.min_shot_s, cue_lifetime_s=settings.cue_lifetime_s,
                              identity_max_age_s=settings.identity_max_age_s)
        self.roster = Roster([])
        self.script_text = ""
        self.enroll_report: dict[str, dict] = {}
        self.engine: FaceEngine | None = None
        self.gallery = Gallery(accept=settings.identity_accept, margin=settings.identity_margin)
        self.worker: IdentityWorker | None = None
        self.semantics: Semantics | None = None
        self.vlm = None
        self.speech = None
        self.mic = None
        self.speech_task: asyncio.Task | None = None
        self.speech_status: dict = {"connected": False, "epoch": 0, "error": ""}
        self.transcript: list[dict] = []
        self.interim: dict | None = None
        self.last_cue: Cue | None = None
        self.last_decision: D.Decision | None = None
        self.pending_cue: Cue | None = None
        self.program_since_wall: float = time.time()
        self.events: list[dict] = []
        self.lat_interpret: list[float] = []
        self.lat_llm: list[float] = []
        self.lat_clause_to_cut: list[float] = []
        self.lat_speech_to_cut: list[float] = []
        self.lat_decide: list[float] = []
        self.cuts = 0
        self.holds = 0
        self.decision_ver = 0
        self.bridge: TeamControlBridge | None = None
        self.recorder: Recorder | None = None
        self.last_recording: dict | None = None
        self.recordings_count = 0
        self.pending_block: str | None = None   # why the pending cue waits: "min-shot" | "identity"
        self.latest_seq = 0                     # newest clause seen by the director
        self._loop_thread: threading.Thread | None = None
        self.dirty = asyncio.Event()
        self._recheck_handle: asyncio.TimerHandle | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------- persistence
    @property
    def data(self) -> Path:
        return self.s.data_dir

    def load_config(self) -> None:
        rp = self.data / "roster.json"
        people: list[Person] = []
        if rp.exists():
            try:
                for p in json.loads(rp.read_text()).get("people", []):
                    people.append(Person(p["id"], p.get("name", p["id"]), list(p.get("aliases", [])), p.get("role", ""), bool(p.get("is_host"))))
            except Exception as e:
                log.warning("roster.json unreadable: %s", e)
        self.roster = Roster(people)
        cp = self.data / "cameras.json"
        cams = DEFAULT_CAMERAS
        if cp.exists():
            try:
                cams = json.loads(cp.read_text()).get("cameras", DEFAULT_CAMERAS)
            except Exception as e:
                log.warning("cameras.json unreadable: %s", e)
        for cid, c in cams.items():
            cam = self.cameras.get(cid) or Camera(cid, c.get("label") or f"Camera {cid}")
            cam.label = c.get("label") or cam.label
            cam.role = c.get("role") or "guest"
            cam.fixed_person = c.get("fixed_person") or None
            self.cameras[cid] = cam
        sp = self.data / "script.md"
        self.script_text = sp.read_text() if sp.exists() else ""

    def save_roster(self) -> None:
        (self.data / "roster.json").write_text(json.dumps({"people": [
            {"id": p.id, "name": p.name, "aliases": p.aliases, "role": p.role, "is_host": p.is_host} for p in self.roster.people
        ]}, indent=2))

    def save_cameras(self) -> None:
        (self.data / "cameras.json").write_text(json.dumps({"cameras": {
            cid: {"label": c.label, "role": c.role, "fixed_person": c.fixed_person} for cid, c in self.cameras.items()
        }}, indent=2))

    # ---------------------------------------------------------------- setup
    def init_identity(self) -> None:
        if not models_present(self.s.models_dir):
            log.error("face models missing in %s (run ./run_server.sh to download). Identity disabled.", self.s.models_dir)
            return
        try:
            self.engine = FaceEngine(self.s.models_dir)
        except Exception as e:  # a truncated download or an incompatible OpenCV: run without identity
            log.error("face models failed to load (%s). Delete models/*.onnx and rerun ./run_server.sh. Identity disabled.", e)
            self.engine = None
            return
        self.reenroll()
        self.worker = IdentityWorker(
            self.engine, self.gallery, self.trackers, self.frames_for_identity,
            interval_s=self.s.ident_interval_s, on_update=self._identity_update_from_thread,
            tracker_factory=lambda: PresenceTracker(self.s.identity_confirmations, 1.5),
        )
        self.worker.start()

    def reenroll(self) -> dict:
        if self.engine is None:
            return {}
        embeddings, report = enroll_people(self.engine, self.data / "people")
        # Only roster people count; a folder without a roster entry is ignored.
        embeddings = {k: v for k, v in embeddings.items() if k in self.roster.ids}
        self.gallery.replace(embeddings)
        self.enroll_report = report
        log.info("enrolled %d embeddings for %d people", self.gallery.size, len(embeddings))
        if self.vlm is not None:
            refs = []
            for p in self.roster.people:
                pdir = self.data / "people" / p.id
                if pdir.exists():
                    for f in sorted(pdir.iterdir()):
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                            th = reference_thumbnail(f)
                            if th:
                                refs.append({"id": p.id, "name": p.name, "jpeg": th})
                                break
            self.vlm.set_references(refs)
        return report

    def init_semantics(self) -> None:
        llm = None
        if self.s.has_llm:
            try:
                llm = LLMParser(self.s.llm_model, self.roster, self.script_text, timeout_s=self.s.llm_timeout_s,
                                base_url=self.s.llm_base_url, api_key=self.s.llm_api_key)
                log.info("LLM interpreter: %s via %s", self.s.llm_model, self.s.llm_base_url or "OpenAI")
            except Exception as e:
                log.error("LLM init failed: %s", e)
        else:
            log.warning("No OPENAI_API_KEY and no CUE_LLM_BASE_URL: rule-based interpreter only (no LLM, no VLM).")
        self.semantics = Semantics(self.roster, self.script_text, self.on_cue, llm=llm, fast_path=self.s.fast_path,
                                   stale_after_s=self.s.cue_lifetime_s)
        if self.s.has_openai and self.s.vlm_enabled:  # the vision fallback needs OpenAI; local text models run without it
            try:
                from .vlm import VLMIdentifier
                self.vlm = VLMIdentifier(self.s.vlm_model, self.s.vlm_min_interval_s, self.s.vlm_timeout_s)
            except Exception as e:
                log.error("VLM init failed: %s", e)

    def roster_keyterms(self) -> list[str]:
        out: list[str] = []
        for p in self.roster.people:
            out.append(p.name)
            out.extend(p.aliases)
        return out

    def refresh_context(self) -> None:
        if self.semantics:
            self.semantics.set_context(self.roster, self.script_text)
        self.reenroll()
        if self.speech is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.speech.reconnect)  # new roster -> new Deepgram key terms
        self.mark_dirty()

    def ensure_mic(self) -> bool:
        """Start the master mic once; Deepgram and the recorder both read from it."""
        if self.mic is not None:
            return True
        if self.loop is None:
            return False
        from .speech import MicSource
        try:
            mic = MicSource(self.loop, self.s.sample_rate, self.s.mic_device)
            mic.feed_queue = self.s.has_deepgram
            mic.start()
            self.mic = mic
            return True
        except Exception as e:
            self.speech_status = {**self.speech_status, "error": f"mic: {e}"[:200]}
            log.error("mic start failed: %s (grant Microphone permission to your terminal, or set CUE_MIC_DEVICE)", e)
            return False

    async def start_speech(self) -> None:
        if not self.s.has_deepgram:
            log.warning("DEEPGRAM_API_KEY missing: live mic transcription disabled. Use the typed input on the UI to test.")
            return
        from .speech import DeepgramStream
        if not self.ensure_mic():
            return
        self.speech = DeepgramStream(self.s.deepgram_api_key, self.s.sample_rate, self.on_speech_event,
                                     self._speech_status, model=self.s.deepgram_model, keyterms=self.roster_keyterms)
        self.speech_task = asyncio.create_task(self.speech.run(self.mic.queue))

    def _speech_status(self, st: dict) -> None:
        self.speech_status = {**self.speech_status, **st}
        self.mark_dirty()

    # ------------------------------------------------------------- identity
    def frames_for_identity(self) -> dict[str, tuple[int, bytes]]:
        with self._lock:
            return {cid: (c.seq, c.latest) for cid, c in self.cameras.items() if c.latest is not None}

    def _identity_update_from_thread(self, cam_id: str, obs, img) -> None:
        if self.loop is None:
            return
        unknown = any(o.decision in ("UNKNOWN", "AMBIGUOUS", "TOO_SMALL") for o in obs)
        self.loop.call_soon_threadsafe(self._identity_updated, cam_id, unknown)

    def _identity_updated(self, cam_id: str, has_unknown: bool) -> None:
        self.mark_dirty()
        if self.pending_cue is not None and self.pending_block == "identity":
            self.reevaluate("identity update")
        if has_unknown and self.vlm is not None and self.vlm.can_call(cam_id):
            cam = self.cameras.get(cam_id)
            if cam and cam.latest:
                asyncio.ensure_future(self._vlm_check(cam_id, cam.latest))

    async def _vlm_check(self, cam_id: str, jpeg: bytes) -> None:
        out = await self.vlm.identify(cam_id, jpeg)
        if out is None:
            return
        tracker = self.trackers.get(cam_id)
        if tracker is None:
            tracker = self.trackers[cam_id] = PresenceTracker(self.s.identity_confirmations, 1.5)
        if out.visible_ids and out.confidence >= 0.6:
            tracker.observe_ids(out.visible_ids, time.monotonic(), "vlm", out.confidence)
            self.log_event("vlm", f"{cam_id}: {', '.join(out.visible_ids)} (conf {out.confidence:.2f}, {self.vlm.stats['last_ms']} ms)")
            if self.pending_cue is not None:
                self.reevaluate("vlm evidence")
        self.mark_dirty()

    def camera_views(self, now: float) -> dict[str, D.CameraView]:
        views = {}
        for cid, c in self.cameras.items():
            tr = self.trackers.get(cid)
            confirmed = tr.confirmed(now, self.tun.identity_max_age_s) if tr else {}
            views[cid] = D.CameraView(cid, c.role, c.healthy(now, self.s.camera_stale_s), confirmed, c.fixed_person)
        return views

    # --------------------------------------------------------------- speech
    async def on_speech_event(self, ev, meta: dict) -> None:
        if isinstance(ev, Interim):
            self.interim = {"utterance_id": ev.utterance_id, "text": ev.text}
            self.mark_dirty()
        elif isinstance(ev, Clause):
            self.interim = None
            self._upsert_transcript(ev.utterance_id, ev.utterance_text, final=ev.is_utterance_end, source=meta.get("source", "mic"))
            self.log_event("clause", ev.text, utterance_id=ev.utterance_id)
            await self.semantics.handle_clause(ev.text, ev.utterance_id, ev.utterance_text, meta.get("recv_mono", time.monotonic()),
                                               {"speech_end_mono": meta.get("speech_end_mono"), "source": meta.get("source", "mic")})
        elif isinstance(ev, UtteranceEnd):
            self._upsert_transcript(ev.utterance_id, ev.text, final=True, source=meta.get("source", "mic"))
            self.interim = None
            self.mark_dirty()

    def _upsert_transcript(self, uid: str, text: str, final: bool, source: str = "mic") -> None:
        for t in self.transcript:
            if t["utterance_id"] == uid:
                t["text"] = text or t["text"]
                t["final"] = t["final"] or final
                source = t["source"]  # the first label wins (typed stays typed)
                break
        else:
            self.transcript.append({"utterance_id": uid, "text": text, "final": final, "wall": time.time(), "source": source})
            self.transcript = self.transcript[-14:]
        if self.recorder is not None:
            self.recorder.log_transcript(uid, text, final, source)
        self.mark_dirty()

    async def say(self, text: str, source: str = "typed") -> None:
        """Typed test input. Same pipeline as the mic; labelled so nobody mistakes it for live speech."""
        text = text.strip()
        if not text or self.semantics is None:
            return
        uid = f"typed-{int(time.time() * 1000) % 100000000}"
        parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()] or [text]
        so_far = []
        for i, part in enumerate(parts):
            so_far.append(part)
            now = time.monotonic()
            ev = Clause(uid, i + 1, part, " ".join(so_far), None, None, i == len(parts) - 1)
            await self.on_speech_event(ev, {"recv_mono": now, "recv_wall": time.time(), "source": source, "speech_end_mono": now})
        await self.on_speech_event(UtteranceEnd(uid, text), {"source": source})

    # ------------------------------------------------------------- directing
    async def on_cue(self, cue: Cue) -> None:
        self.last_cue = cue
        self.lat_interpret.append(cue.latency_ms)
        self.lat_interpret = self.lat_interpret[-200:]
        if cue.source == "llm":
            self.lat_llm.append(cue.latency_ms)
            self.lat_llm = self.lat_llm[-200:]
        self.log_event("cue", f"{cue.action.value} {cue.target_ids} {cue.temporal_intent.value} via {cue.source} ({cue.latency_ms:.0f} ms): '{cue.evidence_text}'",
                       utterance_id=cue.utterance_id)
        seq = int((cue.meta or {}).get("seq", 0) or 0)
        p = self.pending_cue
        p_seq = int((p.meta or {}).get("seq", 0) or 0) if p is not None else 0
        actionable = cue.temporal_intent.value == "NOW" and cue.action in (Action.SHOW, Action.HOST, Action.WIDE)
        if cue.intent.value == "CANCEL":
            if p is not None and seq >= p_seq:
                self.pending_cue = None
        elif p is not None and not actionable:
            same_clause = seq == p_seq and cue.clause_text == p.clause_text
            # An LLM verdict on the very clause the fast path acted on: HOLD means the rules were wrong.
            # An LLM *error* is no information and never drops a deterministic cue.
            llm_disagrees = same_clause and cue.source == "llm" and p.source == "rules-fast"
            # A newer clause (higher seq) supersedes: a new utterance, or a deferral/negation.
            newer = seq > p_seq
            contradicts = cue.temporal_intent.value in ("FUTURE", "NEGATED", "PAST")
            if llm_disagrees or (newer and (cue.utterance_id != p.utterance_id or contradicts)):
                self.pending_cue = None
                self.log_event("decision", f"pending {p.action.value} {p.target_ids} dropped: superseded by '{cue.clause_text}' ({cue.source} {cue.action.value}/{cue.temporal_intent.value})")
        if seq and seq < self.latest_seq and cue.source == "llm" and not actionable:
            return  # a late verdict for an older clause carries nothing new for the director
        self.latest_seq = max(self.latest_seq, seq)
        self.run_director(cue, "cue")

    def run_director(self, cue: Cue | None, why: str, quiet: bool = False) -> D.Decision | None:
        now = time.monotonic()
        t0 = time.perf_counter()
        views = self.camera_views(now)
        decision = D.decide(cue, views, self.state, now, host_id=self.roster.host_id, tun=self.tun)
        self.lat_decide.append((time.perf_counter() - t0) * 1000)
        self.lat_decide = self.lat_decide[-200:]
        prev = self.last_decision
        self.last_decision = decision
        self.decision_ver += 1
        changed = D.apply(self.state, decision, cue, now)
        same_as_before = prev is not None and prev.action == decision.action and prev.camera_id == decision.camera_id and prev.reason == decision.reason
        if decision.action == D.Take.TAKE and changed:
            self.cuts += 1
            self.program_since_wall = time.time()
            if cue is not None:
                self.lat_clause_to_cut.append((now - cue.created_at) * 1000)
                self.lat_clause_to_cut = self.lat_clause_to_cut[-200:]
                se = (cue.meta or {}).get("speech_end_mono")
                if se:
                    self.lat_speech_to_cut.append(max(0.0, (now - se) * 1000))
                    self.lat_speech_to_cut = self.lat_speech_to_cut[-200:]
            self.log_event("cut", f"TAKE {decision.camera_id} [{decision.evidence}] {decision.reason}")
            if self.recorder is not None:
                self.recorder.log_cut(decision.camera_id, decision.evidence, decision.reason, cue.to_json() if cue else None,
                                      self.lat_clause_to_cut[-1] if cue is not None and self.lat_clause_to_cut else None)
            self._mirror_take(decision.camera_id, decision.evidence, decision.reason)
            self.pending_cue = None
            self.pending_block = None
        else:
            if decision.action == D.Take.SLATE and changed:
                self.program_since_wall = time.time()
                self.log_event("cut", f"SLATE [{decision.evidence}] {decision.reason}")
                if self.recorder is not None:
                    self.recorder.log_cut(None, "slate", decision.reason, cue.to_json() if cue else None, None)
            if cue is not None and decision.action != D.Take.TAKE and not quiet:
                self.holds += 1
            if not (quiet and same_as_before):
                self.log_event("decision", f"{decision.action.value} {decision.camera_id or '-'}: {decision.reason}" + (f" ({why})" if quiet else ""))
            actionable = cue is not None and cue.temporal_intent.value == "NOW" and cue.action in (Action.SHOW, Action.HOST, Action.WIDE)
            if actionable:
                wanted = decision.target or (cue.target_ids[0] if cue.target_ids else None)
                live_ok = decision.evidence in ("identity", "static", "host-role") or (cue.action == Action.WIDE and decision.evidence == "wide")
                if live_ok and "min shot" not in decision.reason:
                    # The host's latest wish is already on screen: nothing older may fire later.
                    if self.pending_cue is not None and not quiet:
                        self.log_event("decision", f"pending {self.pending_cue.action.value} {self.pending_cue.target_ids} dropped: '{cue.clause_text}' is already satisfied")
                    self.pending_cue = None
                    self.pending_block = None
                else:
                    self.pending_cue = cue
                    self.pending_block = "min-shot" if "min shot" in decision.reason else "identity"
                    self._schedule_recheck(now, decision.reason)
                    if wanted and self.vlm is not None and self.pending_block == "identity":
                        for cid, cam in self.cameras.items():
                            if cam.latest and self.vlm.can_call(cid):
                                asyncio.ensure_future(self._vlm_check(cid, cam.latest))
        self.mark_dirty()
        return decision

    def _schedule_recheck(self, now: float, reason: str = "") -> None:
        if self.loop is None or self.pending_cue is None:
            return
        if self._recheck_handle:
            self._recheck_handle.cancel()
        remaining_life = self.tun.cue_lifetime_s - (now - self.pending_cue.created_at)
        if "min shot" in reason:
            delay = self.tun.min_shot_s - (now - self.state.last_cut_time) + 0.02
            why = "min-shot elapsed"
        else:  # waiting for identity evidence: identity updates re-run the director; this timer only expires the cue
            delay = remaining_life + 0.02
            why = "cue lifetime check"
        delay = max(0.05, min(delay, remaining_life + 0.02))
        self._recheck_handle = self.loop.call_later(delay, self.reevaluate, why)

    def reevaluate(self, why: str) -> None:
        cue = self.pending_cue
        if cue is None:
            return
        now = time.monotonic()
        if now - cue.created_at > self.tun.cue_lifetime_s:
            self.pending_cue = None
            self.pending_block = None
            self.log_event("decision", f"pending cue expired ({why})")
            self.mark_dirty()
            return
        self.run_director(cue, why, quiet=True)

    def control(self, action: str, camera_id: str | None = None) -> dict:
        now = time.monotonic()
        action = (action or "").lower()
        if action == "hold":
            self.state.mode = D.Mode.HOLD
            self.pending_cue = None
            self.log_event("manual", "HOLD: automatic cuts paused")
            self._mirror_mode(True)
        elif action == "auto":
            self.state.mode = D.Mode.AUTO
            self.state.hold_until = None
            self.log_event("manual", "AUTO resumed")
            self._mirror_mode(False)
        elif action in ("take", "wide", "host"):
            if action == "wide":
                camera_id = next((cid for cid, c in self.cameras.items() if c.role == "wide"), camera_id)
            elif action == "host":
                camera_id = next((cid for cid, c in self.cameras.items() if c.role == "host"), camera_id)
            if not camera_id or camera_id not in self.cameras:
                raise HTTPException(400, "unknown camera")
            if camera_id == self.state.current_camera:
                self.pending_cue = None
                self.pending_block = None
                return {"ok": True, "program": camera_id, "mode": self.state.mode.value, "note": "already live"}
            D.manual_take(self.state, camera_id, now)
            self.pending_cue = None
            self.program_since_wall = time.time()
            self.last_decision = D.Decision(D.Take.TAKE, camera_id, "manual take", "manual")
            self.cuts += 1
            self.log_event("manual", f"TAKE {camera_id}")
            if self.recorder is not None:
                self.recorder.log_cut(camera_id, "manual", "manual take", None, None)
            self._mirror_take(camera_id, "manual", "manual take")
        elif action == "record_start":
            summary = self.start_recording()
            if summary.get("status") == "error":
                raise HTTPException(500, summary.get("error", "recording failed to start"))
        elif action == "record_stop":
            asyncio.ensure_future(self.stop_recording())
        else:
            raise HTTPException(400, "unknown action")
        self.mark_dirty()
        return {"ok": True, "program": self.state.current_camera, "mode": self.state.mode.value}

    # ---------------------------------------------------------- team bridge
    def _mirror_take(self, camera_id: str, evidence: str, reason: str) -> None:
        if self.bridge is None or self.loop is None:
            return

        async def go() -> None:
            res = await self.bridge.mirror_take(camera_id, evidence, reason)
            if res is not None:
                cmd = res.get("renderCommand") or {}
                self.log_event("bridge", f"team GUI: take {cmd.get('cameraId')} (decision {cmd.get('decisionSequence')}, epoch {cmd.get('streamEpoch')})")
            else:
                self.log_event("bridge", f"team GUI: take {camera_id} NOT mirrored ({self.bridge.stats['last_error'] or 'unmapped camera'})")
            self.mark_dirty()

        asyncio.ensure_future(go())

    def _mirror_mode(self, hold: bool) -> None:
        if self.bridge is None or self.loop is None:
            return

        async def go() -> None:
            res = await self.bridge.mirror_mode(hold)
            self.log_event("bridge", f"team GUI mode -> {(res or {}).get('state', res or {}).get('mode', 'unchanged') if res else 'FAILED: ' + self.bridge.stats['last_error']}")
            self.mark_dirty()

        asyncio.ensure_future(go())

    # ------------------------------------------------------------------ LLM
    async def llm_warm_loop(self) -> None:
        """Local models (Ollama) unload after a few idle minutes and take seconds to reload;
        warm once at boot and ping when idle so the first clause of the show is never a cold call."""
        llm = self.semantics.llm if self.semantics else None
        if llm is None:
            return
        ms = await llm.warmup()
        if ms is not None:
            self.log_event("llm", f"warm: {self.s.llm_model} answered in {ms:.0f} ms")
            self.lat_llm.append(ms)
        if not self.s.llm_base_url:
            return
        while True:
            await asyncio.sleep(60)
            if time.monotonic() - self.semantics.last_llm_call > 180:
                ms = await llm.warmup()
                if ms is not None:
                    self.semantics.last_llm_call = time.monotonic()

    # ------------------------------------------------------------ recording
    def program_for_recorder(self) -> tuple[str | None, bytes | None, int]:
        cid = self.state.current_camera
        if not cid:
            return None, None, 0
        with self._lock:
            c = self.cameras.get(cid)
            if c is None or c.latest is None:
                return cid, None, 0
            return cid, c.latest, c.seq

    def start_recording(self) -> dict:
        if self.recorder is not None:
            return self.recorder.summary
        base = self.data / "recordings"
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        rec_dir = base / stamp
        n = 1
        while rec_dir.exists():
            n += 1
            rec_dir = base / f"{stamp}-{n}"
        rec = Recorder(rec_dir, self.program_for_recorder, fps=self.s.record_fps, size=self.s.record_size, sample_rate=self.s.sample_rate)
        try:
            rec.start()
        except Exception as e:
            log.error("recording could not start: %s", e)
            self.log_event("record", f"REC failed to start: {e}")
            try:
                rec.stop()
            except Exception:
                pass
            return {"status": "error", "error": str(e)[:200]}
        self.recorder = rec
        audio = False
        if self.s.record_audio and self.ensure_mic():
            self.mic.taps.append(rec.audio)
            audio = True
        self.log_event("record", f"REC started -> {rec_dir.name}" + ("" if audio else " (no mic: video only)"))
        self.mark_dirty()
        return rec.summary

    async def stop_recording(self) -> dict:
        rec = self.recorder
        if rec is None:
            return self.last_recording or {}
        self.recorder = None
        if self.mic is not None and rec.audio in self.mic.taps:
            self.mic.taps.remove(rec.audio)
        summary = await asyncio.to_thread(rec.stop)
        self.last_recording = summary
        self.recordings_count = len(list_recordings(self.data / "recordings"))
        self.log_event("record", f"REC stopped: {summary.get('final')} ({summary.get('duration_s')}s, {summary.get('cuts')} cuts, audio {summary.get('audio_s')}s{'' if summary.get('muxed') else ', not muxed'})")
        self.mark_dirty()
        return summary

    # ------------------------------------------------------------------ UI
    def log_event(self, kind: str, text: str, **extra) -> None:
        self.events.append({"wall": time.time(), "kind": kind, "text": text, **extra})
        self.events = self.events[-80:]
        self.mark_dirty()

    def mark_dirty(self) -> None:
        if self.loop is not None and self._loop_thread is not None and threading.current_thread() is not self._loop_thread:
            self.loop.call_soon_threadsafe(self.dirty.set)
        else:
            self.dirty.set()

    def snapshot(self) -> dict:
        now = time.monotonic()
        cams = {}
        for cid, c in self.cameras.items():
            tr = self.trackers.get(cid)
            cams[cid] = {
                "id": cid, "label": c.label, "role": c.role, "fixed_person": c.fixed_person,
                "connected": c.connected, "healthy": c.healthy(now, self.s.camera_stale_s),
                "fps": round(c.fps, 1), "kbps": round(c.kbps), "frames": c.frames,
                "last_frame_age_s": round(now - c.last_frame_mono, 2) if c.latest else None,
                "identity": tr.snapshot(now, self.tun.identity_max_age_s) if tr else {"faces": [], "present": {}, "frame_size": [0, 0], "age_s": None},
            }
        prog = self.state.current_camera
        return {
            "type": "state", "now_wall": time.time(), "decision_ver": self.decision_ver,
            "program": {"camera_id": prog, "label": self.cameras[prog].label if prog in self.cameras else None,
                        "since_wall": self.program_since_wall,
                        "reason": self.last_decision.reason if self.last_decision else "", "evidence": self.last_decision.evidence if self.last_decision else ""},
            "mode": self.state.mode.value,
            "cameras": cams,
            "roster": [{"id": p.id, "name": p.name, "aliases": p.aliases, "role": p.role, "is_host": p.is_host,
                        "photos": self.enroll_report.get(p.id, {}).get("photos", 0), "faces": self.enroll_report.get(p.id, {}).get("faces", 0),
                        "failed": self.enroll_report.get(p.id, {}).get("failed", [])} for p in self.roster.people],
            "host_id": self.roster.host_id,
            "transcript": self.transcript[-12:], "interim": self.interim,
            "last_cue": self.last_cue.to_json() if self.last_cue else None,
            "last_decision": self.last_decision.to_json() if self.last_decision else None,
            "pending_cue": self.pending_cue.to_json() if self.pending_cue else None,
            "latency": {"interpret_ms": stat(self.lat_interpret), "llm_ms": stat(self.lat_llm), "decide_ms": stat(self.lat_decide),
                        "clause_to_cut_ms": stat(self.lat_clause_to_cut), "speech_to_cut_ms": stat(self.lat_speech_to_cut),
                        "identity_ms": {"last": self.worker.stats["last_ms"], "avg": self.worker.stats["avg_ms"], "frames": self.worker.stats["frames"]} if self.worker else None},
            "counters": {"cuts": self.cuts, "holds": self.holds, **(self.semantics.stats if self.semantics else {})},
            "services": {
                "deepgram": {"configured": self.s.has_deepgram, **self.speech_status, "model": self.s.deepgram_model,
                             **({"bytes_sent": self.speech.bytes_sent, "messages": self.speech.messages, "results": self.speech.results,
                                 "last_msg_age_s": round(now - self.speech.last_msg_mono, 1) if self.speech.last_msg_mono else None,
                                 "sender_alive": self.speech.sender_alive, "queue": self.mic.queue.qsize() if self.mic else None} if self.speech else {}),
                             "mic_level": round(self.mic.level, 3) if self.mic else None, "mic_dropped": self.mic.dropped if self.mic else None},
                "llm": {"configured": self.semantics is not None and self.semantics.llm is not None, "model": self.s.llm_model,
                        "provider": self.s.llm_provider, "fast_path": self.s.fast_path,
                        "calls": self.semantics.stats["llm_calls"] if self.semantics else 0, "errors": self.semantics.stats["llm_errors"] if self.semantics else 0},
                "vlm": {"enabled": self.vlm is not None, "model": self.s.vlm_model if self.vlm else None, **(self.vlm.stats if self.vlm else {})},
                "identity": {"models_present": self.engine is not None, "gallery_size": self.gallery.size,
                             "max_age_s": self.tun.identity_max_age_s, "accept": self.gallery.accept},
            },
            "bridge": ({"configured": True, "api": self.s.team_api, "event_id": self.s.team_event_id, "camera_map": self.bridge.camera_map,
                        "resume_mode": self.bridge.resume_mode, **self.bridge.stats}
                       if self.bridge else {"configured": False}),
            "recording": {
                "active": self.recorder is not None,
                "dir": self.recorder.out_dir.name if self.recorder else None,
                "elapsed_s": round(self.recorder.elapsed(), 1) if self.recorder else 0,
                "frames": self.recorder.frames if self.recorder else 0,
                "cuts": self.recorder.cuts if self.recorder else 0,
                "audio_s": round(self.recorder.audio_bytes / (2 * self.s.sample_rate), 1) if self.recorder else 0,
                "audio_peak": self.recorder.audio_peak if self.recorder else 0,
                "mic_silent": bool(self.recorder and self.recorder.audio_bytes > 5 * 2 * self.s.sample_rate and self.recorder.audio_peak < 0.004),
                "last": (f"{Path(self.last_recording['dir']).name}/{self.last_recording.get('final')}" if self.last_recording and self.last_recording.get("final") else None),
                "count": self.recordings_count,
            },
            "tunables": {"min_shot_s": self.tun.min_shot_s, "cue_lifetime_s": self.tun.cue_lifetime_s, "camera_stale_s": self.s.camera_stale_s},
            "setup": {"lan_ip": lan_ip(), "port": self.s.port, "script_chars": len(self.script_text)},
            "events": self.events[-40:],
        }

    async def ui_broadcaster(self) -> None:
        period = 1.0 / max(0.5, self.s.ui_state_hz)
        while True:
            try:
                await asyncio.wait_for(self.dirty.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass
            self.dirty.clear()
            if not self.ui_clients:
                await asyncio.sleep(period)
                continue
            payload = json.dumps(self.snapshot())
            for c in list(self.ui_clients):
                if c.alive:
                    c.offer_state(payload)   # never await a network send here: a stalled viewer only stalls itself
                else:
                    self.ui_clients.discard(c)
            await asyncio.sleep(period)

    def relay_frame(self, cam_id: str, jpeg: bytes) -> None:
        if not self.ui_clients:
            return
        cid = cam_id.encode()
        payload = bytes([len(cid)]) + cid + jpeg
        for c in list(self.ui_clients):
            if c.alive:
                c.offer(cam_id, payload)
            else:
                self.ui_clients.discard(c)

    async def health_watch(self) -> None:
        """Failover when the live camera dies (safety still works in HOLD)."""
        last_health: dict[str, bool] = {}
        while True:
            await asyncio.sleep(0.25)
            try:
                self._health_tick(last_health)
            except Exception as e:  # a recorder IO error or a bad camera must never stop failover
                log.warning("health_watch tick failed: %s", e)

    def _health_tick(self, last_health: dict[str, bool]) -> None:
        if True:
            now = time.monotonic()
            changed = False
            for cid, c in self.cameras.items():
                h = c.healthy(now, self.s.camera_stale_s)
                if last_health.get(cid) is not None and last_health[cid] != h:
                    self.log_event("camera", f"{cid} {'healthy' if h else 'UNHEALTHY (no frames)'}")
                    changed = True
                last_health[cid] = h
            live = self.state.current_camera
            if live and not self.cameras.get(live, Camera(live, live)).healthy(now, self.s.camera_stale_s):
                self.run_director(None, "live camera unhealthy")
            elif live is None and any(c.healthy(now, self.s.camera_stale_s) for c in self.cameras.values()) and self.state.mode == D.Mode.AUTO:
                self.run_director(None, "first healthy camera")
            if changed:
                self.mark_dirty()


# --------------------------------------------------------------------- app
settings = load_settings()
show = Show(settings)
app = FastAPI(title="CUE one-shot director")
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
app.mount("/people-photos", StaticFiles(directory=str(settings.data_dir / "people")), name="people")
app.mount("/recordings", StaticFiles(directory=str(settings.data_dir / "recordings")), name="recordings")


@app.on_event("startup")
async def _startup() -> None:
    show.loop = asyncio.get_running_loop()
    show._loop_thread = threading.current_thread()
    show.load_config()
    show.init_semantics()
    await asyncio.to_thread(show.init_identity)
    show.reenroll()
    def _watch(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("background task %s died: %r", task.get_name(), task.exception())

    for coro, name in ((show.ui_broadcaster(), "ui_broadcaster"), (show.health_watch(), "health_watch")):
        t = asyncio.create_task(coro, name=name)
        t.add_done_callback(_watch)
    await show.start_speech()
    show.recordings_count = len(list_recordings(settings.data_dir / "recordings"))
    if settings.auto_record:
        show.start_recording()
    if show.semantics and show.semantics.llm is not None:
        t = asyncio.create_task(show.llm_warm_loop(), name="llm_warm")
        t.add_done_callback(_watch)
    if settings.has_team_bridge:
        show.bridge = TeamControlBridge(settings.team_api, settings.team_event_id, settings.team_producer_secret,
                                        parse_camera_map(settings.team_camera_map), resume_mode=settings.team_resume_mode)
        st = await show.bridge.check()
        if st:
            log.info("team GUI bridge: %s event %s reachable (mode %s, live %s); cameras %s", settings.team_api,
                     settings.team_event_id, st.get("mode"), st.get("liveCameraId"), show.bridge.camera_map)
            show.log_event("bridge", f"team GUI linked: {settings.team_api} event {settings.team_event_id} (mode {st.get('mode')})")
        else:
            log.error("team GUI bridge configured but unreachable: %s", show.bridge.stats["last_error"])
            show.log_event("bridge", f"team GUI UNREACHABLE: {show.bridge.stats['last_error']}")
    from .config import KEY_SOURCES
    for k, src in KEY_SOURCES.items():
        log.info("%s taken from %s", k, src)
    ip = lan_ip()
    log.info("Director UI:  http://%s:%d/   (also http://localhost:%d/)", ip, settings.port, settings.port)
    log.info("Setup page:   http://%s:%d/setup", ip, settings.port)
    log.info("Cameras:      python camera/stream_camera.py --server ws://%s:%d --cam B", ip, settings.port)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if show.recorder is not None:
        await show.stop_recording()
    if show.worker:
        show.worker.stop()
    if show.speech:
        show.speech.stop()
    if show.mic:
        show.mic.stop()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/setup")
async def setup_page() -> FileResponse:
    return FileResponse(settings.static_dir / "setup.html")


@app.get("/cam")
async def cam_page() -> FileResponse:
    return FileResponse(settings.static_dir / "cam.html")


@app.get("/api/health")
async def health() -> dict:
    now = time.monotonic()
    return {"ok": True, "version": APP_VERSION, "pid": os.getpid(),
            "cameras": {cid: c.healthy(now, settings.camera_stale_s) for cid, c in show.cameras.items()},
            "program": show.state.current_camera, "mode": show.state.mode.value,
            "llm": show.semantics is not None and show.semantics.llm is not None, "llm_provider": settings.llm_provider,
            "deepgram": settings.has_deepgram, "identity": show.engine is not None}


@app.get("/api/state")
async def api_state() -> dict:
    return show.snapshot()


@app.post("/api/say")
async def api_say(body: dict) -> dict:
    text = str(body.get("text", ""))
    if not text.strip():
        raise HTTPException(400, "text required")
    await show.say(text, source="typed")
    return {"ok": True, "last_cue": show.last_cue.to_json() if show.last_cue else None,
            "last_decision": show.last_decision.to_json() if show.last_decision else None,
            "program": show.state.current_camera}


@app.post("/api/control")
async def api_control(body: dict) -> dict:
    return show.control(str(body.get("action", "")), body.get("camera_id"))


@app.post("/api/record")
async def api_record(body: dict) -> dict:
    action = str(body.get("action", "")).lower()
    if action == "start":
        summary = show.start_recording()
        if summary.get("status") == "error":
            raise HTTPException(500, summary.get("error", "recording failed to start"))
        return {"ok": True, "recording": summary}
    if action == "stop":
        return {"ok": True, "recording": await show.stop_recording()}
    raise HTTPException(400, "action must be start or stop")


@app.get("/api/recordings")
async def api_recordings() -> dict:
    return {"recordings": list_recordings(settings.data_dir / "recordings")}


@app.get("/api/people")
async def api_people() -> dict:
    return {"people": show.snapshot()["roster"], "host_id": show.roster.host_id}


@app.post("/api/people")
async def api_add_person(name: str = Form(...), aliases: str = Form(""), role: str = Form(""), is_host: str = Form("false"),
                         person_id: str = Form(""), photos: list[UploadFile] = File(default=[])) -> dict:
    name = name.strip()
    if not name:
        raise HTTPException(400, "name required")
    pid = slug(person_id or name.split()[0], fallback=None) or slug(name, fallback=None)
    if pid is None:  # no Latin letters at all: still a unique, stable id
        pid = f"person-{len(show.roster.people) + 1}"
    if pid in show.roster.by_id and show.roster.by_id[pid].name.lower() != name.lower():
        pid = slug(name, fallback=None) or pid
    base_pid, n = pid, 1
    while pid in show.roster.by_id and show.roster.by_id[pid].name.lower() != name.lower():
        n += 1
        pid = f"{base_pid}-{n}"
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
    host = is_host.lower() in ("true", "1", "yes", "on")
    people = [p for p in show.roster.people if p.id != pid]
    if host:
        for p in people:
            p.is_host = False
    people.append(Person(pid, name, alias_list, role.strip(), host))
    show.roster = Roster(people)
    show.save_roster()
    pdir = settings.data_dir / "people" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, up in enumerate(photos):
        data = await up.read()
        if not data:
            continue
        ext = Path(up.filename or "photo.jpg").suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        dest = pdir / f"{int(time.time())}_{i}{ext}"
        dest.write_bytes(data)
        saved += 1
        await asyncio.to_thread(ensure_decodable, dest)  # HEIC from phones -> JPEG
    await asyncio.to_thread(show.refresh_context)
    rep = show.enroll_report.get(pid, {})
    return {"ok": True, "id": pid, "saved_photos": saved, "faces_enrolled": rep.get("faces", 0), "failed": rep.get("failed", [])}


@app.get("/api/people/{pid}/thumb")
async def api_person_thumb(pid: str):
    from fastapi.responses import Response
    sid = slug(pid, fallback=None)
    if sid is None:
        raise HTTPException(404, "no photo")
    pdir = settings.data_dir / "people" / sid
    if pdir.exists():
        for f in sorted(pdir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                data = await asyncio.to_thread(reference_thumbnail, f, 256)
                if data:
                    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    raise HTTPException(404, "no photo")


@app.delete("/api/people/{pid}")
async def api_delete_person(pid: str) -> dict:
    pid = slug(pid, fallback=None)
    if pid is None or pid not in show.roster.by_id:
        raise HTTPException(404, "unknown person")
    show.roster = Roster([p for p in show.roster.people if p.id != pid])
    show.save_roster()
    people_root = (settings.data_dir / "people").resolve()
    pdir = (people_root / pid).resolve()
    if pdir.parent == people_root and pdir.is_dir():
        shutil.rmtree(pdir, ignore_errors=True)
    for c in show.cameras.values():
        if c.fixed_person == pid:
            c.fixed_person = None
    show.save_cameras()
    await asyncio.to_thread(show.refresh_context)
    return {"ok": True}


@app.get("/api/script")
async def api_get_script() -> dict:
    return {"text": show.script_text}


@app.post("/api/script")
async def api_set_script(request: Request) -> dict:
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" in ctype:
        form = await request.form()
        f = form.get("file")
        text = (await f.read()).decode("utf-8", "replace") if f is not None and hasattr(f, "read") else str(form.get("text", ""))
    else:
        body = await request.json()
        text = str(body.get("text", ""))
    show.script_text = text
    (settings.data_dir / "script.md").write_text(text)
    await asyncio.to_thread(show.refresh_context)
    return {"ok": True, "chars": len(text)}


@app.get("/api/cameras")
async def api_get_cameras() -> dict:
    return {"cameras": {cid: {"label": c.label, "role": c.role, "fixed_person": c.fixed_person, "connected": c.connected}
                        for cid, c in show.cameras.items()}}


@app.post("/api/cameras")
async def api_set_cameras(body: dict) -> dict:
    cams = body.get("cameras") or {}
    if not isinstance(cams, dict) or not cams:
        raise HTTPException(400, "cameras required")
    for cid, cfg in cams.items():
        cid = str(cid).strip().upper()[:12]
        if not cid:
            continue
        cam = show.cameras.get(cid) or Camera(cid, f"Camera {cid}")
        cam.label = str(cfg.get("label") or cam.label)
        role = str(cfg.get("role") or "guest").lower()
        cam.role = role if role in ("wide", "host", "guest") else "guest"
        fp = cfg.get("fixed_person") or None
        cam.fixed_person = fp if fp in show.roster.ids else None
        with show._lock:
            show.cameras[cid] = cam
    kept = []
    for cid in [c for c in show.cameras if c not in {str(k).strip().upper()[:12] for k in cams}]:
        if not show.cameras[cid].connected:
            with show._lock:
                del show.cameras[cid]
            show.trackers.pop(cid, None)
        else:
            kept.append(cid)
    show.save_cameras()
    show.mark_dirty()
    return {**(await api_get_cameras()), "kept_connected": kept}


@app.websocket("/ingest")
async def ws_ingest(ws: WebSocket) -> None:
    await ws.accept()
    cam_id = (ws.query_params.get("cam") or "").strip().upper()[:12]
    label = ws.query_params.get("label") or ""
    role = (ws.query_params.get("role") or "").lower()
    if not cam_id:
        await ws.send_text(json.dumps({"type": "error", "error": "cam query param required, e.g. /ingest?cam=B"}))
        await ws.close()
        return
    cam = show.cameras.get(cam_id)
    if cam is None:
        cam = Camera(cam_id, label or f"Camera {cam_id}", role if role in ("wide", "host", "guest") else "guest")
        with show._lock:
            show.cameras[cam_id] = cam
        show.save_cameras()
    elif label and cam.label == f"Camera {cam_id}":
        cam.label = label[:40]  # a label saved on /setup wins over what the laptop announces
    token = object()
    old_ws = cam.owner_ws
    cam.owner = token
    cam.owner_ws = ws
    cam.connected = True
    if old_ws is not None:  # a newer laptop/browser took this id: the older stream is closed, never interleaved
        show.log_event("camera", f"{cam_id}: new connection replaces the previous one")
        try:
            await old_ws.close(code=4000, reason="replaced by a newer connection for this camera id")
        except Exception:
            pass
    cam.client = ws.client.host if ws.client else ""
    show.log_event("camera", f"{cam_id} connected from {cam.client}")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                if cam.owner is not token:
                    break  # we were replaced; stop reading so the client sees the close
                now = time.monotonic()
                with show._lock:
                    cam.push(data, now)
                show.relay_frame(cam_id, data)
                continue
            text = msg.get("text")
            if text:
                try:
                    j = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if j.get("type") == "hello":
                    if j.get("label") and cam.label == f"Camera {cam_id}":
                        cam.label = str(j["label"])[:40]
                    await ws.send_text(json.dumps({"type": "ack", "cam": cam_id, "role": cam.role}))
                elif j.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "t": j.get("t")}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("ingest %s error: %s", cam_id, e)
    finally:
        if cam.owner is token:  # a newer connection for this id may already own the camera
            cam.connected = False
            cam.owner_ws = None
            show.log_event("camera", f"{cam_id} disconnected")
        else:
            show.log_event("camera", f"{cam_id}: stale connection closed, newer stream keeps running")
        show.mark_dirty()


@app.websocket("/ui")
async def ws_ui(ws: WebSocket) -> None:
    await ws.accept()
    client = UIClient(ws)
    show.ui_clients.add(client)
    sender = asyncio.create_task(client.sender())
    try:
        await ws.send_text(json.dumps(show.snapshot()))
        while True:
            text = await ws.receive_text()
            try:
                j = json.loads(text)
            except json.JSONDecodeError:
                continue
            t = j.get("type")
            if t == "control":
                try:
                    show.control(str(j.get("action", "")), j.get("camera_id"))
                except HTTPException as e:
                    await ws.send_text(json.dumps({"type": "error", "error": e.detail}))
            elif t == "say":
                await show.say(str(j.get("text", "")), source="typed")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("ui ws closed: %s", e)
    finally:
        client.alive = False
        client.wake.set()
        sender.cancel()
        show.ui_clients.discard(client)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host=settings.host, port=settings.port, log_level="info", ws_max_size=8 * 1024 * 1024)


if __name__ == "__main__":
    main()
