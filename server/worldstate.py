"""One evolving world state (spec §3). Every loop writes structured observations here; the
director reads a consistent snapshot with a version number. No decisions live in this file."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PersonObs:
    track_id: str
    person_id: str | None          # roster id when identity is confident, else None
    id_conf: float
    box: tuple[int, int, int, int]
    face_w_ratio: float            # face width / frame width
    speaking: float                # visual speech activity 0..1 (mouth motion), EMA
    yaw: float                     # -1 (looking frame-left) .. 0 (at camera) .. +1 (frame-right)
    motion: float                  # local motion energy 0..1
    first_seen: float
    last_seen: float


@dataclass
class CameraObservation:
    cam_id: str
    at: float
    shot_type: str                 # CLOSE_UP | MEDIUM_CLOSE_UP | MEDIUM | WIDE | EMPTY | UNUSABLE
    people: list[PersonObs]
    motion: float                  # global motion 0..1
    frozen: bool
    sharpness: float               # 0..1
    brightness: float              # 0..1
    entrances: list[str]           # track ids that appeared this observation
    exits: list[str]               # track ids that vanished this observation
    frame_size: tuple[int, int] = (0, 0)
    semantic: dict = field(default_factory=dict)   # optional VLM summary: {"action","objects","reaction","at"}


@dataclass
class ConversationState:
    current_speaker: str | None = None      # "host" | roster id | None
    speaker_conf: float = 0.0
    dialogue_act: str = "NONE"              # INTRODUCTION | QUESTION | ANSWER | REBUTTAL | JOKE | EXPLANATION | DEMONSTRATION | TRANSITION | CONCLUSION | AUDIENCE | BACKCHANNEL | NONE
    addressed: list[str] = field(default_factory=list)     # roster ids, "host", "all", "audience"
    expected_next: str | None = None
    backchannel: bool = False
    sentence_complete: bool = True
    salience: float = 0.3                    # narrative / emotional importance 0..1
    references: list[str] = field(default_factory=list)    # mentioned, not addressed
    transition: bool = False
    overlap: bool = False
    subject: str | None = None               # "audience" | "screen" | "object" | None: the visual subject of the moment
    subject_phrase: str | None = None        # the words after "look at the ...", matched against scene tags
    updated_at: float = 0.0
    last_clause: str = ""
    source: str = ""


@dataclass
class ShotEntry:
    cam_id: str | None
    started_at: float
    ended_at: float | None
    reason_codes: list[str]
    origin: str                              # cue | planner | manual | safety | suggestion-accepted


class WorldState:
    def __init__(self, host_id: str | None = None):
        self._lock = threading.Lock()
        self.version = 0
        self.host_id = host_id
        self.cameras: dict[str, CameraObservation] = {}
        self.camera_roles: dict[str, str] = {}
        self.camera_healthy: dict[str, bool] = {}
        self.fixed_person: dict[str, str | None] = {}
        self.scene: dict[str, dict] = {}        # cam id -> {"objects", "people", "setting", "at"} from scene.py
        self.conv = ConversationState()
        self.audio_level = 0.0
        self.speech_active = False
        self.speech_since: float | None = None
        self.silence_since: float | None = None
        self.audio_at = 0.0
        self.shot: ShotEntry | None = None
        self.history: deque[ShotEntry] = deque(maxlen=50)
        self.last_shown: dict[str, float] = {}
        self.cut_times: deque[float] = deque(maxlen=100)
        self.mode = "AUTO"                    # MANUAL | ASSIST | AUTO
        self.hold = False

    # ------------------------------------------------------------ writers
    def _bump(self) -> None:
        self.version += 1

    def update_camera(self, obs: CameraObservation) -> None:
        with self._lock:
            self.cameras[obs.cam_id] = obs
            self._bump()

    def update_health(self, roles: dict[str, str], healthy: dict[str, bool], fixed: dict[str, str | None]) -> None:
        with self._lock:
            self.camera_roles = dict(roles)
            self.camera_healthy = dict(healthy)
            self.fixed_person = dict(fixed)
            self._bump()

    def update_audio(self, level: float, now: float, threshold: float = 0.06) -> None:
        with self._lock:
            self.audio_level = level
            self.audio_at = now
            active = level >= threshold
            if active and not self.speech_active:
                self.speech_since = now
            if not active and self.speech_active:
                self.silence_since = now
            self.speech_active = active
            self._bump()

    def update_scene(self, cam_id: str, info: dict) -> None:
        with self._lock:
            self.scene[cam_id] = dict(info)
            self._bump()

    def update_conv(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                if hasattr(self.conv, k):
                    setattr(self.conv, k, v)
            self._bump()

    def record_cut(self, cam_id: str | None, now: float, codes: list[str], origin: str) -> None:
        with self._lock:
            if self.shot is not None:
                self.shot.ended_at = now
                self.history.append(self.shot)
            self.shot = ShotEntry(cam_id, now, None, list(codes), origin)
            self.cut_times.append(now)
            if cam_id:
                for p in (self.cameras.get(cam_id).people if cam_id in self.cameras else []):
                    if p.person_id:
                        self.last_shown[p.person_id] = now
            self._bump()

    def set_mode(self, mode: str, hold: bool) -> None:
        with self._lock:
            self.mode = mode
            self.hold = hold
            self._bump()

    # ------------------------------------------------------------ readers
    def shot_age(self, now: float) -> float:
        return (now - self.shot.started_at) if self.shot else 1e9

    def cuts_in(self, now: float, window_s: float) -> int:
        return sum(1 for t in self.cut_times if now - t <= window_s)

    def speaking_camera(self, now: float, max_age_s: float = 1.0) -> tuple[str | None, float]:
        """Fuse master-audio activity with per-camera visual speech activity: who is talking
        on which camera. Returns (cam_id, confidence)."""
        if not self.speech_active or now - self.audio_at > max_age_s:
            return None, 0.0
        best, best_v = None, 0.0
        for cid, obs in self.cameras.items():
            if now - obs.at > max_age_s or not self.camera_healthy.get(cid, False):
                continue
            v = max((p.speaking for p in obs.people), default=0.0)
            if v > best_v:
                best, best_v = cid, v
        if best is None or best_v < 0.25:
            return None, 0.0
        return best, min(1.0, best_v)

    def person_cameras(self, person_id: str, now: float, max_age_s: float = 4.0) -> list[tuple[str, PersonObs]]:
        out = []
        for cid, obs in self.cameras.items():
            if now - obs.at > max_age_s:
                continue
            for p in obs.people:
                if p.person_id == person_id:
                    out.append((cid, p))
        return out

    def snapshot(self, now: float) -> dict:
        with self._lock:
            return {
                "version": self.version,
                "cameras": {cid: {
                    "age_s": round(now - o.at, 2), "shot_type": o.shot_type, "motion": round(o.motion, 2), "frozen": o.frozen,
                    "sharpness": round(o.sharpness, 2), "people": [
                        {"track": p.track_id, "person": p.person_id, "id_conf": round(p.id_conf, 2), "speaking": round(p.speaking, 2),
                         "yaw": round(p.yaw, 2), "face_w": round(p.face_w_ratio, 2)} for p in o.people],
                    "semantic": o.semantic,
                } for cid, o in self.cameras.items()},
                "conversation": {
                    "current_speaker": self.conv.current_speaker, "speaker_conf": round(self.conv.speaker_conf, 2),
                    "dialogue_act": self.conv.dialogue_act, "addressed": self.conv.addressed, "expected_next": self.conv.expected_next,
                    "backchannel": self.conv.backchannel, "sentence_complete": self.conv.sentence_complete,
                    "salience": round(self.conv.salience, 2), "references": self.conv.references, "transition": self.conv.transition,
                    "overlap": self.conv.overlap, "subject": self.conv.subject, "subject_phrase": self.conv.subject_phrase,
                    "age_s": round(now - self.conv.updated_at, 1) if self.conv.updated_at else None,
                    "last_clause": self.conv.last_clause, "source": self.conv.source,
                },
                "scene": {cid: {"objects": i.get("objects"), "people": i.get("people"), "setting": i.get("setting"),
                                "age_s": round(now - float(i.get("at", now)), 1)} for cid, i in self.scene.items()},
                "audio": {"level": round(self.audio_level, 3), "speech_active": self.speech_active,
                          "speech_for_s": round(now - self.speech_since, 1) if self.speech_active and self.speech_since else None,
                          "silence_for_s": round(now - self.silence_since, 1) if (not self.speech_active and self.silence_since) else None},
                "shot": {"cam": self.shot.cam_id, "for_s": round(now - self.shot.started_at, 1), "codes": self.shot.reason_codes, "origin": self.shot.origin} if self.shot else None,
                "cuts_last_60s": self.cuts_in(now, 60), "mode": self.mode, "hold": self.hold,
            }
