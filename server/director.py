"""Deterministic director. Adapted from the CUE repo's policy/director.py (C lane).

`decide(cue, cameras, state, now)` is pure: no network, no LLM, no clocks.
The caller passes `now`, gets a Decision naming a camera (or SLATE) plus a
human-readable reason, then calls `apply()` to update the State.

Rules:
- Manual HOLD beats every automatic decision; safety failover still runs.
- Only NOW cues may cut. FUTURE / PAST / NEGATED / UNCERTAIN never cut.
- A named SHOW needs a healthy camera with a FRESH confirmed identity of the
  target (face seen within identity_max_age_s). If none, the setup page's
  fixed person->camera mapping is used (disclosed as "static"). If none, WIDE.
  If no healthy wide either, SLATE. Never guess a guest.
- HOST resolves to the host's identity, else the camera whose role is "host".
- Group scope / WIDE -> the healthy wide camera.
- Minimum shot length min_shot_s, bypassed once per utterance when the new cue
  shares the utterance_id of the cue that made the current cut (correction
  "Sarah, actually Daniel"), or when the live camera is unhealthy.
- Cues older than cue_lifetime_s are stale and ignored.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    AUTO = "AUTO"
    HOLD = "HOLD"


class Take(str, Enum):
    TAKE = "TAKE"
    STAY = "STAY"
    SLATE = "SLATE"


@dataclass
class CameraView:
    id: str
    role: str = "guest"                      # wide | host | guest
    healthy: bool = False
    confirmed: dict[str, float] = field(default_factory=dict)  # person_id -> evidence age (s)
    fixed_person: str | None = None          # setup-page mapping, used only as fallback


@dataclass
class Decision:
    action: Take
    camera_id: str | None
    reason: str
    evidence: str = ""        # identity | static | wide | host-role | manual | none
    target: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"action": self.action.value, "camera_id": self.camera_id, "reason": self.reason,
                "evidence": self.evidence, "target": self.target}


@dataclass
class State:
    current_camera: str | None = None
    last_cut_time: float = -1e9
    mode: Mode = Mode.AUTO
    hold_until: float | None = None
    last_cut_utterance_id: str = ""
    last_cut_target: str | None = None
    recuts_in_utterance: int = 0


@dataclass(frozen=True)
class Tunables:
    min_shot_s: float = 2.5
    cue_lifetime_s: float = 3.0
    identity_max_age_s: float = 4.0
    max_recuts_per_utterance: int = 1


DEFAULT_TUNABLES = Tunables()


# ----------------------------------------------------------------- helpers
def _val(x: Any) -> str:
    if x is None:
        return ""
    return x.value if hasattr(x, "value") else str(x)


def _healthy_wide(cameras: Mapping[str, CameraView]) -> str | None:
    for cid, c in cameras.items():
        if c.role == "wide" and c.healthy:
            return cid
    return None


def _healthy_role(cameras: Mapping[str, CameraView], role: str) -> str | None:
    for cid, c in cameras.items():
        if c.role == role and c.healthy:
            return cid
    return None


def _is_healthy(cameras: Mapping[str, CameraView], cid: str | None) -> bool:
    return bool(cid) and cid in cameras and cameras[cid].healthy


def _safe_fallback(cameras: Mapping[str, CameraView], state: State, reason: str) -> Decision:
    """Stay if the live camera is healthy; else wide; else SLATE."""
    if _is_healthy(cameras, state.current_camera):
        return Decision(Take.STAY, state.current_camera, reason, "none")
    wide = _healthy_wide(cameras)
    if wide:
        if wide == state.current_camera:
            return Decision(Take.STAY, wide, reason + "; wide already live", "wide")
        return Decision(Take.TAKE, wide, reason + "; live camera unhealthy -> wide", "wide")
    any_healthy = next((cid for cid, c in cameras.items() if c.healthy), None)
    if any_healthy:
        if any_healthy == state.current_camera:
            return Decision(Take.STAY, any_healthy, reason, "none")
        return Decision(Take.TAKE, any_healthy, reason + "; no wide, taking the only healthy camera", "none")
    return Decision(Take.SLATE, None, reason + "; no healthy camera -> slate", "none")


def _others_fresh(c: CameraView, target: str | None, tun: Tunables) -> list[str]:
    return [pid for pid, age in c.confirmed.items() if pid != target and age <= tun.identity_max_age_s]


def _pick_named(target: str, cameras: Mapping[str, CameraView], tun: Tunables, current: str | None = None,
                host_id: str | None = None) -> tuple[str | None, str, str]:
    """Fresh identity first (keep the live close-up if it still shows the target, else prefer
    non-wide, then freshest), then the static mapping, but never a camera that freshly shows
    somebody else. The wide camera sees everyone, so being live on it never counts as 'already
    showing' the target; a host-role camera only counts for the host."""
    if current and current in cameras and cameras[current].healthy:
        cur = cameras[current]
        close_up = cur.role == "guest" or (cur.role == "host" and target == host_id)
        age = cur.confirmed.get(target)
        if close_up and age is not None and age <= tun.identity_max_age_s:
            return current, f"fresh identity: {target} still on live camera {current} ({age:.1f}s ago)", "identity"
    best: tuple[float, int, str] | None = None
    for cid, c in cameras.items():
        if not c.healthy or target not in c.confirmed:
            continue
        age = c.confirmed[target]
        if age > tun.identity_max_age_s:
            continue
        rank = {"guest": 0, "host": 1}.get(c.role, 2)  # close-up first, wide last
        key = (rank, age)
        if best is None or key < (best[1], best[0]):
            best = (age, key[0], cid)
    if best is not None:
        age, _, cid = best
        return cid, f"fresh identity: {target} seen on {cid} {age:.1f}s ago", "identity"
    for cid, c in cameras.items():
        if c.healthy and c.fixed_person == target:
            others = _others_fresh(c, target, tun)
            if others:
                return None, f"static mapping {target} -> {cid} refused: {cid} currently shows {', '.join(others)}", ""
            return cid, f"static mapping: {target} -> {cid} (no fresh face match)", "static"
    return None, "", ""


def _propose_take(picked: str, why: str, evidence: str, target: str | None,
                  cameras: Mapping[str, CameraView], state: State, cue: Any, now: float, tun: Tunables) -> Decision:
    if picked == state.current_camera:
        return Decision(Take.STAY, picked, why + " (already live)", evidence, target)
    if state.current_camera and not _is_healthy(cameras, state.current_camera):
        return Decision(Take.TAKE, picked, why + "; live camera unhealthy, safety failover", evidence, target)
    elapsed = now - state.last_cut_time
    if elapsed < tun.min_shot_s:
        utt = getattr(cue, "utterance_id", "") or ""
        same_target = target is not None and target == state.last_cut_target
        if (utt and utt == state.last_cut_utterance_id and state.recuts_in_utterance < tun.max_recuts_per_utterance
                and not same_target):
            return Decision(Take.TAKE, picked, why + f" (in-utterance correction re-cut at {elapsed:.2f}s)", evidence, target)
        return Decision(Take.STAY, state.current_camera,
                        f"min shot {tun.min_shot_s}s not met ({elapsed:.2f}s); held {state.current_camera}. Would have taken {picked}: {why}",
                        "none", target)
    return Decision(Take.TAKE, picked, why, evidence, target)


# ------------------------------------------------------------------ public
def decide(cue: Any, cameras: Mapping[str, CameraView], state: State, now: float, *,
           host_id: str | None = None, tun: Tunables = DEFAULT_TUNABLES) -> Decision:
    holding = state.mode == Mode.HOLD or (state.hold_until is not None and now < state.hold_until)
    if holding:
        return _safe_fallback(cameras, state, "manual HOLD")
    if cue is None:
        return _safe_fallback(cameras, state, "no cue")

    created = float(getattr(cue, "created_at", 0.0) or 0.0)
    if created > 0 and (now - created) > tun.cue_lifetime_s:
        return _safe_fallback(cameras, state, f"cue stale ({now - created:.2f}s > {tun.cue_lifetime_s}s)")

    temporal = _val(getattr(cue, "temporal_intent", None))
    action = _val(getattr(cue, "action", None))
    evidence_text = getattr(cue, "evidence_text", "") or ""
    intent = _val(getattr(cue, "intent", None))
    if intent == "CANCEL":
        # "Sarah, come up. Actually wait..." -> undo the cut we just made, once, back to wide.
        utt = getattr(cue, "utterance_id", "") or ""
        wide = _healthy_wide(cameras)
        if (utt and utt == state.last_cut_utterance_id and (now - state.last_cut_time) < tun.min_shot_s
                and state.recuts_in_utterance < tun.max_recuts_per_utterance and wide and wide != state.current_camera):
            return Decision(Take.TAKE, wide, f"host cancelled ('{evidence_text}') right after the cut -> back to wide", "wide")
        return _safe_fallback(cameras, state, f"cancel ('{evidence_text}'): nothing to undo, staying")
    if temporal != "NOW":
        return _safe_fallback(cameras, state, f"temporal_intent={temporal or 'none'}, not NOW ('{evidence_text}')")
    if action == "HOLD" or not action:
        return _safe_fallback(cameras, state, f"cue says HOLD ('{evidence_text}')")

    if action == "HOST":
        target = host_id
        if target:
            picked, why, ev = _pick_named(target, cameras, tun, state.current_camera, host_id)
            if picked:
                return _propose_take(picked, "return to host; " + why, ev, target, cameras, state, cue, now, tun)
        hc = _healthy_role(cameras, "host")
        if hc and _others_fresh(cameras[hc], target, tun):
            wide = _healthy_wide(cameras)
            why = f"return to host; host camera {hc} currently shows {', '.join(_others_fresh(cameras[hc], target, tun))}"
            if wide:
                return _propose_take(wide, why + " -> wide", "wide", target, cameras, state, cue, now, tun)
            return _safe_fallback(cameras, state, why)
        if hc:
            return _propose_take(hc, "return to host; camera with role host", "host-role", target, cameras, state, cue, now, tun)
        wide = _healthy_wide(cameras)
        if wide:
            return _propose_take(wide, "return to host; no host camera -> wide", "wide", target, cameras, state, cue, now, tun)
        return _safe_fallback(cameras, state, "return to host but no host/wide camera")

    scope = _val(getattr(cue, "scope", None))
    targets = list(getattr(cue, "target_ids", []) or [])
    if action == "WIDE" or scope == "group":
        wide = _healthy_wide(cameras)
        if wide:
            return _propose_take(wide, f"group/wide cue ('{evidence_text}') -> wide", "wide", None, cameras, state, cue, now, tun)
        return _safe_fallback(cameras, state, "group cue but no healthy wide camera")

    if len(targets) != 1:
        return _safe_fallback(cameras, state, f"target count {len(targets)}, need exactly one")
    target = targets[0]
    picked, why, ev = _pick_named(target, cameras, tun, state.current_camera, host_id)
    if picked:
        return _propose_take(picked, why, ev, target, cameras, state, cue, now, tun)
    wide = _healthy_wide(cameras)
    if wide:
        return _propose_take(wide, (why + " -> wide") if why else f"{target} not confirmed on any camera -> wide (never guess)", "wide", target,
                             cameras, state, cue, now, tun)
    if _is_healthy(cameras, state.current_camera):
        return Decision(Take.STAY, state.current_camera, f"{target} not confirmed; no wide; staying", "none", target)
    return Decision(Take.SLATE, None, f"{target} not confirmed; no healthy wide -> slate", "none", target)


def apply(state: State, decision: Decision, cue: Any, now: float) -> bool:
    """Mutate state for a TAKE. Returns True when the program output changed."""
    if decision.action != Take.TAKE or not decision.camera_id:
        if decision.action == Take.SLATE and state.current_camera is not None:
            state.current_camera = None
            return True
        return False
    utt = (getattr(cue, "utterance_id", "") or "") if cue is not None else ""
    if utt and utt == state.last_cut_utterance_id:
        state.recuts_in_utterance += 1
    else:
        state.recuts_in_utterance = 0
    state.last_cut_utterance_id = utt
    state.last_cut_target = decision.target
    state.current_camera = decision.camera_id
    state.last_cut_time = now
    return True


def manual_take(state: State, camera_id: str, now: float) -> None:
    state.current_camera = camera_id
    state.last_cut_time = now
    state.last_cut_utterance_id = ""
    state.last_cut_target = None
    state.recuts_in_utterance = 0
