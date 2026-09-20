"""Editorial director loop (spec §4, §7, §8): candidate shots scored from the world state with
reason codes; executes only when the editorial benefit beats the continuity cost. Pure: no
network, no clocks. The deterministic control layer (director.py guards, app) still validates
health, mode, freshness and minimum duration before anything reaches the program output."""
from __future__ import annotations

from dataclasses import dataclass, field

from .scene import best_camera
from .worldstate import WorldState

MIN_SHOT_S = 2.5
SCENE_MAX_AGE_S = 600.0   # tags older than this say nothing about the room any more
REACTION_MIN_SHOT_S = 1.5
MAX_CUTS_PER_10S = 3


@dataclass
class Candidate:
    cam_id: str
    score: float = 0.0
    codes: list[str] = field(default_factory=list)

    def add(self, v: float, code: str) -> None:
        self.score += v
        self.codes.append(code)


@dataclass
class Proposal:
    decision: str                    # HOLD_CURRENT | CUT_TO_CAMERA | SUGGEST_CAMERA | USE_SAFETY_SHOT | WAIT_FOR_MORE_EVIDENCE
    camera_id: str | None
    confidence: float
    min_hold_ms: int
    reason_codes: list[str]
    alternatives: list[dict]
    state_version: int
    explanation: str = ""

    def to_json(self) -> dict:
        return {"decision": self.decision, "camera_id": self.camera_id, "confidence": round(self.confidence, 2), "min_hold_ms": self.min_hold_ms,
                "reason_codes": self.reason_codes, "alternatives": self.alternatives, "state_version": self.state_version, "explanation": self.explanation}


def _visible(ws: WorldState, cam: str, person: str, now: float, max_age: float = 3.0) -> tuple[bool, float]:
    obs = ws.cameras.get(cam)
    if obs is None or now - obs.at > max_age:
        return False, 0.0
    for p in obs.people:
        if p.person_id == person:
            return True, p.face_w_ratio
    return False, 0.0


def _resolve(ws: WorldState, who: str | None) -> str | None:
    if who == "host":
        return ws.host_id
    return who


def plan(ws: WorldState, now: float, cue=None, cue_age_s: float = 99.0) -> Proposal:
    conv = ws.conv
    current = ws.shot.cam_id if ws.shot else None
    healthy = {c for c, h in ws.camera_healthy.items() if h}
    roles = ws.camera_roles
    wide = next((c for c in roles if roles[c] == "wide" and c in healthy), None)
    host_cam = next((c for c in roles if roles[c] == "host" and c in healthy), None)
    shot_age = ws.shot_age(now)
    codes_global: list[str] = []

    if ws.hold or ws.mode == "MANUAL":
        return Proposal("HOLD_CURRENT", current, 1.0, 0, ["OPERATOR_HOLD" if ws.hold else "MANUAL_MODE"], [], ws.version, "operator owns the program")

    # safety first: the live camera is unhealthy -> best healthy alternative, wide preferred
    if current is not None and current not in healthy:
        target = wide or host_cam or (sorted(healthy)[0] if healthy else None)
        return Proposal("USE_SAFETY_SHOT", target, 1.0, 0, ["CAMERA_UNHEALTHY", "SAFETY_FAILOVER"], [], ws.version, f"{current} is not delivering frames")
    if not healthy:
        return Proposal("HOLD_CURRENT", current, 0.0, 0, ["NO_HEALTHY_CAMERA"], [], ws.version, "no healthy camera")

    speaking_cam, speaking_conf = ws.speaking_camera(now)
    speaker = _resolve(ws, conv.current_speaker)
    addressed = [a for a in ([_resolve(ws, a) for a in conv.addressed]) if a]
    expected = _resolve(ws, conv.expected_next)
    conv_fresh = conv.updated_at and (now - conv.updated_at) < 6.0
    cue_fresh = cue is not None and cue_age_s < 3.0 and getattr(cue, "temporal_intent", None) is not None and cue.temporal_intent.value == "NOW"
    seen_cam, seen_tied = None, []
    if conv_fresh and conv.subject_phrase:
        seen_cam, _, seen_tied = best_camera(ws.scene, conv.subject_phrase, now, SCENE_MAX_AGE_S, ws.camera_healthy)
        role_for = {"audience": ("audience",), "screen": ("demo", "screen")}.get(conv.subject or "", ())
        if any(r in role_for and ws.camera_healthy.get(c) for c, r in ws.camera_roles.items()):
            seen_cam, seen_tied = None, []   # the operator pointed a camera at exactly this: it outranks the tags
    silence_for = (now - ws.silence_since) if (not ws.speech_active and ws.silence_since) else 0.0

    cands: dict[str, Candidate] = {c: Candidate(c) for c in healthy}
    for c, cand in cands.items():
        obs = ws.cameras.get(c)
        age = (now - obs.at) if obs else 99.0
        role = roles.get(c, "guest")
        # ---- narrative relevance
        if cue_fresh and cue.action.value == "SHOW" and cue.target_ids:
            vis, fw = _visible(ws, c, cue.target_ids[0], now)
            if vis:
                cand.add(1.0 if role != "wide" else 0.55, "CUE_TARGET_VISIBLE")
            elif role == "wide":
                cand.add(0.75, "CUE_TARGET_NOT_CONFIRMED_WIDE")   # never guess a guest: the safe shot wins
            elif ws.fixed_person.get(c) == cue.target_ids[0]:
                cand.add(0.6, "CUE_TARGET_STATIC_MAP")
        if cue_fresh and cue.action.value == "HOST":
            if ws.host_id and _visible(ws, c, ws.host_id, now)[0]:
                cand.add(0.9, "RETURN_TO_HOST_VISIBLE")
            elif role == "host":
                cand.add(0.6, "RETURN_TO_HOST_ROLE")
        if cue_fresh and cue.action.value == "WIDE" and role == "wide":
            cand.add(0.9, "GROUP_CUE")
        non_person_subject = conv_fresh and conv.subject in ("audience", "screen", "object")
        if conv_fresh and conv.subject == "audience":
            if role in ("audience", "wide"):
                cand.add(1.0 if role == "audience" else 0.8, "SUBJECT_AUDIENCE")
        if conv_fresh and conv.subject in ("screen", "object") and not (conv.subject == "object" and seen_cam is not None):
            if role in ("demo", "screen"):
                cand.add(1.0, "SUBJECT_DEMONSTRATION")
            elif role == "wide":
                cand.add(0.8, "SUBJECT_DEMONSTRATION_WIDE")
        if seen_cam is not None and c == seen_cam:
            cand.add(1.0, "SUBJECT_SEEN_ON_CAMERA")            # the one camera whose tags show what the host named
        elif seen_cam is None and len(seen_tied) > 1 and role == "wide":
            cand.add(0.8, "SUBJECT_ON_SEVERAL_CAMERAS_WIDE")   # two cameras show it: never guess
        if conv_fresh and conv.dialogue_act == "INTRODUCTION" and obs and role == "wide" and any(now - ws.cameras[c].at < 3.0 for _ in [0]) and obs.entrances:
            cand.add(0.4, "ENTRANCE_ON_WIDE")
        if conv_fresh and conv.transition and role in ("host", "wide") and not cue_fresh:
            cand.add(0.35 if role == "host" else 0.25, "SEGMENT_TRANSITION")
        for a in addressed:
            if a == "all":
                if role == "wide":
                    cand.add(0.7, "ADDRESSED_ALL_WIDE")
            else:
                vis, fw = _visible(ws, c, a, now)
                if vis and role != "wide":
                    cand.add(0.7, "ADDRESSEE_VISIBLE")
        if expected and expected not in addressed:
            vis, _ = _visible(ws, c, expected, now)
            if vis and role != "wide":
                cand.add(0.3, "EXPECTED_SPEAKER_VISIBLE")
        # ---- speaker value (audio + mouth motion); halved when the audience should look at something else
        sv = 0.5 if non_person_subject else 1.0
        if speaking_cam == c:
            cand.add(0.6 * speaking_conf * sv, "ACTIVE_SPEAKER")
        if speaker and _visible(ws, c, speaker, now)[0] and role != "wide":
            cand.add(0.35 * sv, "CURRENT_SPEAKER_VISIBLE")
        # ---- reaction value: somebody else visibly moving while the speaker pauses, only in a short window
        speaker_cam = None
        if speaker:
            seen = ws.person_cameras(speaker, now)
            if seen:
                speaker_cam = max(seen, key=lambda sp: sp[1].face_w_ratio)[0]   # the speaker's close-up, not the wide
        if obs and speaker_cam and c != speaker_cam and role != "wide" and 0.3 <= silence_for <= 1.5 and conv.salience >= 0.5:
            if any(p.motion > 0.35 and p.speaking < 0.2 for p in obs.people):
                cand.add(0.45, "REACTION")
        # ---- overlap / uncertainty
        if conv_fresh and conv.overlap:  # ownership is ambiguous: cover the group, never oscillate between close-ups
            cand.add(0.8 if role == "wide" else -0.5, "OVERLAP_WIDE" if role == "wide" else "OVERLAP_CLOSEUP_PENALTY")
        # ---- visual quality and composition
        if obs:
            if obs.frozen or obs.shot_type == "UNUSABLE":
                cand.add(-1.0, "FRAME_UNUSABLE")
            else:
                cand.add(0.15 * obs.sharpness, "SHARP" if obs.sharpness > 0.5 else "SOFT")
                if role != "wide" and obs.shot_type == "EMPTY":
                    cand.add(-0.6, "EMPTY_CLOSEUP")
                if role != "wide" and obs.people and max(p.face_w_ratio for p in obs.people) < 0.06:
                    cand.add(-0.2, "FACE_TOO_SMALL")
            if age > 1.5:
                cand.add(-0.3, "STALE_OBSERVATION")
        else:
            cand.add(-0.2 if role != "wide" else 0.0, "NO_OBSERVATION")
        # ---- variety
        last_on = None
        for h in reversed(ws.history):
            if h.cam_id == c:
                last_on = h.ended_at
                break
        if c != current and last_on is not None and now - last_on > 25:
            cand.add(0.05, "VARIETY")
        if role == "wide":
            cand.add(0.15, "SAFE_WIDE")
        # ---- continuity
        if c == current:
            cand.add(0.25 if shot_age < 4 else 0.15, "CONTINUITY")
            if speaker and _visible(ws, c, speaker, now)[0] and not conv.sentence_complete:
                cand.add(0.2, "SENTENCE_CONTINUES")
            if conv.salience >= 0.7 and speaker and _visible(ws, c, speaker, now)[0]:
                cand.add(0.35, "EMOTIONAL_MOMENT")

    ranked = sorted(cands.values(), key=lambda k: k.score, reverse=True)
    alternatives = [{"cameraId": k.cam_id, "score": round(k.score, 2), "codes": k.codes} for k in ranked]
    best = ranked[0]
    cur = cands.get(current) if current else None
    cur_score = cur.score if cur else -1.0

    # back-channel: never cut for "mm-hmm"
    if conv_fresh and conv.backchannel and (now - conv.updated_at) < 2.5:
        return Proposal("HOLD_CURRENT", current, 0.9, 1000, ["BACKCHANNEL"] + (cur.codes if cur else []), alternatives, ws.version, "short acknowledgement, not a handoff")

    if current is None:
        return Proposal("CUT_TO_CAMERA" if ws.mode == "AUTO" else "SUGGEST_CAMERA", best.cam_id, 0.8, 2500, best.codes + ["NO_PROGRAM_YET"], alternatives, ws.version, "first shot")

    margin = 0.3
    if conv.salience >= 0.7 and cur and "EMOTIONAL_MOMENT" in cur.codes:
        margin = 0.55
    if cue_fresh:
        margin = 0.15
    if conv_fresh and conv.overlap:
        margin = 0.25
    rapid = ws.cuts_in(now, 10.0)
    gain = best.score - cur_score
    if best.cam_id == current or gain < margin:
        conf = min(1.0, 0.5 + (cur_score - (ranked[1].score if len(ranked) > 1 else 0)) if cur else 0.5)
        return Proposal("HOLD_CURRENT", current, conf, 0, (cur.codes if cur else []) + ["SMALL_ADVANTAGE" if best.cam_id != current else "BEST_SHOT"],
                        alternatives, ws.version, f"{current} keeps the moment" if best.cam_id == current else f"{best.cam_id} is only +{gain:.2f} better")
    is_reaction = "REACTION" in best.codes and not cue_fresh
    min_shot = REACTION_MIN_SHOT_S if is_reaction else MIN_SHOT_S
    if shot_age < min_shot and not (cue_fresh and getattr(cue, "utterance_id", "") and ws.shot and ws.shot.origin == "cue" and shot_age < 2.5 and False):
        return Proposal("WAIT_FOR_MORE_EVIDENCE", best.cam_id, 0.6, int((min_shot - shot_age) * 1000) + 50, best.codes + ["MIN_SHOT_DURATION"], alternatives, ws.version,
                        f"{best.cam_id} preferred (+{gain:.2f}); shot only {shot_age:.1f}s old")
    if rapid >= MAX_CUTS_PER_10S:
        return Proposal("HOLD_CURRENT", current, 0.6, 2000, ["RAPID_CUT_GUARD"] + best.codes, alternatives, ws.version, f"{rapid} cuts in 10 s")
    obs = ws.cameras.get(best.cam_id)
    if obs is None or now - obs.at > 2.0:
        if roles.get(best.cam_id) != "wide":
            return Proposal("WAIT_FOR_MORE_EVIDENCE", best.cam_id, 0.4, 500, best.codes + ["STALE_OBSERVATION"], alternatives, ws.version, "waiting for a fresh look at the candidate")
    conf = min(1.0, 0.5 + gain)
    decision = "CUT_TO_CAMERA" if ws.mode == "AUTO" else "SUGGEST_CAMERA"
    hold_ms = 1500 if is_reaction else 2500
    return Proposal(decision, best.cam_id, conf, hold_ms, best.codes, alternatives, ws.version,
                    f"{best.cam_id} +{gain:.2f} over {current}: " + ", ".join(best.codes[:3]))
