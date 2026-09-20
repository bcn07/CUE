"""Pending-cue semantics of Show (in-process, no server): which later cues drop, keep or
replace a cue that could not be executed yet. These are the round-two review scenarios."""
import asyncio
import threading
import time

import pytest

from server import director as D
from server.app import Camera, Show
from server.config import load_settings
from server.identity import PresenceTracker
from server.semantics import Action, Cue, Intent, Roster, Person, Scope, Semantics, Temporal


def make_show(tmp_path):
    s = load_settings()
    s.data_dir = tmp_path
    show = Show(s)
    show.loop = asyncio.get_running_loop()
    show._loop_thread = threading.current_thread()
    show.roster = Roster([Person("host", "Joe Biden", ["Joe"], "host", True), Person("sarah", "Sarah Tan", ["Sarah"], "guest")])
    show.semantics = Semantics(show.roster, "", show.on_cue, llm=None)
    now = time.monotonic()
    for cid, role in (("A", "wide"), ("B", "host"), ("C", "guest")):
        cam = Camera(cid, f"Camera {cid}", role)
        cam.connected, cam.latest, cam.last_frame_mono = True, b"jpeg", now
        show.cameras[cid] = cam
    show.trackers["C"] = PresenceTracker(1, 1.5)
    show.trackers["C"].observe_ids(["sarah"], now, "test")
    show.trackers["B"] = PresenceTracker(1, 1.5)
    show.trackers["B"].observe_ids(["host"], now, "test")
    return show


def cue(action, targets, *, seq, utt, source="rules-fast", temporal=Temporal.NOW, intent=Intent.INTRODUCE, clause="x"):
    return Cue(list(targets), Scope.SINGLE if len(targets) == 1 else Scope.NONE, intent, temporal, action, "ev",
               utterance_id=utt, clause_text=clause, source=source, created_at=time.monotonic(), meta={"seq": seq})


def cleanup(show):
    if show._recheck_handle:
        show._recheck_handle.cancel()


@pytest.mark.asyncio
async def test_newer_satisfied_cue_clears_an_older_pending_cue(tmp_path):
    show = make_show(tmp_path)
    D.manual_take(show.state, "C", time.monotonic())            # Sarah is live on C
    await show.on_cue(cue(Action.HOST, ["host"], seq=1, utt="u1", clause="Thank you Sarah."))
    assert show.pending_cue is not None and show.pending_block == "min-shot"
    await show.on_cue(cue(Action.SHOW, ["sarah"], seq=2, utt="u2", clause="Sarah, what about latency?"))
    assert show.pending_cue is None, "the host moved on to Sarah, who is already live: nothing older may fire later"
    assert show.state.current_camera == "C"
    cleanup(show)


@pytest.mark.asyncio
async def test_deferral_in_a_new_utterance_drops_pending(tmp_path):
    show = make_show(tmp_path)
    D.manual_take(show.state, "A", time.monotonic())
    await show.on_cue(cue(Action.SHOW, ["sarah"], seq=1, utt="u1"))          # held by min shot
    assert show.pending_cue is not None
    await show.on_cue(cue(Action.HOLD, ["sarah"], seq=2, utt="u2", source="rules", temporal=Temporal.FUTURE, intent=Intent.MENTION,
                          clause="Actually Sarah joins us later."))
    assert show.pending_cue is None
    cleanup(show)


@pytest.mark.asyncio
async def test_llm_error_keeps_pending_but_llm_hold_on_same_clause_drops_it(tmp_path):
    show = make_show(tmp_path)
    D.manual_take(show.state, "A", time.monotonic())
    await show.on_cue(cue(Action.SHOW, ["sarah"], seq=1, utt="u1", clause="Please welcome Sarah."))
    assert show.pending_cue is not None
    await show.on_cue(cue(Action.HOLD, [], seq=1, utt="u1", source="error", temporal=Temporal.UNCERTAIN, intent=Intent.NONE, clause="Please welcome Sarah."))
    assert show.pending_cue is not None, "an LLM error is no information"
    await show.on_cue(cue(Action.HOLD, ["sarah"], seq=1, utt="u1", source="llm", temporal=Temporal.UNCERTAIN, intent=Intent.MENTION, clause="Please welcome Sarah."))
    assert show.pending_cue is None, "the LLM disagreed with the fast path on the same clause"
    cleanup(show)


@pytest.mark.asyncio
async def test_late_llm_verdict_for_an_older_clause_does_not_touch_a_newer_pending(tmp_path):
    show = make_show(tmp_path)
    D.manual_take(show.state, "A", time.monotonic())
    await show.on_cue(cue(Action.HOLD, [], seq=1, utt="u1", source="rules", temporal=Temporal.UNCERTAIN, intent=Intent.NONE, clause="That was a great talk."))
    await show.on_cue(cue(Action.SHOW, ["sarah"], seq=2, utt="u2", clause="Please welcome Sarah!"))
    assert show.pending_cue is not None
    await show.on_cue(cue(Action.HOLD, [], seq=1, utt="u1", source="llm", temporal=Temporal.UNCERTAIN, intent=Intent.NONE, clause="That was a great talk."))
    assert show.pending_cue is not None and show.pending_cue.target_ids == ["sarah"]
    cleanup(show)


@pytest.mark.asyncio
async def test_min_shot_pending_executes_when_the_shot_may_end(tmp_path):
    show = make_show(tmp_path)
    show.tun = D.Tunables(min_shot_s=0.6, cue_lifetime_s=3.0, identity_max_age_s=4.0)
    D.manual_take(show.state, "A", time.monotonic())
    await show.on_cue(cue(Action.SHOW, ["sarah"], seq=1, utt="u1"))
    assert show.state.current_camera == "A" and show.pending_block == "min-shot"
    await asyncio.sleep(0.9)
    assert show.state.current_camera == "C" and show.pending_cue is None
    cleanup(show)


@pytest.mark.asyncio
async def test_identity_blocked_pending_fires_on_new_evidence_not_on_a_timer(tmp_path):
    show = make_show(tmp_path)
    show.trackers["C"] = PresenceTracker(1, 1.5)               # Sarah not confirmed anywhere
    D.manual_take(show.state, "A", time.monotonic() - 10)
    await show.on_cue(cue(Action.SHOW, ["sarah"], seq=1, utt="u1"))
    assert show.state.current_camera == "A" and show.pending_block == "identity"
    await asyncio.sleep(0.3)
    assert show.state.current_camera == "A"
    show.trackers["C"].observe_ids(["sarah"], time.monotonic(), "test")
    show._identity_updated("C", False)
    assert show.state.current_camera == "C" and show.pending_cue is None
    cleanup(show)
