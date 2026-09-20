"""Spec §13 scenarios against the editorial planner, on synthetic world states (no video, no LLM)."""
from server.planner import plan
from server.worldstate import CameraObservation, PersonObs, WorldState


def person(pid, speaking=0.0, fw=0.2, motion=0.1, tid=None, conf=0.9):
    return PersonObs(tid or f"t-{pid}", pid, conf, (100, 100, 200, 200), fw, speaking, 0.0, motion, 0.0, 0.0)


def obs(cam, now, people, shot="MEDIUM_CLOSE_UP", motion=0.1, frozen=False, entrances=()):
    return CameraObservation(cam, now, shot, list(people), motion, frozen, 0.8, 0.5, list(entrances), [])


def world(now, live="C", live_since=None, speaking=None, mode="AUTO", audio=True):
    ws = WorldState(host_id="host")
    ws.update_health({"A": "wide", "B": "host", "C": "guest"}, {"A": True, "B": True, "C": True}, {})
    ws.update_camera(obs("A", now, [person("sarah", fw=0.06), person("host", fw=0.06)], shot="WIDE"))
    ws.update_camera(obs("B", now, [person("host", speaking=0.7 if speaking == "host" else 0.05)]))
    ws.update_camera(obs("C", now, [person("sarah", speaking=0.7 if speaking == "sarah" else 0.05)]))
    if audio:
        ws.update_audio(0.2, now - 2.0)
        ws.update_audio(0.2, now)
    ws.set_mode(mode, False)
    ws.record_cut(live, live_since if live_since is not None else now - 10, ["INITIAL"], "cue")
    return ws


def test_name_mentioned_but_current_speaker_stays():
    """Daniel-style: the host talks ABOUT Sarah while speaking; Sarah's camera must not win."""
    now = 100.0
    ws = world(now, live="B", speaking="host")
    ws.update_conv(current_speaker="host", speaker_conf=0.8, dialogue_act="EXPLANATION", addressed=[], references=["sarah"],
                   sentence_complete=False, salience=0.4, updated_at=now)
    p = plan(ws, now)
    assert p.decision == "HOLD_CURRENT" and p.camera_id == "B", p.to_json()
    assert "CURRENT_SPEAKER_VISIBLE" in p.reason_codes or "ACTIVE_SPEAKER" in p.reason_codes


def test_directed_question_without_a_name_prepares_the_addressee():
    now = 100.0
    ws = world(now, live="B", speaking="host")
    ws.update_conv(current_speaker="host", speaker_conf=0.8, dialogue_act="QUESTION", addressed=["sarah"], expected_next="sarah",
                   sentence_complete=True, salience=0.5, updated_at=now)
    # host stops talking, Sarah starts answering on C
    ws.update_audio(0.0, now + 0.4)
    ws.update_camera(obs("C", now + 1.0, [person("sarah", speaking=0.8)]))
    ws.update_audio(0.2, now + 1.0)
    p = plan(ws, now + 1.0)
    assert p.decision == "CUT_TO_CAMERA" and p.camera_id == "C", p.to_json()
    assert "ADDRESSEE_VISIBLE" in p.reason_codes and "ACTIVE_SPEAKER" in p.reason_codes


def test_backchannel_never_cuts():
    now = 100.0
    ws = world(now, live="C", speaking="host")   # host says "mm-hmm" while Sarah is on air
    ws.update_conv(current_speaker="host", speaker_conf=0.6, dialogue_act="BACKCHANNEL", backchannel=True, updated_at=now)
    p = plan(ws, now)
    assert p.decision == "HOLD_CURRENT" and p.camera_id == "C" and "BACKCHANNEL" in p.reason_codes


def test_emotional_answer_survives_host_acknowledgement():
    now = 100.0
    ws = world(now, live="C", speaking="host")
    ws.update_conv(current_speaker="sarah", speaker_conf=0.9, dialogue_act="ANSWER", salience=0.9, sentence_complete=False, updated_at=now)
    p = plan(ws, now)
    assert p.decision == "HOLD_CURRENT" and p.camera_id == "C" and "EMOTIONAL_MOMENT" in p.reason_codes, p.to_json()


def test_overlapping_speech_goes_wide():
    now = 100.0
    ws = world(now, live="C", speaking="sarah")
    ws.update_conv(current_speaker="sarah", speaker_conf=0.4, overlap=True, dialogue_act="REBUTTAL", updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "A" and p.decision == "CUT_TO_CAMERA" and "OVERLAP_WIDE" in p.reason_codes, p.to_json()


def test_active_camera_failure_uses_safety_shot():
    now = 100.0
    ws = world(now, live="C", speaking="sarah")
    ws.update_health({"A": "wide", "B": "host", "C": "guest"}, {"A": True, "B": True, "C": False}, {})
    p = plan(ws, now)
    assert p.decision == "USE_SAFETY_SHOT" and p.camera_id == "A" and "CAMERA_UNHEALTHY" in p.reason_codes


def test_operator_hold_and_manual_mode_block_everything():
    now = 100.0
    ws = world(now, live="B", speaking="sarah")
    ws.set_mode("AUTO", True)
    assert plan(ws, now).decision == "HOLD_CURRENT"
    ws.set_mode("MANUAL", False)
    assert plan(ws, now).reason_codes == ["MANUAL_MODE"]


def test_assist_mode_suggests_instead_of_cutting():
    now = 100.0
    ws = world(now, live="B", speaking="sarah", mode="ASSIST")
    ws.update_conv(current_speaker="sarah", speaker_conf=0.9, dialogue_act="ANSWER", updated_at=now)
    p = plan(ws, now)
    assert p.decision == "SUGGEST_CAMERA" and p.camera_id == "C"


def test_min_shot_and_rapid_cut_guards():
    now = 100.0
    ws = world(now, live="B", live_since=now - 1.0, speaking="sarah")
    ws.update_conv(current_speaker="sarah", speaker_conf=0.9, dialogue_act="ANSWER", updated_at=now)
    p = plan(ws, now)
    assert p.decision == "WAIT_FOR_MORE_EVIDENCE" and "MIN_SHOT_DURATION" in p.reason_codes
    ws = world(now, live="B", speaking="sarah")
    for t in (now - 9, now - 6, now - 3):
        ws.cut_times.append(t)
    ws.update_conv(current_speaker="sarah", speaker_conf=0.9, dialogue_act="ANSWER", updated_at=now)
    assert "RAPID_CUT_GUARD" in plan(ws, now).reason_codes


def test_applause_line_goes_to_the_audience_camera_or_wide():
    now = 100.0
    ws = world(now, live="B", speaking="host")
    ws.update_conv(current_speaker="host", dialogue_act="AUDIENCE", subject="audience", salience=0.6, updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "A" and "SUBJECT_AUDIENCE" in p.reason_codes
    ws.update_health({"A": "wide", "B": "host", "C": "guest", "D": "audience"}, {"A": True, "B": True, "C": True, "D": True}, {})
    ws.update_camera(obs("D", now, [], shot="WIDE"))
    p = plan(ws, now)
    assert p.camera_id == "D" and p.decision == "CUT_TO_CAMERA"


def test_demonstration_prefers_demo_or_wide_over_the_talking_face():
    now = 100.0
    ws = world(now, live="C", speaking="sarah")
    ws.update_conv(current_speaker="sarah", dialogue_act="DEMONSTRATION", subject="object", salience=0.6, updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "A" and "SUBJECT_DEMONSTRATION_WIDE" in p.reason_codes


def test_look_at_the_tv_cuts_to_the_demo_camera_when_there_is_one():
    from server.dialogue import analyze_rules
    from server.semantics import Roster
    now = 100.0
    conv = analyze_rules("Let's look at the TV.", None, Roster([]), None)  # the rules alone, no LLM
    assert conv["subject"] == "screen" and conv["dialogue_act"] == "DEMONSTRATION"
    ws = world(now, live="B", speaking="host")
    ws.update_health({"A": "wide", "B": "host", "C": "guest", "D": "demo"}, {"A": True, "B": True, "C": True, "D": True}, {})
    ws.update_camera(obs("D", now, [], shot="WIDE"))
    ws.update_conv(current_speaker="host", dialogue_act=conv["dialogue_act"], subject=conv["subject"], salience=conv["salience"], updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "D" and p.decision == "CUT_TO_CAMERA" and "SUBJECT_DEMONSTRATION" in p.reason_codes
    # the demo camera gone stale: the wide shot, never a guess
    ws.update_health({"A": "wide", "B": "host", "C": "guest", "D": "demo"}, {"A": True, "B": True, "C": True, "D": False}, {})
    p = plan(ws, now)
    assert p.camera_id == "A" and "SUBJECT_DEMONSTRATION_WIDE" in p.reason_codes


def test_unknown_identity_with_a_named_cue_falls_back_to_wide_never_a_guess():
    from server.semantics import Action, Cue, Intent, Scope, Temporal
    now = 100.0
    ws = world(now, live="B", speaking="host")
    ws.update_camera(obs("C", now, [person(None, tid="t-x")]))   # a face nobody can name
    cue = Cue(["daniel"], Scope.SINGLE, Intent.INTRODUCE, Temporal.NOW, Action.SHOW, "welcome", created_at=now)
    p = plan(ws, now, cue, 0.2)
    assert p.camera_id == "A" and "CUE_TARGET_NOT_CONFIRMED_WIDE" in p.reason_codes


def test_stale_state_version_is_visible_to_the_executor():
    now = 100.0
    ws = world(now, live="B", speaking="sarah")
    ws.update_conv(current_speaker="sarah", speaker_conf=0.9, dialogue_act="ANSWER", updated_at=now)
    p = plan(ws, now)
    v = ws.version
    ws.update_audio(0.0, now + 0.1)   # the world moved on
    assert p.state_version == v and ws.version != p.state_version


def test_reaction_shot_short_window_only():
    now = 100.0
    ws = world(now, live="C", speaking="sarah")
    ws.update_conv(current_speaker="sarah", speaker_conf=0.9, dialogue_act="JOKE", salience=0.7, sentence_complete=True, updated_at=now)
    ws.update_camera(obs("B", now, [person("host", speaking=0.05, motion=0.6)]))
    ws.update_audio(0.0, now + 0.3)  # punchline pause
    ws.update_audio(0.0, now + 0.9)
    p = plan(ws, now + 0.9)
    # the speaker paused and the host visibly reacts: reaction scores, but only wins if the gain is real
    codes = {a["cameraId"]: a["codes"] for a in p.alternatives}
    assert "REACTION" in codes["B"]
    p2 = plan(ws, now + 5.0)     # window closed
    codes2 = {a["cameraId"]: a["codes"] for a in p2.alternatives}
    assert "REACTION" not in codes2["B"]


def test_look_at_the_flowers_cuts_to_the_camera_whose_tags_show_flowers():
    from server.dialogue import analyze_rules
    from server.semantics import Roster
    now = 100.0
    conv = analyze_rules("Let's look at the flowers.", None, Roster([]), None)
    assert conv["subject_phrase"] == "flowers" and conv["subject"] == "object"
    ws = world(now, live="B", speaking="host")
    ws.update_scene("A", {"objects": ["stairs", "table", "bottle"], "people": 1, "setting": "living room", "at": now - 30})
    ws.update_scene("B", {"objects": ["laptop", "books", "candle"], "people": 1, "setting": "living room", "at": now - 30})
    ws.update_scene("C", {"objects": ["chair", "flowers", "tv"], "people": 1, "setting": "living room", "at": now - 30})
    ws.update_conv(current_speaker="host", dialogue_act=conv["dialogue_act"], subject=conv["subject"], subject_phrase=conv["subject_phrase"],
                   salience=conv["salience"], updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "C" and p.decision == "CUT_TO_CAMERA" and "SUBJECT_SEEN_ON_CAMERA" in p.reason_codes, p.to_json()
    # two cameras show flowers: the wide shot, never a guess
    ws.update_scene("B", {"objects": ["laptop", "flowers"], "people": 1, "setting": "living room", "at": now - 30})
    p = plan(ws, now)
    assert p.camera_id == "A" and "SUBJECT_ON_SEVERAL_CAMERAS_WIDE" in p.reason_codes and "SUBJECT_SEEN_ON_CAMERA" not in p.reason_codes
    # nothing shows a piano: the demonstration fallback, the wide shot
    ws.update_conv(subject_phrase="piano", updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "A" and "SUBJECT_DEMONSTRATION_WIDE" in p.reason_codes
    # stale tags are ignored
    ws.update_conv(subject_phrase="flowers", updated_at=now)
    ws.update_scene("C", {"objects": ["chair", "flowers", "tv"], "people": 1, "setting": "living room", "at": now - 5000})
    ws.update_scene("B", {"objects": ["laptop"], "people": 1, "setting": "living room", "at": now - 30})
    assert "SUBJECT_SEEN_ON_CAMERA" not in plan(ws, now).reason_codes


def test_look_at_the_audience_prefers_the_audience_role_then_the_camera_showing_a_crowd():
    from server.dialogue import analyze_rules
    from server.semantics import Roster
    now = 100.0
    conv = analyze_rules("Let's look at the audience.", None, Roster([]), None)
    ws = world(now, live="B", speaking="host")
    ws.update_scene("C", {"objects": ["chairs", "crowd"], "people": 6, "setting": "hall", "at": now - 10})
    ws.update_conv(current_speaker="host", dialogue_act=conv["dialogue_act"], subject=conv["subject"], subject_phrase=conv["subject_phrase"],
                   salience=conv["salience"], updated_at=now)
    p = plan(ws, now)
    assert p.camera_id == "C" and "SUBJECT_SEEN_ON_CAMERA" in p.reason_codes
    ws.update_health({"A": "wide", "B": "host", "C": "guest", "D": "audience"}, {"A": True, "B": True, "C": True, "D": True}, {})
    ws.update_camera(obs("D", now, [], shot="WIDE"))
    p = plan(ws, now)
    assert p.camera_id == "D" and "SUBJECT_AUDIENCE" in p.reason_codes   # the operator's audience camera outranks the tags
