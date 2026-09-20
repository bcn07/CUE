"""server/desk_ws.py: the desk-UI adapter, pure parts, over synthetic /ui snapshots."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import desk_ws as dw

CAMS = {
    "A": {"id": "A", "label": "Camera A (wide)", "role": "wide", "fixed_person": None, "connected": True, "healthy": True, "fps": 14.9},
    "B": {"id": "B", "label": "Camera B", "role": "host", "fixed_person": None, "connected": True, "healthy": True, "fps": 15.2},
    "C": {"id": "C", "label": "Camera C", "role": "guest", "fixed_person": "sarah", "connected": False, "healthy": False, "fps": 0},
}


def snap(**over):
    base = {
        "type": "state", "decision_ver": 3, "mode": "AUTO",
        "program": {"camera_id": "B"},
        "cameras": CAMS,
        "roster": [{"id": "host", "name": "Brian", "aliases": [], "role": "host", "is_host": True},
                   {"id": "sarah", "name": "Sarah Tan", "aliases": ["Sara"], "role": "guest of honour", "is_host": False}],
        "transcript": [{"utterance_id": "u1", "text": "Welcome everyone.", "final": True}],
        "interim": None,
        "last_cue": {"target_ids": ["sarah"], "scope": "single", "temporal_intent": "NOW", "action": "SHOW",
                     "clause_text": "Sarah, please come up now.", "evidence_text": "please come up now", "latency_ms": 412.0},
        "last_decision": {"action": "TAKE", "camera_id": "C", "reason": "named guest is in frame", "evidence": "identity", "target": "sarah"},
        "pending_cue": None,
        "latency": {"decide_ms": {"n": 3, "last": 2.5, "p50": 2.0, "p95": 3.0}},
        "services": {"deepgram": {"configured": True, "connected": True, "model": "nova-3", "mic_level": 0.21}},
    }
    base.update(over)
    return base


CMAP = dw.camera_map(CAMS)


def test_camera_map_by_role():
    assert CMAP == {"A": "CAM-WIDE", "B": "CAM-HOST", "C": "CAM-GUEST"}


def test_camera_map_prefers_the_camera_that_is_actually_live_per_role():
    # the real case: B is a stale offline "guest" entry, the phone joined as D, also "guest"
    cams = {"A": {"role": "wide", "connected": False, "healthy": False}, "B": {"role": "guest", "connected": False, "healthy": False},
            "C": {"role": "host", "connected": False, "healthy": False}, "D": {"role": "guest", "connected": True, "healthy": True}}
    assert dw.camera_map(cams) == {"A": "CAM-WIDE", "C": "CAM-HOST", "D": "CAM-GUEST", "B": "CAM-GUEST-2"}
    assert dw.rewrite_frame(b"\x01D" + b"jpeg", dw.camera_map(cams)) == b"\tCAM-GUEST" + b"jpeg"
    assert dw.rewrite_frame(b"\x01B" + b"jpeg", dw.camera_map(cams)) == b"\x0bCAM-GUEST-2" + b"jpeg"
    # connected beats offline, healthy beats connected, first listed breaks ties
    cams["B"]["connected"] = True
    assert dw.camera_map(cams)["D"] == "CAM-GUEST" and dw.camera_map(cams)["B"] == "CAM-GUEST-2"
    cams["B"]["healthy"] = True
    assert dw.camera_map(cams)["B"] == "CAM-GUEST" and dw.camera_map(cams)["D"] == "CAM-GUEST-2"
    # ...unless D already holds the tile and is still healthy: no swapping mid-show
    held = {"A": "CAM-WIDE", "C": "CAM-HOST", "D": "CAM-GUEST", "B": "CAM-GUEST-2"}
    assert dw.camera_map(cams, prev=held) == held
    cams["D"]["healthy"] = False
    assert dw.camera_map(cams, prev=held)["B"] == "CAM-GUEST" and dw.camera_map(cams, prev=held)["D"] == "CAM-GUEST-2"
    # roles the desk does not know get their own id; every camera is mapped
    cams["E"] = {"role": "audience", "connected": True, "healthy": True}
    m = dw.camera_map(cams)
    assert m["E"] == "CAM-AUDIENCE" and set(m) == set(cams)
    assert "CAM-AUDIENCE" in dw.camera_states({"cameras": cams}, m)


def test_mode_maps_hold_to_manual_and_everything_else_to_auto():
    assert dw.desk_mode("HOLD") == "manual"
    assert dw.desk_mode("AUTO") == "auto"
    assert dw.desk_mode(None) == "auto"


def test_decision_record_from_a_cue_driven_take():
    rec = dw.decision_record(snap(), CMAP)
    assert rec["decision_seq"] == 3 and rec["action"] == "TAKE" and rec["camera_id"] == "CAM-GUEST"
    assert rec["transcript_span"]["text"] == "Sarah, please come up now."
    assert rec["cue_summary"] == {"target_guest_ids": ["sarah"], "temporal_intent": "NOW", "scope": "single"}
    assert rec["latencies_ms"]["cue_decide_ms"] == 414.5
    assert "guest camera" in rec["plain_reason"]


def test_manual_take_does_not_borrow_the_previous_cue():
    rec = dw.decision_record(snap(last_decision={"action": "TAKE", "camera_id": "A", "reason": "manual take", "evidence": "manual"}), CMAP)
    assert rec["reason"].startswith("manual") and rec["camera_id"] == "CAM-WIDE"
    assert rec["transcript_span"]["text"] == "" and rec["cue_summary"]["target_guest_ids"] == []
    assert rec["latencies_ms"]["cue_decide_ms"] is None
    assert rec["plain_reason"] == "You took the wide shot."


def test_on_connect_sends_the_whole_contract_head():
    types = [m["type"] for m in dw.on_connect_messages(snap(), CMAP)]
    assert types[:6] == ["source", "directing", "caption_status", "deepgram_config", "roster", "mode"]
    assert types.count("camera_state") == 3 and types[-1] == "decision"
    msgs = {m["type"]: m for m in dw.on_connect_messages(snap(), CMAP)}
    assert msgs["source"]["source"] == "mic"
    assert msgs["deepgram_config"]["keyterms"] == ["Brian", "Sarah Tan", "Sara"]
    assert msgs["roster"]["guests"][0] == {"name": "Brian", "role": "Host"}
    # Sarah is the fixed person on camera C, the guest camera, so the desk tile can show her name.
    assert msgs["roster"]["guests"][1] == {"name": "Sarah Tan", "role": "guest of honour", "camera": "CAM-GUEST"}
    assert dw.on_connect_messages(snap(services={"deepgram": {"configured": False}}), CMAP)[0]["source"] == "fixture"


def test_diff_emits_captions_once_and_provisional_on_change():
    a = snap()
    b = snap(interim={"utterance_id": "u2", "text": "Sarah, please"},
             transcript=a["transcript"] + [{"utterance_id": "u2", "text": "Sarah, please come up now.", "final": True}])
    types = [m["type"] for m in dw.diff_messages(a, b, CMAP)]
    assert types.count("caption_final") == 1 and types.count("deepgram_result") == 1 and "caption_provisional" in types
    assert "decision" not in types and "mode" not in types
    # The same snapshot again emits nothing but the mic level.
    assert [m["type"] for m in dw.diff_messages(b, b, CMAP)] == ["mic_level"]


def test_diff_emits_decision_mode_camera_state_prepare_and_stand_down():
    a = snap()
    b = snap(decision_ver=4, mode="HOLD",
             cameras={**CAMS, "C": {**CAMS["C"], "connected": True, "healthy": True, "fps": 12.0}},
             pending_cue={"action": "SHOW", "target_ids": ["sarah"]})
    msgs = dw.diff_messages(a, b, CMAP)
    types = [m["type"] for m in msgs]
    assert "decision" in types and "mode" in types and "directing" in types
    assert {m["camera"] for m in msgs if m["type"] == "camera_state"} == {"CAM-GUEST"}
    assert next(m for m in msgs if m["type"] == "prepare")["camera"] == "CAM-GUEST"
    assert "stand_down" in [m["type"] for m in dw.diff_messages(b, snap(decision_ver=4, mode="HOLD"), CMAP)]


def test_frame_header_is_rewritten_to_the_desk_id_and_unknown_cameras_are_dropped():
    jpeg = b"\xff\xd8\xff\xd9"
    out = dw.rewrite_frame(bytes([1]) + b"C" + jpeg, CMAP)
    assert out == bytes([9]) + b"CAM-GUEST" + jpeg
    assert dw.rewrite_frame(bytes([1]) + b"Z" + jpeg, CMAP) is None


def test_desk_messages_map_to_control_actions():
    assert dw.desk_to_control({"type": "manual_take", "camera": "CAM-WIDE"}, CMAP) == ("take", "A")
    assert dw.desk_to_control({"type": "set_mode", "mode": "auto"}, CMAP) == ("auto", None)
    assert dw.desk_to_control({"type": "set_mode", "mode": "assist"}, CMAP) == ("hold", None)
    assert dw.desk_to_control({"type": "rate", "decision_seq": 3, "right": True}, CMAP) is None
    with pytest.raises(HTTPException):
        dw.desk_to_control({"type": "manual_take", "camera": "CAM-NOPE"}, CMAP)


def test_desk_client_translates_state_and_frames():
    class FakeWS:
        def __init__(self):
            self.texts, self.bins = [], []

        async def send_text(self, t):
            self.texts.append(t)

        async def send_bytes(self, b):
            self.bins.append(b)

    import asyncio, json

    async def run():
        ws = FakeWS()
        client = dw.DeskClient(ws)
        client.offer_state(json.dumps(snap()))
        client.offer("C", bytes([1]) + b"C" + b"jpeg")
        client.offer("Z", bytes([1]) + b"Z" + b"jpeg")
        task = asyncio.create_task(client.sender())
        await asyncio.sleep(0.05)
        client.alive = False
        client.wake.set()
        await asyncio.wait_for(task, 1)
        return ws

    ws = asyncio.run(run())
    kinds = [json.loads(t)["type"] for t in ws.texts]
    assert "decision" in kinds and "camera_state" in kinds and "mic_level" in kinds
    assert ws.bins == [bytes([9]) + b"CAM-GUEST" + b"jpeg"]


def test_held_slate_is_not_an_operator_take():
    rec = dw.decision_record(snap(last_decision={"action": "SLATE", "camera_id": None, "reason": "manual HOLD; no healthy camera -> slate", "evidence": "health"}), CMAP)
    assert rec["action"] == "SLATE" and rec["camera_id"] is None
    assert not rec["reason"].lower().startswith("manual")
    assert rec["plain_reason"] == "No usable camera, so the safe picture."
    stay = dw.decision_record(snap(last_decision={"action": "STAY", "camera_id": "B", "reason": "manual HOLD", "evidence": "hold"}), CMAP)
    assert stay["plain_reason"] == "Automatic cuts are paused." and not stay["reason"].lower().startswith("manual")


SUG = {"camera_id": "C", "confidence": 0.81, "reason_codes": ["ADDRESSED", "FACE_VISIBLE"],
       "explanation": "Sarah was just addressed and her camera is ready.", "origin": "cue", "at_wall": 1.0, "state_version": 7}


def test_assist_director_maps_assist_accept_and_skip_straight_through():
    assert dw.desk_to_control({"type": "set_mode", "mode": "assist"}, CMAP, assist_supported=True) == ("assist", None)
    assert dw.desk_to_control({"type": "accept_suggestion", "decision_seq": 7}, CMAP, assist_supported=True) == ("accept", None)
    assert dw.desk_to_control({"type": "skip_suggestion", "decision_seq": 7}, CMAP, assist_supported=True) == ("skip", None)
    # Older director: assist is HOLD, accept and skip do nothing.
    assert dw.desk_to_control({"type": "set_mode", "mode": "assist"}, CMAP) == ("hold", None)
    assert dw.desk_to_control({"type": "accept_suggestion", "decision_seq": 7}, CMAP) is None
    assert dw.supports_assist(snap()) is False and dw.supports_assist(snap(assist=False)) is True


def test_assist_snapshot_reports_assist_mode_and_a_suggestion_card():
    s = snap(assist=True, suggestion=SUG)
    msgs = dw.on_connect_messages(s, CMAP)
    assert next(m for m in msgs if m["type"] == "mode")["mode"] == "assist"
    assert next(m for m in msgs if m["type"] == "directing")["enabled"] is False
    assert next(m for m in msgs if m["type"] == "prepare")["camera"] == "CAM-GUEST"
    cards = [m["record"] for m in msgs if m["type"] == "decision" and m["record"]["reason"].startswith("suggest")]
    assert len(cards) == 1
    card = cards[0]
    assert card["camera_id"] == "CAM-GUEST" and card["decision_seq"] == 7 and "accepted" not in card
    assert card["plain_reason"] == "Sarah was just addressed and her camera is ready."
    assert card["cue_summary"]["target_guest_ids"] == ["sarah"]


def test_suggestion_appearing_and_clearing_emits_prepare_and_stand_down_once_each():
    a = snap(assist=True, suggestion=None)
    b = snap(assist=True, suggestion=SUG)
    types = [m["type"] for m in dw.diff_messages(a, b, CMAP)]
    assert types.count("prepare") == 1 and types.count("decision") == 1
    assert [m["type"] for m in dw.diff_messages(b, b, CMAP)] == ["mic_level"]
    assert "stand_down" in [m["type"] for m in dw.diff_messages(b, a, CMAP)]
    # A changed suggestion (new state_version) is a new card.
    c = snap(assist=True, suggestion={**SUG, "camera_id": "A", "state_version": 8})
    assert next(m for m in dw.diff_messages(b, c, CMAP) if m["type"] == "prepare")["camera"] == "CAM-WIDE"


def test_roster_is_resent_when_a_camera_assignment_changes():
    a = snap()
    b = snap(cameras={**CAMS, "A": {**CAMS["A"], "fixed_person": "sarah"}, "C": {**CAMS["C"], "fixed_person": None}})
    rosters = [m for m in dw.diff_messages(a, b, CMAP) if m["type"] == "roster"]
    assert len(rosters) == 1
    assert next(g for g in rosters[0]["guests"] if g["name"] == "Sarah Tan")["camera"] == "CAM-WIDE"
    assert not [m for m in dw.diff_messages(b, b, CMAP) if m["type"] == "roster"]


def test_manual_take_reaches_the_desk_as_a_decision(tmp_path):
    """A take from the desk must come back as a decision, or the page keeps showing the old camera."""
    from pathlib import Path
    from server.app import Camera, Show
    from server.config import Settings
    s = Settings(data_dir=tmp_path / "data", models_dir=Path(__file__).resolve().parent.parent / "models", static_dir=Path(__file__).resolve().parent.parent / "server" / "static")
    s.data_dir.mkdir(parents=True)
    show = Show(s)
    for cid, role in (("A", "wide"), ("B", "host"), ("C", "guest")):
        show.cameras[cid] = Camera(cid, f"Camera {cid}", role=role)
        show.cameras[cid].connected = True
    show.state.current_camera = "A"
    before = show.snapshot()
    cmap = dw.camera_map(before["cameras"])
    show.control("take", "C")
    after = show.snapshot()
    assert after["program"]["camera_id"] == "C" and after["decision_ver"] == before["decision_ver"] + 1
    msgs = [m for m in dw.diff_messages(before, after, cmap) if m["type"] == "decision"]
    assert len(msgs) == 1 and msgs[0]["record"]["camera_id"] == "CAM-GUEST" and msgs[0]["record"]["reason"].startswith("manual")
    # the same camera again: nothing changes, nothing is announced
    show.control("take", "C")
    again = show.snapshot()
    assert again["decision_ver"] == after["decision_ver"] and [m for m in dw.diff_messages(after, again, cmap) if m["type"] == "decision"] == []


def test_cameras_message_lists_existing_tiles_in_desk_order_and_only_when_changed():
    import json as _json
    cams = {"A": {"role": "audience", "connected": True, "healthy": True}, "B": {"role": "audience", "connected": False, "healthy": False},
            "D": {"role": "audience", "connected": True, "healthy": True}, "C": {"role": "guest", "connected": True, "healthy": True}}
    assert dw.cameras_message(dw.camera_map(cams)) == {"type": "cameras", "cameras": ["CAM-GUEST", "CAM-AUDIENCE", "CAM-AUDIENCE-2", "CAM-AUDIENCE-3"]}

    class WS:  # never sent to: offer_state only queues text
        pass

    client = dw.DeskClient(WS())
    client.offer_state(_json.dumps(snap(cameras=cams)))
    first = [_json.loads(m) for m in client.texts]
    assert [m for m in first if m["type"] == "cameras"] == [dw.cameras_message(dw.camera_map(cams))]
    client.texts.clear()
    client.offer_state(_json.dumps(snap(cameras=cams)))          # nothing changed: not repeated
    assert [m for m in map(_json.loads, client.texts) if m["type"] == "cameras"] == []
    cams["A"]["role"] = "wide"; cams["B"]["role"] = "host"; del cams["D"]   # roles fixed on /setup, a camera removed
    client.texts.clear()
    client.offer_state(_json.dumps(snap(cameras=cams)))
    msgs = [m for m in map(_json.loads, client.texts) if m["type"] == "cameras"]
    assert msgs == [{"type": "cameras", "cameras": ["CAM-HOST", "CAM-GUEST", "CAM-WIDE"]}]   # the audience tiles are gone
