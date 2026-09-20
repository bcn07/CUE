"""server/scene.py: tags from the vision model's answer, matching a spoken phrase to the camera that
shows it, and the background tagger over a fake model (no network)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from server import scene as sc


def test_parse_tags_from_fenced_json_and_garbage():
    info = sc.parse_tags('```json\n{"people": 3, "objects": ["Flowers", "TV", "books", "flowers"], "setting": "Living Room"}\n```')
    assert info == {"objects": ["flower", "tv", "book"], "people": 3, "setting": "living room"}
    assert sc.parse_tags("The image shows a man in a room.") is None
    assert sc.parse_tags('{"objects": "nope", "people": "x"}') == {"objects": [], "people": 0, "setting": ""}


def test_match_scores_exact_synonym_partial_and_people():
    host = {"objects": ["laptop", "book", "remote control", "candle", "flower"], "people": 2, "setting": "living room"}
    wide = {"objects": ["wall", "ceiling light", "wall art", "plant"], "people": 1, "setting": "living room"}
    guest = {"objects": ["chair", "couch", "microwave", "fire extinguisher"], "people": 1, "setting": "kitchen"}
    assert sc.match_score("flowers", host) == 1.0 and sc.match_score("the lovely flowers", host) == 1.0
    assert sc.match_score("flowers", wide) == 1.0          # plant is a synonym
    assert sc.match_score("flowers", guest) == 0.0
    assert sc.match_score("the remote", host) == 0.6        # inside "remote control"
    assert sc.match_score("the kitchen", guest) == 1.0 and sc.match_score("the kitchen", host) == 0.0
    assert sc.match_score("the TV", {"objects": ["television"], "people": 0, "setting": ""}) == 1.0
    assert sc.match_score("the audience", {"objects": ["chair"], "people": 5, "setting": "hall"}) == 0.6
    assert sc.match_score("the audience", {"objects": ["crowd"], "people": 0, "setting": ""}) == 1.0
    assert sc.match_score("", host) == 0.0 and sc.match_score("flowers", None) == 0.0


def test_best_camera_needs_one_clear_winner_and_fresh_tags():
    now = 1000.0
    scene = {"A": {"objects": ["stairs", "table"], "at": now - 5}, "B": {"objects": ["flower", "tv"], "at": now - 5}, "C": {"objects": ["laptop"], "at": now - 5}}
    healthy = {"A": True, "B": True, "C": True}
    assert sc.best_camera(scene, "the flowers", now, 600, healthy) == ("B", 1.0, ["B"])
    assert sc.best_camera(scene, "the piano", now, 600, healthy) == (None, 0.0, [])
    scene["C"]["objects"].append("flower")
    assert sc.best_camera(scene, "the flowers", now, 600, healthy) == (None, 1.0, ["B", "C"])   # two show it: no guess
    healthy["C"] = False
    assert sc.best_camera(scene, "the flowers", now, 600, healthy)[0] == "B"                     # an offline camera never wins
    scene["B"]["at"] = now - 700
    assert sc.best_camera(scene, "the flowers", now, 600, healthy)[0] is None                    # stale tags say nothing


class _Cam:
    def __init__(self, latest, healthy=True):
        self.latest = latest
        self._h = healthy

    def healthy(self, now, stale_s):
        return self._h


def _show(cams):
    from server.worldstate import WorldState
    events = []
    show = SimpleNamespace(cameras=cams, ws=WorldState(), pending_cue=None, s=SimpleNamespace(camera_stale_s=1.5),
                           log_event=lambda kind, text, **kw: events.append((kind, text)), mark_dirty=lambda: None, events=events)
    return show


def test_tagger_tags_the_oldest_camera_skips_pending_cues_and_survives_bad_answers(tmp_path):
    from pathlib import Path
    jpeg = (Path(__file__).parent / "fixtures" / "obama.jpg").read_bytes()
    show = _show({"A": _Cam(jpeg), "B": _Cam(jpeg), "C": _Cam(None), "D": _Cam(jpeg, healthy=False)})
    asked = []

    async def fake(model, small, prompt):
        asked.append((model, len(small) < len(jpeg), prompt))
        n = len(asked)
        if n == 2:
            return "I cannot see anything clearly."
        return '{"people": %d, "objects": ["flowers", "tv"], "setting": "living room"}' % n

    tg = sc.SceneTagger(show, "fake-vl", "http://127.0.0.1:1/v1", interval_s=5, max_width=200, call=fake)
    assert asyncio.run(tg.tag_next())["objects"] == ["flower", "tv"]
    assert tg.status["last_cam"] == "A" and asked[0][0] == "fake-vl" and asked[0][1] and "JSON" in asked[0][2]
    assert asyncio.run(tg.tag_next()) is None and tg.status["errors"] == 1 and "not the JSON" in tg.status["last_error"]  # B answered prose
    assert asyncio.run(tg.tag_next())["people"] == 3 and tg.status["last_cam"] == "B" and tg.status["last_error"] is None  # B again: still oldest
    assert asyncio.run(tg.tag_next()) is not None and tg.status["last_cam"] == "A"                                        # then A, the older of the two
    assert set(show.ws.scene) == {"A", "B"} and all(k in show.ws.scene["A"] for k in ("objects", "people", "setting", "at"))
    show.pending_cue = object()
    assert asyncio.run(tg.tag_next()) is None and tg.status["skipped_for_cue"] == 1 and len(asked) == 4
    show.pending_cue = None
    assert sum(1 for k, _ in show.events if k == "scene") == 2 + 1   # A, B first tags, and A's people count changed
    snap = show.ws.snapshot(time.monotonic())
    assert snap["scene"]["A"]["objects"] == ["flower", "tv"] and "age_s" in snap["scene"]["A"]


def test_tagger_reports_a_dead_model_and_keeps_going():
    from pathlib import Path
    jpeg = (Path(__file__).parent / "fixtures" / "obama.jpg").read_bytes()
    show = _show({"A": _Cam(jpeg)})

    async def dead(model, small, prompt):
        raise ConnectionError("refused")

    tg = sc.SceneTagger(show, "fake-vl", "http://127.0.0.1:1/v1", call=dead)
    assert asyncio.run(tg.tag_next()) is None and tg.status["errors"] == 1 and "refused" in tg.status["last_error"]
    assert show.ws.scene == {}
