"""server/signup_sync.py: sheet sign-ups enrol themselves, over a fake Apps Script web app that
speaks Code.gs's contract (roster through a 302, Drive-style thumbnails, a sign-in page when the
photos are private)."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from server import signup_sync as ss
from server.app import Camera, Show
from server.config import Settings
from server.identity import models_present

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"


class FakeScript:
    def __init__(self, photo: bytes):
        self.participants: list[dict] = []
        self.photo = photo
        self.private = False   # Drive answers with its sign-in page instead of the image
        self.down = False      # 500 on everything
        self.hits: list[tuple[str, dict]] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D102
                pass

            def _send(self, code, ctype, body, extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                fake.hits.append((u.path, q))
                if fake.down:
                    return self._send(500, "text/plain", b"boom")
                if u.path == "/exec":
                    if q.get("action") != "roster":
                        return self._send(200, "application/json", json.dumps({"ok": False, "error": "unknown action"}).encode())
                    return self._send(302, "text/html", b"", {"Location": "/exec-out?" + u.query})  # like script.google.com
                if u.path == "/exec-out":
                    rows = [p for p in fake.participants if not q.get("eventId") or p.get("event") == q["eventId"]]
                    return self._send(200, "application/json", json.dumps({"ok": True, "sheetUrl": "https://sheets.example/x", "participants": rows}).encode())
                if u.path == "/thumbnail":
                    if fake.private:
                        return self._send(200, "text/html; charset=utf-8", b"<html>Sign in</html>")
                    return self._send(200, "image/jpeg", fake.photo)
                return self._send(404, "text/plain", b"nope")

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/exec"

    def thumb(self, fid: str) -> str:
        return f"http://127.0.0.1:{self.port}/thumbnail?id={fid}&sz=w320"

    def add(self, gid, name, role="", camera="", event="hackmit-demo", consent=True, photos=True):
        self.participants.append({"guestId": gid, "name": name, "role": role, "camera": camera, "event": event,
                                  "front": self.thumb(gid + "-front") if photos else "", "side": self.thumb(gid + "-side") if photos else "",
                                  "consent": consent, "submittedAt": "2026-09-20T12:00:00.000Z"})

    def thumb_hits(self):
        return [h for h in self.hits if h[0] == "/thumbnail"]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def script():
    fs = FakeScript((FX / "obama.jpg").read_bytes())
    yield fs
    fs.close()


def make_show(tmp_path: Path) -> Show:
    s = Settings(data_dir=tmp_path / "data", models_dir=MODELS, static_dir=ROOT / "server" / "static")
    s.data_dir.mkdir(parents=True, exist_ok=True)
    show = Show(s)
    for cid, label, role in (("A", "Camera A (wide)", "wide"), ("B", "Camera B", "host"), ("C", "Camera C", "guest")):
        show.cameras[cid] = Camera(cid, label, role=role)
    return show


def run(coro):
    return asyncio.run(coro)


def sync_for(show, script, **kw):
    return ss.SignupSync(show, script.endpoint, poll_s=2, **kw)


def people_dir(tmp_path, pid):
    return tmp_path / "data" / "people" / pid


# ------------------------------------------------------------------ pure

def test_read_config_js(tmp_path):
    p = tmp_path / "config.js"
    p.write_text("/* endpoint: 'in a comment' */\nwindow.CUE_SIGNUP = {\n  endpoint: 'https://script.google.com/macros/s/abc/exec',\n"
                 "  eventId: \"hackmit-demo\",\n  signupBase: '',\n  pollMs: 5000,\n};\n")
    assert ss.read_config_js(p) == {"endpoint": "https://script.google.com/macros/s/abc/exec", "eventId": "hackmit-demo"}
    assert ss.read_config_js(tmp_path / "missing.js") == {}


def test_urls_and_scrubbing():
    assert ss.roster_url("https://x/exec") == "https://x/exec?action=roster"
    assert ss.roster_url("https://x/exec", "ev 1") == "https://x/exec?action=roster&eventId=ev+1"
    assert ss.larger_thumb("https://drive.google.com/thumbnail?id=F&sz=w320") == "https://drive.google.com/thumbnail?id=F&sz=w1024"
    assert ss.larger_thumb("https://drive.google.com/file/d/F/view") == "https://drive.google.com/file/d/F/view"
    out = ss.scrub("Client error '404' for url 'https://script.google.com/macros/s/SECRET/exec?action=roster'")
    assert "SECRET" not in out and "<url>" in out


def test_camera_for_desk_by_role():
    cams = {"A": Camera("A", "wide", role="wide"), "B": Camera("B", "b", role="host"), "C": Camera("C", "c", role="guest")}
    assert ss.camera_for_desk(cams, "CAM-GUEST") == "C" and ss.camera_for_desk(cams, "cam-host") == "B"
    assert ss.camera_for_desk(cams, "") is None and ss.camera_for_desk(cams, "CAM-NOPE") is None
    assert ss.camera_for_desk({"A": Camera("A", "a", role="demo")}, "CAM-WIDE") is None


# ------------------------------------------------------------------ enrol

def test_new_guest_is_enrolled_with_photos_under_the_sheet_id(tmp_path, script):
    show = make_show(tmp_path)
    script.add("sarah-tan", "Sarah  Tan ", role="Guest of honour")
    sync = sync_for(show, script)
    r = run(sync.poll_once())
    assert r["ok"] and r["added"] == ["sarah-tan"] and r["adopted"] == [] and r["photos"] == 2
    p = show.roster.by_id["sarah-tan"]
    assert (p.name, p.role, p.is_host, p.aliases) == ("Sarah Tan", "Guest of honour", False, [])
    photos = sorted(people_dir(tmp_path, "sarah-tan").iterdir())
    assert len(photos) == 2 and {f.suffix for f in photos} == {".jpg"}
    assert [q.get("sz") for _, q in script.thumb_hits()] == ["w1024", "w1024"]  # the larger thumbnail, first try
    assert json.loads((tmp_path / "data" / "roster.json").read_text())["people"][0]["id"] == "sarah-tan"
    st = json.loads(sync.state_path.read_text())["guests"]["sarah-tan"]
    assert st["pid"] == "sarah-tan" and st["photos"] == 2
    assert (sync.status["enrolled"], sync.status["participants"], sync.status["photos_pending"], sync.status["last_error"]) == (1, 1, 0, None)
    assert any("enrolled Sarah Tan" in e["text"] for e in show.events)
    # idempotent: the same rows again fetch nothing and add nothing
    n = len(script.hits)
    r2 = run(sync.poll_once())
    assert r2["added"] == [] and r2["photos"] == 0 and [h for h in script.hits[n:] if h[0] == "/thumbnail"] == []


@pytest.mark.skipif(not models_present(MODELS), reason="face models missing")
def test_sheet_photos_enrol_a_face(tmp_path, script):
    from server.identity import FaceEngine
    show = make_show(tmp_path)
    show.engine = FaceEngine(MODELS)
    script.add("barack", "Barack Obama")
    run(sync_for(show, script).poll_once())
    assert show.enroll_report["barack"]["faces"] >= 1 and show.gallery.size >= 1
    assert show.snapshot()["roster"][0]["faces"] >= 1


def test_same_name_already_enrolled_is_adopted_not_duplicated(tmp_path, script):
    show = make_show(tmp_path)
    show.upsert_person("Sarah Tan", ["Sara"], "", True, person_id="sarah")  # the operator enrolled her, as the host
    script.add("sarah-tan", "sarah tan", role="Guest of honour")
    r = run(sync_for(show, script).poll_once())
    assert r["adopted"] == ["sarah"] and r["added"] == []
    assert [p.id for p in show.roster.people] == ["sarah"]
    p = show.roster.by_id["sarah"]
    assert p.is_host is True and p.aliases == ["Sara"] and p.role == "Guest of honour"  # empty role filled, nothing else touched
    assert len(list(people_dir(tmp_path, "sarah").iterdir())) == 2  # her sheet photos went to her folder


def test_host_never_comes_from_the_sheet(tmp_path, script):
    show = make_show(tmp_path)
    script.add("brian", "Brian N", role="host")
    run(sync_for(show, script).poll_once())
    assert show.roster.by_id["brian"].is_host is False and show.roster.host_id is None


def test_rows_without_consent_or_id_or_name_are_ignored(tmp_path, script):
    show = make_show(tmp_path)
    script.add("nope", "No Consent", consent=False)
    script.add("", "No Id")
    script.add("x", "N")
    script.participants.append("garbage")
    r = run(sync_for(show, script).poll_once())
    assert r["ok"] and show.roster.people == []


def test_event_id_filters_rows(tmp_path, script):
    show = make_show(tmp_path)
    script.add("a", "Ann Lee", event="other")
    script.add("b", "Bo Chen", event="hackmit-demo")
    run(ss.SignupSync(show, script.endpoint, event_id="hackmit-demo", poll_s=2).poll_once())
    assert [p.id for p in show.roster.people] == ["b"]
    assert [q.get("eventId") for path, q in script.hits if path == "/exec"] == ["hackmit-demo"]


# ----------------------------------------------------------------- photos

def test_private_photos_enrol_the_name_and_retry_later(tmp_path, script):
    show = make_show(tmp_path)
    script.private = True
    script.add("sarah-tan", "Sarah Tan")
    sync = sync_for(show, script)
    r = run(sync.poll_once())
    assert r["added"] == ["sarah-tan"] and r["photos"] == 0
    assert not people_dir(tmp_path, "sarah-tan").exists() or not list(people_dir(tmp_path, "sarah-tan").iterdir())
    assert sync.guests["sarah-tan"]["photo_attempts"] == 1 and sync.status["photos_pending"] == 1
    script.private = False
    r = run(sync.poll_once())
    assert r["photos"] == 2 and sync.status["photos_pending"] == 0 and len(list(people_dir(tmp_path, "sarah-tan").iterdir())) == 2


def test_photo_attempts_are_bounded_and_reported(tmp_path, script):
    show = make_show(tmp_path)
    script.private = True
    script.add("g", "Guest One")
    sync = sync_for(show, script)
    for _ in range(ss.MAX_PHOTO_ATTEMPTS + 3):
        run(sync.poll_once())
    assert sync.guests["g"]["photo_attempts"] == ss.MAX_PHOTO_ATTEMPTS
    assert len(script.thumb_hits()) == ss.MAX_PHOTO_ATTEMPTS * 2 * 2  # front + side, larger then original, per attempt
    assert sum("not fetchable" in e["text"] for e in show.events) == 1
    assert sync.status["photos_pending"] == 0


def test_photos_added_on_setup_stop_the_fetching(tmp_path, script):
    show = make_show(tmp_path)
    script.private = True
    script.add("g", "Guest One")
    sync = sync_for(show, script)
    run(sync.poll_once())
    show.add_person_photos("g", [("me.jpg", (FX / "obama.jpg").read_bytes())])
    n = len(script.thumb_hits())
    run(sync.poll_once())
    assert len(script.thumb_hits()) == n and sync.guests["g"]["photos"] == 1


# ---------------------------------------------------------------- cameras

def test_sheet_camera_fills_an_empty_slot_but_never_overrides_the_operator(tmp_path, script):
    show = make_show(tmp_path)
    show.upsert_person("Some One", [], "", False, person_id="someone")
    show.cameras["C"].fixed_person = "someone"
    script.add("sarah-tan", "Sarah Tan", camera="CAM-GUEST")
    script.add("dan", "Daniel Ho", camera="CAM-HOST")
    sync = sync_for(show, script)
    r = run(sync.poll_once())
    assert show.cameras["C"].fixed_person == "someone"  # the operator's choice stands on first sight
    assert show.cameras["B"].fixed_person == "dan" and r["cameras"] == ["dan -> B"]
    assert json.loads((tmp_path / "data" / "cameras.json").read_text())["cameras"]["B"]["fixed_person"] == "dan"
    script.participants[0]["camera"] = "CAM-WIDE"  # the dashboard moves Sarah: a change is followed
    r = run(sync.poll_once())
    assert show.cameras["A"].fixed_person == "sarah-tan" and r["cameras"] == ["sarah-tan -> A"]
    script.participants[0]["camera"] = "CAM-HOST"  # moved again: she leaves A, Dan is replaced on B, and that is said
    r = run(sync.poll_once())
    assert show.cameras["A"].fixed_person is None and show.cameras["B"].fixed_person == "sarah-tan"
    assert r["cameras"] == ["sarah-tan -> B (replacing Daniel Ho)"]
    assert any("the sheet puts them on CAM-GUEST, but camera C is fixed to Some One here" in e["text"] for e in show.events)
    script.participants[0]["camera"] = ""  # unassigned on the sheet: cleared here
    run(sync.poll_once())
    assert show.cameras["B"].fixed_person is None
    n = len(show.events)
    run(sync.poll_once())  # nothing changed: nothing said
    assert len(show.events) == n


def test_unknown_desk_camera_is_reported_not_guessed(tmp_path, script):
    show = make_show(tmp_path)
    show.cameras["C"].role = "demo"
    script.add("sarah-tan", "Sarah Tan", camera="CAM-GUEST")
    sync = sync_for(show, script)
    run(sync.poll_once())  # first sight: no guest camera here, nothing to fill
    script.participants[0]["camera"] = "CAM-HOST"
    run(sync.poll_once())
    script.participants[0]["camera"] = "CAM-GUEST"
    run(sync.poll_once())
    assert show.cameras["B"].fixed_person is None and all(c.fixed_person != "sarah-tan" for c in show.cameras.values())
    assert any("no camera here has that role" in e["text"] for e in show.events)


# ----------------------------------------------------------------- safety

def test_unreachable_or_odd_sheet_changes_nothing_and_rows_vanishing_removes_nobody(tmp_path, script):
    show = make_show(tmp_path)
    script.add("sarah-tan", "Sarah Tan")
    sync = sync_for(show, script)
    run(sync.poll_once())
    script.down = True
    assert run(sync.poll_once()) == {"ok": False}
    assert "unreachable" in sync.status["last_error"] and sum("unreachable" in e["text"] for e in show.events) == 1
    run(sync.poll_once())  # same problem again: said once
    assert sum("unreachable" in e["text"] for e in show.events) == 1
    script.down = False
    script.participants.clear()  # the row was deleted (or purged) on the sheet
    r = run(sync.poll_once())
    assert r["ok"] and sync.status["last_error"] is None
    assert [p.id for p in show.roster.people] == ["sarah-tan"]
    bad = ss.SignupSync(show, script.endpoint + "?action=nothing", poll_s=2)
    run(bad.poll_once())
    assert "without a roster" in bad.status["last_error"]


def test_someone_removed_on_setup_is_not_brought_back(tmp_path, script):
    show = make_show(tmp_path)
    script.add("sarah-tan", "Sarah Tan")
    sync = sync_for(show, script)
    run(sync.poll_once())
    from server.semantics import Roster
    show.roster = Roster([])  # what DELETE /api/people/sarah-tan does to the roster
    show.save_roster()
    r = run(sync.poll_once())
    assert r["added"] == [] and show.roster.people == [] and sync.status["enrolled"] == 0
    assert any("removed on /setup" in e["text"] for e in show.events)


def test_state_survives_a_restart(tmp_path, script):
    show = make_show(tmp_path)
    script.add("sarah-tan", "Sarah Tan", camera="CAM-GUEST")
    run(sync_for(show, script).poll_once())
    show.cameras["C"].fixed_person = None  # the operator cleared it after the sheet was seen
    n = len(script.hits)
    again = sync_for(show, script)  # a fresh instance over the same data dir, as after a restart
    r = run(again.poll_once())
    assert r["added"] == [] and r["photos"] == 0 and r["cameras"] == []
    assert [h for h in script.hits[n:] if h[0] == "/thumbnail"] == [] and show.cameras["C"].fixed_person is None


def test_endpoint_never_leaks_into_status_events_or_logs(tmp_path, script, caplog):
    show = make_show(tmp_path)
    sync = ss.SignupSync(show, f"http://127.0.0.1:{script.port}/nope/SECRETTOKEN", poll_s=2)  # 404 puts the url in httpx's message
    with caplog.at_level("DEBUG"):  # the root: httpx's own "HTTP Request: GET <url>" lines count too
        run(sync.poll_once())
    assert sync.status["last_error"]
    blob = json.dumps(sync.status) + json.dumps(show.events) + caplog.text + json.dumps(show.snapshot()["signup"])
    assert "SECRETTOKEN" not in blob
    assert any(r.name == "httpx" and "<url>" in r.getMessage() for r in caplog.records)  # the request was logged, redacted


def test_snapshot_says_when_nothing_is_linked(tmp_path):
    assert make_show(tmp_path).snapshot()["signup"] == {"enabled": False}


def test_poll_loop_survives_a_bad_answer(tmp_path, script):
    show = make_show(tmp_path)
    calls = {"n": 0}

    async def flaky(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"ok": True, "participants": []}

    sync = ss.SignupSync(show, script.endpoint, poll_s=2, fetch_json=flaky)
    sync.poll_s = 0.01  # not through the constructor: MIN_POLL_S guards the real thing

    async def go():
        task = asyncio.create_task(sync.run())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if calls["n"] >= 3:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(go())
    assert calls["n"] >= 3 and sync.status["last_error"] is None


# ------------------------------------------------------- review fixes

def test_sync_never_restarts_the_transcription(tmp_path, script, monkeypatch):
    import inspect
    assert inspect.signature(Show.refresh_context).parameters["reconnect_speech"].default is True  # /setup keeps reconnecting
    show = make_show(tmp_path)
    calls = []
    monkeypatch.setattr(show, "refresh_context", lambda reconnect_speech=True: calls.append(reconnect_speech))
    script.add("sarah-tan", "Sarah Tan")
    run(sync_for(show, script).poll_once())
    assert calls == [False]


def test_first_sight_never_pins_a_person_to_two_cameras(tmp_path, script):
    show = make_show(tmp_path)
    show.upsert_person("Sarah Tan", [], "", False, person_id="sarah")
    show.cameras["C"].fixed_person = "sarah"  # the operator's placement
    script.add("sarah-tan", "Sarah Tan", camera="CAM-HOST")
    r = run(sync_for(show, script).poll_once())
    assert r["cameras"] == [] and show.cameras["B"].fixed_person is None and show.cameras["C"].fixed_person == "sarah"
    assert [cid for cid, c in show.cameras.items() if c.fixed_person == "sarah"] == ["C"]
    assert any("already fixed on camera C here" in e["text"] for e in show.events)


def test_only_preset_roles_come_from_the_sheet(tmp_path, script):
    assert ss.clean_role(" judge ") == "Judge" and ss.clean_role("panelist") == "Panellist" and ss.clean_role("Guest of Honor") == "Guest of honour"
    assert ss.clean_role("Host") == "" and ss.clean_role("thank you") == "" and ss.clean_role(None) == ""
    show = make_show(tmp_path)
    script.add("a", "Ann Lee", role="Judge")
    script.add("b", "Bo Chen", role="thank you")
    script.add("c", "Cy Dee", role="Host")
    run(sync_for(show, script).poll_once())
    assert [(p.id, p.role, p.is_host) for p in show.roster.people] == [("a", "Judge", False), ("b", "", False), ("c", "", False)]
    assert sum("role text from the sheet not used" in e["text"] for e in show.events) == 2


def test_one_bad_row_does_not_stop_the_others(tmp_path, script, monkeypatch):
    show = make_show(tmp_path)
    real = show.add_person_photos

    def flaky(pid, files):
        if pid == "bad-one":
            raise OSError("disk full")
        return real(pid, files)

    monkeypatch.setattr(show, "add_person_photos", flaky)
    calls = []
    monkeypatch.setattr(show, "refresh_context", lambda reconnect_speech=True: calls.append(reconnect_speech))
    script.add("bad-one", "Bad One")
    script.add("good-one", "Good One", camera="CAM-GUEST")
    sync = sync_for(show, script)
    r = run(sync.poll_once())
    assert r["added"] == ["bad-one", "good-one"] and r["photos"] == 2 and r["cameras"] == ["good-one -> C"]
    assert calls == [False] and "disk full" in sync.status["last_error"]
    assert set(json.loads(sync.state_path.read_text())["guests"]) == {"bad-one", "good-one"}  # state saved despite the failure


def test_signup_config_precedence(tmp_path):
    from server.config import Settings, apply_signup_config
    cfg = tmp_path / "config.js"
    cfg.write_text("window.CUE_SIGNUP = { endpoint: 'https://script.google.com/macros/s/abc/exec', eventId: 'hackmit-demo' };")
    s = Settings(signup_endpoint="https://env.example/exec")
    apply_signup_config(s, cfg)
    assert (s.signup_endpoint, s.signup_source, s.signup_event_id) == ("https://env.example/exec", "env", "hackmit-demo")
    s = Settings(signup_endpoint="https://env.example/exec", signup_event_id="other")
    apply_signup_config(s, cfg)
    assert s.signup_event_id == "other"
    s = Settings()
    apply_signup_config(s, cfg)
    assert (s.signup_endpoint, s.signup_source, s.signup_event_id) == ("https://script.google.com/macros/s/abc/exec", "config.js", "hackmit-demo")
    assert s.has_signup
    s.signup_enabled = False
    assert not s.has_signup
    s = Settings()
    apply_signup_config(s, tmp_path / "missing.js")
    assert not s.has_signup and s.signup_source == ""
