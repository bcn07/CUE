"""Sheet sign-ups -> this director's people list.

hemadassani/cue-desk-ui's signup.html writes each guest (name, role, two photos, consent) to a
Google Sheet through its Apps Script web app, and dashboard.html assigns each guest a camera in
the same sheet. This module polls that web app (`GET <endpoint>?action=roster`) and enrols every
consenting guest here: name and role as a Person whose id is the sheet's guestId, the two Drive
photos into data/people/<id>/ for face enrolment, and the sheet's camera column as the matching
camera's fixed person. Nothing is ever removed because a row disappeared; the operator removes
people on /setup.

Rules, stated on purpose:
- A guest whose name matches a person already enrolled here (case-insensitive) is adopted, not
  duplicated: two roster entries for one name would make every cue to them ambiguous.
- The host flag never comes from the sheet. The host is whoever the operator marks on /setup.
- Camera assignment follows the sheet only when the sheet's value changes. The first time a
  guest is seen it fills an empty slot but never overrides a camera the operator already fixed.
- Photos are fetched at a larger Drive thumbnail size than the dashboard uses (sz=w1024), falling
  back to the sheet's own URL. A non-image answer (Drive's sign-in page when photos are not
  link-readable) is refused and retried on later polls, up to MAX_PHOTO_ATTEMPTS.
- Someone the operator removed on /setup is not brought back by the sheet.
- Only the sign-up page's preset roles are taken (never "Host"); free text a guest typed is not,
  because Person.role reaches the interpreter's prompt and the spoken-role matcher.
- The sync never restarts the Deepgram stream (that drops the host's clause in flight); new names
  reach Deepgram's key terms at the next operator-driven reconnect.
- The endpoint is a capability URL. It never appears in logs, events or the state snapshot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

log = logging.getLogger("cue.signup")

DESK_TO_ROLE = {"CAM-HOST": "host", "CAM-GUEST": "guest", "CAM-WIDE": "wide"}
ROLE_PRESETS = {"speaker": "Speaker", "guest of honour": "Guest of honour", "guest of honor": "Guest of honour",
                "panellist": "Panellist", "panelist": "Panellist", "judge": "Judge", "performer": "Performer"}
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp")
MAX_PHOTO_ATTEMPTS = 6
PHOTO_WIDTH = 1024
MIN_POLL_S = 2.0

FetchJson = Callable[[str], Awaitable[Any]]
FetchBytes = Callable[[str], Awaitable[tuple[str, bytes]]]


# ------------------------------------------------------------------ pure


def read_config_js(path: Path | str) -> dict[str, str]:
    """`endpoint` and `eventId` out of the desk pages' config.js (a JS object literal), so the
    operator configures the sheet once. Only simple quoted string values are read."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for key in ("endpoint", "eventId"):
        m = re.search(r"\b%s\s*:\s*(['\"])(.*?)\1" % key, text)
        if m:
            out[key] = m.group(2).strip()
    return out


def roster_url(endpoint: str, event_id: str = "") -> str:
    q: dict[str, str] = {"action": "roster"}
    if event_id:
        q["eventId"] = event_id
    return endpoint + ("&" if "?" in endpoint else "?") + urlencode(q)


def larger_thumb(url: str, width: int = PHOTO_WIDTH) -> str:
    """Drive's thumbnail endpoint honours sz=w<px>; the sheet hands out w320 for its <img> tags."""
    return re.sub(r"([?&]sz=)w\d+", lambda m: f"{m.group(1)}w{width}", url)


def camera_for_desk(cameras: dict[str, Any], desk_cam: str) -> str | None:
    """Desk camera id (CAM-HOST / CAM-GUEST / CAM-WIDE) -> this director's camera id, by role.
    First camera per role wins, the same rule desk_ws.camera_map uses in the other direction."""
    role = DESK_TO_ROLE.get(str(desk_cam or "").strip().upper())
    if not role:
        return None
    for cid, cam in cameras.items():
        if str(getattr(cam, "role", "") or "").lower() == role:
            return cid
    return None


def clean_role(text: Any) -> str:
    """A sign-up page preset, canonically spelled, or nothing. "Host" is never taken from the sheet."""
    return ROLE_PRESETS.get(" ".join(str(text or "").split()).lower(), "")


def scrub(text: str) -> str:
    """Never let a URL (the capability endpoint, a Drive link) into an event or a log line."""
    return re.sub(r"https?://\S+", "<url>", str(text))[:200]


class _RedactUrls(logging.Filter):
    """httpx logs every request line at INFO with its full URL, which for the roster call is the
    capability endpoint. Installed once on the `httpx` logger; every URL in its lines becomes <url>."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "http://" in msg or "https://" in msg:
            record.msg = re.sub(r"https?://\S+", "<url>", msg)
            record.args = ()
        return True


def redact_httpx_logs() -> None:
    lg = logging.getLogger("httpx")
    if not any(isinstance(f, _RedactUrls) for f in lg.filters):
        lg.addFilter(_RedactUrls())


# --------------------------------------------------------------- transport


async def default_fetch_json(url: str) -> Any:
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as c:  # Apps Script answers with a 302
        r = await c.get(url)
        r.raise_for_status()
        return r.json()


async def default_fetch_bytes(url: str) -> tuple[str, bytes]:
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.headers.get("content-type", ""), r.content


# ------------------------------------------------------------------ sync


class SignupSync:
    """Polls the sheet's web app and enrols guests into `show`. One instance per director."""

    def __init__(self, show: Any, endpoint: str, event_id: str = "", poll_s: float = 5.0, state_path: Path | str | None = None,
                 fetch_json: FetchJson | None = None, fetch_bytes: FetchBytes | None = None, photo_width: int = PHOTO_WIDTH) -> None:
        self.show = show
        self._endpoint = endpoint.strip()
        self.event_id = event_id.strip()
        self.poll_s = max(MIN_POLL_S, float(poll_s))
        self.state_path = Path(state_path) if state_path else Path(show.s.data_dir) / "signup_sync.json"
        self.fetch_json = fetch_json or default_fetch_json
        self.fetch_bytes = fetch_bytes or default_fetch_bytes
        self.photo_width = photo_width
        redact_httpx_logs()
        self.guests: dict[str, dict] = {}  # guestId -> {pid, camera, photo_attempts, photos}
        self.status: dict[str, Any] = {"enabled": True, "event_id": self.event_id, "poll_s": self.poll_s, "polls": 0,
                                       "last_ok_wall": None, "last_error": None, "participants": 0, "enrolled": 0, "photos_pending": 0}
        self._load()

    # ---------------------------------------------------------- state
    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text())
            guests = data.get("guests") if isinstance(data, dict) else None
            if isinstance(guests, dict):
                self.guests = {str(k): dict(v) for k, v in guests.items() if isinstance(v, dict) and v.get("pid")}
        except (OSError, ValueError):
            self.guests = {}

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"guests": self.guests}, indent=2))
        except OSError as e:
            log.warning("sheet sign-ups: could not save state: %s", e)

    def _error(self, text: str) -> None:
        text = scrub(text)
        if self.status["last_error"] != text:
            log.warning("sheet sign-ups: %s", text)
            self.show.log_event("signup", f"sheet sign-ups: {text}")
        self.status["last_error"] = text
        self.show.mark_dirty()

    # ------------------------------------------------------------ loop
    async def run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the poller must outlive any one bad answer
                self._error(f"{type(e).__name__}: {e}")
            await asyncio.sleep(self.poll_s)

    async def poll_once(self) -> dict:
        self.status["polls"] += 1
        try:
            data = await self.fetch_json(roster_url(self._endpoint, self.event_id))
        except Exception as e:  # noqa: BLE001
            self._error(f"sheet unreachable: {type(e).__name__}: {e}")
            return {"ok": False}
        if not isinstance(data, dict) or not data.get("ok") or not isinstance(data.get("participants"), list):
            err = data.get("error") if isinstance(data, dict) else None
            self._error(f"sheet answered without a roster: {err or type(data).__name__}")
            return {"ok": False}
        result = await self.apply(data["participants"])
        self.status["last_ok_wall"] = time.time()
        if not result.get("errors"):
            self.status["last_error"] = None  # a row that failed stays on the status line until it stops failing
        self.show.mark_dirty()
        return result

    # ----------------------------------------------------------- apply
    async def apply(self, participants: list) -> dict:
        """Enrol what the sheet lists. Idempotent: a second call with the same rows changes nothing.
        One bad row never stops the others; whatever changed is saved and applied at the end."""
        show = self.show
        added: list[str] = []
        adopted: list[str] = []
        cams: list[str] = []
        photos_saved = 0
        roster_changed = photos_changed = cams_changed = False
        seen = errors = 0
        for p in participants:
            if not isinstance(p, dict):
                continue
            gid = str(p.get("guestId") or "").strip()
            name = " ".join(str(p.get("name") or "").split())
            if not gid or len(name) < 2 or p.get("consent") is False:
                continue
            seen += 1
            try:
                rec = self.guests.get(gid)
                if rec is not None and rec.get("pid") not in show.roster.by_id:
                    if not rec.get("removed"):  # the operator removed them on /setup: the sheet does not bring them back
                        rec["removed"] = True
                        show.log_event("signup", f"{name} was removed on /setup; not re-enrolling from the sheet")
                    continue
                if rec is None:
                    raw_role = " ".join(str(p.get("role") or "").split())
                    role = clean_role(raw_role)
                    if raw_role and not role:
                        show.log_event("signup", f"{name}: role text from the sheet not used (only the sign-up page's preset roles are)")
                    existing = next((x for x in show.roster.people if x.name.strip().lower() == name.lower()), None)
                    if existing is not None:
                        pid = existing.id
                        if role and not existing.role:
                            existing.role = role
                            roster_changed = True
                        adopted.append(pid)
                    else:
                        pid = show.upsert_person(name, [], role, False, person_id=gid)  # never the host from the sheet
                        added.append(pid)
                        roster_changed = True
                    rec = {"pid": pid, "camera": None, "photo_attempts": 0, "photos": 0}
                    self.guests[gid] = rec
                pid = rec["pid"]

                # photos: only while this person has none on disk, and only so many tries
                if rec.get("photos", 0) == 0:
                    on_disk = self._count_photos(pid)
                    if on_disk:
                        rec["photos"] = on_disk  # added on /setup meanwhile
                    elif rec.get("photo_attempts", 0) < MAX_PHOTO_ATTEMPTS:
                        urls = [u for u in (p.get("front"), p.get("side")) if isinstance(u, str) and u.startswith("http")]
                        if urls:
                            rec["photo_attempts"] = rec.get("photo_attempts", 0) + 1
                            files = []
                            for i, u in enumerate(urls):
                                blob = await self._photo(u)
                                if blob:
                                    files.append((f"sheet-{'front' if i == 0 else 'side'}.jpg", blob))
                            if files:
                                n = await asyncio.to_thread(show.add_person_photos, pid, files)
                                rec["photos"] = n
                                photos_saved += n
                                photos_changed = True
                            elif rec["photo_attempts"] >= MAX_PHOTO_ATTEMPTS:
                                show.log_event("signup", f"{name}: photos not fetchable after {MAX_PHOTO_ATTEMPTS} tries "
                                                         "(are the Drive photos link-readable?); add photos on /setup")

                # camera column: a change is followed; first sight fills only an empty slot for an unplaced person
                desk_cam = str(p.get("camera") or "").strip().upper()
                prev = rec.get("camera")
                if prev is None or desk_cam != prev:
                    target = camera_for_desk(show.cameras, desk_cam) if desk_cam else None
                    if prev is None:
                        if target is not None:
                            held = [cid for cid, c in show.cameras.items() if c.fixed_person == pid]
                            holder = show.cameras[target].fixed_person
                            if held:
                                show.log_event("signup", f"{name}: already fixed on camera {held[0]} here; the sheet's {desk_cam} not applied")
                            elif holder is None:
                                show.cameras[target].fixed_person = pid
                                cams.append(f"{pid} -> {target}")
                                cams_changed = True
                            else:
                                show.log_event("signup", f"{name}: the sheet puts them on {desk_cam}, but camera {target} is fixed to "
                                                         f"{self._name(holder)} here; change it on /setup or the dashboard")
                        elif desk_cam:
                            show.log_event("signup", f"{name}: the sheet says {desk_cam} but no camera here has that role")
                    else:
                        for cid, cam in show.cameras.items():
                            if cam.fixed_person == pid and cid != target:
                                cam.fixed_person = None
                                cams_changed = True
                        if target is not None and show.cameras[target].fixed_person != pid:
                            holder = show.cameras[target].fixed_person
                            show.cameras[target].fixed_person = pid
                            cams.append(f"{pid} -> {target}" + (f" (replacing {self._name(holder)})" if holder else ""))
                            cams_changed = True
                        elif target is None and desk_cam:
                            show.log_event("signup", f"{name}: the sheet says {desk_cam} but no camera here has that role")
                    rec["camera"] = desk_cam
            except Exception as e:  # noqa: BLE001 - one row must not stop the rest
                errors += 1
                self._error(f"{name}: {type(e).__name__}: {e}")

        if roster_changed:
            show.save_roster()
        if cams_changed:
            show.save_cameras()
        for pid in added:
            show.log_event("signup", f"enrolled {self._name(pid)} from the sheet (id {pid})")
        for pid in adopted:
            show.log_event("signup", f"{self._name(pid)} signed up on the sheet; already enrolled here as {pid}")
        if photos_saved:
            show.log_event("signup", f"{photos_saved} photo(s) fetched from the sheet")
        for line in cams:
            show.log_event("signup", f"sheet camera assignment: {line}")
        self.status["participants"] = seen
        self.status["enrolled"] = sum(1 for g in self.guests.values() if g.get("pid") in show.roster.by_id)
        self.status["photos_pending"] = sum(1 for g in self.guests.values()
                                            if g.get("pid") in show.roster.by_id and g.get("photos", 0) == 0
                                            and g.get("photo_attempts", 0) < MAX_PHOTO_ATTEMPTS)
        self._save()
        if roster_changed or photos_changed:
            await asyncio.to_thread(show.refresh_context, reconnect_speech=False)  # never restart the transcription
        if roster_changed or photos_changed or cams_changed:
            show.mark_dirty()
        return {"ok": True, "added": added, "adopted": adopted, "photos": photos_saved, "cameras": cams, "errors": errors}

    # --------------------------------------------------------- helpers
    def _name(self, pid: str | None) -> str:
        p = self.show.roster.by_id.get(pid or "")
        return p.name if p is not None else str(pid)

    def _count_photos(self, pid: str) -> int:
        pdir = Path(self.show.s.data_dir) / "people" / pid
        if not pdir.is_dir():
            return 0
        return sum(1 for f in pdir.iterdir() if f.suffix.lower() in PHOTO_EXTS)

    async def _photo(self, url: str) -> bytes | None:
        tried: list[str] = []
        for u in (larger_thumb(url, self.photo_width), url):
            if u in tried:
                continue
            tried.append(u)
            try:
                ctype, data = await self.fetch_bytes(u)
            except Exception as e:  # noqa: BLE001
                log.debug("photo fetch failed: %s", scrub(f"{type(e).__name__}: {e}"))
                continue
            if data and ctype.split(";")[0].strip().lower().startswith("image/"):
                return data
        return None
