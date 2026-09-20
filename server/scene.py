"""Scene tags per camera from a local vision model, so "let's look at the flowers" can find the
camera that shows flowers, and "let's look at the audience" the one that shows a crowd.

How this stays inside the rules. The vision model only describes what each camera sees, as short
nouns, slowly and in the background: one camera per `interval_s`, the one whose tags are oldest,
and never while a cue is pending, so it does not compete with the interpreter for the GPU. The
speech side (dialogue.py) only extracts the phrase the host wants to look at. Matching the two is
plain code in `best_camera`: an exact noun match wins, one camera must win outright, and when two
cameras both show the thing the planner takes the wide shot instead of guessing. Tags age out. A
cut never waits for the vision model; it only reads tags already on file.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable

log = logging.getLogger("cue.scene")

PROMPT = ("You describe a live camera view for a TV director. List what is clearly visible as short lowercase nouns. "
          'Answer with JSON only: {"people": <count>, "objects": [up to 8 nouns], "setting": "<3 words>"}. Do not guess; omit anything unclear.')

SYNONYMS: dict[str, set[str]] = {
    "tv": {"television", "telly", "screen", "monitor", "display", "flat screen", "flatscreen"},
    "screen": {"tv", "television", "monitor", "display", "projector", "slide", "presentation"},
    "audience": {"crowd", "people", "spectators", "seats", "chairs", "guests", "viewers"},
    "flower": {"bouquet", "vase", "plant", "rose", "tulip", "orchid"},
    "plant": {"flower", "vase", "pot", "leaves"},
    "laptop": {"computer", "notebook", "macbook"},
    "whiteboard": {"board", "blackboard"},
    "table": {"desk"},
    "phone": {"smartphone", "iphone", "mobile"},
    "stairs": {"staircase", "step"},
    "bottle": {"water bottle"},
    "kitchen": {"microwave", "fridge", "counter", "stove"},
    "food": {"pizza", "snack", "cake", "plate"},
    "door": {"doorway", "entrance"},
}
STOP = {"the", "a", "an", "this", "that", "those", "these", "our", "my", "his", "her", "their", "some", "all", "of", "at", "on", "to", "over",
        "up", "in", "and", "or", "please", "now", "here", "there", "little", "big", "nice", "lovely", "beautiful", "new", "old"}
PRONOUNS = {"me", "us", "you", "him", "her", "them", "it", "this", "that", "those", "these", "yourself", "yourselves", "each other", "one another",
            "myself", "himself", "herself", "themselves", "ourselves"}

Call = Callable[[str, bytes, str], Awaitable[str]]


# ------------------------------------------------------------------ pure


def singular(word: str) -> str:
    if " " in word:
        parts = word.split(" ")
        return " ".join(parts[:-1] + [singular(parts[-1])])
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize(tag: Any) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", str(tag or "").lower()).strip()
    return singular(re.sub(r"\s+", " ", t))


def parse_tags(text: str) -> dict | None:
    """The model's JSON (fenced or not) -> {"objects": [nouns], "people": n, "setting": str}, or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    objects: list[str] = []
    for o in d.get("objects") if isinstance(d.get("objects"), list) else []:
        n = normalize(o)
        if n and n not in objects and len(objects) < 12:
            objects.append(n)
    try:
        people = max(0, int(d.get("people") or 0))
    except (TypeError, ValueError):
        people = 0
    return {"objects": objects, "people": people, "setting": normalize(d.get("setting") or "")[:40]}


def terms_for(phrase: str) -> set[str]:
    """What to look for in the tags: the phrase's words, singular, plus known synonyms."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(phrase or "").lower()) if w not in STOP]
    out: set[str] = set()
    for w in words:
        s = singular(w)
        out.update({w, s})
        for key, syn in SYNONYMS.items():
            if key in (w, s) or w in syn or s in syn:
                out.add(key)
                out.update(syn)
    if len(words) > 1:
        out.add(" ".join(singular(w) for w in words))
    return out


def match_score(phrase: str, info: dict | None) -> float:
    """1.0 when a tag equals a term of the phrase; 0.6 for a term inside a longer tag; 0.5 for the
    setting; 'audience' also matches a camera showing three or more people. 0 otherwise."""
    if not info:
        return 0.0
    terms = terms_for(phrase)
    if not terms:
        return 0.0
    best = 0.0
    for tag in (normalize(t) for t in info.get("objects") or []):
        if tag in terms:
            return 1.0
        if any(len(t) >= 3 and (t in tag.split() or t in tag) for t in terms):
            best = max(best, 0.6)
    setting = normalize(info.get("setting") or "")
    if setting and any(len(t) >= 3 and t in setting for t in terms):
        best = max(best, 0.5)
    if "audience" in terms and int(info.get("people") or 0) >= 3:
        best = max(best, 0.6)
    return best


def best_camera(scene: dict[str, dict], phrase: str, now: float, max_age_s: float, healthy: dict[str, bool]) -> tuple[str | None, float, list[str]]:
    """The one healthy camera whose fresh tags show the phrase's subject. Returns (cid, score, tied):
    cid is None when nothing matches or when two cameras tie at the top, which is never a guess."""
    scores: dict[str, float] = {}
    for cid, info in (scene or {}).items():
        if not healthy.get(cid):
            continue
        if now - float(info.get("at", -1e9)) > max_age_s:
            continue
        s = match_score(phrase, info)
        if s > 0:
            scores[cid] = s
    if not scores:
        return None, 0.0, []
    top = max(scores.values())
    tied = sorted(c for c, s in scores.items() if s >= top - 1e-9)
    return (tied[0] if len(tied) == 1 else None), top, tied


# ---------------------------------------------------------------- tagger


class SceneTagger:
    """Background loop: tag the healthy camera with the oldest tags, one per interval."""

    def __init__(self, show: Any, model: str, base_url: str, interval_s: float = 20.0, max_age_s: float = 600.0, max_width: int = 384,
                 timeout_s: float = 60.0, call: Call | None = None) -> None:
        self.show = show
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.interval_s = max(2.0, float(interval_s))
        self.max_age_s = float(max_age_s)
        self.max_width = int(max_width)
        self.timeout_s = float(timeout_s)
        self.call = call or self._call_openai_compatible
        self.status: dict[str, Any] = {"enabled": True, "model": model, "interval_s": self.interval_s, "calls": 0, "errors": 0,
                                       "last_ms": None, "last_cam": None, "last_error": None, "skipped_for_cue": 0}

    def pick(self, now: float) -> str | None:
        cands = []
        for cid, cam in self.show.cameras.items():
            if cam.latest is None or not cam.healthy(now, self.show.s.camera_stale_s):
                continue
            info = self.show.ws.scene.get(cid)
            cands.append((float(info.get("at", -1e9)) if info else -1e9, cid))
        if not cands:
            return None
        cands.sort()
        return cands[0][1]

    async def run(self) -> None:
        while True:
            try:
                await self.tag_next()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the loop outlives any one bad answer
                self.status["errors"] += 1
                self.status["last_error"] = f"{type(e).__name__}: {e}"[:160]
                log.warning("scene tagger: %s", self.status["last_error"])
            await asyncio.sleep(self.interval_s)

    async def tag_next(self) -> dict | None:
        if getattr(self.show, "pending_cue", None) is not None:  # never compete with a cue in flight
            self.status["skipped_for_cue"] += 1
            return None
        now = time.monotonic()
        cid = self.pick(now)
        if cid is None:
            return None
        from .vlm import downscale_jpeg
        small = await asyncio.to_thread(downscale_jpeg, self.show.cameras[cid].latest, self.max_width, 70)
        t0 = time.perf_counter()
        try:
            text = await asyncio.wait_for(self.call(self.model, small, PROMPT), timeout=self.timeout_s)
        except Exception as e:  # noqa: BLE001
            self.status["errors"] += 1
            self.status["last_error"] = f"{cid}: {type(e).__name__}: {e}"[:160]
            log.warning("scene tagger: %s", self.status["last_error"])
            return None
        self.status["calls"] += 1
        self.status["last_ms"] = round((time.perf_counter() - t0) * 1000)
        self.status["last_cam"] = cid
        info = parse_tags(text)
        if info is None:
            self.status["errors"] += 1
            self.status["last_error"] = f"{cid}: answer was not the JSON asked for"
            return None
        self.status["last_error"] = None
        info["at"] = time.monotonic()
        info["wall"] = time.time()
        prev = self.show.ws.scene.get(cid)
        self.show.ws.update_scene(cid, info)
        if not prev or prev.get("objects") != info["objects"] or prev.get("people") != info["people"]:
            self.show.log_event("scene", f"{cid} sees: {', '.join(info['objects'][:6]) or 'nothing clear'}"
                                         + (f" ({info['people']} people)" if info["people"] else ""))
        self.show.mark_dirty()
        return info

    async def _call_openai_compatible(self, model: str, jpeg: bytes, prompt: str) -> str:
        import httpx

        body = {"model": model, "max_tokens": 120, "temperature": 0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}}]}]}
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(self.base_url + "/chat/completions", json=body)
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
