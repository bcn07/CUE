"""Optional vision-language fallback (OpenAI vision, Responses API).

SFace is the primary identifier (milliseconds). This runs only when a camera
shows faces that SFace could not confirm, or when a named cue cannot find its
target anywhere. It is rate-limited per camera and never on the cut path: its
answer is written into the presence tracker as lower-confidence evidence and
the pending cue is re-evaluated by the director.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

import cv2
import numpy as np
from pydantic import BaseModel

log = logging.getLogger("cue.vlm")


class VLMOut(BaseModel):
    visible_ids: list[str]
    confidence: float
    note: str


SYSTEM = (
    "You identify which of the listed, pre-enrolled people are clearly visible in a live camera frame. "
    "Only answer with ids from the list. If nobody is clearly recognisable, return an empty list. "
    "Never guess: a wrong identification is worse than none. confidence is 0..1 for the whole answer."
)


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def downscale_jpeg(jpeg: bytes, max_w: int = 512, quality: int = 70) -> bytes:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (max_w, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else jpeg


class VLMIdentifier:
    def __init__(self, model: str, min_interval_s: float = 4.0, timeout_s: float = 4.0, max_refs: int = 8):
        from openai import AsyncOpenAI

        self.model = model
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.max_refs = max_refs
        self._client = AsyncOpenAI()
        self._last_call: dict[str, float] = {}
        self._inflight: set[str] = set()
        self._refs: list[dict] = []  # [{"id","name","jpeg"}]
        self.stats = {"calls": 0, "errors": 0, "last_ms": 0.0, "last_result": None}

    def set_references(self, refs: list[dict]) -> None:
        self._refs = refs[: self.max_refs]

    def can_call(self, cam_id: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not self._refs or cam_id in self._inflight:
            return False
        return now - self._last_call.get(cam_id, -1e9) >= self.min_interval_s

    async def identify(self, cam_id: str, jpeg: bytes) -> VLMOut | None:
        """Returns None when rate-limited or on failure."""
        if not self.can_call(cam_id):
            return None
        self._inflight.add(cam_id)
        self._last_call[cam_id] = time.monotonic()
        t0 = time.perf_counter()
        try:
            content: list[dict] = [{"type": "input_text", "text": "Reference photos of the enrolled people:"}]
            for r in self._refs:
                content.append({"type": "input_text", "text": f"id={r['id']} name={r['name']}"})
                content.append({"type": "input_image", "image_url": _data_url(r["jpeg"]), "detail": "low"})
            content.append({"type": "input_text", "text": f"Live frame from camera {cam_id}. Which of the listed ids are clearly visible?"})
            content.append({"type": "input_image", "image_url": _data_url(downscale_jpeg(jpeg)), "detail": "low"})
            kwargs: dict = {}
            if self.model.startswith("gpt-5") and not self.model.startswith("gpt-5-chat"):
                kwargs["reasoning"] = {"effort": "minimal"}
            r = await asyncio.wait_for(
                self._client.responses.parse(
                    model=self.model,
                    input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                    text_format=VLMOut,
                    max_output_tokens=200,
                    **kwargs,
                ),
                timeout=self.timeout_s,
            )
            out = r.output_parsed
            valid = {x["id"] for x in self._refs}
            if out is not None:
                out.visible_ids = [i for i in out.visible_ids if i in valid]
            self.stats["calls"] += 1
            self.stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.stats["last_result"] = {"cam": cam_id, "ids": out.visible_ids if out else [], "confidence": out.confidence if out else 0}
            return out
        except Exception as e:
            self.stats["errors"] += 1
            log.warning("VLM identify failed on %s: %s: %s", cam_id, type(e).__name__, e)
            return None
        finally:
            self._inflight.discard(cam_id)
