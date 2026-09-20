"""Live mic -> Deepgram streaming -> clause events.

Master audio is captured once, here, on the director MacBook. Video cuts never
touch it (CLAUDE.md hard rule 5). Deepgram WebSocket protocol per
https://developers.deepgram.com/reference/speech-to-text-api/listen-streaming

Two Deepgram models, one client. `nova-3` (v1 /listen) sends interim and final
results per clause. `flux-general-en` (v2 /listen, Deepgram's turn-based model)
sends TurnInfo events instead: Update (the turn so far), then EndOfTurn with the
whole turn once the model or a silence timeout decides the speaker is done.
`flux_to_results` turns those into the v1 shapes the assembler already reads,
one final per sentence, so the director still acts clause by clause. Flux has
no KeepAlive message (silence is sent instead) and takes new key terms through
a Configure message, so a roster change does not drop the socket.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

from .assembler import ClauseAssembler, Event

log = logging.getLogger("cue.speech")

EventSink = Callable[[Event, dict], Awaitable[None]]
StatusSink = Callable[[dict], None]

_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def flux_to_results(msg: dict) -> list[dict]:
    """Flux TurnInfo -> the v1 Results shapes ClauseAssembler reads. Progress events (StartOfTurn,
    Update, EagerEndOfTurn, TurnResumed) carry the whole turn so far and become one interim result,
    UI only. EndOfTurn becomes one final result per sentence, the last of them speech_final, so a
    turn of "Don't bring Sarah up yet. Let's welcome Daniel." is two clauses in one utterance and the
    correction rule still applies. Word times are stream seconds, like nova's."""
    event = msg.get("event")
    text = _WS.sub(" ", str(msg.get("transcript") or "").strip())
    words = [{"word": w.get("word"), "start": _num(w.get("start")), "end": _num(w.get("end")), "confidence": _num(w.get("confidence"))}
             for w in (msg.get("words") or []) if isinstance(w, dict)]
    start, end = _num(msg.get("audio_window_start")), _num(msg.get("audio_window_end"))

    def result(t: str, ws: list[dict], is_final: bool, speech_final: bool, st, en) -> dict:
        return {"type": "Results", "is_final": is_final, "speech_final": speech_final, "start": st,
                "duration": (en - st) if (st is not None and en is not None) else None,
                "channel": {"alternatives": [{"transcript": t, "words": ws}]},
                "flux_event": event, "turn_index": msg.get("turn_index")}

    if event != "EndOfTurn":
        return [result(text, words, False, False, start, end)] if text else []
    if not text:
        return [result("", [], True, True, start, end)]  # a silent turn end still closes an open utterance
    sentences = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    if len(sentences) <= 1:
        return [result(text, words, True, True, start, end)]
    aligned = sum(len(s.split()) for s in sentences) == len(words)
    out, i = [], 0
    for k, s in enumerate(sentences):
        n = len(s.split())
        ws = words[i:i + n] if aligned else []
        i += n
        st = min((w["start"] for w in ws if w["start"] is not None), default=start) if ws else start
        en = max((w["end"] for w in ws if w["end"] is not None), default=end) if ws else end
        out.append(result(s, ws, True, k == len(sentences) - 1, st, en))
    return out


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d["name"], "default_samplerate": d.get("default_samplerate")})
    return out


class MicSource:
    """sounddevice callback -> asyncio.Queue of int16 mono PCM chunks (20 ms)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, sample_rate: int = 16000, device: str | int | None = None,
                 block_ms: int = 20, max_queue: int = 200):
        self.loop = loop
        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = int(sample_rate * block_ms / 1000)
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue)
        self._stream = None
        self.level = 0.0  # rough RMS 0..1 for the UI
        self.dropped = 0
        self.chunks = 0
        self.taps: list = []       # extra consumers of the master audio (recorder); called on the loop thread
        self.feed_queue = True     # False when nobody (no Deepgram) drains the queue

    def start(self) -> None:
        import numpy as np
        import sounddevice as sd

        dev = self.device
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)

        def cb(indata, frames, t, status):  # runs on the PortAudio thread
            if status:
                log.debug("mic status: %s", status)
            data = bytes(indata)
            try:
                a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
                self.level = 0.7 * self.level + 0.3 * min(1.0, rms * 4)
            except Exception:
                pass
            self.chunks += 1
            self.loop.call_soon_threadsafe(self._put, data)

        self._stream = sd.RawInputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                                         blocksize=self.blocksize, device=dev, callback=cb)
        self._stream.start()
        log.info("mic started (device=%s, %d Hz)", dev if dev is not None else "default", self.sample_rate)

    def _put(self, data: bytes) -> None:
        for tap in list(self.taps):
            try:
                tap(data)
            except Exception as e:
                log.debug("audio tap failed: %s", e)
        if not self.feed_queue:
            return
        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            self.dropped += 1
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(data)
            except Exception:
                pass

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class DeepgramStream:
    """Owns the Deepgram WebSocket, reconnects with a new audio epoch, feeds the assembler."""

    def __init__(self, api_key: str, sample_rate: int, on_event: EventSink, on_status: StatusSink | None = None,
                 model: str = "nova-3", endpointing_ms: int = 300, utterance_end_ms: int = 1000,
                 clause_timeout_s: float = 1.5, keyterms: Callable[[], list[str]] | None = None,
                 eot_threshold: float = 0.7, eot_timeout_ms: int = 3000):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.on_event = on_event
        self.on_status = on_status or (lambda s: None)
        self.model = model
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.keyterms = keyterms or (lambda: [])
        self.eot_threshold = min(0.9, max(0.5, float(eot_threshold)))   # Flux only
        self.eot_timeout_ms = min(60000, max(500, int(eot_timeout_ms)))  # Flux only: silence backstop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._reconnect_ev: asyncio.Event | None = None
        self.assembler = ClauseAssembler(timeout_s=clause_timeout_s, epoch=1)
        self.epoch = 0
        self.connected = False
        self.stream_t0_wall: float | None = None   # wall time when the first audio byte was sent
        self.stream_t0_mono: float | None = None
        self.bytes_sent = 0
        self.messages = 0
        self.results = 0
        self.last_msg_mono = 0.0
        self.last_send_mono = 0.0
        self.sender_alive = False
        self.last_error = ""
        self._stop = asyncio.Event()

    @property
    def is_flux(self) -> bool:
        return self.model.lower().startswith("flux")

    def keyterm_list(self) -> list[str]:
        """Roster names and aliases as key terms: Deepgram boosts them ("Barack" instead of "Hurrah")."""
        out, seen = [], set()
        for term in self.keyterms():
            t = term.strip()
            if t and t.lower() not in seen and len(seen) < 50:
                seen.add(t.lower())
                out.append(t)
        return out

    def url(self) -> str:
        if self.is_flux:
            q = [("model", self.model), ("encoding", "linear16"), ("sample_rate", str(self.sample_rate)),
                 ("eot_threshold", f"{self.eot_threshold:g}"), ("eot_timeout_ms", str(self.eot_timeout_ms))]
            q += [("keyterm", t) for t in self.keyterm_list()]
            return "wss://api.deepgram.com/v2/listen?" + urlencode(q)
        q = [
            ("model", self.model), ("encoding", "linear16"), ("sample_rate", str(self.sample_rate)), ("channels", "1"),
            ("interim_results", "true"), ("smart_format", "true"), ("punctuate", "true"),
            ("endpointing", str(self.endpointing_ms)), ("utterance_end_ms", str(self.utterance_end_ms)),
            ("vad_events", "true"),
        ]
        q += [("keyterm", t) for t in self.keyterm_list()]
        return "wss://api.deepgram.com/v1/listen?" + urlencode(q)

    def reconnect(self) -> None:
        """New roster terms. Flux takes them in place through a Configure message, nothing in flight is
        lost. nova needs a fresh URL: ask the run loop to drop the socket and reconnect (no close
        handshake wait: Deepgram keeps the socket open while audio flows)."""
        if self.is_flux and self.connected and self._ws is not None and self._loop is not None:
            ws = self._ws
            self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self._configure(ws)))
            return
        if self._reconnect_ev is not None:
            self._reconnect_ev.set()

    async def _configure(self, ws) -> None:
        terms = self.keyterm_list()
        try:
            await ws.send(json.dumps({"type": "Configure", "keyterms": terms}))
            log.info("Deepgram Flux: %d key terms sent in place, no reconnect", len(terms))
        except Exception as e:  # noqa: BLE001
            log.warning("Deepgram Flux configure failed (%s); reconnecting instead", e)
            if self._reconnect_ev is not None:
                self._reconnect_ev.set()

    def stop(self) -> None:
        self._stop.set()

    async def run(self, audio_q: asyncio.Queue[bytes]) -> None:
        from websockets.asyncio.client import connect as ws_connect  # websockets >= 13

        backoff = 1.0
        self._loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            self.epoch += 1
            self.assembler.reset(self.epoch)
            self.stream_t0_wall = None
            self.stream_t0_mono = None
            headers = {"Authorization": f"Token {self.api_key}"}
            try:
                # Protocol pings detect a half-open link (roaming APs) so the reconnect loop can take over.
                async with ws_connect(self.url(), additional_headers=headers, max_size=None, ping_interval=10, ping_timeout=10) as ws:
                    self._ws = ws
                    self.connected = True
                    self.last_error = ""
                    backoff = 1.0
                    self.on_status({"connected": True, "epoch": self.epoch})
                    log.info("Deepgram connected (epoch %d, model %s)", self.epoch, self.model)
                    sender = asyncio.create_task(self._send_guarded(ws, audio_q))
                    self._reconnect_ev = asyncio.Event()
                    receiver = asyncio.create_task(self._recv_loop(ws))
                    waiter = asyncio.create_task(self._reconnect_ev.wait())
                    try:
                        done, _ = await asyncio.wait({receiver, waiter}, return_when=asyncio.FIRST_COMPLETED)
                        if waiter in done:
                            log.info("Deepgram reconnect requested (roster changed): reopening with new key terms")
                            receiver.cancel()
                            try:
                                await ws.send(json.dumps({"type": "CloseStream"}))
                            except Exception:
                                pass
                            backoff = 0.2
                        else:
                            waiter.cancel()
                            receiver.result()  # re-raise a receive error
                    finally:
                        for t in (receiver, waiter):
                            t.cancel()
                        try:
                            await asyncio.wait_for(ws.close(), timeout=2)
                        except Exception:
                            pass
                        sender.cancel()
                        try:
                            await sender
                        except (asyncio.CancelledError, Exception):
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                log.warning("Deepgram stream error: %s", self.last_error)
            finally:
                self._ws = None
                self.connected = False
                self.on_status({"connected": False, "epoch": self.epoch, "error": self.last_error})
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self.messages += 1
            self.last_msg_mono = time.monotonic()
            if msg.get("type") in ("Results", "TurnInfo"):
                self.results += 1
            await self._handle(msg)

    async def _send_guarded(self, ws, audio_q: asyncio.Queue[bytes]) -> None:
        self.sender_alive = True
        try:
            await self._send_loop(ws, audio_q)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = f"sender: {type(e).__name__}: {e}"[:200]
            log.warning("Deepgram sender stopped: %s", self.last_error)
            try:
                await ws.close()  # force the receive loop out so the outer loop reconnects
            except Exception:
                pass
        finally:
            self.sender_alive = False

    async def _send_loop(self, ws, audio_q: asyncio.Queue[bytes]) -> None:
        # Drop audio that piled up while disconnected: stale speech must not create late cuts.
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        last_send = time.monotonic()
        while True:
            try:
                chunk = await asyncio.wait_for(audio_q.get(), timeout=3.0)
            except asyncio.TimeoutError:
                if self.is_flux:
                    await ws.send(bytes(int(self.sample_rate * 0.02) * 2))  # 20 ms of silence: Flux has no KeepAlive
                else:
                    await ws.send(json.dumps({"type": "KeepAlive"}))
                continue
            if self.stream_t0_wall is None:
                self.stream_t0_wall = time.time()
                self.stream_t0_mono = time.monotonic()
            await ws.send(chunk)
            self.bytes_sent += len(chunk)
            self.last_send_mono = time.monotonic()

    async def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "TurnInfo":
            msgs = flux_to_results(msg)
        elif kind in ("Results", "UtteranceEnd"):
            msgs = [msg]
        else:
            if kind == "Error":
                self.last_error = str(msg)[:200]
            elif kind == "ConfigureFailure":
                log.warning("Deepgram Flux refused the new key terms; reconnecting instead")
                if self._reconnect_ev is not None:
                    self._reconnect_ev.set()
            return
        now = time.monotonic()
        events = []
        for m in msgs:
            events.extend(self.assembler.feed(m, now))
        if not events:
            return
        meta_base = {"epoch": self.epoch, "recv_mono": now, "recv_wall": time.time(), "source": "mic"}
        for ev in events:
            meta = dict(meta_base)
            audio_end = getattr(ev, "audio_end_s", None)
            if audio_end is not None and self.stream_t0_mono is not None:
                meta["speech_end_mono"] = self.stream_t0_mono + float(audio_end)
            await self.on_event(ev, meta)
