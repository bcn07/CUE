"""Live mic -> Deepgram streaming -> clause events.

Master audio is captured once, here, on the director MacBook. Video cuts never
touch it (CLAUDE.md hard rule 5). Deepgram WebSocket protocol per
https://developers.deepgram.com/reference/speech-to-text-api/listen-streaming
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

from .assembler import ClauseAssembler, Event

log = logging.getLogger("cue.speech")

EventSink = Callable[[Event, dict], Awaitable[None]]
StatusSink = Callable[[dict], None]


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
                 clause_timeout_s: float = 1.5, keyterms: Callable[[], list[str]] | None = None):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.on_event = on_event
        self.on_status = on_status or (lambda s: None)
        self.model = model
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.keyterms = keyterms or (lambda: [])
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

    def url(self) -> str:
        q = [
            ("model", self.model), ("encoding", "linear16"), ("sample_rate", str(self.sample_rate)), ("channels", "1"),
            ("interim_results", "true"), ("smart_format", "true"), ("punctuate", "true"),
            ("endpointing", str(self.endpointing_ms)), ("utterance_end_ms", str(self.utterance_end_ms)),
            ("vad_events", "true"),
        ]
        # Roster names and aliases as key terms: nova-3 boosts them ("Barack" instead of "Hurrah").
        seen = set()
        for term in self.keyterms():
            t = term.strip()
            if t and t.lower() not in seen and len(seen) < 50:
                seen.add(t.lower())
                q.append(("keyterm", t))
        return "wss://api.deepgram.com/v1/listen?" + urlencode(q)

    def reconnect(self) -> None:
        """Ask the run loop to drop the current socket and reconnect with a fresh URL (new roster terms).
        Does not wait for a close handshake: Deepgram keeps the socket open while audio flows."""
        if self._reconnect_ev is not None:
            self._reconnect_ev.set()

    def stop(self) -> None:
        self._stop.set()

    async def run(self, audio_q: asyncio.Queue[bytes]) -> None:
        from websockets.asyncio.client import connect as ws_connect  # websockets >= 13

        backoff = 1.0
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
            if msg.get("type") == "Results":
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
        if kind not in ("Results", "UtteranceEnd"):
            if kind == "Error":
                self.last_error = str(msg)[:200]
            return
        now = time.monotonic()
        events = self.assembler.feed(msg, now)
        if not events:
            return
        meta_base = {"epoch": self.epoch, "recv_mono": now, "recv_wall": time.time(), "source": "mic"}
        for ev in events:
            meta = dict(meta_base)
            audio_end = getattr(ev, "audio_end_s", None)
            if audio_end is not None and self.stream_t0_mono is not None:
                meta["speech_end_mono"] = self.stream_t0_mono + float(audio_end)
            await self.on_event(ev, meta)
