"""Deepgram client against a fake server: auth header, Results -> Clause, KeepAlive, reconnect epoch."""
import asyncio
import json
import time

import pytest
from websockets.asyncio.server import serve

from server.assembler import Clause
from server.speech import DeepgramStream


@pytest.mark.asyncio
async def test_deepgram_stream_protocol_and_reconnect():
    seen = {"headers": [], "closed_once": False, "keepalive": 0, "audio": 0}
    events: list = []

    async def handler(conn):
        seen["headers"].append(conn.request.headers.get("Authorization"))
        first = not seen["closed_once"]
        async for msg in conn:
            if isinstance(msg, bytes):
                seen["audio"] += len(msg)
                if first:
                    await conn.send(json.dumps({"type": "Results", "start": 0.0, "duration": 1.0, "is_final": True, "speech_final": True,
                                                "channel": {"alternatives": [{"transcript": "Please welcome Sarah.", "confidence": 0.9,
                                                                              "words": [{"word": "sarah", "start": 0.6, "end": 1.0}]}]}}))
                    seen["closed_once"] = True
                    await conn.close(1011, "simulated server drop")
                    return
            else:
                if json.loads(msg).get("type") == "KeepAlive":
                    seen["keepalive"] += 1

    async def on_event(ev, meta):
        events.append((ev, meta))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stream = DeepgramStream("secret-token", 16000, on_event, model="nova-3")
        stream.url = lambda: f"ws://127.0.0.1:{port}/v1/listen?model=nova-3"  # type: ignore[method-assign]
        q: asyncio.Queue[bytes] = asyncio.Queue()
        task = asyncio.create_task(stream.run(q))

        async def mic():  # like MicSource: 20 ms chunks, only while the first connection is up
            while not events:
                if stream.connected:
                    await q.put(bytes(640))
                await asyncio.sleep(0.02)

        mic_task = asyncio.create_task(mic())
        deadline = time.time() + 8
        while time.time() < deadline and not (stream.epoch >= 2 and stream.connected and events):
            await asyncio.sleep(0.05)
        mic_task.cancel()
        # second connection is idle: KeepAlive after 3 s without audio
        await asyncio.sleep(3.6)
        stream.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    assert seen["headers"][0] == "Token secret-token"
    assert seen["audio"] >= 640 and seen["audio"] % 640 == 0
    clauses = [e for e, m in events if isinstance(e, Clause)]
    assert clauses and clauses[0].text == "Please welcome Sarah." and clauses[0].is_utterance_end
    meta = next(m for e, m in events if isinstance(e, Clause))
    assert meta["epoch"] == 1 and meta["source"] == "mic" and meta.get("speech_end_mono") is not None
    assert stream.epoch >= 2, "a server-side drop must reconnect with a new audio epoch"
    assert seen["keepalive"] >= 1
