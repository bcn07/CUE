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


# ------------------------------------------------------------------ Flux

def _turn(event, transcript, words, start, end, turn_index=0):
    return {"type": "TurnInfo", "request_id": "r", "sequence_id": 1, "event": event, "turn_index": turn_index,
            "audio_window_start": str(start), "audio_window_end": str(end), "transcript": transcript,
            "words": [{"word": w, "confidence": "0.9", "start": s, "end": e} for w, s, e in words], "end_of_turn_confidence": "0.8"}


def test_flux_to_results_maps_turn_events_to_v1_shapes():
    from server.speech import flux_to_results
    from server.assembler import ClauseAssembler, Clause, Interim, UtteranceEnd
    assert flux_to_results(_turn("StartOfTurn", "", [], 0.0, 0.1)) == []
    up = flux_to_results(_turn("Update", "Please welcome", [("Please", 0.5, 0.8), ("welcome", 0.8, 1.1)], 0.0, 1.2))
    assert len(up) == 1 and up[0]["is_final"] is False and up[0]["channel"]["alternatives"][0]["transcript"] == "Please welcome"
    two = flux_to_results(_turn("EndOfTurn", "Don't bring Sarah up yet. Let's welcome Daniel.",
                                [("Don't", 0.5, 0.7), ("bring", 0.7, 0.9), ("Sarah", 0.9, 1.2), ("up", 1.2, 1.3), ("yet", 1.3, 1.5),
                                 ("Let's", 2.0, 2.2), ("welcome", 2.2, 2.5), ("Daniel", 2.5, 2.9)], 0.0, 3.1))
    assert [(r["channel"]["alternatives"][0]["transcript"], r["is_final"], r["speech_final"]) for r in two] == [
        ("Don't bring Sarah up yet.", True, False), ("Let's welcome Daniel.", True, True)]
    assert (two[0]["start"], round(two[0]["start"] + two[0]["duration"], 2)) == (0.5, 1.5) and two[1]["start"] == 2.0
    # through the real assembler: two clauses in ONE utterance, then the utterance end
    asm = ClauseAssembler(epoch=1)
    evs = []
    for m in up + two:
        evs.extend(asm.feed(m, 10.0))
    assert isinstance(evs[0], Interim) and evs[0].text == "Please welcome"
    clauses = [e for e in evs if isinstance(e, Clause)]
    assert [c.text for c in clauses] == ["Don't bring Sarah up yet.", "Let's welcome Daniel."]
    assert clauses[0].utterance_id == clauses[1].utterance_id and clauses[1].is_utterance_end and clauses[1].audio_end_s == 2.9
    assert isinstance(evs[-1], UtteranceEnd) and evs[-1].text == "Don't bring Sarah up yet. Let's welcome Daniel."
    # a silent timeout turn end only closes; word/sentence mismatch falls back to the turn span
    assert flux_to_results(_turn("EndOfTurn", "", [], 3.1, 6.1))[0]["speech_final"] is True
    odd = flux_to_results(_turn("EndOfTurn", "One two. Three.", [("one", 0, 1)], 0.0, 2.0))
    assert [r["start"] for r in odd] == [0.0, 0.0] and odd[1]["duration"] == 2.0


@pytest.mark.asyncio
async def test_flux_stream_url_silence_keepalive_and_configure_in_place():
    from server.assembler import Clause, UtteranceEnd
    seen = {"path": None, "texts": [], "audio": []}
    events: list = []

    async def handler(conn):
        seen["path"] = conn.request.path
        await conn.send(json.dumps({"type": "Connected", "request_id": "r", "sequence_id": 0}))
        sent = False
        async for msg in conn:
            if isinstance(msg, bytes):
                seen["audio"].append(len(msg))
                if not sent and sum(seen["audio"]) >= 640 * 3:
                    sent = True
                    await conn.send(json.dumps(_turn("Update", "Please welcome", [("Please", 0.1, 0.4), ("welcome", 0.4, 0.7)], 0.0, 0.8)))
                    await conn.send(json.dumps(_turn("EndOfTurn", "Please welcome Sarah.", [("Please", 0.1, 0.4), ("welcome", 0.4, 0.7), ("Sarah", 0.7, 1.0)], 0.0, 1.2)))
            else:
                seen["texts"].append(json.loads(msg))

    async def on_event(ev, meta):
        events.append((ev, meta))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        terms = ["Sarah Tan", "Sarah"]
        stream = DeepgramStream("secret-token", 16000, on_event, model="flux-general-en", keyterms=lambda: terms, eot_threshold=0.6, eot_timeout_ms=2500)
        real = stream.url()
        assert real.startswith("wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000&eot_threshold=0.6&eot_timeout_ms=2500")
        assert real.endswith("&keyterm=Sarah+Tan&keyterm=Sarah") and "interim_results" not in real and "channels" not in real
        stream.url = lambda: f"ws://127.0.0.1:{port}" + real[real.index("/v2/listen"):]  # type: ignore[method-assign]
        q: asyncio.Queue[bytes] = asyncio.Queue()
        task = asyncio.create_task(stream.run(q))
        deadline = time.time() + 5
        while time.time() < deadline and not stream.connected:
            await asyncio.sleep(0.02)
        for _ in range(4):
            await q.put(bytes(640))
        while time.time() < deadline and not any(isinstance(e, UtteranceEnd) for e, _ in events):
            await asyncio.sleep(0.02)
        kinds = [type(e).__name__ for e, _ in events]
        assert kinds == ["Interim", "Clause", "UtteranceEnd"], kinds
        clause = next(e for e, _ in events if isinstance(e, Clause))
        assert clause.text == "Please welcome Sarah." and clause.is_utterance_end and clause.audio_end_s == 1.0
        assert seen["path"].startswith("/v2/listen?model=flux-general-en")
        # idle: 20 ms of silence goes out instead of a KeepAlive message Flux does not have
        await asyncio.sleep(3.6)
        assert 640 in seen["audio"][4:] and not any(t.get("type") == "KeepAlive" for t in seen["texts"])
        # roster change: key terms travel in a Configure message, the socket and epoch stay
        terms.append("Daniel Ho")
        stream.reconnect()
        await asyncio.sleep(0.3)
        assert [t for t in seen["texts"] if t.get("type") == "Configure"] == [{"type": "Configure", "keyterms": ["Sarah Tan", "Sarah", "Daniel Ho"]}]
        assert stream.epoch == 1 and stream.connected
        stream.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
