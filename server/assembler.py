"""Deepgram streaming messages -> clause events. Pure: no network, no clocks.

Design rule (clauses and corrections): act on every finished clause
for speed. Deepgram marks a finished clause with `is_final: true`; the whole
utterance ends with `speech_final: true` or a separate `UtteranceEnd` message
(or our own timeout). Every clause carries the utterance_id it belongs to, so
the director can allow one fast re-cut when a correction lands inside the
same utterance ("Sarah, actually Daniel").

Message shapes follow Deepgram's documented WebSocket payloads
(https://developers.deepgram.com/docs/interim-results,
 https://developers.deepgram.com/docs/utterance-end).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WS = re.compile(r"\s+")


@dataclass
class Interim:
    """Progressive text for the UI only. Never drives a cut."""
    utterance_id: str
    text: str


@dataclass
class Clause:
    utterance_id: str
    clause_index: int
    text: str                 # this finished segment
    utterance_text: str       # every finished segment of the utterance so far
    audio_start_s: float | None
    audio_end_s: float | None  # Deepgram stream clock of the last word
    is_utterance_end: bool


@dataclass
class UtteranceEnd:
    utterance_id: str
    text: str


Event = Interim | Clause | UtteranceEnd


class ClauseAssembler:
    def __init__(self, timeout_s: float = 1.5, epoch: int = 1, prefix: str = "utt") -> None:
        self._timeout_s = timeout_s
        self._epoch = epoch
        self._prefix = prefix
        self._counter = 0
        self._reset()

    # ------------------------------------------------------------------ public
    def reset(self, epoch: int) -> None:
        """A reconnected Deepgram socket starts a new epoch; drop in-flight text."""
        self._epoch = epoch
        self._reset()

    @property
    def open_utterance_id(self) -> str | None:
        return self._utt_id if self._open else None

    def feed(self, msg: dict[str, Any], now: float) -> list[Event]:
        events: list[Event] = []
        if self._open and self._last_final_at is not None and now - self._last_final_at >= self._timeout_s:
            events.extend(self._close())

        kind = msg.get("type")
        if kind == "UtteranceEnd":
            events.extend(self._close())
            return events
        if kind not in (None, "Results"):
            return events

        alt = self._first_alt(msg)
        transcript = _WS.sub(" ", (alt.get("transcript") or "").strip())
        words = alt.get("words") or []
        is_final = bool(msg.get("is_final"))
        speech_final = bool(msg.get("speech_final"))

        if not is_final:
            if transcript:
                uid = self._utt_id if self._open else self._peek_next_id()
                events.append(Interim(uid, self._join([*self._segments, transcript])))
            return events

        if transcript:
            key = (msg.get("start"), msg.get("duration"), transcript)
            if key in self._seen:
                self._last_final_at = now
            else:
                if not self._open:
                    self._open_new()
                self._seen.add(key)
                self._segments.append(transcript)
                self._idx += 1
                start, end = self._span(msg, words)
                self._last_final_at = now
                events.append(
                    Clause(
                        utterance_id=self._utt_id,
                        clause_index=self._idx,
                        text=transcript,
                        utterance_text=self._join(self._segments),
                        audio_start_s=start,
                        audio_end_s=end,
                        is_utterance_end=speech_final,
                    )
                )
        if speech_final and self._open:
            events.extend(self._close())
        return events

    # ----------------------------------------------------------------- helpers
    def _reset(self) -> None:
        self._open = False
        self._utt_id = ""
        self._idx = 0
        self._segments: list[str] = []
        self._seen: set[tuple[Any, Any, str]] = set()
        self._last_final_at: float | None = None

    def _peek_next_id(self) -> str:
        return f"{self._prefix}-e{self._epoch}-{self._counter + 1}"

    def _open_new(self) -> None:
        self._counter += 1
        self._open = True
        self._utt_id = f"{self._prefix}-e{self._epoch}-{self._counter}"
        self._idx = 0
        self._segments = []

    def _close(self) -> list[Event]:
        if not self._open:
            return []
        ev = UtteranceEnd(self._utt_id, self._join(self._segments))
        self._reset()
        return [ev]

    @staticmethod
    def _join(parts: list[str]) -> str:
        return _WS.sub(" ", " ".join(p for p in parts if p).strip())

    @staticmethod
    def _first_alt(msg: dict[str, Any]) -> dict[str, Any]:
        ch = msg.get("channel") or {}
        if isinstance(ch, dict):
            alts = ch.get("alternatives") or []
            if alts:
                return alts[0] or {}
        return {}

    @staticmethod
    def _span(msg: dict[str, Any], words: list[dict[str, Any]]) -> tuple[float | None, float | None]:
        starts = [w.get("start") for w in words if w.get("start") is not None]
        ends = [w.get("end") for w in words if w.get("end") is not None]
        if starts and ends:
            return float(min(starts)), float(max(ends))
        st = msg.get("start")
        du = msg.get("duration")
        if st is not None and du is not None:
            return float(st), float(st) + float(du)
        return (float(st) if st is not None else None), None
