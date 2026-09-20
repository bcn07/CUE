"""Language loop (spec §3.2): turns each finished clause into conversation state: dialogue act,
who is addressed (even without a name), expected next speaker, back-channel, salience, the
visual subject of the moment (audience, screen, object). Deterministic first (microseconds),
optionally refined by the local LLM. The existing Cue (who/intent/timing) is one input."""
from __future__ import annotations

import asyncio
import logging
import re
import time

from pydantic import BaseModel

from .semantics import Action, Cue, Roster

log = logging.getLogger("cue.dialogue")

_BACKCHANNEL = {"yeah", "yes", "mm-hmm", "mhm", "mm", "uh-huh", "right", "exactly", "sure", "okay", "ok", "wow", "nice", "great",
                "true", "totally", "absolutely", "i see", "got it", "hmm", "yep", "no way", "really", "cool", "oh", "ah", "interesting"}
_AUDIENCE = re.compile(r"\b(round of applause|give (it|them) up|put your hands together|everyone (here|in the room)|the audience|this crowd|"
                       r"everybody (here|in the room|out there)|ladies and gentlemen|all of you|hands up|show of hands|cheer)\b", re.I)
_SCREEN = re.compile(r"\b(on (the|this) screen|the slide|slides|behind me|on the monitor|take a look at (the|this) (screen|slide|chart|graph)|"
                     r"as you can see (here|on)|let'?s watch|roll the (clip|video|tape)|the video)\b", re.I)
_OBJECT = re.compile(r"\b(the prototype|this device|this thing|here'?s how it works|let me show you|check this out|this is what we built|"
                     r"the demo|watch this|hold this|take this|pass (me|it))\b", re.I)
_QUESTION = re.compile(r"\?\s*$|^(what|why|how|when|where|who|which|do you|did you|can you|could you|would you|is it|are you|tell (us|me))\b", re.I)
_TRANSITION = re.compile(r"\b(that brings us to|moving on|let'?s move on|next up|our next|before we (go|move|wrap)|to wrap up|finally|last question|"
                         r"final question|we'?ll be right back|that'?s all|thank you both|thanks everyone|let'?s get started|welcome to)\b", re.I)
_CONCLUSION = re.compile(r"\b(we'?ll be right back|that'?s all (for|from)|thank you both|thanks everyone|see you|good night|wrap(ping)? up)\b", re.I)
_JOKE_CUE = re.compile(r"\b(just kidding|haha|lol|kidding|funny)\b", re.I)
_EMOTIONAL = re.compile(r"\b(passed away|lost|hardest|cried|scared|terrified|proud|grateful|honest(ly)?|personal|struggled|difficult|"
                        r"never forget|changed my life|my (mother|father|mom|dad|family)|diagnos)\w*\b", re.I)
_PLURAL = re.compile(r"\b(both of you|the two of you|all of you|each of you|everyone|everybody|you (guys|all|two))\b", re.I)


class DialogueLLM(BaseModel):
    dialogue_act: str
    addressed: list[str]
    expected_next: str
    backchannel: bool
    sentence_complete: bool
    salience: float
    subject: str


def analyze_rules(text: str, cue: Cue | None, roster: Roster, prev_speaker: str | None) -> dict:
    t = text.strip()
    low = t.lower()
    words = re.findall(r"[a-z'\-]+", low)
    out = {"dialogue_act": "NONE", "addressed": [], "expected_next": None, "backchannel": False, "sentence_complete": True,
           "salience": 0.3, "references": [], "transition": False, "subject": None, "last_clause": t, "source": "rules"}
    if not words:
        return out
    if len(words) <= 3 and (" ".join(words) in _BACKCHANNEL or words[0] in _BACKCHANNEL):
        out.update(dialogue_act="BACKCHANNEL", backchannel=True, salience=0.05)
        return out
    out["sentence_complete"] = bool(re.search(r"[.!?]\s*$", t)) or len(words) >= 6
    if _AUDIENCE.search(low):
        out.update(subject="audience", dialogue_act="AUDIENCE", salience=0.6)
    elif _SCREEN.search(low):
        out.update(subject="screen", dialogue_act="DEMONSTRATION", salience=0.5)
    elif _OBJECT.search(low):
        out.update(subject="object", dialogue_act="DEMONSTRATION", salience=0.6)
    if _CONCLUSION.search(low):
        out.update(dialogue_act="CONCLUSION", transition=True, expected_next="host", salience=0.4)
    elif _TRANSITION.search(low):
        out.update(dialogue_act="TRANSITION", transition=True, salience=0.4)
    if cue is not None:
        if cue.action == Action.SHOW and cue.target_ids:
            out.update(addressed=list(cue.target_ids), expected_next=cue.target_ids[0],
                       dialogue_act="INTRODUCTION" if cue.intent.value == "INTRODUCE" else "QUESTION", salience=max(out["salience"], 0.6))
        elif cue.action == Action.WIDE and cue.target_ids:
            out.update(addressed=list(cue.target_ids), dialogue_act=out["dialogue_act"] if out["dialogue_act"] != "NONE" else "QUESTION", salience=0.5)
        elif cue.action == Action.HOST:
            out.update(addressed=["host"], expected_next="host", dialogue_act="TRANSITION" if out["dialogue_act"] == "NONE" else out["dialogue_act"], transition=True)
        elif cue.target_ids:
            out["references"] = list(cue.target_ids)   # mentioned, not addressed: never a visual owner by itself
    if _PLURAL.search(low) and _QUESTION.search(t):
        out.update(addressed=["all"], dialogue_act="QUESTION", expected_next=None)
    elif _QUESTION.search(t) and out["dialogue_act"] in ("NONE", "TRANSITION"):
        out["dialogue_act"] = "QUESTION"
        if not out["addressed"] and prev_speaker and prev_speaker != "host":
            out["addressed"] = [prev_speaker]            # a follow-up question to whoever just spoke
            out["expected_next"] = prev_speaker
    if _EMOTIONAL.search(low):
        out["salience"] = max(out["salience"], 0.8)
    if _JOKE_CUE.search(low):
        out.update(dialogue_act="JOKE", salience=max(out["salience"], 0.5))
    if out["dialogue_act"] == "NONE":
        out["dialogue_act"] = "EXPLANATION" if len(words) >= 8 else "ANSWER"
    return out


SYSTEM = """You analyse one clause of a live event host's speech for a camera director. Return meaning only.
Roster ids: {roster}. Use "host" for the host, "all" for everyone on stage, "audience" for the crowd.
dialogue_act: INTRODUCTION | QUESTION | ANSWER | REBUTTAL | JOKE | EXPLANATION | DEMONSTRATION | TRANSITION | CONCLUSION | AUDIENCE | BACKCHANNEL
addressed: who the speaker is talking TO right now (ids/host/all/audience), [] if nobody in particular. A person merely talked ABOUT is not addressed.
expected_next: who will most likely speak next (id, host, or "").
backchannel: true for a short acknowledgement (mm-hmm, right, exactly).
sentence_complete: false if the speaker is clearly mid-sentence.
salience: 0..1 how important/emotional the moment is for the audience.
subject: what the audience should look at: "person" | "audience" | "screen" | "object" | "none"."""


class DialogueAnalyzer:
    def __init__(self, roster: Roster, llm_client=None, model: str = "", timeout_s: float = 4.0):
        self.roster = roster
        self._client = llm_client
        self.model = model
        self.timeout_s = timeout_s
        self.stats = {"llm_calls": 0, "llm_errors": 0}

    def set_roster(self, roster: Roster) -> None:
        self.roster = roster

    async def analyze(self, text: str, cue: Cue | None, prev_speaker: str | None) -> dict:
        out = analyze_rules(text, cue, self.roster, prev_speaker)
        if self._client is None or out["backchannel"] or len(text.split()) < 3:
            return out
        try:
            self.stats["llm_calls"] += 1
            r = await asyncio.wait_for(
                self._client.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM.format(roster=sorted(self.roster.ids))},
                              {"role": "user", "content": f"Clause: {text}\nPrevious speaker: {prev_speaker or 'unknown'}\nAnswer with one JSON object."}],
                    response_format=DialogueLLM, max_tokens=200, temperature=0),
                timeout=self.timeout_s)
            p = r.choices[0].message.parsed
            if p is None:
                return out
            valid = self.roster.ids | {"host", "all", "audience"}
            addressed = [a for a in p.addressed if a in valid]
            act = p.dialogue_act.upper()
            if act in ("INTRODUCTION", "QUESTION", "ANSWER", "REBUTTAL", "JOKE", "EXPLANATION", "DEMONSTRATION", "TRANSITION", "CONCLUSION", "AUDIENCE", "BACKCHANNEL"):
                if out["dialogue_act"] in ("EXPLANATION", "ANSWER", "NONE"):
                    out["dialogue_act"] = act
            # the model may ADD an addressee the rules could not see ("what do you think?" after Sarah spoke), never invent an intro cut
            if not out["addressed"] and addressed:
                out["addressed"] = addressed
                if p.expected_next in valid:
                    out["expected_next"] = p.expected_next
            out["salience"] = max(out["salience"], min(1.0, max(0.0, float(p.salience)))) if p.salience else out["salience"]
            if not out["sentence_complete"] or not p.sentence_complete:
                out["sentence_complete"] = False
            if out["subject"] is None and p.subject in ("audience", "screen", "object"):
                out["subject"] = p.subject
            out["source"] = "rules+llm"
        except Exception as e:
            self.stats["llm_errors"] += 1
            log.debug("dialogue LLM failed: %s", e)
        return out
