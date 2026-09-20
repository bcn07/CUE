from server.assembler import Clause, ClauseAssembler, Interim, UtteranceEnd


def results(text, *, is_final, speech_final=False, start=0.0, dur=1.0, words=None):
    return {"type": "Results", "start": start, "duration": dur, "is_final": is_final, "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text, "confidence": 0.9, "words": words or []}]}}


def test_interim_then_two_clauses_in_one_utterance_then_end():
    a = ClauseAssembler(timeout_s=1.5)
    ev = a.feed(results("please wel", is_final=False), now=0.1)
    assert isinstance(ev[0], Interim) and ev[0].text == "please wel"
    ev = a.feed(results("Please welcome Sarah,", is_final=True, start=0.0, dur=1.2), now=0.5)
    assert len(ev) == 1 and isinstance(ev[0], Clause)
    c1 = ev[0]
    assert c1.clause_index == 1 and c1.text == "Please welcome Sarah," and not c1.is_utterance_end
    ev = a.feed(results("actually Daniel.", is_final=True, speech_final=True, start=1.2, dur=0.8), now=1.0)
    assert isinstance(ev[0], Clause) and isinstance(ev[1], UtteranceEnd)
    c2 = ev[0]
    assert c2.utterance_id == c1.utterance_id  # same utterance -> correction re-cut allowed
    assert c2.clause_index == 2 and c2.is_utterance_end
    assert c2.utterance_text == "Please welcome Sarah, actually Daniel."
    assert ev[1].text == "Please welcome Sarah, actually Daniel."
    # next speech opens a new utterance id
    ev = a.feed(results("Thank you.", is_final=True, speech_final=True, start=3.0, dur=0.5), now=3.5)
    assert ev[0].utterance_id != c1.utterance_id


def test_duplicate_final_is_ignored_and_empty_finals_do_not_open():
    a = ClauseAssembler()
    assert a.feed(results("", is_final=True), now=0.0) == []
    assert a.open_utterance_id is None
    ev = a.feed(results("Sarah, come up.", is_final=True), now=0.2)
    assert len(ev) == 1
    assert a.feed(results("Sarah, come up.", is_final=True), now=0.3) == []


def test_utterance_end_message_and_timeout_close():
    a = ClauseAssembler(timeout_s=1.0)
    a.feed(results("Over to Daniel", is_final=True, start=0, dur=1), now=0.0)
    ev = a.feed({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.0}, now=0.5)
    assert isinstance(ev[0], UtteranceEnd) and ev[0].text == "Over to Daniel"
    a.feed(results("And now", is_final=True, start=2, dur=0.5), now=2.0)
    ev = a.feed(results("x", is_final=False), now=3.6)  # timeout closes the stale utterance first
    assert isinstance(ev[0], UtteranceEnd) and ev[0].text == "And now"


def test_audio_span_from_words():
    a = ClauseAssembler()
    words = [{"word": "sarah", "start": 4.1, "end": 4.5}, {"word": "please", "start": 4.6, "end": 4.9}]
    ev = a.feed(results("Sarah please", is_final=True, start=4.0, dur=1.0, words=words), now=5.0)
    assert ev[0].audio_start_s == 4.1 and ev[0].audio_end_s == 4.9


def test_reset_on_new_epoch_drops_in_flight():
    a = ClauseAssembler()
    a.feed(results("Please welcome", is_final=True), now=0.0)
    a.reset(2)
    ev = a.feed(results("Sarah.", is_final=True, speech_final=True), now=0.1)
    assert ev[0].utterance_text == "Sarah." and "e2-" in ev[0].utterance_id
