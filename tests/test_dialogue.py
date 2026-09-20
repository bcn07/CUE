from server.dialogue import analyze_rules
from server.semantics import Person, Roster, RuleParser, validate

R = Roster([Person("host", "Jordan Lee", ["Jordan"], "host", True), Person("sarah", "Sarah Tan", ["Sarah"], "guest"), Person("daniel", "Daniel Reyes", ["Daniel"], "speaker")])
P = RuleParser(R)


def a(text, prev=None):
    return analyze_rules(text, validate(P.parse(text), R), R, prev)


def test_backchannel_and_short_acks():
    for t in ["Mm-hmm.", "Yeah.", "Exactly.", "Right, right."]:
        r = a(t)
        assert r["backchannel"] and r["dialogue_act"] == "BACKCHANNEL", (t, r)


def test_audience_screen_object_subjects():
    assert a("A big round of applause for everyone here tonight!")["subject"] == "audience"
    assert a("Take a look at the screen behind me.")["subject"] == "screen"
    for line in ("Let's look at the TV.", "Everyone, look at the screen for a second.", "Eyes up on the screen please.",
                 "If you look at that monitor you'll see the numbers.", "It's on the big screen now."):
        assert a(line)["subject"] == "screen" and a(line)["dialogue_act"] == "DEMONSTRATION", line
    for line in ("We watched the TV yesterday.", "I bought a new television.", "Sarah looked at me and laughed."):
        assert a(line)["subject"] is None, line
    assert a("Here's how the prototype works.")["subject"] == "object"


def test_introduction_and_question_addressees_from_the_cue():
    r = a("Please welcome Sarah Tan!")
    assert r["dialogue_act"] == "INTRODUCTION" and r["addressed"] == ["sarah"] and r["expected_next"] == "sarah"
    r = a("Sarah, what do you think?")
    assert r["dialogue_act"] == "QUESTION" and r["addressed"] == ["sarah"]


def test_mention_is_a_reference_not_an_addressee():
    r = a("Earlier, Sarah made a great point about accessibility, and I want to build on it.")
    assert r["addressed"] == [] and r["references"] == ["sarah"]


def test_follow_up_question_without_a_name_goes_to_the_previous_speaker():
    r = a("And how did that change your approach?", prev="daniel")
    assert r["dialogue_act"] == "QUESTION" and r["addressed"] == ["daniel"] and r["expected_next"] == "daniel"
    r = a("How would both of you approach this differently?", prev="daniel")
    assert r["addressed"] == ["all"]


def test_transition_and_conclusion():
    assert a("That brings us to our final question.")["transition"]
    r = a("Thank you both, we'll be right back.")
    assert r["dialogue_act"] == "CONCLUSION" and r["expected_next"] == "host"


def test_emotional_salience():
    assert a("Honestly, that was the hardest year of my life, and my mother never saw it finished.")["salience"] >= 0.8


def test_look_at_phrase_becomes_a_subject_phrase_but_never_a_person_or_pronoun():
    from server.semantics import Person, Roster
    roster = Roster([Person("sarah", "Sarah Tan", ["Sara"]), Person("host", "Brian N", [], is_host=True)])
    a = lambda t: analyze_rules(t, None, roster, None)  # noqa: E731
    r = a("Let's look at the flowers over there.")
    assert (r["subject_phrase"], r["subject"], r["dialogue_act"]) == ("flowers", "object", "DEMONSTRATION")
    assert a("Take a look at the TV.")["subject_phrase"] == "tv" and a("Take a look at the TV.")["subject"] == "screen"
    assert a("Now let's look at the audience for a second.")["subject_phrase"] == "audience" and a("Now let's look at the audience for a second.")["subject"] == "audience"
    assert a("Zoom in on the whiteboard, please.")["subject_phrase"] == "whiteboard"
    assert a("Look at Sarah!")["subject_phrase"] is None and a("Look at Sarah!")["subject"] is None
    assert a("Look at me.")["subject_phrase"] is None and a("Look at that.")["subject_phrase"] is None
    assert a("She looked at the report yesterday.")["subject_phrase"] is None
