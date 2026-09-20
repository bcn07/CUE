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
