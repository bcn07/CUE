from server import director as D
from server.semantics import Action, Cue, Intent, Scope, Temporal

TUN = D.Tunables(min_shot_s=2.5, cue_lifetime_s=3.0, identity_max_age_s=4.0)


def cams(**over):
    base = {
        "A": D.CameraView("A", "wide", True, {}, None),
        "B": D.CameraView("B", "host", True, {"host": 0.3}, None),
        "C": D.CameraView("C", "guest", True, {"sarah": 0.5}, "sarah"),
    }
    for k, v in over.items():
        base[k] = v
    return base


def cue(action, targets, temporal=Temporal.NOW, utt="u1", created=10.0, scope=None):
    scope = scope or (Scope.SINGLE if len(targets) == 1 else (Scope.GROUP if targets else Scope.NONE))
    return Cue(list(targets), scope, Intent.INTRODUCE, temporal, action, "ev", utterance_id=utt, created_at=created)


def test_show_uses_fresh_identity():
    st = D.State(current_camera="A", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "C" and d.evidence == "identity"
    assert D.apply(st, d, cue(Action.SHOW, ["sarah"]), 10.0)
    assert st.current_camera == "C"


def test_stale_identity_falls_back_to_static_mapping_then_wide():
    c = cams(C=D.CameraView("C", "guest", True, {"sarah": 9.0}, "sarah"))
    st = D.State(current_camera="B", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "C" and d.evidence == "static"
    c = cams(C=D.CameraView("C", "guest", True, {}, None))
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "A" and d.evidence == "wide"


def test_non_now_never_cuts():
    st = D.State(current_camera="A", last_cut_time=0.0)
    for t in (Temporal.FUTURE, Temporal.PAST, Temporal.NEGATED, Temporal.UNCERTAIN):
        d = D.decide(cue(Action.SHOW, ["sarah"], temporal=t), cams(), st, 10.0, host_id="host", tun=TUN)
        assert d.action == D.Take.STAY and d.camera_id == "A", t


def test_hold_cue_stays():
    st = D.State(current_camera="C", last_cut_time=0.0)
    d = D.decide(cue(Action.HOLD, ["sarah"]), cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and d.camera_id == "C"


def test_min_shot_blocks_then_correction_recut_allowed_once():
    st = D.State(current_camera="A", last_cut_time=0.0)
    c = cams(C=D.CameraView("C", "guest", True, {"sarah": 0.2}, None),
             B=D.CameraView("B", "guest", True, {"host": 0.1}, None))
    c["D"] = D.CameraView("D", "guest", True, {"daniel": 0.2}, None)
    k1 = cue(Action.SHOW, ["sarah"], utt="u7", created=10.0)
    d = D.decide(k1, c, st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "C"
    D.apply(st, d, k1, 10.0)
    # 0.6 s later, same utterance corrects to daniel -> re-cut allowed
    k2 = cue(Action.SHOW, ["daniel"], utt="u7", created=10.6)
    d = D.decide(k2, c, st, 10.6, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "D" and "correction" in d.reason
    D.apply(st, d, k2, 10.6)
    # a second correction in the same utterance is NOT allowed (one fast re-cut only)
    k3 = cue(Action.SHOW, ["sarah"], utt="u7", created=11.0)
    d = D.decide(k3, c, st, 11.0, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and "min shot" in d.reason
    # a different utterance inside the min-shot window is held
    k4 = cue(Action.SHOW, ["sarah"], utt="u8", created=11.5)
    d = D.decide(k4, c, st, 11.5, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and "min shot" in d.reason
    # after 2.5 s it cuts
    d = D.decide(k4, c, st, 13.2, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "C"


def test_manual_hold_beats_cue_but_failover_still_runs():
    st = D.State(current_camera="C", last_cut_time=0.0, mode=D.Mode.HOLD)
    d = D.decide(cue(Action.SHOW, ["sarah"]), cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and "HOLD" in d.reason
    c = cams(C=D.CameraView("C", "guest", False, {}, "sarah"))
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "A"


def test_stale_cue_rejected():
    st = D.State(current_camera="A", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"], created=5.0), cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and "stale" in d.reason


def test_group_and_wide_go_wide():
    st = D.State(current_camera="C", last_cut_time=0.0)
    d = D.decide(cue(Action.WIDE, ["sarah", "daniel"]), cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "A" and d.evidence == "wide"


def test_host_action_prefers_identity_then_role():
    st = D.State(current_camera="C", last_cut_time=0.0)
    d = D.decide(cue(Action.HOST, ["host"]), cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "B" and d.evidence == "identity"
    c = cams(B=D.CameraView("B", "host", True, {}, None))
    d = D.decide(cue(Action.HOST, ["host"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "B" and d.evidence == "host-role"


def test_no_healthy_camera_is_slate_and_recovers():
    st = D.State(current_camera="C", last_cut_time=0.0)
    dead = {k: D.CameraView(k, v.role, False, {}, v.fixed_person) for k, v in cams().items()}
    d = D.decide(None, dead, st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.SLATE
    assert D.apply(st, d, None, 10.0) and st.current_camera is None
    d = D.decide(None, cams(), st, 11.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "A"


def test_identity_prefers_close_up_over_wide_seeing_same_face():
    c = cams(A=D.CameraView("A", "wide", True, {"sarah": 0.1}, None), C=D.CameraView("C", "guest", True, {"sarah": 0.9}, None))
    st = D.State(current_camera="B", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "C"


def test_cancel_right_after_cut_returns_to_wide_once():
    st = D.State(current_camera="A", last_cut_time=0.0)
    k1 = cue(Action.SHOW, ["sarah"], utt="u9", created=10.0)
    d = D.decide(k1, cams(), st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "C"
    D.apply(st, d, k1, 10.0)
    k2 = Cue(["sarah"], Scope.NONE, Intent.CANCEL, Temporal.NOW, Action.HOLD, "actually wait", utterance_id="u9", created_at=10.8)
    d = D.decide(k2, cams(), st, 10.8, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "A" and "cancel" in d.reason.lower()
    D.apply(st, d, k2, 10.8)
    # a cancel long after the cut, or in another utterance, does nothing
    k3 = Cue([], Scope.NONE, Intent.CANCEL, Temporal.NOW, Action.HOLD, "hold on", utterance_id="u10", created_at=14.0)
    d = D.decide(k3, cams(), st, 14.0, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY


def test_same_target_does_not_burn_the_correction_recut():
    """Rules-fast cuts to C for Sarah; the LLM agrees 400 ms later while D also sees Sarah:
    stay on C (no second cut) and keep the one re-cut available for a real correction."""
    st = D.State(current_camera="A", last_cut_time=0.0)
    c = cams(C=D.CameraView("C", "guest", True, {"sarah": 0.3}, None))
    c["D"] = D.CameraView("D", "guest", True, {"sarah": 0.1, "daniel": 0.1}, None)
    k1 = cue(Action.SHOW, ["sarah"], utt="u3", created=10.0)
    d = D.decide(k1, c, st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "D"  # freshest
    D.apply(st, d, k1, 10.0)
    c["C"] = D.CameraView("C", "guest", True, {"sarah": 0.05}, None)  # C now fresher than D
    d = D.decide(cue(Action.SHOW, ["sarah"], utt="u3", created=10.4), c, st, 10.4, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and d.camera_id == "D" and "still on live camera" in d.reason
    assert st.recuts_in_utterance == 0
    k3 = cue(Action.SHOW, ["daniel"], utt="u3", created=10.8)
    d = D.decide(k3, c, st, 10.8, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "D" or d.action == D.Take.STAY  # daniel is also on D: already live


def test_static_mapping_refused_when_camera_shows_someone_else():
    c = cams(C=D.CameraView("C", "guest", True, {"daniel": 0.1}, "sarah"))
    st = D.State(current_camera="B", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "A" and d.evidence == "wide" and "refused" in d.reason


def test_host_role_camera_refused_when_guest_confirmed_on_it():
    c = cams(B=D.CameraView("B", "host", True, {"sarah": 0.2}, None), C=D.CameraView("C", "guest", True, {}, None))
    st = D.State(current_camera="C", last_cut_time=0.0)
    d = D.decide(cue(Action.HOST, ["host"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "A" and d.evidence == "wide"


def test_live_wide_camera_never_absorbs_a_show_as_already_live():
    """The wide sees everyone; being live on it must not satisfy 'SHOW sarah'."""
    c = cams(A=D.CameraView("A", "wide", True, {"sarah": 0.1, "daniel": 0.1, "host": 0.1}, None),
             C=D.CameraView("C", "guest", True, {"sarah": 0.4}, None))
    st = D.State(current_camera="A", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.TAKE and d.camera_id == "C"
    # host-role camera live and seeing a guest is not 'already showing' that guest either
    c = cams(B=D.CameraView("B", "host", True, {"sarah": 0.1, "host": 0.1}, None), C=D.CameraView("C", "guest", True, {"sarah": 0.3}, None))
    st = D.State(current_camera="B", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.camera_id == "C"
    # but a live guest close-up that still shows the target is kept
    st = D.State(current_camera="C", last_cut_time=0.0)
    d = D.decide(cue(Action.SHOW, ["sarah"]), c, st, 10.0, host_id="host", tun=TUN)
    assert d.action == D.Take.STAY and d.camera_id == "C"


def test_host_role_camera_refused_without_a_host_in_the_roster_when_someone_is_on_it():
    c = cams(B=D.CameraView("B", "host", True, {"sarah": 0.2}, None), C=D.CameraView("C", "guest", True, {}, None))
    st = D.State(current_camera="C", last_cut_time=0.0)
    d = D.decide(Cue([], Scope.NONE, Intent.RETURN_HOST, Temporal.NOW, Action.HOST, "thank you", utterance_id="u", created_at=10.0),
                 c, st, 10.0, host_id=None, tun=TUN)
    assert d.camera_id == "A" and d.evidence == "wide"
