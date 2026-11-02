import pytest

from rigsafe.compliance import MISSING, OK, UNKNOWN, ComplianceTracker, PpeObservation, observe
from rigsafe.detector import Detection

PERSON = Detection(100, 100, 200, 400, 0.9, 6, "Person", track_id=7)


def item(label, x1, y1, x2, y2):
    return Detection(x1, y1, x2, y2, 0.8, 0, label)


def test_helmet_on_head_is_worn():
    [obs] = observe([PERSON], [item("helmet", 130, 90, 170, 130)])
    assert obs.track_id == 7
    assert obs.worn["helmet"] is True


def test_helmet_at_feet_does_not_count():
    [obs] = observe([PERSON], [item("helmet", 130, 360, 170, 400)])  # e.g. helmet on the ground
    assert obs.worn["helmet"] is False


def test_no_helmet_box_overrides_helmet_box():
    [obs] = observe([PERSON], [item("helmet", 130, 90, 170, 130), item("no_helmet", 130, 95, 170, 135)])
    assert obs.worn["helmet"] is False


def test_vest_on_torso():
    [obs] = observe([PERSON], [item("vest", 110, 180, 190, 300)])
    assert obs.worn["vest"] is True


def test_head_cut_off_by_frame_top_is_no_evidence():
    at_top = Detection(100, 0, 200, 400, 0.9, 6, "Person", track_id=3)
    [obs] = observe([at_top], [])
    assert obs.worn["helmet"] is None
    assert obs.worn["vest"] is False   # torso still visible, so absence counts


def test_tiny_person_is_no_evidence():
    tiny = Detection(100, 100, 110, 120, 0.9, 6, "Person", track_id=3)
    [obs] = observe([tiny], [])
    assert obs.worn == {"helmet": None, "vest": None}


def test_no_evidence_frames_do_not_change_verdict():
    ct = ComplianceTracker(window=10, min_frames=3)
    for _ in range(3):
        ct.update([obs(1, True)])
    for _ in range(10):
        ct.update([obs(1, None)])
    assert ct.status(1) == OK


def test_untracked_person_is_skipped():
    untracked = Detection(0, 0, 10, 10, 0.9, 6, "Person", track_id=None)
    assert observe([untracked], []) == []


def obs(track_id, helmet):
    return PpeObservation(track_id, {"helmet": helmet})


def test_status_unknown_until_enough_frames():
    ct = ComplianceTracker(window=10, min_frames=5)
    for _ in range(4):
        ct.update([obs(1, False)])
    assert ct.status(1) == UNKNOWN


def test_majority_vote_and_single_change_event():
    ct = ComplianceTracker(window=10, min_frames=5, missing_ratio=0.6)
    changes = []
    for _ in range(6):
        changes += ct.update([obs(1, False)])
    assert ct.status(1) == MISSING
    assert changes == [(1, "helmet", MISSING)]  # reported once, not every frame


def test_one_bad_frame_does_not_flip_ok_worker():
    ct = ComplianceTracker(window=10, min_frames=5)
    for _ in range(8):
        ct.update([obs(1, True)])
    ct.update([obs(1, False)])
    assert ct.status(1) == OK


def test_stale_tracks_are_forgotten():
    ct = ComplianceTracker(window=5, min_frames=1, forget_after=3)
    ct.update([obs(1, True)])
    for _ in range(4):
        ct.update([])
    assert ct.status(1) == UNKNOWN


def test_rejects_bad_window():
    with pytest.raises(ValueError):
        ComplianceTracker(window=3, min_frames=5)


def test_dedupe_keeps_confident_box_drops_nested_duplicate():
    from rigsafe.compliance import dedupe_persons
    tight = Detection(100, 100, 200, 300, 0.9, 6, "Person", track_id=1)
    loose = Detection(90, 60, 220, 310, 0.5, 6, "Person", track_id=2)
    other = Detection(400, 100, 500, 300, 0.6, 6, "Person", track_id=3)
    assert [d.track_id for d in dedupe_persons([loose, tight, other])] == [1, 3]


def test_crouching_worker_helmet_counts():
    crouch = Detection(445, 235, 535, 360, 0.9, 6, "Person", track_id=8)  # real case from site2.mp4
    [obs] = observe([crouch], [item("helmet", 478, 272, 508, 300)])
    assert obs.worn["helmet"] is True
