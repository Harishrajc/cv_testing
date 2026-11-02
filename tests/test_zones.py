import pytest

from rigsafe.compliance import MISSING, OK
from rigsafe.events import ZONE_ENTER, ZONE_EXIT, ZONE_INTRUSION
from rigsafe.zones import TrackedPerson, Zone, ZoneMonitor, feet_visible, load_zones, save_zones

BOX = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
INSIDE, OUTSIDE = (0.5, 0.5), (0.9, 0.9)


def zone(**kw):
    return Zone(name=kw.pop("name", "red"), polygon=BOX, **kw)


def run(monitor, steps):
    """steps: list of (t, [TrackedPerson...]) -> flat list of events."""
    events = []
    for t, people in steps:
        events += monitor.update(t, people)
    return events


def p(foot, helmet=OK, tid=1):
    return TrackedPerson(tid, foot, helmet)


def test_feet_visible_only_when_box_clear_of_bottom_edge():
    from rigsafe.detector import Detection
    assert feet_visible(Detection(0, 0, 10, 90, 0.9, 0, "Person"), frame_height=100)
    assert not feet_visible(Detection(0, 0, 10, 99, 0.9, 0, "Person"), frame_height=100)


def test_zone_validation():
    with pytest.raises(ValueError):
        Zone("x", ((0, 0), (1, 1)))
    with pytest.raises(ValueError):
        Zone("x", ((0, 0), (2, 0), (0, 1)))   # not normalized
    with pytest.raises(ValueError):
        Zone("x", BOX, severity="scary")
    with pytest.raises(ValueError):
        ZoneMonitor([zone(), zone()])          # duplicate names


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "zones.json"
    save_zones(path, [zone(min_dwell_s=3.5, severity="medium")])
    [z] = load_zones(path)
    assert z.name == "red" and z.min_dwell_s == 3.5 and z.severity == "medium"
    assert z.contains(INSIDE)


def test_short_visit_enters_and_exits_without_alert():
    m = ZoneMonitor([zone(min_dwell_s=2.0)], exit_grace_s=0.5)
    events = run(m, [(0.0, [p(INSIDE)]), (1.0, [p(INSIDE)]), (1.2, [p(OUTSIDE)]), (2.0, [p(OUTSIDE)])])
    assert [e.type for e in events] == [ZONE_ENTER, ZONE_EXIT]


def test_dwell_over_limit_raises_one_intrusion():
    m = ZoneMonitor([zone(min_dwell_s=2.0, severity="high")])
    events = run(m, [(t / 2, [p(INSIDE)]) for t in range(10)])  # 0..4.5 s inside
    intrusions = [e for e in events if e.type == ZONE_INTRUSION]
    assert len(intrusions) == 1
    assert intrusions[0].t == 2.0 and intrusions[0].severity == "high"


def test_missing_helmet_escalates_to_critical():
    m = ZoneMonitor([zone(min_dwell_s=0.0, severity="high")])
    [enter, intrusion] = m.update(0.0, [p(INSIDE, helmet=MISSING)])
    assert intrusion.severity == "critical"
    assert intrusion.details["helmet"] == MISSING


def test_flicker_within_grace_keeps_timer():
    m = ZoneMonitor([zone(min_dwell_s=2.0)], exit_grace_s=1.0)
    events = run(m, [(0.0, [p(INSIDE)]), (0.5, [p(OUTSIDE)]), (1.0, []), (2.0, [p(INSIDE)])])
    assert [e.type for e in events] == [ZONE_ENTER, ZONE_INTRUSION]
    assert m.dwell(1, "red", 2.0) == 2.0


def test_lost_track_exits_after_grace():
    m = ZoneMonitor([zone()], exit_grace_s=1.0)
    m.update(0.0, [p(INSIDE)])
    assert m.occupied() == {"red"}
    [exit_event] = m.update(1.5, [])
    assert exit_event.type == ZONE_EXIT
    assert m.occupied() == set()
    assert m.dwell(1, "red", 1.5) is None


def test_two_people_tracked_independently():
    m = ZoneMonitor([zone(min_dwell_s=1.0)])
    events = run(m, [(0.0, [p(INSIDE, tid=1)]), (0.5, [p(INSIDE, tid=1), p(INSIDE, tid=2)]),
                     (1.0, [p(INSIDE, tid=1), p(INSIDE, tid=2)])])
    intrusions = [e.track_id for e in events if e.type == ZONE_INTRUSION]
    assert intrusions == [1]
