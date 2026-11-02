import json

import pytest

from rigsafe.events import PPE_MISSING, ZONE_ENTER, Event, EventLog


def test_event_rejects_unknown_severity():
    with pytest.raises(ValueError):
        Event(ZONE_ENTER, 0.0, 1, "panic")


def test_alert_flag_and_describe():
    e = Event(PPE_MISSING, 3.25, 4, "medium", details={"item": "helmet"})
    assert e.is_alert
    assert "track#4" in e.describe() and "item=helmet" in e.describe()
    assert not Event(ZONE_ENTER, 0.0, 1, "info", "red").is_alert


def test_event_log_writes_jsonl_and_counts(tmp_path):
    path = tmp_path / "sub" / "events.jsonl"
    with EventLog(path) as log:
        log.write(Event(ZONE_ENTER, 1.0, 1, "info", "red"))
        log.write(Event(PPE_MISSING, 2.0, 1, "medium", details={"item": "helmet"}))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [ln["type"] for ln in lines] == [ZONE_ENTER, PPE_MISSING]
    assert lines[1]["details"] == {"item": "helmet"}
    assert log.counts[PPE_MISSING] == 1


def test_alert_cooldown_blocks_repeats_within_window():
    from rigsafe.events import AlertCooldown
    cd = AlertCooldown(seconds=10)
    assert cd.allow((1, "helmet"), 0.0)
    assert not cd.allow((1, "helmet"), 5.0)
    assert cd.allow((2, "helmet"), 5.0)       # different worker
    assert cd.allow((1, "helmet"), 10.5)
