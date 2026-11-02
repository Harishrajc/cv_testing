"""Danger zones and dwell-time intrusion logic (Level 3).

Zone polygons are stored in normalized 0..1 coordinates so the same JSON works
whatever resolution the frames are processed at.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from rigsafe.compliance import EDGE_MARGIN_PX, MISSING
from rigsafe.detector import Detection
from rigsafe.events import SEVERITIES, ZONE_ENTER, ZONE_EXIT, ZONE_INTRUSION, Event
from rigsafe.geometry import Point, point_in_polygon


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[Point, ...]
    min_dwell_s: float = 2.0
    severity: str = "high"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("zone needs a name")
        if len(self.polygon) < 3:
            raise ValueError(f"zone '{self.name}' needs at least 3 points")
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in self.polygon):
            raise ValueError(f"zone '{self.name}' points must be normalized to 0..1")
        if self.min_dwell_s < 0:
            raise ValueError("min_dwell_s must be >= 0")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity}")

    def contains(self, pt: Point) -> bool:
        return point_in_polygon(pt, self.polygon)


def load_zones(path: Path) -> list[Zone]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Zone(
            name=z["name"],
            polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
            min_dwell_s=float(z.get("min_dwell_s", 2.0)),
            severity=z.get("severity", "high"),
        )
        for z in raw["zones"]
    ]


def save_zones(path: Path, zones: list[Zone]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"zones": [
        {"name": z.name, "polygon": [[round(x, 4), round(y, 4)] for x, y in z.polygon],
         "min_dwell_s": z.min_dwell_s, "severity": z.severity}
        for z in zones
    ]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def feet_visible(person: Detection, frame_height: int, margin_px: int = EDGE_MARGIN_PX) -> bool:
    """A box that reaches the bottom edge is cut off: its 'foot point' is really mid-body."""
    return person.y2 < frame_height - margin_px


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    foot: Point          # normalized 0..1
    helmet: str          # compliance status: ok / missing / unknown


@dataclass(frozen=True)
class _Presence:
    entered_at: float
    last_inside: float
    alerted: bool = False


class ZoneMonitor:
    """Tracks who is in which zone and for how long; emits enter / intrusion / exit events.

    exit_grace_s: a person must be outside (or lost by the tracker) this long before we
    call it an exit. Stops one flickering frame from resetting the dwell timer.
    """

    def __init__(self, zones: list[Zone], exit_grace_s: float = 1.0):
        names = [z.name for z in zones]
        if len(names) != len(set(names)):
            raise ValueError("zone names must be unique")
        self._zones = zones
        self._grace = exit_grace_s
        self._state: dict[tuple[int, str], _Presence] = {}

    @property
    def zones(self) -> list[Zone]:
        return list(self._zones)

    def update(self, t: float, people: list[TrackedPerson]) -> list[Event]:
        events = []
        for person in people:
            for zone in self._zones:
                if zone.contains(person.foot):
                    events.extend(self._inside(t, person, zone))
        events.extend(self._expire(t))
        return events

    def dwell(self, track_id: int, zone_name: str, t: float) -> float | None:
        st = self._state.get((track_id, zone_name))
        return None if st is None else t - st.entered_at

    def occupied(self) -> set[str]:
        return {zone for _, zone in self._state}

    def _inside(self, t: float, person: TrackedPerson, zone: Zone) -> list[Event]:
        key = (person.track_id, zone.name)
        st = self._state.get(key)
        events = []
        if st is None:
            st = _Presence(entered_at=t, last_inside=t)
            events.append(Event(ZONE_ENTER, t, person.track_id, "info", zone.name))
        st = replace(st, last_inside=t)
        dwell = t - st.entered_at
        if not st.alerted and dwell >= zone.min_dwell_s:
            severity = "critical" if person.helmet == MISSING else zone.severity
            events.append(Event(ZONE_INTRUSION, t, person.track_id, severity, zone.name,
                                {"dwell_s": round(dwell, 2), "helmet": person.helmet}))
            st = replace(st, alerted=True)
        self._state[key] = st
        return events

    def _expire(self, t: float) -> list[Event]:
        gone = [k for k, st in self._state.items() if t - st.last_inside > self._grace]
        events = []
        for track_id, zone_name in gone:
            st = self._state.pop((track_id, zone_name))
            events.append(Event(ZONE_EXIT, t, track_id, "info", zone_name,
                                {"dwell_s": round(st.last_inside - st.entered_at, 2)}))
        return events
