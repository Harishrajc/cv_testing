"""Per-worker PPE compliance from tracked detections (Level 3).

The trained model detects worn PPE well (helmet AP50 0.94) but "no_helmet" poorly (0.17),
so a violation is inferred from *absence*: no helmet box over a tracked person's head
in most of the recent frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from rigsafe.detector import Detection
from rigsafe.geometry import box_center, contains, overlap_of_smaller, sub_region

OK, MISSING, UNKNOWN = "ok", "missing", "unknown"

# Where each item should sit on a person box, as (top, bottom) fractions of its height
ITEM_REGIONS: dict[str, tuple[float, float]] = {
    # slightly above the box (helmets poke out) down to 45%: a crouching worker's head
    # sits ~40% down the box, a standing worker's ~10%
    "helmet": (-0.10, 0.45),
    "vest": (0.15, 0.75),
}
NEGATIVE_LABELS = {"helmet": "no_helmet", "vest": None}
REGION_PAD_X = 0.10
EDGE_MARGIN_PX = 4        # box touching the frame edge = body part may be out of view
MIN_PERSON_HEIGHT_PX = 40  # smaller than this: too few pixels to judge PPE


@dataclass(frozen=True)
class PpeObservation:
    track_id: int
    worn: dict[str, bool | None]   # item -> seen this frame; None = can't tell (no evidence)


def _box(d: Detection) -> tuple[float, float, float, float]:
    return float(d.x1), float(d.y1), float(d.x2), float(d.y2)


def _observable(person: Detection, item: str) -> bool:
    if person.y2 - person.y1 < MIN_PERSON_HEIGHT_PX:
        return False
    if item == "helmet" and person.y1 <= EDGE_MARGIN_PX:
        return False   # head is cut off by the top of the frame
    return True


def dedupe_persons(persons: list[Detection], max_overlap: float = 0.7) -> list[Detection]:
    """Drop a Person box mostly inside a more confident one (same worker detected twice)."""
    kept: list[Detection] = []
    for p in sorted(persons, key=lambda d: d.confidence, reverse=True):
        if all(overlap_of_smaller(_box(p), _box(k)) < max_overlap for k in kept):
            kept.append(p)
    return kept


def observe(persons: list[Detection], items: list[Detection]) -> list[PpeObservation]:
    """For each tracked person, which PPE items have a box centred in the right body region.

    No evidence is not a violation: when the body part can't be seen, the item is None.
    """
    observations = []
    for person in persons:
        if person.track_id is None:
            continue
        worn: dict[str, bool | None] = {}
        for item, (top, bottom) in ITEM_REGIONS.items():
            if not _observable(person, item):
                worn[item] = None
                continue
            region = sub_region(_box(person), top, bottom, REGION_PAD_X)
            centres = [box_center(_box(d)) for d in items if d.label == item]
            negatives = [box_center(_box(d)) for d in items if d.label == NEGATIVE_LABELS[item]]
            seen = any(contains(region, c) for c in centres)
            refuted = any(contains(region, c) for c in negatives)
            worn[item] = seen and not refuted
        observations.append(PpeObservation(person.track_id, worn))
    return observations


class ComplianceTracker:
    """Majority vote over the last `window` frames per (track, item), so one bad frame can't flip it."""

    def __init__(self, window: int = 15, min_frames: int = 8, missing_ratio: float = 0.6,
                 forget_after: int = 300):
        if not 0 < min_frames <= window:
            raise ValueError("need 0 < min_frames <= window")
        self._window = window
        self._min_frames = min_frames
        self._missing_ratio = missing_ratio
        self._forget_after = forget_after
        self._history: dict[tuple[int, str], deque[bool]] = {}
        self._status: dict[tuple[int, str], str] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def update(self, observations: list[PpeObservation]) -> list[tuple[int, str, str]]:
        """Feed one frame. Returns status changes as (track_id, item, new_status)."""
        self._frame += 1
        changes = []
        for obs in observations:
            self._last_seen[obs.track_id] = self._frame
            for item, worn in obs.worn.items():
                if worn is None:
                    continue   # no evidence this frame: keep the previous verdict
                key = (obs.track_id, item)
                hist = self._history.setdefault(key, deque(maxlen=self._window))
                hist.append(worn)
                new = self._decide(hist)
                if new != self._status.get(key, UNKNOWN):
                    changes.append((obs.track_id, item, new))
                self._status[key] = new
        self._forget_stale()
        return changes

    def status(self, track_id: int, item: str = "helmet") -> str:
        return self._status.get((track_id, item), UNKNOWN)

    def _decide(self, hist: deque[bool]) -> str:
        if len(hist) < self._min_frames:
            return UNKNOWN
        missing = hist.count(False) / len(hist)
        return MISSING if missing >= self._missing_ratio else OK

    def _forget_stale(self) -> None:
        stale = [tid for tid, f in self._last_seen.items() if self._frame - f > self._forget_after]
        for tid in stale:
            del self._last_seen[tid]
            for key in [k for k in self._history if k[0] == tid]:
                self._history.pop(key, None)
                self._status.pop(key, None)
