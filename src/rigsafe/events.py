"""Safety events and a JSON-lines event log (Level 3; a database replaces the file in L4)."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

ZONE_ENTER = "zone_enter"
ZONE_INTRUSION = "zone_intrusion"   # stayed in a zone longer than its dwell limit
ZONE_EXIT = "zone_exit"
PPE_MISSING = "ppe_missing"

SEVERITIES = ("info", "low", "medium", "high", "critical")


@dataclass(frozen=True)
class Event:
    type: str
    t: float                 # seconds from video start
    track_id: int
    severity: str
    zone: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity}")

    @property
    def is_alert(self) -> bool:
        return self.type in (ZONE_INTRUSION, PPE_MISSING)

    def describe(self) -> str:
        where = f" in '{self.zone}'" if self.zone else ""
        extra = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"[{self.t:7.2f}s] {self.severity.upper():8} {self.type} track#{self.track_id}{where} {extra}".rstrip()


class AlertCooldown:
    """Allow an alert for a key at most once per `seconds` (stops flicker spam)."""

    def __init__(self, seconds: float = 10.0):
        self._seconds = seconds
        self._last: dict[Any, float] = {}

    def allow(self, key: Any, t: float) -> bool:
        last = self._last.get(key)
        if last is not None and t - last < self._seconds:
            return False
        self._last[key] = t
        return True


class EventLog:
    """Append events to a .jsonl file (one JSON object per line) and count them."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: TextIO = path.open("w", encoding="utf-8")
        self.counts: Counter[str] = Counter()

    def write(self, event: Event) -> None:
        self._fh.write(json.dumps(asdict(event)) + "\n")
        self._fh.flush()  # don't lose events if the process is killed
        self.counts[event.type] += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()
