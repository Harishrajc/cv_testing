"""PPE class semantics for the Construction-PPE model (Level 2)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rigsafe.detector import Detection

# Class names as they appear in Ultralytics' construction-ppe.yaml
VIOLATION_LABELS = frozenset({"no_helmet", "no_goggle", "no_gloves", "no_boots", "none"})
PERSON_LABELS = frozenset({"Person", "person"})

COLOR_OK = (0, 200, 0)          # BGR green: PPE worn
COLOR_VIOLATION = (0, 0, 255)   # red: PPE missing
COLOR_PERSON = (255, 160, 0)    # blue-ish: the worker box itself


def is_violation(d: Detection) -> bool:
    return d.label in VIOLATION_LABELS


def ppe_color(d: Detection) -> tuple[int, int, int]:
    if is_violation(d):
        return COLOR_VIOLATION
    if d.label in PERSON_LABELS:
        return COLOR_PERSON
    return COLOR_OK


@dataclass(frozen=True)
class PpeSummary:
    people: int
    violations: int
    counts: dict[str, int]

    def hud_lines(self) -> list[str]:
        missing = ", ".join(f"{k}:{v}" for k, v in sorted(self.counts.items()) if k in VIOLATION_LABELS)
        return [
            f"Workers: {self.people}",
            f"Violations: {self.violations}" + (f" ({missing})" if missing else ""),
        ]


def summarize(detections: list[Detection]) -> PpeSummary:
    counts = Counter(d.label for d in detections)
    return PpeSummary(
        people=sum(counts[label] for label in PERSON_LABELS),
        violations=sum(counts[label] for label in VIOLATION_LABELS),
        counts=dict(counts),
    )
