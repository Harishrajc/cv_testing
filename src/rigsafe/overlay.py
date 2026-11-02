"""Drawing for zones and tracked workers (Level 3)."""
from __future__ import annotations

import cv2
import numpy as np

from rigsafe.compliance import MISSING, OK
from rigsafe.detector import Detection
from rigsafe.zones import Zone

ZONE_IDLE = (0, 200, 255)      # BGR amber
ZONE_ACTIVE = (0, 0, 255)      # red when someone is inside
ZONE_ALPHA = 0.25
HELMET_COLORS = {OK: (0, 200, 0), MISSING: (0, 0, 255)}
UNKNOWN_COLOR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def zone_pixels(zone: Zone, width: int, height: int) -> np.ndarray:
    return np.array([[int(x * width), int(y * height)] for x, y in zone.polygon], dtype=np.int32)


def draw_zones(frame: np.ndarray, zones: list[Zone], occupied: set[str]) -> np.ndarray:
    """Semi-transparent filled polygons; red when occupied."""
    h, w = frame.shape[:2]
    fill = frame.copy()
    out = frame.copy()
    for zone in zones:
        pts = zone_pixels(zone, w, h)
        color = ZONE_ACTIVE if zone.name in occupied else ZONE_IDLE
        cv2.fillPoly(fill, [pts], color)
        cv2.polylines(out, [pts], True, color, 2)
    out = cv2.addWeighted(fill, ZONE_ALPHA, out, 1 - ZONE_ALPHA, 0)
    for zone in zones:
        x, y = zone_pixels(zone, w, h)[0]
        cv2.putText(out, zone.name, (int(x) + 4, int(y) + 18), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_worker(frame: np.ndarray, person: Detection, helmet: str, dwell_s: float | None) -> None:
    """Draw one tracked worker in place (frame should already be a copy)."""
    color = HELMET_COLORS.get(helmet, UNKNOWN_COLOR)
    cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), color, 2)
    label = f"#{person.track_id} helmet:{helmet}"
    if dwell_s is not None:
        label += f" {dwell_s:.1f}s"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (person.x1, person.y1 - th - 6), (person.x1 + tw + 4, person.y1), color, -1)
    cv2.putText(frame, label, (person.x1 + 2, person.y1 - 4), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, person.foot_point, 5, (255, 0, 255), -1)


def draw_alert_banner(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 36), (w, h), (0, 0, 200), -1)
    cv2.putText(out, text, (10, h - 11), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out
