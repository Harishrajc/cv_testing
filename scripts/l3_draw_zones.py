#!/usr/bin/env python3
"""Level 3: draw danger zones on the first frame of a video and save them as JSON.

Controls (in the image window):
    left click   add a point
    u            undo last point
    Enter        finish current zone (then type its name in the terminal)
    s            save all zones and quit
    q / Esc      quit without saving

Usage:
    ./scripts/l3_draw_zones.py --source data/videos/site.mp4
    -> writes configs/zones/site.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from rigsafe.events import SEVERITIES
from rigsafe.video_io import open_video, resize_to_width
from rigsafe.zones import Zone, save_zones

WINDOW = "RigSafe zone editor"
KEY_ENTER = (13, 10)
KEY_ESC = 27


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default=None, help="default: configs/zones/<video name>.json")
    p.add_argument("--frame", type=int, default=0, help="which frame to draw on")
    p.add_argument("--width", type=int, default=1280, help="display width")
    return p.parse_args()


def grab_frame(source: str, index: int) -> np.ndarray:
    cap, _ = open_video(source)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} from {source}")
    return frame


def render(base: np.ndarray, zones: list[Zone], points: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    h, w = out.shape[:2]
    for z in zones:
        pts = np.array([[int(x * w), int(y * h)] for x, y in z.polygon], dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 200, 255), 2)
        cv2.putText(out, z.name, tuple(pts[0] + [4, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for i, pt in enumerate(points):
        cv2.circle(out, pt, 4, (0, 0, 255), -1)
        if i:
            cv2.line(out, points[i - 1], pt, (0, 0, 255), 2)
    cv2.putText(out, "click=point  u=undo  Enter=finish zone  s=save  q=quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def ask_zone_details() -> tuple[str, float, str]:
    name = input("Zone name (e.g. red_zone_vdoor): ").strip() or "zone"
    dwell_raw = input("Alert after how many seconds inside? [2.0]: ").strip()
    severity = input(f"Severity {SEVERITIES[1:]} [high]: ").strip() or "high"
    try:
        dwell = float(dwell_raw) if dwell_raw else 2.0
    except ValueError:
        print("  not a number, using 2.0")
        dwell = 2.0
    if severity not in SEVERITIES:
        print("  unknown severity, using high")
        severity = "high"
    return name, dwell, severity


def main() -> int:
    args = parse_args()
    try:
        base = resize_to_width(grab_frame(args.source, args.frame), args.width)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1
    h, w = base.shape[:2]
    output = Path(args.output or f"configs/zones/{Path(args.source).stem}.json")

    zones: list[Zone] = []
    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW, render(base, zones, points))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("u") and points:
                points.pop()
            elif key in KEY_ENTER:
                if len(points) < 3:
                    print("need at least 3 points for a zone")
                    continue
                name, dwell, severity = ask_zone_details()
                if name in {z.name for z in zones}:
                    print(f"  '{name}' already used, pick another name")
                    continue
                polygon = tuple((x / w, y / h) for x, y in points)
                zones.append(Zone(name, polygon, dwell, severity))
                points.clear()
                print(f"  added zone '{name}' ({len(polygon)} points)")
            elif key == ord("s"):
                if not zones:
                    print("no zones to save yet (press Enter to finish one)")
                    continue
                save_zones(output, zones)
                print(f"[saved] {len(zones)} zone(s) -> {output}")
                return 0
            elif key in (ord("q"), KEY_ESC):
                print("quit without saving")
                return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
