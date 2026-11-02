#!/usr/bin/env python3
"""Level 3: track workers, check helmets, watch danger zones, log safety events.

Usage:
    ./scripts/l3_monitor.py --source data/videos/site.mp4 --zones configs/zones/site.json --show
Events are written to outputs/l3_events.jsonl (one JSON object per line).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from rigsafe.compliance import MISSING, ComplianceTracker, dedupe_persons, observe
from rigsafe.detector import Detection, Detector
from rigsafe.events import PPE_MISSING, AlertCooldown, Event, EventLog
from rigsafe.geometry import normalize
from rigsafe.overlay import draw_alert_banner, draw_worker, draw_zones
from rigsafe.ppe import PERSON_LABELS
from rigsafe.video_io import (
    FpsMeter, VideoInfo, draw_hud, iter_frames, make_writer, open_video, resize_to_width, scaled_size,
)
from rigsafe.zones import TrackedPerson, ZoneMonitor, feet_visible, load_zones

BANNER_SECONDS = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--zones", required=True, help="zones JSON from l3_draw_zones.py")
    p.add_argument("--model", default="models/ppe_best.pt")
    p.add_argument("--output", default="outputs/l3_monitor.mp4")
    p.add_argument("--events", default="outputs/l3_events.jsonl")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--resize", type=int, default=960)
    p.add_argument("--exit-grace", type=float, default=1.0, help="seconds outside before an exit counts")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def split(detections: list[Detection]) -> tuple[list[Detection], list[Detection]]:
    persons = [d for d in detections if d.label in PERSON_LABELS and d.track_id is not None]
    items = [d for d in detections if d.label not in PERSON_LABELS]
    return dedupe_persons(persons), items


def ppe_events(t: float, changes: list[tuple[int, str, str]], cooldown: AlertCooldown) -> list[Event]:
    return [Event(PPE_MISSING, t, tid, "medium", details={"item": item})
            for tid, item, status in changes if status == MISSING and cooldown.allow((tid, item), t)]


def main() -> int:
    args = parse_args()
    for path, what in ((args.model, "model"), (args.zones, "zones file")):
        if not Path(path).is_file():
            print(f"[error] {what} not found: {path}", file=sys.stderr)
            return 1
    try:
        cap, info = open_video(int(args.source) if args.source.isdigit() else args.source)
        zones = load_zones(Path(args.zones))
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1

    detector = Detector.from_weights(args.model, conf=args.conf, imgsz=args.imgsz, classes=None)
    compliance = ComplianceTracker()
    cooldown = AlertCooldown(seconds=10.0)
    monitor = ZoneMonitor(zones, exit_grace_s=args.exit_grace)
    out_w, out_h = scaled_size(info.width, info.height, args.resize)
    writer = make_writer(Path(args.output), VideoInfo(out_w, out_h, info.fps, info.frame_count))
    meter = FpsMeter()
    banner, banner_until = "", -1.0
    print(f"[info] {len(zones)} zone(s): {[z.name for z in zones]}")

    with EventLog(Path(args.events)) as log:
        try:
            for index, frame in iter_frames(cap):
                if args.max_frames is not None and index >= args.max_frames:
                    break
                t = index / info.fps  # video time; deterministic, unlike wall clock
                small = resize_to_width(frame, args.resize)
                persons, items = split(detector.track(small))

                events = ppe_events(t, compliance.update(observe(persons, items)), cooldown)
                people = [TrackedPerson(d.track_id, normalize(d.foot_point, out_w, out_h),
                                        compliance.status(d.track_id))
                          for d in persons if feet_visible(d, out_h)]
                events += monitor.update(t, people)

                for e in events:
                    log.write(e)
                    if e.is_alert:
                        print(e.describe())
                        banner, banner_until = e.describe().split("] ", 1)[1], t + BANNER_SECONDS

                annotated = draw_zones(small, monitor.zones, monitor.occupied())
                for d in persons:
                    dwell = max((monitor.dwell(d.track_id, z.name, t) or 0.0 for z in zones), default=0.0)
                    draw_worker(annotated, d, compliance.status(d.track_id), dwell or None)
                annotated = draw_hud(annotated, [
                    f"t={t:.1f}s  workers={len(persons)}",
                    f"alerts={log.counts['zone_intrusion'] + log.counts['ppe_missing']}  FPS {meter.tick():.1f}",
                ])
                if t <= banner_until:
                    annotated = draw_alert_banner(annotated, banner)
                writer.write(annotated)

                if args.show:
                    cv2.imshow("RigSafe L3", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if args.show:
                cv2.destroyAllWindows()

    print(f"[done] video -> {args.output}")
    print(f"[events] {dict(log.counts)} -> {args.events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
