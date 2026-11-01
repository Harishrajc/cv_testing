#!/usr/bin/env python3
"""Level 2: run your Colab-trained PPE model on a local video.

Copy best.pt from Colab/Drive into models/ first, then:
    ./scripts/l2_detect_ppe.py --source data/videos/site.mp4 --model models/ppe_best.pt --show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from rigsafe.detector import Detector, draw_detections
from rigsafe.ppe import ppe_color, summarize
from rigsafe.video_io import (
    FpsMeter, VideoInfo, draw_hud, iter_frames, make_writer, open_video, resize_to_width, scaled_size,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path, RTSP URL, or webcam index")
    p.add_argument("--model", default="models/ppe_best.pt", help="trained PPE weights (.pt)")
    p.add_argument("--output", default="outputs/l2_ppe.mp4")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=640, help="use the size you trained with")
    p.add_argument("--resize", type=int, default=960)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--show", action="store_true", help="preview window (q to quit)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.model).is_file():
        print(f"[error] model not found: {args.model} (download best.pt from Colab first)", file=sys.stderr)
        return 1

    source = int(args.source) if args.source.isdigit() else args.source
    try:
        cap, info = open_video(source)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1

    detector = Detector.from_weights(args.model, conf=args.conf, imgsz=args.imgsz, classes=None)
    print(f"[info] model classes: {detector.names}")

    out_w, out_h = scaled_size(info.width, info.height, args.resize)
    writer = make_writer(Path(args.output), VideoInfo(out_w, out_h, info.fps, info.frame_count))
    meter = FpsMeter()
    processed, frames_with_violation = 0, 0

    try:
        for index, frame in iter_frames(cap):
            if args.max_frames is not None and index >= args.max_frames:
                break
            small = resize_to_width(frame, args.resize)
            detections = detector.detect(small)
            summary = summarize(detections)
            fps = meter.tick()

            annotated = draw_hud(
                draw_detections(small, detections, color_fn=ppe_color, draw_feet=False),
                summary.hud_lines() + [f"Infer: {detector.last_inference_ms:.0f} ms  FPS: {fps:.1f}"],
            )
            writer.write(annotated)
            processed += 1
            frames_with_violation += summary.violations > 0

            if args.show:
                cv2.imshow("RigSafe L2", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if processed:
        print(f"[done] {processed} frames -> {args.output}")
        print(f"[stats] frames with a violation: {frames_with_violation}/{processed}, proc FPS ~{meter.fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
