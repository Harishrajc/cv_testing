#!/usr/bin/env python3
"""Level 1: detect people in a video with a pretrained YOLO model.

Usage:
    ./scripts/l1_detect_people.py --source data/videos/site.mp4 --resize 960 --show
    ./scripts/l1_detect_people.py --source data/videos/site.mp4 --model yolo11s.pt --conf 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigsafe.detector import Detector, draw_detections  # noqa: E402
from rigsafe.video_io import (  # noqa: E402
    FpsMeter, VideoInfo, draw_hud, iter_frames, make_writer, open_video, resize_to_width, scaled_size,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path, RTSP URL, or webcam index")
    p.add_argument("--output", default="outputs/l1_people.mp4")
    p.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m.pt (auto-downloaded to models/)")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold (0-1)")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size (multiple of 32)")
    p.add_argument("--resize", type=int, default=None, help="shrink frames to this width first")
    p.add_argument("--max-frames", type=int, default=None, help="stop early (quick tests)")
    p.add_argument("--show", action="store_true", help="preview window (q to quit)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    try:
        cap, info = open_video(source)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1

    out_w, out_h = scaled_size(info.width, info.height, args.resize)
    print(f"[info] input {info.width}x{info.height} -> processing {out_w}x{out_h} @ {info.fps:.1f} fps")

    MODELS_DIR.mkdir(exist_ok=True)
    weights = args.model if Path(args.model).exists() else str(MODELS_DIR / args.model)
    detector = Detector.from_weights(weights, conf=args.conf, imgsz=args.imgsz)

    writer = make_writer(Path(args.output), VideoInfo(out_w, out_h, info.fps, info.frame_count))
    meter = FpsMeter()
    processed, total_people, total_ms = 0, 0, 0.0

    try:
        for index, frame in iter_frames(cap):
            if args.max_frames is not None and index >= args.max_frames:
                break
            small = resize_to_width(frame, args.resize)
            detections = detector.detect(small)
            fps = meter.tick()

            annotated = draw_hud(draw_detections(small, detections), [
                f"People: {len(detections)}",
                f"Infer: {detector.last_inference_ms:.0f} ms",
                f"Proc FPS: {fps:.1f}",
            ])
            writer.write(annotated)
            processed += 1
            total_people += len(detections)
            total_ms += detector.last_inference_ms

            if args.show:
                cv2.imshow("RigSafe L1", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if processed:
        print(f"[done] {processed} frames -> {args.output}")
        print(f"[stats] avg people/frame {total_people / processed:.1f}, "
              f"avg inference {total_ms / processed:.0f} ms, proc FPS ~{meter.fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
