#!/usr/bin/env python3
"""Level 0: read a video, draw frame counter + FPS, write the result.

Usage:
    python scripts/l0_video_overlay.py --source data/videos/sample.mp4
    python scripts/l0_video_overlay.py --source 0 --show      # webcam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigsafe.video_io import FpsMeter, draw_hud, iter_frames, make_writer, open_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path, RTSP URL, or webcam index")
    p.add_argument("--output", default="outputs/l0_overlay.mp4", help="annotated output video")
    p.add_argument("--show", action="store_true", help="open a preview window (press q to quit)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    try:
        cap, info = open_video(source)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1

    print(f"[info] {info.width}x{info.height} @ {info.fps:.1f} fps, {info.frame_count} frames")
    writer = make_writer(Path(args.output), info)
    meter = FpsMeter()
    processed = 0

    try:
        for index, frame in iter_frames(cap):
            fps = meter.tick()
            annotated = draw_hud(frame, [
                f"Frame: {index}",
                f"Proc FPS: {fps:.1f}",
                f"Time: {index / info.fps:.2f}s",
            ])
            writer.write(annotated)
            processed += 1

            if args.show:
                cv2.imshow("RigSafe L0", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[done] {processed} frames -> {args.output} (avg proc FPS ~{meter.fps:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
