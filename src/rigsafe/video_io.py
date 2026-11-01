"""Video read/write and HUD helpers shared by every level."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

DEFAULT_FPS = 25.0
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.7
HUD_LINE_HEIGHT = 28
HUD_COLOR = (0, 255, 255)  # BGR yellow
HUD_BG = (0, 0, 0)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def open_video(source: str | int) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open a file path, RTSP URL or webcam index. Raises if it can't be read."""
    if isinstance(source, str) and not source.startswith(("rtsp://", "http")):
        if not Path(source).is_file():
            raise FileNotFoundError(f"Video not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps if fps and fps > 0 else DEFAULT_FPS,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, info


def iter_frames(cap: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) until the stream ends; always releases the capture."""
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def make_writer(path: Path, info: VideoInfo) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {path}")
    return writer


def scaled_size(width: int, height: int, target_width: int | None) -> tuple[int, int]:
    """Size after shrinking to target_width, keeping aspect ratio. Never upscales."""
    if target_width is None or target_width >= width:
        return width, height
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    scale = target_width / width
    return target_width, max(1, round(height * scale))


def resize_to_width(frame: np.ndarray, target_width: int | None) -> np.ndarray:
    """Return a frame shrunk to target_width (aspect kept). Returns input as-is if no shrink needed."""
    h, w = frame.shape[:2]
    new_w, new_h = scaled_size(w, h, target_width)
    if (new_w, new_h) == (w, h):
        return frame
    # INTER_AREA is the best interpolation for shrinking (less aliasing than LINEAR)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FpsMeter:
    """Smoothed processing-FPS counter (exponential moving average)."""

    def __init__(self, smoothing: float = 0.9, clock: Callable[[], float] = time.perf_counter):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self._smoothing = smoothing
        self._clock = clock
        self._last: float | None = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def tick(self) -> float:
        now = self._clock()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                self._fps = instant if self._fps == 0 else (
                    self._smoothing * self._fps + (1 - self._smoothing) * instant
                )
        self._last = now
        return self._fps


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Return a copy of the frame with text lines drawn top-left on a dark band."""
    out = frame.copy()
    for i, text in enumerate(lines):
        y = HUD_LINE_HEIGHT * (i + 1)
        (w, h), _ = cv2.getTextSize(text, HUD_FONT, HUD_SCALE, 2)
        cv2.rectangle(out, (5, y - h - 6), (15 + w, y + 6), HUD_BG, -1)
        cv2.putText(out, text, (10, y), HUD_FONT, HUD_SCALE, HUD_COLOR, 2, cv2.LINE_AA)
    return out
