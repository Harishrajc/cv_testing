"""Person detection with a pretrained YOLO model (Level 1)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cv2
import numpy as np

COCO_PERSON_CLASS = 0
BOX_COLOR = (0, 200, 0)  # BGR green
LABEL_TEXT_COLOR = (0, 0, 0)


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-centre of the box: where the person stands. Used for zone checks in L3."""
        return (self.x1 + self.x2) // 2, self.y2


class YoloModel(Protocol):
    names: dict[int, str]

    def predict(self, source: Any, **kwargs: Any) -> list[Any]: ...


class Detector:
    """Thin wrapper over an Ultralytics model that returns plain Detection objects.

    classes: class ids to keep, or None for every class the model knows
    (None is what a custom PPE model wants; (0,) keeps only COCO "person").
    """

    def __init__(self, model: YoloModel, conf: float = 0.35, imgsz: int = 640,
                 classes: tuple[int, ...] | None = (COCO_PERSON_CLASS,)):
        if not 0.0 < conf < 1.0:
            raise ValueError("conf must be between 0 and 1")
        self._model = model
        self._conf = conf
        self._imgsz = imgsz
        self._classes = list(classes) if classes is not None else None
        self.last_inference_ms = 0.0

    @property
    def names(self) -> dict[int, str]:
        return self._model.names

    @classmethod
    def from_weights(cls, weights: str, **kwargs: Any) -> "Detector":
        from ultralytics import YOLO  # imported lazily so tests don't need torch

        return cls(YOLO(weights), **kwargs)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        start = time.perf_counter()
        results = self._model.predict(
            frame, conf=self._conf, imgsz=self._imgsz, classes=self._classes, verbose=False
        )
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        return parse_results(results[0], self._model.names)


def parse_results(result: Any, names: dict[int, str]) -> list[Detection]:
    """Convert one Ultralytics Results object into Detection objects."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = np.asarray(boxes.xyxy).astype(int)
    confs = np.asarray(boxes.conf, dtype=float)
    cls_ids = np.asarray(boxes.cls).astype(int)
    return [
        Detection(int(x1), int(y1), int(x2), int(y2), float(c), int(k), names.get(int(k), str(k)))
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls_ids)
    ]


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    color_fn: Callable[[Detection], tuple[int, int, int]] = lambda _: BOX_COLOR,
    draw_feet: bool = True,
) -> np.ndarray:
    """Return a copy of the frame with boxes, labels and (optionally) foot points drawn."""
    out = frame.copy()
    for d in detections:
        color = color_fn(d)
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (d.x1, d.y1 - th - 6), (d.x1 + tw + 4, d.y1), color, -1)
        cv2.putText(out, text, (d.x1 + 2, d.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    LABEL_TEXT_COLOR, 1, cv2.LINE_AA)
        if draw_feet:
            cv2.circle(out, d.foot_point, 4, (0, 0, 255), -1)
    return out
