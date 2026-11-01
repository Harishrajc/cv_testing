import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigsafe.detector import Detection, Detector, draw_detections, parse_results  # noqa: E402

NAMES = {0: "person"}


class FakeBoxes:
    """Mimics Ultralytics Boxes: rows of (x1, y1, x2, y2, conf, cls)."""

    def __init__(self, rows):
        arr = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = arr[:, :4], arr[:, 4], arr[:, 5]

    def __len__(self):
        return len(self.xyxy)


class FakeModel:
    names = NAMES

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes(self.rows))]


def test_parse_results_converts_boxes():
    result = SimpleNamespace(boxes=FakeBoxes([[10.7, 20.2, 50.9, 120.0, 0.88, 0]]))
    [d] = parse_results(result, NAMES)
    assert (d.x1, d.y1, d.x2, d.y2) == (10, 20, 50, 120)
    assert d.confidence == pytest.approx(0.88)
    assert d.label == "person"


def test_parse_results_empty():
    assert parse_results(SimpleNamespace(boxes=FakeBoxes([])), NAMES) == []
    assert parse_results(SimpleNamespace(boxes=None), NAMES) == []


def test_foot_point_is_bottom_centre():
    d = Detection(10, 20, 50, 120, 0.9, 0, "person")
    assert d.foot_point == (30, 120)


def test_detector_passes_settings_and_times_inference():
    model = FakeModel([[0, 0, 10, 10, 0.9, 0], [5, 5, 20, 30, 0.6, 0]])
    detector = Detector(model, conf=0.5, imgsz=480)
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(detections) == 2
    assert model.calls[0]["conf"] == 0.5
    assert model.calls[0]["imgsz"] == 480
    assert model.calls[0]["classes"] == [0]
    assert detector.last_inference_ms >= 0


def test_detector_rejects_bad_conf():
    with pytest.raises(ValueError):
        Detector(FakeModel([]), conf=1.5)


def test_draw_detections_does_not_mutate_input():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    out = draw_detections(frame, [Detection(20, 40, 100, 180, 0.9, 0, "person")])
    assert frame.sum() == 0
    assert out.sum() > 0
