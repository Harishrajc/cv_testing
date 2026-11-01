import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigsafe.video_io import FpsMeter, draw_hud, open_video, resize_to_width, scaled_size  # noqa: E402


def test_scaled_size_keeps_aspect_ratio():
    assert scaled_size(1920, 1080, 960) == (960, 540)


def test_scaled_size_never_upscales_or_changes_when_none():
    assert scaled_size(640, 360, 1280) == (640, 360)
    assert scaled_size(640, 360, None) == (640, 360)


def test_scaled_size_rejects_non_positive():
    with pytest.raises(ValueError):
        scaled_size(1920, 1080, 0)


def test_resize_to_width_shrinks_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert resize_to_width(frame, 640).shape == (360, 640, 3)
    assert resize_to_width(frame, None) is frame


def fake_clock(times):
    it = iter(times)
    return lambda: next(it)


def test_fps_meter_reports_zero_before_second_tick():
    meter = FpsMeter(clock=fake_clock([0.0]))
    assert meter.tick() == 0.0


def test_fps_meter_first_measurement_is_instant_fps():
    meter = FpsMeter(clock=fake_clock([0.0, 0.1]))
    meter.tick()
    assert meter.tick() == pytest.approx(10.0)


def test_fps_meter_smooths_toward_new_rate():
    meter = FpsMeter(smoothing=0.5, clock=fake_clock([0.0, 0.1, 0.15]))
    meter.tick()
    meter.tick()          # 10 fps
    fps = meter.tick()    # instant 20 fps -> 0.5*10 + 0.5*20
    assert fps == pytest.approx(15.0)


def test_fps_meter_rejects_bad_smoothing():
    with pytest.raises(ValueError):
        FpsMeter(smoothing=1.0)


def test_draw_hud_does_not_mutate_input():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    out = draw_hud(frame, ["hello"])
    assert frame.sum() == 0
    assert out.sum() > 0
    assert out.shape == frame.shape


def test_open_video_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        open_video("does/not/exist.mp4")
