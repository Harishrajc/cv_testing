# RigSafe — CCTV HSE Hazard Monitor

Learning project modelled on industrial AI-safety platforms (e.g. Veunex): detect people/PPE on CCTV,
watch danger zones, raise alerts, show them on a dashboard. Built level by level.

## Levels

| Level | Topic | Difficulty | Runs on |
|---|---|---|---|
| L0 | Video I/O, HUD, FPS | Beginner | Local |
| L1 | Pretrained YOLO person detection | Beginner | Local (CPU) |
| L2 | Train custom PPE model | Beginner→Intermediate | **Colab GPU** (train) + local (infer) |
| L3 | Tracking + red-zone / dwell-time alerts | Intermediate | Local |
| L4 | Event engine + FastAPI + DB | Intermediate | Local |
| L5 | Dashboard + simulated RTSP cameras | Intermediate→Advanced | Local |
| L6 | Line-of-fire, working-at-height (pose) | Advanced | Local infer, Colab if training |
| L7 | ONNX/OpenVINO/TensorRT, Docker, benchmarks | Advanced | Local (OpenVINO) + **Colab** (TensorRT) |

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
```

## Level 0

```bash
.venv/bin/python scripts/l0_video_overlay.py --source data/videos/sample.mp4
.venv/bin/python scripts/l0_video_overlay.py --source data/videos/your_clip.mp4 --show
```

## Level 1

```bash
./scripts/l1_detect_people.py --source data/videos/site.mp4 --resize 960 --show
./scripts/l1_detect_people.py --source data/videos/site.mp4 --model yolo11s.pt --conf 0.25
```

Weights auto-download into `models/` (git-ignored).

## Level 2 — train PPE model on Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → *Upload* → `notebooks/l2_train_ppe.ipynb`
2. `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`
3. Download the `*_best.pt` it gives you, save as `models/ppe_best.pt`
4. Run locally:

```bash
./scripts/l2_detect_ppe.py --source data/videos/site.mp4 --model models/ppe_best.pt --show
```

Dataset: Ultralytics Construction-PPE (1,416 images, 11 classes, AGPL-3.0).

## Level 3 — tracking, PPE compliance, danger zones

```bash
# 1. draw zones on the first frame (click points, Enter = finish zone, s = save)
./scripts/l3_draw_zones.py --source data/videos/site2.mp4
# 2. monitor: tracks workers, checks helmets, raises zone alerts
./scripts/l3_monitor.py --source data/videos/site2.mp4 --zones configs/zones/site2.json --show
```

Example zones: `configs/zones/site2_example.json`. Events go to `outputs/l3_events.jsonl`.

How decisions are made:
- **Tracking**: ByteTrack gives each worker a stable id.
- **Helmet**: violation = no helmet box in the person's head band for ≥60% of the last 15 frames
  (the model's `no_helmet` class is too weak to trust: AP50 0.17). Head cut off by the frame = no evidence, not a violation.
- **Zones**: polygons in 0–1 coordinates; the worker's *foot point* must stay inside ≥ `min_dwell_s`.
  1 s grace before an exit counts. Feet out of frame = not checked.
- **Severity**: in zone = zone severity; in zone without helmet = critical. Same PPE alert max once per 10 s.

Put real test clips (industrial / construction / rig footage) in `data/videos/` — they are git-ignored.
