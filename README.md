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

Put real test clips (industrial / construction / rig footage) in `data/videos/` — they are git-ignored.
