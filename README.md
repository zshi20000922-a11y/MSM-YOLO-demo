# MSM-YOLO-demo

[中文说明](README_zh.md) · [Screenshots](docs/screenshots/README.md) · [Setup](docs/setup.md) · [Architecture](docs/architecture.md) · [ARD100 results](docs/benchmark.md) · [Contributing](CONTRIBUTING.md)

**A local research workbench for motion-guided small-object detection in UAV video.** It connects motion proposals, full-frame/ROI YOLO detection, ByteTrack, visual diagnostics, and a SQLite experiment registry. The project explores control flow and reproducible evaluation around existing detectors; it does **not** modify the YOLO network or claim a new trained architecture.

![MSM-YOLO-demo architecture](diagrams/msm_workbench_architecture.png)

## Workbench preview

The images below are captures of the running Qt app. Click to inspect the full resolution and see the [annotated feature gallery](docs/screenshots/README.md).

| Motion-guided analysis | YOLO threshold lab |
|---|---|
| [![Motion workbench](docs/screenshots/01-motion-workbench.png)](docs/screenshots/01-motion-workbench.png) | [![YOLO confidence and IoU controls](docs/screenshots/03-yolo-thresholds.png)](docs/screenshots/03-yolo-thresholds.png) |
| Our motion pipeline exposes the compensated frame, difference response, mask, and candidate regions together. | One forward pass is cached; confidence and NMS IoU controls update the stage counts and visual results. |

The motion capture uses a synthetic scene. The YOLO and tracking captures use Ultralytics' sample `bus.jpg`; they illustrate the interface and workflow, not UAV detection accuracy or RK3588 performance.

## What this project adds

| Area | Implemented contribution |
|---|---|
| Motion | Interchangeable optical-flow/camera-compensated frame difference, plain frame difference, and MOG2; aligned frames, difference maps, masks, and proposals are exposed for inspection. |
| Decision flow | Explicit global search, local detection, motion extraction, and motion verification states. YOLO confirmation gates ROI locking; the state machine handles misses and recovery. |
| Detection & tracking | Common `Detection`/`Track` contracts, separate original YOLOv5 and Ultralytics `.pt` adapters, and a ByteTrack adapter. Full-image and 640×640 ROI inference map boxes back to source coordinates. |
| Lab workbench | Qt tuning interface with synchronized stage previews, threshold replay on one image, algorithm comparison, network/metric explanations, local training controls, and desktop PT → ONNX → RKNN conversion controls. |
| Reproducibility | SQLite records configuration, input/model paths and hashes, backend/device metadata, metrics, and result locations. ARD100 video-level evaluation script and a [sanitized aggregate report](docs/benchmark.md). |

## Quick start

Python 3.10 is the tested development target. CPU inference works for the workbench; CUDA requires a PyTorch build compatible with your GPU.

```bash
git clone https://github.com/zshi20000922-a11y/MSM-YOLO-demo.git
cd MSM-YOLO-demo
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python msm_tuner_gui.py
```

The workbench opens without bundled models or videos. Choose a local video and compatible detection `.pt` file on the Video page. For original YOLOv5 repository checkpoints, set `MSM_YOLOV5_ROOT` to an external YOLOv5 checkout. For other compatible Ultralytics mainline detection checkpoints, choose the Ultralytics backend. See [environment and model setup](docs/setup.md).

Optional run-time paths:

```bash
export MSM_VIDEO=/path/to/input.mp4
export MSM_WEIGHTS=/path/to/model.pt
export MSM_YOLOV5_ROOT=/path/to/yolov5
export MSM_ARD100_YAML=/path/to/ARD100.yaml
python msm_tuner_gui.py
```

Run the included checks with `python -m unittest discover -s tests -v`. The [ARD100 evaluation entry point](evaluate_ard100_motion.py) accepts dataset and weight paths explicitly; the repository does not contain either asset.

## Code map

- [`msm_yolo_demo.py`](msm_yolo_demo.py): motion proposal and video state machine; native YOLOv5 adapter and ByteTrack bridge.
- [`msm_lab/`](msm_lab): module interfaces and contracts, motion/detection/tracking backends, evaluation, training, deployment, experiment registry.
- [`msm_tuner_gui.py`](msm_tuner_gui.py): Qt workbench.
- [`evaluate_ard100_motion.py`](evaluate_ard100_motion.py): video-level benchmark CLI.
- [`tests/`](tests): contract, algorithm, experiment, and workflow tests.

## Scope and evidence

The published comparison is a **combined ROI pipeline** comparison: motion scheduling and detection scale change together. The desktop RKNN path has been exercised for conversion; RK3588 on-device accuracy, latency, and throughput have **not** been verified here. No model weights, videos, ARD100 files, per-frame predictions, calibration images, or runtime databases are published.

## License and third-party software

Project code and documentation are released under [AGPL-3.0](LICENSE). Ultralytics, original YOLOv5, RKNN-Toolkit2, datasets, and model weights retain their own licenses and must be obtained separately. See [third-party and licensing notes](docs/third_party.md). Contributions are welcome through issues and pull requests.
