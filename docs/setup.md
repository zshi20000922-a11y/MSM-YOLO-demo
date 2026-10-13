# Environment and run guide / 环境与运行

## Requirements / 依赖

- Python 3.10, Linux desktop with a Qt display (or `QT_QPA_PLATFORM=offscreen` for a smoke check).
- `python -m pip install -r requirements.txt` installs the desktop workbench dependencies. The tested Ultralytics package is pinned to 8.4.117; NumPy, OpenCV, PySide6, LAP, and PyYAML are constrained in the file.
- PyTorch is installed as an Ultralytics dependency. For GPU inference/training, install a PyTorch build appropriate for your CUDA driver and GPU before installing the rest. CPU is supported for basic use.
- `requirements-deploy.txt` adds ONNX tooling for PT export. RKNN-Toolkit2 is **not** installed or redistributed by this repository; create a separate environment with a version supported by your RK3588 deployment stack.

Create a virtual environment, install requirements, then run `python msm_tuner_gui.py`. The GUI starts without local assets; select a video and weights in its input panel. On a headless system, only smoke/import checks are expected; the interactive workbench needs a display.

## Paths / 路径

| Variable | Purpose / 用途 | Default |
|---|---|---|
| `MSM_VIDEO` | Initial video selection | `phantom03.mp4` beside the script, only if it exists |
| `MSM_WEIGHTS` | Initial detection checkpoint selection | `best.pt` beside the script, only if it exists |
| `MSM_YOLOV5_ROOT` | Original YOLOv5 repository checkout for original YOLOv5 checkpoints | `external/yolov5` if present; legacy adjacent checkout fallback |
| `MSM_ARD100_YAML` | Optional training dataset YAML suggestion | `data/ARD100.yaml` if present |
| `MSM_RKNN_PYTHON` | Optional separate RKNN-Toolkit2 environment Python | blank |

The Qt training page defaults to the Python interpreter running the GUI and lets you choose a different interpreter. The deployment page lets you choose both the export and RKNN interpreters. Runtime configuration, metrics, model paths/hashes, training output, and deployment output are stored under `runs/`, which is ignored by Git.

## Model backends / 模型后端

The **original YOLOv5** adapter imports `models.common` and other inference utilities from a separate YOLOv5 checkout. Clone the [original YOLOv5 repository](https://github.com/ultralytics/yolov5) outside this project, or into ignored `external/yolov5`, then set `MSM_YOLOV5_ROOT` if needed. The exact revision should match the checkpoint's architecture, particularly for custom modules. An original-repository checkpoint is not interchangeable with a newer Ultralytics checkpoint loader.

The **Ultralytics** adapter accepts compatible detection `.pt` models supported by the installed Ultralytics version. Model-family names in the workbench are educational references, not a guarantee that every third-party YOLOv6/7 checkpoint can load. Select a backend manually if auto-detection cannot determine the format.

## CLI examples / 命令行示例

```bash
python msm_yolo_demo.py --source /path/to/video.mp4 --weights /path/to/original-yolov5.pt --device cpu --no-display
python evaluate_ard100_motion.py --dataset /path/to/ARD100 --weights /path/to/detector.pt --backend ultralytics --device cpu --max-videos 1
python -m unittest discover -s tests -v
```

The first CLI currently uses the native original YOLOv5 adapter. The ARD100 command requires the dataset and checkpoint to be supplied locally; its results go under ignored `runs/`. Training requires a YOLO-format dataset YAML pointing to local train/val data. For INT8 RKNN PTQ conversion, choose at least eight representative local calibration images. Board-side runtime verification is a separate task.

## Troubleshooting / 故障排查

- **Missing YOLOv5 code:** set `MSM_YOLOV5_ROOT` to a compatible checkout. Do not install that source into this repository's Git history.
- **Qt reports an `xcb` plugin error:** launch through the same Python environment in which PySide6 is installed. The GUI clears OpenCV's conflicting Qt plugin path before Qt loads; a desktop display and system `xcb` libraries are still required.
- **CUDA unavailable:** verify `python -c "import torch; print(torch.cuda.is_available())"` in the selected interpreter, or choose CPU.
- **RKNN converter unavailable:** configure a separate RKNN Python interpreter in the deployment page; desktop conversion also requires toolkit installation and its own license acceptance.
