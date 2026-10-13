#!/usr/bin/env python3
"""Qt tuning workbench for the MSM-YOLO motion and tracking demo."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# cv2 sets this to its bundled Qt plugins during import; restore the plugin
# directory matching PySide6 before QtCore loads its platform integration.
_py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
_PYSIDE_PLATFORMS = Path(sys.prefix) / "lib" / _py_version / "site-packages" / "PySide6" / "Qt" / "plugins" / "platforms"
if _PYSIDE_PLATFORMS.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_PYSIDE_PLATFORMS)

from PySide6.QtCore import QByteArray, QObject, Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QRadioButton, QSlider,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QScrollArea, QGridLayout,
    QGroupBox, QPlainTextEdit,
)
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PySide6.QtSvgWidgets import QSvgWidget

from msm_lab.detectors import create_detector
from msm_lab.deployment import (DeploymentRequest, calibration_images,
                                convert_command, export_command, validate_request)
from msm_lab.experiment_store import ExperimentStore, runtime_versions, sha256_file
from msm_lab.motion_algorithms import create_motion_algorithm, motion_algorithm_choices
from msm_lab.model_catalog import MODEL_PROFILES, architecture_svg
from msm_lab.trackers import create_tracker, tracker_backend_choices
from msm_lab.training import (UltralyticsTrainingBackend, TrainingRequest,
                              TrainingState)
from msm_yolo_demo import MSMVideoPipeline


HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = Path(os.environ.get("MSM_VIDEO", str(HERE / "phantom03.mp4"))).expanduser()
DEFAULT_WEIGHTS = Path(os.environ.get("MSM_WEIGHTS", str(HERE / "best.pt"))).expanduser()
RUNS = HERE / "runs" / "tuner"
DATABASE = HERE / "runs" / "msm_lab.sqlite3"
ARD100_DATA_YAML = Path(
    os.environ.get("MSM_ARD100_YAML", str(HERE / "data" / "ARD100.yaml"))
).expanduser()
TRAINING_SETTINGS = HERE / "runs" / "training_center.json"
DEPLOYMENT_SETTINGS = HERE / "runs" / "deployment_center.json"
DEPLOYMENT_WORKER = HERE / "msm_lab" / "deployment_worker.py"

TRAINING_PARAMETER_GROUPS = [
    ("训练策略", [
        ("time", "最长训练时间（小时）", "float", 0.0, 0.0, 100000.0),
        ("save", "保存检查点", "bool", True),
        ("save_period", "检查点保存间隔（epoch）", "int", -1, -1, 10000),
        ("exist_ok", "允许复用同名输出目录", "bool", False),
        ("cache", "图像缓存", "choice", False, [("关闭", False), ("内存", "ram"), ("磁盘", "disk")]),
        ("pretrained", "使用预训练权重", "bool", True),
        ("cls_remap", "按类别名称映射权重", "bool", True),
        ("verbose", "输出详细日志", "bool", True),
        ("seed", "随机种子", "int", 0, 0, 2147483647),
        ("deterministic", "确定性算法", "bool", True),
        ("single_cls", "单类别训练", "bool", False),
        ("rect", "矩形批处理", "bool", False),
        ("cos_lr", "余弦学习率调度", "bool", False),
        ("close_mosaic", "关闭 Mosaic 的末尾轮数", "int", 10, 0, 10000),
        ("amp", "自动混合精度", "bool", True),
        ("fraction", "训练集使用比例", "float", 1.0, 0.001, 1.0),
        ("profile", "性能分析", "bool", False),
        ("freeze", "冻结层（空/整数/逗号列表）", "text", ""),
        ("multi_scale", "多尺度训练比例", "float", 0.0, 0.0, 1.0),
        ("compile", "编译模型（PyTorch 2.x）", "bool", False),
        ("channels_last", "Channels Last 内存格式", "bool", False),
        ("end2end", "端到端检测头（模型支持时）", "choice", None, [("自动", None), ("启用", True), ("关闭", False)]),
        ("quantize", "量化训练模式（模型支持时）", "text", ""),
    ]),
    ("验证与输出", [
        ("val", "每轮执行验证", "bool", True),
        ("split", "验证数据集分区", "choice", "val", [("train", "train"), ("val", "val"), ("test", "test")]),
        ("save_json", "导出 COCO JSON 指标", "bool", False),
        ("conf", "验证置信度阈值（空=默认）", "optional_float", None, 0.0, 1.0),
        ("iou", "验证 NMS IoU", "float", 0.7, 0.0, 1.0),
        ("max_det", "每张图最大检测数", "int", 300, 1, 10000),
        ("plots", "保存训练曲线与诊断图", "bool", True),
    ]),
    ("优化器与学习率", [
        ("lr0", "初始学习率", "float", 0.01, 0.000001, 10.0),
        ("lrf", "最终学习率比例", "float", 0.01, 0.0, 1.0),
        ("momentum", "动量 / Adam beta1", "float", 0.937, 0.0, 1.0),
        ("weight_decay", "权重衰减", "float", 0.0005, 0.0, 1.0),
        ("warmup_epochs", "学习率预热轮数", "float", 3.0, 0.0, 100.0),
        ("warmup_momentum", "预热初始动量", "float", 0.8, 0.0, 1.0),
        ("warmup_bias_lr", "偏置预热学习率", "float", 0.1, 0.0, 10.0),
        ("nbs", "名义批次大小", "int", 64, 1, 65536),
    ]),
    ("损失函数", [
        ("box", "边框损失权重", "float", 7.5, 0.0, 100.0),
        ("cls", "分类损失权重", "float", 0.5, 0.0, 100.0),
        ("cls_pw", "分类正样本权重", "float", 0.0, 0.0, 100.0),
        ("dfl", "Distribution Focal Loss 权重", "float", 1.5, 0.0, 100.0),
        ("distill_model", "蒸馏教师模型（可选）", "text", ""),
        ("dis", "蒸馏损失权重", "float", 6.0, 0.0, 100.0),
    ]),
    ("数据增强", [
        ("hsv_h", "HSV 色调", "float", 0.015, 0.0, 1.0),
        ("hsv_s", "HSV 饱和度", "float", 0.7, 0.0, 1.0),
        ("hsv_v", "HSV 明度", "float", 0.4, 0.0, 1.0),
        ("degrees", "旋转角度（°）", "float", 0.0, 0.0, 180.0),
        ("translate", "平移比例", "float", 0.1, 0.0, 1.0),
        ("scale", "缩放强度", "float", 0.5, 0.0, 1.0),
        ("shear", "错切角度（°）", "float", 0.0, 0.0, 180.0),
        ("perspective", "透视变换强度", "float", 0.0, 0.0, 1.0),
        ("flipud", "上下翻转概率", "float", 0.0, 0.0, 1.0),
        ("fliplr", "左右翻转概率", "float", 0.5, 0.0, 1.0),
        ("bgr", "BGR 通道交换概率", "float", 0.0, 0.0, 1.0),
        ("mosaic", "Mosaic 概率", "float", 1.0, 0.0, 1.0),
        ("mixup", "MixUp 概率", "float", 0.0, 0.0, 1.0),
        ("cutmix", "CutMix 概率", "float", 0.0, 0.0, 1.0),
        ("copy_paste", "Copy-Paste 概率", "float", 0.0, 0.0, 1.0),
        ("copy_paste_mode", "Copy-Paste 模式", "choice", "flip", [("翻转", "flip"), ("混合", "mixup")]),
        ("auto_augment", "自动增强策略", "choice", "randaugment", [("RandAugment", "randaugment"), ("AutoAugment", "autoaugment"), ("AugMix", "augmix"), ("关闭", None)]),
        ("erasing", "随机擦除概率", "float", 0.4, 0.0, 1.0),
    ]),
]


def cv_image_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    return QImage(image.data, width, height, image.strides[0], QImage.Format.Format_RGB888).copy()


def draw_caption(image: np.ndarray, message: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (18, 34, 54), -1)
    cv2.putText(image, message, (13, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (255, 255, 255), 2, cv2.LINE_AA)


def make_diff_view(diff: np.ndarray) -> np.ndarray:
    """Stretch sparse frame-difference values into a visible diagnostic map."""
    positive = diff[diff > 0]
    if positive.size == 0:
        return np.zeros((*diff.shape, 3), np.uint8)
    upper = max(1.0, float(np.percentile(positive, 99.5)))
    stretched = np.clip(diff.astype(np.float32) * (255.0 / upper), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(stretched, cv2.COLORMAP_TURBO)


def make_mask_view(mask: np.ndarray, candidate_count: int = 0) -> np.ndarray:
    view = np.full((*mask.shape, 3), (24, 38, 55), np.uint8)
    view[mask > 0] = (210, 240, 240)
    label = (f"CANDIDATES  {candidate_count}" if candidate_count else
             "NO CANDIDATE REGION")
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (18, 30, 44), -1)
    cv2.putText(view, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (225, 238, 248), 2, cv2.LINE_AA)
    return view


def preview_image(image: np.ndarray | None, size=(760, 390), boxes=(), roi=None,
                  roi_status: str = "") -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), (20, 33, 49), np.uint8)
    if image is None:
        return canvas
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    out_w, out_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    ox, oy = (width - out_w) // 2, (height - out_h) // 2
    canvas[oy:oy + out_h, ox:ox + out_w] = resized
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        px1, py1 = round(ox + x1 * scale), round(oy + y1 * scale)
        px2, py2 = round(ox + x2 * scale), round(oy + y2 * scale)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (70, 245, 170), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"C{i}", (px1, max(18, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (70, 245, 170), 2, cv2.LINE_AA)
    if roi is not None:
        x1, y1, x2, y2 = roi
        px1, py1 = round(ox + x1 * scale), round(oy + y1 * scale)
        px2, py2 = round(ox + x2 * scale), round(oy + y2 * scale)
        roi_color = (230, 80, 255)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), roi_color, 4, cv2.LINE_AA)
        label = roi_status or "CANDIDATE ROI"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                 0.55, 2)
        label_y = max(label_h + 8, py1)
        cv2.rectangle(canvas, (px1, label_y - label_h - 8),
                      (min(width - 1, px1 + label_w + 14), label_y + 4),
                      (50, 20, 65), -1)
        cv2.putText(canvas, label, (px1 + 7, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 255), 2,
                    cv2.LINE_AA)
    return canvas


class ProcessingWorker(QObject):
    progress = Signal(int, int, str, int, int, float)
    snapshot = Signal(int, object, str)
    status = Signal(str)
    done = Signal(str, str, dict)
    failed = Signal(str)

    def __init__(self, video: Path, start_sec: float, end_sec: float,
                 settings: dict, mode: str, weights: str, device: str,
                 motion_algorithm: str = "flow_frame_difference",
                 detector_backend: str = "auto", project_name: str = "默认项目",
                 output_root: Path = RUNS,
                 tracker_backend: str = "bytetrack_ultralytics"):
        super().__init__()
        self.video = video
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.settings = settings
        self.mode = mode
        self.weights = weights
        self.device = device
        self.motion_algorithm = motion_algorithm
        self.detector_backend = detector_backend
        self.project_name = project_name or "默认项目"
        self.output_root = Path(output_root).expanduser()
        self.tracker_backend = tracker_backend
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    @staticmethod
    def _candidate_text(frame: np.ndarray, candidates: list[tuple[int, int, int, int]]) -> None:
        for index, (x1, y1, x2, y2) in enumerate(candidates, 1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 245, 170), 2, cv2.LINE_AA)
            cv2.putText(frame, f"C{index}", (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 245, 170), 2, cv2.LINE_AA)

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video))
        writer = None
        detector = None
        tracker = None
        experiment_store = None
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        run_dir = self.output_root / experiment_id
        actual_detector_backend = None
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{self.video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            first = max(0, int(self.start_sec * fps))
            last = min(int(self.end_sec * fps), total_frames)
            if last <= first:
                raise RuntimeError("所选时间范围超出视频长度。")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            run_dir.mkdir(parents=True, exist_ok=True)
            output_name = {"motion": "motion_preview.mp4", "detect": "detection_only.mp4",
                           "motion_yolo": "motion_yolo.mp4", "full": "full_system.mp4",
                           "compare": "motion_comparison.mp4"}[self.mode]
            output = run_dir / output_name
            motion_config = {key: self.settings[key] for key in (
                "threshold_sigma", "threshold_offset", "threshold_window", "morph_kernel",
                "open_iterations", "close_iterations", "min_area", "max_proposals",
                "history_window")}
            motion_config["edge_margin_ratio"] = (
                self.settings["edge_margin_percent"] / 100.0
            )
            motion_config.update({
                "mog2_history": self.settings["mog2_history"],
                "mog2_var_threshold": self.settings["mog2_var_threshold"],
                "mog2_detect_shadows": self.settings["mog2_detect_shadows"],
            })
            algorithm_configs = self.settings.get("motion_algorithm_configs", {})

            def config_for_algorithm(algorithm_id: str) -> dict:
                config = dict(motion_config)
                config.update(algorithm_configs.get(algorithm_id, {}))
                if "edge_margin_percent" in config:
                    config["edge_margin_ratio"] = float(config["edge_margin_percent"]) / 100.0
                return config

            writer_size = (width * 2, height * 2) if self.mode == "compare" else (width, height)
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, writer_size)
            if not writer.isOpened():
                raise RuntimeError(f"无法创建输出视频：{output}")

            experiment_store = ExperimentStore(DATABASE)
            project_id = experiment_store.ensure_project(self.project_name)
            video_hash = sha256_file(self.video)
            model_hash = sha256_file(self.weights) if self.mode in (
                "detect", "motion_yolo", "full") else None
            model_id = None
            experiment_store.start_experiment(
                experiment_id=experiment_id, project_id=project_id, mode=self.mode,
                video_path=self.video, video_sha256=video_hash, model_id=model_id,
                model_path=self.weights if model_hash else None, model_sha256=model_hash,
                detector_backend=self.detector_backend if model_hash else None,
                tracker_backend=self.tracker_backend if self.mode == "full" else None,
                tracker_version=None, motion_algorithms={
                    "selected": self.motion_algorithm,
                    "comparison": [key for key, _ in motion_algorithm_choices()]
                    if self.mode == "compare" else [],
                }, parameters=self.settings, device=self.device if model_hash else None,
                output_dir=run_dir,
            )

            pipeline = None
            algorithms = {}
            if self.mode == "compare":
                algorithms = {key: create_motion_algorithm(key, config_for_algorithm(key))
                              for key, _ in motion_algorithm_choices()}
            elif self.mode == "motion":
                algorithms[self.motion_algorithm] = create_motion_algorithm(
                    self.motion_algorithm, config_for_algorithm(self.motion_algorithm))
            if self.mode in ("detect", "motion_yolo", "full"):
                self.status.emit("正在加载模型并检查 GPU 运算……")
                detector = create_detector(self.weights, self.device, self.detector_backend)
                actual_backend = getattr(detector, "backend_id", self.detector_backend)
                actual_detector_backend = actual_backend
                if model_hash:
                    model_id = experiment_store.register_model(
                        project_id, self.weights, model_hash, actual_backend,
                        str(getattr(detector, "version", "unknown")))
                    experiment_store.bind_experiment_model(
                        experiment_id, model_id=model_id, model_path=self.weights,
                        model_sha256=model_hash, detector_backend=actual_backend)
                if detector.device.type == "cuda":
                    self.status.emit(f"模型运行在 {detector.torch.cuda.get_device_name(detector.device)}")
                if self.mode in ("motion_yolo", "full"):
                    selected_motion = create_motion_algorithm(
                        self.motion_algorithm, config_for_algorithm(self.motion_algorithm))
                    pipeline = MSMVideoPipeline(
                        detector, self.settings["roi_size"], self.settings["confidence"],
                        self.settings["confidence"], self.settings["global_interval"],
                        self.settings["imgsz"],
                        iou_threshold=self.settings["iou_threshold"],
                        motion_config=motion_config,
                        motion_algorithm=selected_motion)
                if self.mode == "full":
                    tracker = create_tracker(self.tracker_backend,
                                             self.settings["confidence"], fps,
                                             self.settings["track_buffer"])
                    experiment_store.update_experiment_backends(
                        experiment_id, detector_backend=actual_backend,
                        tracker_version=tracker.version)
                elif detector is not None:
                    experiment_store.update_experiment_backends(
                        experiment_id, detector_backend=actual_backend)

            max_frames = last - first
            processed = 0
            candidate_frames = 0
            candidate_total = 0
            detection_total = 0
            algorithm_stats = {key: {"candidate_frames": 0, "candidate_boxes": 0}
                               for key in algorithms}
            t0 = time.perf_counter()
            show_every = max(1, self.settings["show_every"])
            while processed < max_frames and not self._stopped:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number = first + processed
                active_roi = None
                roi_status = ""
                if self.mode in ("motion", "compare"):
                    if self.mode == "motion":
                        key = self.motion_algorithm
                        result = algorithms[key].process(frame)
                        boxes = [item[0] for item in result.proposals]
                        aligned, diff, mask = result.aligned_previous, result.diff, result.mask
                        annotated = frame.copy()
                        self._candidate_text(annotated, boxes)
                        draw_caption(annotated,
                                     f"{result.diagnostics.get('algorithm', key)} | frame {frame_number} | candidates {len(boxes)}")
                        panels = [preview_image(annotated, boxes=boxes),
                                  preview_image(aligned),
                                  preview_image(make_diff_view(diff) if diff is not None else None, boxes=boxes),
                                  preview_image(make_mask_view(mask, len(boxes)) if mask is not None else None, boxes=boxes)]
                    else:
                        results = {key: algorithm.process(frame)
                                   for key, algorithm in algorithms.items()}
                        boxes_by_algorithm = {
                            key: [item[0] for item in result.proposals]
                            for key, result in results.items()
                        }
                        for key, boxes_for_algorithm in boxes_by_algorithm.items():
                            algorithm_stats[key]["candidate_boxes"] += len(boxes_for_algorithm)
                            algorithm_stats[key]["candidate_frames"] += bool(boxes_for_algorithm)
                        flow_key, diff_key, mog_key = (
                            "flow_frame_difference", "frame_difference", "mog2")
                        annotated_by_algorithm = {}
                        for key, result in results.items():
                            vis = frame.copy()
                            candidate_boxes = boxes_by_algorithm[key]
                            self._candidate_text(vis, candidate_boxes)
                            draw_caption(vis, f"{key} | candidates {len(candidate_boxes)}")
                            annotated_by_algorithm[key] = vis
                        annotated = frame.copy()
                        draw_caption(annotated, f"ALGORITHM COMPARISON | frame {frame_number}")
                        montage = np.vstack((
                            np.hstack((frame, annotated_by_algorithm[flow_key])),
                            np.hstack((annotated_by_algorithm[diff_key], annotated_by_algorithm[mog_key])),
                        ))
                        writer.write(montage)
                        panels = [preview_image(frame),
                                  preview_image(annotated_by_algorithm[flow_key]),
                                  preview_image(annotated_by_algorithm[diff_key]),
                                  preview_image(annotated_by_algorithm[mog_key])]
                        boxes = []
                        aligned = diff = mask = None
                    if self.mode == "motion":
                        writer.write(annotated)
                        if boxes:
                            candidate_frames += 1
                            candidate_total += len(boxes)
                        state_summary = f"{self.motion_algorithm} · 候选 {len(boxes)} 个"
                    else:
                        candidate_total += sum(len(v) for v in boxes_by_algorithm.values())
                        state_summary = "并行对比：" + " · ".join(
                            f"{key} {len(value)}" for key, value in boxes_by_algorithm.items())
                elif self.mode == "detect":
                    detections = detector.predict(
                        frame, self.settings["confidence"], self.settings["imgsz"],
                        self.settings["iou_threshold"])
                    detection_total += len(detections)
                    annotated = frame.copy()
                    for detection in detections:
                        cv2.rectangle(annotated, (detection.x1, detection.y1),
                                      (detection.x2, detection.y2), (70, 210, 120), 2,
                                      cv2.LINE_AA)
                        cv2.putText(annotated, f"{detection.class_id} {detection.confidence:.2f}",
                                    (detection.x1, max(18, detection.y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 210, 120), 2,
                                    cv2.LINE_AA)
                    draw_caption(annotated,
                                 f"DETECTION | frame {frame_number} | detections {len(detections)}")
                    aligned, diff, mask, boxes = frame, None, None, []
                    state_summary = f"检测 {len(detections)} 个 · 追踪未启用"
                else:
                    annotated, detections = pipeline.process(frame)
                    detection_total += len(detections)
                    tracks = tracker.update(detections, frame.shape) if self.mode == "full" else []
                    if self.mode == "full":
                        for track in tracks:
                            cv2.rectangle(annotated, (track.x1, track.y1), (track.x2, track.y2),
                                          (235, 90, 195), 2, cv2.LINE_AA)
                            cv2.putText(annotated, f"ID {track.track_id}",
                                        (track.x1, max(18, track.y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 90, 195), 2)
                    draw_caption(annotated, f"{pipeline.mode} | frame {frame_number} | detections {len(detections)}")
                    motion_result = pipeline.motion_result
                    aligned, diff, mask = (motion_result.aligned_previous,
                                           motion_result.diff, motion_result.mask)
                    boxes = [item[0] for item in motion_result.proposals]
                    active_roi = pipeline.preview_roi
                    roi_status = pipeline.preview_roi_status
                    state_summary = (f"{pipeline.mode} · 检测 {len(detections)} 个" +
                                     (f" · 轨迹 {len(tracks)} 条" if self.mode == "full" else " · 追踪未启用"))
                    state_summary += f" · {pipeline.preview_roi_status}"

                if self.mode not in ("compare", "motion"):
                    writer.write(annotated)
                if processed % show_every == 0 or processed == max_frames - 1:
                    if self.mode in ("compare", "motion"):
                        self.snapshot.emit(frame_number, panels, state_summary)
                        processed += 1
                        if processed % 5 == 0 or processed == max_frames:
                            elapsed = max(0.001, time.perf_counter() - t0)
                            self.progress.emit(processed, max_frames,
                                               f"处理 {processed}/{max_frames} 帧 · {processed / elapsed:.1f} FPS",
                                               candidate_total, 0, processed / elapsed)
                        continue
                    diff_view = make_diff_view(diff) if diff is not None else None
                    mask_view = make_mask_view(mask, len(boxes)) if mask is not None else None
                    panels = [
                        preview_image(annotated, boxes=boxes, roi=active_roi,
                                      roi_status=roi_status),
                        preview_image(aligned),
                        preview_image(diff_view, boxes=boxes),
                        preview_image(mask_view, boxes=boxes),
                    ]
                    self.snapshot.emit(frame_number, panels, state_summary)
                processed += 1
                if processed % 5 == 0 or processed == max_frames:
                    elapsed = max(0.001, time.perf_counter() - t0)
                    cands = (candidate_total if self.mode in ("motion", "compare") else
                             pipeline.motion_candidate_total if pipeline is not None else 0)
                    dets = 0 if self.mode in ("motion", "compare") else detection_total
                    speed = processed / elapsed
                    self.progress.emit(processed, max_frames,
                                       f"处理 {processed}/{max_frames} 帧 · {speed:.1f} FPS",
                                       cands, dets, speed)

            elapsed = max(0.001, time.perf_counter() - t0)
            if self.mode in ("motion", "compare"):
                compare_stats = algorithm_stats
                if self.mode == "compare":
                    for values in compare_stats.values():
                        values["candidate_frames"] = int(values["candidate_frames"])
                summary = {"processed_frames": processed, "candidate_frames": candidate_frames,
                           "candidate_boxes": candidate_total, "elapsed_seconds": round(elapsed, 3),
                           "algorithm_metrics": compare_stats}
                message = (f"完成 {processed} 帧；运动候选框累计 {candidate_total} 个；"
                           f"耗时 {elapsed:.1f} 秒。")
            elif self.mode == "detect":
                summary = {"processed_frames": processed,
                           "detection_boxes": detection_total,
                           "average_fps": round(processed / elapsed, 3),
                           "elapsed_seconds": round(elapsed, 3), "device": self.device}
                message = (f"检测完成 {processed} 帧；检测框 {detection_total} 个；"
                           f"平均 {processed / elapsed:.1f} FPS。追踪未启用。")
            elif self.mode == "motion_yolo":
                summary = {"processed_frames": processed,
                           "candidate_frames": pipeline.motion_candidate_frames,
                           "candidate_boxes": pipeline.motion_candidate_total,
                           "detection_boxes": detection_total,
                           "detection_frames": pipeline.detection_frames,
                           "state_frames": dict(pipeline.state_frame_counts),
                           "average_fps": round(processed / elapsed, 3),
                           "elapsed_seconds": round(elapsed, 3), "device": self.device}
                message = (f"运动模块 + YOLO 检测完成 {processed} 帧；检测框 {detection_total} 个；"
                           f"平均 {processed / elapsed:.1f} FPS。追踪未启用。")
            else:
                summary = {"processed_frames": processed,
                           "candidate_frames": pipeline.motion_candidate_frames,
                           "candidate_boxes": pipeline.motion_candidate_total,
                           "detection_boxes": detection_total,
                           "detection_frames": pipeline.detection_frames,
                           "tracked_frames": tracker.track_frame_count,
                           "unique_track_ids": len(tracker.seen_track_ids),
                           "state_frames": dict(pipeline.state_frame_counts),
                           "average_fps": round(processed / elapsed, 3),
                           "elapsed_seconds": round(elapsed, 3), "device": self.device}
                message = (f"完成 {processed} 帧；检测框 {detection_total} 个；"
                           f"平均 {processed / elapsed:.1f} FPS。")
            manifest = {"mode": self.mode, "video": str(self.video),
                        "start_seconds": self.start_sec, "end_seconds": self.end_sec,
                        "weights": self.weights if self.mode != "motion" else None,
                        "device": self.device if self.mode != "motion" else None,
                        "parameters": self.settings, "summary": summary,
                        "status": "stopped" if self._stopped else "completed",
                        "output_video": str(output)}
            manifest.update({"experiment_id": experiment_id,
                             "motion_algorithm": self.motion_algorithm,
                             "detector_backend": actual_detector_backend or self.detector_backend,
                             "tracker_backend": getattr(tracker, "backend_id", None),
                             "tracker_version": getattr(tracker, "version", None),
                             "runtime": runtime_versions()})
            (run_dir / "parameters.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            writer.release()
            writer = None
            experiment_store.finish_experiment(
                experiment_id, status="stopped" if self._stopped else "completed",
                summary=summary, output_video=output)
            self.done.emit(("已停止。" if self._stopped else "") + message, str(output), summary)
        except Exception as exc:
            if experiment_store is not None:
                try:
                    experiment_store.finish_experiment(
                        experiment_id, status="failed", error=f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass


class ProcessingThread(QThread):
    """Run the plain processing worker off the GUI thread without moving its QObject."""

    def __init__(self, worker: ProcessingWorker, parent: QWidget | None = None):
        super().__init__(parent)
        self.worker = worker

    def run(self) -> None:
        self.worker.run()


class TrainingPreflightWorker(QObject):
    finished = Signal(object, object, object, object)

    def __init__(self, backend: UltralyticsTrainingBackend,
                 request: TrainingRequest):
        super().__init__()
        self.backend = backend
        self.request = request

    @Slot()
    def run(self) -> None:
        errors = self.backend.validate(self.request)
        if errors:
            self.finished.emit(self.request, errors, None, {})
            return
        try:
            run_id = self.backend.start(self.request, validated=True)
        except Exception as exc:
            self.finished.emit(self.request, [f"无法启动训练进程：{exc}"], None, {})
            return
        try:
            runtime = self.backend.runtime_info(self.request)
        except Exception as exc:
            runtime = {"runtime_probe_error": str(exc)}
        self.finished.emit(self.request, [], run_id, runtime)


class DeploymentPreflightThread(QThread):
    completed = Signal(object)

    def __init__(self, request: DeploymentRequest,
                 export: bool = True, convert: bool = True, parent=None):
        super().__init__(parent)
        self.request = request
        self.export = export
        self.convert = convert

    def run(self) -> None:
        errors = validate_request(
            self.request, export=self.export, convert=self.convert)
        self.completed.emit(errors)


class DeploymentThread(QThread):
    line = Signal(str)
    stage = Signal(str)
    completed = Signal(bool, str, str)

    def __init__(self, steps: list[tuple[str, list[str]]], parent=None):
        super().__init__(parent)
        self.steps = steps
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = False
        self._log_lines: list[str] = []

    def cancel(self) -> None:
        self._cancel_requested = True
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def force_kill(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def run(self) -> None:
        success = False
        message = ""
        for stage_name, command in self.steps:
            if self._cancel_requested:
                message = "任务已取消。"
                break
            self.stage.emit(stage_name)
            command_text = "$ " + shlex.join(command)
            self.line.emit(command_text)
            self._log_lines.append(command_text)
            try:
                process = subprocess.Popen(
                    command, cwd=str(HERE), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1,
                )
                self._process = process
                if self._cancel_requested and process.poll() is None:
                    process.terminate()
                assert process.stdout is not None
                for raw_line in iter(process.stdout.readline, ""):
                    line = raw_line.rstrip("\r\n")
                    self._log_lines.append(line)
                    self.line.emit(line)
                    if self._cancel_requested and process.poll() is None:
                        process.terminate()
                return_code = process.wait()
                self._process = None
            except Exception as exc:
                self._process = None
                message = f"无法启动部署命令：{exc}"
                self._log_lines.append(message)
                break
            if self._cancel_requested:
                message = "任务已取消。"
                break
            if return_code != 0:
                message = f"{stage_name}失败，退出码 {return_code}。"
                break
        else:
            success = True
            message = "部署转换流程已完成。"
        self.completed.emit(success, message, "\n".join(self._log_lines[-1200:]))


class ImagePanel(QFrame):
    def __init__(self, title: str, with_description: bool = False):
        super().__init__()
        self.setObjectName("imagePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        self.title = QLabel(title)
        self.title.setObjectName("panelTitle")
        self.image = QLabel("等待分析")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(320, 200)
        self.image.setObjectName("imageSurface")
        self._source_pixmap: QPixmap | None = None
        self.description = QLabel()
        self.description.setObjectName("hint")
        self.description.setWordWrap(True)
        self.description.setVisible(with_description)
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.description)

    def show_image(self, image: np.ndarray) -> None:
        self._source_pixmap = QPixmap.fromImage(cv_image_to_qimage(image))
        self._scale_image()

    def _scale_image(self) -> None:
        if self._source_pixmap is not None:
            self.image.setPixmap(self._source_pixmap.scaled(
                self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        self._scale_image()
        super().resizeEvent(event)


class YoloTraceWorker(QObject):
    """Own the single-image detector session off the GUI thread."""

    ready = Signal(object)
    failed = Signal(int, str)
    status = Signal(int, str)
    shutdown_finished = Signal()

    def __init__(self):
        super().__init__()
        self.detector = None
        self.snapshot = None
        self.image = None
        self.image_path = ""
        self.request_id = 0

    def _close_detector(self) -> None:
        if self.detector is not None:
            try:
                self.detector.close()
            finally:
                self.detector = None
        self.snapshot = None
        self.image = None

    @Slot(int, str, str, str, str, int, float, float)
    def analyze(self, request_id: int, image_path: str, weights: str, device: str,
                backend: str, imgsz: int, confidence: float, iou: float) -> None:
        self.request_id = request_id
        self._close_detector()
        try:
            self.status.emit(request_id, "正在读取图片并加载模型…")
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法解码图片：{image_path}")
            self.detector = create_detector(weights, device=device, backend=backend)
            self.status.emit(request_id, "模型已加载，正在执行一次前向推理…")
            self.snapshot = self.detector.trace_inference(
                image, int(imgsz),
            )
            self.image = image
            self.image_path = image_path
            self._emit_trace(confidence, iou)
        except Exception as exc:
            self._close_detector()
            self.failed.emit(request_id, str(exc))

    @Slot(float, float)
    def update_thresholds(self, confidence: float, iou: float) -> None:
        if self.detector is None or self.snapshot is None or self.image is None:
            return
        try:
            self._emit_trace(confidence, iou)
        except Exception as exc:
            self.failed.emit(self.request_id, f"后处理预览失败：{exc}")

    def _emit_trace(self, confidence: float, iou: float) -> None:
        trace = self.detector.trace_postprocess(self.snapshot, confidence, iou)
        self.ready.emit({
            "request_id": self.request_id,
            "image": self.image,
            "image_path": self.image_path,
            "resized_image": self.snapshot.resized_image,
            "normalized_image": self.snapshot.normalized_image,
            "backend": self.snapshot.backend,
            "input_shape": self.snapshot.input_shape,
            "tensor_range": self.snapshot.tensor_range,
            "end2end": self.snapshot.end2end,
            "trace": trace,
            "names": getattr(self.detector, "names", {}),
            "device": str(getattr(self.detector, "device", "unknown")),
        })

    @Slot(int)
    def invalidate(self, request_id: int) -> None:
        self.request_id = request_id
        self._close_detector()
        self.status.emit(request_id, "输入或模型设置已变化，请重新分析单图。")

    @Slot()
    def shutdown(self) -> None:
        self._close_detector()
        self.shutdown_finished.emit()


class TunerWindow(QMainWindow):
    yolo_inference_requested = Signal(int, str, str, str, str, int, float, float)
    yolo_thresholds_requested = Signal(float, float)
    yolo_invalidate_requested = Signal(int)
    yolo_shutdown_requested = Signal()

    DEFAULTS = {
        "threshold_sigma": 2.0, "threshold_offset": 8.0, "threshold_window": 21,
        "morph_kernel": 3, "open_iterations": 1, "close_iterations": 2,
        "min_area": 12, "max_proposals": 20, "edge_margin_percent": 3.0,
        "history_window": 3,
        "roi_size": 640,
        "confidence": 0.60, "iou_threshold": 0.45,
        "global_interval": 15, "imgsz": 640,
        "track_buffer": 30, "show_every": 15,
        "mog2_history": 500, "mog2_var_threshold": 16.0,
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSM-YOLO 动态区域调参工作台")
        self.resize(1580, 990)
        self.setMinimumSize(1200, 760)
        self.snapshots: list[tuple[int, list[np.ndarray], str]] = []
        self.thread: QThread | None = None
        self.worker: ProcessingWorker | None = None
        self.controls = {}
        self.mode = "motion"
        self.training_backend = UltralyticsTrainingBackend()
        self.training_store = ExperimentStore(DATABASE)
        self.deployment_store = self.training_store
        self.training_run_id: str | None = None
        self.training_request: TrainingRequest | None = None
        self._training_log_prefix = ""
        self.training_validation_thread: QThread | None = None
        self.training_validation_worker: TrainingPreflightWorker | None = None
        self.deployment_thread: DeploymentThread | None = None
        self.deployment_preflight_thread: DeploymentPreflightThread | None = None
        self.deployment_run_id: str | None = None
        self.deployment_started_at: str | None = None
        self._build()
        self._start_yolo_trace_worker()
        self.training_timer = QTimer(self)
        self.training_timer.setInterval(1000)
        self.training_timer.timeout.connect(self._poll_training)

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(20, 16, 20, 18)
        main.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("MSM-YOLO 调参工作台")
        title.setObjectName("pageTitle")
        subtitle = QLabel("动态区域提取  ·  局部检测  ·  ByteTrack")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.status_badge = QLabel("就绪")
        self.status_badge.setObjectName("statusBadge")
        header.addWidget(self.status_badge, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        main.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(382)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(10)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("settingsTabs")
        self.tabs.addTab(self._video_tab(), "视频")
        self.tabs.addTab(self._motion_tab(), "运动参数")
        self.tabs.addTab(self._tracking_tab(), "检测追踪")
        self.tabs.addTab(self._experiments_tab(), "实验记录")
        self._active_algorithm_key = motion_algorithm_choices()[0][0]
        self.algorithm_configs = {
            key: self._capture_algorithm_config() for key, _ in motion_algorithm_choices()
        }
        self.motion_algorithm_combo.currentIndexChanged.connect(
            self._on_motion_algorithm_changed)
        side.addWidget(self.tabs, 1)
        actions = QHBoxLayout()
        self.start_button = QPushButton("▶  开始分析")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        actions.addWidget(self.start_button, 1)
        actions.addWidget(self.stop_button)
        side.addLayout(actions)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        side.addWidget(self.progress_bar)
        self.status_text = QLabel("先选取较短片段，观察运动诊断图再调参。")
        self.status_text.setObjectName("statusText")
        self.status_text.setWordWrap(True)
        side.addWidget(self.status_text)
        body.addWidget(self.sidebar)

        content = QVBoxLayout()
        content.setSpacing(10)
        metrics = QHBoxLayout()
        self.metric_values = []
        for label in ("已处理帧数", "候选区域", "检测框", "处理速度"):
            card = QFrame()
            card.setObjectName("metricCard")
            v = QVBoxLayout(card)
            v.setContentsMargins(14, 10, 14, 10)
            name = QLabel(label)
            name.setObjectName("metricName")
            value = QLabel("—")
            value.setObjectName("metricValue")
            v.addWidget(name)
            v.addWidget(value)
            metrics.addWidget(card, 1)
            self.metric_values.append(value)
        content.addLayout(metrics)

        diag_header = QHBoxLayout()
        diag_title = QLabel("过程诊断")
        diag_title.setObjectName("sectionTitle")
        diag_header.addWidget(diag_title)
        diag_header.addStretch(1)
        self.frame_info = QLabel("等待分析")
        self.frame_info.setObjectName("subtitle")
        diag_header.addWidget(self.frame_info)
        content.addLayout(diag_header)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.panels = [
            ImagePanel("候选区域 / 检测与追踪结果"),
            ImagePanel("相机补偿后的前一帧"),
            ImagePanel("帧差 · 百分位对比度增强"),
            ImagePanel("二值运动掩膜 · 候选框标注"),
        ]
        for index, panel in enumerate(self.panels):
            grid.addWidget(panel, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        content.addLayout(grid, 1)
        scrub = QHBoxLayout()
        scrub.addWidget(QLabel("预览帧"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.show_snapshot)
        scrub.addWidget(self.frame_slider, 1)
        self.scrub_label = QLabel("0 / 0")
        self.scrub_label.setMinimumWidth(75)
        scrub.addWidget(self.scrub_label)
        content.addLayout(scrub)
        body.addLayout(content, 1)
        self.workspace_tabs = QTabWidget()
        dashboard = QWidget()
        dashboard_layout = QVBoxLayout(dashboard)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.addLayout(body)
        self.workspace_tabs.addTab(dashboard, "分析工作台")
        self.workspace_tabs.addTab(self._algorithm_guide_tab(), "算法原理与可视化")
        self.workspace_tabs.addTab(self._training_center_page(), "训练中心")
        self.workspace_tabs.addTab(self._deployment_center_page(), "RK3588 部署中心")
        main.addWidget(self.workspace_tabs, 1)
        self.setStyleSheet(self._stylesheet())

    @staticmethod
    def _scroll_page(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _algorithm_guide_tab(self) -> QWidget:
        guide = QWidget()
        layout = QVBoxLayout(guide)
        layout.setContentsMargins(10, 10, 10, 10)
        heading = QLabel("算法学习与过程实验室")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        intro = QLabel(
            "沿着“视频变化 → 候选区域 → 单图检测 → 误差评估”学习。选择自己的视频或图像，"
            "逐阶段查看本项目实际计算出的中间结果；讲解同时列出参数影响、适用条件和常见失败原因。"
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        reading = QLabel(
            "读图顺序：Backbone 逐层降采样提特征 → Neck 将高/低分辨率语义融合 → Head 在多个尺度预测类别与边框。"
            "P3/stride 8 保留更细空间信息，常用于小目标；P4、P5 感受野更大。实际小目标效果仍要通过验证集检查。"
            "注意：原版 YOLOv5 与 Ultralytics YOLOv5u 的检测头不同，图中的代际名不能替代权重格式说明。"
        )
        reading.setObjectName("hint")
        reading.setWordWrap(True)
        layout.addWidget(reading)
        references = QLabel(
            '参考资料：<a href="https://docs.opencv.org/5.0/tutorials/others/optical_flow.html">'
            'OpenCV 光流</a> · '
            '<a href="https://docs.opencv.org/4.13.0/d1/dc5/tutorial_background_subtraction.html">'
            'OpenCV 背景减除</a> · '
            '<a href="https://docs.ultralytics.com/guides/yolo-architecture/">'
            'Ultralytics 网络结构</a> · '
            '<a href="https://docs.voxel51.com/user_guide/evaluation.html">'
            'FiftyOne 误差分析</a>'
        )
        references.setOpenExternalLinks(True)
        references.setObjectName("hint")
        layout.addWidget(references)
        self.guide_tabs = QTabWidget()
        self.guide_tabs.addTab(self._motion_steps_page(), "运动提取实验")
        self.guide_tabs.addTab(self._yolo_guide_page(), "YOLO 推理实验")
        self.guide_tabs.addTab(self._model_architecture_page(), "网络结构与代际")
        self.guide_tabs.addTab(self._metrics_page(), "指标与误差分析")
        layout.addWidget(self.guide_tabs, 1)
        return guide

    def _motion_steps_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("运动算法"))
        self.lab_algorithm_combo = QComboBox()
        for algorithm_id, label in motion_algorithm_choices():
            self.lab_algorithm_combo.addItem(label, algorithm_id)
        self.lab_algorithm_combo.currentIndexChanged.connect(
            self._update_motion_learning_card)
        controls.addWidget(self.lab_algorithm_combo)
        controls.addWidget(QLabel("当前帧"))
        self.lab_frame_spin = QSpinBox()
        self.lab_frame_spin.setRange(0, 10000000)
        self.lab_frame_spin.setValue(30)
        controls.addWidget(self.lab_frame_spin)
        self.lab_simulate_button = QPushButton("提取这一步的结果")
        self.lab_simulate_button.clicked.connect(self.simulate_motion_steps)
        controls.addWidget(self.lab_simulate_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.motion_learning_card = QGroupBox("算法机理 · 假设 · 失效场景")
        learning_layout = QVBoxLayout(self.motion_learning_card)
        self.motion_learning_text = QLabel()
        self.motion_learning_text.setObjectName("hint")
        self.motion_learning_text.setWordWrap(True)
        learning_layout.addWidget(self.motion_learning_text)
        layout.addWidget(self.motion_learning_card)
        self._update_motion_learning_card()
        self.lab_status = QLabel("视频路径来自“分析工作台”的输入视频。选一个帧号后点击按钮查看各阶段。")
        self.lab_status.setObjectName("statusText")
        self.lab_status.setWordWrap(True)
        layout.addWidget(self.lab_status)
        self.motion_diagnosis_label = QLabel(
            "结果判读：观察前景比例、连通域数量和候选框位置。运动掩膜只提供时空变化线索，"
            "需结合 YOLO 与连续帧确认目标。"
        )
        self.motion_diagnosis_label.setObjectName("hint")
        self.motion_diagnosis_label.setWordWrap(True)
        layout.addWidget(self.motion_diagnosis_label)
        self.lab_stage_panels = [ImagePanel("等待算法模拟", with_description=True)
                                 for _ in range(8)]
        for panel in self.lab_stage_panels:
            panel.image.setMinimumSize(220, 170)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for index, panel in enumerate(self.lab_stage_panels):
            grid.addWidget(panel, index // 3, index % 3)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        for row in range(3):
            grid.setRowStretch(row, 1)
        layout.addLayout(grid)
        return self._scroll_page(page)

    def _update_motion_learning_card(self, *_args) -> None:
        if not hasattr(self, "motion_learning_text"):
            return
        algorithm_id = self.lab_algorithm_combo.currentData()
        guides = {
            "flow_frame_difference": (
                "机理：灰度化 + 5×5 高斯平滑 → Shi–Tomasi 角点 → 金字塔 LK 稀疏光流 → "
                "RANSAC 单应性估计（重投影阈值 3 px）→ 把前一帧变换到当前视角 → "
                "D(x,y)=|Iₜ−warp(Iₜ₋₁,H)| → 局部统计阈值 T=μ+kσ+C → 开/闭运算 → 连通域与 ROI。\n"
                "适用：相机缓慢平移/转动、背景有足够纹理；相机补偿后再做帧差，能压低整幅背景抖动。\n"
                "关键参数：角点密度影响配准覆盖；阈值窗口、k 与偏置 C 控制运动像素灵敏度；边缘屏蔽用于排除 warp 无效区。\n"
                "容易失效：低纹理天空/水面、运动模糊、视差/滚动快门、前景占比过大时单应性会被目标带偏。"
            ),
            "frame_difference": (
                "机理：灰度化 + 高斯平滑 → D(x,y)=|Iₜ−Iₜ₋₁| → 局部阈值 T=μ+kσ+C → "
                "形态学开运算去孤立点、闭运算连接断裂 → 连通域按面积与边缘规则筛候选框。\n"
                "适用：固定相机或相邻帧背景变化小，目标运动能形成明显像素差。\n"
                "关键参数：窗口越大，局部背景统计范围越广；k/C 越大，弱变化越容易被舍弃；面积上下限决定候选尺度。\n"
                "容易失效：相机移动、树叶/水面摆动、自动曝光变化都会产生大面积差分；该方法只说明‘像素变了’，不等于‘有目标’。"
            ),
            "mog2": (
                "机理：每个像素维护自适应高斯混合背景模型；新像素与模型匹配后更新背景/前景。"
                "当前实现用 history 控制学习历史、varThreshold 控制前景匹配门限；OpenCV 阴影标记为 127，"
                "本流程仅保留确定前景（>200），前两帧作为模型预热。\n"
                "适用：固定视角、背景逐渐变化、持续视频输入；比单帧差更能吸收缓慢亮度与背景变化。\n"
                "关键参数：history 越大背景适应越慢；varThreshold 越低越敏感；阴影检测可减少阴影误检但会增加处理。\n"
                "容易失效：摄像机大幅运动、长期静止的目标逐渐被学成背景、快速照明突变或动态纹理会造成误检/漏检。"
            ),
        }
        self.motion_learning_text.setText(guides.get(algorithm_id, "选择算法以查看对应说明。"))

    def simulate_motion_steps(self) -> None:
        video_path = Path(self.video_edit.text()).expanduser()
        if not video_path.is_file():
            QMessageBox.warning(self, "视频不存在", f"找不到视频文件：\n{video_path}")
            return
        frame_number = self.lab_frame_spin.value()
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or total < 2:
                raise RuntimeError("无法打开视频，或视频帧数不足 2 帧。")
            frame_number = min(frame_number, total - 1)
        except Exception as exc:
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        finally:
            cap.release()

        try:
            algorithm_id = self.lab_algorithm_combo.currentData()
            values = self.settings()
            config = dict(values.get("motion_algorithm_configs", {}).get(algorithm_id, {}))
            config.update({key: values[key] for key in (
                "threshold_sigma", "threshold_offset", "threshold_window", "morph_kernel",
                "open_iterations", "close_iterations", "min_area", "max_proposals",
                "mog2_history", "mog2_var_threshold", "mog2_detect_shadows",
            ) if key not in config})
            if "edge_margin_percent" in config:
                config["edge_margin_ratio"] = float(config["edge_margin_percent"]) / 100.0
            algorithm = create_motion_algorithm(algorithm_id, config)
            start_frame = max(0, frame_number - 300)
            cap = cv2.VideoCapture(str(video_path))
            result = None
            current = None
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for index in range(start_frame, frame_number + 1):
                    ok, current = cap.read()
                    if not ok:
                        raise RuntimeError(f"无法读取视频第 {index} 帧。")
                    result = algorithm.process(current)
                    if index % 20 == 0 or index == frame_number:
                        self.lab_status.setText(
                            f"正在预热算法状态：{index - start_frame + 1}/"
                            f"{frame_number - start_frame + 1} 帧…")
                        QApplication.processEvents()
            finally:
                cap.release()
            if result is None:
                raise RuntimeError("未能生成算法输出。")
        except Exception as exc:
            QMessageBox.warning(self, "算法处理失败", str(exc))
            return

        descriptions = {
            "原始输入帧": "当前帧 Iₜ，后续所有阶段都从该帧和前序状态计算。对照候选框时注意这里仍是原始分辨率。",
            "预处理灰度帧": "BGR 转灰度后使用 5×5 GaussianBlur。减少颜色通道带来的计算量，并压低高频噪声；过强平滑会削弱小 UAV 的微弱边缘。",
            "光流特征点与相机补偿": "绿色圆点是 LK 跟踪到的下一帧位置，黄色线段表示角点位移。RANSAC 用多数一致匹配估计全局单应性；稀疏/集中匹配时补偿不可靠。",
            "对齐后上一帧": "通过 H 将前一帧 warp 到当前视角。差分只在有效重叠区计算，图像边缘外推区域被屏蔽，避免黑边成为候选。",
            "帧差响应": "D(x,y)=|Iₜ(x,y)−Iₜ₋₁(x,y)|；亮度表示像素变化幅度，不包含类别或目标语义。",
            "MOG2 前景响应": "每像素高斯混合背景模型输出的前景强度；阴影常被编码为中间灰度，当前实现随后将其排除。",
            "阈值分割结果": "帧差使用 T=局部均值+k×局部标准差+C；仅 D>T 的像素置为白色。MOG2 则保留确定前景像素。",
            "形态学清理后掩膜": "先开运算去除小孤点，再闭运算填小孔/连接邻近像素；操作过强会吞掉微小目标或把背景粘连起来。",
            "连通区域候选框": "连通域按像素面积、边缘带和 top-K 规则筛选；候选框是给后续 YOLO 的搜索区域，不是最终检测结论。",
        }
        if algorithm_id == "mog2":
            descriptions["阈值分割结果"] = (
                "将 MOG2 输出大于 200 的确定前景保留为白色；阴影值 127 与背景值 0 均置零。")
        if algorithm_id == "frame_difference":
            descriptions["光流特征点与相机补偿"] = (
                "纯帧差不做光流或相机补偿。若相机移动，背景纹理也会进入帧差响应；可与当前帧差算法并排对照。")
        for index, panel in enumerate(self.lab_stage_panels):
            if index >= len(result.stages):
                panel.hide()
                continue
            name, image = list(result.stages.items())[index]
            panel.show()
            panel.title.setText(name)
            display = make_diff_view(image) if ("响应" in name or "帧差" in name) else image
            panel.show_image(preview_image(display, size=(560, 300)))
            panel.description.setText(descriptions.get(name, "中间处理结果。"))
            panel.setToolTip(descriptions.get(name, "中间处理结果。"))
        diagnostics = result.diagnostics
        self.lab_status.setText(
            f"视频：{video_path.name}　帧：{frame_number}　算法：{algorithm_id}　"
            f"候选框：{len(result.proposals)}　前景比例："
            f"{diagnostics.get('foreground_ratio', 0.0):.3%}　"
            f"连通域：{diagnostics.get('component_count', '—')}"
        )
        foreground_ratio = float(diagnostics.get("foreground_ratio", 0.0))
        components = diagnostics.get("component_count")
        if not result.proposals:
            advice = (
                "没有候选框：先看‘帧差响应/前景响应’是否有目标信号；若响应可见但掩膜为空，"
                "降低阈值偏置或 k、检查面积下限；若前景响应本身为空，检查相邻帧、MOG2 预热或相机补偿。"
            )
        elif foreground_ratio > 0.25:
            advice = (
                "前景比例偏高，背景运动/纹理可能占主导。提高阈值或 MOG2 varThreshold，"
                "检查相机补偿是否失效，并在候选框阶段排查边缘框。"
            )
        elif components is not None and int(components) > 150:
            advice = (
                "连通域很多但候选有限：噪点正在被面积/top-K 规则裁剪。可调整开运算、最小面积与阈值，"
                "同时核对小目标是否被形态学操作删除。"
            )
        elif foreground_ratio < 0.0001:
            advice = (
                "前景极稀疏：若目标肉眼可见，降低阈值 k/偏置或 varThreshold；"
                "若是 MOG2，还要检查目标是否已被背景模型吸收。"
            )
        else:
            advice = (
                "掩膜与候选数量处于可观察范围。逐框检查是否覆盖真实移动物；"
                "相邻帧候选位置应连续，再由 YOLO 验证外观类别。"
            )
        self.motion_diagnosis_label.setText(f"本帧判读：{advice}")

    def _yolo_guide_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        explanation = QLabel(
            "检测过程分成两类变化：前处理只改变图像坐标/数值表示；后处理只筛选模型已经产生的候选。"
            "拖动阈值时复用同一次前向输出，可直接观察‘候选 → 置信度筛选 → NMS/端到端输出 → 原图映射’的数量与位置变化。"
        )
        explanation.setObjectName("hint")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        layout.addWidget(self._yolo_single_image_lab())
        layout.addStretch(1)
        return self._scroll_page(page)

    def _model_architecture_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        intro = QLabel(
            "按代表性目标检测代际查看分阶段网络图。图中只强调主要特征流和关键模块；"
            "不同 n/s/m/l/x 规格共享阶段拓扑，但深度与通道规模不同。"
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        architecture_guide = QLabel(
            "阅读顺序：Backbone 把图像编码成不同层级的特征；Neck 将高层语义与高分辨率细节融合；"
            "Head 在多个尺度上输出类别与边界框。P3 通常保留更细的空间信息，适合观察小目标特征，"
            "但是否提升 UAV 检测效果仍要用同一验证集实测。"
            "图是便于学习的阶段示意，具体层数、通道数和连接以所选模型 YAML 为准。"
        )
        architecture_guide.setObjectName("hint")
        architecture_guide.setWordWrap(True)
        layout.addWidget(architecture_guide)
        compatibility = QLabel(
            "兼容提示：原版 YOLOv5 仓库权重与 Ultralytics 的 YOLOv5u 属于不同实现/权重路径；"
            "加载器应按来源选择，不能只凭文件名推断结构。模型代际图展示代表性目标检测结构，不表示所有变体完全相同。"
        )
        compatibility.setWordWrap(True)
        compatibility.setStyleSheet("padding: 9px; border-left: 3px solid #d39b36; background: #fff8e8;")
        layout.addWidget(compatibility)

        selector = QHBoxLayout()
        selector.addWidget(QLabel("模型代际"))
        self.model_profile_combo = QComboBox()
        for profile in MODEL_PROFILES:
            self.model_profile_combo.addItem(profile["name"], profile["id"])
        selector.addWidget(self.model_profile_combo, 1)
        self.model_source_link = QLabel()
        self.model_source_link.setOpenExternalLinks(True)
        selector.addWidget(self.model_source_link)
        layout.addLayout(selector)

        diagram_card = QGroupBox("Backbone → Neck → Head · 多尺度特征流")
        diagram_layout = QVBoxLayout(diagram_card)
        self.model_architecture_view = QSvgWidget()
        self.model_architecture_view.setMinimumHeight(360)
        diagram_layout.addWidget(self.model_architecture_view)
        layout.addWidget(diagram_card)

        details = QGridLayout()
        self.model_backbone_card = self._model_detail_card("Backbone · 主干")
        self.model_neck_card = self._model_detail_card("Neck · 特征融合")
        self.model_head_card = self._model_detail_card("Head · 检测头与输出")
        self.model_change_card = self._model_detail_card("关键变化与阅读提示")
        details.addWidget(self.model_backbone_card, 0, 0)
        details.addWidget(self.model_neck_card, 0, 1)
        details.addWidget(self.model_head_card, 1, 0)
        details.addWidget(self.model_change_card, 1, 1)
        layout.addLayout(details)

        self.model_profile_combo.currentIndexChanged.connect(self._show_model_profile)
        self._show_model_profile(0)
        layout.addStretch(1)
        return self._scroll_page(page)

    @staticmethod
    def _model_detail_card(title: str) -> QGroupBox:
        card = QGroupBox(title)
        layout = QVBoxLayout(card)
        label = QLabel()
        label.setWordWrap(True)
        label.setMinimumHeight(58)
        layout.addWidget(label)
        card.detail_label = label
        return card

    def _show_model_profile(self, index: int) -> None:
        if not MODEL_PROFILES or index < 0:
            return
        profile = MODEL_PROFILES[index]
        self.model_architecture_view.load(QByteArray(architecture_svg(profile)))
        self.model_backbone_card.detail_label.setText(profile["backbone"])
        self.model_neck_card.detail_label.setText(profile["neck"])
        self.model_head_card.detail_label.setText(
            f"{profile['head']}\n输出：{profile['outputs']}\n{profile['detail']}"
        )
        self.model_change_card.detail_label.setText(
            f"{profile['change']}\n结构参考：{profile['yaml']}"
        )
        self.model_source_link.setText(
            f'<a href="{profile["url"]}">官方模型说明 ↗</a>'
        )

    def _metrics_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "这些指标用于比较模型检测质量、运行速度和计算规模。Precision/Recall/AP/mAP 必须从带标注验证集计算；"
            "FPS、延迟、参数量和 FLOPs 也要注明设备、输入尺寸、精度与 batch。"
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        evaluation_flow = QHBoxLayout()
        for title, body in (
            ("1 · 固定条件", "同一验证集、类别、输入尺寸、设备和置信度扫值范围。"),
            ("2 · 匹配标注", "预测框按类别与评测 IoU 匹配真值；NMS IoU 与评测 IoU 是两回事。"),
            ("3 · 回看样本", "拆分 TP / FP / FN，再按尺寸、位置、类别和分数找误差来源。"),
        ):
            card = QGroupBox(title)
            card_layout = QVBoxLayout(card)
            label = QLabel(body)
            label.setWordWrap(True)
            label.setMinimumHeight(54)
            card_layout.addWidget(label)
            evaluation_flow.addWidget(card, 1)
        layout.addLayout(evaluation_flow)
        table = QTableWidget(9, 3)
        table.setHorizontalHeaderLabels(["指标", "它回答什么", "比较时要注意"])
        rows = [
            ("Precision", "预测框中有多少是真正目标。", "误检越多通常越低；依赖置信度阈值。"),
            ("Recall", "标注目标中有多少被检出。", "漏检越多通常越低；依赖置信度阈值。"),
            ("AP / mAP", "PR 曲线下的面积 / 各类别 AP 均值。", "需同一验证集、匹配规则和类别集合。"),
            ("mAP50–95", "IoU 0.50 至 0.95 多档阈值平均 AP。", "较高阈值会更严格评价定位精度。"),
            ("AP_S", "小尺寸目标上的 AP。", "报告小目标定义与数据集尺寸分布。"),
            ("IoU", "两个矩形框的交集面积 / 并集面积。", "后处理 NMS IoU 不等同评测匹配 IoU。"),
            ("延迟 / FPS", "单帧耗时 / 单位时间处理帧数。", "注明设备、预热、精度、输入尺寸和 batch。"),
            ("参数量", "可学习参数规模。", "不直接代表内存峰值或实际速度。"),
            ("FLOPs", "一次前向的大致浮点计算量。", "不同工具的乘加计数口径可能不同。"),
        ]
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setMinimumHeight(380)
        layout.addWidget(table)
        error_card = QGroupBox("面向 UAV 小目标的误差排查")
        error_layout = QVBoxLayout(error_card)
        error_label = QLabel(
            "漏检 FN：按目标像素尺寸、模糊/遮挡、置信度区间和画面位置分组；先检查是否缩放后只剩几像素。\n"
            "误检 FP：把森林纹理、建筑边缘、阴影、运动补偿边缘伪影单独标记；检查运动候选是否把 YOLO 引到错误 ROI。\n"
            "定位错误：在固定评测 IoU 下看框偏移；先分清模型定位误差与 letterbox 坐标逆变换错误。\n"
            "推荐视图：按 TP/FP/FN 筛选具体图片、画置信度/IoU 分布，再回到原图复核标注。"
        )
        error_label.setWordWrap(True)
        error_layout.addWidget(error_label)
        layout.addWidget(error_card)
        note = QLabel(
            "本工作台当前提供定义和误差分析流程，不会从无标注视频生成 Precision、Recall 或 mAP。"
            "接入带标注验证集后，才能计算真实结果并按实验保存。参考："
            '<a href="https://docs.voxel51.com/user_guide/evaluation.html">FiftyOne 检测评估</a> · '
            '<a href="https://docs.ultralytics.com/modes/val/">Ultralytics 验证模式</a>'
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _training_center_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        intro = QLabel("Ultralytics · 目标检测训练中心")
        intro.setObjectName("sectionTitle")
        layout.addWidget(intro)
        intro_text = QLabel(
            "训练由所选 Conda 环境中的 Ultralytics 原生训练器执行，参数以可编辑控件呈现并写入每次实验快照。"
            "支持 Ultralytics 目标检测模型；训练在后台子进程运行，不会阻塞页面，也不会自动安装依赖。"
        )
        intro_text.setObjectName("hint")
        intro_text.setWordWrap(True)
        layout.addWidget(intro_text)

        config = QGroupBox("运行环境与训练配置")
        form = QGridLayout(config)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        self.training_edits: dict[str, QLineEdit] = {}

        saved = self._load_training_settings()
        defaults = {
            "python": sys.executable,
            "dataset": str(ARD100_DATA_YAML) if ARD100_DATA_YAML.is_file() else "",
            "model": "yolo11n.pt",
            "cfg": "",
            "output": str(HERE / "runs" / "training"),
            "name": "",
        }
        defaults.update(saved)
        if "yolov5_rk" in str(defaults.get("python", "")):
            defaults["python"] = sys.executable
        if str(defaults.get("model", "")) == str(DEFAULT_WEIGHTS):
            defaults["model"] = "yolo11n.pt"

        path_specs = [
            ("python", "训练环境 Python", "python"),
            ("dataset", "数据集 YAML", "yaml"),
            ("model", "预训练权重 / 模型 YAML / 模型名", "model"),
            ("cfg", "训练参数 YAML（可选）", "yaml"),
            ("output", "训练输出目录", "directory"),
        ]
        for row, (key, label, kind) in enumerate(path_specs):
            edit = QLineEdit(defaults[key])
            edit.setPlaceholderText({
                "python": "例如 ~/miniconda3/envs/ultralytics/bin/python",
                "dataset": "数据集配置文件，例如 ARD100.yaml",
                "model": "例如 yolo11n.pt、YOLO 配置 YAML 或本地权重",
                "cfg": "可选 Ultralytics 训练配置 YAML；控件参数将覆盖同名项",
                "output": "保存训练记录、权重和结果图表的目录",
            }[key])
            self.training_edits[key] = edit
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(edit, row, 1)
            browse = QPushButton("浏览…")
            browse.clicked.connect(lambda _checked=False, field=edit, field_kind=kind:
                                   self._browse_training_path(field, field_kind))
            form.addWidget(browse, row, 2)

        run_name = QLineEdit(defaults["name"])
        run_name.setPlaceholderText("留空时自动生成训练任务名称")
        self.training_edits["name"] = run_name
        form.addWidget(QLabel("任务名称"), len(path_specs), 0)
        form.addWidget(run_name, len(path_specs), 1, 1, 2)

        self.training_spins: dict[str, QSpinBox] = {}
        numeric = [
            ("epochs", "训练轮数", 1, 10000, 100),
            ("batch", "Batch size（-1 自动）", -1, 512, 16),
            ("imgsz", "输入尺寸", 32, 4096, 640),
            ("workers", "数据加载 workers", 0, 64, 0),
            ("patience", "EarlyStopping patience", 0, 10000, 100),
        ]
        numeric_row = len(path_specs) + 1
        for index, (key, label, minimum, maximum, initial) in enumerate(numeric):
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(int(saved.get(key, initial)))
            spin.setKeyboardTracking(False)
            self.training_spins[key] = spin
            column = (index % 3) * 2
            row = numeric_row + index // 3
            form.addWidget(QLabel(label), row, column)
            form.addWidget(spin, row, column + 1)

        choice_row = numeric_row + 2
        form.addWidget(QLabel("设备"), choice_row, 0)
        self.training_device_combo = QComboBox()
        self.training_device_combo.addItem("GPU 0", "0")
        self.training_device_combo.addItem("CPU", "cpu")
        self.training_device_combo.setCurrentIndex(
            1 if str(saved.get("device", "0")).lower() == "cpu" else 0)
        form.addWidget(self.training_device_combo, choice_row, 1)
        form.addWidget(QLabel("优化器"), choice_row, 2)
        self.training_optimizer_combo = QComboBox()
        for optimizer in ("auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"):
            self.training_optimizer_combo.addItem(optimizer, optimizer)
        optimizer_index = self.training_optimizer_combo.findData(saved.get("optimizer", "auto"))
        self.training_optimizer_combo.setCurrentIndex(max(0, optimizer_index))
        form.addWidget(self.training_optimizer_combo, choice_row, 3)
        layout.addWidget(config)

        advanced = QGroupBox("训练参数（按类别完整配置）")
        advanced_layout = QVBoxLayout(advanced)
        advanced_hint = QLabel(
            "以下参数会直接传入 Ultralytics train()；将鼠标悬停在参数名上可查看说明。"
            "空值表示采用 Ultralytics 默认行为。不同模型结构可能忽略不适用的参数。"
        )
        advanced_hint.setObjectName("hint")
        advanced_hint.setWordWrap(True)
        advanced_layout.addWidget(advanced_hint)
        self.training_param_controls: dict[str, QWidget] = {}
        self.training_param_defaults: dict[str, object] = {}
        self.training_param_tabs = QTabWidget()
        for group_name, specs in TRAINING_PARAMETER_GROUPS:
            tab = QWidget()
            grid = QGridLayout(tab)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(6)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            for index, spec in enumerate(specs):
                key, label, kind, default, *extra = spec
                saved_value = saved.get(key, default)
                control = self._make_training_parameter_control(
                    key, kind, default, saved_value, extra)
                self.training_param_controls[key] = control
                self.training_param_defaults[key] = default
                grid_row, pair = divmod(index, 2)
                col = pair * 2
                label_widget = QLabel(label)
                label_widget.setToolTip(f"Ultralytics 参数：{key}")
                control.setToolTip(f"参数名：{key}；当前值会记录到训练配置快照。")
                grid.addWidget(label_widget, grid_row, col)
                grid.addWidget(control, grid_row, col + 1)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            self.training_param_tabs.addTab(tab, group_name)
            tab.setLayout(grid)
        advanced_layout.addWidget(self.training_param_tabs)
        layout.addWidget(advanced)

        status_row = QHBoxLayout()
        self.training_preflight_label = QLabel("配置训练环境后，点击“开始训练”将先执行依赖与数据集预检。")
        self.training_preflight_label.setObjectName("hint")
        self.training_preflight_label.setWordWrap(True)
        status_row.addWidget(self.training_preflight_label, 1)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setMinimumWidth(180)
        status_row.addWidget(self.training_progress)
        layout.addLayout(status_row)

        metric_group = QGroupBox("最近轮次指标")
        metric_layout = QHBoxLayout(metric_group)
        self.training_metrics_label = QLabel("等待训练任务")
        self.training_metrics_label.setWordWrap(True)
        metric_layout.addWidget(self.training_metrics_label)
        layout.addWidget(metric_group)

        actions = QHBoxLayout()
        self.training_start_button = QPushButton("▶  开始训练")
        self.training_start_button.setObjectName("primaryButton")
        self.training_start_button.clicked.connect(self.start_training)
        self.training_stop_button = QPushButton("停止")
        self.training_stop_button.setEnabled(False)
        self.training_stop_button.clicked.connect(self.stop_training)
        self.training_resume_button = QPushButton("从 last.pt 恢复所选任务")
        self.training_resume_button.clicked.connect(self.resume_selected_training)
        self.training_refresh_button = QPushButton("刷新记录")
        self.training_refresh_button.clicked.connect(self.refresh_training_runs)
        actions.addWidget(self.training_start_button)
        actions.addWidget(self.training_stop_button)
        actions.addWidget(self.training_resume_button)
        actions.addStretch(1)
        actions.addWidget(self.training_refresh_button)
        layout.addLayout(actions)

        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumBlockCount(4000)
        self.training_log.setMinimumHeight(130)
        self.training_log.setPlaceholderText("任务预检、训练日志和错误信息会显示在这里。")
        log_layout.addWidget(self.training_log)
        layout.addWidget(log_group)

        history_group = QGroupBox("本地训练记录")
        history_layout = QVBoxLayout(history_group)
        self.training_table = QTableWidget(0, 7)
        self.training_table.setHorizontalHeaderLabels(
            ["状态", "轮次", "任务名称", "数据集", "mAP50-95", "best.pt", "输出目录"])
        self.training_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.training_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.training_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.training_table.verticalHeader().setVisible(False)
        self.training_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch)
        self.training_table.setMinimumHeight(180)
        history_layout.addWidget(self.training_table)
        history_actions = QHBoxLayout()
        self.training_open_button = QPushButton("打开所选输出目录")
        self.training_open_button.clicked.connect(self.open_selected_training_output)
        self.training_use_weight_button = QPushButton("将 best.pt 用于检测")
        self.training_use_weight_button.clicked.connect(self.use_selected_training_weight)
        history_actions.addWidget(self.training_open_button)
        history_actions.addWidget(self.training_use_weight_button)
        history_actions.addStretch(1)
        history_layout.addLayout(history_actions)
        layout.addWidget(history_group)
        layout.addStretch(1)
        self.refresh_training_runs()
        return self._scroll_page(page)

    def _load_deployment_settings(self) -> dict:
        try:
            value = json.loads(DEPLOYMENT_SETTINGS.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_deployment_settings(self) -> None:
        values = {
            "weights": self.deploy_weights_edit.text().strip(),
            "onnx": self.deploy_onnx_edit.text().strip(),
            "output": self.deploy_output_edit.text().strip(),
            "ultralytics_python": self.deploy_ultralytics_python_edit.text().strip(),
            "rknn_python": self.deploy_rknn_python_edit.text().strip(),
            "imgsz": self.deploy_imgsz_spin.value(),
            "opset": int(self.deploy_opset_combo.currentData()),
            "simplify": self.deploy_simplify_check.isChecked(),
            "precision": str(self.deploy_precision_combo.currentData()),
            "calibration_dir": self.deploy_calibration_edit.text().strip(),
            "calibration_limit": self.deploy_calibration_limit_spin.value(),
            "algorithm": str(self.deploy_algorithm_combo.currentData()),
            "method": str(self.deploy_method_combo.currentData()),
        }
        DEPLOYMENT_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        DEPLOYMENT_SETTINGS.write_text(
            json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    def _deployment_center_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        title = QLabel("RK3588 · 模型部署中心")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        intro = QLabel(
            "把训练完成的检测模型导出为 ONNX，再由 RKNN-Toolkit2 转换为面向 RK3588 的 .rknn。"
            "所有命令在各自 Conda 环境中以后台进程运行；页面记录参数、日志和产物。"
            "PT 自动导出当前走 Ultralytics；原版 YOLOv5 可先用对应仓库导出 ONNX，再使用‘仅转换 RKNN’。"
            "PC 构建通过只说明转换产物已生成，板端仍需用匹配的 Runtime 实测。"
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        flow = QHBoxLayout()
        for heading, caption in (
            ("01 选择权重", ".pt 与当前导出后端"),
            ("02 导出 ONNX", "固定尺寸、opset、简化"),
            ("03 RKNN 构建", "FP16 基线或 INT8 PTQ"),
            ("04 板端验证", "Runtime、精度与耗时"),
        ):
            card = QGroupBox(heading)
            card_layout = QVBoxLayout(card)
            label = QLabel(caption)
            label.setObjectName("hint")
            label.setWordWrap(True)
            card_layout.addWidget(label)
            flow.addWidget(card, 1)
        layout.addLayout(flow)

        saved = self._load_deployment_settings()
        config = QGroupBox("模型、环境与 ONNX 参数")
        form = QGridLayout(config)
        form.setHorizontalSpacing(9)
        form.setVerticalSpacing(7)
        self.deploy_weights_edit = QLineEdit(
            saved.get("weights", str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.is_file() else "")
        )
        self.deploy_onnx_edit = QLineEdit(saved.get("onnx", ""))
        self.deploy_output_edit = QLineEdit(saved.get("output", str(HERE / "runs" / "deployment")))
        self.deploy_ultralytics_python_edit = QLineEdit(saved.get(
            "ultralytics_python", sys.executable))
        self.deploy_rknn_python_edit = QLineEdit(saved.get(
            "rknn_python", os.environ.get("MSM_RKNN_PYTHON", "")))
        path_rows = [
            ("训练权重 .pt", self.deploy_weights_edit, "weights"),
            ("ONNX 文件（导出目标 / 转换输入）", self.deploy_onnx_edit, "onnx"),
            ("部署产物目录", self.deploy_output_edit, "directory"),
            ("Ultralytics 环境 Python", self.deploy_ultralytics_python_edit, "python"),
            ("RKNN-Toolkit2 环境 Python", self.deploy_rknn_python_edit, "python"),
        ]
        for row, (label, edit, kind) in enumerate(path_rows):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(edit, row, 1)
            button = QPushButton("浏览…")
            button.clicked.connect(lambda _checked=False, field=edit, field_kind=kind:
                                   self._browse_deployment_path(field, field_kind))
            form.addWidget(button, row, 2)

        option_row = len(path_rows)
        form.addWidget(QLabel("目标平台"), option_row, 0)
        self.deploy_target_label = QLabel("RK3588 · 静态输入模型")
        self.deploy_target_label.setObjectName("statusText")
        form.addWidget(self.deploy_target_label, option_row, 1)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("输入尺寸"))
        self.deploy_imgsz_spin = QSpinBox()
        self.deploy_imgsz_spin.setRange(32, 2048)
        self.deploy_imgsz_spin.setSingleStep(32)
        self.deploy_imgsz_spin.setValue(int(saved.get("imgsz", 640)))
        opts.addWidget(self.deploy_imgsz_spin)
        opts.addSpacing(12)
        opts.addWidget(QLabel("ONNX opset"))
        self.deploy_opset_combo = QComboBox()
        for opset in (12, 13, 17, 18, 19):
            self.deploy_opset_combo.addItem(str(opset), opset)
        opset_index = self.deploy_opset_combo.findData(int(saved.get("opset", 12)))
        self.deploy_opset_combo.setCurrentIndex(max(0, opset_index))
        opts.addWidget(self.deploy_opset_combo)
        self.deploy_simplify_check = QCheckBox("简化 ONNX 图")
        self.deploy_simplify_check.setChecked(bool(saved.get("simplify", True)))
        opts.addWidget(self.deploy_simplify_check)
        opts.addStretch(1)
        form.addLayout(opts, option_row + 1, 1, 1, 2)
        layout.addWidget(config)

        quant = QGroupBox("RKNN 精度与量化")
        quant_form = QGridLayout(quant)
        quant_form.addWidget(QLabel("转换精度"), 0, 0)
        self.deploy_precision_combo = QComboBox()
        self.deploy_precision_combo.addItem("浮点基线 · FP16", "fp16")
        self.deploy_precision_combo.addItem("INT8 · PTQ 校准量化", "int8_ptq")
        precision_index = self.deploy_precision_combo.findData(saved.get("precision", "int8_ptq"))
        self.deploy_precision_combo.setCurrentIndex(max(0, precision_index))
        quant_form.addWidget(self.deploy_precision_combo, 0, 1)
        quant_form.addWidget(QLabel("量化算法"), 0, 2)
        self.deploy_algorithm_combo = QComboBox()
        self.deploy_algorithm_combo.addItem("normal · 常规", "normal")
        self.deploy_algorithm_combo.addItem("mmse · 最小均方误差", "mmse")
        algorithm_index = self.deploy_algorithm_combo.findData(saved.get("algorithm", "normal"))
        self.deploy_algorithm_combo.setCurrentIndex(max(0, algorithm_index))
        quant_form.addWidget(self.deploy_algorithm_combo, 0, 3)
        quant_form.addWidget(QLabel("量化粒度"), 1, 0)
        self.deploy_method_combo = QComboBox()
        self.deploy_method_combo.addItem("channel · 按通道", "channel")
        self.deploy_method_combo.addItem("layer · 按层", "layer")
        method_index = self.deploy_method_combo.findData(saved.get("method", "channel"))
        self.deploy_method_combo.setCurrentIndex(max(0, method_index))
        quant_form.addWidget(self.deploy_method_combo, 1, 1)
        self.deploy_calibration_edit = QLineEdit(saved.get("calibration_dir", ""))
        self.deploy_calibration_edit.setPlaceholderText("选择来自实际部署场景的代表性图像目录")
        quant_form.addWidget(QLabel("校准图像目录"), 1, 2)
        quant_form.addWidget(self.deploy_calibration_edit, 1, 3)
        calibration_browse = QPushButton("选择目录…")
        calibration_browse.clicked.connect(
            lambda: self._browse_deployment_path(self.deploy_calibration_edit, "directory"))
        quant_form.addWidget(calibration_browse, 1, 4)
        self.deploy_calibration_count_label = QLabel("尚未统计图片")
        self.deploy_calibration_count_label.setObjectName("hint")
        quant_form.addWidget(self.deploy_calibration_count_label, 2, 2)
        quant_form.addWidget(QLabel("校准图片上限"), 2, 0)
        self.deploy_calibration_limit_spin = QSpinBox()
        self.deploy_calibration_limit_spin.setRange(8, 10000)
        self.deploy_calibration_limit_spin.setValue(int(saved.get("calibration_limit", 256)))
        self.deploy_calibration_limit_spin.setToolTip("按文件名排序后取前 N 张图片；8 张仅是预检下限，优先用具代表性的图片覆盖不同场景。")
        quant_form.addWidget(self.deploy_calibration_limit_spin, 2, 1)
        quant_note = QLabel(
            "INT8 PTQ 使用无标签校准图片估计量化范围；建议从目标部署视频/数据集中抽取有代表性的画面。"
            "FP16 用作精度基线，不需要校准集。QAT 权重导入暂留后续适配。"
        )
        quant_note.setObjectName("hint")
        quant_note.setWordWrap(True)
        quant_form.addWidget(quant_note, 3, 0, 1, 5)
        self.deploy_precision_combo.currentIndexChanged.connect(self._deployment_quant_controls_changed)
        self.deploy_calibration_edit.textChanged.connect(self._refresh_calibration_count)
        self.deploy_calibration_limit_spin.valueChanged.connect(self._refresh_calibration_count)
        self._deployment_quant_controls_changed()
        self._refresh_calibration_count()
        layout.addWidget(quant)

        status_row = QHBoxLayout()
        self.deployment_status = QLabel("选择权重和量化方式后执行预检。板端验证当前尚未连接。")
        self.deployment_status.setObjectName("hint")
        self.deployment_status.setWordWrap(True)
        status_row.addWidget(self.deployment_status, 1)
        self.deployment_progress = QProgressBar()
        self.deployment_progress.setRange(0, 2)
        self.deployment_progress.setValue(0)
        self.deployment_progress.setMinimumWidth(170)
        status_row.addWidget(self.deployment_progress)
        layout.addLayout(status_row)

        actions = QHBoxLayout()
        self.deployment_preflight_button = QPushButton("环境与参数预检")
        self.deployment_preflight_button.clicked.connect(lambda: self._begin_deployment("preflight"))
        self.deployment_export_button = QPushButton("仅导出 ONNX")
        self.deployment_export_button.clicked.connect(lambda: self._begin_deployment("export"))
        self.deployment_convert_button = QPushButton("仅转换 RKNN")
        self.deployment_convert_button.clicked.connect(lambda: self._begin_deployment("convert"))
        self.deployment_all_button = QPushButton("▶  一键 ONNX → RKNN")
        self.deployment_all_button.setObjectName("primaryButton")
        self.deployment_all_button.clicked.connect(lambda: self._begin_deployment("all"))
        self.deployment_stop_button = QPushButton("停止")
        self.deployment_stop_button.setEnabled(False)
        self.deployment_stop_button.clicked.connect(self._stop_deployment)
        self.deployment_open_button = QPushButton("打开产物目录")
        self.deployment_open_button.clicked.connect(self._open_deployment_output)
        for button in (self.deployment_preflight_button, self.deployment_export_button,
                       self.deployment_convert_button, self.deployment_all_button,
                       self.deployment_stop_button, self.deployment_open_button):
            actions.addWidget(button)
        layout.addLayout(actions)

        log_group = QGroupBox("转换日志")
        log_layout = QVBoxLayout(log_group)
        self.deployment_log = QPlainTextEdit()
        self.deployment_log.setReadOnly(True)
        self.deployment_log.setMaximumBlockCount(5000)
        self.deployment_log.setMinimumHeight(170)
        self.deployment_log.setPlaceholderText("预检、ONNX 导出、RKNN 构建及错误日志会显示在这里。")
        log_layout.addWidget(self.deployment_log)
        layout.addWidget(log_group)

        history_group = QGroupBox("部署记录 · SQLite")
        history_layout = QVBoxLayout(history_group)
        self.deployment_table = QTableWidget(0, 5)
        self.deployment_table.setHorizontalHeaderLabels(
            ["状态", "时间", "精度", "RKNN 产物", "输出目录"])
        self.deployment_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.deployment_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.deployment_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.deployment_table.verticalHeader().setVisible(False)
        self.deployment_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.deployment_table.setMinimumHeight(145)
        history_layout.addWidget(self.deployment_table)
        self.deployment_refresh_button = QPushButton("刷新记录")
        self.deployment_refresh_button.clicked.connect(self.refresh_deployment_runs)
        history_layout.addWidget(self.deployment_refresh_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(history_group)
        self.refresh_deployment_runs()
        layout.addStretch(1)
        return self._scroll_page(page)

    def _browse_deployment_path(self, field: QLineEdit, kind: str) -> None:
        current = field.text().strip() or str(Path.home())
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(self, "选择目录", current)
        elif kind == "python":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 Conda 环境 Python", current,
                "Python 可执行文件 (python python.exe);;所有文件 (*)")
        elif kind == "weights":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择训练权重", current, "PyTorch 权重 (*.pt);;所有文件 (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 ONNX 模型", current, "ONNX 模型 (*.onnx);;所有文件 (*)")
        if path:
            field.setText(path)

    def _deployment_quant_controls_changed(self, *_args) -> None:
        enabled = self.deploy_precision_combo.currentData() == "int8_ptq"
        for widget in (self.deploy_algorithm_combo, self.deploy_method_combo,
                       self.deploy_calibration_edit, self.deploy_calibration_limit_spin):
            widget.setEnabled(enabled)
        if hasattr(self, "deploy_calibration_count_label"):
            self.deploy_calibration_count_label.setEnabled(enabled)

    def _refresh_calibration_count(self, *_args) -> None:
        if not hasattr(self, "deploy_calibration_count_label"):
            return
        images = calibration_images(self.deploy_calibration_edit.text().strip(),
                                    self.deploy_calibration_limit_spin.value())
        self.deploy_calibration_count_label.setText(
            f"当前会使用 {len(images)} 张可识别图片（预检至少 8 张）")

    def _deployment_request(self) -> DeploymentRequest:
        weights = self.deploy_weights_edit.text().strip()
        onnx_path = self.deploy_onnx_edit.text().strip()
        if not onnx_path and weights:
            onnx_path = str(Path(self.deploy_output_edit.text().strip()).expanduser() /
                            f"{Path(weights).stem}_{self.deploy_imgsz_spin.value()}.onnx")
            self.deploy_onnx_edit.setText(onnx_path)
        return DeploymentRequest(
            weights=weights,
            output_dir=self.deploy_output_edit.text().strip(),
            ultralytics_python=self.deploy_ultralytics_python_edit.text().strip(),
            rknn_python=self.deploy_rknn_python_edit.text().strip(),
            onnx_file=onnx_path,
            imgsz=self.deploy_imgsz_spin.value(),
            opset=int(self.deploy_opset_combo.currentData()),
            simplify=self.deploy_simplify_check.isChecked(),
            precision=str(self.deploy_precision_combo.currentData()),
            calibration_dir=self.deploy_calibration_edit.text().strip(),
            calibration_limit=self.deploy_calibration_limit_spin.value(),
            quantized_algorithm=str(self.deploy_algorithm_combo.currentData()),
            quantized_method=str(self.deploy_method_combo.currentData()),
        )

    def _begin_deployment(self, action: str) -> None:
        if self.deployment_thread is not None or self.deployment_preflight_thread is not None:
            return
        request = self._deployment_request()
        export = action in {"export", "all", "preflight"}
        convert = action in {"convert", "all", "preflight"}
        self.deployment_pending_action = action
        self.deployment_request = request
        self._save_deployment_settings()
        self.deployment_status.setText("正在后台检查模型文件、环境依赖、校准图片和输出目录…")
        self.deployment_log.clear()
        self.deployment_progress.setValue(0)
        self._set_deployment_buttons(False)
        thread = DeploymentPreflightThread(request, export=export, convert=convert, parent=self)
        thread.completed.connect(self._on_deployment_preflight_completed)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._deployment_preflight_finished)
        self.deployment_preflight_thread = thread
        thread.start()

    def _deployment_preflight_finished(self) -> None:
        self.deployment_preflight_thread = None

    def _on_deployment_preflight_completed(self, errors: list[str]) -> None:
        action = getattr(self, "deployment_pending_action", "preflight")
        if errors:
            message = "预检未通过：\n• " + "\n• ".join(errors)
            self.deployment_status.setText(message)
            self.deployment_log.appendPlainText(message)
            self._set_deployment_buttons(True)
            return
        if action == "preflight":
            message = "预检通过：Ultralytics 导出环境、RKNN-Toolkit2 环境、模型输入和量化校准目录可用。"
            self.deployment_status.setText(message)
            self.deployment_log.appendPlainText(message)
            self._set_deployment_buttons(True)
            return
        request = self.deployment_request
        steps: list[tuple[str, list[str]]] = []
        if action in {"export", "all"}:
            if request.onnx_path.exists():
                answer = QMessageBox.question(
                    self, "覆盖 ONNX 文件？", f"目标文件已存在，将覆盖：\n{request.onnx_path}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    self.deployment_status.setText("已取消，未覆盖已有 ONNX 文件。")
                    self._set_deployment_buttons(True)
                    return
            steps.append(("PT → ONNX", export_command(request, DEPLOYMENT_WORKER)))
        if action in {"convert", "all"}:
            if request.rknn_path.exists():
                answer = QMessageBox.question(
                    self, "覆盖 RKNN 文件？", f"目标文件已存在，将覆盖：\n{request.rknn_path}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    self.deployment_status.setText("已取消，未覆盖已有 RKNN 文件。")
                    self._set_deployment_buttons(True)
                    return
            steps.append(("ONNX → RKNN", convert_command(request, DEPLOYMENT_WORKER)))
        self.deployment_run_id = uuid.uuid4().hex
        self.deployment_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.deployment_store.create_deployment_run(
            run_id=self.deployment_run_id,
            request={**request.snapshot(), "weights_sha256":
                     sha256_file(request.weights_path) if request.weights_path.is_file() else None},
            output_dir=request.output_path, onnx_path=request.onnx_path,
            rknn_path=request.rknn_path)
        self.deployment_progress.setRange(0, len(steps))
        self.deployment_progress.setValue(0)
        thread = DeploymentThread(steps, parent=self)
        thread.line.connect(self.deployment_log.appendPlainText)
        thread.stage.connect(self._on_deployment_stage)
        thread.completed.connect(self._on_deployment_completed)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._deployment_thread_finished)
        self.deployment_thread = thread
        self.deployment_stop_button.setEnabled(True)
        self.deployment_status.setText("部署任务运行中…")
        thread.start()

    def _set_deployment_buttons(self, enabled: bool) -> None:
        for button in (self.deployment_preflight_button, self.deployment_export_button,
                       self.deployment_convert_button, self.deployment_all_button):
            button.setEnabled(enabled)
        self.deployment_stop_button.setEnabled(False)

    def _on_deployment_stage(self, stage: str) -> None:
        self.deployment_status.setText(f"正在执行：{stage}")
        self.deployment_progress.setValue(min(
            self.deployment_progress.maximum(), self.deployment_progress.value() + 1))

    def _stop_deployment(self) -> None:
        if self.deployment_thread is not None:
            self.deployment_status.setText("正在请求停止当前转换步骤…")
            self.deployment_stop_button.setEnabled(False)
            self.deployment_thread.cancel()

    def _on_deployment_completed(self, success: bool, message: str, log_tail: str) -> None:
        run_id = self.deployment_run_id
        status = "completed" if success else ("stopped" if "取消" in message else "failed")
        if run_id:
            try:
                self.deployment_store.update_deployment_run(
                    run_id, status=status, log_tail=log_tail,
                    error=None if success else message)
            except Exception as exc:
                self.deployment_log.appendPlainText(f"写入部署记录失败：{exc}")
        self.deployment_log.appendPlainText(message)
        self.deployment_status.setText(message)
        if success:
            self.deployment_progress.setValue(self.deployment_progress.maximum())
        self._set_deployment_buttons(True)
        self.refresh_deployment_runs()
        if success and self.deployment_request.rknn_path.is_file():
            self.deployment_log.appendPlainText(
                f"RKNN 文件已生成：{self.deployment_request.rknn_path}\n"
                "尚未执行 RK3588 板端 Runtime 验证。")
        self.deployment_run_id = None

    def _deployment_thread_finished(self) -> None:
        self.deployment_thread = None

    def refresh_deployment_runs(self) -> None:
        if not hasattr(self, "deployment_table"):
            return
        try:
            records = self.deployment_store.recent_deployment_runs(100)
        except Exception as exc:
            self.deployment_status.setText(f"读取部署记录失败：{exc}")
            return
        self.deployment_table.setRowCount(len(records))
        for row, record in enumerate(records):
            try:
                request = json.loads(record["request_json"])
            except (TypeError, json.JSONDecodeError):
                request = {}
            precision = "INT8 PTQ" if request.get("precision") == "int8_ptq" else "FP16"
            values = [record["status"], record["started_at"][:19], precision,
                      record["rknn_path"], record["output_dir"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.deployment_table.setItem(row, column, item)

    def _open_deployment_output(self) -> None:
        row = self.deployment_table.currentRow() if hasattr(self, "deployment_table") else -1
        if row >= 0:
            record = self.deployment_store.recent_deployment_runs(100)[row]
            path = Path(record["output_dir"])
        elif hasattr(self, "deployment_request"):
            path = self.deployment_request.output_path
        else:
            path = Path(self.deploy_output_edit.text().strip()).expanduser()
        if not path.exists():
            QMessageBox.information(self, "产物目录不存在", str(path))
            return
        os.startfile(path) if hasattr(os, "startfile") else subprocess.Popen(["xdg-open", str(path)])

    @staticmethod
    def _make_training_parameter_control(key: str, kind: str, default,
                                         value, extra: list):
        if kind == "bool":
            control = QCheckBox()
            control.setChecked(bool(value))
            return control
        if kind == "int":
            minimum, maximum = extra
            control = QSpinBox()
            control.setRange(int(minimum), int(maximum))
            control.setValue(int(default if value is None else value))
            control.setKeyboardTracking(False)
            return control
        if kind == "float":
            minimum, maximum = extra
            control = QDoubleSpinBox()
            control.setRange(float(minimum), float(maximum))
            control.setDecimals(6)
            control.setSingleStep(0.01)
            control.setValue(float(default if value is None else value))
            control.setKeyboardTracking(False)
            return control
        if kind == "optional_float":
            control = QLineEdit("" if value is None else str(value))
            control.setPlaceholderText("留空使用默认值")
            return control
        if kind == "choice":
            control = QComboBox()
            choices = extra[0]
            for label, item_value in choices:
                control.addItem(label, item_value)
            index = control.findData(value)
            if index < 0:
                index = control.findData(default)
            control.setCurrentIndex(max(0, index))
            return control
        control = QLineEdit("" if value is None else str(value))
        control.setPlaceholderText("留空使用默认值")
        return control

    def _read_training_parameter_controls(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, control in self.training_param_controls.items():
            if isinstance(control, QCheckBox):
                value = control.isChecked()
            elif isinstance(control, QSpinBox):
                value = control.value()
            elif isinstance(control, QDoubleSpinBox):
                value = control.value()
            elif isinstance(control, QComboBox):
                value = control.currentData()
            elif isinstance(control, QLineEdit):
                text = control.text().strip()
                if not text:
                    value = None
                elif key == "freeze":
                    pieces = [part.strip() for part in text.split(",") if part.strip()]
                    try:
                        value = int(pieces[0]) if len(pieces) == 1 else [int(part) for part in pieces]
                    except ValueError:
                        value = text
                elif key == "distill_model":
                    value = text
                elif key == "conf":
                    try:
                        value = float(text)
                    except ValueError:
                        value = text
                else:
                    value = text
            else:
                continue
            if key == "time" and value == 0:
                value = None
            values[key] = value
        return values

    def _load_training_settings(self) -> dict:
        try:
            value = json.loads(TRAINING_SETTINGS.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_training_settings(self) -> None:
        values = {key: field.text().strip()
                  for key, field in self.training_edits.items()}
        values.update({key: spin.value() for key, spin in self.training_spins.items()})
        values["device"] = str(self.training_device_combo.currentData())
        values["optimizer"] = str(self.training_optimizer_combo.currentData())
        values.update(self._read_training_parameter_controls())
        TRAINING_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        TRAINING_SETTINGS.write_text(
            json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    def _browse_training_path(self, field: QLineEdit, kind: str) -> None:
        current = field.text().strip()
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(
                self, "选择目录", current or str(Path.home()))
        elif kind == "python":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 Conda 环境 Python", current or str(Path.home()),
                "Python 可执行文件 (python python.exe);;所有文件 (*)")
        elif kind == "model":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择预训练权重或模型 YAML", current or str(Path.home()),
                "模型来源 (*.pt *.yaml *.yml);;PyTorch 权重 (*.pt);;模型 YAML (*.yaml *.yml)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择数据集/超参数 YAML", current or str(Path.home()),
                "YAML 文件 (*.yaml *.yml);;所有文件 (*)")
        if path:
            field.setText(path)

    def _make_training_request(self, *, run_id: str | None = None,
                               resume_checkpoint: str = "") -> TrainingRequest:
        values = {key: field.text().strip()
                  for key, field in self.training_edits.items()}
        name = values["name"].strip()
        if not name:
            name = "ultralytics_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            self.training_edits["name"].setText(name)
        name = name.replace("/", "_").replace("\\", "_").strip(" .") or "ultralytics_run"
        options = {
            "name": name,
            "epochs": self.training_spins["epochs"].value(),
            "batch": self.training_spins["batch"].value(),
            "imgsz": self.training_spins["imgsz"].value(),
            "workers": self.training_spins["workers"].value(),
            "patience": self.training_spins["patience"].value(),
            "device": str(self.training_device_combo.currentData()),
            "optimizer": str(self.training_optimizer_combo.currentData()),
        }
        options.update(self._read_training_parameter_controls())
        if values.get("cfg"):
            options["cfg"] = values["cfg"]
        else:
            options.pop("cfg", None)
        return TrainingRequest(
            backend_id=UltralyticsTrainingBackend.backend_id,
            model=values["model"], dataset=values["dataset"], options=options,
            python_executable=values["python"],
            output_dir=values["output"], run_id=run_id or uuid.uuid4().hex,
            resume_checkpoint=resume_checkpoint,
        )

    def start_training(self) -> None:
        if self.training_validation_thread is not None:
            return
        if self.training_run_id:
            QMessageBox.information(self, "训练任务运行中", "请等待当前训练完成或先停止当前任务。")
            return
        request = self._make_training_request()
        self._begin_training_preflight(request)

    def _begin_training_preflight(self, request: TrainingRequest) -> None:
        self._save_training_settings()
        self.training_request = request
        self.training_start_button.setEnabled(False)
        self.training_stop_button.setEnabled(False)
        self.training_resume_button.setEnabled(False)
        self.training_progress.setValue(0)
        self.training_metrics_label.setText("正在检查 Ultralytics 环境、参数、数据集和设备……")
        self.training_preflight_label.setText("后台执行 Ultralytics 预检；不会阻塞界面或自动安装依赖。")
        if request.resume_checkpoint:
            prior = self.training_store.training_run(request.run_id)
            self._training_log_prefix = prior.get("log_tail", "") if prior else ""
            self.training_log.setPlainText(self._training_log_prefix)
        else:
            self._training_log_prefix = ""
            self.training_log.clear()
        thread = QThread(self)
        worker = TrainingPreflightWorker(self.training_backend, request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_training_preflight_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.training_validation_thread = thread
        self.training_validation_worker = worker
        thread.finished.connect(self._on_training_validation_thread_finished)
        thread.start()

    @Slot(object, object, object, object)
    def _on_training_preflight_finished(self, request: TrainingRequest,
                                        errors: list[str], run_id: str | None,
                                        runtime: dict) -> None:
        if errors or not run_id:
            message = "\n".join(errors or ["训练预检未能启动任务。"])
            self.training_preflight_label.setText(message)
            self.training_metrics_label.setText("预检失败")
            self.training_log.appendPlainText(message)
            self.training_start_button.setEnabled(True)
            self.training_refresh_button.setEnabled(True)
            return
        try:
            request_data = request.to_dict()
            output_dir = Path(request.output_dir).expanduser().resolve() / str(request.options["name"])
            prior_record = (self.training_store.training_run(run_id)
                            if request.resume_checkpoint else None)
            dataset_hash = (sha256_file(request.dataset) if Path(request.dataset).is_file()
                            else (prior_record.get("dataset_sha256") if prior_record else None))
            model_hash = (sha256_file(request.model) if Path(request.model).is_file()
                          else (prior_record.get("model_source_sha256") if prior_record else None))
            self.training_store.create_training_run(
                run_id=run_id, request=request_data,
                dataset_sha256=dataset_hash,
                model_source_sha256=model_hash,
                output_dir=output_dir, environment={
                    **runtime,
                    "python_executable": str(Path(request.python_executable).resolve()),
                    "device": request.options.get("device", "0"),
                })
        except Exception as exc:
            self.training_backend.cancel(run_id)
            message = f"训练已启动，但无法登记 SQLite 记录：{exc}"
            self.training_preflight_label.setText(message)
            self.training_log.appendPlainText(message)
            self.training_start_button.setEnabled(True)
            self.training_stop_button.setEnabled(False)
            return
        self.training_run_id = run_id
        self.training_request = request
        device = (runtime.get("gpu") if str(request.options.get("device", "0")).lower() != "cpu"
                  else "CPU") or str(request.options.get("device", "cpu"))
        self.training_preflight_label.setText(
            f"训练运行中 · Ultralytics {runtime.get('ultralytics', '?')} · Python {runtime.get('python', '?')} · "
            f"PyTorch {runtime.get('torch', '?')} · {device}")
        self.training_metrics_label.setText("等待第一个 epoch 指标写入 results.csv……")
        self.training_stop_button.setEnabled(True)
        self.training_timer.start()
        self.refresh_training_runs()

    @Slot()
    def _on_training_validation_thread_finished(self) -> None:
        thread = self.training_validation_thread
        if thread is None:
            return
        self.training_validation_thread = None
        self.training_validation_worker = None
        if not self.training_run_id:
            self.training_start_button.setEnabled(True)
            self.training_refresh_button.setEnabled(True)

    def stop_training(self) -> None:
        if not self.training_run_id:
            return
        self.training_backend.cancel(self.training_run_id)
        self.training_stop_button.setEnabled(False)
        self.training_preflight_label.setText(
            "已发送停止信号；恢复时将从训练器最近保存的 last.pt 检查点继续。")

    def _poll_training(self) -> None:
        if not self.training_run_id:
            self.training_timer.stop()
            return
        event = self.training_backend.status(self.training_run_id)
        record = self.training_store.training_run(self.training_run_id)
        output_dir = Path(record["output_dir"]) if record else Path()
        active_log = self.training_backend.log_tail(self.training_run_id)
        log_tail = "\n".join(part for part in (self._training_log_prefix, active_log) if part)
        error = None
        if event.state == TrainingState.FAILED:
            error = (log_tail.splitlines()[-1] if log_tail else event.message)
        model_id = None
        best = output_dir / "weights" / "best.pt"
        if event.state == TrainingState.COMPLETED and best.is_file() and record:
            try:
                project_id = self.training_store.ensure_project(self.project_edit.text())
                framework = json.loads(record.get("environment_json") or "{}")
                model_id = self.training_store.register_model(
                    project_id, best, sha256_file(best), "ultralytics_native",
                    str(framework.get("torch", "unknown")), task="detect")
            except Exception as exc:
                self.training_log.appendPlainText(f"登记 best.pt 模型信息失败：{exc}")
        self.training_store.update_training_run(
            self.training_run_id, status=event.state.value, epoch=event.epoch,
            epochs=event.epochs, progress=event.progress, metrics=dict(event.metrics),
            log_tail=log_tail, artifacts=list(event.artifacts),
            return_code=event.return_code, error=error, model_id=model_id)
        if event.log_lines:
            self.training_log.appendPlainText("\n".join(event.log_lines))
        self.training_progress.setValue(round(event.progress * 100))
        if event.epoch is not None:
            self.training_metrics_label.setText(self._format_training_metrics(event))
        if event.state in {TrainingState.COMPLETED, TrainingState.STOPPED, TrainingState.FAILED}:
            self.training_timer.stop()
            self.training_run_id = None
            self.training_start_button.setEnabled(True)
            self.training_stop_button.setEnabled(False)
            self.training_preflight_label.setText(event.message)
            self.refresh_training_runs()
        else:
            self.training_preflight_label.setText(
                f"{event.message} · epoch {event.epoch or 0}/{event.epochs or '?'}")
            self.refresh_training_runs()

    @staticmethod
    def _format_training_metrics(event) -> str:
        metrics = event.metrics
        values = [f"Epoch {event.epoch or 0}/{event.epochs or '?'}"]
        for key, label in (("metrics/precision", "P"), ("metrics/precision(B)", "P"),
                            ("metrics/recall", "R"), ("metrics/recall(B)", "R"),
                            ("metrics/mAP_0.5", "mAP50"),
                            ("metrics/mAP_0.5:0.95", "mAP50-95"),
                            ("metrics/mAP50(B)", "mAP50"),
                            ("metrics/mAP50-95(B)", "mAP50-95"),
                            ("train/box_loss", "box loss"),
                            ("train/obj_loss", "obj loss"),
                            ("train/cls_loss", "cls loss")):
            if key in metrics:
                values.append(f"{label} {metrics[key]:.4f}")
        return "　·　".join(values)

    def refresh_training_runs(self) -> None:
        if not hasattr(self, "training_table"):
            return
        try:
            records = self.training_store.recent_training_runs(100)
        except Exception as exc:
            self.training_preflight_label.setText(f"读取训练记录失败：{exc}")
            return
        self.training_table.setRowCount(len(records))
        for row, record in enumerate(records):
            metrics = json.loads(record.get("metrics_json") or "{}")
            dataset = Path(record.get("dataset_yaml") or "—").name
            best = record.get("best_checkpoint") or ""
            values = [record.get("status", "—"),
                      f"{record.get('epoch') or 0}/{record.get('epochs') or '?'}",
                      json.loads(record.get("request_json") or "{}").get("options", {}).get("name", record["id"][:8]),
                      dataset,
                      (f"{metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP_0.5:0.95')):.4f}"
                       if metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP_0.5:0.95')) is not None
                       else "—"),
                      Path(best).name if best else "—",
                      record.get("output_dir") or "—"]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 6:
                    item.setToolTip(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record["id"])
                if column == 5:
                    item.setData(Qt.ItemDataRole.UserRole, best)
                self.training_table.setItem(row, column, item)

    def _selected_training_record(self) -> dict | None:
        row = self.training_table.currentRow() if hasattr(self, "training_table") else -1
        if row < 0:
            return None
        run_id = self.training_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return self.training_store.training_run(run_id) if run_id else None

    def resume_selected_training(self) -> None:
        if self.training_run_id or self.training_validation_thread is not None:
            QMessageBox.information(self, "训练任务运行中", "请等待当前任务停止或完成后再恢复。")
            return
        record = self._selected_training_record()
        if not record:
            QMessageBox.information(self, "未选择任务", "请先在训练记录中选择一个任务。")
            return
        if record["status"] not in {"stopped", "failed"}:
            QMessageBox.information(self, "无法恢复", "只支持恢复已停止或异常退出的训练任务。")
            return
        checkpoint = record.get("last_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            QMessageBox.warning(self, "检查点不存在", "所选任务没有可用的 weights/last.pt。")
            return
        try:
            old = TrainingRequest.from_dict(json.loads(record["request_json"]))
            if old.backend_id != UltralyticsTrainingBackend.backend_id:
                QMessageBox.information(
                    self, "训练后端不同",
                    "该记录由 Spinosaurus YOLOv5 旧后端创建，不能使用 Ultralytics 恢复入口。")
                return
            output = Path(record["output_dir"])
            request = TrainingRequest(
                backend_id=UltralyticsTrainingBackend.backend_id,
                model=old.model, dataset=old.dataset, options={**old.options, "name": output.name},
                python_executable=self.training_edits["python"].text().strip() or old.python_executable,
                output_dir=str(output.parent), run_id=record["id"],
                resume_checkpoint=checkpoint,
            )
            self.training_edits["name"].setText(output.name)
            self.training_edits["dataset"].setText(old.dataset)
            self.training_edits["model"].setText(old.model)
            self.training_spins["epochs"].setValue(int(old.options.get("epochs", 100)))
            self._begin_training_preflight(request)
        except Exception as exc:
            QMessageBox.warning(self, "恢复配置失败", str(exc))

    def open_selected_training_output(self) -> None:
        record = self._selected_training_record()
        if not record:
            return
        path = Path(record["output_dir"])
        if not path.exists():
            QMessageBox.warning(self, "目录不存在", str(path))
            return
        os.startfile(path) if hasattr(os, "startfile") else __import__("subprocess").Popen(["xdg-open", str(path)])

    def use_selected_training_weight(self) -> None:
        record = self._selected_training_record()
        if not record:
            return
        best = record.get("best_checkpoint")
        if not best or not Path(best).is_file():
            QMessageBox.information(self, "权重暂不可用", "训练完成后才会生成 best.pt。")
            return
        self.weights_edit.setText(best)
        ultralytics_index = self.detector_backend_combo.findData("ultralytics")
        if ultralytics_index >= 0:
            self.detector_backend_combo.setCurrentIndex(ultralytics_index)
        self.workspace_tabs.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0)
        self.status_text.setText(f"已载入训练权重：{best}")

    def _yolo_single_image_lab(self) -> QWidget:
        group = QGroupBox("单图阈值实验：一次推理，多次后处理预览")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        intro = QLabel(
            "先运行一次模型前向，再拖动阈值查看筛选结果变化；调节时复用已缓存的原始预测，不重复推理。"
            "置信度和 IoU 也同步到视频分析，并保存在参数与实验记录中。"
        )
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        example_image = HERE / "validation" / "model_only_02s.jpg"
        self.yolo_image_edit = QLineEdit(str(example_image) if example_image.is_file() else "")
        self.yolo_image_edit.setPlaceholderText("选择一张 JPG / PNG 图片")
        row.addWidget(self.yolo_image_edit, 1)
        browse = QPushButton("选择图片…")
        browse.clicked.connect(self.browse_yolo_image)
        row.addWidget(browse)
        run = QPushButton("▶  分析单图")
        run.setObjectName("primaryButton")
        run.clicked.connect(self.simulate_yolo_steps)
        self.yolo_trace_run_button = run
        row.addWidget(run)
        layout.addLayout(row)

        self.yolo_trace_status = QLabel(
            "选择示例图片后，使用“检测追踪”页设置的权重和推理设备。"
        )
        self.yolo_trace_status.setObjectName("hint")
        self.yolo_trace_status.setWordWrap(True)
        layout.addWidget(self.yolo_trace_status)

        threshold_card = QGroupBox("实时后处理参数")
        threshold_layout = QGridLayout(threshold_card)
        self.yolo_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.yolo_confidence_slider.setRange(5, 95)
        self.yolo_confidence_value = QDoubleSpinBox()
        self.yolo_confidence_value.setRange(0.05, 0.95)
        self.yolo_confidence_value.setSingleStep(0.01)
        self.yolo_confidence_value.setDecimals(2)
        self.yolo_confidence_value.setValue(self.controls["confidence"].value())
        self.yolo_iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.yolo_iou_slider.setRange(5, 95)
        self.yolo_iou_value = QDoubleSpinBox()
        self.yolo_iou_value.setRange(0.05, 0.95)
        self.yolo_iou_value.setSingleStep(0.01)
        self.yolo_iou_value.setDecimals(2)
        self.yolo_iou_value.setValue(self.controls["iou_threshold"].value())
        threshold_layout.addWidget(QLabel("置信度阈值"), 0, 0)
        threshold_layout.addWidget(self.yolo_confidence_slider, 0, 1)
        threshold_layout.addWidget(self.yolo_confidence_value, 0, 2)
        threshold_layout.addWidget(QLabel("NMS IoU 阈值"), 1, 0)
        threshold_layout.addWidget(self.yolo_iou_slider, 1, 1)
        threshold_layout.addWidget(self.yolo_iou_value, 1, 2)
        self.yolo_iou_note = QLabel("IoU 越低，重叠框抑制越强；端到端免 NMS 路径不使用此参数。")
        self.yolo_iou_note.setObjectName("hint")
        self.yolo_iou_note.setWordWrap(True)
        threshold_layout.addWidget(self.yolo_iou_note, 2, 1, 1, 2)
        layout.addWidget(threshold_card)

        replay_group = QGroupBox("阈值影响 · 分数分布 · 阶段账本")
        replay_layout = QGridLayout(replay_group)
        self.yolo_confidence_histogram = QLabel("运行单图后显示候选框置信度分布")
        self.yolo_confidence_histogram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.yolo_confidence_histogram.setMinimumHeight(150)
        self.yolo_confidence_histogram.setScaledContents(True)
        self.yolo_confidence_histogram.setStyleSheet(
            "background:#F7FAFD; border:1px solid #DCE5EF; border-radius:8px;")
        replay_layout.addWidget(self.yolo_confidence_histogram, 0, 0, 1, 2)
        self.yolo_stage_table = QTableWidget(4, 4)
        self.yolo_stage_table.setHorizontalHeaderLabels(
            ["阶段", "执行操作", "框数量", "坐标空间"])
        self.yolo_stage_table.verticalHeader().setVisible(False)
        self.yolo_stage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.yolo_stage_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.yolo_stage_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.yolo_stage_table.setMinimumHeight(166)
        replay_layout.addWidget(self.yolo_stage_table, 0, 2, 1, 2)
        self.yolo_model_note = QLabel(
            "模型输出说明：YOLO 不同代际的检测头和分数定义可能不同；请结合当前后端、模型结构与端到端状态阅读。"
        )
        self.yolo_model_note.setObjectName("hint")
        self.yolo_model_note.setWordWrap(True)
        replay_layout.addWidget(self.yolo_model_note, 1, 0, 1, 4)
        layout.addWidget(replay_group)

        stages = [
            ("1 · 原图与尺寸", "输入图像、原始高宽和像素坐标。"),
            ("2 · 等比例缩放", "r=min(S/H,S/W)，保持长宽比缩放，不直接拉伸目标。"),
            ("3 · Letterbox", "在短边填充像素，使输入符合网络尺寸；记录比例与 padding 供逆映射。"),
            ("4 · 颜色通道", "OpenCV 图像 BGR 转为模型需要的 RGB。"),
            ("5 · 张量归一化", "HWC→CHW，转浮点并缩放到 [0,1]；显示真实输入张量尺寸和值域。"),
            ("6 · 低阈值候选", "用低 conf 查看解码后的候选框；框坐标仍处于模型输入空间。"),
            ("7 · 置信度筛选", "保留 score≥conf 的候选；阈值越高通常误检减少、漏检风险增大。"),
            ("8 · NMS / 端到端", "常规路径按类别执行 NMS；支持 one-to-one 的模型可绕过 IoU-NMS。"),
            ("9 · 原图坐标映射", "撤销 letterbox padding 与缩放比例，并裁剪到原图边界。"),
        ]
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.yolo_trace_panels = []
        for index, (title, description) in enumerate(stages):
            panel = ImagePanel(title, with_description=True)
            panel.image.setMinimumSize(220, 135)
            panel.description.setText(description)
            grid.addWidget(panel, index // 3, index % 3)
            self.yolo_trace_panels.append(panel)
        layout.addLayout(grid)
        return group

    def _start_yolo_trace_worker(self) -> None:
        self._yolo_request_id = 0
        self.yolo_trace_thread = QThread(self)
        self.yolo_trace_worker = YoloTraceWorker()
        self.yolo_trace_worker.moveToThread(self.yolo_trace_thread)
        self.yolo_inference_requested.connect(self.yolo_trace_worker.analyze)
        self.yolo_thresholds_requested.connect(self.yolo_trace_worker.update_thresholds)
        self.yolo_invalidate_requested.connect(self.yolo_trace_worker.invalidate)
        self.yolo_shutdown_requested.connect(self.yolo_trace_worker.shutdown)
        self.yolo_trace_worker.ready.connect(self._display_yolo_trace)
        self.yolo_trace_worker.failed.connect(self._yolo_trace_failed)
        self.yolo_trace_worker.status.connect(self._yolo_trace_status)
        self.yolo_trace_worker.shutdown_finished.connect(
            self.yolo_trace_thread.quit, Qt.ConnectionType.DirectConnection)
        self.yolo_trace_thread.finished.connect(self.yolo_trace_worker.deleteLater)
        self.yolo_trace_thread.start()

        self.yolo_threshold_timer = QTimer(self)
        self.yolo_threshold_timer.setSingleShot(True)
        self.yolo_threshold_timer.setInterval(90)
        self.yolo_threshold_timer.timeout.connect(self._queue_yolo_threshold_update)
        self.yolo_confidence_slider.valueChanged.connect(
            lambda value: self.yolo_confidence_value.setValue(value / 100.0))
        self.yolo_confidence_value.valueChanged.connect(
            lambda value: self.yolo_confidence_slider.setValue(round(value * 100)))
        self.yolo_iou_slider.valueChanged.connect(
            lambda value: self.yolo_iou_value.setValue(value / 100.0))
        self.yolo_iou_value.valueChanged.connect(
            lambda value: self.yolo_iou_slider.setValue(round(value * 100)))
        self.yolo_confidence_value.valueChanged.connect(
            lambda value: self.controls["confidence"].setValue(value))
        self.controls["confidence"].valueChanged.connect(
            lambda value: self.yolo_confidence_value.setValue(value))
        self.yolo_iou_value.valueChanged.connect(
            lambda value: self.controls["iou_threshold"].setValue(value))
        self.controls["iou_threshold"].valueChanged.connect(
            lambda value: self.yolo_iou_value.setValue(value))
        self.controls["confidence"].valueChanged.connect(self._schedule_yolo_threshold_update)
        self.controls["iou_threshold"].valueChanged.connect(self._schedule_yolo_threshold_update)

        self.yolo_image_edit.textChanged.connect(self._invalidate_yolo_trace)
        self.weights_edit.textChanged.connect(self._invalidate_yolo_trace)
        self.detector_backend_combo.currentIndexChanged.connect(self._invalidate_yolo_trace)
        self.device_combo.currentIndexChanged.connect(self._invalidate_yolo_trace)
        self.controls["imgsz"].valueChanged.connect(self._invalidate_yolo_trace)

    def _schedule_yolo_threshold_update(self, *_args) -> None:
        if hasattr(self, "yolo_threshold_timer"):
            self.yolo_threshold_timer.start()

    def _queue_yolo_threshold_update(self) -> None:
        self.yolo_thresholds_requested.emit(
            float(self.controls["confidence"].value()),
            float(self.controls["iou_threshold"].value()),
        )

    def _invalidate_yolo_trace(self, *_args) -> None:
        if not hasattr(self, "yolo_trace_thread"):
            return
        self._yolo_request_id += 1
        self.yolo_trace_run_button.setEnabled(True)
        for panel in self.yolo_trace_panels:
            panel.image.clear()
            panel.image.setText("参数已变化，请重新分析")
        self.yolo_invalidate_requested.emit(self._yolo_request_id)

    def _yolo_trace_status(self, request_id: int, message: str) -> None:
        if request_id == self._yolo_request_id:
            self.yolo_trace_status.setText(message)

    def _yolo_trace_failed(self, request_id: int, message: str) -> None:
        if request_id != self._yolo_request_id:
            return
        self.yolo_trace_run_button.setEnabled(True)
        self.yolo_trace_status.setText(f"单图分析失败：{message}")
        QMessageBox.critical(self, "YOLO 流程分析失败", message)

    def _display_yolo_trace(self, payload: dict) -> None:
        if payload.get("request_id", self._yolo_request_id) != self._yolo_request_id:
            return
        image = payload["image"]
        trace = payload["trace"]
        names = payload["names"]
        rows = {
            "candidates": trace.candidates,
            "confidence_filtered": trace.confidence_filtered,
            "nms": trace.nms,
            "final": trace.final,
        }
        padded = payload["resized_image"]
        height, width = image.shape[:2]
        padded_h, padded_w = padded.shape[:2]
        scale = min(padded_w / width, padded_h / height)
        resized_only = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        images = [
            image,
            resized_only,
            padded,
            self._show_rgb_planes(padded),
            payload["normalized_image"],
            self._draw_yolo_trace_boxes(padded, rows["candidates"], names, "MODEL CANDIDATES"),
            self._draw_yolo_trace_boxes(padded, rows["confidence_filtered"], names, "CONFIDENCE FILTER"),
            self._draw_yolo_trace_boxes(
                padded, rows["nms"], names,
                "END-TO-END OUTPUT" if trace.end2end else f"NMS  IoU={trace.iou_threshold:.2f}",
            ),
            self._draw_yolo_trace_boxes(image, rows["final"], names, "FINAL DETECTIONS"),
        ]
        stage_explanations = [
            "原图参考：保留输入宽高与原始像素坐标；最终框要映射回这里。",
            "等比例 resize：缩放率 r=min(Wₙ/W,Hₙ/H)，避免目标形状被拉伸。",
            "Letterbox：把缩放后的图像填充到模型尺寸；记录左右/上下 padding。",
            "颜色排列：将 OpenCV 的 BGR 通道调整为 RGB，避免颜色语义错位。",
            "数值表示：批量维度 + CHW 布局；浮点归一化到 [0,1]，实际 shape/range 见上方。",
            "候选框：以低 conf 与接近关闭的 IoU 抑制观察网络输出；不是未经解码的 logits，也不是最终框。",
            f"置信度：仅保留 score ≥ {trace.confidence_threshold:.2f}。阈值高会减少低分框，也可能漏掉微小/模糊 UAV。",
            ("端到端 one-to-one 输出：模型已学习去重，当前 NMS IoU 不生效。"
             if trace.end2end else
             f"NMS：按类别保留高分框，并抑制 IoU > {trace.iou_threshold:.2f} 的重叠框。"),
            "坐标逆变换：从模型输入坐标减去 padding、除以缩放率，最后裁剪到原图边界。",
        ]
        for panel, stage_image, description in zip(self.yolo_trace_panels, images, stage_explanations):
            panel.show_image(preview_image(stage_image, size=(600, 320)))
            panel.description.setText(description)
        scores = (trace.candidates[:, 4].astype(np.float32)
                  if len(trace.candidates) else np.empty(0, dtype=np.float32))
        histogram = self._render_confidence_histogram(scores, trace.confidence_threshold)
        self.yolo_confidence_histogram.setPixmap(
            QPixmap.fromImage(cv_image_to_qimage(histogram)))
        stage_rows = [
            ("低阈值候选", "conf≈0.001 / IoU≈1.0", len(trace.candidates), "模型输入像素"),
            ("置信度筛选", f"score ≥ {trace.confidence_threshold:.2f}",
             len(trace.confidence_filtered), "模型输入像素"),
            ("NMS / one-to-one", "端到端旁路" if trace.end2end else f"IoU={trace.iou_threshold:.2f}",
             len(trace.nms), "模型输入像素"),
            ("原图映射", "反填充 / 反缩放 / 边界裁剪", len(trace.final), "原图像素"),
        ]
        for row, values in enumerate(stage_rows):
            for column, value in enumerate(values):
                self.yolo_stage_table.setItem(row, column, QTableWidgetItem(str(value)))
        if trace.end2end:
            model_note = (
                "当前模型报告 end-to-end 输出：本次检测使用 one-to-one 路径，NMS IoU 不参与候选去重。"
                "若要比较不同阈值，置信度仍可调；请同时确认模型/推理后端实际启用的 head。"
            )
        elif payload["backend"] == "legacy_yolov5":
            model_note = (
                "当前使用原版 YOLOv5：anchor-based 多尺度检测头输出 objectness 与类别分数；"
                "候选置信度按 objectness×class score 组合，随后由 NMS 去重。"
            )
        else:
            model_note = (
                "当前使用 Ultralytics 检测后端：框经过模型解码后以 boxes.conf 展示。"
                "YOLOv8/11 等 anchor-free 头与原版 YOLOv5 的候选生成方式不同，"
                "因此跨架构比较时要固定数据、输入尺寸和阈值口径。"
            )
        self.yolo_model_note.setText(model_note)
        self.yolo_iou_slider.setEnabled(not trace.end2end)
        self.yolo_iou_value.setEnabled(not trace.end2end)
        self.yolo_iou_note.setText(
            "当前推理路径为端到端输出，NMS IoU 不参与筛选。"
            if trace.end2end else
            "IoU 越低，重叠框抑制越强；当前阈值同步用于视频分析。"
        )
        self.yolo_trace_run_button.setEnabled(True)
        self.yolo_trace_status.setText(
            f"图片：{Path(payload['image_path']).name}（{width}×{height}）　"
            f"后端：{payload['backend']}　设备：{payload['device']}　"
            f"张量：{payload['input_shape']}　范围："
            f"[{payload['tensor_range'][0]:.3f}, {payload['tensor_range'][1]:.3f}]　"
            f"置信度 {trace.confidence_threshold:.2f}　"
            f"{'端到端输出' if trace.end2end else f'IoU {trace.iou_threshold:.2f}'}　"
            f"候选 {len(trace.candidates)} → 置信度后 {len(trace.confidence_filtered)} "
            f"→ 最终 {len(trace.final)}"
        )

    @staticmethod
    def _render_confidence_histogram(scores: np.ndarray, threshold: float,
                                     width: int = 900, height: int = 170) -> np.ndarray:
        """Draw a compact confidence histogram with the active cutoff."""
        canvas = np.full((height, width, 3), (250, 252, 254), np.uint8)
        left, right, top, bottom = 48, width - 18, 28, height - 34
        cv2.putText(canvas, "Confidence distribution", (left, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (42, 62, 83), 1, cv2.LINE_AA)
        cv2.line(canvas, (left, bottom), (right, bottom), (135, 153, 171), 1, cv2.LINE_AA)
        cv2.line(canvas, (left, top), (left, bottom), (135, 153, 171), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0", (left - 4, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (80, 96, 112), 1, cv2.LINE_AA)
        cv2.putText(canvas, "1.0", (right - 18, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (80, 96, 112), 1, cv2.LINE_AA)
        if scores.size:
            counts, _ = np.histogram(np.clip(scores, 0, 1), bins=20, range=(0, 1))
            maximum = max(1, int(counts.max()))
            plot_width = right - left
            bin_width = plot_width / len(counts)
            for index, count in enumerate(counts):
                x1 = int(left + index * bin_width + 2)
                x2 = int(left + (index + 1) * bin_width - 2)
                y = int(bottom - (bottom - top) * float(count) / maximum)
                cv2.rectangle(canvas, (x1, y), (max(x1 + 1, x2), bottom),
                              (214, 164, 72), -1, cv2.LINE_AA)
            summary = f"{scores.size} candidates  |  mean {float(scores.mean()):.3f}"
        else:
            summary = "No candidate confidence scores"
        cutoff_x = int(left + np.clip(threshold, 0, 1) * (right - left))
        cv2.line(canvas, (cutoff_x, top), (cutoff_x, bottom), (45, 75, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"conf {threshold:.2f}",
                    (min(cutoff_x + 6, right - 95), top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 75, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, summary, (left + 165, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (76, 92, 107), 1, cv2.LINE_AA)
        return canvas

    def browse_yolo_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择单张图像", self.yolo_image_edit.text(),
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if path:
            self.yolo_image_edit.setText(path)

    @staticmethod
    def _draw_yolo_trace_boxes(image: np.ndarray, rows: np.ndarray,
                               names: dict[int, str], title: str,
                               limit: int = 180) -> np.ndarray:
        view = image.copy()
        if view.ndim == 2:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        for row in rows[:limit]:
            x1, y1, x2, y2 = [int(round(float(v))) for v in row[:4]]
            score, class_id = float(row[4]), int(row[5])
            color = (40, 220, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{names.get(class_id, str(class_id))} {score:.2f}"
            cv2.putText(view, label, (max(0, x1), max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        draw_caption(view, f"{title}  |  {len(rows)} boxes" +
                     (f" (showing {min(len(rows), limit)})" if len(rows) > limit else ""))
        return view

    @staticmethod
    def _show_rgb_planes(image: np.ndarray) -> np.ndarray:
        planes = []
        for channel, name, color in ((2, "R", (0, 0, 255)),
                                     (1, "G", (0, 255, 0)),
                                     (0, "B", (255, 0, 0))):
            plane = image[:, :, channel]
            tinted = np.zeros_like(image)
            tinted[:, :, channel] = plane
            cv2.rectangle(tinted, (0, 0), (tinted.shape[1], 32), (18, 34, 54), -1)
            cv2.putText(tinted, f"{name} channel", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 245, 250), 2, cv2.LINE_AA)
            planes.append(tinted)
        return np.hstack(planes)

    def simulate_yolo_steps(self) -> None:
        image_path = Path(self.yolo_image_edit.text()).expanduser()
        weights = Path(self.weights_edit.text()).expanduser()
        if not image_path.is_file():
            QMessageBox.warning(self, "图像不存在", f"找不到图片：\n{image_path}")
            return
        if not weights.is_file():
            QMessageBox.warning(self, "模型不存在", f"找不到权重：\n{weights}")
            return
        device = "0" if self.device_combo.currentIndex() == 0 else "cpu"
        self.yolo_trace_run_button.setEnabled(False)
        self._yolo_request_id += 1
        self.yolo_trace_status.setText("分析任务已提交到后台…")
        for panel in self.yolo_trace_panels:
            panel.image.clear()
            panel.image.setText("正在执行…")
        self.yolo_inference_requested.emit(
            self._yolo_request_id,
            str(image_path.resolve()), str(weights.resolve()), device,
            self.detector_backend_combo.currentData(),
            int(self.controls["imgsz"].value()),
            float(self.controls["confidence"].value()),
            float(self.controls["iou_threshold"].value()),
        )

    def _video_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 16, 12, 10)
        layout.setSpacing(10)
        layout.addWidget(self._section("输入视频"))
        self.video_edit = QLineEdit(str(DEFAULT_VIDEO) if DEFAULT_VIDEO.is_file() else "")
        layout.addLayout(self._file_row(self.video_edit, "video"))
        layout.addWidget(QLabel("模型权重"))
        self.weights_edit = QLineEdit(str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.is_file() else "")
        layout.addLayout(self._file_row(self.weights_edit, "weights"))
        layout.addWidget(QLabel("项目名称"))
        self.project_edit = QLineEdit("MSM-YOLO 实验")
        layout.addWidget(self.project_edit)
        layout.addWidget(QLabel("实验输出目录"))
        self.output_root_edit = QLineEdit(str(RUNS))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_root_edit, 1)
        output_browse = QPushButton("浏览…")
        output_browse.clicked.connect(self.browse_output_root)
        output_row.addWidget(output_browse)
        layout.addLayout(output_row)
        layout.addSpacing(8)
        layout.addWidget(self._section("处理时间段（秒）"))
        time_row = QHBoxLayout()
        self.start_sec = QDoubleSpinBox()
        self.start_sec.setRange(0, 100000)
        self.start_sec.setDecimals(1)
        self.start_sec.setValue(0)
        self.end_sec = QDoubleSpinBox()
        self.end_sec.setRange(0.1, 100000)
        self.end_sec.setDecimals(1)
        self.end_sec.setValue(8)
        time_row.addWidget(QLabel("从"))
        time_row.addWidget(self.start_sec)
        time_row.addWidget(QLabel("到"))
        time_row.addWidget(self.end_sec)
        layout.addLayout(time_row)
        layout.addSpacing(8)
        layout.addWidget(self._section("分析方式"))
        self.motion_radio = QRadioButton("运动模块预览（快速）")
        self.motion_radio.setChecked(True)
        self.detect_radio = QRadioButton("仅目标检测（YOLO）")
        self.motion_yolo_radio = QRadioButton("运动模块 + YOLO 检测")
        self.full_radio = QRadioButton("运动模块 + YOLO + ByteTrack 追踪")
        self.compare_radio = QRadioButton("运动算法并行对比")
        group = QButtonGroup(page)
        group.addButton(self.motion_radio)
        group.addButton(self.detect_radio)
        group.addButton(self.motion_yolo_radio)
        group.addButton(self.full_radio)
        group.addButton(self.compare_radio)
        note = QLabel("只运行对齐、帧差和候选提取，不加载模型。")
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(self._radio_option(self.motion_radio, note))
        note_detect = QLabel("逐帧运行 YOLO，不进行目标关联；用于单独查看检测效果。")
        note_detect.setObjectName("hint")
        note_detect.setWordWrap(True)
        layout.addWidget(self._radio_option(self.detect_radio, note_detect))
        note_motion_yolo = QLabel("先做相机运动补偿与候选区域提取，再交由 YOLO 复核；不做轨迹关联。")
        note_motion_yolo.setObjectName("hint")
        note_motion_yolo.setWordWrap(True)
        layout.addWidget(self._radio_option(self.motion_yolo_radio, note_motion_yolo))
        note2 = QLabel("在运动模块 + YOLO 的基础上增加 ByteTrack 目标关联。")
        note2.setObjectName("hint")
        note2.setWordWrap(True)
        layout.addWidget(self._radio_option(self.full_radio, note2))
        compare_note = QLabel("同一视频帧分别输入光流+帧差、纯帧差和 MOG2，各自维护状态并输出对比视频。")
        compare_note.setObjectName("hint")
        compare_note.setWordWrap(True)
        layout.addWidget(self._radio_option(self.compare_radio, compare_note))
        self.motion_algorithm_combo = QComboBox()
        self.motion_algorithm_combo.addItems([name for _, name in motion_algorithm_choices()])
        layout.addLayout(self._labeled_row("单算法/检测所用运动模块", self.motion_algorithm_combo))
        layout.addStretch(1)
        return self._scroll_page(page)

    def _motion_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 16, 12, 10)
        layout.setSpacing(7)
        layout.addWidget(self._section("运动区域提取"))
        hint = QLabel("稀疏帧差将自动拉伸显示；候选区域会在图中描边。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        preset_row = QHBoxLayout()
        for text, name in (("原始", "original"), ("偏灵敏", "sensitive"), ("偏稳健", "conservative")):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, preset=name: self.set_preset(preset))
            button.setObjectName("presetButton")
            preset_row.addWidget(button)
        layout.addLayout(preset_row)
        layout.addSpacing(4)
        self._add_spin(layout, "阈值标准差系数", "threshold_sigma", 0, 4, 0.1, True)
        self._add_spin(layout, "阈值额外偏置", "threshold_offset", 0, 30, 1)
        self._add_spin(layout, "运动历史窗口（帧）", "history_window", 2, 5, 1)
        self._add_spin(layout, "局部阈值窗口", "threshold_window", 3, 61, 2)
        self._add_spin(layout, "形态学核尺寸", "morph_kernel", 1, 9, 2)
        self._add_spin(layout, "开运算次数", "open_iterations", 0, 3, 1)
        self._add_spin(layout, "闭运算次数", "close_iterations", 0, 4, 1)
        self._add_spin(layout, "最小连通区域面积", "min_area", 1, 300, 1)
        self._add_spin(layout, "边缘屏蔽比例 (%)", "edge_margin_percent", 0, 10, 0.5, True)
        self._add_spin(layout, "候选区域上限", "max_proposals", 1, 50, 1)
        layout.addSpacing(8)
        layout.addWidget(self._section("MOG2 专属参数"))
        self._add_spin(layout, "背景学习历史帧数", "mog2_history", 30, 2000, 10)
        self._add_spin(layout, "前景方差阈值", "mog2_var_threshold", 2, 100, 1, True)
        self.mog2_shadows = QCheckBox("检测阴影（阴影不计入前景候选）")
        self.mog2_shadows.setChecked(True)
        self.controls["mog2_detect_shadows"] = self.mog2_shadows
        layout.addWidget(self.mog2_shadows)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _tracking_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 16, 12, 10)
        layout.setSpacing(7)
        layout.addWidget(self._section("检测与追踪"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU 0 · CUDA", "CPU"])
        layout.addLayout(self._labeled_row("推理设备", self.device_combo))
        self.detector_backend_combo = QComboBox()
        self.detector_backend_combo.addItem("自动识别", "auto")
        self.detector_backend_combo.addItem("原版 YOLOv5", "legacy_yolov5")
        self.detector_backend_combo.addItem("Ultralytics", "ultralytics")
        layout.addLayout(self._labeled_row("检测后端", self.detector_backend_combo))
        self.tracker_backend_combo = QComboBox()
        for backend_id, label in tracker_backend_choices():
            self.tracker_backend_combo.addItem(label, backend_id)
        layout.addLayout(self._labeled_row("追踪后端", self.tracker_backend_combo))
        self._add_spin(layout, "ROI 边长", "roi_size", 320, 1280, 32)
        self._add_spin(layout, "检测置信度", "confidence", 0.05, 0.95, 0.05, True)
        self._add_spin(layout, "NMS IoU 阈值", "iou_threshold", 0.05, 0.95, 0.05, True)
        self._add_spin(layout, "全图重捕获间隔（帧）", "global_interval", 1, 60, 1)
        self._add_spin(layout, "YOLO 输入尺寸", "imgsz", 320, 1280, 32)
        self._add_spin(layout, "ByteTrack 缓冲（帧）", "track_buffer", 1, 120, 1)
        self._add_spin(layout, "预览抽帧间隔", "show_every", 1, 120, 1)
        note = QLabel("视频逐帧处理；该参数只控制界面保留多少张预览帧。")
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        save = QPushButton("保存参数")
        load = QPushButton("载入参数")
        save.clicked.connect(self.save_config)
        load.clicked.connect(self.load_config)
        buttons.addWidget(save)
        buttons.addWidget(load)
        layout.addLayout(buttons)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _experiments_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        title_row = QHBoxLayout()
        title_row.addWidget(self._section("本地 SQLite 实验库"))
        title_row.addStretch(1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_experiments)
        title_row.addWidget(refresh)
        layout.addLayout(title_row)
        self.experiment_table = QTableWidget(0, 6)
        self.experiment_table.setHorizontalHeaderLabels(
            ["状态", "模式", "开始时间", "运动算法", "检测后端", "结果视频"])
        self.experiment_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.experiment_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.experiment_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.experiment_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch)
        self.experiment_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.experiment_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.experiment_table, 1)
        open_button = QPushButton("打开所选结果视频")
        open_button.clicked.connect(self.open_selected_experiment)
        layout.addWidget(open_button)
        self.refresh_experiments()
        return self._scroll_page(page)

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    @staticmethod
    def _radio_option(radio: QRadioButton, explanation: QLabel) -> QFrame:
        option = QFrame()
        option.setObjectName("radioOption")
        layout = QVBoxLayout(option)
        layout.setContentsMargins(7, 5, 7, 6)
        layout.setSpacing(3)
        explanation.setContentsMargins(25, 0, 3, 0)
        layout.addWidget(radio)
        layout.addWidget(explanation)
        return option

    def _file_row(self, edit: QLineEdit, kind: str) -> QHBoxLayout:
        row = QHBoxLayout()
        button = QPushButton("浏览…")
        button.clicked.connect(lambda: self.browse(edit, kind))
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _labeled_row(self, text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(text)
        label.setMinimumWidth(105)
        label.setMaximumWidth(138)
        label.setWordWrap(True)
        row.addWidget(label)
        row.addWidget(widget, 1)
        return row

    def _add_spin(self, layout: QVBoxLayout, label: str, key: str,
                  minimum: float, maximum: float, step: float,
                  decimal: bool = False) -> None:
        box = QDoubleSpinBox() if decimal else QSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        if decimal:
            box.setDecimals(2)
        box.setValue(self.DEFAULTS[key])
        self.controls[key] = box
        layout.addLayout(self._labeled_row(label, box))

    @staticmethod
    def _algorithm_parameter_keys() -> tuple[str, ...]:
        return ("threshold_sigma", "threshold_offset", "threshold_window",
                "morph_kernel", "open_iterations", "close_iterations", "min_area",
                "max_proposals", "edge_margin_percent", "mog2_history",
                "mog2_var_threshold", "mog2_detect_shadows")

    def _capture_algorithm_config(self) -> dict:
        return {key: (widget.isChecked() if isinstance(widget, QCheckBox) else widget.value())
                for key, widget in self.controls.items()
                if key in self._algorithm_parameter_keys()}

    def _apply_algorithm_config(self, config: dict) -> None:
        for key, value in config.items():
            widget = self.controls.get(key)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(value)

    def _on_motion_algorithm_changed(self, index: int) -> None:
        if not hasattr(self, "algorithm_configs"):
            return
        self.algorithm_configs[self._active_algorithm_key] = self._capture_algorithm_config()
        selected = motion_algorithm_choices()[index][0]
        self._active_algorithm_key = selected
        self._apply_algorithm_config(self.algorithm_configs.get(selected, {}))

    def _stylesheet(self) -> str:
        return """
        QMainWindow, QWidget { background: #F3F6FA; color: #24364B; }
        QLabel { font-size: 10.5pt; }
        #pageTitle { font-size: 21pt; font-weight: 700; color: #152D46; }
        #subtitle { color: #718198; }
        #hint { color: #718198; font-size: 9.5pt; }
        #sectionTitle { font-size: 14pt; font-weight: 700; color: #203A56; }
        #sectionLabel { font-size: 12pt; font-weight: 700; color: #203A56; padding: 5px 0; }
        #statusBadge { background: #E7F5EC; color: #28784E; border-radius: 12px; padding: 7px 14px; font-weight: 700; }
        #statusRunning { background: #FFF2D9; color: #A6630B; border-radius: 12px; padding: 7px 14px; font-weight: 700; }
        #sidebar, #metricCard, #imagePanel { background: #FFFFFF; border: 1px solid #DFE6EE; border-radius: 12px; }
        #radioOption { background: #F7F9FC; border: 1px solid #E5EBF2; border-radius: 8px; }
        QScrollArea { border: 0; }
        QGroupBox { background: #FFFFFF; border: 1px solid #DFE6EE; border-radius: 9px; margin-top: 12px; padding: 10px; font-weight: 700; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #203A56; }
        #metricName { color: #718198; font-size: 9.5pt; }
        #metricValue { color: #17324E; font-size: 19pt; font-weight: 700; }
        #panelTitle { font-weight: 700; color: #2A4058; }
        #imageSurface { background: #142235; color: #B8C6D7; border-radius: 8px; }
        QTabWidget::pane { border: 1px solid #E0E7EF; border-radius: 9px; background: white; top: -1px; }
        QTabBar::tab { background: #EDF2F7; color: #51657B; padding: 10px 13px; margin-right: 4px; border-radius: 7px 7px 0 0; }
        QTabBar::tab:selected { background: #FFFFFF; color: #24699F; font-weight: 700; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #FFFFFF; border: 1px solid #CBD6E2; border-radius: 6px; padding: 7px 8px; min-height: 20px; selection-background-color: #3586C4; }
        QPushButton { background: #EAF0F6; border: 1px solid #D4DEE8; border-radius: 7px; padding: 8px 10px; color: #29425B; font-weight: 600; }
        QPushButton:hover { background: #DCE8F3; }
        #primaryButton { background: #287FBD; color: white; border: 0; font-size: 11pt; padding: 11px; }
        #primaryButton:hover { background: #216EA6; }
        #secondaryButton { background: #F0F3F7; }
        #presetButton { padding: 7px 4px; }
        QRadioButton { spacing: 8px; font-weight: 600; }
        QProgressBar { background: #DCE5EF; border: 0; border-radius: 4px; }
        QProgressBar::chunk { background: #3792D0; border-radius: 4px; }
        QSlider::groove:horizontal { background: #D9E3ED; height: 5px; border-radius: 2px; }
        QSlider::handle:horizontal { background: #2C82BE; width: 15px; margin: -5px 0; border-radius: 7px; }
        """

    def browse(self, edit: QLineEdit, kind: str) -> None:
        if kind == "video":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", edit.text(),
                                                   "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", edit.text(),
                                                   "PyTorch 权重 (*.pt);;所有文件 (*)")
        if path:
            edit.setText(path)

    def browse_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择实验输出目录", self.output_root_edit.text())
        if path:
            self.output_root_edit.setText(path)

    def set_preset(self, name: str) -> None:
        presets = {
            "original": (2.0, 8, 21, 3, 1, 2, 12, 3.0, 20),
            "sensitive": (1.4, 3, 21, 1, 0, 1, 4, 2.0, 30),
            "conservative": (2.5, 12, 31, 5, 1, 1, 20, 4.0, 20),
        }
        keys = ("threshold_sigma", "threshold_offset", "threshold_window", "morph_kernel",
                "open_iterations", "close_iterations", "min_area", "edge_margin_percent",
                "max_proposals")
        for key, value in zip(keys, presets[name]):
            self.controls[key].setValue(value)
        self.status_text.setText({"original": "已载入原始参数。",
                                  "sensitive": "偏灵敏：保留更多小候选，也可能增加噪声。",
                                  "conservative": "偏稳健：候选更少，微小区域也可能被过滤。"}[name])

    def settings(self) -> dict:
        values = {key: (widget.isChecked() if isinstance(widget, QCheckBox)
                        else widget.value()) for key, widget in self.controls.items()}
        if hasattr(self, "algorithm_configs"):
            self.algorithm_configs[self._active_algorithm_key] = self._capture_algorithm_config()
            values["motion_algorithm_configs"] = {
                key: dict(value) for key, value in self.algorithm_configs.items()
            }
        for key in ("threshold_window", "morph_kernel"):
            values[key] = int(values[key]) | 1
        return values

    def save_config(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "保存参数配置", str(RUNS / "motion_tuning.json"),
                                               "JSON 配置 (*.json)")
        if not path:
            return
        data = {"version": 1, "parameters": self.settings(),
                "video": self.video_edit.text(), "weights": self.weights_edit.text(),
                "device": self.device_combo.currentText(),
                "detector_backend": self.detector_backend_combo.currentData(),
                "tracker_backend": self.tracker_backend_combo.currentData(),
                "motion_algorithm": motion_algorithm_choices()[self.motion_algorithm_combo.currentIndex()][0],
                "project": self.project_edit.text(), "output_root": self.output_root_edit.text(),
                "start_seconds": self.start_sec.value(), "end_seconds": self.end_sec.value()}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_text.setText(f"参数已保存：{path}")

    def load_config(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, "载入参数配置", str(RUNS), "JSON 配置 (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for key, value in data.get("parameters", {}).items():
                if key in self.controls:
                    widget = self.controls[key]
                    widget.setChecked(bool(value)) if isinstance(widget, QCheckBox) else widget.setValue(value)
            saved_algorithm_configs = data.get("parameters", {}).get("motion_algorithm_configs")
            if isinstance(saved_algorithm_configs, dict):
                self.algorithm_configs.update(saved_algorithm_configs)
            self.video_edit.setText(data.get("video", self.video_edit.text()))
            self.weights_edit.setText(data.get("weights", self.weights_edit.text()))
            self.project_edit.setText(data.get("project", self.project_edit.text()))
            self.output_root_edit.setText(data.get("output_root", self.output_root_edit.text()))
            if "device" in data:
                self.device_combo.setCurrentIndex(0 if str(data["device"]).startswith("GPU") else 1)
            if data.get("detector_backend"):
                index = self.detector_backend_combo.findData(data["detector_backend"])
                if index >= 0:
                    self.detector_backend_combo.setCurrentIndex(index)
            if data.get("tracker_backend"):
                index = self.tracker_backend_combo.findData(data["tracker_backend"])
                if index >= 0:
                    self.tracker_backend_combo.setCurrentIndex(index)
            if data.get("motion_algorithm"):
                index = [key for key, _ in motion_algorithm_choices()].index(data["motion_algorithm"])
                self.motion_algorithm_combo.setCurrentIndex(index)
                self._apply_algorithm_config(self.algorithm_configs[data["motion_algorithm"]])
            self.start_sec.setValue(float(data.get("start_seconds", self.start_sec.value())))
            self.end_sec.setValue(float(data.get("end_seconds", self.end_sec.value())))
            self.status_text.setText(f"参数已载入：{path}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "载入失败", str(exc))

    def start(self) -> None:
        if self.thread is not None:
            return
        video = Path(self.video_edit.text()).expanduser()
        if not video.is_file():
            QMessageBox.warning(self, "视频不存在", f"找不到视频文件：\n{video}")
            return
        if self.end_sec.value() <= self.start_sec.value():
            QMessageBox.warning(self, "时间范围无效", "结束时间必须大于起始时间。")
            return
        self.mode = ("compare" if self.compare_radio.isChecked() else
                     "full" if self.full_radio.isChecked() else
                     "motion_yolo" if self.motion_yolo_radio.isChecked() else
                     "detect" if self.detect_radio.isChecked() else "motion")
        if self.mode in ("detect", "motion_yolo", "full") and not Path(self.weights_edit.text()).is_file():
            QMessageBox.warning(self, "模型不存在", f"找不到模型权重：\n{self.weights_edit.text()}")
            return
        device = "0" if self.device_combo.currentIndex() == 0 else "cpu"
        self.snapshots.clear()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.scrub_label.setText("0 / 0")
        panel_titles = {
            "motion": ("候选区域 / 运动提取结果", "相机补偿后的前一帧",
                       "帧差 · 对比度增强", "二值运动掩膜 · 候选框标注"),
            "detect": ("YOLO 检测结果", "当前输入帧", "帧差（检测模式未运行）",
                       "运动掩膜（检测模式未运行）"),
            "motion_yolo": ("运动模块 + YOLO 检测结果", "相机补偿后的前一帧",
                            "帧差 · 对比度增强", "二值运动掩膜 · 候选框标注"),
            "full": ("运动模块 + YOLO + ByteTrack", "相机补偿后的前一帧",
                     "帧差 · 对比度增强", "二值运动掩膜 · 候选框标注"),
            "compare": ("原始视频帧", "光流 + 帧差", "纯帧差", "MOG2 背景减除"),
        }[self.mode]
        for panel, title in zip(self.panels, panel_titles):
            panel.title.setText(title)
        for panel in self.panels:
            panel.image.setText("正在准备分析…")
            panel.image.setPixmap(QPixmap())
        for value in self.metric_values:
            value.setText("—")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_badge.setText("运行中")
        self.status_badge.setObjectName("statusRunning")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.status_text.setText("正在启动处理线程……")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        selected_algorithm = motion_algorithm_choices()[self.motion_algorithm_combo.currentIndex()][0]
        self.worker = ProcessingWorker(
            video, self.start_sec.value(), self.end_sec.value(), self.settings(), self.mode,
            self.weights_edit.text(), device, selected_algorithm,
            self.detector_backend_combo.currentData(), self.project_edit.text(),
            Path(self.output_root_edit.text()).expanduser(),
            self.tracker_backend_combo.currentData())
        self.worker.setParent(self)
        self.thread = ProcessingThread(self.worker, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.status.connect(self.status_text.setText)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        run_thread = self.thread
        self.thread.finished.connect(lambda thread=run_thread: self._on_thread_finished(thread))
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def refresh_experiments(self) -> None:
        try:
            records = ExperimentStore(DATABASE).recent_experiments(200)
        except Exception as exc:
            QMessageBox.warning(self, "读取实验库失败", str(exc))
            return
        self.experiment_table.setRowCount(len(records))
        for row, record in enumerate(records):
            algorithms = json.loads(record["motion_algorithms_json"] or "{}")
            algorithm_label = ("三算法并行对比" if algorithms.get("comparison")
                               else algorithms.get("selected", "—"))
            values = [record["status"], record["mode"], record["started_at"],
                      algorithm_label, record["detector_backend"] or "—",
                      record["output_video"] or record["output_dir"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 5:
                    item.setToolTip(str(value))
                self.experiment_table.setItem(row, column, item)
            self.experiment_table.item(row, 0).setData(Qt.ItemDataRole.UserRole,
                                                         record.get("output_video") or record["output_dir"])

    def open_selected_experiment(self) -> None:
        row = self.experiment_table.currentRow()
        if row < 0:
            return
        item = self.experiment_table.item(row, 0)
        path = Path(item.data(Qt.ItemDataRole.UserRole) or "")
        if path.is_dir():
            QMessageBox.information(self, "实验目录", str(path))
            return
        if not path.is_file():
            QMessageBox.warning(self, "文件不存在", f"结果文件不可用：\n{path}")
            return
        os.startfile(path) if hasattr(os, "startfile") else __import__("subprocess").Popen(["xdg-open", str(path)])

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.status_text.setText("正在停止并保存当前结果……")

    def on_progress(self, current: int, total: int, message: str,
                    candidates: int, detections: int, speed: float) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        self.metric_values[0].setText(f"{current:,}")
        self.metric_values[1].setText(f"{candidates:,}")
        self.metric_values[2].setText(f"{detections:,}")
        self.metric_values[3].setText(f"{speed:.1f} FPS")
        self.status_text.setText(message)

    def on_snapshot(self, frame_number: int, panels: list[np.ndarray], summary: str) -> None:
        self.snapshots.append((frame_number, panels, summary))
        index = len(self.snapshots) - 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, index)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self.show_snapshot(index)

    def show_snapshot(self, index: int) -> None:
        if not self.snapshots:
            return
        index = max(0, min(index, len(self.snapshots) - 1))
        frame_number, panels, summary = self.snapshots[index]
        self.frame_info.setText(f"帧 {frame_number}  ·  {summary}")
        self.scrub_label.setText(f"{index + 1} / {len(self.snapshots)}")
        for panel, image in zip(self.panels, panels):
            panel.show_image(image)

    def on_done(self, message: str, output: str, summary: dict) -> None:
        stopped = message.startswith("已停止。")
        self._finished("已停止" if stopped else "已完成",
                       "#A6630B" if stopped else "#28784E")
        self.status_text.setText(f"{message}\n结果视频：{output}")

    def on_failed(self, message: str) -> None:
        self._finished("出错", "#B13E4A")
        self.status_text.setText(message)

    def _finished(self, text: str, color: str) -> None:
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(f"background:#F2F5F8;color:{color};border-radius:12px;padding:7px 14px;font-weight:700;")

    def _on_thread_finished(self, finished_thread: QThread) -> None:
        if self.thread is not finished_thread:
            return
        finished_worker = self.worker
        finished_thread.worker = None
        self.thread = None
        self.worker = None
        if finished_worker is not None:
            finished_worker.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event) -> None:
        if self.deployment_preflight_thread is not None and self.deployment_preflight_thread.isRunning():
            self.deployment_preflight_thread.wait()
        if self.deployment_thread is not None and self.deployment_thread.isRunning():
            self.deployment_thread.cancel()
            if not self.deployment_thread.wait(15_000):
                self.deployment_thread.force_kill()
                self.deployment_thread.wait(2_000)
            if self.deployment_run_id:
                try:
                    self.deployment_store.update_deployment_run(
                        self.deployment_run_id, status="stopped",
                        error="工作台关闭时停止部署转换。")
                except Exception:
                    pass
        if self.training_validation_thread is not None and self.training_validation_thread.isRunning():
            self.training_validation_thread.wait()
            QApplication.processEvents()
        if self.training_run_id:
            run_id = self.training_run_id
            self.training_backend.shutdown(timeout=15)
            stopped = self.training_backend.status(run_id)
            try:
                record = self.training_store.training_run(run_id)
                self.training_store.update_training_run(
                    run_id, status=stopped.state.value, epoch=stopped.epoch,
                    epochs=stopped.epochs, progress=stopped.progress,
                    metrics=dict(stopped.metrics), log_tail=self.training_backend.log_tail(run_id),
                    artifacts=list(stopped.artifacts), return_code=stopped.return_code,
                    error="桌面工作台关闭时停止训练进程。" if stopped.state == TrainingState.STOPPED else None)
            except Exception:
                pass
            self.training_timer.stop()
        if self.worker is not None and self.thread is not None and self.thread.isRunning():
            self.worker.stop()
            self.thread.wait()
        if hasattr(self, "yolo_trace_thread") and self.yolo_trace_thread.isRunning():
            self.yolo_shutdown_requested.emit()
            self.yolo_trace_thread.wait()
        event.accept()


def main() -> None:
    app = QApplication([])
    font_file = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_id = QFontDatabase.addApplicationFont(str(font_file)) if font_file.is_file() else -1
    registered = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    preferred = next((name for name in registered if "CJK SC" in name), "Noto Sans CJK SC")
    app.setFont(QFont(preferred, 10))
    app.setStyle("Fusion")
    window = TunerWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
