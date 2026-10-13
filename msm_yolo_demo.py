#!/usr/bin/env python3
"""Paper-inspired MSM-YOLO video demo using an unchanged YOLOv5 checkpoint.

Implements temporal control flow (global detection, motion proposals, and
next-frame local ROI detection). This is not the paper's trained model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, deque
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from msm_lab.contracts import (Detection, DetectionTrace, DetectionTraceSnapshot,
                               MotionResult, Track)
from msm_lab.interfaces import DetectorBackend, MotionAlgorithm, TrackerBackend


TrackedObject = Track


def clip_box(x1: float, y1: float, x2: float, y2: float,
             width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


class YOLOv5Detector:
    """Run YOLOv5 checkpoints with the repository's native inference stack."""

    backend_id = "legacy_yolov5"

    def __init__(self, weights: str, device: str | None = None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "当前 conda 环境缺少 PyTorch，请切换到 yolo 环境后运行。"
            ) from exc

        # best.pt is a YOLOv5 checkpoint; use its matching local code rather
        # than the newer Ultralytics predictor, whose model API is incompatible.
        configured_root = os.environ.get("MSM_YOLOV5_ROOT")
        yolov5_root = (
            Path(configured_root).expanduser().resolve()
            if configured_root else Path(__file__).resolve().parent / "external" / "yolov5"
        )
        if not yolov5_root.is_dir() and not configured_root:
            # Keep existing local installations usable during migration.
            yolov5_root = Path(__file__).resolve().parents[2] / "RK-yolo" / "yolov5"
        if not yolov5_root.is_dir():
            raise RuntimeError(
                f"找不到原版 YOLOv5 代码目录：{yolov5_root}。"
                "请设置 MSM_YOLOV5_ROOT 指向已安装的 YOLOv5 仓库。"
            )
        sys.path.insert(0, str(yolov5_root))
        try:
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox
            from utils.general import check_img_size, non_max_suppression, scale_boxes
            from utils.torch_utils import select_device
        except ImportError as exc:
            raise RuntimeError(
                f"无法导入 YOLOv5 推理模块，请检查 yolo 环境依赖：{yolov5_root}"
            ) from exc

        self.torch = torch
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.device = select_device(device or "")
        self.version = "local-yolov5"
        if self.device.type == "cuda":
            try:
                probe = torch.ones((2, 2), device=self.device)
                _ = probe @ probe
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"PyTorch 检测到 GPU {self.device}，但 CUDA 运算失败。"
                    "当前 PyTorch/CUDA 构建可能不包含该显卡架构的内核；"
                    "请安装支持当前 GPU 的 CUDA 版 PyTorch。"
                ) from exc
        # PyTorch 2.6 changed torch.load's default to weights_only=True;
        # this locally supplied legacy YOLOv5 checkpoint stores model objects.
        # Temporarily opt into legacy loading only while reading this file.
        original_torch_load = torch.load

        def trusted_checkpoint_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = trusted_checkpoint_load
        try:
            self.model = DetectMultiBackend(
                weights, device=self.device, dnn=False, data=None,
                fp16=self.device.type != "cpu",
            )
        finally:
            torch.load = original_torch_load
        self.stride = self.model.stride
        self.names = self.model.names
        if isinstance(self.names, (list, tuple)):
            self.names = {index: name for index, name in enumerate(self.names)}
        self.imgsz = check_img_size(640, s=self.stride)
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        self._trace_snapshot = None
        self._trace_candidates = None

    def predict(self, image: np.ndarray, confidence: float,
                imgsz: int, iou: float = 0.45) -> list[Detection]:
        # YOLOv5 letterbox + RGB/NCHW preprocessing.
        padded = self.letterbox(
            image, new_shape=imgsz, stride=self.stride, auto=self.model.pt
        )[0]
        tensor = padded[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor)
        tensor = self.torch.from_numpy(tensor).to(self.device)
        tensor = tensor.half() if self.model.fp16 else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        prediction = self.model(tensor, augment=False, visualize=False)
        prediction = self.non_max_suppression(
            prediction, confidence, float(iou), max_det=1000
        )[0]
        if not len(prediction):
            return []
        prediction[:, :4] = self.scale_boxes(
            tensor.shape[2:], prediction[:, :4], image.shape
        ).round()
        values = prediction.detach().cpu().numpy()
        h, w = image.shape[:2]
        return [
            Detection(*clip_box(*row[:4], w, h), float(row[4]), int(row[5]))
            for row in values
        ]

    def trace_inference(self, image: np.ndarray,
                        imgsz: int) -> DetectionTraceSnapshot:
        """Run one native YOLOv5 forward pass for threshold-only replay."""
        padded = self.letterbox(
            image, new_shape=imgsz, stride=self.stride, auto=self.model.pt
        )[0]
        array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        tensor = self.torch.from_numpy(array).to(self.device)
        tensor = tensor.half() if self.model.fp16 else tensor.float()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)
        with self.torch.inference_mode():
            raw = self.model(tensor, augment=False, visualize=False)
        normalized = np.clip(
            tensor[0].detach().float().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0,
            0, 255,
        ).astype(np.uint8)
        snapshot = DetectionTraceSnapshot(
            backend=self.backend_id, raw_prediction=raw,
            resized_image=padded.copy(), normalized_image=normalized,
            original_shape=tuple(image.shape), input_shape=tuple(tensor.shape),
            tensor_range=(float(tensor.min().item()), float(tensor.max().item())),
            end2end=False,
        )
        self._trace_snapshot = snapshot
        self._trace_candidates = None
        return snapshot

    def trace_postprocess(self, snapshot: DetectionTraceSnapshot,
                          confidence: float, iou: float) -> DetectionTrace:
        raw = snapshot.raw_prediction
        if self._trace_snapshot is snapshot and self._trace_candidates is not None:
            candidates = self._trace_candidates
        else:
            candidates = self.non_max_suppression(
                raw.clone() if hasattr(raw, "clone") else raw,
                0.001, 1.0, max_det=3000,
            )[0]
            self._trace_snapshot = snapshot
            self._trace_candidates = candidates
        filtered = (candidates[candidates[:, 4] >= float(confidence)]
                    if len(candidates) else candidates)
        kept = self.non_max_suppression(
            raw.clone() if hasattr(raw, "clone") else raw,
            float(confidence), float(iou), max_det=1000,
        )[0]
        final = kept.clone()
        if len(final):
            final[:, :4] = self.scale_boxes(
                snapshot.input_shape[2:], final[:, :4], snapshot.original_shape
            ).round()
        arrays = [item.detach().cpu().numpy() for item in (candidates, filtered, kept, final)]
        return DetectionTrace(*arrays, float(confidence), float(iou), False)

    def trace(self, image: np.ndarray, confidence: float, imgsz: int,
              iou: float = 0.45) -> dict:
        snapshot = self.trace_inference(image, imgsz)
        stages = self.trace_postprocess(snapshot, confidence, iou)
        return {
            "resized": snapshot.resized_image,
            "normalized": snapshot.normalized_image,
            "candidates": stages.candidates,
            "confidence_filtered": stages.confidence_filtered,
            "nms": stages.nms, "final": stages.final,
            "input_shape": snapshot.input_shape,
            "tensor_range": snapshot.tensor_range,
            "backend": snapshot.backend, "end2end": stages.end2end,
        }

    def close(self) -> None:
        self.model = None


class ByteTrackAdapter:
    """Pass per-frame detections into Ultralytics ByteTrack."""

    backend_id = "bytetrack_ultralytics"

    def __init__(self, confidence: float = 0.60, frame_rate: float = 30.0,
                 track_buffer: int = 30):
        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack 需要 yolo 环境中的 ultralytics 和 lap 依赖。"
            ) from exc
        self.Boxes = Boxes
        tracker_args = SimpleNamespace(
            track_high_thresh=confidence,
            track_low_thresh=min(0.10, confidence),
            new_track_thresh=confidence,
            track_buffer=track_buffer,
            match_thresh=0.80,
            fuse_score=True,
        )
        try:
            self.tracker = BYTETracker(
                tracker_args, frame_rate=max(1, round(frame_rate))
            )
        except TypeError as exc:
            # Newer Ultralytics versions removed the frame_rate constructor arg.
            if "frame_rate" not in str(exc):
                raise
            self.tracker = BYTETracker(tracker_args)
        self.track_frame_count = 0
        self.seen_track_ids: set[int] = set()
        try:
            from importlib.metadata import version
            self.version = version("ultralytics")
        except Exception:
            self.version = "unknown"

    def update(self, detections: list[Detection],
               frame_shape: tuple[int, ...]) -> list[TrackedObject]:
        height, width = frame_shape[:2]
        rows = np.asarray([
            [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id]
            for d in detections
        ], dtype=np.float32).reshape(-1, 6)
        boxes = self.Boxes(rows, (height, width))
        raw_tracks = self.tracker.update(boxes)
        tracks = [
            TrackedObject(
                *clip_box(*row[:4], width, height),
                int(row[4]), float(row[5]), int(row[6]),
            )
            for row in raw_tracks
        ]
        if tracks:
            self.track_frame_count += 1
            self.seen_track_ids.update(track.track_id for track in tracks)
        return tracks

    def reset(self) -> None:
        tracker_type = type(self.tracker)
        try:
            self.tracker = tracker_type(
                self.tracker.args,
                frame_rate=max(1, round(self.tracker.frame_rate)),
            )
        except (AttributeError, TypeError):
            self.tracker = tracker_type(self.tracker.args)
        self.track_frame_count = 0
        self.seen_track_ids.clear()


class MotionProposer:
    """Camera-compensated temporal difference with adaptive thresholding."""

    algorithm_id = "flow_frame_difference"
    display_name = "光流 + 帧差（当前算法）"

    def __init__(self, min_area: int = 12, max_area_ratio: float = 0.12,
                 threshold_sigma: float = 2.0, threshold_offset: float = 8.0,
                 threshold_window: int = 21, morph_kernel: int = 3,
                 open_iterations: int = 1, close_iterations: int = 2,
                 max_proposals: int = 20, edge_margin_ratio: float = 0.03,
                 history_window: int = 3):
        self.previous_gray: np.ndarray | None = None
        self.history_window = max(2, min(5, int(history_window)))
        self.history_frames: deque[np.ndarray] = deque(
            maxlen=self.history_window - 1
        )
        # Each matrix maps one retained frame to the next retained frame.
        self.history_transforms: deque[np.ndarray] = deque(
            maxlen=max(0, self.history_window - 2)
        )
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.threshold_sigma = threshold_sigma
        self.threshold_offset = threshold_offset
        self.threshold_window = max(3, int(threshold_window) | 1)
        self.morph_kernel = max(1, int(morph_kernel) | 1)
        self.open_iterations = max(0, int(open_iterations))
        self.close_iterations = max(0, int(close_iterations))
        self.max_proposals = max(1, int(max_proposals))
        self.edge_margin_ratio = max(0.0, min(float(edge_margin_ratio), 0.15))
        self.target_template: np.ndarray | None = None
        self.last_aligned: np.ndarray | None = None
        self.last_diff: np.ndarray | None = None
        self.last_threshold: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_proposals: list[tuple[tuple[int, int, int, int], float]] = []
        self.last_stage_images: dict[str, np.ndarray] = {}
        self.last_history_count = 0
        self.last_compensation_valid = False

    def update_target_template(self, frame: np.ndarray,
                               detection: Detection) -> None:
        """Keep a compact appearance template from a detector-confirmed target."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = clip_box(
            detection.x1, detection.y1, detection.x2, detection.y2,
            gray.shape[1], gray.shape[0],
        )
        patch = gray[y1:y2, x1:x2]
        if patch.size and min(patch.shape[:2]) >= 3:
            self.target_template = cv2.resize(
                patch, (32, 32), interpolation=cv2.INTER_AREA
            )

    def propose(self, frame: np.ndarray
                ) -> list[tuple[tuple[int, int, int, int], float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous = self.previous_gray
        if previous is None or previous.shape != gray.shape:
            self.history_frames.clear()
            self.history_transforms.clear()
            self.history_frames.append(gray.copy())
            self.previous_gray = gray
            self.last_history_count = 0
            self.last_compensation_valid = False
            self.last_aligned = gray.copy()
            self.last_diff = np.zeros_like(gray)
            self.last_threshold = np.zeros_like(gray)
            self.last_mask = np.zeros_like(gray)
            self.last_proposals = []
            self.last_stage_images = {
                "原始输入帧": frame.copy(),
                "预处理灰度帧": gray.copy(),
                "光流特征点与相机补偿": frame.copy(),
                "对齐后上一帧": gray.copy(),
                "帧差响应": np.zeros_like(gray),
                "阈值分割结果": np.zeros_like(gray),
                "形态学清理后掩膜": np.zeros_like(gray),
                "连通区域候选框": frame.copy(),
            }
            return []

        aligned_previous = previous
        flow_visual = frame.copy()
        valid_alignment = np.full_like(gray, 255, dtype=np.uint8)
        homography = None
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=500, qualityLevel=0.01, minDistance=8,
            blockSize=7,
        )
        if points is not None and len(points) >= 4:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                previous, gray, points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          30, 0.01),
            )
            if moved is not None and status is not None:
                good_old = points[status.reshape(-1) == 1]
                good_new = moved[status.reshape(-1) == 1]
                if len(good_old) >= 4:
                    homography, _ = cv2.findHomography(
                        good_old, good_new, cv2.RANSAC, 3.0
                    )
                    if (homography is not None
                            and homography.shape == (3, 3)
                            and bool(np.isfinite(homography).all())):
                        h, w = gray.shape
                        aligned_previous = cv2.warpPerspective(
                            previous, homography, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        valid_alignment = cv2.warpPerspective(
                            np.full_like(previous, 255, dtype=np.uint8),
                            homography, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        for old_point, new_point in zip(
                                good_old.reshape(-1, 2)[::max(1, len(good_old) // 100)],
                                good_new.reshape(-1, 2)[::max(1, len(good_new) // 100)]):
                            old_xy = tuple(np.round(old_point).astype(int))
                            new_xy = tuple(np.round(new_point).astype(int))
                            cv2.line(flow_visual, old_xy, new_xy,
                                     (60, 220, 255), 1, cv2.LINE_AA)
                            cv2.circle(flow_visual, new_xy, 2,
                                       (80, 255, 100), -1, cv2.LINE_AA)

        compensation_valid = (
            homography is not None
            and homography.shape == (3, 3)
            and bool(np.isfinite(homography).all())
        )
        prior_frames = list(self.history_frames)
        prior_transforms = list(self.history_transforms)
        aligned_frames: list[tuple[np.ndarray, np.ndarray]] = []

        # Align each retained history frame into the current frame. Adjacent
        # homographies are composed from newest to oldest; max residual fusion
        # preserves a weak target response from any frame in the short window.
        cumulative = (homography.astype(np.float64) if compensation_valid
                      else np.eye(3, dtype=np.float64))
        if prior_frames:
            for lag, old in enumerate(reversed(prior_frames), start=1):
                if lag > 1:
                    transform_index = len(prior_frames) - lag
                    if (not compensation_valid
                            or transform_index < 0
                            or transform_index >= len(prior_transforms)):
                        break
                    cumulative = cumulative @ prior_transforms[transform_index]
                if old.shape != gray.shape:
                    continue
                aligned = cv2.warpPerspective(
                    old, cumulative, (gray.shape[1], gray.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                valid = cv2.warpPerspective(
                    np.full_like(old, 255), cumulative,
                    (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                aligned_frames.append((aligned, valid))
        if not aligned_frames:
            aligned_frames = [(aligned_previous, valid_alignment)]
        self.last_history_count = len(aligned_frames)
        self.last_compensation_valid = compensation_valid

        differences = []
        for aligned, valid in aligned_frames:
            current_diff = cv2.absdiff(gray, aligned)
            current_diff[valid == 0] = 0
            differences.append(current_diff)
        diff = np.maximum.reduce(differences)
        # Invalid borders are already zeroed independently for each history
        # comparison. Keep pixels valid in the adjacent alignment usable.
        valid_region = valid_alignment.copy()

        if compensation_valid:
            self.history_transforms.append(homography.astype(np.float64))
        else:
            # Old cumulative transforms are unsafe after a failed fit. Start a
            # fresh adjacent pair and let the next valid fit rebuild history.
            self.history_frames.clear()
            self.history_transforms.clear()
            self.history_frames.append(previous.copy())
        self.history_frames.append(gray.copy())
        self.previous_gray = gray
        aligned_previous = aligned_frames[0][0]

        height, width = gray.shape
        edge_margin = int(round(min(height, width) * self.edge_margin_ratio))
        valid_region = valid_alignment
        if edge_margin > 0:
            valid_region[:edge_margin, :] = 0
            valid_region[-edge_margin:, :] = 0
            valid_region[:, :edge_margin] = 0
            valid_region[:, -edge_margin:] = 0
        # Local mean + 2*std threshold, approximated with box-filter moments.
        diff_f = diff.astype(np.float32)
        window = (self.threshold_window, self.threshold_window)
        mean = cv2.boxFilter(diff_f, -1, window, normalize=True)
        mean_sq = cv2.boxFilter(diff_f * diff_f, -1, window, normalize=True)
        std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
        thresholded = (diff_f > (mean + self.threshold_sigma * std +
                                 self.threshold_offset)).astype(np.uint8) * 255
        thresholded[valid_region == 0] = 0
        kernel_size = self.morph_kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = thresholded
        if self.open_iterations:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=self.open_iterations
            )
        if self.close_iterations:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iterations
            )
        # Morphology can regrow pixels into excluded frame/warp borders.
        mask[valid_region == 0] = 0
        self.last_aligned = aligned_previous.copy()
        self.last_diff = diff.copy()
        self.last_threshold = thresholded.copy()
        self.last_mask = mask.copy()

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        max_area = int(height * width * self.max_area_ratio)
        proposals: list[tuple[tuple[int, int, int, int], float]] = []
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if self.min_area <= area <= max_area and w >= 2 and h >= 2:
                # Reject components that are clipped by or touch the excluded
                # border band; these are commonly alignment/warping artifacts.
                if (x <= edge_margin or y <= edge_margin
                        or x + w >= width - edge_margin
                        or y + h >= height - edge_margin):
                    continue
                box = (int(x), int(y), int(x + w), int(y + h))
                # Paper's feature-correlation step: compare each suspicious
                # region's appearance to the last detector-confirmed target.
                # This lightweight demo uses normalized grayscale NCC in place
                # of the paper's learned feature-map correlation convolution.
                score = 0.0
                if self.target_template is not None:
                    region = gray[y:y + h, x:x + w]
                    if min(region.shape[:2]) >= 3:
                        region = cv2.resize(region, (32, 32),
                                            interpolation=cv2.INTER_AREA)
                        response = cv2.matchTemplate(
                            region, self.target_template, cv2.TM_CCOEFF_NORMED
                        )
                        score = float(response[0, 0])
                proposals.append((box, score))
        proposals.sort(key=lambda item: (
            item[1], (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])
        ), reverse=True)
        self.last_proposals = proposals[:self.max_proposals]
        candidate_view = frame.copy()
        for index, (box, _) in enumerate(self.last_proposals, 1):
            x1, y1, x2, y2 = box
            cv2.rectangle(candidate_view, (x1, y1), (x2, y2),
                          (70, 245, 170), 2, cv2.LINE_AA)
            cv2.putText(candidate_view, f"C{index}", (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 245, 170), 2,
                        cv2.LINE_AA)
        self.last_stage_images = {
            "原始输入帧": frame.copy(),
            "预处理灰度帧": gray.copy(),
                "光流特征点与相机补偿": flow_visual,
                "对齐后上一帧": aligned_previous.copy(),
            "帧差响应": diff.copy(),
            "阈值分割结果": thresholded.copy(),
            "形态学清理后掩膜": mask.copy(),
            "连通区域候选框": candidate_view,
        }
        return self.last_proposals

    def process(self, frame: np.ndarray) -> MotionResult:
        proposals = self.propose(frame)
        mask = self.last_mask
        return MotionResult(
            proposals=proposals,
            mask=mask,
            diff=self.last_diff,
            aligned_previous=self.last_aligned,
            threshold=self.last_threshold,
            stages=self.last_stage_images,
            diagnostics={
                "algorithm": self.algorithm_id,
                "candidate_count": len(proposals),
                "history_window": self.history_window,
                "history_count": self.last_history_count,
                "compensation_valid": self.last_compensation_valid,
                "foreground_ratio": (
                    float(np.count_nonzero(mask) / mask.size)
                    if mask is not None and mask.size else 0.0
                ),
            },
        )

    def reset(self) -> None:
        self.previous_gray = None
        self.history_frames.clear()
        self.history_transforms.clear()
        self.target_template = None
        self.last_aligned = None
        self.last_diff = None
        self.last_threshold = None
        self.last_mask = None
        self.last_proposals = []
        self.last_stage_images = {}
        self.last_history_count = 0
        self.last_compensation_valid = False


class PipelineState(Enum):
    GLOBAL_SEARCH = auto()
    LOCAL_DETECT = auto()
    MOTION_EXTRACT = auto()
    MOTION_VERIFY = auto()


class MSMVideoPipeline:
    def __init__(self, detector: DetectorBackend, roi_size: int = 640,
                 global_conf: float = 0.60, local_conf: float = 0.60,
                 global_interval: int = 15, imgsz: int = 640,
                 iou_threshold: float = 0.45,
                 motion_config: dict | None = None,
                 motion_algorithm: MotionAlgorithm | None = None,
                 trajectory_filtering: bool = True):
        self.detector = detector
        self.roi_size = roi_size
        self.global_conf = global_conf
        self.local_conf = local_conf
        self.global_interval = max(1, global_interval)
        self.imgsz = imgsz
        self.iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
        self.motion = motion_algorithm or MotionProposer(**(motion_config or {}))
        self.trajectory_filtering = bool(trajectory_filtering)
        self.roi: tuple[int, int, int, int] | None = None
        self.frame_number = 0
        self.state = PipelineState.GLOBAL_SEARCH
        self.mode = self.state.name
        self.motion_misses = 0
        self.verify_misses = 0
        self.verify_miss_limit = 3
        # Associate motion proposals over time instead of independently picking
        # the largest connected component on each frame.
        self.motion_anchor_box: tuple[int, int, int, int] | None = None
        self.motion_velocity = (0.0, 0.0)
        self.rejected_candidate_box: tuple[int, int, int, int] | None = None
        self.rejected_candidate_until = -1
        self.candidate_history: deque[bool] = deque(maxlen=3)
        self.candidate_misses = 0
        self.candidate_miss_limit = 5
        self.candidate_confirm_hits = 2
        self.preview_roi: tuple[int, int, int, int] | None = None
        self.preview_roi_status = "SEARCHING"
        self.state_frame_counts: Counter[str] = Counter()
        self.state_transition_counts: Counter[str] = Counter()
        self.motion_candidate_frames = 0
        self.motion_candidate_total = 0
        self.detection_frames = 0
        self.last_frame_timing = {"motion_ms": 0.0, "detector_ms": 0.0,
                                  "global_detector_ms": 0.0, "roi_detector_ms": 0.0,
                                  "pipeline_ms": 0.0}

    def _predict(self, image: np.ndarray, confidence: float,
                 role: str = "global") -> list[Detection]:
        started = time.perf_counter()
        try:
            return self.detector.predict(
                image, confidence, self.imgsz, self.iou_threshold
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.last_frame_timing["detector_ms"] += elapsed_ms
            key = "roi_detector_ms" if role == "roi" else "global_detector_ms"
            self.last_frame_timing[key] += elapsed_ms

    def _centered_roi(self, center_x: float, center_y: float,
                      shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        side_w, side_h = min(self.roi_size, w), min(self.roi_size, h)
        x1, y1 = center_x - side_w / 2, center_y - side_h / 2
        x1 = min(max(0, x1), w - side_w)
        y1 = min(max(0, y1), h - side_h)
        return int(x1), int(y1), int(x1 + side_w), int(y1 + side_h)

    def _detect_crop(self, frame: np.ndarray,
                     crop: tuple[int, int, int, int], confidence: float
                     ) -> list[Detection]:
        x1, y1, x2, y2 = crop
        detections = self._predict(frame[y1:y2, x1:x2], confidence, role="roi")
        return [Detection(d.x1 + x1, d.y1 + y1, d.x2 + x1, d.y2 + y1,
                          d.confidence, d.class_id) for d in detections]

    @staticmethod
    def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def _select_motion_candidate(
        self, proposal_scores: list[tuple[tuple[int, int, int, int], float]]
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """Choose the proposal closest to the predicted prior-candidate location."""
        if not proposal_scores:
            return None
        if not self.trajectory_filtering:
            return proposal_scores[0]
        if (self.rejected_candidate_box is not None
                and self.frame_number > self.rejected_candidate_until):
            self.rejected_candidate_box = None
        eligible = proposal_scores
        if self.rejected_candidate_box is not None:
            rejected_x, rejected_y = self._box_center(self.rejected_candidate_box)
            reject_radius = max(48.0, min(float(self.roi_size) * 0.15, 100.0))
            eligible = [
                item for item in proposal_scores
                if np.hypot(self._box_center(item[0])[0] - rejected_x,
                            self._box_center(item[0])[1] - rejected_y) > reject_radius
            ]
        if not eligible:
            return None
        if self.motion_anchor_box is None:
            return eligible[0]

        anchor_x, anchor_y = self._box_center(self.motion_anchor_box)
        predicted_x = anchor_x + self.motion_velocity[0]
        predicted_y = anchor_y + self.motion_velocity[1]
        gate = max(64.0, min(float(self.roi_size) * 0.25, 180.0))
        anchor_w = max(1, self.motion_anchor_box[2] - self.motion_anchor_box[0])
        anchor_h = max(1, self.motion_anchor_box[3] - self.motion_anchor_box[1])
        best = None
        best_score = float("-inf")
        for box, appearance in eligible:
            center_x, center_y = self._box_center(box)
            distance = float(np.hypot(center_x - predicted_x, center_y - predicted_y))
            if distance > gate:
                continue
            width = max(1, box[2] - box[0])
            height = max(1, box[3] - box[1])
            area_ratio = (width * height) / float(anchor_w * anchor_h)
            size_penalty = min(abs(float(np.log(max(area_ratio, 1e-3)))), 2.0) / 2.0
            distance_score = 1.0 - distance / gate
            appearance_score = (max(-1.0, min(1.0, appearance)) + 1.0) / 2.0
            score = (0.65 * distance_score + 0.20 * appearance_score
                     + 0.15 * (1.0 - size_penalty))
            if score > best_score:
                best = (box, appearance)
                best_score = score
        return best

    def _update_motion_anchor(self, box: tuple[int, int, int, int]) -> None:
        if self.motion_anchor_box is not None:
            old_x, old_y = self._box_center(self.motion_anchor_box)
            new_x, new_y = self._box_center(box)
            # Bound one-frame motion so a single bad association cannot make
            # the next prediction gate leap across the image.
            dx = max(-60.0, min(60.0, new_x - old_x))
            dy = max(-60.0, min(60.0, new_y - old_y))
            self.motion_velocity = (
                0.5 * self.motion_velocity[0] + 0.5 * dx,
                0.5 * self.motion_velocity[1] + 0.5 * dy,
            )
        else:
            self.motion_velocity = (0.0, 0.0)
        self.motion_anchor_box = box

    def _candidate_roi(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.motion_anchor_box is None:
            return None
        center_x, center_y = self._box_center(self.motion_anchor_box)
        return self._centered_roi(center_x, center_y, frame.shape)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        pipeline_started = time.perf_counter()
        self.last_frame_timing = {"motion_ms": 0.0, "detector_ms": 0.0,
                                  "global_detector_ms": 0.0, "roi_detector_ms": 0.0,
                                  "pipeline_ms": 0.0}
        self.frame_number += 1
        height, width = frame.shape[:2]
        self.preview_roi = self.roi
        self.preview_roi_status = "TARGET ROI" if self.roi is not None else "SEARCHING"
        motion_started = time.perf_counter()
        self.motion_result = self.motion.process(frame)
        self.last_frame_timing["motion_ms"] = (
            time.perf_counter() - motion_started
        ) * 1000.0
        proposal_scores = self.motion_result.proposals
        proposals = [item[0] for item in proposal_scores]
        self.state_frame_counts[self.state.name] += 1
        if proposals:
            self.motion_candidate_frames += 1
            self.motion_candidate_total += len(proposals)
        detections: list[Detection] = []
        next_state = self.state

        if self.state is PipelineState.GLOBAL_SEARCH:
            self.mode = "GLOBAL_SEARCH"
            detections = self._predict(frame, self.global_conf)
            if detections:
                next_state = PipelineState.LOCAL_DETECT
            else:
                next_state = PipelineState.MOTION_EXTRACT

        elif self.state is PipelineState.LOCAL_DETECT:
            self.mode = "LOCAL_DETECT"
            if self.roi is None:
                next_state = PipelineState.GLOBAL_SEARCH
            else:
                detections = self._detect_crop(frame, self.roi, self.local_conf)
                if detections:
                    next_state = PipelineState.LOCAL_DETECT
                else:
                    self.preview_roi = self.roi
                    self.preview_roi_status = "TARGET LOST · REACQUIRE"
                    self.roi = None
                    next_state = PipelineState.MOTION_EXTRACT
            # Periodic full-frame reacquisition limits drift and handles fast motion.
            if self.frame_number % self.global_interval == 0:
                global_dets = self._predict(frame, self.global_conf)
                if global_dets:
                    detections = global_dets
                    next_state = PipelineState.LOCAL_DETECT

        elif self.state is PipelineState.MOTION_EXTRACT:
            self.mode = "MOTION_EXTRACT_3.3"
            candidate = self._select_motion_candidate(proposal_scores)
            if candidate is not None:
                best_box, best_score = candidate
                self._update_motion_anchor(best_box)
                self.candidate_misses = 0
                self.motion_misses = 0
                self.candidate_history.append(True)
                candidate_roi = self._candidate_roi(frame)
                self.preview_roi = candidate_roi
                hits = sum(self.candidate_history)
                self.preview_roi_status = f"CANDIDATE · CONTINUITY {hits}/3"
                appearance_ok = (getattr(self.motion, "target_template", None) is None
                                 or best_score >= 0.10)
                if appearance_ok and hits >= self.candidate_confirm_hits:
                    self.roi = candidate_roi
                    self.verify_misses = 0
                    self.preview_roi_status = "CANDIDATE · YOLO VERIFY"
                    next_state = PipelineState.MOTION_VERIFY
                else:
                    next_state = PipelineState.MOTION_EXTRACT
            else:
                self.candidate_history.append(False)
                self.candidate_misses += 1
                self.motion_misses += 1
                if not self.trajectory_filtering:
                    self.motion_anchor_box = None
                    self.motion_velocity = (0.0, 0.0)
                    self.candidate_history.clear()
                    self.candidate_misses = 0
                self.preview_roi = self._candidate_roi(frame)
                if self.preview_roi is not None and self.candidate_misses < self.candidate_miss_limit:
                    self.preview_roi_status = (
                        f"CANDIDATE · HOLD {self.candidate_misses}/"
                        f"{self.candidate_miss_limit}"
                    )
                else:
                    self.preview_roi_status = "SEARCHING · NO CONTINUOUS CANDIDATE"
                    self.motion_anchor_box = None
                    self.motion_velocity = (0.0, 0.0)
                    self.candidate_history.clear()
            # Reacquire globally only after a longer candidate-loss interval.
            if self.motion_misses >= self.global_interval:
                next_state = PipelineState.GLOBAL_SEARCH
                self.motion_misses = 0

        elif self.state is PipelineState.MOTION_VERIFY:
            self.mode = "MOTION_VERIFY"
            candidate = self._select_motion_candidate(proposal_scores)
            if candidate is not None:
                self._update_motion_anchor(candidate[0])
                self.candidate_misses = 0
                tracked_roi = self._candidate_roi(frame)
                if tracked_roi is not None:
                    self.roi = tracked_roi
            candidate_roi = self.roi
            if self.roi is not None:
                detections = self._detect_crop(frame, self.roi, self.local_conf)
            if detections:
                self.verify_misses = 0
                self.motion_misses = 0
                next_state = PipelineState.LOCAL_DETECT
            else:
                self.verify_misses += 1
                self.preview_roi = candidate_roi
                if self.verify_misses < self.verify_miss_limit:
                    self.preview_roi_status = (
                        f"CANDIDATE · YOLO VERIFY {self.verify_misses}/"
                        f"{self.verify_miss_limit}"
                    )
                    next_state = PipelineState.MOTION_VERIFY
                else:
                    self.preview_roi_status = "ROI REJECTED · YOLO MISS"
                    self.rejected_candidate_box = self.motion_anchor_box
                    self.rejected_candidate_until = (
                        self.frame_number + max(self.global_interval, 15)
                    )
                    self.motion_anchor_box = None
                    self.motion_velocity = (0.0, 0.0)
                    self.roi = None
                    self.verify_misses = 0
                    self.candidate_history.clear()
                    next_state = PipelineState.MOTION_EXTRACT

        # A detector-confirmed box seeds the next-frame ROI and appearance template.
        if detections:
            self.detection_frames += 1
            best = max(detections, key=lambda det: det.confidence)
            self.motion.update_target_template(frame, best)
            self._update_motion_anchor((best.x1, best.y1, best.x2, best.y2))
            self.candidate_history.clear()
            self.candidate_misses = 0
            self.roi = self._centered_roi(
                (best.x1 + best.x2) / 2, (best.y1 + best.y2) / 2, frame.shape
            )
            self.preview_roi = self.roi
            self.preview_roi_status = "YOLO CONFIRMED · TARGET ROI"
            self.motion_misses = 0
        elif next_state is PipelineState.GLOBAL_SEARCH:
            self.roi = None
            self.motion_anchor_box = None
            self.motion_velocity = (0.0, 0.0)
            self.candidate_history.clear()
        if next_state is not self.state:
            transition = f"{self.state.name}->{next_state.name}"
            self.state_transition_counts[transition] += 1
        self.state = next_state

        output = frame.copy()
        for x1, y1, x2, y2 in proposals:
            cv2.rectangle(output, (x1, y1), (x2, y2), (80, 180, 255), 1)
        for det in detections:
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2),
                          (40, 230, 70), 2)
            class_name = self.detector.names.get(det.class_id, str(det.class_id))
            cv2.putText(output, f"{class_name} {det.confidence:.2f}",
                        (det.x1, max(18, det.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 230, 70), 1,
                        cv2.LINE_AA)
        if self.preview_roi is not None:
            x1, y1, x2, y2 = self.preview_roi
            color = ((230, 80, 255) if "CONFIRMED" in self.preview_roi_status
                     else (0, 165, 255) if "REJECTED" in self.preview_roi_status
                     else (255, 190, 30))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            cv2.putText(output, self.preview_roi_status,
                        (x1, max(22, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color, 2, cv2.LINE_AA)
        cv2.putText(output, f"MSM | {self.mode} | frame {self.frame_number}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        self.last_frame_timing["pipeline_ms"] = (
            time.perf_counter() - pipeline_started
        ) * 1000.0
        return output, detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MSM-YOLO 论文框架轻量演示：全图检测 + 运动候选 + 局部 ROI"
    )
    parser.add_argument("--weights", required=True,
                        help="YOLOv5 checkpoint 路径（例如 best.pt）")
    parser.add_argument("--source", default="0",
                        help="视频文件路径或摄像头编号，默认 0")
    parser.add_argument("--output", default="runs/msm_demo_bytetrack.mp4",
                        help="输出视频路径；传入空字符串可关闭保存")
    parser.add_argument("--device", default="0",
                        help="YOLOv5 推理设备，默认 GPU 0；CPU 请显式指定 cpu")
    parser.add_argument("--roi-size", type=int, default=640,
                        help="局部 ROI 边长，默认 640 像素")
    parser.add_argument("--global-conf", type=float, default=0.60,
                        help="全图检测置信度阈值")
    parser.add_argument("--local-conf", type=float, default=0.60,
                        help="局部 ROI/运动候选复核置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值，默认 0.45")
    parser.add_argument("--global-interval", type=int, default=15,
                        help="每隔多少帧执行一次全图重捕获")
    parser.add_argument("--motion-history", type=int, choices=range(2, 6),
                        default=3, help="GMC 运动残差历史窗口，默认 3 帧")
    parser.add_argument("--track-buffer", type=int, default=30,
                        help="ByteTrack 保留丢失轨迹的最大帧数")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="检测器输入尺寸")
    parser.add_argument("--no-display", action="store_true",
                        help="不打开实时预览窗口")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开输入源：{args.source}")
    detector = YOLOv5Detector(args.weights, args.device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    tracker = ByteTrackAdapter(args.global_conf, fps, args.track_buffer)
    pipeline = MSMVideoPipeline(
        detector, args.roi_size, args.global_conf, args.local_conf,
        args.global_interval, args.imgsz, iou_threshold=args.iou,
        motion_config={"history_window": args.motion_history},
    )
    writer = None
    output_path = Path(args.output) if args.output else None
    start = time.perf_counter()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, detections = pipeline.process(frame)
            tracks = tracker.update(detections, frame.shape)
            for track in tracks:
                color = (255, 80, 220)
                cv2.rectangle(annotated, (track.x1, track.y1),
                              (track.x2, track.y2), color, 2)
                cv2.putText(
                    annotated, f"ID {track.track_id}",
                    (track.x1, max(18, track.y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
                )
            frames += 1
            if frames % 300 == 0:
                elapsed_now = time.perf_counter() - start
                print(
                    f"进度：{frames} 帧；状态：{pipeline.mode}；"
                    f"平均 {frames / elapsed_now:.2f} FPS",
                    flush=True,
                )
            if output_path:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height),
                    )
                    if not writer.isOpened():
                        raise SystemExit(f"无法创建输出视频：{output_path}")
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("MSM-YOLO framework demo", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - start
    print(f"处理帧数：{frames}；平均处理速度：{frames / elapsed:.2f} FPS")
    print(f"状态帧数：{dict(pipeline.state_frame_counts)}")
    print(f"状态转换：{dict(pipeline.state_transition_counts)}")
    print(
        f"运动候选：{pipeline.motion_candidate_frames} 帧有候选，"
        f"共 {pipeline.motion_candidate_total} 个候选框；"
        f"检测器命中 {pipeline.detection_frames} 帧"
    )
    print(
        f"ByteTrack：{tracker.track_frame_count} 帧有活跃轨迹；"
        f"本段视频分配 {len(tracker.seen_track_ids)} 个唯一 ID"
    )
    if output_path:
        print(f"结果视频：{output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
