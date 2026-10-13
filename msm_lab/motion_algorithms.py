from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .contracts import Box, Detection, MotionResult


class _OpenCVMotionAlgorithm:
    algorithm_id = "base"
    display_name = "OpenCV motion algorithm"

    def __init__(self, *, min_area: int = 12, max_area_ratio: float = 0.12,
                 edge_margin_ratio: float = 0.03, morph_kernel: int = 3,
                 open_iterations: int = 1, close_iterations: int = 2,
                 max_proposals: int = 20):
        self.min_area = max(1, int(min_area))
        self.max_area_ratio = max(0.001, min(float(max_area_ratio), 0.9))
        self.edge_margin_ratio = max(0.0, min(float(edge_margin_ratio), 0.15))
        self.morph_kernel = max(1, int(morph_kernel) | 1)
        self.open_iterations = max(0, int(open_iterations))
        self.close_iterations = max(0, int(close_iterations))
        self.max_proposals = max(1, int(max_proposals))
        self.last_diff: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_threshold: np.ndarray | None = None
        self.last_aligned: np.ndarray | None = None
        self.last_proposals: list[tuple[Box, float]] = []

    def reset(self) -> None:
        self.last_diff = None
        self.last_mask = None
        self.last_threshold = None
        self.last_aligned = None
        self.last_proposals = []

    def update_target_template(self, frame: np.ndarray,
                               detection: Detection) -> None:
        # Appearance correlation is implemented by the flow+frame-difference
        # legacy algorithm. Other algorithms rely on temporal association and YOLO.
        return None

    def _edge_margin(self, shape: tuple[int, ...]) -> int:
        height, width = shape[:2]
        return int(round(min(height, width) * self.edge_margin_ratio))

    def _extract(self, frame: np.ndarray, diff: np.ndarray,
                 thresholded: np.ndarray,
                 valid_region: np.ndarray | None = None,
                 preprocessed: np.ndarray | None = None,
                 method_stage: str = "帧差") -> MotionResult:
        height, width = frame.shape[:2]
        edge = self._edge_margin(frame.shape)
        if valid_region is None:
            valid_region = np.full((height, width), 255, np.uint8)
        else:
            valid_region = valid_region.copy()
        if edge:
            valid_region[:edge, :] = 0
            valid_region[-edge:, :] = 0
            valid_region[:, :edge] = 0
            valid_region[:, -edge:] = 0
        threshold_preview = thresholded.copy()
        thresholded = thresholded.copy()
        thresholded[valid_region == 0] = 0
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        mask = thresholded
        if self.open_iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=self.open_iterations)
        if self.close_iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=self.close_iterations)
        mask[valid_region == 0] = 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        max_area = int(height * width * self.max_area_ratio)
        proposals: list[tuple[Box, float]] = []
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if not (self.min_area <= area <= max_area and w >= 2 and h >= 2):
                continue
            if (x <= edge or y <= edge or x + w >= width - edge
                    or y + h >= height - edge):
                continue
            proposals.append(((int(x), int(y), int(x + w), int(y + h)), 0.0))
        proposals.sort(
            key=lambda item: (item[0][2] - item[0][0]) *
                             (item[0][3] - item[0][1]),
            reverse=True,
        )
        self.last_diff = diff.copy()
        self.last_threshold = thresholded.copy()
        self.last_mask = mask.copy()
        self.last_aligned = frame.copy()
        self.last_proposals = proposals[:self.max_proposals]
        candidate_view = frame.copy()
        for index, (box, _) in enumerate(self.last_proposals, 1):
            x1, y1, x2, y2 = box
            cv2.rectangle(candidate_view, (x1, y1), (x2, y2),
                          (70, 245, 170), 2, cv2.LINE_AA)
            cv2.putText(candidate_view, f"C{index}", (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 245, 170), 2,
                        cv2.LINE_AA)
        stages = {
            "原始输入帧": frame.copy(),
            "预处理灰度帧": (preprocessed.copy() if preprocessed is not None
                              else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
            f"{method_stage}响应": diff.copy(),
            "阈值分割结果": threshold_preview,
            "形态学清理后掩膜": mask.copy(),
            "连通区域候选框": candidate_view,
        }
        return MotionResult(
            proposals=self.last_proposals,
            mask=self.last_mask,
            diff=self.last_diff,
            aligned_previous=self.last_aligned,
            threshold=self.last_threshold,
            stages=stages,
            diagnostics={
                "algorithm": self.algorithm_id,
                "component_count": count - 1,
                "candidate_count": len(self.last_proposals),
                "foreground_ratio": float(np.count_nonzero(mask) / mask.size),
                "edge_margin_px": edge,
            },
        )

    def process(self, frame: np.ndarray) -> MotionResult:
        raise NotImplementedError


class FrameDifferenceAlgorithm(_OpenCVMotionAlgorithm):
    algorithm_id = "frame_difference"
    display_name = "帧差法"

    def __init__(self, **kwargs):
        base_options = {
            key: kwargs[key] for key in (
                "min_area", "max_area_ratio", "edge_margin_ratio",
                "morph_kernel", "open_iterations", "close_iterations",
                "max_proposals",
            ) if key in kwargs
        }
        super().__init__(**base_options)
        self.threshold_sigma = float(kwargs.get("threshold_sigma", 2.0))
        self.threshold_offset = float(kwargs.get("threshold_offset", 8.0))
        self.threshold_window = max(3, int(kwargs.get("threshold_window", 21)) | 1)
        self.previous_gray: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.previous_gray = None

    def process(self, frame: np.ndarray) -> MotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous = self.previous_gray
        self.previous_gray = gray
        if previous is None or previous.shape != gray.shape:
            return self._extract(frame, np.zeros_like(gray), np.zeros_like(gray),
                                 preprocessed=gray)
        diff = cv2.absdiff(gray, previous)
        diff_f = diff.astype(np.float32)
        window = (self.threshold_window, self.threshold_window)
        mean = cv2.boxFilter(diff_f, -1, window, normalize=True)
        mean_sq = cv2.boxFilter(diff_f * diff_f, -1, window, normalize=True)
        std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
        threshold = (diff_f > (mean + self.threshold_sigma * std +
                               self.threshold_offset)).astype(np.uint8) * 255
        return self._extract(frame, diff, threshold, preprocessed=gray)


class MOG2Algorithm(_OpenCVMotionAlgorithm):
    algorithm_id = "mog2"
    display_name = "MOG2 背景减除"

    def __init__(self, *, history: int = 500, var_threshold: float = 16.0,
                 detect_shadows: bool = True, **kwargs):
        base_options = {
            key: kwargs[key] for key in (
                "min_area", "max_area_ratio", "edge_margin_ratio",
                "morph_kernel", "open_iterations", "close_iterations",
                "max_proposals",
            ) if key in kwargs
        }
        super().__init__(**base_options)
        self.history = max(1, int(history))
        self.var_threshold = max(1.0, float(var_threshold))
        self.detect_shadows = bool(detect_shadows)
        self.frames_processed = 0
        self._create_subtractor()

    def _create_subtractor(self) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def reset(self) -> None:
        super().reset()
        self.frames_processed = 0
        self._create_subtractor()

    def process(self, frame: np.ndarray) -> MotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        foreground = self.subtractor.apply(gray)
        # OpenCV MOG2 encodes shadows as 127; retain only definite foreground.
        threshold = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)[1]
        self.frames_processed += 1
        if self.frames_processed <= 2:
            threshold.fill(0)
        return self._extract(frame, foreground, threshold, preprocessed=gray,
                             method_stage="MOG2 前景")


class FlowFrameDifferenceAdapter:
    algorithm_id = "flow_frame_difference"
    display_name = "光流 + 帧差（当前算法）"

    def __init__(self, config: dict):
        from msm_yolo_demo import MotionProposer

        supported = {
            key: value for key, value in config.items()
            if key in {
                "min_area", "max_area_ratio", "threshold_sigma",
                "threshold_offset", "threshold_window", "morph_kernel",
                "open_iterations", "close_iterations", "max_proposals",
                "edge_margin_ratio", "history_window",
            }
        }
        self.impl = MotionProposer(**supported)

    def __getattr__(self, name):
        return getattr(self.impl, name)

    def reset(self) -> None:
        self.impl.reset()

    def update_target_template(self, frame: np.ndarray,
                               detection: Detection) -> None:
        self.impl.update_target_template(frame, detection)

    def process(self, frame: np.ndarray) -> MotionResult:
        proposals = self.impl.propose(frame)
        diagnostics = {
            "algorithm": self.algorithm_id,
            "candidate_count": len(proposals),
            "history_window": self.impl.history_window,
            "history_count": self.impl.last_history_count,
            "compensation_valid": self.impl.last_compensation_valid,
            "foreground_ratio": (
                float(np.count_nonzero(self.impl.last_mask) / self.impl.last_mask.size)
                if self.impl.last_mask is not None and self.impl.last_mask.size else 0.0
            ),
        }
        return MotionResult(
            proposals=proposals,
            mask=self.impl.last_mask,
            diff=self.impl.last_diff,
            aligned_previous=self.impl.last_aligned,
            threshold=self.impl.last_threshold,
            stages=self.impl.last_stage_images,
            diagnostics=diagnostics,
        )


MOTION_ALGORITHMS: dict[str, tuple[str, Callable[[dict], object]]] = {
    "flow_frame_difference": ("光流 + 帧差（当前算法）", FlowFrameDifferenceAdapter),
    "frame_difference": ("帧差法", lambda config: FrameDifferenceAlgorithm(**config)),
    "mog2": ("MOG2 背景减除", lambda config: MOG2Algorithm(**config)),
}


def create_motion_algorithm(algorithm_id: str, config: dict | None = None):
    try:
        _, factory = MOTION_ALGORITHMS[algorithm_id]
    except KeyError as exc:
        raise ValueError(f"未知运动算法：{algorithm_id}") from exc
    normalized = dict(config or {})
    normalized.setdefault("history", normalized.get("mog2_history", 500))
    normalized.setdefault("var_threshold", normalized.get("mog2_var_threshold", 16.0))
    normalized.setdefault("detect_shadows", normalized.get("mog2_detect_shadows", True))
    return factory(normalized)


def motion_algorithm_choices() -> list[tuple[str, str]]:
    return [(algorithm_id, value[0])
            for algorithm_id, value in MOTION_ALGORITHMS.items()]
