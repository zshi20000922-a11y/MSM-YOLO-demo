from __future__ import annotations

from typing import Protocol

import numpy as np

from .contracts import Detection, DetectionTrace, DetectionTraceSnapshot, MotionResult, Track


class MotionAlgorithm(Protocol):
    algorithm_id: str
    display_name: str

    def process(self, frame: np.ndarray) -> MotionResult: ...
    def reset(self) -> None: ...
    def update_target_template(self, frame: np.ndarray, detection: Detection) -> None: ...


class DetectorBackend(Protocol):
    backend_id: str
    names: dict[int, str]

    def predict(self, image: np.ndarray, confidence: float,
                imgsz: int, iou: float = 0.45) -> list[Detection]: ...
    def trace_inference(self, image: np.ndarray,
                        imgsz: int) -> DetectionTraceSnapshot: ...
    def trace_postprocess(self, snapshot: DetectionTraceSnapshot,
                          confidence: float, iou: float) -> DetectionTrace: ...
    def close(self) -> None: ...


class TrackerBackend(Protocol):
    backend_id: str

    def update(self, detections: list[Detection],
               frame_shape: tuple[int, ...]) -> list[Track]: ...
    def reset(self) -> None: ...
