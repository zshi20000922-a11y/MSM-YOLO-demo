from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """Detection coordinates are always xyxy in the input frame's pixels."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


@dataclass(frozen=True)
class Track:
    """Tracker output using the same original-frame xyxy coordinate space."""

    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    confidence: float
    class_id: int


@dataclass
class MotionResult:
    """One algorithm's candidate and diagnostics for a single input frame."""

    proposals: list[tuple[Box, float]] = field(default_factory=list)
    mask: np.ndarray | None = None
    diff: np.ndarray | None = None
    aligned_previous: np.ndarray | None = None
    threshold: np.ndarray | None = None
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    path: str
    backend: str
    framework_version: str
    device: str
    task: str = "detect"
    sha256: str | None = None


@dataclass(frozen=True)
class DetectionTraceSnapshot:
    """One model forward pass plus the tensors needed for threshold-only replay."""

    backend: str
    raw_prediction: Any
    resized_image: np.ndarray
    normalized_image: np.ndarray
    original_shape: tuple[int, ...]
    input_shape: tuple[int, ...]
    tensor_range: tuple[float, float]
    end2end: bool = False


@dataclass(frozen=True)
class DetectionTrace:
    """Visualizable detection stages derived from a cached forward pass."""

    candidates: np.ndarray
    confidence_filtered: np.ndarray
    nms: np.ndarray
    final: np.ndarray
    confidence_threshold: float
    iou_threshold: float
    end2end: bool = False
