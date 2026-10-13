from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import Detection, DetectionTrace, DetectionTraceSnapshot


class UltralyticsDetector:
    """Detection-only adapter for the installed Ultralytics model API."""

    backend_id = "ultralytics"

    def __init__(self, weights: str | Path, device: str | None = None):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics 后端不可用：当前 Python 环境需要安装 torch 与 ultralytics。"
            ) from exc
        self.torch = torch
        self.device_arg = device or "0"
        if str(self.device_arg).lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("选择了 GPU，但当前 PyTorch 未检测到可用 CUDA 设备。")
            device_index = int(str(self.device_arg).split(":")[-1])
            self.device = torch.device(f"cuda:{device_index}")
            try:
                probe = torch.ones((2, 2), device=self.device)
                _ = probe @ probe
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise RuntimeError(f"Ultralytics 后端 GPU 检查失败：{exc}") from exc
        self.weights = str(Path(weights).expanduser().resolve())
        try:
            self.model = YOLO(self.weights, task="detect")
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics 后端无法加载此权重。若它来自原版 YOLOv5 仓库，"
                "请改用‘自动识别’或‘原版 YOLOv5’后端；其他自定义结构需提供对应模型代码。"
            ) from exc
        self.names = self._normalize_names(getattr(self.model, "names", {}))
        self.version = self._version()
        self._trace_snapshot = None
        self._trace_candidates = None

    @staticmethod
    def _normalize_names(names) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    @staticmethod
    def _version() -> str:
        try:
            from importlib.metadata import version
            return version("ultralytics")
        except Exception:
            return "unknown"

    def predict(self, image: np.ndarray, confidence: float,
                imgsz: int, iou: float = 0.45) -> list[Detection]:
        try:
            results = self.model.predict(
                source=image,
                conf=float(confidence),
                iou=float(iou),
                imgsz=int(imgsz),
                device=self.device_arg,
                verbose=False,
                task="detect",
            )
        except Exception as exc:
            raise RuntimeError(f"Ultralytics 推理失败：{exc}") from exc
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            height, width = image.shape[:2]
            for box, score, class_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = [int(round(float(value))) for value in box]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))
                detections.append(Detection(
                    x1, y1, x2, y2, float(score), int(class_id)
                ))
        return detections

    def trace_inference(self, image: np.ndarray,
                        imgsz: int) -> DetectionTraceSnapshot:
        """Run one forward pass and retain only its raw predictions for replay."""
        from ultralytics.utils import nms
        from ultralytics.models.yolo.detect import DetectionPredictor

        # Build Ultralytics' official predictor around the already-loaded model;
        # unlike calling model.predict first, this avoids a discarded extra pass.
        predictor = DetectionPredictor(overrides={
            "model": self.weights, "task": "detect", "mode": "predict",
            "conf": 0.001, "iou": 1.0, "imgsz": int(imgsz),
            "device": self.device_arg, "verbose": False,
        }, _callbacks=self.model.callbacks)
        predictor.setup_model(model=self.model.model, verbose=False)
        predictor.setup_source(source=[image])
        batch = predictor.preprocess([image])
        with self.torch.inference_mode():
            raw = predictor.inference(batch)
        tensor_image = batch[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        normalized = np.clip(tensor_image[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
        transformed = predictor.pre_transform([image])[0]
        self._trace_predictor = predictor
        snapshot = DetectionTraceSnapshot(
            backend=self.backend_id,
            raw_prediction=raw,
            resized_image=transformed.copy(),
            normalized_image=normalized,
            original_shape=tuple(image.shape),
            input_shape=tuple(batch.shape),
            tensor_range=(float(batch.min().item()), float(batch.max().item())),
            end2end=bool(getattr(predictor.model, "end2end", False)),
        )
        self._trace_snapshot = snapshot
        self._trace_candidates = None
        return snapshot

    def trace_postprocess(self, snapshot: DetectionTraceSnapshot,
                          confidence: float, iou: float) -> DetectionTrace:
        from ultralytics.utils import nms, ops

        raw = snapshot.raw_prediction
        end2end = snapshot.end2end
        common = {"max_det": 1000, "nc": len(self.names), "end2end": end2end}
        if self._trace_snapshot is snapshot and self._trace_candidates is not None:
            candidates = self._trace_candidates
        else:
            candidates = nms.non_max_suppression(
                self._clone_prediction(raw), 0.001, 1.0,
                max_det=3000, nc=len(self.names), end2end=end2end,
                max_time_img=2.0, max_nms=3000,
            )[0]
            self._trace_snapshot = snapshot
            self._trace_candidates = candidates
        filtered = (candidates[candidates[:, 4] >= float(confidence)]
                    if len(candidates) else candidates)
        kept = nms.non_max_suppression(
            self._clone_prediction(raw), float(confidence), float(iou), **common,
            max_time_img=1.0,
        )[0]
        final = kept.clone()
        if len(final):
            final[:, :4] = ops.scale_boxes(
                tuple(snapshot.input_shape[2:]), final[:, :4], snapshot.original_shape
            ).round()
        arrays = [item.detach().cpu().numpy() for item in (candidates, filtered, kept, final)]
        return DetectionTrace(*arrays, float(confidence), float(iou), end2end)

    def trace(self, image: np.ndarray, confidence: float, imgsz: int,
              iou: float = 0.45) -> dict:
        snapshot = self.trace_inference(image, imgsz)
        stages = self.trace_postprocess(snapshot, confidence, iou)
        return self._trace_dict(snapshot, stages)

    @staticmethod
    def _clone_prediction(raw):
        if isinstance(raw, tuple):
            return tuple(item.clone() if hasattr(item, "clone") else item for item in raw)
        if isinstance(raw, list):
            return [item.clone() if hasattr(item, "clone") else item for item in raw]
        return raw.clone() if hasattr(raw, "clone") else raw

    @staticmethod
    def _trace_dict(snapshot: DetectionTraceSnapshot, stages: DetectionTrace) -> dict:
        return {
            "resized": snapshot.resized_image,
            "normalized": snapshot.normalized_image,
            "candidates": stages.candidates,
            "confidence_filtered": stages.confidence_filtered,
            "nms": stages.nms,
            "final": stages.final,
            "input_shape": snapshot.input_shape,
            "tensor_range": snapshot.tensor_range,
            "backend": snapshot.backend,
            "end2end": stages.end2end,
        }

    def close(self) -> None:
        self._trace_predictor = None
        self._trace_snapshot = None
        self._trace_candidates = None
        self.model = None


def create_detector(weights: str | Path, device: str | None = None,
                    backend: str = "auto"):
    weights_path = Path(weights).expanduser()
    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights_path}")
    if weights_path.suffix.lower() != ".pt":
        raise ValueError("首期检测后端仅支持 .pt 权重文件。")
    if backend not in {"auto", "legacy_yolov5", "ultralytics"}:
        raise ValueError(f"未知检测后端：{backend}")
    if backend in {"auto", "ultralytics"}:
        try:
            return UltralyticsDetector(weights_path, device)
        except Exception as ultralytics_error:
            if backend == "ultralytics":
                raise
            ultralytics_message = str(ultralytics_error)
    else:
        ultralytics_message = "用户选择了原版 YOLOv5 后端"

    try:
        from msm_yolo_demo import YOLOv5Detector
        detector = YOLOv5Detector(str(weights_path), device)
        detector.backend_id = "legacy_yolov5"
        detector.version = "local-yolov5"
        return detector
    except Exception as legacy_error:
        raise RuntimeError(
            "模型无法由支持的检测后端加载。\n"
            f"Ultralytics 后端：{ultralytics_message}\n"
            f"原版 YOLOv5 后端：{legacy_error}"
        ) from legacy_error
