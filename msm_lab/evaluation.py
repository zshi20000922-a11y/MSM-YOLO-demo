"""Offline ARD100 evaluation for frame and motion-guided YOLO pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from msm_yolo_demo import Detection, MSMVideoPipeline, MotionProposer
from msm_lab.motion_algorithms import create_motion_algorithm


IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))
EXPERIMENTS = (
    {"id": "yolo_baseline", "label": "YOLO 单帧基线", "algorithm": None,
     "history_window": None, "trajectory_filtering": False},
    {"id": "mog2_roi", "label": "MOG2 + YOLO ROI", "algorithm": "mog2",
     "history_window": None, "trajectory_filtering": False},
    {"id": "frame_difference_roi", "label": "帧差 + YOLO ROI", "algorithm": "frame_difference",
     "history_window": None, "trajectory_filtering": False},
    {"id": "gmc_2frame_roi", "label": "GMC + 两帧差分 + YOLO ROI", "algorithm": "gmc",
     "history_window": 2, "trajectory_filtering": False},
    {"id": "gmc_3frame_roi", "label": "GMC + 三帧差分 + YOLO ROI", "algorithm": "gmc",
     "history_window": 3, "trajectory_filtering": False},
    {"id": "gmc_3frame_roi_trajectory", "label": "三帧 GMC + ROI + 轨迹筛选",
     "algorithm": "gmc", "history_window": 3, "trajectory_filtering": True},
)


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def yolo_labels_to_boxes(label_path: Path, image_shape: tuple[int, ...]
                         ) -> list[Detection]:
    """Read one YOLO txt file and return absolute xyxy boxes."""
    height, width = image_shape[:2]
    boxes: list[Detection] = []
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return boxes
    for line_number, line in enumerate(content.splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected class x y w h")
        class_id = int(fields[0])
        center_x, center_y, box_w, box_h = map(float, fields[1:])
        if not all(math.isfinite(value) for value in (center_x, center_y, box_w, box_h)):
            raise ValueError(f"{label_path}:{line_number}: non-finite box value")
        if box_w <= 0 or box_h <= 0:
            raise ValueError(f"{label_path}:{line_number}: box size must be positive")
        x1 = max(0, min(width - 1, int(round((center_x - box_w / 2) * width))))
        y1 = max(0, min(height - 1, int(round((center_y - box_h / 2) * height))))
        x2 = max(x1 + 1, min(width, int(round((center_x + box_w / 2) * width))))
        y2 = max(y1 + 1, min(height, int(round((center_y + box_h / 2) * height))))
        boxes.append(Detection(x1, y1, x2, y2, 1.0, class_id))
    return boxes


def _frame_key(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_([0-9]+)", stem)
    if not match:
        raise ValueError(f"Cannot identify video and frame number from {stem!r}")
    return match.group(1), int(match.group(2))


def load_test_sequences(dataset_root: str | Path) -> dict[str, list[tuple[Path, Path]]]:
    """Load official ARD100 test frames grouped into source-video sequences."""
    root = Path(dataset_root).expanduser().resolve()
    yolo_root = root / "yolo_dataset" if (root / "yolo_dataset").is_dir() else root
    images_dir = yolo_root / "images" / "test"
    labels_dir = yolo_root / "labels" / "test"
    split_manifest = yolo_root / "video_splits.csv"
    for required in (images_dir, labels_dir, split_manifest):
        if not required.exists():
            raise FileNotFoundError(f"ARD100 test asset is missing: {required}")

    video_splits: dict[str, str] = {}
    with split_manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video = row.get("video", "").strip()
            split = row.get("official_split", "").strip()
            if not video or not split or video in video_splits:
                raise ValueError(f"Invalid or duplicate row in {split_manifest}: {row}")
            video_splits[video] = split
    official_test = {video for video, split in video_splits.items() if split == "test"}
    if len(official_test) != 35:
        raise ValueError(f"Expected 35 official test videos, found {len(official_test)}")

    sequences: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image_path in images_dir.glob("*.jpg"):
        video, frame_number = _frame_key(image_path.stem)
        if video not in official_test:
            raise ValueError(f"Test image {image_path.name} is not in the official test split")
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file for {image_path}: {label_path}")
        sequences[video].append((image_path, label_path))
    if not sequences:
        raise ValueError(f"No test images found in {images_dir}")
    for video, frames in sequences.items():
        frames.sort(key=lambda pair: _frame_key(pair[0].stem)[1])
    return dict(sorted(sequences.items()))


def source_video_fps(dataset_root: str | Path, video: str) -> float | None:
    root = Path(dataset_root).expanduser().resolve()
    video_root = root if (root / "test_videos").is_dir() else root.parent
    source = video_root / "test_videos" / f"{video}.mp4"
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            return None
        value = float(capture.get(cv2.CAP_PROP_FPS))
        return value if math.isfinite(value) and value > 0 else None
    finally:
        capture.release()


def sampling_interval(frames: list[tuple[Path, Path]]) -> float:
    frame_numbers = [_frame_key(image_path.stem)[1] for image_path, _ in frames]
    gaps = np.diff(frame_numbers)
    positive_gaps = gaps[gaps > 0]
    return float(np.median(positive_gaps)) if positive_gaps.size else 1.0


def _prediction_rows(frames: list[tuple[list[Detection], list[Detection]]]
                     ) -> list[tuple[int, float, Detection]]:
    rows = []
    for frame_index, (_, predictions) in enumerate(frames):
        for prediction in predictions:
            rows.append((frame_index, prediction.confidence, prediction))
    return sorted(rows, key=lambda row: row[1], reverse=True)


def _average_precision(tp: np.ndarray, fp: np.ndarray, gt_count: int) -> float | None:
    if gt_count == 0:
        return None
    if tp.size == 0:
        return 0.0
    recall = np.cumsum(tp) / gt_count
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    samples = np.linspace(0, 1, 101)
    return float(np.mean([precision[recall >= value].max(initial=0.0)
                          for value in samples]))


def _match_frame(ground_truth: list[Detection], predictions: list[Detection],
                 iou_threshold: float, confidence_floor: float = 0.0
                 ) -> tuple[int, int, int, set[int]]:
    eligible = [item for item in predictions if item.confidence >= confidence_floor]
    eligible.sort(key=lambda item: item.confidence, reverse=True)
    matched: set[int] = set()
    tp = fp = 0
    for prediction in eligible:
        best_index, best_iou = -1, 0.0
        for index, target in enumerate(ground_truth):
            if index in matched or target.class_id != prediction.class_id:
                continue
            overlap = box_iou(
                (prediction.x1, prediction.y1, prediction.x2, prediction.y2),
                (target.x1, target.y1, target.x2, target.y2),
            )
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(ground_truth) - len(matched), matched


def detection_metrics(frames: list[tuple[list[Detection], list[Detection]]],
                     confidence_threshold: float) -> dict:
    gt_count = sum(len(gt) for gt, _ in frames)
    tp, fp, fn = 0, 0, 0
    buckets = {"lt8px": [0, 0], "8to16px": [0, 0],
               "16to32px": [0, 0], "ge32px": [0, 0]}
    for gt, predictions in frames:
        frame_tp, frame_fp, frame_fn, matched = _match_frame(
            gt, predictions, 0.5, confidence_threshold
        )
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        for index, target in enumerate(gt):
            size = max(target.x2 - target.x1, target.y2 - target.y1)
            bucket = ("lt8px" if size < 8 else "8to16px" if size < 16
                      else "16to32px" if size < 32 else "ge32px")
            buckets[bucket][1] += 1
            if index in matched:
                buckets[bucket][0] += 1

    ap_by_iou = {}
    rows = _prediction_rows(frames)
    for threshold in IOU_THRESHOLDS:
        used_by_frame: dict[int, set[int]] = defaultdict(set)
        true_positive = []
        false_positive = []
        for frame_index, _, prediction in rows:
            ground_truth = frames[frame_index][0]
            best_index, best_iou = -1, 0.0
            for index, target in enumerate(ground_truth):
                if index in used_by_frame[frame_index] or target.class_id != prediction.class_id:
                    continue
                overlap = box_iou(
                    (prediction.x1, prediction.y1, prediction.x2, prediction.y2),
                    (target.x1, target.y1, target.x2, target.y2),
                )
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= threshold:
                used_by_frame[frame_index].add(best_index)
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)
        ap_by_iou[f"{threshold:.2f}"] = _average_precision(
            np.asarray(true_positive), np.asarray(false_positive), gt_count
        )

    ap50 = ap_by_iou.get("0.50")
    valid_aps = [value for value in ap_by_iou.values() if value is not None]
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / gt_count if gt_count else None,
        "mAP50": ap50,
        "mAP50_95": float(np.mean(valid_aps)) if valid_aps else None,
        "ap_by_iou": ap_by_iou,
        "gt_count": gt_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "size_bucket_recall": {
            name: (matched_count / count if count else None)
            for name, (matched_count, count) in buckets.items()
        },
        "size_bucket_gt": {name: count for name, (_, count) in buckets.items()},
    }


def proposal_metrics(frames: list[tuple[list[Detection], list[tuple[int, int, int, int]]]]
                     ) -> dict:
    gt_count = proposal_recall = center_covered = total_proposals = false_proposals = 0
    for ground_truth, proposals in frames:
        total_proposals += len(proposals)
        covered_gt: set[int] = set()
        matched_proposals: set[int] = set()
        overlaps: list[tuple[float, int, int]] = []
        for target_index, target in enumerate(ground_truth):
            gt_box = (target.x1, target.y1, target.x2, target.y2)
            center_x = (target.x1 + target.x2) / 2.0
            center_y = (target.y1 + target.y2) / 2.0
            if any(x1 <= center_x < x2 and y1 <= center_y < y2
                   for x1, y1, x2, y2 in proposals):
                center_covered += 1
            for proposal_index, proposal in enumerate(proposals):
                overlap = box_iou(gt_box, proposal)
                if overlap >= 0.5:
                    overlaps.append((overlap, target_index, proposal_index))
        for _, target_index, proposal_index in sorted(overlaps, reverse=True):
            if (target_index not in covered_gt
                    and proposal_index not in matched_proposals):
                covered_gt.add(target_index)
                matched_proposals.add(proposal_index)
        gt_count += len(ground_truth)
        proposal_recall += len(covered_gt)
        false_proposals += len(proposals) - len(matched_proposals)
    frame_count = len(frames)
    return {
        "proposal_recall_iou50": proposal_recall / gt_count if gt_count else None,
        "proposal_gt_center_coverage": center_covered / gt_count if gt_count else None,
        "mean_proposals_per_frame": total_proposals / frame_count if frame_count else 0.0,
        "false_proposals_per_frame": false_proposals / frame_count if frame_count else 0.0,
        "total_proposals": total_proposals,
        "false_proposals": false_proposals,
        "gt_count": gt_count,
        "frame_count": frame_count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_evaluation(*, dataset_root: str | Path, detector, output_dir: str | Path,
                   imgsz: int = 640, roi_size: int = 640,
                   metric_confidence: float = 0.25, prediction_floor: float = 0.001,
                   iou_threshold: float = 0.45, global_interval: int = 15,
                   max_videos: int | None = None, save_visualizations: bool = False,
                   motion_options: dict | None = None) -> dict:
    dataset_path = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    sequences = load_test_sequences(dataset_path)
    if max_videos is not None:
        sequences = dict(list(sequences.items())[:max_videos])
    experiments = EXPERIMENTS
    modes = tuple(item["id"] for item in experiments)
    per_video: dict[str, dict[str, list]] = {
        mode: {} for mode in modes
    }
    video_rows = []
    visualization_root = output_path / "visualizations"
    if save_visualizations:
        visualization_root.mkdir(parents=True, exist_ok=True)

    for video, samples in sequences.items():
        for experiment in experiments:
            mode = experiment["id"]
            detections_for_video = []
            proposals_for_video = []
            timings = defaultdict(list)
            motion_diagnostics = defaultdict(list)
            pipeline = None
            if experiment["algorithm"] is not None:
                options = dict(motion_options or {})
                if experiment["algorithm"] == "gmc":
                    options["history_window"] = experiment["history_window"]
                    motion_algorithm = MotionProposer(**options)
                else:
                    motion_algorithm = create_motion_algorithm(
                        experiment["algorithm"], options
                    )
                pipeline = MSMVideoPipeline(
                    detector, roi_size=roi_size,
                    global_conf=prediction_floor, local_conf=prediction_floor,
                    global_interval=global_interval, imgsz=imgsz,
                    iou_threshold=iou_threshold,
                    motion_algorithm=motion_algorithm,
                    trajectory_filtering=experiment["trajectory_filtering"],
                )
            writer = None
            if save_visualizations:
                first_image = cv2.imread(str(samples[0][0]))
                if first_image is None:
                    raise ValueError(f"Cannot decode test image: {samples[0][0]}")
                height, width = first_image.shape[:2]
                video_root = dataset_path if (dataset_path / "test_videos").is_dir() else dataset_path.parent
                source_video = video_root / "test_videos" / f"{video}.mp4"
                source_cap = cv2.VideoCapture(str(source_video))
                original_fps = source_cap.get(cv2.CAP_PROP_FPS) if source_cap.isOpened() else 30.0
                source_cap.release()
                output_video = visualization_root / f"{video}_{mode}.mp4"
                writer = cv2.VideoWriter(
                    str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, original_fps / 5.0), (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create visualization: {output_video}")
            try:
                for image_path, label_path in samples:
                    frame_started = time.perf_counter()
                    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f"Cannot decode test image: {image_path}")
                    targets = yolo_labels_to_boxes(label_path, frame.shape)
                    if experiment["algorithm"] is None:
                        started = time.perf_counter()
                        predictions = detector.predict(
                            frame, prediction_floor, imgsz, iou_threshold
                        )
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        timings["detector_ms"].append(elapsed_ms)
                        timings["pipeline_ms"].append(elapsed_ms)
                        proposals = []
                        annotated = frame.copy()
                        for prediction in predictions:
                            cv2.rectangle(
                                annotated, (prediction.x1, prediction.y1),
                                (prediction.x2, prediction.y2), (40, 230, 70), 2,
                            )
                            cv2.putText(
                                annotated, f"{prediction.confidence:.2f}",
                                (prediction.x1, max(18, prediction.y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 230, 70), 1,
                                cv2.LINE_AA,
                            )
                    else:
                        assert pipeline is not None
                        annotated, predictions = pipeline.process(frame)
                        proposals = [item[0] for item in pipeline.motion_result.proposals]
                        for key in ("history_count", "compensation_valid"):
                            value = pipeline.motion_result.diagnostics.get(key)
                            if value is not None:
                                motion_diagnostics[key].append(float(value))
                        for key, value in pipeline.last_frame_timing.items():
                            timings[key].append(float(value))
                    detections_for_video.append((targets, predictions))
                    proposals_for_video.append((targets, proposals))
                    timings["end_to_end_ms"].append(
                        (time.perf_counter() - frame_started) * 1000.0
                    )
                    if writer is not None:
                        for target in targets:
                            cv2.rectangle(annotated, (target.x1, target.y1),
                                          (target.x2, target.y2), (255, 180, 0), 1)
                        writer.write(annotated)
            finally:
                if writer is not None:
                    writer.release()
                if pipeline is not None:
                    pipeline.motion.reset()

            per_video[mode][video] = {
                "detections": detections_for_video,
                "proposals": proposals_for_video,
                "timings": dict(timings),
                "motion_diagnostics": dict(motion_diagnostics),
            }
            det_summary = detection_metrics(detections_for_video, metric_confidence)
            prop_summary = proposal_metrics(proposals_for_video)
            row = {"video": video, "mode": mode, **{
                key: value for key, value in det_summary.items()
                if key not in ("ap_by_iou", "size_bucket_recall", "size_bucket_gt")
            }, **prop_summary}
            row.update({f"recall_{name}": value for name, value
                        in det_summary["size_bucket_recall"].items()})
            for key, values in motion_diagnostics.items():
                metric_name = ("compensation_valid_ratio" if key == "compensation_valid"
                               else "mean_history_frames")
                row[metric_name] = float(np.mean(values)) if values else 0.0
            row.update({f"avg_{name}": float(np.mean(values)) if values else 0.0
                        for name, values in timings.items()})
            row["average_fps"] = (1000.0 / row.get("avg_end_to_end_ms", 0)
                                   if row.get("avg_end_to_end_ms", 0) > 0 else 0.0)
            source_fps = source_video_fps(dataset_path, video)
            frame_stride = sampling_interval(samples)
            row["source_video_fps"] = source_fps
            row["sampled_video_fps"] = (
                source_fps / frame_stride if source_fps is not None else None
            )
            row["false_proposals_per_minute"] = (
                row["false_proposals_per_frame"] * source_fps / frame_stride * 60.0
                if source_fps is not None else None
            )
            row["experiment_label"] = experiment["label"]
            video_rows.append(row)
            print(f"[{mode}] {video}: {len(samples)} annotated frames", flush=True)

    summaries = {}
    for experiment in experiments:
        mode = experiment["id"]
        all_detections = [frame for video_data in per_video[mode].values()
                          for frame in video_data["detections"]]
        all_proposals = [frame for video_data in per_video[mode].values()
                         for frame in video_data["proposals"]]
        metrics = detection_metrics(all_detections, metric_confidence)
        metrics.update(proposal_metrics(all_proposals))
        all_timing_values: dict[str, list[float]] = defaultdict(list)
        all_motion_diagnostics: dict[str, list[float]] = defaultdict(list)
        for video_data in per_video[mode].values():
            for key, values in video_data["timings"].items():
                all_timing_values[key].extend(values)
            for key, values in video_data["motion_diagnostics"].items():
                all_motion_diagnostics[key].extend(values)
        timing_summary = {
            f"mean_{key}": float(np.mean(values)) if values else 0.0
            for key, values in all_timing_values.items()
        }
        metrics.update(timing_summary)
        for key, values in all_motion_diagnostics.items():
            metric_name = ("compensation_valid_ratio" if key == "compensation_valid"
                           else "mean_history_frames")
            metrics[metric_name] = float(np.mean(values)) if values else 0.0
        frame_count = metrics["frame_count"]
        end_to_end_ms = timing_summary.get("mean_end_to_end_ms", 0.0)
        metrics["average_fps"] = 1000.0 / end_to_end_ms if end_to_end_ms > 0 else 0.0
        metrics["frames"] = frame_count
        metrics["videos"] = len(per_video[mode])
        metrics["experiment_label"] = experiment["label"]
        video_durations_minutes = []
        source_fps_values = []
        for video, video_data in per_video[mode].items():
            source_fps = source_video_fps(dataset_path, video)
            samples = sequences[video]
            stride = sampling_interval(samples)
            if source_fps is not None:
                source_fps_values.append(source_fps)
                video_durations_minutes.append(
                    len(video_data["proposals"]) * stride / source_fps / 60.0
                )
        metrics["mean_source_video_fps"] = (
            float(np.mean(source_fps_values)) if source_fps_values else None
        )
        metrics["false_proposals_per_minute"] = (
            metrics["false_proposals"] / sum(video_durations_minutes)
            if video_durations_minutes and sum(video_durations_minutes) > 0 else None
        )
        summaries[mode] = metrics

    csv_path = output_path / "per_video.csv"
    if video_rows:
        columns = list(dict.fromkeys(key for row in video_rows for key in row))
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(video_rows)
    manifest = {
        "dataset_root": str(dataset_path),
        "dataset_protocol": "ARD100 官方视频级测试划分；按源视频分组的标注抽帧",
        "test_videos": list(sequences),
        "weights": str(Path(detector.weights).resolve()) if hasattr(detector, "weights") else None,
        "weights_sha256": _sha256(Path(detector.weights)) if hasattr(detector, "weights") else None,
        "detector_backend": getattr(detector, "backend_id", "unknown"),
        "imgsz": imgsz,
        "roi_size": roi_size,
        "metric_confidence": metric_confidence,
        "prediction_floor_and_pipeline_gate": prediction_floor,
        "iou_threshold": iou_threshold,
        "global_interval": global_interval,
        "motion_options": motion_options or {},
        "experiments": [dict(item) for item in experiments],
        "modes": list(modes),
        "created_unix": time.time(),
        "visualizations": str(visualization_root) if save_visualizations else None,
    }
    result = {"manifest": manifest, "summary": summaries}
    (output_path / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_comparison_report(output_path / "comparison_report_zh.md", manifest,
                             summaries, video_rows)
    return result


def _write_comparison_report(path: Path, manifest: dict, summaries: dict,
                             video_rows: list[dict] | None = None) -> None:
    lines = [
        "# ARD100 运动候选与 YOLO 综合对比",
        "",
        f"- 数据协议：{manifest['dataset_protocol']}。本次覆盖 {len(manifest['test_videos'])} 个视频。",
        f"- 权重：`{manifest.get('weights')}`；SHA256：`{manifest.get('weights_sha256')}`。",
        f"- 输入尺寸：{manifest['imgsz']}；ROI：{manifest['roi_size']}；点指标置信度：{manifest['metric_confidence']}；候选/推理阈值：{manifest['prediction_floor_and_pipeline_gate']}。",
        "- 目标尺寸桶按原图框最大边长计算；候选召回要求 IoU ≥ 0.5，中心覆盖统计 GT 中心是否落入候选框。",
        "- 速度为当前主机端到端实测。测试图片按数据集抽帧间隔取样；误候选/分钟按原视频 FPS 和抽帧间隔折算。",
        "",
        "| 实验 | Precision | Recall | mAP50 | mAP50:95 | <8px Recall | 8–16px Recall | 16–32px Recall | ≥32px Recall | 候选 IoU召回 | GT中心覆盖 | 候选/帧 | 误候选/分钟 | motion ms | 全图YOLO ms | ROI YOLO ms | YOLO总 ms | pipeline ms | E2E ms | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in manifest["experiments"]:
        mode = experiment["id"]
        metrics = summaries[mode]
        bucket = metrics["size_bucket_recall"]
        values = [
            experiment["label"], metrics["precision"], metrics["recall"],
            metrics["mAP50"], metrics["mAP50_95"], bucket["lt8px"],
            bucket["8to16px"], bucket["16to32px"], bucket["ge32px"],
            metrics["proposal_recall_iou50"],
            metrics["proposal_gt_center_coverage"],
            metrics["mean_proposals_per_frame"],
            metrics["false_proposals_per_minute"],
            metrics.get("mean_motion_ms", 0.0),
            metrics.get("mean_global_detector_ms", 0.0),
            metrics.get("mean_roi_detector_ms", 0.0),
            metrics.get("mean_detector_ms", 0.0),
            metrics.get("mean_pipeline_ms", 0.0),
            metrics.get("mean_end_to_end_ms", 0.0), metrics["average_fps"],
        ]
        formatted = ["—" if value is None else
                     (str(value) if isinstance(value, str) else f"{value:.4f}")
                     for value in values]
        lines.append("| " + " | ".join(formatted) + " |")
    lines.extend([
        "",
        "## 综合对比",
        "",
        "Precision/Recall 使用 manifest 中的点指标置信度；mAP 使用低阈值收集预测后按 0.50:0.05:0.95 计算。",
    ])
    baseline = summaries.get("yolo_baseline", {})
    if baseline:
        for experiment in manifest["experiments"]:
            mode = experiment["id"]
            if mode == "yolo_baseline":
                continue
            metrics = summaries[mode]
            delta_map = metrics["mAP50"] - baseline["mAP50"]
            delta_recall = metrics["recall"] - baseline["recall"]
            win_text = ""
            if video_rows:
                paired = {}
                for row in video_rows:
                    paired.setdefault(row["video"], {})[row["mode"]] = row
                wins = sum(float(pair[mode]["mAP50"]) > float(pair["yolo_baseline"]["mAP50"])
                           for pair in paired.values()
                           if mode in pair and "yolo_baseline" in pair)
                losses = sum(float(pair[mode]["mAP50"]) < float(pair["yolo_baseline"]["mAP50"])
                             for pair in paired.values()
                             if mode in pair and "yolo_baseline" in pair)
                win_text = f"；视频级 mAP50 胜/负 {wins}/{losses}"
            lines.append(
                f"- **{experiment['label']}**：相对单帧基线 ΔmAP50 {delta_map:+.4f}、"
                f"ΔRecall {delta_recall:+.4f}{win_text}。"
            )
        if "gmc_2frame_roi" in summaries and "gmc_3frame_roi" in summaries:
            two = summaries["gmc_2frame_roi"]
            three = summaries["gmc_3frame_roi"]
            lines.append(
                f"- 两帧扩为三帧：mAP50 变化 {three['mAP50'] - two['mAP50']:+.4f}，"
                f"端到端速度变化 {three['average_fps'] - two['average_fps']:+.2f} FPS；"
                f"候选数从 {two['mean_proposals_per_frame']:.2f} 增到 "
                f"{three['mean_proposals_per_frame']:.2f} 个/帧。"
            )
        if ("gmc_3frame_roi" in summaries
                and "gmc_3frame_roi_trajectory" in summaries):
            three = summaries["gmc_3frame_roi"]
            trajectory = summaries["gmc_3frame_roi_trajectory"]
            lines.append(
                f"- 增加轨迹筛选：mAP50 变化 "
                f"{trajectory['mAP50'] - three['mAP50']:+.4f}，Recall 变化 "
                f"{trajectory['recall'] - three['recall']:+.4f}，端到端速度变化 "
                f"{trajectory['average_fps'] - three['average_fps']:+.2f} FPS。"
            )
    lines.extend([
        "- 所有运动组同时改变了运动候选调度与 ROI 检测尺度；单帧 YOLO 与它们的差值属于组合流程差异，不能解释成运动模块的独立因果收益。",
        "- GMC 候选 IoU 召回与中心覆盖需一起看：极小 GT 中心落框不等于候选框和目标有足够面积重叠。",
        "- 本次没有加入独立的 E4 无 ROI 变体，因此未单独量化 ROI 放大对多帧 GMC 的贡献。",
        "",
        "## 尚未覆盖的实验",
        "",
        "- E7 可学习时序特征融合：需要成对序列数据管线、新模型结构和独立训练权重；当前对比使用同一现成 `.pt` 权重，未将帧差输入或跟踪后处理冒充为特征融合。",
        "- AP_small 未计算；报告提供原图像素尺寸桶 Recall。ARD100 框没有目标身份 ID，因此未报告 IDF1、ID switches 等跟踪指标。",
        "- RK3588 NPU/RGA/V4L2 板端延迟未测；当前速度来自主机端 `.pt` 后端。",
        "",
        "## 文件",
        "",
        "按视频汇总和各视频结果见 `per_video.csv`；复现实验参数和环境见 `manifest.json`。",
        "",
        "若所用训练权重的训练/验证帧来自同一批训练池视频，验证集数值不作为独立视频泛化结果；本报告主表只使用官方 35 个测试视频。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
