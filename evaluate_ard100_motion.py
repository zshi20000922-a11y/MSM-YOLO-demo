#!/usr/bin/env python3
"""Compare frame YOLO and camera-motion-guided YOLO on official ARD100 test videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msm_lab.detectors import create_detector
from msm_lab.evaluation import run_evaluation
from msm_lab.experiment_store import runtime_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compare YOLO baseline, MOG2, frame difference, 2/3-frame GMC, "
                     "and trajectory filtering on the official ARD100 test videos.")
    )
    parser.add_argument("--dataset", required=True,
                        help="ARD100 root or yolo_dataset directory")
    parser.add_argument("--weights", required=True, help="YOLO .pt checkpoint")
    parser.add_argument("--output", default="runs/ard100_motion_eval",
                        help="Result directory for JSON/CSV and optional videos")
    parser.add_argument("--backend", choices=("auto", "legacy_yolov5", "ultralytics"),
                        default="auto")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--roi-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold used for point metrics")
    parser.add_argument("--prediction-floor", type=float, default=0.001,
                        help="Low inference and pipeline gate threshold for AP collection")
    parser.add_argument("--iou", type=float, default=0.45, help="Detector NMS IoU threshold")
    parser.add_argument("--global-interval", type=int, default=15)
    parser.add_argument("--max-videos", type=int,
                        help="Optional leading subset of official test videos for a smoke run")
    parser.add_argument("--save-visualizations", action="store_true",
                        help="Write annotated sampled-frame videos for each mode and test video")
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--threshold-sigma", type=float, default=2.0)
    parser.add_argument("--threshold-offset", type=float, default=8.0)
    parser.add_argument("--threshold-window", type=int, default=21)
    parser.add_argument("--max-proposals", type=int, default=20)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.03)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.roi_size <= 0:
        parser.error("--imgsz and --roi-size must be positive")
    if not 0 <= args.prediction_floor <= args.conf <= 1:
        parser.error("Require 0 <= --prediction-floor <= --conf <= 1")
    if not 0 <= args.iou <= 1:
        parser.error("--iou must be between 0 and 1")
    if args.global_interval < 1:
        parser.error("--global-interval must be positive")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    return args


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    detector = create_detector(weights, args.device, args.backend)
    # The legacy YOLOv5 adapter does not expose its source checkpoint path.
    detector.weights = str(weights)
    try:
        results = run_evaluation(
            dataset_root=args.dataset,
            detector=detector,
            output_dir=args.output,
            imgsz=args.imgsz,
            roi_size=args.roi_size,
            metric_confidence=args.conf,
            prediction_floor=args.prediction_floor,
            iou_threshold=args.iou,
            global_interval=args.global_interval,
            max_videos=args.max_videos,
            save_visualizations=args.save_visualizations,
            motion_options={
                "min_area": args.min_area,
                "threshold_sigma": args.threshold_sigma,
                "threshold_offset": args.threshold_offset,
                "threshold_window": args.threshold_window,
                "max_proposals": args.max_proposals,
                "edge_margin_ratio": args.edge_margin_ratio,
                "morph_kernel": 3,
                "open_iterations": 1,
                "close_iterations": 2,
            },
        )
        results["manifest"]["runtime"] = runtime_versions()
        output_dir = Path(args.output).expanduser().resolve()
        (output_dir / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(results["manifest"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Results: {output_dir / 'summary.json'}")
        print(f"Per-video table: {output_dir / 'per_video.csv'}")
        for mode, summary in results["summary"].items():
            print(
                f"{mode}: mAP50={summary['mAP50']}, "
                f"mAP50:95={summary['mAP50_95']}, "
                f"Recall={summary['recall']}, "
                f"proposal recall={summary['proposal_recall_iou50']}, "
                f"FPS={summary['average_fps']:.2f}"
            )
    finally:
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
