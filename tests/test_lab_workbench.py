import tempfile
import unittest
from collections import deque
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import cv2
import numpy as np

from msm_lab.experiment_store import ExperimentStore, sha256_file
from msm_lab.evaluation import (detection_metrics, load_test_sequences,
                                proposal_metrics, yolo_labels_to_boxes)
from msm_lab.motion_algorithms import create_motion_algorithm, motion_algorithm_choices
from msm_lab.contracts import Detection, DetectionTrace, DetectionTraceSnapshot
from msm_lab.model_catalog import MODEL_PROFILES, architecture_svg
from msm_lab.training import (SpinosaurusYOLOv5Backend, TrainingRequest,
                              TrainingState, _ProcessRecord)
from msm_yolo_demo import ByteTrackAdapter


class MotionAlgorithmTests(unittest.TestCase):
    def test_algorithms_keep_independent_state_and_return_common_result(self):
        algorithms = {
            key: create_motion_algorithm(key, {
                "min_area": 4, "threshold_offset": 1,
                "mog2_history": 40, "mog2_var_threshold": 10,
            })
            for key, _ in motion_algorithm_choices()
        }
        first = np.zeros((96, 128, 3), np.uint8)
        moved = first.copy()
        cv2.rectangle(moved, (48, 35), (65, 52), (255, 255, 255), -1)
        first_results = {key: alg.process(first) for key, alg in algorithms.items()}
        second_results = {key: alg.process(moved) for key, alg in algorithms.items()}
        self.assertEqual(set(first_results), set(second_results))
        for result in second_results.values():
            self.assertEqual(result.mask.shape, moved.shape[:2])
            self.assertEqual(result.diff.shape, moved.shape[:2])
            self.assertIsInstance(result.proposals, list)
            self.assertIn("algorithm", result.diagnostics)
            self.assertIn("原始输入帧", result.stages)
            self.assertIn("阈值分割结果", result.stages)
            self.assertEqual(result.stages["原始输入帧"].shape, moved.shape)
        self.assertEqual(algorithms["mog2"].history, 40)

    def test_gmc_temporal_window_handles_warmup_motion_and_shape_reset(self):
        from msm_yolo_demo import MotionProposer

        rng = np.random.default_rng(7)
        background = rng.integers(0, 150, (128, 160), dtype=np.uint8)
        frames = []
        for x in (40, 44, 48):
            gray = background.copy()
            cv2.rectangle(gray, (x, 54), (x + 10, 64), 255, -1)
            frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        motion = MotionProposer(min_area=2, open_iterations=0,
                               close_iterations=0, edge_margin_ratio=0,
                               history_window=3)
        first = motion.process(frames[0])
        self.assertEqual(first.diagnostics["history_count"], 0)
        second = motion.process(frames[1])
        third = motion.process(frames[2])
        self.assertGreaterEqual(second.diagnostics["history_count"], 1)
        self.assertEqual(third.diagnostics["history_count"], 2)
        self.assertTrue(third.diagnostics["compensation_valid"])
        self.assertLessEqual(third.diagnostics["history_count"], 2)
        self.assertEqual(third.diff.shape, frames[2].shape[:2])

        resized = np.zeros((96, 120, 3), dtype=np.uint8)
        reset_result = motion.process(resized)
        self.assertEqual(reset_result.diagnostics["history_count"], 0)
        self.assertEqual(reset_result.proposals, [])

    def test_blank_frames_fall_back_when_homography_cannot_be_estimated(self):
        from msm_yolo_demo import MotionProposer

        motion = MotionProposer(edge_margin_ratio=0)
        blank = np.zeros((64, 80, 3), dtype=np.uint8)
        motion.process(blank)
        result = motion.process(blank.copy())
        self.assertFalse(result.diagnostics["compensation_valid"])
        self.assertEqual(result.diff.shape, blank.shape[:2])


class ExperimentStoreTests(unittest.TestCase):
    def test_experiment_records_hashes_parameters_and_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.mp4"
            source.write_bytes(b"video")
            database = root / "catalog.sqlite3"
            store = ExperimentStore(database)
            project = store.ensure_project("acceptance")
            store.start_experiment(
                experiment_id="test-run", project_id=project, mode="motion",
                video_path=source, video_sha256=sha256_file(source), model_id=None,
                model_path=None, model_sha256=None, detector_backend=None,
                tracker_backend=None, tracker_version=None,
                motion_algorithms={"selected": "mog2"},
                parameters={"history": 40}, device=None, output_dir=root,
            )
            output = root / "result.mp4"
            output.write_bytes(b"result")
            store.finish_experiment("test-run", status="completed",
                                    summary={"frames": 1}, output_video=output)
            record = store.recent_experiments(1)[0]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["video_sha256"], sha256_file(source))
            self.assertEqual(record["output_video"], str(output.resolve()))
            self.assertEqual(record["project_name"], "acceptance")

    def test_training_run_metadata_is_persisted_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data.yaml"
            dataset.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [drone]\n")
            checkpoint = root / "weights" / "last.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            request = TrainingRequest(
                backend_id="spinosaurus_yolov5", model=str(root / "best.pt"),
                dataset=str(dataset), options={"name": "exp", "epochs": 5},
                repo_root=str(root), python_executable="/usr/bin/python3",
                output_dir=str(root / "runs"), run_id="training-record",
            )
            store = ExperimentStore(root / "catalog.sqlite3")
            store.create_training_run(
                run_id="training-record", request=request.to_dict(),
                dataset_sha256="data-hash", model_source_sha256="model-hash",
                output_dir=root / "runs" / "exp", environment={"torch": "2.9.1"},
            )
            store.update_training_run(
                "training-record", status="stopped", epoch=2, epochs=5,
                progress=0.4, metrics={"metrics/mAP_0.5:0.95": 0.3},
                artifacts=[str(checkpoint)], log_tail="stopped after epoch 2",
            )
            record = store.training_run("training-record")
            self.assertEqual(record["status"], "stopped")
            self.assertEqual(record["last_checkpoint"], str(checkpoint))
            self.assertEqual(record["epoch"], 2)
            self.assertEqual(len(store.recent_training_runs()), 1)


class TrackerAdapterTests(unittest.TestCase):
    def test_empty_frames_and_common_detection_contract(self):
        tracker = ByteTrackAdapter(confidence=0.25, frame_rate=30, track_buffer=30)
        self.assertEqual(tracker.update([], (480, 640, 3)), [])
        detection = Detection(100, 100, 160, 150, 0.9, 0)
        tracks = tracker.update([detection], (480, 640, 3))
        if not tracks:
            tracks = tracker.update([detection], (480, 640, 3))
        self.assertEqual(len(tracks), 1)
        track_id = tracks[0].track_id
        next_tracks = tracker.update(
            [Detection(102, 100, 162, 150, 0.88, 0)], (480, 640, 3))
        self.assertEqual(next_tracks[0].track_id, track_id)


class MotionPipelineGeometryTests(unittest.TestCase):
    def test_candidate_anchor_and_crop_coordinates_are_stable(self):
        from msm_yolo_demo import MSMVideoPipeline

        class FakeDetector:
            names = {0: "Drone"}

            def predict(self, image, confidence, imgsz, iou=0.45):
                return [Detection(2, 3, 12, 13, 0.9, 0)]

        pipeline = MSMVideoPipeline(FakeDetector(), roi_size=32)
        pipeline._update_motion_anchor((10, 10, 20, 20))
        pipeline._update_motion_anchor((14, 10, 24, 20))
        selected = pipeline._select_motion_candidate([((18, 10, 28, 20), 0.0)])
        self.assertIsNotNone(selected)
        mapped = pipeline._detect_crop(
            np.zeros((64, 64, 3), dtype=np.uint8), (10, 20, 30, 40), 0.1
        )
        self.assertEqual((mapped[0].x1, mapped[0].y1,
                          mapped[0].x2, mapped[0].y2), (12, 23, 22, 33))

    def test_candidate_misses_expire_the_motion_anchor(self):
        from msm_yolo_demo import MSMVideoPipeline, PipelineState
        from msm_lab.contracts import MotionResult

        class EmptyMotion:
            def process(self, frame):
                return MotionResult()

            def update_target_template(self, frame, detection):
                pass

        class FakeDetector:
            names = {0: "Drone"}

            def predict(self, image, confidence, imgsz, iou=0.45):
                return []

        pipeline = MSMVideoPipeline(FakeDetector(), motion_algorithm=EmptyMotion())
        pipeline.state = PipelineState.MOTION_EXTRACT
        pipeline.motion_anchor_box = (20, 20, 30, 30)
        for _ in range(pipeline.candidate_miss_limit):
            pipeline.process(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIsNone(pipeline.motion_anchor_box)


class ModelLearningAssetsTests(unittest.TestCase):
    def test_representative_model_diagrams_are_valid_svg(self):
        self.assertEqual(len(MODEL_PROFILES), 6)
        for profile in MODEL_PROFILES:
            root = ElementTree.fromstring(architecture_svg(profile))
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertIn(profile["backbone"], architecture_svg(profile).decode())


class EvaluationMetricTests(unittest.TestCase):
    def test_detection_and_proposal_metrics_include_tiny_and_empty_frames(self):
        gt = [Detection(10, 10, 17, 17, 1.0, 0),
              Detection(30, 30, 44, 44, 1.0, 0)]
        predictions = [Detection(10, 10, 17, 17, 0.9, 0),
                       Detection(0, 0, 5, 5, 0.8, 0)]
        metrics = detection_metrics([(gt, predictions), ([], [])], 0.25)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["mAP50"], 0.505, places=3)
        self.assertEqual(metrics["size_bucket_gt"]["lt8px"], 1)
        proposal = proposal_metrics([
            (gt, [(10, 10, 17, 17), (0, 0, 4, 4)]), ([], [])
        ])
        self.assertEqual(proposal["proposal_recall_iou50"], 0.5)
        self.assertEqual(proposal["false_proposals_per_frame"], 0.5)

    def test_yolo_labels_accept_empty_files_and_restore_pixel_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "label.txt"
            label.write_text("0 0.5 0.5 0.25 0.5\n", encoding="utf-8")
            boxes = yolo_labels_to_boxes(label, (100, 200, 3))
            self.assertEqual((boxes[0].x1, boxes[0].y1,
                              boxes[0].x2, boxes[0].y2), (75, 25, 125, 75))
            label.write_text("", encoding="utf-8")
            self.assertEqual(yolo_labels_to_boxes(label, (100, 200, 3)), [])

    def test_test_frames_are_grouped_by_official_video_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "yolo_dataset"
            (root / "images" / "test").mkdir(parents=True)
            (root / "labels" / "test").mkdir(parents=True)
            rows = ["video,official_split"]
            rows.extend(f"video{i:02d},test" for i in range(35))
            (root / "video_splits.csv").write_text("\n".join(rows), encoding="utf-8")
            for stem in ("video00_0001", "video00_0006", "video01_0001"):
                cv2.imwrite(str(root / "images" / "test" / f"{stem}.jpg"),
                            np.zeros((16, 16, 3), dtype=np.uint8))
                (root / "labels" / "test" / f"{stem}.txt").write_text("", encoding="utf-8")
            groups = load_test_sequences(root)
            self.assertEqual(list(groups["video00"])[0][0].stem, "video00_0001")
            self.assertEqual(len(groups["video00"]), 2)
            self.assertEqual(set(groups), {"video00", "video01"})


class YoloThresholdReplayTests(unittest.TestCase):
    def test_threshold_changes_reuse_single_model_forward(self):
        from msm_tuner_gui import YoloTraceWorker

        class FakeDetector:
            backend_id = "fake"
            names = {0: "uav"}
            device = "cpu"

            def __init__(self):
                self.inference_calls = 0
                self.postprocess_calls = []

            def trace_inference(self, image, imgsz):
                self.inference_calls += 1
                return DetectionTraceSnapshot(
                    backend="fake", raw_prediction=object(),
                    resized_image=image.copy(), normalized_image=image.copy(),
                    original_shape=tuple(image.shape), input_shape=(1, 3, 32, 32),
                    tensor_range=(0.0, 1.0),
                )

            def trace_postprocess(self, snapshot, confidence, iou):
                self.postprocess_calls.append((confidence, iou))
                empty = np.empty((0, 6), dtype=np.float32)
                return DetectionTrace(empty, empty, empty, empty,
                                      confidence, iou, False)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
            detector = FakeDetector()
            worker = YoloTraceWorker()
            with patch("msm_tuner_gui.create_detector", return_value=detector):
                worker.analyze(4, str(image_path), "weights.pt", "cpu",
                               "fake", 640, 0.60, 0.45)
                worker.update_thresholds(0.72, 0.30)
                worker.update_thresholds(0.40, 0.70)
            self.assertEqual(detector.inference_calls, 1)
            self.assertEqual(detector.postprocess_calls,
                             [(0.60, 0.45), (0.72, 0.30), (0.40, 0.70)])
            worker.shutdown()


class TrainingBackendTests(unittest.TestCase):
    def test_command_uses_argument_vector_for_weights_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "backend-code").mkdir()
            weights = root / "best.pt"
            weights.write_bytes(b"weights")
            data = root / "data.yaml"
            data.write_text("nc: 1\nnames: [drone]\ntrain: .\nval: .\n")
            request = TrainingRequest(
                backend_id="spinosaurus_yolov5", model=str(weights),
                dataset=str(data), options={"name": "demo", "epochs": 7,
                    "batch_size": 8, "imgsz": 512, "workers": 0,
                    "patience": 10, "device": "cpu", "optimizer": "AdamW"},
                repo_root=str(root), python_executable="/usr/bin/python3",
                output_dir=str(root / "runs"),
            )
            command = SpinosaurusYOLOv5Backend().build_command(request)
            self.assertIn("--weights", command)
            self.assertIn("--optimizer", command)
            self.assertEqual(command[command.index("--optimizer") + 1], "AdamW")
            self.assertEqual(command[command.index("--epochs") + 1], "7")
            self.assertIn("--cfg", command)
            self.assertEqual(command[command.index("--cfg") + 1], "")
            resumed = TrainingRequest(
                backend_id=request.backend_id, model=request.model,
                dataset=request.dataset, options=request.options,
                repo_root=request.repo_root, python_executable=request.python_executable,
                output_dir=request.output_dir, resume_checkpoint=str(root / "last.pt"),
            )
            self.assertEqual(SpinosaurusYOLOv5Backend().build_command(resumed)[-2:],
                             ["--resume", str(root / "last.pt")])

    def test_dataset_yaml_errors_are_reported_before_runtime_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "backend-code").mkdir()
            weights = root / "best.pt"
            weights.write_bytes(b"weights")
            dataset = root / "data.yaml"
            dataset.write_text("path: missing\ntrain: images/train\nval: images/val\nnc: 2\nnames: [drone]\n")
            request = TrainingRequest(
                backend_id="spinosaurus_yolov5", model=str(weights),
                dataset=str(dataset), options={"device": "cpu"},
                repo_root=str(root), python_executable="/usr/bin/python3",
                output_dir=str(root / "runs"),
            )
            errors = SpinosaurusYOLOv5Backend().validate(request)
            self.assertTrue(any("names 数量" in error for error in errors))
            self.assertTrue(any("train 路径不存在" in error for error in errors))

    def test_results_csv_parser_reports_epoch_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            csv_path.write_text(
                " epoch , metrics/precision , metrics/mAP_0.5:0.95 \n 1 , 0.8 , 0.5 \n")
            epoch, metrics = SpinosaurusYOLOv5Backend._latest_metrics(csv_path)
            self.assertEqual(epoch, 2)
            self.assertEqual(metrics["metrics/precision"], 0.8)
            self.assertEqual(metrics["metrics/mAP_0.5:0.95"], 0.5)
            self.assertEqual(SpinosaurusYOLOv5Backend.ANSI_ESCAPE.sub("", "\x1b[31merror\x1b[0m"), "error")

    def test_resume_rejects_checkpoint_without_optimizer_state(self):
        import sys
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = root / "backend-code"
            trainer.mkdir()
            (trainer / "train.py").write_text("# trainer placeholder")
            dataset_root = root / "dataset"
            for split in ("train", "val"):
                image_dir = dataset_root / "images" / split
                image_dir.mkdir(parents=True)
                (image_dir / "one.jpg").write_bytes(b"image")
            data = root / "data.yaml"
            data.write_text(
                f"path: {dataset_root}\ntrain: images/train\nval: images/val\n"
                "nc: 1\nnames: [drone]\n")
            model = root / "best.pt"
            model.write_bytes(b"weights")
            run = root / "runs" / "exp"
            weights = run / "weights"
            weights.mkdir(parents=True)
            checkpoint = weights / "last.pt"
            torch.save({"epoch": -1, "optimizer": None}, checkpoint)
            (run / "opt.yaml").write_text("epochs: 2\n")
            request = TrainingRequest(
                backend_id="spinosaurus_yolov5", model=str(model),
                dataset=str(data), options={"device": "cpu"},
                repo_root=str(root), python_executable=sys.executable,
                output_dir=str(root / "runs"), resume_checkpoint=str(checkpoint),
            )
            errors = SpinosaurusYOLOv5Backend().validate(request)
            self.assertTrue(any("不含可恢复的优化器状态" in error for error in errors))

    def test_cancel_marks_job_stopped_after_child_exits(self):
        class FakeProcess:
            pid = 1234
            stdout = StringIO("")

            def __init__(self):
                self.returncode = None
                self.signals = []

            def poll(self):
                return self.returncode

            def send_signal(self, value):
                self.signals.append(value)

        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            request = TrainingRequest("spinosaurus_yolov5", "model.pt", "data.yaml")
            record = _ProcessRecord(request, process, TrainingState.RUNNING, Path(tmp))
            backend = SpinosaurusYOLOv5Backend()
            backend._records["run"] = record
            backend._active_run_id = "run"
            with patch("msm_lab.training.os.killpg") as killpg:
                backend.cancel("run")
                killpg.assert_called_once()
            process.returncode = 130
            event = backend.status("run")
            self.assertEqual(event.state, TrainingState.STOPPED)
            self.assertEqual(event.return_code, 130)


if __name__ == "__main__":
    unittest.main()
