"""Local subprocess adapter for the Spinosaurus YOLOv5 detector trainer.

The trainer stays in its own Conda environment and checkout.  The adapter
never installs dependencies and passes command arguments without a shell.
"""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml


class TrainingState(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class TrainingRequest:
    """Serializable configuration for one local YOLOv5 training run."""

    backend_id: str
    model: str
    dataset: str
    options: Mapping[str, Any] = field(default_factory=dict)
    repo_root: str = ""
    python_executable: str = ""
    output_dir: str = ""
    run_id: str = ""
    resume_checkpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRequest":
        return cls(
            backend_id=str(value.get("backend_id", "spinosaurus_yolov5")),
            model=str(value.get("model", "")),
            dataset=str(value.get("dataset", "")),
            options=dict(value.get("options", {})),
            repo_root=str(value.get("repo_root", "")),
            python_executable=str(value.get("python_executable", "")),
            output_dir=str(value.get("output_dir", "")),
            run_id=str(value.get("run_id", "")),
            resume_checkpoint=str(value.get("resume_checkpoint", "")),
        )


@dataclass(frozen=True)
class TrainingEvent:
    """A point-in-time view of task state, progress, logs and artifacts."""

    run_id: str
    state: TrainingState
    progress: float = 0.0
    epoch: int | None = None
    epochs: int | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    message: str = ""
    artifacts: tuple[str, ...] = ()
    log_lines: tuple[str, ...] = ()
    return_code: int | None = None


@dataclass
class _ProcessRecord:
    request: TrainingRequest
    process: subprocess.Popen[str]
    state: TrainingState
    output_dir: Path
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=2000))
    pending_logs: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    cancel_requested_at: float | None = None
    terminate_sent_at: float | None = None
    return_code: int | None = None
    reader_thread: threading.Thread | None = None


class TrainingBackend(Protocol):
    """Common interface for local or future remote training backends."""

    backend_id: str

    def validate(self, request: TrainingRequest) -> list[str]: ...
    def start(self, request: TrainingRequest) -> str: ...
    def resume(self, request: TrainingRequest, checkpoint: str) -> str: ...
    def cancel(self, run_id: str) -> None: ...
    def status(self, run_id: str) -> TrainingEvent: ...


class SpinosaurusYOLOv5Backend:
    """Run ``backend-code/train.py`` in a separate configured Python env."""

    backend_id = "spinosaurus_yolov5"
    STOP_GRACE_SECONDS = 12.0
    ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self):
        self._records: dict[str, _ProcessRecord] = {}
        self._lock = threading.RLock()
        self._active_run_id: str | None = None

    @staticmethod
    def _resolved(path: str | Path) -> Path:
        return Path(path).expanduser().resolve()

    @staticmethod
    def _dataset_errors(dataset_path: Path) -> list[str]:
        errors: list[str] = []
        if not dataset_path.is_file():
            return [f"数据集 YAML 不存在：{dataset_path}"]
        try:
            data = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            return [f"无法读取数据集 YAML：{exc}"]
        if not isinstance(data, dict):
            return ["数据集 YAML 顶层必须是映射结构。"]
        try:
            count = int(data.get("nc", 0))
        except (TypeError, ValueError):
            count = 0
        names = data.get("names")
        if count < 1:
            errors.append("数据集 YAML 的 nc 必须是正整数。")
        if isinstance(names, dict):
            name_count = len(names)
        elif isinstance(names, (list, tuple)):
            name_count = len(names)
        else:
            name_count = 0
        if name_count < 1 or (count and name_count != count):
            errors.append("数据集 YAML 的 names 数量必须与 nc 一致。")

        dataset_root_value = data.get("path", "")
        dataset_root = Path(os.path.expandvars(str(dataset_root_value))).expanduser() if dataset_root_value else dataset_path.parent
        if not dataset_root.is_absolute():
            dataset_root = dataset_path.parent / dataset_root
        dataset_root = dataset_root.resolve()
        for split in ("train", "val"):
            value = data.get(split)
            if not value:
                errors.append(f"数据集 YAML 缺少 {split} 路径。")
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                candidate = Path(os.path.expandvars(str(item))).expanduser()
                if not candidate.is_absolute():
                    candidate = dataset_root / candidate
                if not candidate.exists():
                    errors.append(f"数据集 {split} 路径不存在：{candidate}")
                    break
                if not SpinosaurusYOLOv5Backend._split_contains_image(candidate, dataset_root):
                    errors.append(f"数据集 {split} 路径中没有可读取的图像：{candidate}")
                    break
        return errors

    @staticmethod
    def _split_contains_image(path: Path, dataset_root: Path) -> bool:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dng"}
        if path.is_dir():
            for directory, _, filenames in os.walk(path):
                if any(Path(name).suffix.lower() in suffixes for name in filenames):
                    return True
            return False
        if path.is_file() and path.suffix.lower() == ".txt":
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = Path(os.path.expandvars(line.strip())).expanduser()
                    if not line.strip():
                        continue
                    if not item.is_absolute():
                        item = dataset_root / item
                    if item.is_file() and item.suffix.lower() in suffixes:
                        return True
            except OSError:
                return False
            return False
        return path.is_file() and path.suffix.lower() in suffixes

    def validate(self, request: TrainingRequest) -> list[str]:
        errors: list[str] = []
        if request.backend_id != self.backend_id:
            errors.append(f"不支持的训练后端：{request.backend_id}")
        repo_root = self._resolved(request.repo_root) if request.repo_root else Path()
        if not request.repo_root or not repo_root.is_dir():
            errors.append("请选择包含 backend-code/train.py 的 YOLOv5 仓库目录。")
        else:
            train_script = repo_root / "backend-code" / "train.py"
            if not train_script.is_file():
                errors.append(f"训练入口不存在：{train_script}")

        python_executable = self._resolved(request.python_executable) if request.python_executable else Path()
        if not request.python_executable or not python_executable.is_file() or not os.access(python_executable, os.X_OK):
            errors.append("请选择专用 Conda 环境中的可执行 python 文件。")

        dataset_path = self._resolved(request.dataset) if request.dataset else Path()
        if not request.dataset:
            errors.append("请选择数据集 YAML 文件。")
        else:
            errors.extend(self._dataset_errors(dataset_path))

        model_path = self._resolved(request.model) if request.model else Path()
        if not request.resume_checkpoint:
            if not request.model or not model_path.is_file():
                errors.append("请选择存在的预训练 .pt 权重或 YOLOv5 模型 .yaml 文件。")
            elif model_path.suffix.lower() not in {".pt", ".yaml", ".yml"}:
                errors.append("模型来源只支持 .pt 权重或 .yaml/.yml 模型结构。")

        options = dict(request.options)
        hyp = str(options.get("hyp", "")).strip()
        if hyp and not request.resume_checkpoint and not self._resolved(hyp).is_file():
            errors.append(f"超参数 YAML 不存在：{hyp}")
        try:
            epochs = int(options.get("epochs", 100))
            batch_size = int(options.get("batch_size", 16))
            imgsz = int(options.get("imgsz", 640))
            workers = int(options.get("workers", 4))
            patience = int(options.get("patience", 100))
        except (TypeError, ValueError):
            errors.append("轮数、批次、图像尺寸、workers 和 patience 必须为整数。")
            epochs = batch_size = imgsz = workers = patience = 0
        if epochs < 1:
            errors.append("训练轮数必须大于 0。")
        if batch_size < 1:
            errors.append("批次大小必须大于 0。")
        if imgsz < 32:
            errors.append("输入尺寸不能小于 32。")
        if workers < 0:
            errors.append("workers 不能为负数。")
        if patience < 0:
            errors.append("EarlyStopping patience 不能为负数。")
        if options.get("optimizer", "SGD") not in {"SGD", "Adam", "AdamW"}:
            errors.append("优化器只支持 SGD、Adam 或 AdamW。")

        output_dir = self._resolved(request.output_dir) if request.output_dir else None
        if output_dir is None:
            errors.append("请选择训练输出目录。")
        else:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                probe = output_dir / ".msm_write_check"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                errors.append(f"训练输出目录不可写：{exc}")

        if request.resume_checkpoint:
            checkpoint = self._resolved(request.resume_checkpoint)
            if not checkpoint.is_file():
                errors.append(f"恢复检查点不存在：{checkpoint}")
            elif checkpoint.name != "last.pt":
                errors.append("仅支持从训练产物 weights/last.pt 恢复。")
            if not (checkpoint.parent.parent / "opt.yaml").is_file():
                errors.append("检查点缺少 opt.yaml，无法恢复原训练参数。")
            if checkpoint.is_file() and request.python_executable:
                trainer_root = self._resolved(request.repo_root) / "backend-code"
                inspect = (
                    "import sys,torch; sys.path.insert(0,sys.argv[2]); "
                    "p=sys.argv[1]; "
                    "exec('try:\\n c=torch.load(p,map_location=\\\"cpu\\\",weights_only=False)\\n'"
                    "+ 'except TypeError:\\n c=torch.load(p,map_location=\\\"cpu\\\")'); "
                    "assert isinstance(c,dict) and c.get('optimizer') is not None "
                    "and int(c.get('epoch',-1)) >= 0, 'checkpoint has no optimizer state'"
                )
                try:
                    check = subprocess.run(
                        [str(self._resolved(request.python_executable)), "-c", inspect,
                         str(checkpoint), str(trainer_root)],
                        capture_output=True, text=True, timeout=90, check=False,
                        env={**os.environ, "YOLOv5_AUTOINSTALL": "false"},
                    )
                    if check.returncode != 0:
                        errors.append(
                            "last.pt 不含可恢复的优化器状态，可能已完成训练或检查点不完整；"
                            "请选择一个已停止且保留优化器状态的任务。")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"无法检查恢复检查点：{exc}")

        if errors:
            return errors

        environment = os.environ.copy()
        environment.update({"YOLOv5_AUTOINSTALL": "false", "PYTHONUNBUFFERED": "1"})
        command = [str(python_executable), "-u", str(repo_root / "backend-code" / "train.py"), "--help"]
        try:
            result = subprocess.run(command, cwd=str(repo_root / "backend-code"),
                                    env=environment, capture_output=True, text=True,
                                    timeout=45, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f"训练环境预检失败：{exc}"]
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[-1] if detail else "训练入口导入失败。"
            return [f"训练环境依赖未就绪：{message}。请在专用 Conda 环境安装仓库训练依赖。"]

        device = str(options.get("device", "0"))
        if device.lower() != "cpu":
            probe = (
                "import torch; "
                "assert torch.cuda.is_available(), 'PyTorch 未检测到 CUDA'; "
                "d=torch.device('cuda:0'); x=torch.ones((2,2),device=d); "
                "torch.cuda.synchronize(d); print(torch.cuda.get_device_name(d))"
            )
            try:
                gpu = subprocess.run([str(python_executable), "-c", probe],
                                     cwd=str(repo_root / "backend-code"), env=environment,
                                     capture_output=True, text=True, timeout=45, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return [f"GPU 预检失败：{exc}"]
            if gpu.returncode != 0:
                detail = (gpu.stderr + "\n" + gpu.stdout).strip().splitlines()
                meaningful = next((line.strip() for line in reversed(detail)
                                   if "RuntimeError:" in line or "AssertionError:" in line), None)
                if meaningful is None:
                    meaningful = next((line.strip() for line in detail
                                       if "not compatible" in line.lower()), None)
                message = meaningful or (detail[-1].strip() if detail else "CUDA 张量运算失败。")
                return [f"所选训练环境无法使用 GPU：{message}；可切换 CPU 或配置兼容的 PyTorch/CUDA。"]
        return []

    def build_command(self, request: TrainingRequest) -> list[str]:
        options = dict(request.options)
        python_executable = str(self._resolved(request.python_executable))
        repo_root = self._resolved(request.repo_root)
        output_dir = self._resolved(request.output_dir)
        train_script = repo_root / "backend-code" / "train.py"
        compatibility_entry = Path(__file__).resolve().with_name("yolov5_train_entry.py")
        command = [python_executable, "-u", str(compatibility_entry), str(train_script)]
        if request.resume_checkpoint:
            command.extend(["--resume", str(self._resolved(request.resume_checkpoint))])
            return command

        model = self._resolved(request.model)
        if model.suffix.lower() == ".pt":
            command.extend(["--weights", str(model), "--cfg", ""])
        else:
            command.extend(["--weights", "", "--cfg", str(model)])
        command.extend([
            "--data", str(self._resolved(request.dataset)),
            "--epochs", str(int(options.get("epochs", 100))),
            "--batch-size", str(int(options.get("batch_size", 16))),
            "--imgsz", str(int(options.get("imgsz", 640))),
            "--device", str(options.get("device", "0")),
            "--workers", str(int(options.get("workers", 4))),
            "--optimizer", str(options.get("optimizer", "SGD")),
            "--patience", str(int(options.get("patience", 100))),
            "--project", str(output_dir),
            "--name", str(options.get("name", "train")),
            "--exist-ok",
        ])
        hyp = str(options.get("hyp", "")).strip()
        if hyp:
            command.extend(["--hyp", str(self._resolved(hyp))])
        return command

    def start(self, request: TrainingRequest, *, validated: bool = False) -> str:
        if not validated:
            errors = self.validate(request)
            if errors:
                raise ValueError("\n".join(errors))
        with self._lock:
            if self._active_run_id:
                active = self._records.get(self._active_run_id)
                if active and active.process.poll() is None:
                    raise RuntimeError("当前已有训练任务运行，请先停止或等待其结束。")
            run_id = request.run_id or uuid.uuid4().hex
            if not request.run_id:
                request = TrainingRequest(
                    backend_id=request.backend_id, model=request.model,
                    dataset=request.dataset, options=request.options,
                    repo_root=request.repo_root,
                    python_executable=request.python_executable,
                    output_dir=request.output_dir, run_id=run_id,
                    resume_checkpoint=request.resume_checkpoint,
                )
            env = os.environ.copy()
            env.update({"YOLOv5_AUTOINSTALL": "false", "PYTHONUNBUFFERED": "1"})
            kwargs: dict[str, Any] = {}
            if os.name != "nt":
                kwargs["start_new_session"] = True
            process = subprocess.Popen(
                self.build_command(request), cwd=str(self._resolved(request.repo_root) / "backend-code"),
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
            )
            output_dir = self._resolved(request.output_dir) / str(request.options.get("name", "train"))
            record = _ProcessRecord(request, process, TrainingState.RUNNING, output_dir)
            self._records[run_id] = record
            self._active_run_id = run_id
            record.reader_thread = threading.Thread(
                target=self._read_output, args=(run_id, record), daemon=True,
                name=f"train-log-{run_id[:8]}")
            record.reader_thread.start()
            return run_id

    def resume(self, request: TrainingRequest, checkpoint: str) -> str:
        resumed = TrainingRequest(
            backend_id=request.backend_id, model=request.model,
            dataset=request.dataset, options=request.options,
            repo_root=request.repo_root,
            python_executable=request.python_executable,
            output_dir=request.output_dir,
            run_id=request.run_id or uuid.uuid4().hex,
            resume_checkpoint=checkpoint,
        )
        return self.start(resumed)

    def runtime_info(self, request: TrainingRequest) -> dict[str, Any]:
        script = (
            "import json,platform,sys,torch; "
            "print(json.dumps({'python':sys.version.split()[0],"
            "'platform':platform.platform(),'torch':torch.__version__,"
            "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        )
        env = os.environ.copy()
        env["YOLOv5_AUTOINSTALL"] = "false"
        result = subprocess.run(
            [str(self._resolved(request.python_executable)), "-c", script],
            capture_output=True, text=True, timeout=30, check=True, env=env,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    @staticmethod
    def _read_output(run_id: str, record: _ProcessRecord) -> None:
        stream = record.process.stdout
        if stream is None:
            return
        for raw_line in stream:
            line = SpinosaurusYOLOv5Backend.ANSI_ESCAPE.sub("", raw_line).rstrip("\r\n")
            if line:
                with record.lock:
                    record.logs.append(line)
                    record.pending_logs.append(line)
        stream.close()

    def cancel(self, run_id: str) -> None:
        record = self._records.get(run_id)
        if record is None or record.process.poll() is not None:
            return
        record.state = TrainingState.STOPPING
        record.cancel_requested_at = time.monotonic()
        try:
            if os.name != "nt":
                os.killpg(record.process.pid, signal.SIGINT)
            else:
                record.process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            try:
                record.process.send_signal(signal.SIGINT)
            except OSError:
                pass

    @staticmethod
    def _latest_metrics(results_csv: Path) -> tuple[int | None, dict[str, float]]:
        if not results_csv.is_file():
            return None, {}
        try:
            with results_csv.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                return None, {}
            row = {str(key).strip(): value for key, value in rows[-1].items() if key}
            epoch = int(float(row.get("epoch", "-1"))) + 1
            metrics = {}
            for key, value in row.items():
                if key == "epoch" or value is None:
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return epoch, metrics
        except (OSError, ValueError, csv.Error):
            return None, {}

    @staticmethod
    def _artifacts(output_dir: Path) -> tuple[str, ...]:
        candidates = (
            output_dir / "weights" / "best.pt",
            output_dir / "weights" / "last.pt",
            output_dir / "results.csv",
            output_dir / "results.png",
            output_dir / "confusion_matrix.png",
            output_dir,
        )
        return tuple(str(path.resolve()) for path in candidates if path.exists())

    def status(self, run_id: str) -> TrainingEvent:
        record = self._records.get(run_id)
        if record is None:
            return TrainingEvent(run_id, TrainingState.FAILED, message="此任务不在当前进程中，请从训练记录恢复。")
        process = record.process
        code = process.poll()
        if code is not None and record.reader_thread is not None:
            record.reader_thread.join(timeout=0.5)
        if code is None and record.cancel_requested_at is not None:
            now = time.monotonic()
            if (record.terminate_sent_at is None
                    and now - record.cancel_requested_at >= self.STOP_GRACE_SECONDS):
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except OSError:
                    pass
                record.terminate_sent_at = now
            elif record.terminate_sent_at is not None and now - record.terminate_sent_at >= 5.0:
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except OSError:
                    pass
        if code is not None:
            record.return_code = code
            if record.cancel_requested_at is not None:
                record.state = TrainingState.STOPPED
            else:
                record.state = TrainingState.COMPLETED if code == 0 else TrainingState.FAILED
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

        epoch, metrics = self._latest_metrics(record.output_dir / "results.csv")
        epochs = int(record.request.options.get("epochs", 0)) or None
        progress = min(1.0, epoch / epochs) if epoch is not None and epochs else 0.0
        with record.lock:
            logs = tuple(record.pending_logs)
            record.pending_logs.clear()
            log_tail = "\n".join(record.logs)
        message = {
            TrainingState.RUNNING: "训练进行中",
            TrainingState.STOPPING: "正在停止训练进程",
            TrainingState.STOPPED: "训练已停止，可从 last.pt 恢复",
            TrainingState.COMPLETED: "训练完成",
            TrainingState.FAILED: f"训练进程退出，返回码 {record.return_code}",
        }.get(record.state, record.state.value)
        return TrainingEvent(
            run_id=run_id, state=record.state, progress=progress,
            epoch=epoch, epochs=epochs, metrics=metrics, message=message,
            artifacts=self._artifacts(record.output_dir), log_lines=logs,
            return_code=record.return_code,
        )

    def log_tail(self, run_id: str) -> str:
        record = self._records.get(run_id)
        if record is None:
            return ""
        with record.lock:
            return "\n".join(record.logs)

    def shutdown(self, timeout: float = 15.0) -> None:
        run_id = self._active_run_id
        if not run_id:
            return
        self.cancel(run_id)
        record = self._records.get(run_id)
        if record is not None:
            try:
                record.process.wait(timeout=max(0.1, timeout))
            except subprocess.TimeoutExpired:
                try:
                    if os.name != "nt":
                        os.killpg(record.process.pid, signal.SIGKILL)
                    else:
                        record.process.kill()
                except OSError:
                    pass
                record.process.wait(timeout=3)
            self.status(run_id)


class UltralyticsTrainingBackend(SpinosaurusYOLOv5Backend):
    """Run Ultralytics detection training in the configured Python environment.

    The implementation shares the process lifecycle, CSV reader and stop logic
    with the legacy adapter, but delegates the actual training call to the
    installed Ultralytics package rather than a separate source checkout.
    """

    backend_id = "ultralytics"

    @staticmethod
    def _latest_metrics(results_csv: Path) -> tuple[int | None, dict[str, float]]:
        """Ultralytics writes epoch numbers as one-based values in results.csv."""
        if not results_csv.is_file():
            return None, {}
        try:
            with results_csv.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                return None, {}
            row = {str(key).strip(): value for key, value in rows[-1].items() if key}
            epoch = int(float(row.get("epoch", "0")))
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key == "epoch" or value is None:
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return epoch, metrics
        except (OSError, ValueError, csv.Error):
            return None, {}

    def validate(self, request: TrainingRequest) -> list[str]:
        errors: list[str] = []
        if request.backend_id != self.backend_id:
            errors.append(f"不支持的训练后端：{request.backend_id}")
        python_executable = self._resolved(request.python_executable) if request.python_executable else Path()
        if not request.python_executable or not python_executable.is_file() or not os.access(python_executable, os.X_OK):
            errors.append("请选择已安装 Ultralytics 的 Conda 环境 Python。")

        dataset_path = self._resolved(request.dataset) if request.dataset else Path()
        if not request.dataset:
            errors.append("请选择数据集 YAML 文件。")
        else:
            errors.extend(self._dataset_errors(dataset_path))

        model = request.model.strip()
        if not model:
            errors.append("请选择本地 .pt/.yaml 模型，或输入 Ultralytics 支持的模型名称。")
        elif Path(model).suffix.lower() in {".pt", ".yaml", ".yml"}:
            model_path = self._resolved(model)
            if not model_path.is_file() and (Path(model).is_absolute() or "/" in model or "\\" in model):
                errors.append(f"模型文件不存在：{model_path}")
            elif not model_path.is_file() and Path(model).suffix.lower() in {".yaml", ".yml"}:
                errors.append(f"模型 YAML 不存在：{model_path}")
        elif "/" in model or "\\" in model:
            errors.append("模型名称无法识别；本地权重请使用 .pt 文件路径。")

        options = dict(request.options)
        cfg_value = str(options.get("cfg", "") or "").strip()
        if cfg_value:
            cfg_path = self._resolved(cfg_value)
            if not cfg_path.is_file():
                errors.append(f"Ultralytics 参数 YAML 不存在：{cfg_path}")
            else:
                try:
                    parsed_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                    if parsed_cfg is not None and not isinstance(parsed_cfg, dict):
                        errors.append("Ultralytics 参数 YAML 顶层必须是键值映射。")
                except (OSError, yaml.YAMLError) as exc:
                    errors.append(f"无法读取 Ultralytics 参数 YAML：{exc}")
        output_dir = self._resolved(request.output_dir) if request.output_dir else None
        if output_dir is None:
            errors.append("请选择训练输出目录。")
        else:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                probe = output_dir / ".msm_write_check"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                errors.append(f"训练输出目录不可写：{exc}")

        if request.resume_checkpoint:
            checkpoint = self._resolved(request.resume_checkpoint)
            if not checkpoint.is_file() or checkpoint.name != "last.pt":
                errors.append("恢复训练需要存在的 weights/last.pt。")
            if not (checkpoint.parent.parent / "args.yaml").is_file():
                errors.append("检查点目录缺少 args.yaml，无法恢复原训练配置。")

        if errors:
            return errors

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        probe_script = (
            "import json,sys,torch,ultralytics; "
            "from ultralytics.cfg import DEFAULT_CFG_DICT; "
            "opts=json.loads(sys.argv[1]); unknown=sorted(set(opts)-set(DEFAULT_CFG_DICT)-{'name'}); "
            "assert not unknown, '未知训练参数: '+','.join(unknown); "
            "print(json.dumps({'python':sys.version.split()[0],"
            "'torch':torch.__version__,'ultralytics':ultralytics.__version__,"
            "'cuda_available':torch.cuda.is_available()}))"
        )
        try:
            result = subprocess.run(
                [str(python_executable), "-c", probe_script,
                 json.dumps(options, ensure_ascii=False)],
                capture_output=True, text=True, timeout=45, check=False, env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f"Ultralytics 环境预检失败：{exc}"]
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[-1] if detail else "Ultralytics 导入失败。"
            return [f"Ultralytics 依赖或参数检查失败：{message}"]

        if str(options.get("device", "0")).lower() not in {"cpu", "none", ""}:
            gpu_probe = (
                "import torch; assert torch.cuda.is_available(), 'PyTorch 未检测到 CUDA'; "
                "x=torch.ones((2,2),device='cuda:0'); torch.cuda.synchronize(); "
                "print(torch.cuda.get_device_name(0))"
            )
            try:
                gpu = subprocess.run([str(python_executable), "-c", gpu_probe],
                                     capture_output=True, text=True, timeout=45,
                                     check=False, env=env)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return [f"GPU 预检失败：{exc}"]
            if gpu.returncode != 0:
                lines = (gpu.stderr + "\n" + gpu.stdout).strip().splitlines()
                message = next((line.strip() for line in reversed(lines)
                                if "RuntimeError:" in line or "AssertionError:" in line),
                               lines[-1].strip() if lines else "CUDA 张量运算失败。")
                return [f"所选环境无法使用 GPU：{message}；可切换 CPU 或修复 PyTorch/CUDA。"]
        return []

    def build_command(self, request: TrainingRequest) -> list[str]:
        entry = Path(__file__).resolve().with_name("ultralytics_train_entry.py")
        return [str(self._resolved(request.python_executable)), "-u", str(entry)]

    def start(self, request: TrainingRequest, *, validated: bool = False) -> str:
        if not validated:
            errors = self.validate(request)
            if errors:
                raise ValueError("\n".join(errors))
        with self._lock:
            if self._active_run_id:
                active = self._records.get(self._active_run_id)
                if active and active.process.poll() is None:
                    raise RuntimeError("当前已有训练任务运行，请先停止或等待其结束。")
            run_id = request.run_id or uuid.uuid4().hex
            if not request.run_id:
                request = TrainingRequest(
                    backend_id=request.backend_id, model=request.model,
                    dataset=request.dataset, options=request.options,
                    repo_root=request.repo_root,
                    python_executable=request.python_executable,
                    output_dir=request.output_dir, run_id=run_id,
                    resume_checkpoint=request.resume_checkpoint,
                )
            output_dir = self._resolved(request.output_dir) / str(request.options.get("name", "train"))
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": request.resume_checkpoint or request.model,
                "dataset": request.dataset,
                "project": request.output_dir,
                "options": dict(request.options),
                "resume": bool(request.resume_checkpoint),
            }
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1",
                        "MSM_ULTRALYTICS_TRAIN_REQUEST": json.dumps(payload, ensure_ascii=False)})
            kwargs: dict[str, Any] = {}
            if os.name != "nt":
                kwargs["start_new_session"] = True
            process = subprocess.Popen(
                self.build_command(request), cwd=str(output_dir.parent), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
            )
            record = _ProcessRecord(request, process, TrainingState.RUNNING, output_dir)
            self._records[run_id] = record
            self._active_run_id = run_id
            record.reader_thread = threading.Thread(
                target=self._read_output, args=(run_id, record), daemon=True,
                name=f"train-log-{run_id[:8]}")
            record.reader_thread.start()
            return run_id

    def runtime_info(self, request: TrainingRequest) -> dict[str, Any]:
        script = (
            "import json,platform,sys,torch,ultralytics; "
            "print(json.dumps({'python':sys.version.split()[0],"
            "'platform':platform.platform(),'torch':torch.__version__,"
            "'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        )
        result = subprocess.run(
            [str(self._resolved(request.python_executable)), "-c", script],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])


def training_event_to_dict(event: TrainingEvent) -> dict[str, Any]:
    value = asdict(event)
    value["state"] = event.state.value
    value["metrics"] = dict(event.metrics)
    value["artifacts"] = list(event.artifacts)
    value["log_lines"] = list(event.log_lines)
    return value


def training_event_from_dict(value: Mapping[str, Any]) -> TrainingEvent:
    return TrainingEvent(
        run_id=str(value.get("run_id", "")),
        state=TrainingState(str(value.get("state", TrainingState.QUEUED.value))),
        progress=float(value.get("progress", 0.0)),
        epoch=value.get("epoch"), epochs=value.get("epochs"),
        metrics=dict(value.get("metrics", {})),
        message=str(value.get("message", "")),
        artifacts=tuple(value.get("artifacts", ())),
        log_lines=tuple(value.get("log_lines", ())),
        return_code=value.get("return_code"),
    )
