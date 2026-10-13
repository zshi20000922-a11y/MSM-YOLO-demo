"""Configuration and command construction for local RK3588 deployment."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DeploymentRequest:
    weights: str
    output_dir: str
    ultralytics_python: str
    rknn_python: str
    onnx_file: str = ""
    imgsz: int = 640
    opset: int = 12
    simplify: bool = True
    precision: str = "int8_ptq"
    calibration_dir: str = ""
    calibration_limit: int = 256
    quantized_algorithm: str = "normal"
    quantized_method: str = "channel"
    target_platform: str = "rk3588"

    def snapshot(self) -> dict:
        return asdict(self)

    @property
    def weights_path(self) -> Path:
        return Path(self.weights).expanduser().resolve()

    @property
    def model_stem(self) -> str:
        if self.weights.strip():
            return self.weights_path.stem
        if self.onnx_file.strip():
            return Path(self.onnx_file).expanduser().stem
        return "model"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def onnx_path(self) -> Path:
        if self.onnx_file.strip():
            return Path(self.onnx_file).expanduser().resolve()
        return self.output_path / f"{self.model_stem}_{self.imgsz}.onnx"

    @property
    def rknn_path(self) -> Path:
        suffix = "int8" if self.precision == "int8_ptq" else "fp16"
        return self.output_path / f"{self.onnx_path.stem}_{self.target_platform}_{suffix}.rknn"


def calibration_images(directory: str | Path, limit: int = 0) -> list[Path]:
    if not str(directory).strip():
        return []
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []
    images = sorted(path for path in root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return images[:limit] if limit > 0 else images


def validate_request(request: DeploymentRequest, *, check_environments: bool = True,
                     export: bool = True, convert: bool = True) -> list[str]:
    errors: list[str] = []
    weights = request.weights_path
    if export:
        if not weights.is_file():
            errors.append(f"模型权重不存在：{weights}")
        elif weights.suffix.lower() != ".pt":
            errors.append("PT→ONNX 导出入口只接受 .pt 权重。")
    if request.imgsz < 32 or request.imgsz % 32:
        errors.append("输入尺寸必须大于等于 32 且为 32 的倍数。")
    if export and request.opset not in {12, 13, 17, 18, 19}:
        errors.append("当前提供的 ONNX opset 选项为 12、13、17、18、19。")
    if request.precision not in {"fp16", "int8_ptq"}:
        errors.append("暂不支持的转换精度选项。")
    if request.quantized_algorithm not in {"normal", "mmse"}:
        errors.append("RKNN-Toolkit2 2.x 当前支持 normal 或 mmse 量化算法。")
    if request.quantized_method not in {"channel", "layer"}:
        errors.append("量化粒度只支持 channel 或 layer。")
    if request.target_platform != "rk3588":
        errors.append("当前部署工作流固定面向 RK3588。")
    output_dir = request.output_path
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".msm_deploy_write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        errors.append(f"输出目录不可写：{exc}")
    if convert and not export and not request.onnx_path.is_file():
        errors.append(f"待转换的 ONNX 文件不存在：{request.onnx_path}")
    if convert and request.precision == "int8_ptq":
        images = calibration_images(request.calibration_dir, request.calibration_limit)
        if len(images) < 8:
            errors.append(f"INT8 PTQ 至少需要 8 张可读校准图片，当前找到 {len(images)} 张。")
    if check_environments:
        environments = []
        if export:
            environments.append(("Ultralytics", request.ultralytics_python, "ultralytics, onnx"))
        if convert:
            environments.append(("RKNN-Toolkit2", request.rknn_python, "rknn.api"))
        for label, executable, module in environments:
            path = Path(executable).expanduser()
            if not path.is_file() or not os.access(path, os.X_OK):
                errors.append(f"{label} Python 不存在或不可执行：{path}")
                continue
            probe_code = (
                "import importlib; "
                + "; ".join(f"importlib.import_module('{name}')" for name in module.split(", "))
                + "; print('environment ready')"
            )
            try:
                result = subprocess.run(
                    [str(path), "-c", probe_code], capture_output=True,
                    text=True, timeout=45, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{label} 环境检查失败：{exc}")
                continue
            if result.returncode:
                detail = (result.stderr or result.stdout).strip().splitlines()
                errors.append(f"{label} 环境缺少可用依赖：{detail[-1] if detail else '未知错误'}")
    return errors


def export_command(request: DeploymentRequest, worker_script: str | Path) -> list[str]:
    return [
        str(Path(request.ultralytics_python).expanduser()), str(Path(worker_script).resolve()),
        "export", "--weights", str(request.weights_path), "--output", str(request.onnx_path),
        "--imgsz", str(request.imgsz), "--opset", str(request.opset),
        "--simplify", "1" if request.simplify else "0",
    ]


def convert_command(request: DeploymentRequest, worker_script: str | Path,
                    onnx_path: str | Path | None = None) -> list[str]:
    command = [
        str(Path(request.rknn_python).expanduser()), str(Path(worker_script).resolve()),
        "convert", "--onnx", str(Path(onnx_path or request.onnx_path).expanduser().resolve()),
        "--output", str(request.rknn_path), "--target", request.target_platform,
        "--precision", request.precision, "--algorithm", request.quantized_algorithm,
        "--method", request.quantized_method,
    ]
    if request.precision == "int8_ptq":
        command.extend([
            "--calibration-dir", str(Path(request.calibration_dir).expanduser().resolve()),
            "--calibration-limit", str(request.calibration_limit),
            "--calibration-list", str(request.output_path / "calibration_images.txt"),
        ])
    return command
