"""Isolated worker entry point for ONNX export and RKNN-Toolkit2 builds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(args: argparse.Namespace) -> int:
    from ultralytics import YOLO
    import onnx

    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"导出模型：{weights}", flush=True)
    model = YOLO(str(weights))
    result = model.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset,
        simplify=bool(args.simplify), dynamic=False, device="cpu",
    )
    exported = Path(str(result)).expanduser().resolve()
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics 未生成 ONNX 文件：{exported}")
    if exported != output:
        shutil.copy2(exported, output)
    print(f"校验 ONNX 图：{output}", flush=True)
    graph = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(graph)
    print(f"ONNX 导出完成：{output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)
    return 0


def convert_rknn(args: argparse.Namespace) -> int:
    from rknn.api import RKNN

    onnx_path = Path(args.onnx).resolve()
    output = Path(args.output).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 文件不存在：{onnx_path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    calibration_list = None
    if args.precision == "int8_ptq":
        root = Path(args.calibration_dir).resolve()
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        images = sorted(path for path in root.rglob("*")
                        if path.is_file() and path.suffix.lower() in suffixes)
        if args.calibration_limit > 0:
            images = images[:args.calibration_limit]
        if len(images) < 8:
            raise ValueError(f"INT8 PTQ 至少需要 8 张校准图片，当前找到 {len(images)} 张。")
        calibration_list = Path(args.calibration_list).resolve()
        calibration_list.parent.mkdir(parents=True, exist_ok=True)
        calibration_list.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
        print(f"校准图片：{len(images)} 张；清单：{calibration_list}", flush=True)

    rknn = RKNN(verbose=True)
    try:
        print(f"配置目标平台：{args.target}", flush=True)
        ret = rknn.config(
            mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            quant_img_RGB2BGR=False,
            target_platform=args.target,
            quantized_algorithm=args.algorithm,
            quantized_method=args.method,
            float_dtype="float16",
        )
        if ret != 0:
            raise RuntimeError(f"RKNN config 失败，返回码 {ret}")
        print(f"加载 ONNX：{onnx_path}", flush=True)
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"RKNN 加载 ONNX 失败，返回码 {ret}")
        quantize = args.precision == "int8_ptq"
        print("开始转换构建（耗时与模型规模、Toolkit 版本和主机配置有关）…", flush=True)
        ret = rknn.build(
            do_quantization=quantize,
            dataset=str(calibration_list) if calibration_list else None,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN 构建失败，返回码 {ret}")
        print(f"导出 RKNN：{output}", flush=True)
        ret = rknn.export_rknn(str(output))
        if ret != 0:
            raise RuntimeError(f"RKNN 导出失败，返回码 {ret}")
    finally:
        rknn.release()

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_platform": args.target,
        "precision": args.precision,
        "quantized_algorithm": args.algorithm,
        "quantized_method": args.method,
        "rknn_toolkit2_version": importlib.metadata.version("rknn-toolkit2"),
        "calibration_list": str(calibration_list) if calibration_list else None,
        "calibration_image_count": len(images) if args.precision == "int8_ptq" else 0,
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "rknn_path": str(output),
        "rknn_sha256": sha256(output),
        "rknn_bytes": output.stat().st_size,
        "board_runtime_validation": "not_run",
    }
    metadata_path = output.with_suffix(".deployment.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RKNN 转换完成：{output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)
    print(f"部署快照：{metadata_path}", flush=True)
    print("注意：PC 端成功构建不等于 RK3588 板端已验证。", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--weights", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--imgsz", type=int, required=True)
    export_parser.add_argument("--opset", type=int, required=True)
    export_parser.add_argument("--simplify", type=int, choices=(0, 1), required=True)
    export_parser.set_defaults(run=export_onnx)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--onnx", required=True)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--target", default="rk3588")
    convert_parser.add_argument("--precision", choices=("fp16", "int8_ptq"), required=True)
    convert_parser.add_argument("--algorithm", choices=("normal", "mmse"), default="normal")
    convert_parser.add_argument("--method", choices=("channel", "layer"), default="channel")
    convert_parser.add_argument("--calibration-dir", default="")
    convert_parser.add_argument("--calibration-limit", type=int, default=0)
    convert_parser.add_argument("--calibration-list", default="")
    convert_parser.set_defaults(run=convert_rknn)
    args = parser.parse_args()
    try:
        return int(args.run(args))
    except Exception as exc:
        print(f"部署步骤失败：{exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
