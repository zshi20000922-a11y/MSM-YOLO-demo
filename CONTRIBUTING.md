# Contributing / 参与贡献

Thank you for improving MSM-YOLO-demo. Issues with a short reproduction recipe, platform details, and anonymized logs are welcome. Please avoid uploading raw videos, datasets, model weights, local databases, calibration sets, or credentials.

感谢参与改进。提交问题时请附复现步骤、运行环境和脱敏日志；请勿上传原始视频、数据集、模型权重、实验数据库、量化校准集或凭据。

## Local workflow / 本地开发

1. Install Python 3.10 and dependencies as described in [docs/setup.md](docs/setup.md).
2. Create a branch for one focused change. Keep motion, detection, and tracking backends behind the interfaces in `msm_lab/interfaces.py`.
3. Run `python -m unittest discover -s tests -v` before opening a pull request.
4. Describe the behavior, affected algorithms/backends, and any data or model assumptions. If metrics change, state the dataset split, checkpoint, thresholds, hardware, and measurement method.

本地流程：按环境文档安装依赖；在独立分支修改；保持 `msm_lab/interfaces.py` 的统一契约；运行现有测试；提交 PR 时说明数据划分、权重、阈值、硬件与测量方法。

## Scope / 范围

This repository is an experimental workbench for motion-guided object detection. Please do not describe changes as a new YOLO architecture unless the training/model code actually changes. RKNN desktop conversion and RK3588 on-device inference are separate claims and require separate evidence.

本仓库是运动引导检测实验工作台。未经模型结构修改与验证，请勿将工程集成描述为新 YOLO 网络；PC 端 RKNN 转换与 RK3588 板端推理也需要分别验证。

By contributing, you agree that your contribution is licensed under the repository's AGPL-3.0 license.
