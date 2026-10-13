# Third-party software and asset licenses / 第三方许可

This repository publishes project-authored source, tests, documentation, and diagrams under [AGPL-3.0](../LICENSE). It does not bundle third-party source checkouts, pretrained model weights, datasets, videos, or RKNN binaries.

本仓库以 AGPL-3.0 发布项目自身代码、测试、文档与图示；不随仓库分发第三方源码仓库、预训练权重、数据集、视频或 RKNN 工具包。

| Dependency / Asset | Role | License and installation boundary |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Mainline detection, ByteTrack, training, ONNX export | Upstream AGPL-3.0 or separate enterprise licensing; install from its official package/source. Read upstream terms for your use case. |
| [Original YOLOv5](https://github.com/ultralytics/yolov5) | Native inference of original-repository checkpoints | Separate checkout with its upstream AGPL-3.0 terms; point `MSM_YOLOV5_ROOT` at it. Its weights and code are not copied here. |
| [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), [PySide6](https://www.qt.io/qt-for-python), [LAP](https://pypi.org/project/lap/), [PyYAML](https://pyyaml.org/) | Runtime libraries | Installed as dependencies; their own package licenses apply. |
| [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) | Optional desktop ONNX → RKNN conversion | External Rockchip toolkit with its own [license](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE); install and use separately. No toolkit wheel or runtime binary is distributed here. |
| ARD100 dataset, user checkpoints, and calibration images | Local experiments | Obtain from their owners under applicable terms; never included in Git. |

The project license does not grant rights to third-party data, weights, or SDKs. Check those sources before redistribution. These notes describe repository boundaries and are not a substitute for the upstream license texts.
