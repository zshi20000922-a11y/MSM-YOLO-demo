# Motion Detection + YOLO：空对空反无人机小目标检测论文调研

> 调研日期：2026-10-09  
> 研究平台：RK3588 / RKNN / RGA / V4L2  
> 目标任务：Air-to-Air UAV Detection（空对空无人机检测）  
> 实验数据集：ARD100  
> 研究重点：Motion Detection（运动检测）、Ego-Motion Compensation（自身运动补偿）、Temporal Fusion（时序融合）、Motion-Guided ROI（运动引导的感兴趣区域检测）

---

## 1. 研究背景与目标

### 1.1 问题定义

空对空无人机检测面临以下挑战：

1. 目标极小：远距离 UAV 可能只有 5×5～20×20 像素。
2. 相机运动：载机运动导致整个背景产生光流。
3. 复杂背景：天空、云层、树林、建筑物和地面纹理。
4. 信息损失：4K 图像缩放至 640 后，微小目标可能只剩 1～3 像素。
5. 算力约束：RK3588 的 CPU、NPU 和内存带宽资源有限。
6. 误警问题：鸟类、云层变化、背景视差和图像噪声均可能产生虚假运动候选。

### 1.2 总体研究目标

构建：

**Global Detection + Motion Proposal + High-Resolution ROI Detection + Temporal Tracking**

推荐系统流程：

```text
4K / 2K Camera
       |
     V4L2
       |
    DMA-BUF
       |
      RGA
       |
       +----------------------+
       |                      |
 Global YOLO             Motion Branch
       |                      |
       |                Motion Compensation
       |                      |
       |                Multi-frame Difference
       |                      |
       |                Motion Proposals
       |                      |
       +----------+-----------+
                  |
             ROI Scheduler
                  |
              RGA Crop
                  |
            ROI YOLO / RKNN
                  |
            Detection Fusion
                  |
          Tracking / Prediction
                  |
             Next-frame ROI
```

核心研究问题：

- 运动信息能否帮助 YOLO 发现单帧漏检的微小 UAV？
- 运动补偿能否抑制载机运动造成的虚假运动？
- 运动候选能否减少高分辨率 YOLO 的计算量？
- 多帧时序融合是否优于简单的帧差候选？
- 哪些方法适合 RK3588 的实时部署？

---

## 2. 六篇核心论文

### Paper 1：MG-VTOD（2023）

**英文标题：** Video Tiny-Object Detection Guided by the Spatial-Temporal Motion Information

**发表：** CVPR Workshops 2023

**论文链接：**

- 官方页面：https://openaccess.thecvf.com/content/CVPR2023W/Anti-UAV/html/Yang_Video_Tiny-Object_Detection_Guided_by_the_Spatial-Temporal_Motion_Information_CVPRW_2023_paper.html
- PDF：https://openaccess.thecvf.com/content/CVPR2023W/Anti-UAV/papers/Yang_Video_Tiny-Object_Detection_Guided_by_the_Spatial-Temporal_Motion_Information_CVPRW_2023_paper.pdf
- DOI：https://doi.org/10.1109/CVPRW59228.2023.00307

**数据集：** Anti-UAV-2021

#### 核心方法

MG-VTOD 受到生物视网膜运动感知机制启发，将连续视频帧配准后构建 Spatio-Temporal Motion Strength Map（时空运动强度图）。

主要流程：

1. 读取连续视频帧。
2. 执行 Frame Alignment（帧间配准）。
3. 构建时空帧序列。
4. 计算 Motion Strength Map（运动强度图）。
5. 将运动响应与图像卷积特征结合。
6. 利用 YOLOv5s 完成目标检测。

论文采用运动强度图与图像卷积响应的特征融合，而不是简单地在 YOLO 后面添加一个运动检测器。

#### 创新点

- 利用多帧信息弥补单帧微小目标纹理不足。
- 使用运动强度图增强目标响应。
- 将 Motion Guidance（运动引导）融入 YOLOv5 特征提取过程。

#### 局限性

- 依赖连续帧配准质量。
- 目标与背景运动差异较小时可能失效。
- 红外数据集结果不能直接推广到可见光 ARD100。

#### RK3588 借鉴方向

先独立计算低成本 Motion Map，再比较：

- RGB-only YOLO
- Motion-enhanced RGB YOLO
- Motion Map + ROI YOLO

**复现优先级：中高。**

---

### Paper 2：MGMD（2024）

**英文标题：** Motion-guided small MAV detection in complex and non-planar scenes

**发表：** Pattern Recognition Letters, Vol. 186, pp. 98–105

**论文链接：**

- 期刊：https://www.sciencedirect.com/science/article/pii/S0167865524002757
- arXiv：https://arxiv.org/abs/2410.10527
- DOI：https://doi.org/10.1016/j.patrec.2024.09.013

**数据集：** ARD-MAV（注意不是 ARD100）

#### 核心方法

MGMD 是一套多阶段 Motion-Guided Detection（运动引导检测）框架。

```text
Video Frames
      |
Frame Alignment
      |
Multi-frame Difference
      |
Motion Feature Enhancement
      |
Motion Candidate Generation
      |
Multi-object Tracking
      |
Trajectory Filtering
      |
Appearance Classification
      |
Local Fine Detection
      |
Final UAV Detection
```

主要包括三个模块：

**模块 A：Motion Feature Enhancement**

- 对齐连续帧。
- 计算多帧差分。
- 聚合微弱运动响应。
- 生成疑似微小 UAV 区域。

**模块 B：Trajectory Filtering**

- 对运动候选进行跨帧关联。
- 根据轨迹特征过滤背景视差导致的虚假候选。
- 降低复杂非平面场景中的误警。

**模块 C：Local Fine Detection**

- 从原始图像裁剪候选区域。
- 使用外观分类器过滤非 UAV 候选。
- 使用改进 YOLOv5s 对局部区域精细检测。

#### 创新点

- 多帧差分增强极小目标运动信号。
- 使用轨迹而不是单帧阈值判断目标真实性。
- 针对非平面背景中的 Motion Parallax（运动视差）设计过滤机制。
- 高分辨率局部检测避免整幅图像缩放导致目标消失。

#### 局限性

- 依赖目标的时序运动可观测性。
- 轨迹建立可能带来检测延迟。
- 运动候选过多时会增加 ROI 检测开销。
- 复杂背景下仍需处理运动补偿残差。

#### RK3588 借鉴方向

最推荐的工程复现方案：

```text
V4L2
  |
Motion Compensation
  |
Multi-frame Difference
  |
Connected Components
  |
Trajectory Filtering
  |
RGA Crop
  |
RKNN YOLO
```

建议重点验证：

- Motion Proposal Recall（运动候选召回率）
- False Positives per Frame（每帧误警数）
- ROI Count per Frame（每帧 ROI 数量）
- End-to-End FPS（端到端帧率）

**复现优先级：最高。**

---

### Paper 3：MSM-YOLO（2026）

**英文标题：** Spatio-Temporal Feature Fusion for Anti-UAV Detection: Integrating Inter-Frame Dynamics and Appearance

**发表：** Sensors, 2026, 26(5), 1492

**论文链接：**

- 全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC12986653/
- PDF：https://pmc.ncbi.nlm.nih.gov/articles/PMC12986653/pdf/sensors-26-01492.pdf
- 期刊：https://www.mdpi.com/1424-8220/26/5/1492

#### 核心方法

论文结合改进 YOLO11 与帧间运动检测。

**空间检测分支：**

- YOLO11
- SPD-Conv
- BiFPN
- 高分辨率检测头

**运动检测分支：**

- LK Optical Flow（LK 光流）
- RANSAC
- Frame Alignment（帧间配准）
- Frame Difference（帧差）
- Morphological Processing（形态学处理）

**最终阶段：**

融合空间检测与运动检测结果。

#### 创新点

- 将外观检测与帧间运动检测结合。
- 使用运动候选补充单帧漏检。
- 将小目标网络改进与 RK3588 工程部署结合。

#### 局限性

- 多个模块同时改进，需要严格消融。
- 模型复杂度与检测精度存在权衡。
- 论文报告的 RK3588 FPS 需要核对测量边界，不能直接视为 4K 视频完整流水线吞吐率。

#### RK3588 借鉴方向

重点参考：

1. Motion Detection 的实现与融合策略。
2. RK3588 模型转换和部署方式。
3. 网络与运动模块的独立消融。
4. 端到端性能分析。

建议先固定 YOLO 模型，仅增加 Motion Branch，测量独立贡献。

**复现优先级：最高。**

---

### Paper 4：A2ANet（2026 在线发表）

**英文标题：** A2ANet: Ego-motion-compensated temporal detection for air-to-air UAVs in moving-platform airborne imagery

**发表：** Expert Systems with Applications，期刊卷期标注为 2027，论文 DOI 为 2026 年编号。

**论文链接：**

- 期刊：https://www.sciencedirect.com/science/article/pii/S0957417426031477
- DOI：https://doi.org/10.1016/j.eswa.2026.134241

**数据集：** ARD100、UAVSwarm

#### 核心方法

A2ANet 是网络内部的 Motion-Aware Temporal Detection（运动感知时序检测）方法。

包含两个关键模块：

**GECM：Global Ego-Motion Compensation Module**

- 估计短时间窗口内的背景运动。
- 使用全局仿射变换近似背景位移。
- 对齐历史帧的特征。

**MGTF：Motion-Guided Temporal Feature Fusion**

融合：

- 当前帧特征。
- 对齐后的历史帧特征。
- 帧间差分响应。
- 粗运动先验。

通过 Learnable Gate（可学习门控）控制历史信息注入强度。

#### 训练机制

论文结合：

- Detection Loss（检测损失）
- Feature Consistency Loss（特征一致性损失）
- Motion Smoothness Regularization（运动平滑正则）
- Affine Regularization（仿射正则）

#### 报告结果

ARD100：

- A2ANet：79.1 ± 0.2% mAP50
- YOLOMG：73.5 ± 0.3% mAP50
- A2ANet：54.4 ± 0.3% mAP50:95

这些是论文自身实验条件下的结果。

#### 创新点

- 先对齐历史特征，再执行时序融合。
- 利用运动残差增强微弱目标。
- 门控机制降低错误历史特征的影响。
- 联合优化检测与运动补偿相关目标。

#### 局限性

- 特征缓存和时序融合增加内存与计算开销。
- RKNN 算子兼容性需要验证。
- 全局仿射模型无法完整描述复杂三维视差。

#### RK3588 借鉴方向

建议设计轻量版本：

```text
Current P2/P3 Features
          +
Aligned Historical Features
          |
     Gated Fusion
          |
      YOLO Head
```

首先在 RTX 5090 上验证收益，再评估 RK3588 部署。

**复现优先级：算法研究最高，工程实现第二阶段。**

---

### Paper 5：Optical Flow + YOLOv5（2023）

**英文标题：** Enhancing UAV Detection in Surveillance Camera Videos through Spatiotemporal Information and Optical Flow

**发表：** Sensors, 2023, 23(13), 6037

**论文链接：**

- 全文：https://www.mdpi.com/1424-8220/23/13/6037
- DOI：https://doi.org/10.3390/s23136037

#### 核心方法

将连续视频中的 Spatiotemporal Information（时空信息）与 Optical Flow（光流）引入 YOLOv5 检测流程。

重点是让检测网络同时利用：

- Spatial Appearance（空间外观）
- Temporal Motion（时序运动）

#### 创新点

- 不再只依赖单帧 RGB 特征。
- 利用运动信息辅助 UAV 检测。
- 提供运动特征与 YOLO 联合建模的实验参考。

#### 局限性

- 主要研究监控摄像机视频。
- 移动载机下背景光流更复杂。
- 光流计算会增加处理延迟。

#### RK3588 借鉴方向

建立光流辅助 YOLO 的对照实验：

- RGB YOLO
- RGB + Optical Flow
- Motion Compensation + Optical Flow + YOLO

具体输入融合方式应在阅读原文方法图和实现细节后确定。

**复现优先级：中。**

---

### Paper 6：TAD（CVPR Workshop 2023）

**英文标题：** A Real-Time and Lightweight Method for Tiny Airborne Object Detection

**发表：** CVPR Workshops 2023

**论文链接：**

- 官方页面：https://openaccess.thecvf.com/content/CVPR2023W/Anti-UAV/html/Lyu_A_Real-Time_and_Lightweight_Method_for_Tiny_Airborne_Object_Detection_CVPRW_2023_paper.html
- PDF：https://openaccess.thecvf.com/content/CVPR2023W/Anti-UAV/papers/Lyu_A_Real-Time_and_Lightweight_Method_for_Tiny_Airborne_Object_Detection_CVPRW_2023_paper.pdf
- DOI：https://doi.org/10.1109/CVPRW59228.2023.00303

**数据集：** AOT、NPS-Drones、Anti-UAV Benchmark

#### 核心方法

TAD 通过局部运动不一致性检测微小飞行目标。

```text
Adjacent Frames
       |
Spatial Feature Extraction
       |
Pixel Correspondence
       |
Local Motion Consistency
       |
Motion Inconsistency Representation
       |
Object Probability Heatmap
       |
Center / Scale Regression
       |
Bounding Boxes
```

#### 创新点

- 不依赖目标本身的丰富纹理。
- 重点建模目标与周围背景之间的运动差异。
- 使用轻量网络生成目标概率热力图。
- 使用独立分支恢复目标位置和尺寸。

#### 与 YOLO 的关系

TAD 本身不是 YOLO 网络。

可将其思想迁移为：

```text
Motion Consistency
        |
Candidate Heatmap
        |
ROI Proposal
        |
YOLO Detection
```

但这属于后续提出的组合方案，并非论文原始实现。

#### 局限性

- 需要额外的运动特征计算。
- 极低对比度或相对静止目标可能缺少运动线索。
- 需要验证在 RK3588 上的真实时延。

**复现优先级：中高。**

---

## 3. 六篇论文横向比较

| 论文 | 运动方法 | 与 YOLO 的关系 | 主要特点 | 推荐用途 |
|---|---|---|---|---|
| MG-VTOD | 多帧运动强度图 | YOLOv5 特征融合 | 增强微弱运动响应 | 多帧运动增强 |
| MGMD | 帧间配准、多帧差分、轨迹过滤 | 局部 YOLOv5s 精检 | 复杂背景、视差抑制 | 工程主方案 |
| MSM-YOLO | 光流、RANSAC、帧差 | 改进 YOLO11 + 决策融合 | 空间与运动互补 | RK3588 参考 |
| A2ANet | 自身运动补偿、特征对齐、门控 | YOLO 特征级时序融合 | 可学习时序增强 | 算法研究 |
| Optical Flow + YOLOv5 | 光流、时空信息 | 网络联合建模 | 运动特征辅助检测 | 基础对照 |
| TAD | 局部运动不一致性 | 独立检测器 | 轻量运动感知 | 运动候选研究 |

**注意：不同论文的实验结果不能直接比较。必须统一数据集、划分、训练配置和评价指标。**

---

## 4. 面向 RK3588 的技术路线

### 第一阶段：传统运动检测

实现：

- Frame Difference
- MOG2
- LK Optical Flow
- RANSAC
- Affine / Homography Alignment
- Multi-frame Difference

目标：

建立可运行的 Motion Proposal Baseline。

### 第二阶段：运动引导 ROI 检测

实现：

```text
Motion Candidate
      |
ROI Expansion
      |
RGA Crop
      |
RKNN YOLO
      |
Coordinate Mapping
      |
NMS / Detection Fusion
```

需要注意：

ROI 应从原始高分辨率图像裁剪，并保留必要上下文。不能将几像素的运动区域直接放大后期待恢复不存在的细节。

### 第三阶段：轨迹过滤

实现：

- Kalman Filter
- Candidate Association
- Trajectory Consistency
- False-positive Filtering
- ROI Prediction

目标：

降低复杂背景误警，提高连续检测稳定性。

### 第四阶段：时序特征融合

参考 A2ANet：

- Feature Alignment
- Temporal Feature Cache
- Motion Residual
- Gated Fusion

先在 RTX 5090 训练和测试，再评估 RKNN 兼容性。

---

## 5. ARD100 实验设计

### 5.1 数据集检查

数据路径：

`<ARD100 数据集目录>`

已整理数据：

`<ARD100 数据集目录>/yolo_dataset`

Agent 必须先检查：

- 数据目录结构。
- train / val / test 数量。
- 图片与标签是否匹配。
- 类别定义。
- 原始视频与图片的对应关系。
- 是否存在相邻视频帧跨数据集划分。
- 是否存在重复或近重复图像。

**不得假定现有 yolo_dataset 已正确按视频序列划分。**

### 5.2 实验矩阵

| ID | 方法 | 实验目的 |
|---|---|---|
| E0 | YOLO Baseline | 单帧检测基线 |
| E1 | MOG2 + YOLO | 背景建模对照 |
| E2 | Frame Difference + YOLO | 基础运动检测 |
| E3 | GMC + Frame Difference + YOLO | 运动补偿贡献 |
| E4 | GMC + Multi-frame Difference + YOLO | 多帧运动增强 |
| E5 | E4 + ROI YOLO | 高分辨率局部检测 |
| E6 | E5 + Trajectory Filtering | 轨迹过滤贡献 |
| E7 | Temporal Feature Fusion + YOLO | 时序特征融合对照 |

GMC = Global Motion Compensation（全局运动补偿）。

### 5.3 评价指标

检测指标：

- Precision
- Recall
- mAP50
- mAP50:95
- AP_small
- Recall for targets <8 px
- Recall for targets 8–16 px
- Recall for targets 16–32 px

运动候选指标：

- Proposal Recall
- Mean Proposals per Frame
- Proposal Processing Time
- False Proposals per Frame

视频指标：

- False Alarms per Minute
- Time to First Detection
- Track Fragmentation
- ID Switches（如评估多目标跟踪）

部署指标：

- CPU Latency
- RGA Latency
- NPU Latency
- End-to-End Latency
- FPS
- Memory Usage

对于 Tiny Recall，需要事先明确目标尺寸的定义，建议统一使用原始图像中的 bounding box 宽高及分组规则。

---

## 6. 推荐研究顺序

### 优先级 P0：先完成工程基线

参考 MGMD 与 MSM-YOLO。

目标：

- 实现运动补偿。
- 实现多帧差分。
- 生成 Motion ROI。
- 通过 RGA 裁剪高分辨率区域。
- 使用 RKNN YOLO 检测。
- 比较有无运动模块的收益。

### 优先级 P1：提高运动候选可靠性

参考 MG-VTOD 与 TAD。

目标：

- 比较帧差与运动强度图。
- 研究局部运动一致性。
- 过滤视差和云层造成的假候选。
- 降低每帧候选 ROI 数量。

### 优先级 P2：研究可学习时序融合

参考 A2ANet。

目标：

- 缓存历史特征。
- 执行特征对齐。
- 设计轻量门控融合。
- 比较精度、FLOPs 和部署延迟。

---

## 7. 给后续 Agent 的执行任务

### Task 1：论文核验与精读

对六篇论文逐一完成：

1. 下载并读取原始 PDF。
2. 提取论文方法框架图。
3. 识别每个模块的输入、输出和算法。
4. 区分作者原始实现与可迁移的工程建议。
5. 整理公开代码仓库；没有核实到仓库时标记为未确认。
6. 记录实验数据集、划分方式、评价指标、计算量和速度测试条件。
7. 生成每篇论文的详细技术笔记。

### Task 2：ARD100 数据集审计

检查本地数据目录及 `yolo_dataset`：

1. 统计样本和标签。
2. 分析视频序列来源。
3. 检查 train/val/test 泄漏。
4. 统计微小目标尺寸分布。
5. 给出规范的数据划分方案。

未经用户确认，不修改原始数据集。

### Task 3：实现 Motion Detection Baseline

依次实现：

1. Two-frame Difference
2. Three-frame Difference
3. MOG2
4. LK Optical Flow + RANSAC
5. Motion Compensation + Multi-frame Difference

输出：

- 可视化视频。
- 候选框。
- 运动残差图。
- 处理时延。
- 候选召回率。

### Task 4：Motion ROI + YOLO 联调

在已有 YOLO 模型基础上：

1. 读取运动候选框。
2. 将候选映射到原始图像坐标。
3. 生成固定尺寸 ROI。
4. 裁剪并送入 YOLO。
5. 将预测结果映射回全图。
6. 融合 Global 与 ROI 检测结果。
7. 记录检测精度和时延。

### Task 5：RK3588 部署

检查：

- RGA Crop / Resize。
- DMA-BUF 共享。
- RKNN 模型兼容性。
- CPU 与 NPU 任务调度。
- ROI 数量限制。
- 异步流水线。
- 端到端 FPS 与延迟。

### Task 6：论文创新点筛选

优先研究：

- 低成本且高召回的 Motion Proposal。
- 复杂非平面背景下的视差抑制。
- 运动置信度驱动的 ROI 调度。
- 轻量时序特征对齐与门控融合。
- 检测精度与计算开销联合优化。

所有创新点都必须通过消融实验验证，不能仅凭模块组合宣称创新。

---

## 8. Agent 工作约束

- 优先使用论文原文和官方实现。
- 不得编造论文结果、代码仓库或实验数据。
- 区分已验证事实、论文作者声明和工程推测。
- 不得直接比较不同数据集上的 mAP。
- 不得将 ARD-MAV 与 ARD100 视为同一数据集。
- 不得使用测试集进行反复调参。
- 视频数据应优先按序列划分，避免相邻帧泄漏。
- 保留原始数据与实验配置。
- 每个实验记录随机种子、训练参数、模型版本和环境版本。
- 部署性能必须明确统计边界，区分模型推理 FPS 和完整系统 FPS。

---

## 9. 最终研究建议

建议先完成：

**GMC + Multi-frame Difference + Motion ROI + YOLO + Trajectory Filtering**

这是当前最适合 RK3588 工程落地的研究基线。

在此基础上，再探索：

**Motion-Aware Temporal Feature Fusion**

目标不是简单堆叠更多网络模块，而是证明：

> 利用可靠的运动信息，在有限计算预算下提高空对空微小无人机的检测召回率，同时控制误警与端到端延迟。

