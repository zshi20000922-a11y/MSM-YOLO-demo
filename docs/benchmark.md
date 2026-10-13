# ARD100 video-level evaluation / ARD100 视频级评测

This is a sanitized aggregate from a local experiment. The raw ARD100 dataset, sampled frames, checkpoint, per-frame outputs, and runtime database are **not** published. These numbers are evidence for this specific protocol, not a claim of general model performance.

此页仅保留本地实验的汇总指标。原始数据、抽帧、权重、逐帧结果和数据库均不发布；指标只适用于下述实验协议。

## Protocol / 协议

- Official ARD100 video-level test partition; 35 source videos, evaluated on grouped labeled sampled frames.
- Same locally trained Ultralytics YOLO11n detection checkpoint in all rows; model input and ROI size 640 px.
- Point Precision/Recall confidence 0.25; detection collection threshold 0.001 for AP. Candidate recall uses IoU ≥ 0.5. FPS is end-to-end on the desktop host for sampled frames, including motion work and detection.
- The checkpoint is omitted under the repository's asset policy. Exact reproduction requires a compatible checkpoint and the same dataset split and manifest.

## Results / 结果

| Pipeline / 流程 | Precision | Recall | mAP50 | mAP50:95 | Candidate IoU recall | E2E FPS |
|---|---:|---:|---:|---:|---:|---:|
| Full-frame YOLO | 0.3566 | 0.1492 | 0.1250 | 0.0543 | — | 121.98 |
| MOG2 + ROI YOLO | 0.4862 | 0.3140 | 0.2925 | 0.1254 | 0.0184 | 19.18 |
| Frame difference + ROI YOLO | 0.4851 | 0.3132 | 0.2914 | 0.1250 | 0.0066 | 33.58 |
| GMC + 2-frame difference + ROI YOLO | 0.4847 | 0.3109 | 0.2893 | 0.1237 | 0.0135 | 19.04 |
| GMC + 3-frame difference + ROI YOLO | 0.4866 | 0.3136 | 0.2923 | 0.1251 | 0.0098 | 17.69 |
| GMC + ROI + track filtering | 0.4838 | 0.3084 | 0.2867 | 0.1228 | 0.0098 | 17.67 |

## Interpretation / 解读

All motion rows also change ROI scheduling and image scale. The difference from full-frame YOLO is the **combined pipeline effect**, not an isolated causal gain from the motion algorithm. Candidate IoU recall is low despite higher final recall, so proposal quality remains a research problem. The speed comparison is for sampled-frame desktop execution; it is neither original-video throughput nor RK3588 performance. AP_small and tracking identity metrics were not measured.

运动组同时改变了 ROI 调度和目标放大尺度，不能把相对全图 YOLO 的差值归因于运动算法本身。候选框 IoU 召回偏低，仍需改进。此处速度是 PC 上对抽样帧的端到端测量，不能外推到原视频吞吐或 RK3588。未测 AP_small 和轨迹身份指标。
