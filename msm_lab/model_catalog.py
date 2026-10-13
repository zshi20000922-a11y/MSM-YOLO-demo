"""Local, offline model-learning cards for representative YOLO detectors."""

from __future__ import annotations

from html import escape


MODEL_PROFILES = (
    {
        "id": "yolov5_original",
        "name": "YOLOv5 · 原版检测器",
        "yaml": "ultralytics/cfg/models/v5/yolov5.yaml（结构参考）",
        "url": "https://docs.ultralytics.com/yolov5/tutorials/architecture-description/",
        "backbone": "CSPDarknet · C3",
        "neck": "SPPF + FPN / PAN",
        "head": "Anchor-based · 解耦前的耦合检测头",
        "outputs": "P3 / P4 / P5 · 需要 NMS",
        "detail": "原版 best.pt 由独立 YOLOv5 仓库适配器加载。Ultralytics 包内的 YOLOv5u 沿用 C3 主干，但检测头改为 anchor-free；不要把两种 checkpoint 当作同一权重格式。",
        "change": "关键点：C3 CSP 主干、SPPF 快速池化、PAN 自底向上融合；锚框与 NMS 构成传统检测流程。",
    },
    {
        "id": "yolov8",
        "name": "YOLOv8",
        "yaml": "ultralytics/cfg/models/v8/yolov8.yaml",
        "url": "https://docs.ultralytics.com/models/yolov8/",
        "backbone": "CSP · C2f + SPPF",
        "neck": "FPN / PAN 多尺度融合",
        "head": "Anchor-free · 解耦 Detect + DFL",
        "outputs": "P3 / P4 / P5 · 常规 NMS",
        "detail": "C2f 通过分支与特征拼接增强梯度/特征流；检测头分别处理分类与框回归。P3 高分辨率层对小目标较敏感，但最终效果仍取决于数据、训练和输入尺寸。",
        "change": "相对原版 v5：C2f 替代 C3，采用 anchor-free 解耦检测头与 DFL 框回归。",
    },
    {
        "id": "yolov10",
        "name": "YOLOv10",
        "yaml": "ultralytics/cfg/models/v10/yolov10n.yaml",
        "url": "https://docs.ultralytics.com/models/yolov10/",
        "backbone": "C2f + SCDown + PSA",
        "neck": "FPN / PAN · C2fCIB",
        "head": "一致双重分配 · 双检测头",
        "outputs": "One-to-many / one-to-one · 支持 NMS-free",
        "detail": "通过一致双重标签分配同时训练 one-to-many 与 one-to-one 分支；推理可走免 NMS 的 one-to-one 分支，兼顾监督信号和端到端部署。",
        "change": "关键点：SCDown 下采样、C2fCIB/PSA 模块与端到端检测设计。",
    },
    {
        "id": "yolo11",
        "name": "YOLO11",
        "yaml": "ultralytics/cfg/models/11/yolo11.yaml",
        "url": "https://docs.ultralytics.com/models/yolo11/",
        "backbone": "C3k2 + SPPF + C2PSA",
        "neck": "FPN / PAN 多尺度融合",
        "head": "Anchor-free · 解耦 Detect + DFL",
        "outputs": "P3 / P4 / P5 · 常规 NMS",
        "detail": "C3k2 提供可配置的 CSP 特征处理，C2PSA 在高层特征上加入位置/空间注意力；仍沿用多尺度、解耦检测头范式。",
        "change": "相对 v8：主干/颈部改用 C3k2，并在 SPPF 后加入 C2PSA 注意力。",
    },
    {
        "id": "yolo12",
        "name": "YOLO12",
        "yaml": "ultralytics/cfg/models/12/yolo12.yaml",
        "url": "https://docs.ultralytics.com/models/yolo12/",
        "backbone": "C3k2 + A2C2f 注意力",
        "neck": "A2C2f / C3k2 多尺度融合",
        "head": "Anchor-free · Detect",
        "outputs": "P3 / P4 / P5 · 依模型配置后处理",
        "detail": "A2C2f 将区域注意力与 CSP 特征处理结合。注意力结构的计算/显存开销会随设备、输入分辨率和模型规格变化，应使用同一验证集与设备实测。",
        "change": "关键点：在 C3k2 基础上引入区域注意力 A2C2f，扩展长程空间建模。",
    },
    {
        "id": "yolo26",
        "name": "YOLO26",
        "yaml": "ultralytics/cfg/models/26/yolo26.yaml",
        "url": "https://docs.ultralytics.com/models/yolo26/",
        "backbone": "C3k2 + SPPF + C2PSA",
        "neck": "FPN / PAN 多尺度融合",
        "head": "One-to-many / one-to-one 双分支",
        "outputs": "One-to-one 免 NMS；one-to-many 使用 NMS",
        "detail": "YOLO26 可通过 one-to-one 路径进行端到端免 NMS 推理，也保留 one-to-many 路径。当前适配器会读取模型实际输出类型：若输出已完成端到端筛选，IoU 阈值不会参与；若输出原始候选，则按置信度与 IoU 执行 NMS。模型头使用直接框回归，不依赖 DFL。",
        "change": "关键点：端到端 one-to-one 推理选项、移除 DFL，并保留 one-to-many 训练监督。",
    },
)


def architecture_svg(profile: dict) -> bytes:
    """Return a readable stage-level SVG diagram, requiring no network access."""
    title = escape(profile["name"])
    backbone = escape(profile["backbone"])
    neck = escape(profile["neck"])
    head = escape(profile["head"])
    outputs = escape(profile["outputs"])

    def text(x: int, y: int, value: str, size: int = 17,
             color: str = "#24364B", weight: int = 500) -> str:
        return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
                f'font-weight="{weight}" font-family="Noto Sans CJK SC, sans-serif">'
                f'{escape(value)}</text>')

    levels = []
    for index, (label, color) in enumerate((("P3 / 8 · 小目标", "#CFF2E7"),
                                             ("P4 / 16 · 中目标", "#D9E9FF"),
                                             ("P5 / 32 · 大目标", "#E9E0FF"))):
        y = 205 + index * 66
        levels.append(
            f'<rect x="539" y="{y-28}" width="206" height="46" rx="12" fill="{color}"/>'
            + text(556, y, label, 15, "#263C52", 600)
            + f'<path d="M465 {y-5} H530" stroke="#70839A" stroke-width="2.5" marker-end="url(#arrow)"/>'
            + f'<path d="M746 {y-5} H817" stroke="#70839A" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="500" viewBox="0 0 1160 500">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#70839A"/></marker></defs>
<rect width="1160" height="500" rx="24" fill="#FFFFFF"/>
{text(34, 45, title, 24, "#17324E", 700)}
{text(34, 74, "分阶段示意图 · 表示主要特征流，不按层数/通道精确绘制", 14, "#718198", 400)}
<rect x="28" y="160" width="150" height="170" rx="18" fill="#F0F5FA" stroke="#DCE5EF"/>
{text(56, 215, "输入图像", 19, "#25425D", 700)}
{text(54, 252, "RGB · H×W", 15)}
{text(50, 282, "缩放 / 填充", 15)}
<path d="M180 245 H214" stroke="#70839A" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="220" y="130" width="245" height="230" rx="18" fill="#E6F0FB" stroke="#C8D9EA"/>
{text(247, 164, "Backbone · 主干", 19, "#234B73", 700)}
<rect x="243" y="183" width="198" height="46" rx="10" fill="#FFFFFF"/>{text(260, 212, "Conv / 下采样", 16)}
<rect x="243" y="242" width="198" height="46" rx="10" fill="#FFFFFF"/>{text(260, 271, backbone, 15)}
<rect x="243" y="301" width="198" height="38" rx="10" fill="#FFFFFF"/>{text(260, 326, "多层级语义特征", 15)}
<path d="M466 245 H508" stroke="#70839A" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="514" y="130" width="265" height="230" rx="18" fill="#FFF2DD" stroke="#F0D8AF"/>
{text(540, 164, "Neck · 特征融合", 19, "#805A20", 700)}
{''.join(levels)}
<path d="M780 245 H820" stroke="#70839A" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="826" y="130" width="305" height="230" rx="18" fill="#F1EAFE" stroke="#D9C9F2"/>
{text(853, 164, "Head · 检测头", 19, "#5A3C83", 700)}
<rect x="850" y="188" width="255" height="56" rx="10" fill="#FFFFFF"/>{text(865, 222, head, 14, "#344A60", 600)}
<rect x="850" y="260" width="255" height="66" rx="10" fill="#FFFFFF"/>{text(865, 286, "框坐标 + 类别/置信度", 14)}{text(865, 309, outputs, 13, "#526A82", 500)}
<rect x="220" y="390" width="911" height="58" rx="14" fill="#F5F8FB"/>
{text(245, 426, "Backbone", 15, "#234B73", 700)}{text(345, 426, backbone, 15)}
{text(570, 426, "Neck", 15, "#805A20", 700)}{text(630, 426, neck, 15)}
</svg>'''
    return svg.encode("utf-8")
