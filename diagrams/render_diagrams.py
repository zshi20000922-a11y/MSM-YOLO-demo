#!/usr/bin/env python3
"""Generate slide-sized SVG architecture diagrams for the MSM-YOLO demo."""

from __future__ import annotations

from html import escape
from pathlib import Path
import shutil
import subprocess


OUT = Path(__file__).resolve().parent
W, H = 1920, 1080
FONT = "Noto Sans CJK SC, sans-serif"


def start(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B"/></marker>',
        '<marker id="arrowGreen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#3E9B70"/></marker>',
        '<marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9272D4"/></marker>',
        '<filter id="shadow" x="-10%" y="-10%" width="130%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="8" flood-color="#18324F" flood-opacity="0.08"/></filter>',
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="#F6F8FC"/>',
        f'<text x="70" y="76" font-family="{FONT}" font-size="42" font-weight="700" fill="#142B45">{escape(title)}</text>',
        f'<text x="72" y="119" font-family="{FONT}" font-size="21" fill="#64748B">{escape(subtitle)}</text>',
        '<line x1="70" y1="145" x2="1850" y2="145" stroke="#DCE4EE" stroke-width="2"/>',
    ]


def card(svg: list[str], x: int, y: int, w: int, h: int, step: str,
         title: str, lines: list[str], fill: str, stroke: str,
         title_color: str = "#172B4D", dashed: bool = False) -> None:
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="2.5" filter="url(#shadow)"{dash}/>')
    svg.append(f'<rect x="{x+18}" y="{y+17}" width="44" height="34" rx="12" fill="{stroke}"/>')
    svg.append(f'<text x="{x+40}" y="{y+41}" text-anchor="middle" font-family="{FONT}" font-size="17" font-weight="700" fill="#FFFFFF">{escape(step)}</text>')
    svg.append(f'<text x="{x+78}" y="{y+43}" font-family="{FONT}" font-size="22" font-weight="700" fill="{title_color}">{escape(title)}</text>')
    line_y = y + 84
    for line in lines:
        svg.append(f'<text x="{x+26}" y="{line_y}" font-family="{FONT}" font-size="17" fill="#42566F">{escape(line)}</text>')
        line_y += 31


def label(svg: list[str], x: int, y: int, text: str, color: str = "#52657C",
          size: int = 15, anchor: str = "middle") -> None:
    svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" font-weight="600" fill="{color}">{escape(text)}</text>')


def path(svg: list[str], d: str, color: str = "#64748B", width: int = 3,
         marker: str = "arrow", dashed: bool = False) -> None:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{marker})"{dash}/>')


def finish(svg: list[str], filename: str) -> None:
    svg.append("</svg>")
    svg_path = OUT / filename
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    inkscape = shutil.which("inkscape")
    if inkscape:
        png_path = svg_path.with_suffix(".png")
        subprocess.run([
            inkscape, str(svg_path), "--export-type=png",
            f"--export-filename={png_path}", f"--export-width={W}",
        ], check=True)


def motion_diagram() -> None:
    s = start(
        "动态区域选择｜论文 3.3 运动区域提取",
        "相邻帧运动补偿 → 帧差候选提取 → 外观相关筛选 → 为下一帧生成局部检测 ROI",
    )
    # Row 1: estimate and compensate camera motion.
    xs = [60, 520, 980, 1440]
    w, h, y1, y2, y3 = 410, 195, 185, 440, 695
    card(s, xs[0], y1, w, h, "1", "连续帧输入", [
        "前一帧 I(t−1)     当前帧 I(t)",
        "转为灰度并提取角点特征",
        "同一时刻的画面进入配准",
    ], "#EAF2FF", "#78A5E2")
    card(s, xs[1], y1, w, h, "2", "特征点匹配", [
        "Shi–Tomasi 角点检测",
        "LK 光流跟踪相邻帧特征点",
        "生成前后帧对应点集合",
    ], "#EAF4FF", "#67A3E6")
    card(s, xs[2], y1, w, h, "3", "相机运动估计", [
        "RANSAC 剔除误匹配点",
        "估计单应矩阵 H",
        "用背景运动描述全局位移",
    ], "#EAF4FF", "#67A3E6")
    card(s, xs[3], y1, w, h, "4", "前帧变换对齐", [
        "将 I(t−1) 变换到 I(t) 坐标",
        "W(I(t−1), H)",
        "补偿相机运动，突出独立运动",
    ], "#E2F0FF", "#4588D0")
    for x in (470, 930, 1390):
        path(s, f"M{x} 283 H{x+42}")

    # Serpentine row 2: difference image preprocessing, right-to-left.
    card(s, xs[3], y2, w, h, "5", "帧间差分", [
        "D(t) = | I(t) − W(I(t−1), H) |",
        "相机补偿后计算像素差异",
        "得到运动响应图",
    ], "#FFF8E8", "#E7B34E", "#714A12")
    card(s, xs[2], y2, w, h, "6", "自适应阈值", [
        "根据局部像素统计动态设阈值",
        "阈值约为局部均值 + 2σ",
        "demo 另加 8 灰度级偏置",
    ], "#FFF8E8", "#E7B34E", "#714A12")
    card(s, xs[1], y2, w, h, "7", "形态学去噪", [
        "开运算去除零散噪点",
        "闭运算连接断裂区域 / 填洞",
        "整理二值运动掩膜",
    ], "#FFF3D6", "#D89B2B", "#714A12")
    card(s, xs[0], y2, w, h, "8", "连通区域筛选", [
        "提取二值图连通分量",
        "按面积与尺寸过滤噪声",
        "输出运动候选框与区域",
    ], "#FFF3D6", "#D89B2B", "#714A12")
    path(s, "M1645 380 V410 H1645 V440", color="#B17B1B")
    label(s, 1685, 414, "配准帧", "#9A6A19", 14, "start")
    for left_x, right_x in zip(xs, xs[1:]):
        path(s, f"M{right_x} 538 H{left_x+w}", color="#B17B1B")

    # Row 3: target template correlation and ROI delivery.
    card(s, xs[0], y3, w, h, "9", "候选区域 + 目标模板", [
        "运动候选来自连通区域筛选",
        "模板来自上一帧模型确认框",
        "初始无模板时按运动优先级筛选",
    ], "#F3EEFF", "#9D83DE", "#513A88")
    card(s, xs[1], y3, w, h, "10", "相关性筛选与排序", [
        "比较候选区域与目标外观模板",
        "demo 用灰度 NCC 做轻量近似",
        "论文原方法为特征图相关卷积",
    ], "#F0E9FF", "#9072D4", "#513A88")
    card(s, xs[2], y3, w, h, "11", "生成局部 ROI", [
        "选择高相关性 / 优先级最高候选",
        "以候选中心扩展为 640×640",
        "将局部坐标映射回原帧坐标",
    ], "#EAF6EF", "#58A57A", "#286849")
    card(s, xs[3], y3, w, h, "12", "送入下一帧", [
        "局部 YOLO 检测与候选复核",
        "模型确认后更新目标模板",
        "漏检则继续运动搜索 / 全图重捕获",
    ], "#DFF3E7", "#3E9B70", "#286849")
    path(s, "M265 635 V695", color="#B17B1B")
    label(s, 306, 674, "候选区域", "#9A6A19", 14, "start")
    for x in (470, 930, 1390):
        path(s, f"M{x} 793 H{x+42}", color="#4C956E", marker="arrowGreen")
    path(s, "M1645 890 V930 H265 V890", color="#9272D4", marker="arrowPurple", dashed=True)
    label(s, 950, 923, "模型确认框用于刷新外观模板", "#755CB3", 15)
    s.append('<text x="70" y="1010" font-family="Noto Sans CJK SC, sans-serif" font-size="16" fill="#64748B">实现说明：论文 ROI 边长为 160 px；本 demo 按当前要求使用 640×640。灰度 NCC 是对论文特征图相关卷积的工程近似。</text>')
    finish(s, "01_motion_region_selection.svg")


def system_diagram() -> None:
    s = start(
        "完整系统框架｜动态区域选择 + YOLO + ByteTrack",
        "现有 YOLOv5 模型保持不变；全图 / 局部 / 运动候选按状态切换，检测结果逐帧送入 ByteTrack",
    )
    # Shared model badge.
    s.append('<rect x="610" y="170" width="510" height="82" rx="18" fill="#EDF2F7" stroke="#8295AA" stroke-width="2" filter="url(#shadow)"/>')
    label(s, 865, 204, "现有模型  best.pt  ·  YOLOv5", "#253B56", 21)
    label(s, 865, 233, "权重与网络结构不修改 / 不重训", "#64748B", 15)

    # State machine boundary.
    s.append('<rect x="300" y="280" width="1080" height="640" rx="28" fill="#FAFCFF" stroke="#BDCDDE" stroke-width="2.5" stroke-dasharray="12 9"/>')
    label(s, 330, 315, "检测区域选择状态机", "#35536F", 19, "start")

    # Source and state boxes.
    s.append('<rect x="55" y="510" width="200" height="140" rx="22" fill="#EAF2FF" stroke="#78A5E2" stroke-width="2.5" filter="url(#shadow)"/>')
    label(s, 155, 560, "视频 / 摄像头", "#234A78", 20)
    label(s, 155, 594, "逐帧输入", "#4E6885", 17)
    label(s, 155, 625, "1920×1080 示例", "#71839A", 14)

    card(s, 365, 365, 390, 155, "A", "GLOBAL_SEARCH", [
        "全图 YOLO · conf 0.6",
        "初始化搜索 / 周期性重捕获",
    ], "#EAF2FF", "#6E9DDD", "#234A78")
    card(s, 855, 365, 440, 155, "B", "LOCAL_DETECT", [
        "640×640 ROI YOLO · conf 0.6",
        "命中后更新 ROI 中心与外观模板",
    ], "#E8F5ED", "#55A177", "#286849")
    card(s, 365, 665, 390, 180, "C", "MOTION_EXTRACT · 3.3", [
        "LK + RANSAC 对齐与相机补偿",
        "帧差 / 自适应阈值 / 形态学",
        "连通区域 → NCC 近似排序",
    ], "#FFF5E0", "#D5A03D", "#795115")
    card(s, 855, 665, 440, 155, "D", "MOTION_VERIFY", [
        "候选区域裁成 640×640 ROI",
        "调用同一个 YOLO 模型复核",
    ], "#F1ECFF", "#9275D6", "#513A88")

    # State transitions.
    path(s, "M255 580 H320 V442 H365")
    path(s, "M755 442 H855", color="#3E9B70", marker="arrowGreen")
    label(s, 805, 423, "检出 / 生成 ROI", "#357C59", 14)
    path(s, "M560 520 V665", color="#B17B1B")
    label(s, 595, 600, "全图漏检", "#9A6A19", 14, "start")
    path(s, "M1075 520 V665", color="#B17B1B")
    label(s, 1108, 600, "局部漏检", "#9A6A19", 14, "start")
    path(s, "M755 755 H855", color="#B17B1B")
    label(s, 805, 737, "找到运动候选", "#9A6A19", 14)
    path(s, "M1075 665 V565 Q1075 545 1055 545 H970 V520", color="#3E9B70", marker="arrowGreen")
    label(s, 1128, 570, "复核成功", "#357C59", 14, "start")
    path(s, "M855 780 H800 V820 H755", color="#9272D4", marker="arrowPurple")
    label(s, 805, 810, "复核失败", "#755CB3", 14)
    path(s, "M855 420 H820 V335 H560 V365", color="#64748B", dashed=True)
    label(s, 690, 328, "每 15 帧全图重捕获", "#52657C", 14)
    label(s, 330, 890, "连续无候选 / 连续复核失败 → 回全图搜索", "#52657C", 13, "start")

    # Detection aggregation and tracking pipeline.
    s.append('<rect x="1430" y="390" width="205" height="150" rx="20" fill="#E8F5ED" stroke="#55A177" stroke-width="2.5" filter="url(#shadow)"/>')
    label(s, 1532, 438, "检测结果汇聚", "#286849", 19)
    label(s, 1532, 472, "全图 / 局部 / 候选复核", "#426C57", 14)
    label(s, 1532, 501, "原图坐标 · 置信度 · 类别", "#426C57", 14)
    label(s, 1532, 527, "空帧也更新追踪器", "#426C57", 13)
    path(s, "M1380 480 H1430", color="#3E9B70", marker="arrowGreen")
    label(s, 1405, 462, "命中框", "#357C59", 13)

    s.append('<rect x="1665" y="390" width="205" height="150" rx="20" fill="#FCEAF4" stroke="#D45C9B" stroke-width="2.5" filter="url(#shadow)"/>')
    label(s, 1767, 438, "ByteTrack", "#8A3267", 20)
    label(s, 1767, 472, "IoU 关联 + Kalman 预测", "#784466", 14)
    label(s, 1767, 501, "连续追踪 / Track ID", "#784466", 14)
    path(s, "M1635 465 H1665", color="#64748B")
    label(s, 1650, 447, "每帧更新", "#52657C", 12)

    s.append('<rect x="1665" y="655" width="205" height="145" rx="20" fill="#EAF2FF" stroke="#78A5E2" stroke-width="2.5" filter="url(#shadow)"/>')
    label(s, 1767, 705, "标注视频", "#234A78", 20)
    label(s, 1767, 740, "检测框 + Track ID", "#4E6885", 15)
    label(s, 1767, 769, "ROI / 状态信息", "#4E6885", 15)
    path(s, "M1767 540 V655", color="#64748B")

    # Compact settings and visual legend.
    s.append('<rect x="1430" y="645" width="205" height="155" rx="18" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2"/>')
    label(s, 1532, 681, "运行设置", "#35536F", 17)
    label(s, 1450, 713, "ROI 640×640", "#52657C", 14, "start")
    label(s, 1450, 740, "检测阈值 0.6", "#52657C", 14, "start")
    label(s, 1450, 767, "ByteTrack buffer 30", "#52657C", 14, "start")
    path(s, "M1532 540 V610 H1532 V645", color="#A0AEC0", dashed=True)

    s.append('<line x1="70" y1="975" x2="1850" y2="975" stroke="#DCE4EE" stroke-width="2"/>')
    s.append('<text x="70" y="1015" font-family="Noto Sans CJK SC, sans-serif" font-size="16" fill="#64748B">边界说明：检测网络保持原样；运动模板相关使用 demo 灰度 NCC 近似论文特征图卷积。运行验证在 yolo conda 环境 CPU 完成。</text>')
    finish(s, "02_full_system_bytetrack.svg")


if __name__ == "__main__":
    motion_diagram()
    system_diagram()
    print(f"Wrote SVG diagrams to {OUT}")
