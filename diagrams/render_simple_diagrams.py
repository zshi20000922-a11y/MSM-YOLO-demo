#!/usr/bin/env python3
"""Generate two simplified, slide-ready architecture diagrams."""

from html import escape
from pathlib import Path
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
W, H = 1920, 1080
FONT = "Noto Sans CJK SC, sans-serif"
INK = "#18324F"
MUTED = "#52657C"
ARROW = "#6B7F95"


def base(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="%s"/></marker></defs>' % ARROW,
        f'<rect width="{W}" height="{H}" fill="#F7F9FC"/>',
        f'<text x="90" y="105" font-family="{FONT}" font-size="48" font-weight="700" fill="{INK}">{escape(title)}</text>',
        f'<text x="92" y="155" font-family="{FONT}" font-size="25" fill="{MUTED}">{escape(subtitle)}</text>',
        '<line x1="90" y1="195" x2="1830" y2="195" stroke="#DCE4EE" stroke-width="2"/>',
    ]


def box(s: list[str], x: int, y: int, w: int, h: int, title: str,
        detail: str = "", fill: str = "#EAF2FF", stroke: str = "#78A5E2",
        title_size: int = 28, detail_size: int = 20) -> None:
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    cy = y + h // 2 - (10 if detail else 0)
    s.append(f'<text x="{x+w//2}" y="{cy}" text-anchor="middle" font-family="{FONT}" font-size="{title_size}" font-weight="700" fill="{INK}">{escape(title)}</text>')
    if detail:
        s.append(f'<text x="{x+w//2}" y="{cy+42}" text-anchor="middle" font-family="{FONT}" font-size="{detail_size}" fill="{MUTED}">{escape(detail)}</text>')


def arrow(s: list[str], x1: int, y1: int, x2: int, y2: int, color: str = ARROW,
          dashed: bool = False) -> None:
    dash = ' stroke-dasharray="10 9"' if dashed else ""
    s.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="4" marker-end="url(#arrow)"{dash}/>')


def text(s: list[str], x: int, y: int, value: str, size: int = 20,
         color: str = MUTED, weight: int = 500, anchor: str = "middle") -> None:
    s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')


def save(s: list[str], filename: str) -> None:
    s.append("</svg>")
    svg = OUT / filename
    svg.write_text("\n".join(s), encoding="utf-8")
    inkscape = shutil.which("inkscape")
    if inkscape:
        subprocess.run([inkscape, str(svg), "--export-type=png",
                        f"--export-filename={svg.with_suffix('.png')}",
                        f"--export-width={W}"], check=True)


def motion() -> None:
    s = base("动态区域选择", "利用帧间运动信息生成下一帧局部检测区域")
    y, h, w = 390, 190, 250
    xs = [90, 400, 710, 1020, 1330, 1640]
    stages = [
        ("相邻视频帧", "前一帧 + 当前帧", "#EAF2FF", "#78A5E2"),
        ("运动补偿", "特征匹配 · 帧对齐", "#EAF2FF", "#78A5E2"),
        ("运动提取", "帧差 · 去噪", "#FFF5E0", "#D5A03D"),
        ("候选筛选", "运动区域 · 外观相关", "#F1ECFF", "#9275D6"),
        ("生成 ROI", "目标周围 640×640", "#E8F5ED", "#55A177"),
        ("局部检测", "YOLO · 阈值 0.6", "#E8F5ED", "#55A177"),
    ]
    for x, (title, detail, fill, stroke) in zip(xs, stages):
        box(s, x, y, w, h, title, detail, fill, stroke, 26, 19)
    for x in xs[:-1]:
        arrow(s, x+w+9, y+h//2, x+310-9, y+h//2)
    # Feedback loop: confirmed detections guide next frame's ROI choice.
    s.append(f'<path d="M1765 610 V755 H835 V610" fill="none" stroke="#9275D6" stroke-width="4" stroke-dasharray="12 10" marker-end="url(#arrow)"/>')
    text(s, 1300, 705, "确认框更新目标模板，辅助下一帧候选筛选", 21, "#684DA7", 600)
    text(s, 960, 900, "输出：下一帧优先搜索区域（640×640）", 24, INK, 700)
    save(s, "03_motion_region_simple.svg")


def system() -> None:
    s = base("无人机小目标检测系统框架", "动态区域选择负责确定检测范围，现有 YOLO 检测后由 ByteTrack 关联目标")
    # Main pipeline
    y, h, w = 390, 200, 300
    xs = [60, 420, 780, 1140, 1500]
    stages = [
        ("视频输入", "无人机视频 / 摄像头", "#EAF2FF", "#78A5E2"),
        ("区域选择", "全图 / 局部 ROI / 运动候选", "#FFF5E0", "#D5A03D"),
        ("YOLO 检测", "现有 best.pt · conf 0.6", "#E8F5ED", "#55A177"),
        ("ByteTrack", "目标关联 · 分配 Track ID", "#FCEAF4", "#D45C9B"),
        ("结果输出", "检测框 · 目标编号", "#EAF2FF", "#78A5E2"),
    ]
    for x, (title, detail, fill, stroke) in zip(xs, stages):
        box(s, x, y, w, h, title, detail, fill, stroke, 28, 19)
    for x in xs[:-1]:
        arrow(s, x+w+10, y+h//2, x+360-10, y+h//2)
    # Small state-machine strip, deliberately summarized.
    text(s, 960, 715, "检测区域状态切换", 28, INK, 700)
    sy, sh, sw = 805, 115, 410
    state_xs = [180, 755, 1330]
    states = [
        ("全图搜索", "未检出 / 定期重捕获", "#EAF2FF", "#78A5E2"),
        ("局部检测", "目标已定位", "#E8F5ED", "#55A177"),
        ("运动搜索", "局部漏检时寻找候选", "#FFF5E0", "#D5A03D"),
    ]
    for x, (title, detail, fill, stroke) in zip(state_xs, states):
        box(s, x, sy, sw, sh, title, detail, fill, stroke, 25, 18)
    arrow(s, 590, sy+sh//2, 745, sy+sh//2)
    text(s, 665, sy+sh//2-25, "检出", 16, MUTED, 600)
    arrow(s, 1165, sy+sh//2, 1320, sy+sh//2)
    text(s, 1240, sy+sh//2-25, "漏检", 16, MUTED, 600)
    s.append(f'<path d="M1535 {sy+sh} V970 H385 V{sy+sh}" fill="none" stroke="#D5A03D" stroke-width="3" stroke-dasharray="10 9" marker-end="url(#arrow)"/>')
    text(s, 960, 940, "发现候选后重新检测；连续失败则返回全图搜索", 18, MUTED, 500)
    save(s, "04_system_framework_simple.svg")


if __name__ == "__main__":
    motion()
    system()
    print(f"Wrote simplified diagrams to {OUT}")
