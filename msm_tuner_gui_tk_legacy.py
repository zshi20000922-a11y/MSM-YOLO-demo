#!/usr/bin/env python3
"""Tkinter video tuning interface for the MSM-YOLO demo.

Run with the project's conda ``yolo`` environment. The motion-only mode does
not load YOLO, so it is useful for quickly inspecting alignment and masks.
"""

from __future__ import annotations

import queue
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from msm_yolo_demo import ByteTrackAdapter, MSMVideoPipeline, MotionProposer, YOLOv5Detector


HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = HERE / "phantom03.mp4"
DEFAULT_WEIGHTS = HERE / "best.pt"
RUNS = HERE / "runs" / "tuner"


def draw_text(image: np.ndarray, message: str, y: int = 28,
              color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2, cv2.LINE_AA)


class TunerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MSM-YOLO 动态区域调参工具")
        self.root.geometry("1550x980")
        self.root.minsize(1180, 760)
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.snapshots: list[tuple[int, list[np.ndarray], str]] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.running = False
        self.active_mode = "motion"
        self.active_weights = str(DEFAULT_WEIGHTS)
        families = {name.casefold(): name for name in tkfont.families(root)}
        self.ui_font = next((families[name.casefold()] for name in
                             ("Song Ti", "FangSong Ti", "Mincho")
                             if name.casefold() in families),
                            tkfont.nametofont("TkDefaultFont").actual("family"))
        self.active_device = "0"

        self.video_var = tk.StringVar(value=str(DEFAULT_VIDEO))
        self.weights_var = tk.StringVar(value=str(DEFAULT_WEIGHTS))
        self.start_var = tk.StringVar(value="0")
        self.end_var = tk.StringVar(value="8")
        self.mode_var = tk.StringVar(value="motion")
        self.device_var = tk.StringVar(value="GPU 0")
        self.status_var = tk.StringVar(value="选择视频后点击“开始分析”。建议先用运动模块模式快速调参。")
        self.position_var = tk.IntVar(value=0)
        self.frame_label_var = tk.StringVar(value="预览帧：—")

        self.params: dict[str, tk.Variable] = {
            "threshold_sigma": tk.DoubleVar(value=2.0),
            "threshold_offset": tk.DoubleVar(value=8.0),
            "threshold_window": tk.IntVar(value=21),
            "morph_kernel": tk.IntVar(value=3),
            "open_iterations": tk.IntVar(value=1),
            "close_iterations": tk.IntVar(value=2),
            "min_area": tk.IntVar(value=12),
            "max_proposals": tk.IntVar(value=20),
            "roi_size": tk.IntVar(value=640),
            "confidence": tk.DoubleVar(value=0.60),
            "global_interval": tk.IntVar(value=15),
            "imgsz": tk.IntVar(value=640),
            "track_buffer": tk.IntVar(value=30),
            "show_every": tk.IntVar(value=15),
        }

        self._build()
        self.root.after(150, self._poll_events)

    def _build(self) -> None:
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                     "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name).configure(family=self.ui_font, size=10)
            except tk.TclError:
                pass
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#EEF2F7")
        style.configure("TFrame", background="#EEF2F7")
        style.configure("Card.TFrame", background="#FFFFFF")
        style.configure("TLabel", background="#EEF2F7", foreground="#20334A", font=(self.ui_font, 10))
        style.configure("Card.TLabel", background="#FFFFFF", foreground="#20334A", font=(self.ui_font, 10))
        style.configure("Title.TLabel", font=(self.ui_font, 21, "bold"), foreground="#132A43")
        style.configure("Sub.TLabel", foreground="#65768B", font=(self.ui_font, 10))
        style.configure("Metric.TLabel", background="#FFFFFF", foreground="#132A43", font=(self.ui_font, 19, "bold"))
        style.configure("MetricName.TLabel", background="#FFFFFF", foreground="#65768B", font=(self.ui_font, 9))
        style.configure("TNotebook", background="#EEF2F7", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 8), font=(self.ui_font, 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")], foreground=[("selected", "#245B8F")])
        style.configure("Accent.TButton", font=(self.ui_font, 10, "bold"), padding=(10, 8),
                        background="#2F78B7", foreground="#FFFFFF")
        style.map("Accent.TButton", background=[("active", "#27679F")])
        style.configure("TProgressbar", troughcolor="#DDE5EF", background="#3F91D0", thickness=10)

        outer = ttk.Frame(self.root, padding=(18, 14))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="MSM-YOLO 调参工作台", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="运动区域提取 · 局部检测 · ByteTrack", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))
        self.header_status = ttk.Label(header, text="● 就绪", foreground="#27865B", font=(self.ui_font, 10, "bold"))
        self.header_status.pack(side="right", padx=8)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=390, style="Card.TFrame", padding=10)
        controls.pack(side="left", fill="y", padx=(0, 12))
        controls.pack_propagate(False)
        preview = ttk.Frame(body)
        preview.pack(side="right", fill="both", expand=True)

        tabs = ttk.Notebook(controls)
        tabs.pack(fill="both", expand=True)
        input_tab = ttk.Frame(tabs, padding=12)
        motion_tab = ttk.Frame(tabs, padding=12)
        tracking_tab = ttk.Frame(tabs, padding=12)
        tabs.add(input_tab, text="视频")
        tabs.add(motion_tab, text="运动参数")
        tabs.add(tracking_tab, text="检测追踪")

        ttk.Label(input_tab, text="输入视频", font=(self.ui_font, 12, "bold")).pack(anchor="w", pady=(0, 4))
        self._path_row(input_tab, "视频文件", self.video_var, "video")
        self._path_row(input_tab, "模型权重", self.weights_var, "weights")
        ttk.Label(input_tab, text="处理时间段（秒）", font=(self.ui_font, 11, "bold")).pack(anchor="w", pady=(16, 4))
        range_row = ttk.Frame(input_tab)
        range_row.pack(fill="x", pady=(0, 12))
        ttk.Label(range_row, text="从").pack(side="left")
        ttk.Entry(range_row, textvariable=self.start_var, width=9).pack(side="left", padx=6)
        ttk.Label(range_row, text="到").pack(side="left", padx=(8, 0))
        ttk.Entry(range_row, textvariable=self.end_var, width=9).pack(side="left", padx=6)
        ttk.Label(input_tab, text="分析方式", font=(self.ui_font, 11, "bold")).pack(anchor="w", pady=(8, 4))
        ttk.Radiobutton(input_tab, text="运动模块预览（快速）", variable=self.mode_var,
                        value="motion").pack(anchor="w", pady=3)
        ttk.Label(input_tab, text="只运行对齐、帧差和候选提取，不加载模型。", style="Sub.TLabel", wraplength=330).pack(anchor="w", padx=(24, 0))
        ttk.Radiobutton(input_tab, text="完整系统（YOLO + ByteTrack）", variable=self.mode_var,
                        value="full").pack(anchor="w", pady=(14, 3))
        ttk.Label(input_tab, text="使用当前权重和追踪器处理所选片段。", style="Sub.TLabel", wraplength=330).pack(anchor="w", padx=(24, 0))
        ttk.Label(input_tab, text="建议先选 5–10 秒片段调整运动参数。", style="Sub.TLabel", wraplength=330).pack(anchor="w", pady=(22, 0))

        ttk.Label(motion_tab, text="运动提取", font=(self.ui_font, 12, "bold")).pack(anchor="w")
        ttk.Label(motion_tab, text="选择预设后仍可单独微调参数。", style="Sub.TLabel").pack(anchor="w", pady=(2, 8))
        presets = ttk.Frame(motion_tab)
        presets.pack(fill="x", pady=(0, 8))
        ttk.Button(presets, text="原始", command=lambda: self._preset("original")).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(presets, text="偏灵敏", command=lambda: self._preset("sensitive")).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(presets, text="偏稳健", command=lambda: self._preset("conservative")).pack(side="left", expand=True, fill="x", padx=(3, 0))
        ttk.Separator(motion_tab).pack(fill="x", pady=6)
        self._spin(motion_tab, "阈值标准差系数", "threshold_sigma", 0.0, 4.0, 0.1)
        self._spin(motion_tab, "阈值额外偏置", "threshold_offset", 0, 30, 1)
        self._spin(motion_tab, "局部阈值窗口", "threshold_window", 3, 61, 2)
        self._spin(motion_tab, "形态学核尺寸", "morph_kernel", 1, 9, 2)
        self._spin(motion_tab, "开运算次数", "open_iterations", 0, 3, 1)
        self._spin(motion_tab, "闭运算次数", "close_iterations", 0, 4, 1)
        self._spin(motion_tab, "最小连通区域面积", "min_area", 1, 300, 1)
        self._spin(motion_tab, "候选区域上限", "max_proposals", 1, 50, 1)

        ttk.Label(tracking_tab, text="检测与追踪", font=(self.ui_font, 12, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(tracking_tab, text="推理设备", font=(self.ui_font, 10, "bold")).pack(anchor="w", pady=(4, 2))
        self.device_combo = ttk.Combobox(tracking_tab, textvariable=self.device_var,
                                         values=("GPU 0", "CPU"), state="readonly", width=15)
        self.device_combo.pack(anchor="w", pady=(0, 8))
        self._spin(tracking_tab, "ROI 边长", "roi_size", 320, 1280, 32)
        self._spin(tracking_tab, "检测置信度", "confidence", 0.05, 0.95, 0.05)
        self._spin(tracking_tab, "全图重捕获间隔（帧）", "global_interval", 1, 60, 1)
        self._spin(tracking_tab, "YOLO 输入尺寸", "imgsz", 320, 1280, 32)
        self._spin(tracking_tab, "ByteTrack 缓冲（帧）", "track_buffer", 1, 120, 1)
        ttk.Separator(tracking_tab).pack(fill="x", pady=9)
        self._spin(tracking_tab, "预览抽帧间隔", "show_every", 1, 120, 1)
        ttk.Label(tracking_tab, text="抽帧间隔只影响界面预览密度；视频会逐帧处理。", style="Sub.TLabel", wraplength=330).pack(anchor="w", pady=(6, 12))
        config_buttons = ttk.Frame(tracking_tab)
        config_buttons.pack(fill="x", pady=8)
        ttk.Button(config_buttons, text="保存参数", command=self._save_config).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(config_buttons, text="载入参数", command=self._load_config).pack(side="left", expand=True, fill="x", padx=(3, 0))

        actions = ttk.Frame(controls, style="Card.TFrame", padding=(2, 8, 2, 0))
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="▶  开始分析", command=self.start, style="Accent.TButton")
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(actions, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(7, 0), ipadx=8, ipady=5)
        self.progress = ttk.Progressbar(actions, mode="determinate")
        self.progress.pack(fill="x", pady=(10, 4))
        ttk.Label(actions, textvariable=self.status_var, style="Sub.TLabel", wraplength=350,
                  justify="left").pack(anchor="w", pady=(2, 4))

        metrics = ttk.Frame(preview)
        metrics.pack(fill="x", pady=(0, 8))
        self.metric_vars = [tk.StringVar(value="—") for _ in range(4)]
        metric_names = ["已处理帧数", "候选总数", "检测框总数", "处理速度"]
        for i, (name, value) in enumerate(zip(metric_names, self.metric_vars)):
            card = ttk.Frame(metrics, style="Card.TFrame", padding=(12, 8))
            card.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 6, 0))
            ttk.Label(card, text=name, style="MetricName.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=value, style="Metric.TLabel").pack(anchor="w", pady=(2, 0))

        top = ttk.Frame(preview)
        top.pack(fill="x", pady=(2, 4))
        ttk.Label(top, text="过程诊断", font=(self.ui_font, 13, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.frame_label_var, style="Sub.TLabel").pack(side="right")

        grid = ttk.Frame(preview)
        grid.pack(fill="both", expand=True, pady=4)
        self.image_labels: list[ttk.Label] = []
        self.panel_titles = ["候选框 / 检测框与轨迹", "相机补偿后的前一帧", "补偿后帧差", "二值运动区域"]
        for i, title in enumerate(self.panel_titles):
            frame = ttk.LabelFrame(grid, text=title, padding=5)
            frame.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            label = ttk.Label(frame, anchor="center", text="等待分析", background="#162435", foreground="#D5DFEA")
            label.pack(fill="both", expand=True)
            self.image_labels.append(label)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.slider = ttk.Scale(preview, from_=0, to=0, orient="horizontal",
                                command=self._slider_changed)
        self.slider.pack(fill="x", padx=6)
        ttk.Label(preview, text="拖动底部滑块查看分析过程保存的预览帧").pack(anchor="w", padx=6)

    def _path_row(self, parent: ttk.Frame, title: str, variable: tk.StringVar,
                  kind: str) -> None:
        ttk.Label(parent, text=title).pack(anchor="w", pady=(4, 1))
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="浏览", width=6,
                   command=lambda: self._browse(variable, kind)).pack(side="left", padx=(4, 0))

    def _browse(self, variable: tk.StringVar, kind: str) -> None:
        if kind == "video":
            path = filedialog.askopenfilename(
                title="选择输入视频",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                title="选择模型权重", filetypes=[("PyTorch 权重", "*.pt"), ("所有文件", "*.*")]
            )
        if path:
            variable.set(path)

    def _spin(self, parent: ttk.Frame, label: str, key: str,
              start: float, end: float, step: float) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Spinbox(row, from_=start, to=end, increment=step,
                    textvariable=self.params[key], width=8).pack(side="right")

    def _settings(self) -> dict:
        values = {key: variable.get() for key, variable in self.params.items()}
        for key in ("threshold_window", "morph_kernel"):
            values[key] = int(values[key]) | 1
        for key in ("threshold_window", "morph_kernel", "open_iterations", "close_iterations",
                    "min_area", "max_proposals", "roi_size", "global_interval", "imgsz",
                    "track_buffer", "show_every"):
            values[key] = int(values[key])
        for key in ("threshold_sigma", "threshold_offset", "confidence"):
            values[key] = float(values[key])
        return values

    def _preset(self, name: str) -> None:
        presets = {
            "original": (2.0, 8, 21, 3, 1, 2, 12, 20),
            "sensitive": (1.4, 3, 21, 1, 0, 1, 4, 30),
            "conservative": (2.5, 12, 31, 5, 1, 1, 20, 20),
        }
        keys = ("threshold_sigma", "threshold_offset", "threshold_window", "morph_kernel",
                "open_iterations", "close_iterations", "min_area", "max_proposals")
        for key, value in zip(keys, presets[name]):
            self.params[key].set(value)
        labels = {"original": "原始参数已载入", "sensitive": "偏灵敏预设已载入：更容易保留小候选，也可能增加噪声",
                  "conservative": "偏稳健预设已载入：候选更少，微小区域也可能被过滤"}
        self.status_var.set(labels[name])

    def _save_config(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="保存调参配置", initialdir=str(RUNS), initialfile="motion_tuning.json",
            defaultextension=".json", filetypes=[("JSON 配置", "*.json")],
        )
        if not path:
            return
        payload = {"version": 1, "parameters": self._settings(),
                   "video": self.video_var.get(), "weights": self.weights_var.get(),
                   "device": self.device_var.get(),
                   "start_seconds": self.start_var.get(), "end_seconds": self.end_var.get()}
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status_var.set(f"参数已保存：{path}")

    def _load_config(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="载入调参配置", initialdir=str(RUNS), filetypes=[("JSON 配置", "*.json")]
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            values = payload.get("parameters", payload)
            for key, variable in self.params.items():
                if key in values:
                    variable.set(values[key])
            if "video" in payload:
                self.video_var.set(payload["video"])
            if "weights" in payload:
                self.weights_var.set(payload["weights"])
            if "device" in payload:
                self.device_var.set(payload["device"])
            if "start_seconds" in payload:
                self.start_var.set(payload["start_seconds"])
            if "end_seconds" in payload:
                self.end_var.set(payload["end_seconds"])
            self.status_var.set(f"参数已载入：{path}")
        except (OSError, json.JSONDecodeError, tk.TclError) as exc:
            messagebox.showerror("载入失败", f"配置文件无法读取：\n{exc}")

    def start(self) -> None:
        if self.running:
            return
        video = Path(self.video_var.get()).expanduser()
        if not video.is_file():
            messagebox.showerror("视频不存在", f"找不到视频文件：\n{video}")
            return
        try:
            start_sec, end_sec = float(self.start_var.get()), float(self.end_var.get())
            settings = self._settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("参数无效", f"请检查时间范围和参数值。\n{exc}")
            return
        if start_sec < 0 or end_sec <= start_sec:
            messagebox.showerror("时间范围无效", "结束秒数需要大于起始秒数。")
            return
        if self.mode_var.get() == "full" and not Path(self.weights_var.get()).is_file():
            messagebox.showerror("模型不存在", f"找不到权重文件：\n{self.weights_var.get()}")
            return
        self.snapshots.clear()
        self.cancel_event.clear()
        self.active_mode = self.mode_var.get()
        self.active_weights = self.weights_var.get()
        self.active_device = "0" if self.device_var.get().startswith("GPU") else "cpu"
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.configure(value=0, maximum=1)
        for metric in self.metric_vars:
            metric.set("—")
        self.status_var.set("正在初始化……")
        self.header_status.configure(text="● 运行中", foreground="#C47B18")
        self.worker = threading.Thread(
            target=self._process, args=(video, start_sec, end_sec, settings,
                                       self.active_mode, self.active_weights,
                                       self.active_device), daemon=True
        )
        self.worker.start()

    def stop(self) -> None:
        self.cancel_event.set()
        self.status_var.set("正在停止并保存当前结果……")

    @staticmethod
    def _small(image: np.ndarray | None, size: tuple[int, int] = (680, 370)) -> np.ndarray:
        if image is None:
            return np.zeros((size[1], size[0], 3), np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = min(size[0] / w, size[1] / h)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        y, x = (size[1] - resized.shape[0]) // 2, (size[0] - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    def _process(self, video: Path, start_sec: float, end_sec: float,
                 settings: dict, mode: str, weights: str, device: str) -> None:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            self.events.put(("error", f"无法打开视频：{video}"))
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first, last = int(start_sec * fps), min(int(end_sec * fps), total_frames)
        if last <= first:
            cap.release()
            self.events.put(("error", "所选时间超出视频长度。"))
            return
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        run_dir = RUNS / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            out_video = run_dir / ("motion_preview.mp4" if mode == "motion" else "full_system.mp4")
            manifest = {
                "mode": mode,
                "video": str(video),
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "weights": weights if mode == "full" else None,
                "device": device if mode == "full" else None,
                "parameters": settings,
            }
            (run_dir / "parameters.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            cap.release()
            self.events.put(("error", f"无法创建运行结果目录：{exc}"))
            return
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            cap.release()
            self.events.put(("error", f"无法创建预览视频：{out_video}"))
            return

        motion_config = {key: settings[key] for key in (
            "threshold_sigma", "threshold_offset", "threshold_window", "morph_kernel",
            "open_iterations", "close_iterations", "min_area", "max_proposals")
        }
        proposer: MotionProposer | None = None
        pipeline: MSMVideoPipeline | None = None
        tracker: ByteTrackAdapter | None = None
        try:
            if mode == "motion":
                proposer = MotionProposer(**motion_config)
            else:
                self.events.put(("status", "加载现有 YOLO 模型……"))
                detector = YOLOv5Detector(weights, device)
                if detector.device.type == "cuda":
                    self.events.put(("status", f"模型已加载到 GPU：{detector.torch.cuda.get_device_name(detector.device)}"))
                else:
                    self.events.put(("status", "模型已加载到 CPU（界面中选择了 CPU）"))
                pipeline = MSMVideoPipeline(
                    detector, settings["roi_size"], settings["confidence"],
                    settings["confidence"], settings["global_interval"],
                    settings["imgsz"], motion_config=motion_config,
                )
                tracker = ByteTrackAdapter(settings["confidence"], fps,
                                           settings["track_buffer"])
            max_frames = last - first
            self.events.put(("maximum", max_frames))
            processed = 0
            detections_count = 0
            candidate_frames = 0
            candidate_total = 0
            t0 = time.perf_counter()
            stride = max(1, settings["show_every"])
            while processed < max_frames and not self.cancel_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number = first + processed
                if mode == "motion":
                    scores = proposer.propose(frame)
                    proposals = [item[0] for item in scores]
                    if proposals:
                        candidate_frames += 1
                        candidate_total += len(proposals)
                    annotated = frame.copy()
                    for x1, y1, x2, y2 in proposals:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 170, 255), 2)
                    draw_text(annotated, f"MOTION | frame {frame_number} | proposals {len(proposals)}")
                    aligned, diff, mask = proposer.last_aligned, proposer.last_diff, proposer.last_mask
                    summary = f"候选 {len(proposals)} 个 · 累计 {candidate_total} 个"
                else:
                    annotated, detections = pipeline.process(frame)
                    tracks = tracker.update(detections, frame.shape)
                    detections_count += len(detections)
                    for track in tracks:
                        cv2.rectangle(annotated, (track.x1, track.y1), (track.x2, track.y2),
                                      (255, 70, 220), 2)
                        cv2.putText(annotated, f"ID {track.track_id}",
                                    (track.x1, max(18, track.y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 70, 220), 2)
                    aligned = pipeline.motion.last_aligned
                    diff = pipeline.motion.last_diff
                    mask = pipeline.motion.last_mask
                    draw_text(annotated, f"{pipeline.mode} | frame {frame_number} | det {len(detections)}")
                    summary = f"状态 {pipeline.mode} · 本帧检测 {len(detections)} · 轨迹 {len(tracks)}"
                writer.write(annotated)
                if processed % stride == 0 or processed == max_frames - 1:
                    panels = [self._small(annotated), self._small(aligned),
                              self._small(diff), self._small(mask)]
                    self.events.put(("snapshot", (frame_number, panels, summary)))
                processed += 1
                if processed % 5 == 0 or processed == max_frames:
                    elapsed = max(0.001, time.perf_counter() - t0)
                    if mode == "motion":
                        current_candidates, current_detections = candidate_total, 0
                    else:
                        current_candidates = pipeline.motion.motion_candidate_total
                        current_detections = detections_count
                    self.events.put(("progress", (processed, max_frames,
                        f"处理 {processed}/{max_frames} 帧 · {processed / elapsed:.1f} FPS",
                        current_candidates, current_detections, processed / elapsed)))
            elapsed = time.perf_counter() - t0
            if mode == "motion":
                details = (f"完成 {processed} 帧，运动候选出现于 {candidate_frames} 帧，"
                           f"共 {candidate_total} 个候选框；耗时 {elapsed:.1f}s。")
                result_summary = {"processed_frames": processed,
                                  "candidate_frames": candidate_frames,
                                  "candidate_boxes": candidate_total,
                                  "elapsed_seconds": round(elapsed, 3)}
            else:
                details = (f"完成 {processed} 帧，检测框累计 {detections_count} 个；"
                           f"平均 {processed / max(0.001, elapsed):.1f} FPS。")
                result_summary = {"processed_frames": processed,
                                  "candidate_frames": pipeline.motion.motion_candidate_frames,
                                  "candidate_boxes": pipeline.motion.motion_candidate_total,
                                  "detection_boxes": detections_count,
                                  "detection_frames": pipeline.detection_frames,
                                  "state_frames": dict(pipeline.state_frame_counts),
                                  "elapsed_seconds": round(elapsed, 3),
                                  "average_fps": round(processed / max(0.001, elapsed), 3)}
            manifest["summary"] = result_summary
            cancelled = self.cancel_event.is_set()
            manifest["status"] = "stopped" if cancelled else "completed"
            manifest["output_video"] = str(out_video)
            (run_dir / "parameters.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cap.release()
            writer.release()
            if cancelled:
                details = "已按要求停止。" + details
            self.events.put(("done", (details, str(out_video))))
        except Exception as exc:  # surface worker errors in the UI
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            cap.release()
            writer.release()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "maximum":
                    self.progress.configure(maximum=payload, value=0)
                elif kind == "progress":
                    current, total, message, candidates, detections, speed = payload
                    self.progress.configure(maximum=total, value=current)
                    self.status_var.set(message)
                    self.metric_vars[0].set(str(current))
                    self.metric_vars[1].set(str(candidates))
                    self.metric_vars[2].set(str(detections))
                    self.metric_vars[3].set(f"{speed:.1f} FPS")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "snapshot":
                    self.snapshots.append(payload)
                    self.slider.configure(to=max(0, len(self.snapshots) - 1))
                    self._show_snapshot(len(self.snapshots) - 1)
                elif kind == "done":
                    details, path = payload
                    self.status_var.set(f"{details}\n结果视频：{path}")
                    self._set_running(False)
                    self.header_status.configure(text="● 已完成", foreground="#27865B")
                    messagebox.showinfo("分析完成", f"{details}\n\n结果视频已保存：\n{path}")
                elif kind == "error":
                    self.status_var.set(payload)
                    self._set_running(False)
                    self.header_status.configure(text="● 出错", foreground="#B34B4B")
                    messagebox.showerror("处理失败", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_events)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        if not running and self.header_status.cget("text") == "● 运行中":
            self.header_status.configure(text="● 已停止", foreground="#65768B")

    def _slider_changed(self, value: str) -> None:
        if self.snapshots:
            self._show_snapshot(int(float(value)))

    def _show_snapshot(self, index: int) -> None:
        if not self.snapshots:
            return
        index = max(0, min(index, len(self.snapshots) - 1))
        frame_no, panels, summary = self.snapshots[index]
        self.frame_label_var.set(f"帧 {frame_no} · {summary}")
        self.photo_refs = []
        for label, array in zip(self.image_labels, panels):
            rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            label.configure(image=photo, text="")
            self.photo_refs.append(photo)


def main() -> None:
    root = tk.Tk()
    TunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
