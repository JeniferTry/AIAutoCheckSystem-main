#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能图纸标注系统 - 整合版

功能概述：
1. 支持导入PDF图纸（自动转换为图片）和图片文件（PNG/JPG等）
2. 一键自动标注：调用大模型API识别图纸标注信息
3. 泡泡图展示：带圆圈数字的交互式标注可视化
4. 支持手动导入JSON标注文件
5. 智能标签偏移、重叠检测、分组排列等优化显示功能

依赖：
    pip install pillow PyMuPDF requests

运行：
    python smart_drawing_annotator.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import base64
import threading
import io
import logging
import logging.handlers
import os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple


# =========================================================================
# 【关键修复】在 import tkinter/PIL 之前，立即在主线程预热加载 torch。
# 否则 Tk/GUI/输入法/显卡 hook 会先加载冲突的 VC runtime 或 OpenMP
# DLL（如 jdk-11\bin、TortoiseSVN\bin、PyCharm\bin 里的 vcruntime140.dll），
# 之后 torch 的 c10.dll 会因 DLL 版本不兼容抛 WinError 1114
# "DLL 初始化例程失败"，YOLO 就回退到整图模式。
# 预热成功与否都记录到全局，绝不影响主程序启动。
# =========================================================================
def _warmup_torch_for_yolo() -> Tuple[bool, str]:
    """在所有 GUI 库导入前预热 torch，返回 (是否OK, 说明)"""
    try:
        import site as _site
        # 移除用户site包，避免覆盖当前环境的 numpy/torch
        _user_site = _site.getusersitepackages()
        if _user_site and _user_site in sys.path:
            sys.path.remove(_user_site)

        # 项目 libs 放末尾兜底（防止 libs 里的旧 numpy 插队）
        _libs_dir = str(Path(__file__).parent / "libs")
        if _libs_dir not in sys.path:
            sys.path.append(_libs_dir)

        import ctypes as _ctypes
        _env_dir = Path(sys.executable).parent

        # 把 conda 环境 DLL 目录注入到 Windows DLL 搜索路径最前面
        _torch_lib = _env_dir / "Lib" / "site-packages" / "torch" / "lib"
        _dll_paths = [_torch_lib, _env_dir / "Library" / "bin",
                      _env_dir / "Library" / "lib", _env_dir / "Scripts"]
        for _d in _dll_paths:
            if _d.exists():
                try:
                    os.add_dll_directory(str(_d))
                except (OSError, AttributeError):
                    pass

        # 预加载 torch 自带的 VC runtime + Intel OpenMP（正确版本）
        if _torch_lib.exists():
            for _dll in ("vcruntime140.dll", "vcruntime140_1.dll",
                         "msvcp140.dll", "libiomp5md.dll"):
                _p = _torch_lib / _dll
                if _p.exists():
                    try:
                        _ctypes.CDLL(str(_p))
                    except OSError:
                        pass

        import torch as _torch
        from ultralytics import YOLO as _YOLO
        _msg = (f"预热成功: torch {_torch.__version__}, "
                f"CUDA={_torch.cuda.is_available()}, ultralytics 已就绪")
        print(f"[YOLO预热] {_msg}")
        return True, _msg
    except Exception as _e:
        _msg = f"预热失败: {type(_e).__name__}: {_e}"
        print(f"[YOLO预热] {_msg}   (YOLO+AI标注将回退整图模式)")
        return False, _msg


YOLO_WARMUP_OK, YOLO_WARMUP_MSG = _warmup_torch_for_yolo()


try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
except Exception as exc:
    print(f"无法导入 tkinter：{exc}")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageTk = None
    ImageDraw = None
    ImageFont = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import requests
except Exception:
    requests = None


# ==================== API 配置 ====================
# API_URL = "http://1487379763047589.cn-shenzhen.pai-eas.aliyuncs.com/api/predict/qwen3_5_9b_4bitfinetune2/v1/chat/completions"
# API_HEADERS = {
#    "Authorization": "OTI4ZjMxZDBmZDAxOWQzNzNkZmVhNTEyN2NmODMzODY4YjE1OWFiZg==",
#    "Content-Type": "application/json"
# }
# 本地部署的模型
API_URL = "http://192.168.10.248:8006/v1/chat/completions"
API_HEADERS = {
   "Authorization": "",
   "Content-Type": "application/json"
}

# API_URL ="http://1487379763047589.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/qwen3_5_9b_4bitfinetune2/v1/chat/completions"
# API_HEADERS ={"Authorization": "NjNhYjY1YWI2NTFiMmVhNDcyN2YxNTU1YTJmMjlkYWZhMTg3YmI5Yg==","Content-Type": "application/json"}
# API_URL ="http://1867994711041679.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/levelfield_1/v1/chat/completions"
# API_HEADERS ={"Authorization": "OTI4ZjMxZDBmZDAxOWQzNzNkZmVhNTEyN2NmODMzODY4YjE1OWFiZg==","Content-Type": "application/json"}
# 精简版prompt - AI返回紧凑格式，本地转换为x-anylabeling标准格式
# 注意：JSON示例中的花括号需转义为{{和}}，仅保留{image_width}和{image_height}作为真正占位符
# API_PROMPT_TEMPLATE = (
#     '你是机械工程图纸尺寸标注信息识别模型。请识别这张机械图纸中的所有尺寸标注、'
#     '技术要求和关键信息，并要给出信息的像素坐标位置，严格按照下面示例的格式输出，'
#     '必须包含每个目标的坐标框 x1,y1,x2,y2 和标签。\n'
#     '【重要】输出格式为“图片预估尺寸+以 #JSON# 开头和 #JSON# 结尾的一段紧凑JSON数组”，除了格式示例中的内容外，不要包含任何解释或思考过程。\n'
#     '格式示例（严格遵守）：\n'
#     '图片预估尺寸:1024*1680.#JSON#[{{"label":"长度尺寸","description":"标注=22 ±0.05","score":90,'
#     '"x1":838.9073,"y1":733.3333,"x2":909.2777,"y2":992.5925}}]#JSON#'
# )

# API_PROMPT_TEMPLATE = (
#     '你是机械工程图纸尺寸标注信息识别模型。请识别这张机械图纸中的所有尺寸标注、'
#     '技术要求和关键信息，并要给出信息的像素坐标位置，严格按照下面示例的格式输出，'
#     '必须包含每个目标的坐标框 x1,y1,x2,y2 和标签。\n'
#     '【重要】你的回答必须且只能包含以 #JSON# 开头和 #JSON# 结尾的一段JSON数据，'
#     '不要在#JSON#标记之外输出任何文字、解释或思考过程。\n'
#     '输出格式为紧凑JSON数组，输入图片的尺寸为{image_width}*{image_height}。\n'
#     '格式示例（严格遵守）：\n'
#     '#JSON#[{{"label":"长度尺寸","description":"标注=22 ±0.05","score":90,'
#     '"x1":838.9073,"y1":733.3333,"x2":909.2777,"y2":992.5925}}]#JSON#'
# )
API_PROMPT_TEMPLATE =(
'输入图片的尺寸为{image_width}*{image_height}，请识别这张机械图纸的所有尺寸标注、技术要求和关键信息，并要给出每个特征的像素坐标位置。请严格按照下面示例的格式输出，'
'不可遗漏。【重要】你的回答必须且只能包含以 #JSON# 开头和 #JSON# 结尾的一段JSON数据格式，示例（严格遵守）：'
'#JSON#[{{"label":"长度尺寸","description":"标注=22 ±0.05","score":90,"x1":838.9073,"y1":733.3333,"x2":909.2777,"y2":992.5925}}]#JSON# 。'
)
# API_PROMPT_TEMPLATE2 =(
# '请识别这张机械图纸剪裁拼接后的图片中的所有尺寸标注、技术要求和关键信息，并要给出每个特征的像素坐标位置。请严格按照下面示例的格式输出，'
# '不可遗漏每个目标的坐标框 x1,y1,x2,y2 和标签。【重要】你的回答必须且只能包含以 #JSON# 开头和 #JSON# 结尾的一段JSON数据格式，示例（严格遵守）：'
# '#JSON#[{{"label":"长度尺寸","description":"标注=22 ±0.05","score":90,"x1":838.9073,"y1":733.3333,"x2":909.2777,"y2":992.5925}}]#JSON# 。'
# 'label类别：技术要求、标题栏、长度尺寸、直径尺寸、半径尺寸、角度尺寸、倒角、线轮廓度、面轮廓度、平行度、垂直度、倾斜度、同轴度、对称度、位置度、圆跳动、全跳动、直线度、平面度、圆柱度、圆度、粗糙度'
# )

# API_PROMPT_TEMPLATE = (
#     '你是资深工艺工程师。请仔细分析这张机械图纸,读取并整理零件名称、图号、材料、尺寸与公差（包含角度尺寸和形位公差）、表面粗糙度及其他技术要求，并告知坐标位置，以便帮助我绘制泡泡图和编制工艺文件。告诉我你的结果。'
# )

# JSON数据标记 - 用于从AI回复中提取有效数据
JSON_TAG = "#JSON#"

# 最大token数（精简prompt不需要太多token）
MAX_TOKENS = 8192 #5632

# 标准标签列表（从 class.txt 加载），用于将AI/导入的 label 对齐到标准类别
CLASS_FILE_PATH = Path(__file__).parent / "class.txt"


def _load_class_labels() -> List[str]:
    """加载 class.txt 中的标准标签列表（格式: "0: 技术要求" -> "技术要求"）"""
    labels: List[str] = []
    try:
        if CLASS_FILE_PATH.exists():
            for line in CLASS_FILE_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                # 支持 "0: 技术要求" 或 "技术要求" 两种格式
                if ":" in line:
                    parts = line.split(":", 1)
                    label = parts[1].strip()
                else:
                    label = line
                if label:
                    labels.append(label)
    except Exception:
        pass
    return labels


STANDARD_LABELS: List[str] = _load_class_labels()


class DateFileHandler(logging.Handler):
    """按日期命名文件的日志处理器，文件名格式: {prefix}_YYYY-MM-DD.log"""

    def __init__(self, log_dir: Path, prefix: str, encoding: str = "utf-8"):
        super().__init__()
        self.log_dir = log_dir
        self.prefix = prefix
        self.encoding = encoding
        self._current_date: Optional[str] = None
        self._file_handle: Optional[Any] = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{self.prefix}_{today}.log"

    def _get_or_create_file(self) -> Any:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_date != today or self._file_handle is None:
            if self._file_handle is not None:
                self._file_handle.close()
            file_path = self._get_file_path()
            self._file_handle = open(str(file_path), "a", encoding=self.encoding)
            self._current_date = today
        return self._file_handle

    def emit(self, record: logging.LogRecord) -> None:
        try:
            handle = self._get_or_create_file()
            msg = self.format(record)
            handle.write(msg + "\n")
            handle.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        super().close()


class LogManager:
    """日志管理器 - 按日期命名文件，完整记录AI交互内容"""

    LOG_DIR = Path(__file__).parent / "logs"

    def __init__(self) -> None:
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.operation_logger = self._create_logger("operation")
        self.ai_logger = self._create_logger("ai_interaction")
        self.error_logger = self._create_logger("error")

    def _create_logger(self, prefix: str) -> logging.Logger:
        logger = logging.getLogger(f"{prefix}_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        file_handler = DateFileHandler(self.LOG_DIR, prefix)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def log_operation(self, message: str) -> None:
        self.operation_logger.info(message)

    def log_ai(self, message: str) -> None:
        self.ai_logger.info(message)

    def log_ai_request(self, url: str, prompt: str) -> None:
        self.ai_logger.info(f"[REQUEST] URL={url}")
        self.ai_logger.info(f"[PROMPT] {prompt}")

    def log_ai_response(self, response: str) -> None:
        self.ai_logger.info(f"[RESPONSE] {response}")

    def log_ai_success(self, count: int) -> None:
        self.ai_logger.info(f"[SUCCESS] AI识别完成，共返回 {count} 个标注")

    def log_error(self, message: str, exc_info: bool = False) -> None:
        self.error_logger.error(message, exc_info=exc_info)


@dataclass
class Marker:
    """标注数据类，存储单个标注的所有信息"""
    number: int
    x: float
    y: float
    label: str = ""
    description: str = ""
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0
    shape_type: str = "rectangle"
    display_x: float = 0.0
    display_y: float = 0.0
    font_size_override: Optional[int] = None


class SmartDrawingAnnotator(tk.Tk):
    """智能图纸标注系统 - 整合版主应用类"""

    # 参数配置
    RADIUS_FONT_RATIO = 0.20
    RADIUS_DIGIT_RATIO = 0.20
    RADIUS_BASE = 5
    OFFSET_DISTANCE_RATIO = 1.5
    THIN_ASPECT_RATIO = 1
    WIDE_ASPECT_RATIO = 1
    OVERLAP_MIN_DISTANCE = 4
    OVERLAP_MAX_ITERATIONS = 15
    OVERLAP_FONT_MIN = 6
    GROUP_THRESHOLD_RATIO = 6.0
    GROUP_STAGGER_RATIO = 1.5
    EDGE_PADDING_RATIO = 0.3
    EDGE_PENALTY_FACTOR = 0.6

    COLOR_NORMAL = "#2563eb"
    COLOR_SELECTED = "#dc2626"
    COLOR_LINE = "#6b7280"
    CIRCLE_FILL = ""
    CIRCLE_LINE_WIDTH_NORMAL = 2
    CIRCLE_LINE_WIDTH_SELECTED = 3

    def __init__(self) -> None:
        """初始化应用窗口和所有组件"""
        super().__init__()
        self.title("智能图纸标注系统 - 整合版")
        self.geometry("1400x850")
        self.minsize(1000, 600)

        # 文件路径
        self.image_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.annotation_path: Optional[Path] = None

        # 图像
        self.original_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.image_canvas_id: Optional[int] = None
        # PhotoImage 复用：尺寸一致时用 paste，避免 Tk 端图片句柄累积
        self._cached_tk_size: tuple[int, int] = (0, 0)

        # 缩放
        self.zoom = 1.0
        self.zoom_factor = 1.0
        self.fit_zoom = 1.0
        self._resize_after_id: Optional[str] = None

        # 标注数据
        self.markers: dict[int, Marker] = {}
        self.marker_items: dict[int, tuple] = {}

        # 交互状态
        self.selected_number: Optional[int] = None
        self.dragging_number: Optional[int] = None
        self.drag_offset: tuple[float, float] = (0, 0)
        self.checked_markers: set[int] = set()
        # 拖动节流：合并连续 B1-Motion，避免每帧重绘
        self._drag_after_id: Optional[str] = None

        # 加载状态
        self.is_loading = False

        # UI变量
        self.font_size_var = tk.IntVar(value=11)
        self.marker_font = tkfont.Font(family="Arial", size=self.font_size_var.get(), weight="bold")
        self.label_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择图纸文件（PDF或图片）")

        # 进度窗口
        self.progress_window: Optional[tk.Toplevel] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.progress_label: Optional[ttk.Label] = None

        # 日志管理器
        self.log = LogManager()
        self.log.log_operation("应用启动")

        # YOLO预热状态
        if YOLO_WARMUP_OK:
            self.log.log_ai(f"YOLO预热成功: {YOLO_WARMUP_MSG}")
            self.status_var.set("YOLO 已就绪，请选择图纸文件（PDF或图片）")
        else:
            self.log.log_error(f"YOLO预热失败: {YOLO_WARMUP_MSG}  YOLO+AI标注将回退到整图模式")

        self._build_ui()

    def _build_ui(self) -> None:
        """构建完整的UI界面（使用PanedWindow实现可拖拽分割）"""
        # 主分割容器 - 左右可拖拽调整
        style = ttk.Style()
        style.configure("TPanedwindow", sashwidth=6)

        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # 左侧：图片显示区
        self.left = ttk.Frame(self.paned)
        self.paned.add(self.left, weight=3)

        # 右侧：控制面板
        self.right = ttk.Frame(self.paned, padding=10)
        self.paned.add(self.right, weight=1)

        # 设置初始比例（左侧75%，右侧25%）
        self.after(50, lambda: self.paned.sashpos(0, int(self.winfo_width() * 0.75)))

        # 工具栏
        toolbar = ttk.Frame(self.left, padding=(8, 8, 8, 4))
        toolbar.pack(fill=tk.X)

        # 文件操作
        ttk.Label(toolbar, text="📁 文件:").pack(side=tk.LEFT, padx=(0, 5))
        self.btn_import_pdf = ttk.Button(toolbar, text="导入PDF", command=self.import_pdf)
        self.btn_import_pdf.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_import_image = ttk.Button(toolbar, text="导入图片", command=self.open_image)
        self.btn_import_image.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # 自动标注
        self.btn_auto_annotate = ttk.Button(toolbar, text="🔄 一键自动标注", command=self.auto_annotate)
        self.btn_auto_annotate.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_yolo_vlm = ttk.Button(toolbar, text="🎯 YOLO+AI标注", command=self.yolo_vlm_annotate)
        self.btn_yolo_vlm.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_import_annotation = ttk.Button(toolbar, text="导入标注", command=self.import_annotation)
        self.btn_import_annotation.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # 图纸归一化
        normalize_frame = ttk.Frame(toolbar)
        normalize_frame.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(normalize_frame, text="图纸归一化", command=self.on_normalize_click).pack(side=tk.LEFT)
        self.auto_normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(normalize_frame, text="自动", variable=self.auto_normalize_var).pack(side=tk.LEFT, padx=(6, 0))

        # 结果比对
        ttk.Button(toolbar, text="结果比对", command=self.open_comparison_dialog).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # 视图控制
        ttk.Button(toolbar, text="智能偏移", command=self.calculate_all_offsets).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="放大", command=lambda: self.set_zoom_factor(self.zoom_factor * 1.25)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="缩小", command=lambda: self.set_zoom_factor(self.zoom_factor / 1.25)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="适应窗口", command=lambda: self.set_zoom_factor(1.0)).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(toolbar, textvariable=self.status_var, foreground="#666").pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        # 画布
        canvas_wrap = ttk.Frame(self.left)
        canvas_wrap.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_wrap, bg="#f3f4f6", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(canvas_wrap, orient=tk.VERTICAL, command=self.canvas.yview)
        x_scroll = ttk.Scrollbar(canvas_wrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # 底部像素信息栏
        info_bar = ttk.Frame(self.left, padding=(8, 2))
        info_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.pixel_info_var = tk.StringVar(value="图片尺寸: - | 文件: -")
        ttk.Label(info_bar, textvariable=self.pixel_info_var, foreground="#555").pack(side=tk.LEFT)

        self._build_right_panel()

    def _build_right_panel(self) -> None:
        """构建右侧控制面板（使用可滚动容器 + 鼠标滚轮支持）"""
        # 使用Canvas实现右侧面板的整体滚动
        self._right_canvas = tk.Canvas(self.right, highlightthickness=0)
        right_vsb = ttk.Scrollbar(self.right, orient=tk.VERTICAL, command=self._right_canvas.yview)
        self._right_canvas.configure(yscrollcommand=right_vsb.set)

        self._right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # 内部Frame
        self.right_inner = ttk.Frame(self._right_canvas)
        self._right_canvas_window = self._right_canvas.create_window((0, 0), window=self.right_inner, anchor=tk.NW)

        def on_inner_configure(event):
            self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all"))
        self.right_inner.bind("<Configure>", on_inner_configure)

        def on_canvas_configure(event):
            self._right_canvas.itemconfig(self._right_canvas_window, width=event.width)
        self._right_canvas.bind("<Configure>", on_canvas_configure)

        # 鼠标滚轮绑定
        def on_mousewheel(event):
            self._right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._right_canvas.bind("<Enter>", lambda e: self._right_canvas.bind_all("<MouseWheel>", on_mousewheel))
        self._right_canvas.bind("<Leave>", lambda e: self._right_canvas.unbind_all("<MouseWheel>"))

        # 新增/更新标注表单
        ttk.Label(self.right_inner, text="新增或更新标注", font=("", 11, "bold")).pack(anchor=tk.W)

        ttk.Label(self.right_inner, text="快速输入，如：(120,80)").pack(anchor=tk.W, pady=(10, 2))
        self.quick_var = tk.StringVar()
        quick_entry = ttk.Entry(self.right_inner, textvariable=self.quick_var)
        quick_entry.pack(fill=tk.X)
        quick_entry.bind("<Return>", lambda _: self.add_or_update_from_input())

        fields = ttk.Frame(self.right_inner)
        fields.pack(fill=tk.X, pady=(8, 0))
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()

        ttk.Label(fields, text="X").grid(row=0, column=0, sticky="w")
        ttk.Entry(fields, textvariable=self.x_var, width=10).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Label(fields, text="Y").grid(row=0, column=1, sticky="w")
        ttk.Entry(fields, textvariable=self.y_var, width=10).grid(row=1, column=1, sticky="ew", padx=(0, 6))
        ttk.Label(fields, text="Label").grid(row=0, column=2, sticky="w")
        ttk.Entry(fields, textvariable=self.label_var, width=12).grid(row=1, column=2, sticky="ew")
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=1)
        fields.columnconfigure(2, weight=2)

        desc_frame = ttk.Frame(self.right_inner)
        desc_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(desc_frame, text="Description").pack(side=tk.LEFT)
        ttk.Entry(desc_frame, textvariable=self.description_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        btns = ttk.Frame(self.right_inner)
        btns.pack(fill=tk.X, pady=10)
        ttk.Button(btns, text="添加/更新", command=self.add_or_update_from_input).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="删除选中", command=self.delete_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        # Label对齐按钮：将所有Label对齐到 class.txt 标准类别
        align_btn = ttk.Button(self.right_inner, text="对齐Label到class.txt", command=self.align_all_labels)
        align_btn.pack(fill=tk.X, pady=(0, 6))

        font_box = ttk.Frame(self.right_inner)
        font_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(font_box, text="圆圈数字字体大小").pack(side=tk.LEFT)
        font_spin = ttk.Spinbox(font_box, from_=6, to=72, width=6, textvariable=self.font_size_var, command=self.on_font_size_change)
        font_spin.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(self.right_inner).pack(fill=tk.X, pady=8)
        ttk.Label(self.right_inner, text="标注列表", font=("", 11, "bold")).pack(anchor=tk.W)

        table_frame = ttk.Frame(self.right_inner)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.tree = ttk.Treeview(table_frame, columns=("check", "number", "label", "description", "x", "y", "up", "down", "delete"), show="headings", selectmode="browse", height=25)
        self.tree.heading("check", text="✓")
        self.tree.heading("number", text="序号")
        self.tree.heading("label", text="Label")
        self.tree.heading("description", text="Description")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("up", text="↑")
        self.tree.heading("down", text="↓")
        self.tree.heading("delete", text="×")

        self.tree.column("check", width=40, anchor=tk.CENTER, stretch=False)
        self.tree.column("number", width=54, anchor=tk.CENTER, stretch=False)
        self.tree.column("label", width=90, anchor=tk.W, stretch=False)
        self.tree.column("description", width=140, anchor=tk.W, stretch=False)
        self.tree.column("x", width=60, anchor=tk.CENTER, stretch=False)
        self.tree.column("y", width=60, anchor=tk.CENTER, stretch=False)
        self.tree.column("up", width=30, anchor=tk.CENTER, stretch=False)
        self.tree.column("down", width=30, anchor=tk.CENTER, stretch=False)
        self.tree.column("delete", width=30, anchor=tk.CENTER, stretch=False)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table_vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        table_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        table_hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        table_hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(yscrollcommand=table_vsb.set, xscrollcommand=table_hsb.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)
        self.tree.bind("<Double-1>", self.load_selected_to_form)

        # Treeview的鼠标滚轮支持
        def tree_on_mousewheel(event):
            self.tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.tree.bind("<Enter>", lambda e: self.tree.bind_all("<MouseWheel>", tree_on_mousewheel))
        self.tree.bind("<Leave>", lambda e: self.tree.unbind_all("<MouseWheel>"))

        hint = "提示：\n1. 点击'导入PDF'或'导入图片'加载图纸\n2. 点击'一键自动标注'调用AI识别\n3. 坐标使用原图像素坐标\n4. 点击图片标注可选中对应表格行\n5. 拖动图片标注会自动更新坐标\n6. 双击表格Label/Description列可编辑"
        ttk.Label(self.right_inner, text=hint, justify=tk.LEFT, foreground="#4b5563").pack(anchor=tk.W, pady=(10, 0))

    # ==================== PDF 导入 ====================

    def import_pdf(self) -> None:
        """导入PDF文件并转换为图片显示"""
        if fitz is None:
            messagebox.showerror("缺少依赖", "请先安装 PyMuPDF：pip install PyMuPDF")
            self.log.log_error("导入PDF失败: PyMuPDF未安装")
            return
        if Image is None or ImageTk is None:
            messagebox.showerror("缺少依赖", "请先安装 Pillow：pip install pillow")
            self.log.log_error("导入PDF失败: Pillow未安装")
            return

        path = filedialog.askopenfilename(
            title="选择PDF文件",
            filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")],
        )
        if not path:
            return

        self.log.log_operation(f"开始导入PDF: {path}")

        try:
            doc = fitz.open(path)
            total_pages = len(doc)

            # 多页PDF让用户选择页码
            page_num = 0
            if total_pages > 1:
                page_num = self._ask_pdf_page(total_pages)
                if page_num is None:
                    doc.close()
                    return

            page = doc[page_num]
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()

            pdf_path = Path(path)
            if total_pages == 1:
                img_name = f"{pdf_path.stem}.png"
            else:
                img_name = f"{pdf_path.stem}_page_{page_num + 1}.png"

            temp_img_path = pdf_path.parent / img_name
            self._set_image(img, temp_img_path)

            self.status_var.set(f"PDF已加载: {pdf_path.name} (第{page_num + 1}/{total_pages}页) | {img.width}x{img.height}")
            self.log.log_operation(f"PDF导入成功: {pdf_path.name}, 第{page_num + 1}/{total_pages}页, 尺寸{img.width}x{img.height}")
            messagebox.showinfo("成功", f"PDF转换完成！\n页数：{total_pages}\n当前页：第{page_num + 1}页\n图片尺寸：{img.width}x{img.height}")

        except Exception as exc:
            self.log.log_error(f"PDF导入失败: {path}, 错误: {exc}", exc_info=True)
            messagebox.showerror("PDF导入失败", f"无法处理PDF文件：{exc}")

    def _ask_pdf_page(self, total_pages: int) -> Optional[int]:
        """弹窗让用户选择PDF页码"""
        dialog = tk.Toplevel(self)
        dialog.title("选择PDF页码")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        result = [None]
        ttk.Label(dialog, text=f"该PDF共有 {total_pages} 页，请选择要加载的页码：").pack(pady=(15, 10), padx=20)

        page_var = tk.IntVar(value=1)
        ttk.Spinbox(dialog, from_=1, to=total_pages, textvariable=page_var, width=10).pack(pady=5)

        def on_confirm():
            result[0] = page_var.get() - 1
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="确定", command=on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        return result[0]

    # ==================== 图片导入 ====================

    def open_image(self) -> None:
        """打开图片文件"""
        if Image is None or ImageTk is None:
            messagebox.showerror("缺少依赖", "请先安装 Pillow：pip install pillow")
            self.log.log_error("打开图片失败: Pillow未安装")
            return

        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not path:
            return

        self.log.log_operation(f"开始打开图片: {path}")

        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:
            self.log.log_error(f"打开图片失败: {path}, 错误: {exc}", exc_info=True)
            messagebox.showerror("打开失败", f"无法打开图片：{exc}")
            return

        self._set_image(img, Path(path))

    def _set_image(self, img: Image.Image, path: Path) -> None:
        """设置图片并初始化显示"""
        self.image_path = path
        self.config_path = self.image_path.with_name(self.image_path.name + ".json")
        # 释放上一张图的 PhotoImage 与 canvas 图元，避免 Tk 端图片句柄累积
        if self.image_canvas_id is not None:
            self.canvas.delete(self.image_canvas_id)
            self.image_canvas_id = None
        self.tk_image = None
        self._cached_tk_size = (0, 0)
        self.original_image = img
        self.zoom = 1.0
        self.zoom_factor = 1.0
        self.fit_zoom = 1.0
        self.markers.clear()
        self.checked_markers.clear()
        self.selected_number = None
        self.load_config()
        self.update_fit_zoom()
        self.refresh_table()
        self.log.log_operation(f"图片加载完成: {path.name}, 尺寸{img.width}x{img.height}")
        self.status_var.set(f"{self.image_path.name} | {img.width}x{img.height} | 配置：{self.config_path.name}")
        self.pixel_info_var.set(f"图片尺寸: {img.width} × {img.height} px | 文件: {path.name}")

    # ==================== 图纸归一化 ====================

    NORMALIZE_LONG_SIDE = 2000  # 自动模式：长边目标像素

    def on_normalize_click(self) -> None:
        """归一化按钮点击：自动模式直接执行，手动模式弹对话框"""
        if self.original_image is None:
            messagebox.showinfo("提示", "请先导入图片或PDF文件")
            return
        if self.auto_normalize_var.get():
            self._auto_normalize()
        else:
            self.open_normalize_dialog()

    def _auto_normalize(self) -> None:
        """自动归一化：长边压缩到2000px（只缩小不放大），保存到Normalization目录"""
        img = self.original_image
        orig_w, orig_h = img.width, img.height
        long_side = max(orig_w, orig_h)

        if long_side <= self.NORMALIZE_LONG_SIDE:
            messagebox.showinfo("自动归一化",
                f"当前图片长边为 {long_side}px，已≤{self.NORMALIZE_LONG_SIDE}px，无需归一化。")
            return

        # 计算缩放比例（只缩小，不放大）
        ratio = self.NORMALIZE_LONG_SIDE / long_side
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))

        # 输出目录：当前文件所在目录/Normalization
        if self.image_path:
            output_dir = str(self.image_path.parent / "Normalization")
        else:
            output_dir = str(Path.cwd() / "Normalization")

        self._do_normalize(self.image_path, new_w, new_h, output_dir)

    def open_normalize_dialog(self) -> None:
        """打开图纸归一化对话框（手动模式）"""
        if self.original_image is None:
            messagebox.showinfo("提示", "请先导入图片或PDF文件")
            return

        img = self.original_image
        orig_w, orig_h = img.width, img.height
        orig_path = self.image_path

        dialog = tk.Toplevel(self)
        dialog.title("图纸归一化")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # 居中
        dialog.update_idletasks()
        w, h = 420, 380
        sw = dialog.winfo_screenwidth()
        sh = dialog.winfo_screenheight()
        dialog.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        pad = {"padx": 12, "pady": 6}

        # 原图信息
        info_frame = ttk.LabelFrame(dialog, text="原图信息", padding=8)
        info_frame.pack(fill=tk.X, **pad)
        file_label = orig_path.name if orig_path else "未知"
        ttk.Label(info_frame, text=f"文件: {file_label}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"原始尺寸: {orig_w} × {orig_h} px").pack(anchor=tk.W)

        # 调整模式
        mode_frame = ttk.LabelFrame(dialog, text="调整模式", padding=8)
        mode_frame.pack(fill=tk.X, **pad)

        mode_var = tk.StringVar(value="ratio")
        ttk.Radiobutton(mode_frame, text="按比例缩放 (%)", variable=mode_var, value="ratio").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="按像素尺寸", variable=mode_var, value="pixel").pack(anchor=tk.W, pady=(4, 0))

        # 参数区
        param_frame = ttk.Frame(dialog)
        param_frame.pack(fill=tk.X, **pad)

        # --- 比例模式参数 ---
        ratio_frame = ttk.Frame(param_frame)
        ratio_frame.pack(fill=tk.X)

        ttk.Label(ratio_frame, text="缩放比例:").pack(side=tk.LEFT)
        ratio_var = tk.StringVar(value="100")
        ratio_entry = ttk.Entry(ratio_frame, textvariable=ratio_var, width=8)
        ratio_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(ratio_frame, text="%").pack(side=tk.LEFT)

        # --- 像素模式参数 ---
        pixel_frame = ttk.Frame(param_frame)

        ttk.Label(pixel_frame, text="目标宽度:").grid(row=0, column=0, sticky=tk.W)
        tw_var = tk.StringVar(value=str(orig_w))
        ttk.Entry(pixel_frame, textvariable=tw_var, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(pixel_frame, text="px").grid(row=0, column=2)

        ttk.Label(pixel_frame, text="目标高度:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        th_var = tk.StringVar(value=str(orig_h))
        ttk.Entry(pixel_frame, textvariable=th_var, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))
        ttk.Label(pixel_frame, text="px").grid(row=1, column=2, pady=(6, 0))

        # 锁定宽高比
        lock_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pixel_frame, text="保持宽高比", variable=lock_var).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        def switch_mode():
            if mode_var.get() == "ratio":
                pixel_frame.pack_forget()
                ratio_frame.pack(fill=tk.X)
            else:
                ratio_frame.pack_forget()
                pixel_frame.pack(fill=tk.X)

        mode_var.trace_add("write", lambda *_: switch_mode())
        switch_mode()

        # 输出目录
        output_frame = ttk.LabelFrame(dialog, text="输出设置", padding=8)
        output_frame.pack(fill=tk.X, **pad)

        default_output = str(orig_path.parent / "归一化输出") if orig_path else str(Path.cwd() / "归一化输出")
        output_dir_var = tk.StringVar(value=default_output)

        def browse_dir():
            d = filedialog.askdirectory(title="选择输出目录", initialdir=output_dir_var.get())
            if d:
                output_dir_var.set(d)

        dir_row = ttk.Frame(output_frame)
        dir_row.pack(fill=tk.X)
        ttk.Entry(dir_row, textvariable=output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(dir_row, text="浏览...", command=browse_dir).pack(side=tk.RIGHT)

        # 保存原文件标记
        ttk.Label(output_frame, text="注意：不会修改原导入文件，将保存到上述目录", foreground="#888").pack(anchor=tk.W, pady=(6, 0))

        # 按钮区
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=(12, 8), padx=12)

        def on_confirm():
            try:
                # 计算目标尺寸
                if mode_var.get() == "ratio":
                    ratio = float(ratio_var.get()) / 100.0
                    if ratio <= 0:
                        raise ValueError("缩放比例必须大于0")
                    new_w = max(1, int(orig_w * ratio))
                    new_h = max(1, int(orig_h * ratio))
                else:
                    tw = int(tw_var.get())
                    th = int(th_var.get())
                    if tw <= 0 or th <= 0:
                        raise ValueError("目标尺寸必须大于0")
                    if lock_var.get():
                        ratio_w = tw / orig_w
                        ratio_h = th / orig_h
                        r = min(ratio_w, ratio_h)
                        new_w = max(1, int(orig_w * r))
                        new_h = max(1, int(orig_h * r))
                    else:
                        new_w, new_h = tw, th

                out_dir = output_dir_var.get()
                if not out_dir:
                    raise ValueError("请选择输出目录")

                dialog.destroy()
                self._do_normalize(orig_path, new_w, new_h, out_dir)

            except ValueError as e:
                messagebox.showerror("错误", str(e), parent=dialog)

        ttk.Button(btn_frame, text="确定", command=on_confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT)

    def _do_normalize(self, src_path: Path, target_w: int, target_h: int, output_dir: str) -> None:
        """执行图纸归一化：格式转换 -> 缩放 -> 保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            self.log.log_operation(f"开始图纸归一化: {src_path.name} -> {target_w}x{target_h}")

            # 步骤1: 获取或加载图片
            if self.original_image is not None and self.image_path == src_path:
                img = self.original_image.copy()
            else:
                if src_path.suffix.lower() == ".pdf":
                    if fitz is None:
                        raise RuntimeError("缺少 PyMuPDF 依赖")
                    doc = fitz.open(str(src_path))
                    page = doc[0]
                    mat = fitz.Matrix(300 / 72, 300 / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    doc.close()
                else:
                    img = Image.open(str(src_path))

            # 步骤2: 格式转换为PNG（如果不是PNG）
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            # 步骤3: 缩放
            resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            # 步骤4: 保存到输出目录
            base_name = src_path.stem
            out_file = output_path / f"{base_name}_{target_w}x{target_h}.png"

            # 若同名则加序号
            counter = 1
            while out_file.exists():
                out_file = output_path / f"{base_name}_{target_w}x{target_h}_{counter}.png"
                counter += 1

            resized.save(str(out_file), "PNG")

            self.log.log_operation(f"归一化完成: {out_file.name} ({target_w}x{target_h})")
            messagebox.showinfo("归一化完成", f"处理成功！\n\n输出文件: {out_file}\n尺寸: {target_w} × {target_h} px")

        except Exception as e:
            self.log.log_error(f"归一化失败: {src_path.name} - {str(e)}", exc_info=True)
            messagebox.showerror("错误", f"归一化失败：{str(e)}")

    # ==================== 结果比对 ====================

    @staticmethod
    def _normalize_desc(desc: str) -> str:
        """标准化描述文本：去除多余空格，统一标点符号（中文→英文，标点差异忽略）"""
        import re as _re
        # 中文标点统一映射到英文标点：标点不同（中/英）视为相同描述
        punct_map = str.maketrans({
            "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?",
            "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
            "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
            "、": ",", "·": ".", "—": "-", "～": "~",
            "「": "(", "」": ")", "『": "(", "』": ")",
        })
        desc = desc.translate(punct_map)
        desc = desc.strip()
        desc = _re.sub(r'\s+', ' ', desc)
        return desc

    @staticmethod
    def _is_approx_match(pred_norm: str, gt_norm: str) -> bool:
        """近似匹配：一方描述包含另一方（即一方是另一方的子串），允许预测多出字符"""
        if not pred_norm or not gt_norm:
            return False
        return pred_norm in gt_norm or gt_norm in pred_norm

    # ==================== Label 对齐到 class.txt ====================

    # description 中需要去除的前缀（提示词中加入这些前缀有助于提高AI回复质量，转换时去掉）
    _DESC_PREFIXES: Tuple[str, ...] = (
        "标注=", "值=", "数值=", "数据=", "数值：", "标注：", "值：", "数据：",
    )

    @classmethod
    def _strip_desc_prefix(cls, description: str) -> str:
        """
        去除 description 中的前缀（如"标注="、"值="、"数值="、"数据="等）。

        这些前缀在提示词中用于提高AI回复质量，转换到标准格式时需要去掉。
        仅去除开头的单个前缀，避免误伤内容中包含"="的描述。
        """
        if not description:
            return description
        text = description.strip()
        for prefix in cls._DESC_PREFIXES:
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    @staticmethod
    def _align_label_to_class(label: str) -> str:
        """
        将任意 label 对齐到 class.txt 中的标准标签。

        匹配策略（按优先级）：
        1. 完全相同 -> 直接返回
        2. 标准标签是输入的子串（如 "长度尺寸标注" 含 "长度尺寸"） -> 返回标准
        3. 输入是标准标签的子串 -> 返回标准
        4. 分词后关键字符匹配（包含"长度"→"长度尺寸"，包含"直径"→"直径尺寸"等）
        5. 都不匹配 -> 返回原 label

        若 STANDARD_LABELS 为空（class.txt 缺失），返回原 label
        """
        if not label:
            return label
        if not STANDARD_LABELS:
            return label

        raw = label.strip()

        # 1. 完全相同
        for std in STANDARD_LABELS:
            if raw == std:
                return std

        # 2. 标准标签是输入的子串（取最长的优先，避免"半径尺寸"匹配到"尺寸"）
        best_match: Optional[str] = None
        best_len = 0
        for std in STANDARD_LABELS:
            if std in raw and len(std) > best_len:
                best_match = std
                best_len = len(std)
        if best_match:
            return best_match

        # 3. 输入是标准标签的子串
        for std in STANDARD_LABELS:
            if raw in std:
                return std

        # 4. 关键词映射表（常见AI返回值 -> 标准标签）
        keyword_map = [
            ("长度", "长度尺寸"),
            ("直径", "直径尺寸"),
            ("半径", "半径尺寸"),
            ("角度", "角度尺寸"),
            ("线轮廓", "线轮廓度"),
            ("面轮廓", "面轮廓度"),
            ("平行", "平行度"),
            ("垂直", "垂直度"),
            ("倾斜", "倾斜度"),
            ("同轴", "同轴度"),
            ("对称", "对称度"),
            ("位置", "位置度"),
            ("圆跳", "圆跳动"),
            ("全跳", "全跳动"),
            ("粗糙", "粗糙度"),
            ("直线", "直线度"),
            ("平面", "平面度"),
            ("圆柱", "圆柱度"),
            ("圆度", "圆度"),
            ("倒角", "倒角"),
            ("技术要求", "技术要求"),
            ("标题", "标题栏"),
            ("标题框", "标题栏"),
            ("明细", "明细表"),
            ("附加", "附加栏"),
            ("螺纹", "直径尺寸"),
            ("厚度", "长度尺寸"),
            ("孔", "直径尺寸"),
            ("其他", "其他"),
        ]
        for kw, std_label in keyword_map:
            if kw in raw and std_label in STANDARD_LABELS:
                return std_label

        # 5. 都不匹配，返回原值
        return raw

    def align_all_labels(self) -> int:
        """
        对当前所有标注的 label 执行对齐到 class.txt 的操作。

        返回: 被修改的标注数量
        """
        if not self.markers:
            messagebox.showinfo("提示", "当前没有标注数据")
            return 0

        changed = 0
        for marker in self.markers.values():
            original = marker.label
            aligned = self._align_label_to_class(original)
            if aligned != original:
                marker.label = aligned
                changed += 1
                self.log.log_operation(f"Label对齐: '{original}' -> '{aligned}'")

        if changed > 0:
            self.refresh_table()
            self.save_config()
            messagebox.showinfo("对齐完成", f"已对齐 {changed} 个标注的 Label 到 class.txt 标准类别")
        else:
            messagebox.showinfo("对齐完成", "所有 Label 已符合 class.txt 标准类别，无需调整")
        return changed

    def _compare_annotations(
        self, predicted: List[Dict[str, str]], ground_truth: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        比对两组标注数据（不依赖坐标，仅比对 label + description）

        使用多集合(bag-of-items)匹配策略，正确处理多个相同尺寸的情况。
        匹配优先级：完全匹配（含标点差异忽略） > 近似匹配（描述互相包含） > label匹配但description不同 > 完全不匹配

        返回: {
            "total": 预测总数,
            "gt_total": 正确答案总数,
            "matched": 完全匹配数,
            "approx": 近似匹配数（标签相同且描述互相包含）,
            "extra": 多余预测数（不在正确答案中）,
            "missing": 遗漏数（正确答案中未被预测）,
            "mismatch": label匹配但description不一致数,
            "accuracy": 准确率,
            "matched_items": [...],  # 匹配详情
            "approx_items": [...],   # 近似匹配详情
            "extra_items": [...],    # 多余预测
            "missing_items": [...],  # 遗漏项
            "mismatch_items": [...], # 标签匹配但描述不同
            "comparison_log": [...], # 详细比对过程步骤
        }
        """
        pred_used = [False] * len(predicted)
        gt_used = [False] * len(ground_truth)

        matched_items: List[Dict[str, Any]] = []
        approx_items: List[Dict[str, Any]] = []
        mismatch_items: List[Dict[str, Any]] = []
        extra_items: List[Dict[str, Any]] = []
        missing_items: List[Dict[str, Any]] = []
        comparison_log: List[Dict[str, Any]] = []  # 详细比对过程日志

        # 第一轮：label+description 完全匹配
        comparison_log.append({
            "step": "round1",
            "title": "【第1轮】完全匹配（label + description 完全相同）",
            "detail": "",
        })
        for gi, gt_item in enumerate(ground_truth):
            gt_label = gt_item["label"]
            gt_desc = self._normalize_desc(gt_item["description"])
            found_match = False
            for pi, pred_item in enumerate(predicted):
                if pred_used[pi]:
                    continue
                if pred_item["label"] == gt_label and self._normalize_desc(pred_item["description"]) == gt_desc:
                    pred_used[pi] = True
                    gt_used[gi] = True
                    matched_items.append({
                        "pred_no": pi + 1,
                        "pred_label": pred_item["label"],
                        "pred_desc": pred_item["description"],
                        "gt_no": gi + 1,
                        "gt_label": gt_item["label"],
                        "gt_desc": gt_item["description"],
                    })
                    comparison_log.append({
                        "step": "round1_match",
                        "title": "",
                        "detail": (
                            f"✅ 预测[#{pi + 1}] {pred_item['label']} | {pred_item['description']}"
                            f"  ←完全一致→  答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        ),
                    })
                    found_match = True
                    break
            if not found_match:
                comparison_log.append({
                    "step": "round1_skip",
                    "title": "",
                    "detail": (
                        f"➖ 答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        f"  → 本轮未找到完全匹配，留待下一轮处理"
                    ),
                })

        # 第二轮：近似匹配（label 相同 + description 互相包含，但非完全相同）
        comparison_log.append({
            "step": "round2_approx",
            "title": "【第2轮】近似匹配（标签相同且描述互相包含，多出字符可接受）",
            "detail": "",
        })
        for gi, gt_item in enumerate(ground_truth):
            if gt_used[gi]:
                continue
            gt_label = gt_item["label"]
            gt_desc_norm = self._normalize_desc(gt_item["description"])
            found_match = False
            for pi, pred_item in enumerate(predicted):
                if pred_used[pi]:
                    continue
                if pred_item["label"] != gt_label:
                    continue
                pred_desc_norm = self._normalize_desc(pred_item["description"])
                if pred_desc_norm == gt_desc_norm:
                    continue  # 理论上 round1 已命中，保险起见跳过
                if self._is_approx_match(pred_desc_norm, gt_desc_norm):
                    pred_used[pi] = True
                    gt_used[gi] = True
                    approx_items.append({
                        "pred_no": pi + 1,
                        "pred_label": pred_item["label"],
                        "pred_desc": pred_item["description"],
                        "gt_no": gi + 1,
                        "gt_label": gt_item["label"],
                        "gt_desc": gt_item["description"],
                    })
                    comparison_log.append({
                        "step": "round2_approx",
                        "title": "",
                        "detail": (
                            f"🟡 预测[#{pi + 1}] {pred_item['label']} | {pred_item['description']}"
                            f"  ←近似匹配（描述互相包含）→  答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        ),
                    })
                    found_match = True
                    break
            if not found_match:
                comparison_log.append({
                    "step": "round2_approx_skip",
                    "title": "",
                    "detail": (
                        f"➖ 答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        f"  → 本轮未找到近似匹配，留待下一轮处理"
                    ),
                })

        # 第三轮：仅 label 匹配（description 不同）
        comparison_log.append({
            "step": "round2",
            "title": "【第3轮】仅 label 匹配（描述不同，记为不匹配项）",
            "detail": "",
        })
        for gi, gt_item in enumerate(ground_truth):
            if gt_used[gi]:
                continue
            gt_label = gt_item["label"]
            found_match = False
            for pi, pred_item in enumerate(predicted):
                if pred_used[pi]:
                    continue
                if pred_item["label"] == gt_label:
                    pred_used[pi] = True
                    gt_used[gi] = True
                    mismatch_items.append({
                        "pred_no": pi + 1,
                        "pred_label": pred_item["label"],
                        "pred_desc": pred_item["description"],
                        "gt_no": gi + 1,
                        "gt_label": gt_item["label"],
                        "gt_desc": gt_item["description"],
                    })
                    comparison_log.append({
                        "step": "round2_mismatch",
                        "title": "",
                        "detail": (
                            f"⚠️ 预测[#{pi + 1}] {pred_item['label']} | {pred_item['description']}"
                            f"  ←标签相同，描述不同→  答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        ),
                    })
                    found_match = True
                    break
            if not found_match:
                comparison_log.append({
                    "step": "round2_skip",
                    "title": "",
                    "detail": (
                        f"➖ 答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        f"  → 未找到相同标签的预测，将记为【遗漏】"
                    ),
                })

        # 剩余未匹配的预测 = 多余
        comparison_log.append({
            "step": "extra",
            "title": "【剩余】多余预测（答案中不存在的项）",
            "detail": "",
        })
        for pi, pred_item in enumerate(predicted):
            if not pred_used[pi]:
                extra_items.append({
                    "pred_no": pi + 1,
                    "pred_label": pred_item["label"],
                    "pred_desc": pred_item["description"],
                })
                comparison_log.append({
                    "step": "extra_item",
                    "title": "",
                    "detail": (
                        f"❌ 预测[#{pi + 1}] {pred_item['label']} | {pred_item['description']}"
                        f"  → 多余预测（答案中无对应项）"
                    ),
                })

        # 剩余未匹配的 GT = 遗漏
        comparison_log.append({
            "step": "missing",
            "title": "【剩余】遗漏项（预测中未识别到的正确答案）",
            "detail": "",
        })
        for gi, gt_item in enumerate(ground_truth):
            if not gt_used[gi]:
                missing_items.append({
                    "gt_no": gi + 1,
                    "gt_label": gt_item["label"],
                    "gt_desc": gt_item["description"],
                })
                comparison_log.append({
                    "step": "missing_item",
                    "title": "",
                    "detail": (
                        f"❌ 答案[#{gi + 1}] {gt_item['label']} | {gt_item['description']}"
                        f"  → 遗漏（预测中未识别）"
                    ),
                })

        matched = len(matched_items)
        approx = len(approx_items)
        mismatch = len(mismatch_items)
        extra = len(extra_items)
        missing = len(missing_items)
        total = len(predicted)
        gt_total = len(ground_truth)

        denominator = matched + mismatch + extra + missing
        accuracy = matched / denominator if denominator > 0 else 0.0

        return {
            "total": total,
            "gt_total": gt_total,
            "matched": matched,
            "approx": approx,
            "extra": extra,
            "missing": missing,
            "mismatch": mismatch,
            "accuracy": accuracy,
            "matched_items": matched_items,
            "approx_items": approx_items,
            "extra_items": extra_items,
            "missing_items": missing_items,
            "mismatch_items": mismatch_items,
            "comparison_log": comparison_log,
        }

    def open_comparison_dialog(self) -> None:
        """打开结果比对对话框"""
        if not self.markers:
            messagebox.showinfo("提示", "当前没有标注数据，请先执行自动标注或导入标注")
            return

        # 加载预测数据
        predicted = [
            {"label": m.label, "description": m.description}
            for m in self.markers.values()
        ]

        # 选择正确答案文件
        gt_path = filedialog.askopenfilename(
            title="选择正确答案（标注JSON文件）",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if not gt_path:
            return

        try:
            gt_data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("错误", f"读取正确答案失败：{e}")
            return

        # 解析正确答案
        gt_shapes = gt_data.get("shapes", [])
        if not isinstance(gt_shapes, list) or not gt_shapes:
            messagebox.showwarning("警告", "正确答案文件中没有有效的标注数据")
            return

        ground_truth = [
            {"label": s.get("label", ""), "description": s.get("description", "")}
            for s in gt_shapes
        ]

        # 执行比对
        result = self._compare_annotations(predicted, ground_truth)
        self._show_comparison_report(result, gt_path)

    def _show_comparison_report(self, result: Dict[str, Any], gt_path: str) -> None:
        """显示比对报告对话框"""
        dialog = tk.Toplevel(self)
        dialog.title("结果比对报告")
        dialog.geometry("960x720")
        dialog.transient(self)

        # 居中
        dialog.update_idletasks()
        w, h = 960, 720
        sw = dialog.winfo_screenwidth()
        sh = dialog.winfo_screenheight()
        dialog.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # 顶部摘要
        summary_frame = ttk.LabelFrame(dialog, text="比对摘要", padding=12)
        summary_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        accuracy = result["accuracy"]
        acc_color = "#16a34a" if accuracy >= 0.8 else "#ca8a04" if accuracy >= 0.5 else "#dc2626"

        stats = [
            ("AI 预测数", result["total"], "#1d4ed8"),
            ("正确答案数", result["gt_total"], "#6b21a8"),
            ("完全匹配", result["matched"], "#16a34a"),
            ("近似匹配", result["approx"], "#ca8a04"),
            ("标签匹配但描述不同", result["mismatch"], "#ea580c"),
            ("多余预测（错误）", result["extra"], "#dc2626"),
            ("遗漏（未识别）", result["missing"], "#dc2626"),
        ]

        for i, (label, value, color) in enumerate(stats):
            row, col = divmod(i, 3)
            cell = ttk.Frame(summary_frame)
            cell.grid(row=row, column=col, padx=15, pady=4, sticky="w")
            ttk.Label(cell, text=label + ":", foreground="#555").pack(side=tk.LEFT)
            ttk.Label(cell, text=str(value), foreground=color, font=("", 12, "bold")).pack(side=tk.LEFT, padx=(4, 0))

        # 准确率
        acc_frame = ttk.Frame(summary_frame)
        acc_frame.grid(row=3, column=0, columnspan=3, pady=(8, 0), sticky="ew")
        ttk.Label(acc_frame, text="综合准确率:", foreground="#555").pack(side=tk.LEFT)
        ttk.Label(acc_frame, text=f"{accuracy:.1%}", foreground=acc_color, font=("", 16, "bold")).pack(side=tk.LEFT, padx=(8, 0))

        # 答案文件信息
        info_frame = ttk.Frame(summary_frame)
        info_frame.grid(row=4, column=0, columnspan=3, pady=(6, 0), sticky="ew")
        ttk.Label(info_frame, text=f"正确答案文件: {Path(gt_path).name}", foreground="#666").pack(anchor=tk.W)

        # 详细对比表格
        detail_frame = ttk.LabelFrame(dialog, text="详细对比（表格列：预测序号/答案序号/标签/描述）", padding=6)
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Notebook 分标签页
        notebook = ttk.Notebook(detail_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        def build_tree(parent: ttk.Frame, columns: List[Tuple[str, str, int]], data: List[Dict[str, Any]]):
            frame = ttk.Frame(parent)
            tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=10)
            for col_id, text, width in columns:
                tree.heading(col_id, text=text)
                tree.column(col_id, width=width, anchor=tk.W, stretch=True if col_id == columns[-1][0] else False)
            vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            for row_data in data:
                values = [str(row_data.get(c[0], "")) for c in columns]
                tree.insert("", tk.END, values=values)
            return frame

        # Tab 0: 详细比对过程（新增）
        log_frame = ttk.Frame(notebook)
        log_tree_container = ttk.Frame(log_frame)
        log_tree_container.pack(fill=tk.BOTH, expand=True)

        # 使用可滚动的 Text 控件显示详细过程，支持多行和颜色
        log_text = tk.Text(log_tree_container, wrap=tk.WORD, height=12, font=("Consolas", 10))
        log_vsb = ttk.Scrollbar(log_tree_container, orient=tk.VERTICAL, command=log_text.yview)
        log_hsb = ttk.Scrollbar(log_tree_container, orient=tk.HORIZONTAL, command=log_text.xview)
        log_text.configure(yscrollcommand=log_vsb.set, xscrollcommand=log_hsb.set)
        log_text.grid(row=0, column=0, sticky="nsew")
        log_vsb.grid(row=0, column=1, sticky="ns")
        log_hsb.grid(row=1, column=0, sticky="ew")
        log_tree_container.rowconfigure(0, weight=1)
        log_tree_container.columnconfigure(0, weight=1)

        # 配置颜色标签
        log_text.tag_configure("header", foreground="#1d4ed8", font=("Microsoft YaHei", 10, "bold"))
        log_text.tag_configure("match", foreground="#16a34a")
        log_text.tag_configure("approx", foreground="#ca8a04")
        log_text.tag_configure("mismatch", foreground="#ea580c")
        log_text.tag_configure("error", foreground="#dc2626")
        log_text.tag_configure("skip", foreground="#6b7280")
        log_text.tag_configure("normal", foreground="#111827")

        # 填入比对过程日志
        log_text.configure(state=tk.NORMAL)
        log_text.insert(tk.END, "═══════════════════════════════════════════════════════════════\n", "normal")
        log_text.insert(tk.END, "  详细比对过程（逐步骤）\n", "header")
        log_text.insert(tk.END, "  预测序号：当前AI识别的标注序号  |  答案序号：正确答案JSON中的序号\n", "normal")
        log_text.insert(tk.END, "═══════════════════════════════════════════════════════════════\n\n", "normal")

        for item in result.get("comparison_log", []):
            title = item.get("title", "")
            detail = item.get("detail", "")
            step = item.get("step", "")

            if title:
                log_text.insert(tk.END, "\n" + title + "\n", "header")
                log_text.insert(tk.END, "─" * 70 + "\n", "normal")

            if detail:
                # 根据不同步骤类型使用不同颜色
                if step in ("round1_match",):
                    tag = "match"
                elif step in ("round2_approx",):
                    tag = "approx"
                elif step in ("round2_mismatch",):
                    tag = "mismatch"
                elif step in ("extra_item", "missing_item"):
                    tag = "error"
                elif step in ("round1_skip", "round2_skip", "round2_approx_skip"):
                    tag = "skip"
                else:
                    tag = "normal"
                log_text.insert(tk.END, detail + "\n", tag)

        log_text.insert(tk.END, "\n═══════════════════════════════════════════════════════════════\n", "normal")
        log_text.insert(tk.END, f"  汇总：完全匹配 {result['matched']} | 近似匹配 {result['approx']} | 描述不同 {result['mismatch']} | 多余 {result['extra']} | 遗漏 {result['missing']}\n", "header")
        log_text.insert(tk.END, "═══════════════════════════════════════════════════════════════\n", "normal")
        log_text.configure(state=tk.DISABLED)  # 只读

        log_count = len(result.get("comparison_log", []))
        notebook.add(log_frame, text=f"📜 详细比对过程 ({log_count})")

        # Tab 1: 匹配
        if result["matched_items"]:
            cols1 = [
                ("pred_no", "预测#", 60),
                ("gt_no", "答案#", 60),
                ("pred_label", "标签", 110),
                ("gt_desc", "描述", 320),
            ]
            tab1 = build_tree(ttk.Frame(notebook), cols1, result["matched_items"])
            notebook.add(tab1, text=f"✅ 匹配 ({result['matched']})")

        # Tab 2: 近似匹配
        if result["approx_items"]:
            cols_approx = [
                ("pred_no", "预测#", 55),
                ("gt_no", "答案#", 55),
                ("pred_label", "标签", 90),
                ("pred_desc", "AI描述", 190),
                ("gt_desc", "正确描述", 190),
            ]
            tab_approx = build_tree(ttk.Frame(notebook), cols_approx, result["approx_items"])
            notebook.add(tab_approx, text=f"🟡 近似匹配 ({result['approx']})")

        # Tab 3: 描述不匹配
        if result["mismatch_items"]:
            cols2 = [
                ("pred_no", "预测#", 55),
                ("gt_no", "答案#", 55),
                ("pred_label", "标签", 90),
                ("pred_desc", "AI描述", 190),
                ("gt_desc", "正确描述", 190),
            ]
            tab2 = build_tree(ttk.Frame(notebook), cols2, result["mismatch_items"])
            notebook.add(tab2, text=f"⚠️ 描述不同 ({result['mismatch']})")

        # Tab 3: 多余预测
        if result["extra_items"]:
            cols3 = [
                ("pred_no", "预测#", 60),
                ("pred_label", "标签", 120),
                ("pred_desc", "AI描述", 380),
            ]
            tab3 = build_tree(ttk.Frame(notebook), cols3, result["extra_items"])
            notebook.add(tab3, text=f"❌ 多余 ({result['extra']})")

        # Tab 4: 遗漏
        if result["missing_items"]:
            cols4 = [
                ("gt_no", "答案#", 60),
                ("gt_label", "标签", 120),
                ("gt_desc", "正确描述", 380),
            ]
            tab4 = build_tree(ttk.Frame(notebook), cols4, result["missing_items"])
            notebook.add(tab4, text=f"❌ 遗漏 ({result['missing']})")

        if not result["matched_items"] and not result["approx_items"] and not result["extra_items"] and not result["missing_items"] and not result["mismatch_items"]:
            ttk.Label(detail_frame, text="没有可对比的数据", foreground="#888").pack(pady=50)

        # 底部：导出按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=12, pady=(6, 12))

        def export_report():
            save_path = filedialog.asksaveasfilename(
                title="保存比对报告",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json")],
                initialfile="comparison_report.json"
            )
            if save_path:
                try:
                    Path(save_path).write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    messagebox.showinfo("导出成功", f"报告已保存到：{save_path}")
                except Exception as e:
                    messagebox.showerror("导出失败", str(e))

        ttk.Button(btn_frame, text="导出报告", command=export_report).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT)

        self.log.log_operation(f"结果比对完成: 准确率{accuracy:.1%}, 匹配{result['matched']}/近似{result['approx']}/多余{result['extra']}/遗漏{result['missing']}/描述不同{result['mismatch']}")

    # ==================== 自动标注 ====================

    def auto_annotate(self) -> None:
        """一键自动标注：调用大模型API"""
        if self.original_image is None:
            messagebox.showinfo("未加载图片", "请先导入PDF或图片文件")
            return
        if requests is None:
            messagebox.showerror("缺少依赖", "请先安装 requests：pip install requests")
            self.log.log_error("自动标注失败: requests未安装")
            return
        if self.is_loading:
            messagebox.showinfo("处理中", "正在进行自动标注，请稍候...")
            return

        if self.markers:
            if not messagebox.askyesno("确认", "当前已有标注数据，执行自动标注将清除现有标注。是否继续？"):
                return

        self.log.log_operation("开始自动标注")
        self._show_progress_window("正在调用AI模型识别...")
        self.is_loading = True
        self.btn_auto_annotate.config(state="disabled")
        self.btn_import_pdf.config(state="disabled")
        self.btn_import_image.config(state="disabled")

        thread = threading.Thread(target=self._do_auto_annotate, daemon=True)
        thread.start()

    def _do_auto_annotate(self) -> None:
        """执行自动标注（子线程）"""
        try:
            image = self.original_image
            image_width = image.width
            image_height = image.height

            self._update_progress(20, "正在处理图片...")

            # 图片转base64
            img_buffer = io.BytesIO()
            image.convert("RGB").save(img_buffer, format="JPEG", quality=95)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            self._update_progress(40, "正在发送到AI模型...")

            prompt = API_PROMPT_TEMPLATE.format(image_width=image_width, image_height=image_height)
            self.log.log_ai_request(API_URL, prompt)

            data = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }],
                "temperature": 0.1,
                "max_tokens": MAX_TOKENS
            }

            self._update_progress(50, "等待AI响应...")

            response = requests.post(API_URL, headers=API_HEADERS, json=data, timeout=120)

            self._update_progress(80, "处理AI响应...")

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                self.log.log_ai_response(content)
                annotation_data = self._parse_api_response(content)
                self._update_progress(95, "加载标注数据...")
                self.after(0, self._finish_auto_annotate, annotation_data)
            else:
                error_msg = f"API请求失败，状态码：{response.status_code}\n{response.text}"
                self.log.log_error(f"API请求失败: 状态码{response.status_code}, URL={API_URL}")
                self.after(0, self._handle_auto_annotate_error, error_msg)

        except requests.exceptions.Timeout:
            self.log.log_error("API请求超时")
            self.after(0, self._handle_auto_annotate_error, "API请求超时，请检查网络或稍后重试")
        except Exception as exc:
            self.log.log_error(f"自动标注异常: {exc}", exc_info=True)
            self.after(0, self._handle_auto_annotate_error, f"自动标注失败：{str(exc)}")

    def _parse_api_response(self, content: str) -> Any:
        """
        解析API返回的JSON数据（多级回退策略）

        解析优先级：
        1. #JSON#...#JSON# 标记包裹（最可靠）
        2. 直接尝试整体JSON解析
        3. markdown ```json 代码块
        4. 提取 JSON 数组 [...]
        5. 提取 JSON 对象 {...}
        """
        # ---- 优先级1: 使用 #JSON# 标记提取 ----
        tag_start = content.find(JSON_TAG)
        if tag_start != -1:
            after_start = content[tag_start + len(JSON_TAG):]
            tag_end = after_start.find(JSON_TAG)
            if tag_end != -1:
                json_str = after_start[:tag_end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # 标记内解析失败，继续尝试其他方式
                    pass

        # ---- 优先级2: 直接尝试整体JSON解析 ----
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # ---- 优先级3: markdown ```json 代码块 ----
        json_match = re.search(r'```json\s*\n?(.*?)```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # ---- 优先级4: 提取 JSON 数组 ----
        # 依次尝试每个 [ 到对应 ] 的组合，优先返回包含字典项的最长数组
        best_list = None
        best_score = -1
        for i in range(len(content)):
            if content[i] == '[':
                depth = 0
                for j in range(i, len(content)):
                    if content[j] == '[':
                        depth += 1
                    elif content[j] == ']':
                        depth -= 1
                        if depth == 0:
                            candidate = content[i:j + 1]
                            if candidate.count('[') == candidate.count(']'):
                                try:
                                    result = json.loads(candidate)
                                    if isinstance(result, list):
                                        # 评分：包含字典项越多越好，其次长度越长越好
                                        dict_count = sum(1 for item in result if isinstance(item, dict))
                                        score = dict_count * 1000 + len(result)
                                        if score > best_score:
                                            best_score = score
                                            best_list = result
                                except json.JSONDecodeError:
                                    pass
                            break
        if best_list is not None:
            return best_list

        # ---- 优先级5: 提取 JSON 对象 {...} ----
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            candidate = content[start_idx:end_idx + 1]
            if candidate.count('{') == candidate.count('}'):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        return None

    def _convert_compact_to_anylabeling(self, compact_data: List[Dict[str, Any]],
                                         image_width: int, image_height: int,
                                         image_path: str) -> Dict[str, Any]:
        """
        将紧凑格式转换为x-anylabeling标准格式

        紧凑格式: [{"label":"长度尺寸标注","description":"标注=22 ±0.05","score":90,"x1":838.9,"y1":733.3,"x2":909.3,"y2":992.6}]
        标准格式: {version, flags, shapes:[{label, score, points:[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], group_id, description, difficult, shape_type, flags, attributes}], imagePath, imageData, imageHeight, imageWidth, description}
        """
        shapes = []
        for idx, item in enumerate(compact_data, start=1):
            label = self._align_label_to_class(item.get("label", ""))
            description = self._strip_desc_prefix(item.get("description", ""))
            score = item.get("score", 90)
            x1 = item.get("x1", 0)
            y1 = item.get("y1", 0)
            x2 = item.get("x2", 0)
            y2 = item.get("y2", 0)

            # x1,y1,x2,y2 -> 四角坐标: 左上、右上、右下、左下
            points = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

            shape = {
                "label": label,
                "score": score,
                "points": points,
                "group_id": idx,
                "description": description,
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
            shapes.append(shape)

        return {
            "version": "3.2.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
            "description": image_path,
        }

    def _finish_auto_annotate(self, annotation_data: Any) -> None:
        """完成自动标注"""
        self._hide_progress_window()
        self.is_loading = False
        self.btn_auto_annotate.config(state="normal")
        self.btn_import_pdf.config(state="normal")
        self.btn_import_image.config(state="normal")

        if annotation_data is None:
            self.log.log_error("AI响应解析失败: 返回数据无法解析为JSON")
            messagebox.showerror("解析失败", "无法解析AI返回的标注数据")
            return

        # 如果是紧凑数组格式，转换为标准格式
        if isinstance(annotation_data, list):
            if self.original_image is None or self.image_path is None:
                messagebox.showerror("数据错误", "缺少图片信息，无法转换标注格式")
                return
            annotation_data = self._convert_compact_to_anylabeling(
                annotation_data,
                image_width=self.original_image.width,
                image_height=self.original_image.height,
                image_path=self.image_path.name,
            )
            self.log.log_ai(f"格式转换完成: {len(annotation_data.get('shapes', []))} 个标注")

        count = self._load_shapes_from_dict(annotation_data)
        self.log.log_ai_success(count)
        if count > 0:
            self.log.log_operation(f"自动标注完成: 识别{count}个标注")
            messagebox.showinfo("自动标注完成", f"AI识别完成！\n共识别 {count} 个标注")
        else:
            self.log.log_operation("自动标注完成: 未识别到有效标注")
            messagebox.showwarning("标注为空", "AI未识别到有效的标注数据")

    def _handle_auto_annotate_error(self, error_msg: str) -> None:
        """处理自动标注错误"""
        self._hide_progress_window()
        self.is_loading = False
        self.btn_auto_annotate.config(state="normal")
        self.btn_import_pdf.config(state="normal")
        self.btn_import_image.config(state="normal")
        self.log.log_error(f"自动标注失败: {error_msg}")
        messagebox.showerror("自动标注失败", error_msg)

    def _show_progress_window(self, message: str) -> None:
        """显示进度窗口"""
        self.progress_window = tk.Toplevel(self)
        self.progress_window.title("处理中")
        self.progress_window.geometry("300x100")
        self.progress_window.transient(self)
        self.progress_window.grab_set()
        self.progress_window.resizable(False, False)

        self.progress_window.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 300) // 2
        y = self.winfo_y() + (self.winfo_height() - 100) // 2
        self.progress_window.geometry(f"+{x}+{y}")

        self.progress_label = ttk.Label(self.progress_window, text=message)
        self.progress_label.pack(pady=(15, 10))

        self.progress_bar = ttk.Progressbar(self.progress_window, mode='determinate', maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20)
        self.progress_bar['value'] = 10

    def _update_progress(self, value: int, message: str) -> None:
        """更新进度（子线程安全）"""
        self.after(0, self._do_update_progress, value, message)

    def _do_update_progress(self, value: int, message: str) -> None:
        """实际更新进度UI"""
        if self.progress_window and self.progress_bar and self.progress_label:
            self.progress_bar['value'] = value
            self.progress_label.config(text=message)

    def _hide_progress_window(self) -> None:
        """隐藏进度窗口"""
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
            self.progress_bar = None
            self.progress_label = None

    # ==================== 标注导入 ====================

    def import_annotation(self) -> None:
        """从JSON文件导入标注"""
        if self.original_image is None:
            messagebox.showinfo("未打开图片", "请先导入PDF或图片文件")
            return

        path = filedialog.askopenfilename(title="选择标注JSON文件", filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")])
        if not path:
            return

        self.log.log_operation(f"开始导入标注文件: {path}")
        self.annotation_path = Path(path)
        try:
            data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log.log_error(f"读取标注文件失败: {path}, 错误: {exc}", exc_info=True)
            messagebox.showerror("读取失败", f"无法读取JSON文件：{exc}")
            return

        count = self._load_shapes_from_dict(data)
        if count > 0:
            self.log.log_operation(f"标注导入成功: {path}, 共{count}个标注")
            messagebox.showinfo("导入成功", f"成功导入 {count} 个标注")
        else:
            self.log.log_operation(f"标注导入完成但无有效数据: {path}")
            messagebox.showwarning("导入警告", "未找到有效的标注数据")

    def _load_shapes_from_dict(self, data: Dict[str, Any]) -> int:
        """从字典数据加载标注（公共方法）"""
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            return 0

        self.markers.clear()
        self.checked_markers.clear()
        self.selected_number = None

        count = 0
        for item in shapes:
            label = self._align_label_to_class(item.get("label", ""))
            description = self._strip_desc_prefix(item.get("description", ""))
            points = item.get("points", [])
            shape_type = item.get("shape_type", "rectangle")

            if not points or not isinstance(points, list):
                continue

            x, y = self._calculate_center(points)
            if x is None or y is None:
                continue

            xmin, ymin, xmax, ymax = self._calculate_bbox(points)

            count += 1
            self.markers[count] = Marker(
                number=count, x=x, y=y, label=label, description=description,
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                shape_type=shape_type, display_x=x, display_y=y,
            )

        self.renumber_markers()
        # 智能偏移已关闭（导入标注后不再自动偏移）
        # self.calculate_all_offsets()
        self.refresh_table()
        self.save_config()

        if self.annotation_path:
            self.status_var.set(f"已导入 {count} 个标注 | {self.annotation_path.name}")
        elif self.image_path:
            self.status_var.set(f"AI识别完成 | {self.image_path.name} | {count} 个标注")

        return count

    def _calculate_center(self, points: List[List[float]]) -> tuple:
        """计算多边形点集的几何中心"""
        if not points:
            return None, None
        xs = [p[0] for p in points if isinstance(p, list) and len(p) >= 2]
        ys = [p[1] for p in points if isinstance(p, list) and len(p) >= 2]
        if not xs or not ys:
            return None, None
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _calculate_bbox(self, points: List[List[float]]) -> tuple:
        """计算多边形点集的边界框"""
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        xs = [p[0] for p in points if isinstance(p, list) and len(p) >= 2]
        ys = [p[1] for p in points if isinstance(p, list) and len(p) >= 2]
        if not xs or not ys:
            return 0.0, 0.0, 0.0, 0.0
        return min(xs), min(ys), max(xs), max(ys)

    # ==================== 智能偏移算法 ====================

    def calculate_all_offsets(self) -> None:
        """计算所有标注的最佳显示位置"""
        if not self.markers or self.original_image is None:
            return
        image_width = self.original_image.width
        image_height = self.original_image.height

        for marker in self.markers.values():
            marker.font_size_override = None
            marker.display_x = marker.x
            marker.display_y = marker.y

        for marker in self.markers.values():
            self._calculate_single_offset(marker, image_width, image_height)

        self._resolve_overlaps()
        self._group_stagger()
        self.redraw_markers()
        self.save_config()

    def _calculate_single_offset(self, marker: Marker, image_width: float, image_height: float) -> None:
        """计算单个标注的最佳偏移"""
        bbox_width = marker.xmax - marker.xmin
        bbox_height = marker.ymax - marker.ymin

        if bbox_width <= 0 or bbox_height <= 0:
            marker.display_x = marker.x
            marker.display_y = marker.y
            return

        aspect_ratio = bbox_width / bbox_height
        font_size = marker.font_size_override if marker.font_size_override else self.font_size_var.get()
        radius = self._calculate_radius(font_size, len(str(marker.number)))

        if marker.shape_type == "polygon" or aspect_ratio > self.THIN_ASPECT_RATIO:
            directions = self._evaluate_top_bottom(marker, radius, image_width, image_height)
        elif aspect_ratio < self.WIDE_ASPECT_RATIO:
            directions = self._evaluate_left_right(marker, radius, image_width, image_height)
        else:
            directions = self._evaluate_all_directions(marker, radius, image_width, image_height)

        if directions:
            best_dir = max(directions, key=lambda d: d[1])
            marker.display_x = marker.x + best_dir[0][0]
            marker.display_y = marker.y + best_dir[0][1]

        marker.display_x = max(radius, min(image_width - radius, marker.display_x))
        marker.display_y = max(radius, min(image_height - radius, marker.display_y))

    def _calculate_radius(self, font_size: int, digits: int) -> float:
        """计算圆圈半径"""
        return max(font_size * 0.5 + 1, int(font_size * self.RADIUS_FONT_RATIO + digits * font_size * self.RADIUS_DIGIT_RATIO + self.RADIUS_BASE * self.OFFSET_DISTANCE_RATIO))

    def _evaluate_top_bottom(self, marker, radius, image_width, image_height):
        """评估上下方向"""
        directions = []
        dist_to_top = marker.y - marker.ymin
        dist_to_bottom = marker.ymax - marker.y
        offset_up = dist_to_top - radius
        offset_down = dist_to_bottom - radius
        top_space = marker.ymin - radius
        bottom_space = image_height - marker.ymax - radius

        if offset_up > 0 and top_space > 0:
            score = top_space
            if marker.ymin < image_height * self.EDGE_PADDING_RATIO:
                score *= self.EDGE_PENALTY_FACTOR
            directions.append(((0, -offset_up), score))

        if offset_down > 0 and bottom_space > 0:
            score = bottom_space
            if marker.ymax > image_height * (1 - self.EDGE_PADDING_RATIO):
                score *= self.EDGE_PENALTY_FACTOR
            directions.append(((0, offset_down), score))
        return directions

    def _evaluate_left_right(self, marker, radius, image_width, image_height):
        """评估左右方向"""
        directions = []
        dist_to_left = marker.x - marker.xmin
        dist_to_right = marker.xmax - marker.x
        offset_left = dist_to_left - radius
        offset_right = dist_to_right - radius
        left_space = marker.xmin - radius
        right_space = image_width - marker.xmax - radius

        if offset_left > 0 and left_space > 0:
            score = left_space
            if marker.xmin < image_width * self.EDGE_PADDING_RATIO:
                score *= self.EDGE_PENALTY_FACTOR
            directions.append(((-offset_left, 0), score))

        if offset_right > 0 and right_space > 0:
            score = right_space
            if marker.xmax > image_width * (1 - self.EDGE_PADDING_RATIO):
                score *= self.EDGE_PENALTY_FACTOR
            directions.append(((offset_right, 0), score))
        return directions

    def _evaluate_all_directions(self, marker, radius, image_width, image_height):
        """评估四个方向"""
        directions = []
        dist_to_top = marker.y - marker.ymin
        dist_to_bottom = marker.ymax - marker.y
        dist_to_left = marker.x - marker.xmin
        dist_to_right = marker.xmax - marker.x
        offset_up = dist_to_top - radius
        offset_down = dist_to_bottom - radius
        offset_left = dist_to_left - radius
        offset_right = dist_to_right - radius
        top_space = marker.ymin - radius
        bottom_space = image_height - marker.ymax - radius
        left_space = marker.xmin - radius
        right_space = image_width - marker.xmax - radius

        if offset_up > 0 and top_space > 0:
            directions.append(((0, -offset_up), top_space))
        if offset_down > 0 and bottom_space > 0:
            directions.append(((0, offset_down), bottom_space))
        if offset_left > 0 and left_space > 0:
            directions.append(((-offset_left, 0), left_space))
        if offset_right > 0 and right_space > 0:
            directions.append(((offset_right, 0), right_space))
        return directions

    def _resolve_overlaps(self) -> None:
        """解决标注重叠"""
        if len(self.markers) < 2:
            return
        font_size = self.font_size_var.get()
        changed = True
        iterations = 0

        while changed and iterations < self.OVERLAP_MAX_ITERATIONS:
            changed = False
            iterations += 1
            marker_list = sorted(self.markers.values(), key=lambda m: m.number)

            for i, m1 in enumerate(marker_list):
                r1 = self._calculate_radius(m1.font_size_override if m1.font_size_override else font_size, len(str(m1.number)))
                for j in range(i + 1, len(marker_list)):
                    m2 = marker_list[j]
                    r2 = self._calculate_radius(m2.font_size_override if m2.font_size_override else font_size, len(str(m2.number)))

                    dist = math.hypot(m1.display_x - m2.display_x, m1.display_y - m2.display_y)
                    min_dist = r1 + r2 + self.OVERLAP_MIN_DISTANCE

                    if dist < min_dist:
                        changed = True
                        overlap = min_dist - dist
                        if m1.font_size_override is None and font_size > self.OVERLAP_FONT_MIN:
                            new_size = font_size - 1
                            m1.font_size_override = new_size
                            m2.font_size_override = new_size
                        else:
                            dx = m1.display_x - m2.display_x
                            dy = m1.display_y - m2.display_y
                            if dist > 0:
                                nx, ny = dx / dist, dy / dist
                                m1.display_x += nx * overlap * 0.5
                                m1.display_y += ny * overlap * 0.5
                                m2.display_x -= nx * overlap * 0.5
                                m2.display_y -= ny * overlap * 0.5

    def _group_stagger(self) -> None:
        """分组错开排列"""
        if len(self.markers) < 2:
            return
        font_size = self.font_size_var.get()
        radius = self._calculate_radius(font_size, 2)
        group_threshold = radius * self.GROUP_THRESHOLD_RATIO

        marker_list = sorted(self.markers.values(), key=lambda m: (m.x, m.y))
        groups = []

        for marker in marker_list:
            added = False
            for group in groups:
                for m in group:
                    dist = math.hypot(marker.x - m.x, marker.y - m.y)
                    if dist < group_threshold:
                        group.append(marker)
                        added = True
                        break
                if added:
                    break
            if not added:
                groups.append([marker])

        for group in groups:
            if len(group) <= 1:
                continue
            avg_x = sum(m.x for m in group) / len(group)
            avg_y = sum(m.y for m in group) / len(group)
            vertical = all(abs(m.x - avg_x) < abs(m.y - avg_y) * 1.5 for m in group)

            for idx, marker in enumerate(group):
                if vertical:
                    stagger_y = (idx - (len(group) - 1) / 2) * radius * self.GROUP_STAGGER_RATIO
                    marker.display_y = avg_y + stagger_y
                else:
                    stagger_x = (idx - (len(group) - 1) / 2) * radius * self.GROUP_STAGGER_RATIO
                    marker.display_x = avg_x + stagger_x

    # ==================== 配置文件 ====================

    def load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path or not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            font_size = data.get("font_size") if isinstance(data, dict) else None
            if font_size is not None:
                self.font_size_var.set(max(6, min(72, int(font_size))))
                self.marker_font.configure(size=self.font_size_var.get())
            raw_markers = data.get("markers", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for item in raw_markers:
                marker = Marker(
                    number=int(item["number"]), x=float(item["x"]), y=float(item["y"]),
                    label=item.get("label", ""), description=item.get("description", ""),
                    xmin=float(item.get("xmin", 0)), ymin=float(item.get("ymin", 0)),
                    xmax=float(item.get("xmax", 0)), ymax=float(item.get("ymax", 0)),
                    shape_type=item.get("shape_type", "rectangle"),
                    display_x=float(item.get("display_x", item.get("x", 0))),
                    display_y=float(item.get("display_y", item.get("y", 0))),
                    font_size_override=item.get("font_size_override"),
                )
                self.markers[marker.number] = marker
            self.renumber_markers()
        except Exception as exc:
            messagebox.showwarning("配置读取失败", f"配置文件格式可能不正确：{exc}")

    def save_config(self) -> None:
        """保存配置"""
        if not self.config_path:
            return
        data = {
            "image": self.image_path.name if self.image_path else "",
            "font_size": self.font_size_var.get(),
            "markers": [
                {
                    "number": m.number, "x": round(m.x, 3), "y": round(m.y, 3),
                    "label": m.label, "description": m.description,
                    "xmin": round(m.xmin, 3), "ymin": round(m.ymin, 3),
                    "xmax": round(m.xmax, 3), "ymax": round(m.ymax, 3),
                    "shape_type": m.shape_type,
                    "display_x": round(m.display_x, 3), "display_y": round(m.display_y, 3),
                    "font_size_override": m.font_size_override,
                }
                for m in sorted(self.markers.values(), key=lambda item: item.number)
            ],
        }
        try:
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.log.log_error(f"保存配置失败: {self.config_path}, 错误: {exc}", exc_info=True)

    # ==================== 渲染绘制 ====================

    def render_image(self) -> None:
        """渲染图片"""
        if self.original_image is None:
            return
        self.marker_items.clear()

        width = max(1, int(self.original_image.width * self.zoom))
        height = max(1, int(self.original_image.height * self.zoom))

        background = self.original_image.resize((width, height)).convert("RGBA")
        foreground = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(foreground)

        for marker in sorted(self.markers.values(), key=lambda item: item.number):
            self._draw_marker_on_image(draw, marker)

        combined = Image.alpha_composite(background, foreground)

        target_size = (width, height)
        if self.tk_image is None or self._cached_tk_size != target_size:
            self.tk_image = ImageTk.PhotoImage(combined)
            self._cached_tk_size = target_size
            if self.image_canvas_id is not None:
                self.canvas.delete(self.image_canvas_id)
            self.image_canvas_id = self.canvas.create_image(
                0, 0, image=self.tk_image, anchor=tk.NW, tags=("image",)
            )
        else:
            self.tk_image.paste(combined)
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def _draw_marker_on_image(self, draw, marker):
        """在PIL上绘制标注"""
        sx, sy = int(marker.x * self.zoom), int(marker.y * self.zoom)
        dx, dy = int(marker.display_x * self.zoom), int(marker.display_y * self.zoom)
        text = str(marker.number)
        digits = len(text)
        font_size = marker.font_size_override if marker.font_size_override else self.font_size_var.get()
        radius = int(self._calculate_radius(font_size, digits))
        is_selected = marker.number == self.selected_number

        line_color = self._hex_to_rgba(self.COLOR_LINE, alpha=200)
        normal_color = self._hex_to_rgba(self.COLOR_NORMAL, alpha=255)
        selected_color = self._hex_to_rgba(self.COLOR_SELECTED, alpha=255)
        color = selected_color if is_selected else normal_color

        if abs(sx - dx) > 2 or abs(sy - dy) > 2:
            draw.line([sx, sy, dx, dy], fill=line_color, width=2 if is_selected else 1)

        circle_width = self.CIRCLE_LINE_WIDTH_SELECTED if is_selected else self.CIRCLE_LINE_WIDTH_NORMAL
        draw.ellipse([dx - radius, dy - radius, dx + radius, dy + radius], fill=(0, 0, 0, 0), outline=color, width=circle_width)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((dx, dy), text, fill=color, font=font, anchor="mm")

    def _hex_to_rgba(self, hex_color: str, alpha: int = 255) -> tuple:
        """十六进制转RGBA"""
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (r, g, b, alpha)

    def redraw_markers(self) -> None:
        """重新绘制标注"""
        self.render_image()

    # ==================== 缩放 ====================

    def set_zoom(self, value: float) -> None:
        """设置缩放"""
        if self.original_image is None:
            return
        self.zoom = min(8.0, max(0.1, value))
        self.render_image()

    def set_zoom_factor(self, value: float) -> None:
        """设置缩放因子"""
        if self.original_image is None:
            return
        self.zoom_factor = min(8.0, max(0.1, value))
        self.update_fit_zoom()

    def update_fit_zoom(self) -> None:
        """更新适应窗口缩放"""
        if self.original_image is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if canvas_width <= 1 or canvas_height <= 1:
            self.after(50, self.update_fit_zoom)
            return
        self.fit_zoom = min(canvas_width / self.original_image.width, canvas_height / self.original_image.height)
        self.fit_zoom = max(0.05, min(8.0, self.fit_zoom))
        self.set_zoom(self.fit_zoom * self.zoom_factor)

    def on_canvas_resize(self, _event) -> None:
        """画布resize"""
        if self.original_image is None:
            return
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(120, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        self._resize_after_id = None
        self.update_fit_zoom()

    # ==================== 标注编辑 ====================

    def on_font_size_change(self) -> None:
        """字体大小变更"""
        try:
            size = int(self.font_size_var.get())
        except Exception:
            size = 11
        size = max(6, min(72, size))
        self.font_size_var.set(size)
        self.marker_font.configure(size=size)
        for marker in self.markers.values():
            marker.font_size_override = None
        # 智能偏移已关闭（重置字体后不再自动偏移）
        # self.calculate_all_offsets()
        self.save_config()

    def parse_input(self) -> tuple:
        """解析坐标输入"""
        quick = self.quick_var.get().strip()
        if quick:
            normalized = quick.replace("，", ",").replace("（", "(").replace("）", ")")
            match = re.search(r"^\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*$", normalized)
            if not match:
                raise ValueError("快速输入格式应类似：(120,80)")
            return float(match.group(1)), float(match.group(2))
        if not self.x_var.get().strip() or not self.y_var.get().strip():
            raise ValueError("请输入 X、Y，或使用快速输入格式")
        return float(self.x_var.get()), float(self.y_var.get())

    def add_or_update_from_input(self) -> None:
        """添加或更新标注"""
        if self.original_image is None:
            messagebox.showinfo("未打开图片", "请先导入PDF或图片文件")
            return
        try:
            x, y = self.parse_input()
        except Exception as exc:
            messagebox.showerror("输入错误", str(exc))
            return

        label = self.label_var.get().strip()
        description = self.description_var.get().strip()

        if self.quick_var.get().strip() or self.selected_number is None:
            marker = Marker(number=len(self.markers) + 1, x=x, y=y, label=label, description=description,
                            xmin=x-10, ymin=y-10, xmax=x+10, ymax=y+10, display_x=x, display_y=y)
            self.markers[marker.number] = marker
        else:
            marker = self.markers.get(self.selected_number)
            if marker is None:
                marker = Marker(number=len(self.markers) + 1, x=x, y=y, label=label, description=description,
                                xmin=x-10, ymin=y-10, xmax=x+10, ymax=y+10, display_x=x, display_y=y)
                self.markers[marker.number] = marker
            else:
                marker.x = x
                marker.y = y
                marker.label = label
                marker.description = description

        self.selected_number = marker.number
        self.quick_var.set("")
        self.fill_form(marker)
        # 智能偏移已关闭（添加/编辑标注后不再自动偏移）
        # self.calculate_all_offsets()
        self.refresh_table()
        self.select_tree_row(marker.number)
        self.save_config()

    def delete_selected(self) -> None:
        if self.selected_number is None:
            messagebox.showinfo("未选择", "请先选择一个标注")
            return
        self.delete_marker(self.selected_number)

    def delete_marker(self, number: int) -> None:
        selected_marker = None
        if self.selected_number is not None and self.selected_number != number:
            selected_marker = self.markers.get(self.selected_number)
        self.markers.pop(number, None)
        self.checked_markers.discard(number)
        self.renumber_markers()
        if self.selected_number == number:
            self.selected_number = None
            self.x_var.set("")
            self.y_var.set("")
            self.label_var.set("")
            self.description_var.set("")
        elif selected_marker is not None:
            self.selected_number = selected_marker.number
        # 智能偏移已关闭（删除标注后不再自动偏移）
        # self.calculate_all_offsets()
        self.refresh_table()
        if self.selected_number is not None:
            self.select_tree_row(self.selected_number)
        self.save_config()

    def move_marker(self, number: int, direction: int) -> None:
        marker = self.markers.get(number)
        target = self.markers.get(number + direction)
        if not marker or not target:
            return
        marker.number, target.number = target.number, marker.number
        self.markers[number], self.markers[number + direction] = target, marker
        self.selected_number = target.number
        self.fill_form(marker)
        self.refresh_table()
        self.select_tree_row(target.number)
        self.redraw_markers()
        self.save_config()

    def renumber_markers(self) -> None:
        ordered = sorted(self.markers.values(), key=lambda item: item.number)
        self.markers = {}
        for index, marker in enumerate(ordered, start=1):
            marker.number = index
            self.markers[index] = marker

    # ==================== 表格 ====================

    def refresh_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for marker in sorted(self.markers.values(), key=lambda item: item.number):
            check = "✓" if marker.number in self.checked_markers else ""
            self.tree.insert("", tk.END, iid=str(marker.number),
                values=(check, marker.number, marker.label, marker.description,
                        self.format_coord(marker.x), self.format_coord(marker.y), "↑", "↓", "×"))

    def select_tree_row(self, number: int) -> None:
        iid = str(number)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)

    def on_tree_select(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        number = int(selection[0])
        if number not in self.markers:
            return
        self.selected_number = number
        self.fill_form(self.markers[number])
        self.redraw_markers()

    def on_tree_click(self, event) -> None:
        row_id = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        if not row_id:
            return
        try:
            number = int(row_id)
        except ValueError:
            return

        if column_id == "#1":
            if number in self.checked_markers:
                self.checked_markers.remove(number)
            else:
                self.checked_markers.add(number)
            self.refresh_table()
            self.select_tree_row(number)
        elif column_id == "#3":
            self.start_edit_cell(number, "label", event)
        elif column_id == "#4":
            self.start_edit_cell(number, "description", event)
        elif column_id == "#7":
            self.move_marker(number, -1)
        elif column_id == "#8":
            self.move_marker(number, 1)
        elif column_id == "#9":
            self.delete_marker(number)

    def start_edit_cell(self, number: int, field: str, event) -> None:
        marker = self.markers.get(number)
        if not marker:
            return
        cell = self.tree.identify("cell", event.x, event.y)
        if not cell:
            return
        column = self.tree.identify_column(event.x)
        x, y, width, height = self.tree.bbox(str(number), column)
        if x is None:
            return

        self.edit_entry = ttk.Entry(self.right)
        self.edit_entry.insert(0, getattr(marker, field))
        self.edit_entry.place(x=x, y=y + 56, width=width, height=height)
        self.edit_entry.focus()
        self.edit_entry.select_range(0, tk.END)

        def on_confirm(event=None):
            setattr(marker, field, self.edit_entry.get())
            self.edit_entry.destroy()
            self.refresh_table()
            self.select_tree_row(number)
            self.save_config()

        self.edit_entry.bind("<Return>", on_confirm)
        self.edit_entry.bind("<Escape>", lambda e: self.edit_entry.destroy())
        self.edit_entry.bind("<FocusOut>", on_confirm)

    def load_selected_to_form(self, _event=None) -> None:
        if self.selected_number and self.selected_number in self.markers:
            self.fill_form(self.markers[self.selected_number])

    def fill_form(self, marker: Marker) -> None:
        self.x_var.set(self.format_coord(marker.x))
        self.y_var.set(self.format_coord(marker.y))
        self.label_var.set(marker.label)
        self.description_var.set(marker.description)

    # ==================== 画布交互 ====================

    def on_canvas_click(self, event) -> None:
        if self.original_image is None:
            return
        x = self.canvas.canvasx(event.x) / self.zoom
        y = self.canvas.canvasy(event.y) / self.zoom
        number = self.find_marker_at(x, y)

        if number is None:
            self.x_var.set(self.format_coord(x))
            self.y_var.set(self.format_coord(y))
            self.label_var.set("")
            self.description_var.set("")
            self.selected_number = None
            self.tree.selection_remove(self.tree.selection())
            self.redraw_markers()
            self.dragging_number = None
            return

        self.selected_number = number
        marker = self.markers[number]
        self.dragging_number = number
        self.drag_offset = (marker.display_x - x, marker.display_y - y)
        self.fill_form(marker)
        self.select_tree_row(number)
        self.redraw_markers()

    def on_canvas_drag(self, event) -> None:
        if self.dragging_number is None or self.original_image is None:
            return
        marker = self.markers.get(self.dragging_number)
        if not marker:
            return
        x = self.canvas.canvasx(event.x) / self.zoom + self.drag_offset[0]
        y = self.canvas.canvasy(event.y) / self.zoom + self.drag_offset[1]
        marker.display_x = min(max(x, 0), self.original_image.width - 1)
        marker.display_y = min(max(y, 0), self.original_image.height - 1)
        # 节流：合并连续 B1-Motion 事件，约 60fps 上限
        # 拖动只改 display_x/y，marker.x/y 不变，故无需 fill_form / refresh_table / select_tree_row
        if self._drag_after_id is None:
            self._drag_after_id = self.after(16, self._flush_drag_redraw)

    def _flush_drag_redraw(self) -> None:
        self._drag_after_id = None
        if self.dragging_number is not None:
            self.redraw_markers()

    def on_canvas_release(self, _event) -> None:
        # 取消挂起的节流回调，确保最后一次位置被绘制
        if self._drag_after_id is not None:
            self.after_cancel(self._drag_after_id)
            self._drag_after_id = None
            self.redraw_markers()
        if self.dragging_number is not None:
            self.save_config()
        self.dragging_number = None

    def find_marker_at(self, x: float, y: float) -> Optional[int]:
        best_number = None
        best_distance = float("inf")
        font_size = self.font_size_var.get()
        for marker in self.markers.values():
            fs = marker.font_size_override if marker.font_size_override else font_size
            radius = self._calculate_radius(fs, len(str(marker.number)))
            dist = math.hypot(marker.display_x - x, marker.display_y - y)
            if dist <= radius and dist < best_distance:
                best_number = marker.number
                best_distance = dist
        return best_number

    @staticmethod
    def format_coord(value: float) -> str:
        if abs(value - round(value)) < 0.001:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")


    # ==================== YOLO+VLM 流水线标注 ====================

    # YOLO模型路径（None则回退到整图模式）
    YOLO_MODEL_PATH: Optional[str] = r"E:\1.软件库\1.工具\X-AnyLabeling-main\exp8\weights\best.pt"

    def yolo_vlm_annotate(self) -> None:
        """YOLO+AI标注入口"""
        if self.original_image is None:
            messagebox.showinfo("未加载图片", "请先导入PDF或图片文件")
            return
        if self.is_loading:
            messagebox.showinfo("处理中", "正在处理，请稍候...")
            return

        if self.markers:
            if not messagebox.askyesno("确认", "当前已有标注数据，执行YOLO+AI标注将清除现有标注。是否继续？"):
                return

        self.log.log_operation("开始YOLO+AI流水线标注")
        self._show_progress_window("正在初始化YOLO+AI流水线...")
        self.is_loading = True
        self.btn_yolo_vlm.config(state="disabled")
        self.btn_auto_annotate.config(state="disabled")
        self.btn_import_pdf.config(state="disabled")
        self.btn_import_image.config(state="disabled")

        thread = threading.Thread(target=self._do_yolo_vlm_annotate, daemon=True)
        thread.start()

    def _do_yolo_vlm_annotate(self) -> None:
        """执行YOLO+VLM流水线（子线程）"""
        try:
            from yolovcut import YoloVlmPipeline

            self._update_progress(5, "创建流水线...")

            pipeline = YoloVlmPipeline(
                api_url=API_URL,
                api_headers=API_HEADERS,
                yolo_model_path=self.YOLO_MODEL_PATH,
                max_tokens=MAX_TOKENS,
                max_per_batch=20,
                max_concurrency=5,
            )

            yolo_status = "可用" if pipeline.yolo_available else "不可用(整图模式)"
            self.log.log_ai(f"YOLO+AI流水线启动, YOLO={yolo_status}")

            def progress_cb(pct, msg):
                self._update_progress(pct, msg)

            def log_cb(msg):
                self.log.log_ai(msg)

            result = pipeline.run(
                image=self.original_image,
                image_path=self.image_path.name if self.image_path else "image.png",
                progress_callback=progress_cb,
                log_callback=log_cb,
            )

            self._update_progress(95, "加载标注数据...")
            self.after(0, self._finish_yolo_vlm_annotate, result)

        except ImportError:
            self.log.log_error("yolovcut模块导入失败")
            self.after(0, self._handle_auto_annotate_error, "yolovcut模块导入失败，请确保yolovcut目录存在")
        except Exception as exc:
            self.log.log_error(f"YOLO+AI标注异常: {exc}", exc_info=True)
            self.after(0, self._handle_auto_annotate_error, f"YOLO+AI标注失败：{str(exc)}")

    def _finish_yolo_vlm_annotate(self, annotation_data: Any) -> None:
        """完成YOLO+AI标注"""
        self._hide_progress_window()
        self.is_loading = False
        self.btn_yolo_vlm.config(state="normal")
        self.btn_auto_annotate.config(state="normal")
        self.btn_import_pdf.config(state="normal")
        self.btn_import_image.config(state="normal")

        if annotation_data is None or not isinstance(annotation_data, dict):
            self.log.log_error("YOLO+AI流水线返回数据无效")
            messagebox.showerror("标注失败", "YOLO+AI流水线未返回有效数据")
            return

        count = self._load_shapes_from_dict(annotation_data)
        shapes = annotation_data.get("shapes", [])
        self.log.log_ai_success(count)

        if count > 0:
            yolo_count = sum(1 for s in shapes if s.get("score", 0) > 0)
            self.log.log_operation(f"YOLO+AI标注完成: 识别{count}个标注")
            messagebox.showinfo("标注完成", f"YOLO+AI识别完成！\n共识别 {count} 个标注")
        else:
            self.log.log_operation("YOLO+AI标注完成: 未识别到有效标注")
            messagebox.showwarning("标注为空", "未识别到有效的标注数据")


def main() -> None:
    app = SmartDrawingAnnotator()
    app.protocol("WM_DELETE_WINDOW", lambda: (
        app.log.log_operation("应用关闭"),
        app.destroy()
    ))
    app.mainloop()


if __name__ == "__main__":
    main()
