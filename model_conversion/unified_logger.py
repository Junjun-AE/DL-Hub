#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志模块 (Unified Logger)

合并 log_formatter.py 和 logging_config.py 的功能，提供：
1. 美观、对齐的控制台输出
2. 标准 Python logging 集成
3. 彩色输出支持
4. 统一的 API 接口
5. 多种输出格式（树形、表格、进度条等）

使用方法:
    from unified_logger import Logger, console
    
    # 初始化（在 main.py 中调用一次）
    Logger.init(verbose=False, log_file="output/conversion.log")
    
    # 获取 logger 实例
    log = Logger.get("ModelConverter")
    
    # 使用各种输出方法
    log.info("开始转换")
    log.stage(1, "模型加载")
    log.tree([("模型", "resnet50"), ("精度", "fp16")])
    log.success("转换完成")
    
    # 使用全局控制台输出
    console.header("模型转换流水线")
    console.table(headers, rows)

Author: Model Converter Team
Version: 3.0.0
"""

import atexit
import logging
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================================
# 颜色和样式
# ============================================================================

class Colors:
    """ANSI 颜色代码"""
    
    # 检测是否支持颜色
    ENABLED = (
        hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        and os.environ.get('NO_COLOR') is None
        and os.environ.get('TERM') != 'dumb'
    )
    
    # Windows 特殊处理
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            ENABLED = False
    
    # 基础颜色
    RESET = '\033[0m' if ENABLED else ''
    BOLD = '\033[1m' if ENABLED else ''
    DIM = '\033[2m' if ENABLED else ''
    ITALIC = '\033[3m' if ENABLED else ''
    UNDERLINE = '\033[4m' if ENABLED else ''
    
    # 前景色
    BLACK = '\033[30m' if ENABLED else ''
    RED = '\033[31m' if ENABLED else ''
    GREEN = '\033[32m' if ENABLED else ''
    YELLOW = '\033[33m' if ENABLED else ''
    BLUE = '\033[34m' if ENABLED else ''
    MAGENTA = '\033[35m' if ENABLED else ''
    CYAN = '\033[36m' if ENABLED else ''
    WHITE = '\033[37m' if ENABLED else ''
    
    # 亮色
    BRIGHT_RED = '\033[91m' if ENABLED else ''
    BRIGHT_GREEN = '\033[92m' if ENABLED else ''
    BRIGHT_YELLOW = '\033[93m' if ENABLED else ''
    BRIGHT_BLUE = '\033[94m' if ENABLED else ''
    BRIGHT_MAGENTA = '\033[95m' if ENABLED else ''
    BRIGHT_CYAN = '\033[96m' if ENABLED else ''
    BRIGHT_WHITE = '\033[97m' if ENABLED else ''
    
    # 背景色
    BG_RED = '\033[41m' if ENABLED else ''
    BG_GREEN = '\033[42m' if ENABLED else ''
    BG_YELLOW = '\033[43m' if ENABLED else ''
    BG_BLUE = '\033[44m' if ENABLED else ''
    
    @classmethod
    def disable(cls):
        """禁用颜色"""
        for attr in dir(cls):
            if attr.isupper() and attr != 'ENABLED':
                setattr(cls, attr, '')
        cls.ENABLED = False
    
    @classmethod
    def enable(cls):
        """启用颜色（重新加载模块）"""
        cls.ENABLED = True


# ============================================================================
# 图标定义
# ============================================================================

class Icons:
    """状态图标"""
    
    # 状态
    SUCCESS = "✓"      # 更简洁
    ERROR = "✗"
    WARNING = "!"
    INFO = "•"
    
    # 带颜色的状态
    @staticmethod
    def success(text: str = "") -> str:
        icon = f"{Colors.BRIGHT_GREEN}✓{Colors.RESET}"
        return f"{icon} {text}" if text else icon
    
    @staticmethod
    def error(text: str = "") -> str:
        icon = f"{Colors.BRIGHT_RED}✗{Colors.RESET}"
        return f"{icon} {text}" if text else icon
    
    @staticmethod
    def warning(text: str = "") -> str:
        icon = f"{Colors.BRIGHT_YELLOW}!{Colors.RESET}"
        return f"{icon} {text}" if text else icon
    
    @staticmethod
    def info(text: str = "") -> str:
        icon = f"{Colors.BRIGHT_BLUE}•{Colors.RESET}"
        return f"{icon} {text}" if text else icon
    
    # Stage 图标
    STAGE_ICONS = {
        1: "①",
        2: "②",
        3: "③",
        4: "④",
        5: "⑤",
        6: "⑥",
        7: "⑦",
        8: "⑧",
    }
    
    # 其他图标
    ARROW = "→"
    BULLET = "•"
    CHECK = "✓"
    CROSS = "✗"
    STAR = "★"
    CLOCK = "⏱"


# ============================================================================
# 布局常量
# ============================================================================

class Layout:
    """布局常量"""
    
    # 宽度
    LINE_WIDTH = 70
    LABEL_WIDTH = 20      # 标签对齐宽度
    VALUE_WIDTH = 40      # 值对齐宽度
    INDENT = 4            # 默认缩进
    
    # 树形结构
    TREE_BRANCH = "├──"
    TREE_LAST = "└──"
    TREE_VERTICAL = "│  "
    TREE_SPACE = "   "
    
    # 表格
    TABLE_H = "─"
    TABLE_V = "│"
    TABLE_TL = "┌"
    TABLE_TR = "┐"
    TABLE_BL = "└"
    TABLE_BR = "┘"
    TABLE_TM = "┬"
    TABLE_BM = "┴"
    TABLE_ML = "├"
    TABLE_MR = "┤"
    TABLE_MM = "┼"


# ============================================================================
# 时间追踪
# ============================================================================

class Timer:
    """简单的时间追踪器"""
    
    _start_time: float = None
    _stages: Dict[str, float] = {}
    _current_stage: Optional[str] = None
    _stage_start: float = None
    
    @classmethod
    def start(cls):
        """开始全局计时"""
        cls._start_time = time.time()
    
    @classmethod
    def start_stage(cls, name: str):
        """开始阶段计时"""
        cls._current_stage = name
        cls._stage_start = time.time()
    
    @classmethod
    def end_stage(cls) -> float:
        """结束阶段计时"""
        if cls._current_stage and cls._stage_start:
            elapsed = time.time() - cls._stage_start
            cls._stages[cls._current_stage] = elapsed
            cls._current_stage = None
            return elapsed
        return 0.0
    
    @classmethod
    def elapsed(cls) -> float:
        """总耗时"""
        if cls._start_time:
            return time.time() - cls._start_time
        return 0.0
    
    @classmethod
    def format_time(cls, seconds: float) -> str:
        """格式化时间"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {secs:.1f}s"
    
    @staticmethod
    def now() -> str:
        """当前时间 HH:MM:SS"""
        return datetime.now().strftime("%H:%M:%S")
    
    @staticmethod
    def now_full() -> str:
        """完整时间"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# 文本工具
# ============================================================================

class TextUtils:
    """文本处理工具"""
    
    @staticmethod
    def display_width(s: str) -> int:
        """计算显示宽度（考虑中文）"""
        width = 0
        for char in s:
            # 去除 ANSI 转义序列
            if char == '\033':
                continue
            # 中文字符占2个宽度
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            elif '\uff00' <= char <= '\uffef':  # 全角字符
                width += 2
            else:
                width += 1
        return width
    
    @staticmethod
    def strip_ansi(s: str) -> str:
        """移除 ANSI 转义序列"""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', s)
    
    @staticmethod
    def pad_right(s: str, width: int, char: str = ' ') -> str:
        """右填充（考虑中文）"""
        current_width = TextUtils.display_width(TextUtils.strip_ansi(s))
        padding = width - current_width
        return s + char * max(0, padding)
    
    @staticmethod
    def pad_left(s: str, width: int, char: str = ' ') -> str:
        """左填充（考虑中文）"""
        current_width = TextUtils.display_width(TextUtils.strip_ansi(s))
        padding = width - current_width
        return char * max(0, padding) + s
    
    @staticmethod
    def center(s: str, width: int, char: str = ' ') -> str:
        """居中对齐"""
        current_width = TextUtils.display_width(TextUtils.strip_ansi(s))
        padding = width - current_width
        left = padding // 2
        right = padding - left
        return char * max(0, left) + s + char * max(0, right)
    
    @staticmethod
    def truncate(s: str, width: int, suffix: str = "...") -> str:
        """截断字符串"""
        if TextUtils.display_width(s) <= width:
            return s
        
        result = ""
        current_width = 0
        for char in s:
            char_width = 2 if '\u4e00' <= char <= '\u9fff' else 1
            if current_width + char_width + len(suffix) > width:
                break
            result += char
            current_width += char_width
        
        return result + suffix


# ============================================================================
# 控制台输出类
# ============================================================================

class Console:
    """
    控制台输出类
    
    提供美观的格式化输出，包括：
    - 标题和分隔线
    - 树形结构
    - 表格
    - 进度条
    - 状态消息
    """
    
    def __init__(self):
        self._indent_level = 0
    
    # ==================== 基础输出 ====================
    
    def _print(self, *args, **kwargs):
        """内部打印方法"""
        print(*args, **kwargs)
    
    def line(self, char: str = "═", width: int = Layout.LINE_WIDTH):
        """打印分隔线"""
        self._print(char * width)
    
    def blank(self, count: int = 1):
        """打印空行"""
        for _ in range(count):
            self._print()
    
    # ==================== 标题和 Stage ====================
    
    def header(self, title: str, subtitle: str = ""):
        """
        打印主标题
        
        ══════════════════════════════════════════════════════════════════════
        模型转换流水线
        ══════════════════════════════════════════════════════════════════════
        """
        self.line("═")
        self._print(f"{Colors.BOLD}{title}{Colors.RESET}")
        if subtitle:
            self._print(f"{Colors.DIM}{subtitle}{Colors.RESET}")
        self.line("═")
    
    def stage(self, num: int, title: str, subtitle: str = ""):
        """
        打印 Stage 标题
        
        ══════════════════════════════════════════════════════════════════════
        [Stage 5] 量化转换与验证
        ══════════════════════════════════════════════════════════════════════
        """
        Timer.start_stage(f"Stage {num}")
        
        self.blank()
        self.line("═")
        
        stage_text = f"Stage {num}"
        colored_stage = f"{Colors.BRIGHT_CYAN}[{stage_text}]{Colors.RESET}"
        self._print(f"{colored_stage} {Colors.BOLD}{title}{Colors.RESET}")
        
        if subtitle:
            self._print(f"         {Colors.DIM}{subtitle}{Colors.RESET}")
        
        self.line("═")
    
    def section(self, title: str):
        """
        打印小节标题
        
        ─── 模型信息 ────────────────────────────────────────────────────────
        """
        title_width = TextUtils.display_width(title)
        line_width = Layout.LINE_WIDTH - title_width - 6
        
        self._print(
            f"{Colors.DIM}───{Colors.RESET} "
            f"{Colors.BOLD}{title}{Colors.RESET} "
            f"{Colors.DIM}{'─' * line_width}{Colors.RESET}"
        )
    
    # ==================== 日志条目 ====================
    
    def log(self, level: str, message: str, details: List[Tuple[str, str]] = None):
        """
        打印日志条目
        
        [10:02:31] INFO    加载模型配置
                   ├── 模型: wide_resnet50_2
                   └── 精度: fp16
        """
        timestamp = Timer.now()
        
        # 级别颜色
        level_colors = {
            'INFO': Colors.BRIGHT_GREEN,
            'WARNING': Colors.BRIGHT_YELLOW,
            'ERROR': Colors.BRIGHT_RED,
            'DEBUG': Colors.DIM,
        }
        level_color = level_colors.get(level.upper(), '')
        level_str = TextUtils.pad_right(level.upper(), 7)
        
        # 主消息
        self._print(
            f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
            f"{level_color}{level_str}{Colors.RESET} "
            f"{message}"
        )
        
        # 详情树
        if details:
            self.tree(details, indent=11)
    
    def info(self, message: str, **details):
        """INFO 日志"""
        detail_list = [(k, str(v)) for k, v in details.items()] if details else None
        self.log("INFO", message, detail_list)
    
    def warning(self, message: str, **details):
        """WARNING 日志"""
        detail_list = [(k, str(v)) for k, v in details.items()] if details else None
        self.log("WARNING", message, detail_list)
    
    def error(self, message: str, **details):
        """ERROR 日志"""
        detail_list = [(k, str(v)) for k, v in details.items()] if details else None
        self.log("ERROR", message, detail_list)
    
    def debug(self, message: str):
        """DEBUG 日志"""
        self.log("DEBUG", message)
    
    # ==================== 树形结构 ====================
    
    def tree(self, items: List[Tuple[str, Any]], indent: int = 0):
        """
        打印树形结构
        
        ├── 模型类型: classification
        ├── 类别数: 1000
        └── 状态: ✓ 成功
        """
        if not items:
            return
        
        padding = " " * indent
        
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            branch = Layout.TREE_LAST if is_last else Layout.TREE_BRANCH
            
            if isinstance(item, tuple) and len(item) >= 2:
                key, value = item[0], item[1]
                # 对齐键
                key_padded = TextUtils.pad_right(str(key), 18)
                self._print(
                    f"{padding}{Colors.DIM}{branch}{Colors.RESET} "
                    f"{Colors.CYAN}{key_padded}{Colors.RESET}: {value}"
                )
            else:
                self._print(f"{padding}{Colors.DIM}{branch}{Colors.RESET} {item}")
    
    def tree_nested(self, data: Dict[str, Any], indent: int = 0, _is_last: bool = True):
        """打印嵌套树形结构"""
        items = list(data.items())
        
        for i, (key, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            
            if isinstance(value, dict):
                self.tree([(key, "")], indent)
                self.tree_nested(value, indent + 4, is_last)
            else:
                self.tree([(key, value)], indent)
    
    # ==================== 表格 ====================
    
    def table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        indent: int = 0,
        align: List[str] = None,
    ):
        """
        打印表格
        
        ┌──────────────┬─────────┬──────────┐
        │ 组件类型     │ 数量    │ 占比     │
        ├──────────────┼─────────┼──────────┤
        │ Conv2d       │ 53      │ 27.75%   │
        │ BatchNorm2d  │ 53      │ 27.75%   │
        └──────────────┴─────────┴──────────┘
        """
        if not headers or not rows:
            return
        
        padding = " " * indent
        
        # 计算列宽
        col_widths = []
        for i, header in enumerate(headers):
            max_width = TextUtils.display_width(str(header))
            for row in rows:
                if i < len(row):
                    cell_width = TextUtils.display_width(str(row[i]))
                    max_width = max(max_width, cell_width)
            col_widths.append(max_width + 2)
        
        # 默认对齐方式
        if not align:
            align = ['l'] * len(headers)
        
        L = Layout
        
        # 顶部边框
        top = L.TABLE_TL
        for i, w in enumerate(col_widths):
            top += L.TABLE_H * w
            top += L.TABLE_TM if i < len(col_widths) - 1 else L.TABLE_TR
        self._print(f"{padding}{Colors.DIM}{top}{Colors.RESET}")
        
        # 表头
        header_line = L.TABLE_V
        for i, (header, w) in enumerate(zip(headers, col_widths)):
            cell = TextUtils.pad_right(f" {header}", w)
            header_line += f"{Colors.BOLD}{cell}{Colors.RESET}{Colors.DIM}{L.TABLE_V}{Colors.RESET}"
        self._print(f"{padding}{Colors.DIM}{L.TABLE_V}{Colors.RESET}{header_line[1:]}")
        
        # 分隔线
        sep = L.TABLE_ML
        for i, w in enumerate(col_widths):
            sep += L.TABLE_H * w
            sep += L.TABLE_MM if i < len(col_widths) - 1 else L.TABLE_MR
        self._print(f"{padding}{Colors.DIM}{sep}{Colors.RESET}")
        
        # 数据行
        for row in rows:
            row_line = ""
            for i, w in enumerate(col_widths):
                cell = str(row[i]) if i < len(row) else ""
                if align[i] == 'r':
                    cell = TextUtils.pad_left(cell + " ", w)
                else:
                    cell = TextUtils.pad_right(f" {cell}", w)
                row_line += f"{cell}{Colors.DIM}{L.TABLE_V}{Colors.RESET}"
            self._print(f"{padding}{Colors.DIM}{L.TABLE_V}{Colors.RESET}{row_line}")
        
        # 底部边框
        bottom = L.TABLE_BL
        for i, w in enumerate(col_widths):
            bottom += L.TABLE_H * w
            bottom += L.TABLE_BM if i < len(col_widths) - 1 else L.TABLE_BR
        self._print(f"{padding}{Colors.DIM}{bottom}{Colors.RESET}")
    
    def kv_table(self, items: List[Tuple[str, Any]], indent: int = 0, title: str = None):
        """
        打印键值对表格
        
        ┌────────────────────┬────────────────────────────┐
        │ 属性               │ 值                         │
        ├────────────────────┼────────────────────────────┤
        │ 模型大小           │ 262.67 MB                  │
        └────────────────────┴────────────────────────────┘
        """
        if not items:
            return
        
        headers = [title or "属性", "值"]
        rows = [[k, str(v)] for k, v in items]
        self.table(headers, rows, indent)
    
    # ==================== 状态消息 ====================
    
    def success(self, message: str, **details):
        """成功消息"""
        self._print(f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} {message}")
        if details:
            self.tree([(k, v) for k, v in details.items()], indent=2)
    
    def fail(self, message: str, **details):
        """失败消息"""
        self._print(f"{Colors.BRIGHT_RED}✗{Colors.RESET} {message}")
        if details:
            self.tree([(k, v) for k, v in details.items()], indent=2)
    
    def warn(self, message: str):
        """警告消息"""
        self._print(f"{Colors.BRIGHT_YELLOW}!{Colors.RESET} {message}")
    
    # ==================== 进度和结果 ====================
    
    def progress(self, current: int, total: int, message: str = "", width: int = 30):
        """
        打印进度条
        
        [██████████████████──────────────] 60% 正在处理...
        """
        percent = (current / total) * 100 if total > 0 else 0
        filled = int(width * current / total) if total > 0 else 0
        empty = width - filled
        
        bar = f"{Colors.BRIGHT_GREEN}{'█' * filled}{Colors.RESET}{Colors.DIM}{'─' * empty}{Colors.RESET}"
        
        self._print(f"\r[{bar}] {percent:5.1f}% {message}", end='', flush=True)
        
        if current >= total:
            self._print()  # 换行
    
    def result(
        self,
        success: bool,
        title: str,
        items: List[Tuple[str, str]] = None,
        elapsed: float = None,
    ):
        """
        打印结果块
        
        ✓ 转换完成
          ├── 输出文件: output/model.onnx
          ├── 模型大小: 262.67 MB
          └── 耗时: 12.34s
        """
        icon = Icons.success() if success else Icons.error()
        color = Colors.BRIGHT_GREEN if success else Colors.BRIGHT_RED
        
        self._print(f"{icon} {color}{title}{Colors.RESET}")
        
        # 合并 items 和 elapsed
        all_items = list(items) if items else []
        if elapsed is not None:
            elapsed_str = Timer.format_time(elapsed)
            all_items.append(("耗时", elapsed_str))
        
        if all_items:
            self.tree(all_items, indent=2)
    
    # ==================== 汇总报告 ====================
    
    def summary_header(self, title: str = "执行摘要"):
        """打印汇总标题"""
        self.blank()
        self.line("═")
        self._print(f"{Colors.BOLD}{title}{Colors.RESET}")
        self.line("═")
    
    def summary_section(self, title: str, items: List[Tuple[str, str]] = None, icon: str = None):
        """
        打印汇总小节
        
        ▸ 转换配置
          ├── 目标后端: tensorrt
          └── 精度模式: int8
        
        Args:
            title: 小节标题
            items: 键值对列表（可选）
            icon: 图标（可选，如果第二个参数是字符串则视为图标）
        """
        # 兼容旧的调用方式：summary_section("标题", "💡")
        if isinstance(items, str):
            icon = items
            items = None
        
        if icon:
            self._print(f"\n{Colors.BRIGHT_CYAN}▸{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET} {icon}")
        else:
            self._print(f"\n{Colors.BRIGHT_CYAN}▸{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET}")
        
        if items:
            self.tree(items, indent=2)
    
    def final_status(self, success: bool, elapsed: float = None, end_time: str = None):
        """
        打印最终状态
        
        ══════════════════════════════════════════════════════════════════════
        ✓ 执行完成 | 总耗时: 1m 23.4s | 结束时间: 2025-12-24 10:59:00
        ══════════════════════════════════════════════════════════════════════
        
        Args:
            success: 是否成功
            elapsed: 耗时（秒）
            end_time: 结束时间字符串
        """
        self.blank()
        self.line("═")
        
        if success:
            status = f"{Colors.BRIGHT_GREEN}✓ 执行完成{Colors.RESET}"
        else:
            status = f"{Colors.BRIGHT_RED}✗ 执行失败{Colors.RESET}"
        
        parts = [status]
        
        if elapsed is not None:
            elapsed_str = Timer.format_time(elapsed)
            parts.append(f"总耗时: {elapsed_str}")
        
        if end_time is not None:
            parts.append(f"结束时间: {end_time}")
        
        self._print(f" {Colors.DIM}|{Colors.RESET} ".join(parts))
        
        self.line("═")


# ============================================================================
# Logger 类（集成 Python logging）
# ============================================================================

class UnifiedFormatter(logging.Formatter):
    """统一日志格式化器"""
    
    FORMATS = {
        logging.DEBUG: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.DIM}DEBUG  {Colors.RESET} %(name)s - %(message)s",
        logging.INFO: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.BRIGHT_GREEN}INFO   {Colors.RESET} %(name)s - %(message)s",
        logging.WARNING: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.BRIGHT_YELLOW}WARNING{Colors.RESET} %(name)s - %(message)s",
        logging.ERROR: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.BRIGHT_RED}ERROR  {Colors.RESET} %(name)s - %(message)s",
        logging.CRITICAL: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.BG_RED}{Colors.WHITE}CRITICAL{Colors.RESET} %(name)s - %(message)s",
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """无颜色格式化器（用于文件）"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class Logger:
    """
    统一 Logger 类
    
    集成 Python logging 和美化控制台输出。
    """
    
    _initialized = False
    _console = Console()
    _loggers: Dict[str, 'LoggerWrapper'] = {}
    _lock = threading.Lock()
    
    @classmethod
    def init(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        verbose: bool = False,
        quiet: bool = False,
        no_color: bool = False,
    ):
        """
        初始化日志系统
        
        Args:
            level: 日志级别
            log_file: 日志文件路径
            verbose: 详细模式
            quiet: 静默模式
            no_color: 禁用颜色
        """
        with cls._lock:
            if cls._initialized:
                return
            
            # 禁用颜色
            if no_color:
                Colors.disable()
            
            # 确定级别
            if verbose:
                level = "DEBUG"
            elif quiet:
                level = "WARNING"
            
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
            }
            log_level = level_map.get(level.upper(), logging.INFO)
            
            # 配置根 logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # 清除现有处理器
            root_logger.handlers.clear()
            
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(UnifiedFormatter())
            root_logger.addHandler(console_handler)
            
            # 文件处理器
            if log_file:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(PlainFormatter())
                root_logger.addHandler(file_handler)
            
            # 降低第三方库日志级别
            for lib in ['PIL', 'matplotlib', 'urllib3', 'onnx', 'onnxruntime', 'tensorrt']:
                logging.getLogger(lib).setLevel(logging.WARNING)
            
            # 启动计时器
            Timer.start()
            
            cls._initialized = True
    
    @classmethod
    def reset(cls):
        """
        重置日志系统状态
        
        用于在模块被重新加载时重新初始化日志系统。
        这对于多次调用转换流程非常重要。
        """
        with cls._lock:
            cls._initialized = False
            cls._loggers.clear()
    
    @classmethod
    def get(cls, name: str) -> 'LoggerWrapper':
        """获取命名 logger"""
        if name not in cls._loggers:
            cls._loggers[name] = LoggerWrapper(name)
        return cls._loggers[name]
    
    @classmethod
    def console(cls) -> Console:
        """获取控制台输出器"""
        return cls._console


class LoggerWrapper:
    """
    Logger 包装器
    
    同时支持 Python logging 和美化输出。
    """
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)
        self._console = Logger._console
    
    # Python logging 方法
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)
    
    # 美化输出方法
    def stage(self, num: int, title: str, subtitle: str = ""):
        """打印 Stage 标题"""
        self._console.stage(num, title, subtitle)
    
    def section(self, title: str):
        """打印小节标题"""
        self._console.section(title)
    
    def tree(self, items: List[Tuple[str, Any]], indent: int = 0):
        """打印树形结构"""
        self._console.tree(items, indent)
    
    def table(self, headers: List[str], rows: List[List[Any]], **kwargs):
        """打印表格"""
        self._console.table(headers, rows, **kwargs)
    
    def success(self, msg: str, **details):
        """成功消息"""
        self._console.success(msg, **details)
    
    def fail(self, msg: str, **details):
        """失败消息"""
        self._console.fail(msg, **details)
    
    def result(self, success: bool, title: str, **kwargs):
        """打印结果"""
        self._console.result(success, title, **kwargs)


# ============================================================================
# 全局实例
# ============================================================================

# 全局控制台
console = Console()

# 便捷函数
def get_logger(name: str) -> LoggerWrapper:
    """获取 logger（兼容旧 API）"""
    return Logger.get(name)


def setup_logging(**kwargs):
    """设置日志（兼容旧 API）"""
    Logger.init(**kwargs)


# ============================================================================
# 测试
# ============================================================================

def _test():
    """测试统一日志模块"""
    
    # 初始化
    Logger.init(verbose=True)
    log = Logger.get("TestModule")
    
    # 测试标题
    console.header("模型转换流水线", "PyTorch → ONNX → TensorRT")
    
    # 测试 Stage
    console.stage(1, "模型加载与分析")
    
    log.info("加载模型配置")
    console.tree([
        ("模型名称", "wide_resnet50_2"),
        ("模型来源", "torchvision"),
        ("预训练权重", "ImageNet-1K"),
    ], indent=4)
    
    # 测试表格
    console.blank()
    console.section("模型结构分析")
    console.table(
        ["组件类型", "数量", "参数量", "占比"],
        [
            ["Conv2d", "53", "68.8M", "99.2%"],
            ["BatchNorm2d", "53", "0.1M", "0.1%"],
            ["Linear", "1", "0.5M", "0.7%"],
        ],
        indent=4,
    )
    
    # 测试 Stage 2
    console.stage(5, "量化转换", "INT8 精度 → TensorRT 后端")
    
    log.info("开始 INT8 量化")
    console.tree([
        ("校准方法", "entropy"),
        ("校准样本", "300"),
        ("目标后端", "tensorrt"),
    ], indent=4)
    
    # 模拟进度
    console.blank()
    for i in range(0, 101, 20):
        console.progress(i, 100, "校准中...")
        time.sleep(0.1)
    
    # 测试结果
    console.blank()
    console.result(
        success=True,
        title="INT8 转换完成",
        items=[
            ("输出文件", "output/model_int8.engine"),
            ("模型大小", "65.5 MB"),
            ("压缩率", "4.0x"),
        ],
        elapsed=12.34,
    )
    
    # 测试警告
    console.blank()
    log.warning("验证精度低于阈值")
    console.tree([
        ("Cosine Similarity", "0.9821"),
        ("阈值", "0.99"),
        ("建议", "增加校准样本或使用 MIXED 精度"),
    ], indent=4)
    
    # 测试汇总
    console.summary_header("执行摘要")
    
    console.summary_section("转换配置", [
        ("目标后端", "tensorrt"),
        ("精度模式", "int8"),
        ("校准样本", "300"),
    ])
    
    console.summary_section("输出文件", [
        ("模型文件", "output/model_int8.engine (65.5 MB)"),
        ("校准缓存", "output/model_calib.cache"),
        ("转换报告", "output/conversion_report.json"),
    ])
    
    console.summary_section("性能指标", [
        ("推理延迟", "2.3 ms"),
        ("吞吐量", "434 FPS"),
        ("内存占用", "1.2 GB"),
    ])
    
    console.final_status(success=True, elapsed=Timer.elapsed())


if __name__ == "__main__":
    _test()
