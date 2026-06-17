"""
SegFormer 语义分割训练工具 - Gradio Web界面
与目标检测工具保持一致的UI风格

功能：环境检查、模型训练、模型转换、批量推理

已集成 DL-Hub 支持：
- 支持 --task-dir 参数指定任务目录
- 支持 --port 参数指定端口
- 自动保存/加载UI参数
"""
import os
import sys
import threading
import time
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import gradio as gr
import numpy as np

import logging
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

def init_dlhub_adapter():
    """初始化 DL-Hub 适配器"""
    try:
        dlhub_path = Path(__file__).parent.parent / 'dlhub_project' / 'dlhub'
        if dlhub_path.exists():
            sys.path.insert(0, str(dlhub_path.parent))
        from dlhub.app_adapters.base_adapter import get_adapter
        adapter = get_adapter(default_port=7861)
        print(f'[DL-Hub] 适配器已初始化，模式: {('DL-Hub' if adapter.is_dlhub_mode else '独立')}')
        return adapter
    except ImportError:
        print('[DL-Hub] 适配器未找到，以独立模式运行')
        return None
    except Exception as e:
        print(f'[DL-Hub] 初始化失败: {e}，以独立模式运行')
        return None

def init_dlhub_params():
    """初始化 DL-Hub 参数管理器（使用单例模式）"""
    try:
        from dlhub_params import get_dlhub_params
        params = get_dlhub_params()
        return params
    except ImportError:
        print('[DL-Hub] 参数管理器未找到，参数不会持久化')
        return None
    except Exception as e:
        print(f'[DL-Hub] 参数管理器初始化失败: {e}')
        return None
dlhub_adapter = init_dlhub_adapter()
dlhub_params = init_dlhub_params()

def get_output_dir(default: str='./output') -> Path:
    """获取输出目录，优先使用 DL-Hub 任务目录"""
    if dlhub_params and dlhub_params.is_dlhub_mode:
        return dlhub_params.get_output_dir()
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode:
        return dlhub_adapter.get_output_dir()
    return Path(default)

def get_saved_param(key: str, default=None):
    """获取保存的参数值"""
    if dlhub_params:
        return dlhub_params.get(key, default)
    return default

def save_all_params(params_dict: dict) -> bool:
    """保存所有参数"""
    if dlhub_params:
        return dlhub_params.save(params_dict)
    return False
CUSTOM_CSS = "\n/* 整体容器 */\n.gradio-container {\n    max-width: 100% !important;\n    padding-left: 20px !important;\n    padding-right: 20px !important;\n    margin: 0 !important;\n}\n\n.main {\n    max-width: 100% !important;\n}\n\n.contain {\n    max-width: 100% !important;\n}\n\n/* 行布局优化 */\n.row {\n    gap: 20px !important;\n}\n\n/* 组件间距 */\n.group {\n    padding: 15px !important;\n}\n\n/* 日志框 */\n.log-box textarea {\n    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;\n    font-size: 13px !important;\n    background: #1a1a2e !important;\n    color: #eee !important;\n    border-radius: 8px !important;\n    line-height: 1.5 !important;\n}\n\n/* 按钮样式 */\n.primary-btn {\n    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;\n    border: none !important;\n    font-weight: bold !important;\n}\n\n.stop-btn {\n    background: linear-gradient(135deg, #f43f5e 0%, #ec4899 100%) !important;\n    border: none !important;\n}\n\n/* 预览图像容器 */\n.preview-container img {\n    max-height: 300px !important;\n    object-fit: contain !important;\n    cursor: zoom-in !important;\n}\n\n/* Plotly图表 */\n.plotly-graph-div {\n    border-radius: 8px !important;\n    overflow: hidden !important;\n}\n\n/* 滚动条 */\n::-webkit-scrollbar {\n    width: 8px;\n    height: 8px;\n}\n\n::-webkit-scrollbar-track {\n    background: #f1f1f1;\n    border-radius: 4px;\n}\n\n::-webkit-scrollbar-thumb {\n    background: #c1c1c1;\n    border-radius: 4px;\n}\n\n::-webkit-scrollbar-thumb:hover {\n    background: #a1a1a1;\n}\n\n/* 图像查看器按钮悬停效果 */\n.iv-btn:hover {\n    background: rgba(255, 255, 255, 0.3) !important;\n}\n"
IMAGE_VIEWER_HTML = '\n<div id="imageViewerOverlay" style="\n    display: none;\n    position: fixed;\n    top: 0;\n    left: 0;\n    width: 100%;\n    height: 100%;\n    background: rgba(0, 0, 0, 0.95);\n    z-index: 10000;\n    overflow: hidden;\n">\n    <img id="imageViewerImg" style="\n        position: absolute;\n        top: 50%;\n        left: 50%;\n        transform-origin: center center;\n        cursor: grab;\n        user-select: none;\n    " draggable="false">\n    \n    <div id="imageViewerInfo" style="\n        position: fixed;\n        top: 20px;\n        left: 20px;\n        background: rgba(0,0,0,0.7);\n        color: white;\n        padding: 10px 20px;\n        border-radius: 20px;\n        font-size: 14px;\n        z-index: 10001;\n    ">100%</div>\n    \n    <button id="imageViewerClose" style="\n        position: fixed;\n        top: 20px;\n        right: 20px;\n        width: 50px;\n        height: 50px;\n        background: rgba(255,255,255,0.2);\n        border: none;\n        border-radius: 50%;\n        color: white;\n        font-size: 28px;\n        cursor: pointer;\n        z-index: 10001;\n    ">×</button>\n    \n    <div style="\n        position: fixed;\n        bottom: 80px;\n        left: 50%;\n        transform: translateX(-50%);\n        color: rgba(255,255,255,0.7);\n        font-size: 13px;\n        z-index: 10001;\n    ">🖱️ 滚轮缩放 | 拖拽移动 | ESC关闭</div>\n    \n    <div style="\n        position: fixed;\n        bottom: 25px;\n        left: 50%;\n        transform: translateX(-50%);\n        display: flex;\n        gap: 10px;\n        z-index: 10001;\n    ">\n        <button class="iv-btn" id="ivBtnZoomIn" style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-size: 14px;">🔍+ 放大</button>\n        <button class="iv-btn" id="ivBtnZoomOut" style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-size: 14px;">🔍- 缩小</button>\n        <button class="iv-btn" id="ivBtnReset" style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-size: 14px;">↺ 重置</button>\n        <button class="iv-btn" id="ivBtnFit" style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-size: 14px;">⛶ 适应</button>\n    </div>\n</div>\n'
IMAGE_VIEWER_HEAD = '\n<script>\n(function() {\n    // 防止重复初始化\n    if (window.imageViewerReady) return;\n    window.imageViewerReady = true;\n    \n    console.log(\'[ImageViewer] 脚本加载中...\');\n    \n    // ===== 状态变量 =====\n    var scale = 1;\n    var translateX = 0;\n    var translateY = 0;\n    var isDragging = false;\n    var dragStartX = 0;\n    var dragStartY = 0;\n    var lastTranslateX = 0;\n    var lastTranslateY = 0;\n    var imgNaturalWidth = 0;\n    var imgNaturalHeight = 0;\n    var isOpen = false;\n    \n    // ===== 获取元素 =====\n    function getElements() {\n        return {\n            overlay: document.getElementById(\'imageViewerOverlay\'),\n            img: document.getElementById(\'imageViewerImg\'),\n            info: document.getElementById(\'imageViewerInfo\'),\n            closeBtn: document.getElementById(\'imageViewerClose\'),\n            zoomInBtn: document.getElementById(\'ivBtnZoomIn\'),\n            zoomOutBtn: document.getElementById(\'ivBtnZoomOut\'),\n            resetBtn: document.getElementById(\'ivBtnReset\'),\n            fitBtn: document.getElementById(\'ivBtnFit\')\n        };\n    }\n    \n    // ===== 更新变换 =====\n    function updateTransform() {\n        var els = getElements();\n        if (!els.img) return;\n        \n        els.img.style.transform = \'translate(-50%, -50%) translate(\' + translateX + \'px, \' + translateY + \'px) scale(\' + scale + \')\';\n        \n        if (els.info) {\n            els.info.textContent = Math.round(scale * 100) + \'%\';\n        }\n    }\n    \n    // ===== 打开查看器 =====\n    window.openImageViewer = function(src) {\n        var els = getElements();\n        if (!els.overlay || !els.img) {\n            console.error(\'[ImageViewer] 元素未找到\');\n            return;\n        }\n        \n        console.log(\'[ImageViewer] 打开\');\n        \n        scale = 1;\n        translateX = 0;\n        translateY = 0;\n        isDragging = false;\n        isOpen = true;\n        \n        els.img.onload = function() {\n            imgNaturalWidth = els.img.naturalWidth;\n            imgNaturalHeight = els.img.naturalHeight;\n            \n            var maxW = window.innerWidth * 0.9;\n            var maxH = window.innerHeight * 0.85;\n            var scaleX = maxW / imgNaturalWidth;\n            var scaleY = maxH / imgNaturalHeight;\n            scale = Math.min(scaleX, scaleY, 1);\n            \n            updateTransform();\n        };\n        \n        els.img.src = src;\n        els.overlay.style.display = \'block\';\n        document.body.style.overflow = \'hidden\';\n    };\n    \n    // ===== 关闭查看器 =====\n    window.closeImageViewer = function() {\n        var els = getElements();\n        if (els.overlay) {\n            els.overlay.style.display = \'none\';\n            document.body.style.overflow = \'\';\n            isOpen = false;\n            console.log(\'[ImageViewer] 关闭\');\n        }\n    };\n    \n    // ===== 初始化事件 =====\n    function initEvents() {\n        var els = getElements();\n        if (!els.overlay) {\n            console.log(\'[ImageViewer] 等待元素...\');\n            setTimeout(initEvents, 300);\n            return;\n        }\n        \n        console.log(\'[ImageViewer] 绑定事件...\');\n        \n        // 滚轮缩放\n        els.overlay.addEventListener(\'wheel\', function(e) {\n            if (!isOpen) return;\n            e.preventDefault();\n            e.stopPropagation();\n            \n            var rect = els.img.getBoundingClientRect();\n            var imgCenterX = rect.left + rect.width / 2;\n            var imgCenterY = rect.top + rect.height / 2;\n            var mouseOffsetX = e.clientX - imgCenterX;\n            var mouseOffsetY = e.clientY - imgCenterY;\n            \n            var factor = e.deltaY < 0 ? 1.1 : 0.9;\n            var newScale = Math.max(0.1, Math.min(20, scale * factor));\n            var ratio = newScale / scale;\n            \n            translateX = translateX - mouseOffsetX * (ratio - 1);\n            translateY = translateY - mouseOffsetY * (ratio - 1);\n            scale = newScale;\n            \n            updateTransform();\n        }, { passive: false });\n        \n        // 拖拽开始\n        els.img.addEventListener(\'mousedown\', function(e) {\n            if (e.button !== 0) return;\n            e.preventDefault();\n            \n            isDragging = true;\n            dragStartX = e.clientX;\n            dragStartY = e.clientY;\n            lastTranslateX = translateX;\n            lastTranslateY = translateY;\n            els.img.style.cursor = \'grabbing\';\n        });\n        \n        // 拖拽移动\n        document.addEventListener(\'mousemove\', function(e) {\n            if (!isDragging || !isOpen) return;\n            e.preventDefault();\n            \n            translateX = lastTranslateX + (e.clientX - dragStartX);\n            translateY = lastTranslateY + (e.clientY - dragStartY);\n            updateTransform();\n        });\n        \n        // 拖拽结束\n        document.addEventListener(\'mouseup\', function() {\n            if (isDragging) {\n                isDragging = false;\n                var els2 = getElements();\n                if (els2.img) els2.img.style.cursor = \'grab\';\n            }\n        });\n        \n        // ESC关闭\n        document.addEventListener(\'keydown\', function(e) {\n            if (e.key === \'Escape\' && isOpen) {\n                closeImageViewer();\n            }\n        });\n        \n        // 点击背景关闭\n        els.overlay.addEventListener(\'click\', function(e) {\n            if (e.target === els.overlay) {\n                closeImageViewer();\n            }\n        });\n        \n        // 关闭按钮\n        if (els.closeBtn) {\n            els.closeBtn.onclick = function(e) {\n                e.stopPropagation();\n                closeImageViewer();\n            };\n        }\n        \n        // 控制按钮\n        if (els.zoomInBtn) {\n            els.zoomInBtn.onclick = function(e) {\n                e.stopPropagation();\n                scale = Math.min(scale * 1.25, 20);\n                updateTransform();\n            };\n        }\n        \n        if (els.zoomOutBtn) {\n            els.zoomOutBtn.onclick = function(e) {\n                e.stopPropagation();\n                scale = Math.max(scale * 0.8, 0.1);\n                updateTransform();\n            };\n        }\n        \n        if (els.resetBtn) {\n            els.resetBtn.onclick = function(e) {\n                e.stopPropagation();\n                scale = 1;\n                translateX = 0;\n                translateY = 0;\n                updateTransform();\n            };\n        }\n        \n        if (els.fitBtn) {\n            els.fitBtn.onclick = function(e) {\n                e.stopPropagation();\n                var maxW = window.innerWidth * 0.9;\n                var maxH = window.innerHeight * 0.85;\n                scale = Math.min(maxW / imgNaturalWidth, maxH / imgNaturalHeight);\n                translateX = 0;\n                translateY = 0;\n                updateTransform();\n            };\n        }\n        \n        // 阻止图像点击冒泡\n        els.img.addEventListener(\'click\', function(e) {\n            e.stopPropagation();\n        });\n        \n        console.log(\'[ImageViewer] 事件绑定完成\');\n    }\n    \n    // ===== 为图像添加点击事件 =====\n    function setupGradioImages() {\n        var images = document.querySelectorAll(\'img\');\n        \n        for (var i = 0; i < images.length; i++) {\n            var img = images[i];\n            \n            if (img.id === \'imageViewerImg\') continue;\n            if (img.getAttribute(\'data-iv-bindend\')) continue;\n            if (!img.src) continue;\n            \n            // 检查是否是预览图像\n            var rect = img.getBoundingClientRect();\n            var isPreviewImage = (\n                img.closest && (\n                    img.closest(\'[data-testid="image"]\') ||\n                    img.closest(\'.image-container\')\n                )\n            ) || (rect.width > 100 && rect.height > 100);\n            \n            if (!isPreviewImage) continue;\n            \n            img.setAttribute(\'data-iv-bindend\', \'true\');\n            img.style.cursor = \'zoom-in\';\n            \n            (function(imgEl) {\n                imgEl.addEventListener(\'click\', function(e) {\n                    e.preventDefault();\n                    e.stopPropagation();\n                    if (this.src) {\n                        console.log(\'[ImageViewer] 点击图像\');\n                        openImageViewer(this.src);\n                    }\n                });\n            })(img);\n        }\n    }\n    \n    // ===== 启动 =====\n    function start() {\n        console.log(\'[ImageViewer] 初始化...\');\n        initEvents();\n        setupGradioImages();\n        \n        // 监听DOM变化\n        var observer = new MutationObserver(function() {\n            setupGradioImages();\n        });\n        observer.observe(document.body, { childList: true, subtree: true });\n        \n        // 定时检查\n        setInterval(setupGradioImages, 2000);\n        \n        console.log(\'[ImageViewer] 初始化完成\');\n    }\n    \n    // 页面加载后启动\n    if (document.readyState === \'loading\') {\n        document.addEventListener(\'DOMContentLoaded\', start);\n    } else {\n        start();\n    }\n})();\n</script>\n'

class TrainingState:
    """训练状态管理"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.is_training = False
        self.should_stop = False
        self.trainer = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_losses = []
        self.val_mIoU = []
        self.val_mDice = []
        self.per_class_IoU = {}
        self.logs = []
        self.best_mIoU = 0.0
        self.best_epoch = 0
        self.output_dir = ''
        self.start_time = None
        self.dataset_dir = ''
        self.class_names = []
        self.segmentation_preview = None
        self.batch_images_dir = ''
        self.batch_output_dir = ''

    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

class ConversionState:
    """转换状态管理"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.is_running = False
        self.logs = []
        self.output_path = ''
        self.start_time = None

    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
state = TrainingState()
conversion_state = ConversionState()
preview_seed = [42]
try:
    import yolo_seg_integration as yolo_seg

    def _parse_gpu_for_yoloseg(choice):
        from utils.env_validator import parse_gpu_choice
        return parse_gpu_choice(choice)
    yolo_seg.bind(state=state, conversion_state=conversion_state, dlhub_params=dlhub_params, get_output_dir=get_output_dir, parse_gpu_choice=_parse_gpu_for_yoloseg)
    _YOLO_SEG_AVAILABLE = True
    print('[YOLO-seg] 实例分割集成已加载')
except Exception as _e:
    yolo_seg = None
    _YOLO_SEG_AVAILABLE = False
    print(f'[YOLO-seg] 实例分割集成加载失败(将仅提供SegFormer): {_e}')

def check_environment() -> str:
    """检查运行环境"""
    try:
        from utils.env_validator import validate_environment
        success, message, info = validate_environment()
        return message
    except Exception as e:
        return f'❌ 环境验证出错: {str(e)}'

def get_gpu_options() -> list:
    """获取GPU选项列表"""
    try:
        from utils.env_validator import get_gpu_choices
        return get_gpu_choices()
    except Exception:
        return ['CPU']

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f'{seconds:.0f}秒'
    elif seconds < 3600:
        return f'{seconds / 60:.1f}分'
    else:
        return f'{seconds / 3600:.1f}时'

def get_trained_models_seg() -> List[str]:
    """获取训练好的分割模型列表"""
    models = []
    output_dir = get_output_dir()
    if output_dir.exists():
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir() and 'SegFormer' in model_dir.name:
                weights_dir = model_dir / 'weights'
                if weights_dir.exists():
                    for model_file in ['best_model.pth', 'last_model.pth']:
                        model_path = weights_dir / model_file
                        if model_path.exists():
                            models.append(str(model_path))
                work_dirs = model_dir / 'work_dirs'
                if work_dirs.exists():
                    for pth_file in work_dirs.glob('*.pth'):
                        models.append(str(pth_file))
    return sorted(models, reverse=True)

def get_output_base_dir() -> Path:
    """获取output基础目录"""
    if dlhub_params and dlhub_params.is_dlhub_mode:
        return dlhub_params.get_output_dir()
    return Path('./output')
_folder_path_map = {}
_model_path_map = {}

def scan_model_folders_seg() -> Dict:
    """扫描output目录下的训练文件夹，只显示文件夹名"""
    global _folder_path_map
    _folder_path_map = {}
    base_path = get_output_base_dir()
    if not base_path.exists():
        return gr.update(choices=[], value=None)
    folders = []
    for item in base_path.iterdir():
        if item.is_dir():
            has_models = any(item.rglob('*.pth')) or any(item.rglob('*.pt'))
            if has_models:
                folder_name = item.name
                _folder_path_map[folder_name] = str(item)
                folders.append(folder_name)
    folders = sorted(folders, reverse=True)
    if folders:
        return gr.update(choices=folders, value=folders[0])
    return gr.update(choices=[], value=None)

def scan_models_in_selected_folder_seg(folder_name: str) -> Dict:
    """扫描选定文件夹下的模型文件，只显示相对路径"""
    global _model_path_map
    _model_path_map = {}
    if not folder_name:
        return gr.update(choices=[], value=None)
    folder_path = _folder_path_map.get(folder_name)
    if not folder_path:
        return gr.update(choices=[], value=None)
    folder = Path(folder_path)
    if not folder.exists():
        return gr.update(choices=[], value=None)
    models = list(folder.rglob('*.pth')) + list(folder.rglob('*.pt'))
    model_items = []
    seen = set()
    for m in models:
        rel_path = str(m.relative_to(folder))
        if rel_path in seen:
            continue
        seen.add(rel_path)
        _model_path_map[rel_path] = str(m)
        model_items.append(rel_path)
    model_items = sorted(model_items, reverse=True)
    if model_items:
        return gr.update(choices=model_items, value=model_items[0])
    return gr.update(choices=[], value=None)

def get_full_model_path_seg(rel_path: str) -> str:
    """根据相对路径获取完整模型路径"""
    if not rel_path:
        return ''
    if rel_path in _model_path_map:
        return _model_path_map[rel_path]
    if os.path.isabs(rel_path) or os.path.exists(rel_path):
        return rel_path
    return rel_path

def scan_models_in_folder_seg(folder_path: str) -> Dict:
    """扫描文件夹中的分割模型 - 用于推理Tab"""
    global _model_path_map
    _model_path_map = {}
    if not folder_path or not folder_path.strip():
        base_path = get_output_base_dir()
        if not base_path.exists():
            return gr.update(choices=[], value=None)
        folder_path = str(base_path)
    else:
        folder_path = folder_path.strip()
    if os.path.isfile(folder_path):
        if folder_path.endswith('.pth') or folder_path.endswith('.pt'):
            file_name = os.path.basename(folder_path)
            _model_path_map[file_name] = folder_path
            return gr.update(choices=[file_name], value=file_name)
        return gr.update(choices=[], value=None)
    if not os.path.exists(folder_path):
        return gr.update(choices=[], value=None)
    folder = Path(folder_path)
    models = list(folder.rglob('*.pth')) + list(folder.rglob('*.pt'))
    model_items = []
    seen = set()
    for m in models:
        try:
            rel_path = str(m.relative_to(folder))
        except ValueError:
            rel_path = m.name
        if rel_path in seen:
            continue
        seen.add(rel_path)
        _model_path_map[rel_path] = str(m)
        model_items.append(rel_path)
    model_items = sorted(model_items, reverse=True)
    if model_items:
        return gr.update(choices=model_items, value=model_items[0])
    return gr.update(choices=[], value=None)

def update_backend_options(backend: str) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """更新后端选项的显示状态"""
    is_tensorrt = backend == 'tensorrt'
    return (gr.update(visible=is_tensorrt), gr.update(visible=is_tensorrt), gr.update(visible=True), gr.update(visible=True), gr.update(visible=is_tensorrt), gr.update(visible=True), gr.update(visible=True))

def update_calib_visibility(precision: str) -> Dict:
    """更新校准数据输入框的可见性"""
    needs_calib = precision in ['int8', 'mixed']
    return gr.update(visible=needs_calib)

def validate_calib_data_seg(images_dir: str) -> str:
    """验证分割模型校准数据"""
    if not images_dir or not images_dir.strip():
        return '⏳ 请输入校准图像文件夹路径'
    images_dir = images_dir.strip()
    if not os.path.exists(images_dir):
        return f'❌ 路径不存在: {images_dir}'
    if not os.path.isdir(images_dir):
        return '❌ 请输入文件夹路径，不是文件'
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    total_images = 0
    for ext in image_extensions:
        total_images += len(list(Path(images_dir).rglob(f'*{ext}')))
        total_images += len(list(Path(images_dir).rglob(f'*{ext.upper()}')))
    if total_images == 0:
        return '❌ 未找到图片文件'
    if total_images < 100:
        warning = '⚠️ 建议至少100张图片\n'
    else:
        warning = ''
    return f'✅ 验证通过\n{warning}📊 共 {total_images} 张图片'

def get_conversion_logs() -> str:
    """获取转换日志"""
    return '\n'.join(conversion_state.logs[-100:]) if conversion_state.logs else '暂无日志'

def get_conversion_status() -> Tuple[str, str]:
    """获取转换状态"""
    if conversion_state.is_running:
        elapsed = time.time() - conversion_state.start_time if conversion_state.start_time else 0
        status = f'🔄 转换中... ({format_time(elapsed)})'
    elif conversion_state.output_path:
        status = f'✅ 转换完成: {conversion_state.output_path}'
    else:
        status = '⏳ 等待开始...'
    logs = get_conversion_logs()
    return (status, logs)

def on_start_conversion_seg(model_path: str, target_backend: str, precision: str, device: str, workspace_gb: int, dynamic_batch: bool, min_batch: int, opt_batch: int, max_batch: int, output_dir: str, calib_images_dir: str='', end2end: bool=True) -> Tuple[str, str]:
    """开始模型转换 (自动区分 SegFormer / YOLO-seg)。"""
    global conversion_state, state
    if state.is_training:
        return ('⚠️ 训练正在进行中，请等待训练完成后再进行转换', '')
    if conversion_state.is_running:
        return ('⚠️ 转换正在进行中...', get_conversion_logs())
    if not model_path:
        return ('❌ 请选择要转换的模型', '')
    full_model_path = get_full_model_path_seg(model_path)
    if not os.path.exists(full_model_path):
        return (f'❌ 模型文件不存在: {full_model_path}', '')
    if _YOLO_SEG_AVAILABLE and yolo_seg is not None and yolo_seg.is_yolo_seg_model(full_model_path):
        if precision in ['int8', 'mixed']:
            if not calib_images_dir or not calib_images_dir.strip():
                return ('❌ INT8/Mixed精度需要提供校准图像文件夹路径', '')
            calib_images_dir = calib_images_dir.strip()
            if not os.path.exists(calib_images_dir):
                return (f'❌ 校准图像路径不存在: {calib_images_dir}', '')
        if dlhub_params:
            cur = dlhub_params.get_all()
            cur['conversion'] = {'target_backend': target_backend, 'precision': precision, 'device': device, 'workspace_gb': workspace_gb, 'dynamic_batch': dynamic_batch, 'min_batch': min_batch, 'opt_batch': opt_batch, 'max_batch': max_batch, 'output_dir': output_dir, 'end2end': end2end}
            dlhub_params.save(cur)
        return yolo_seg.start_conversion(model_path=full_model_path, target_backend=target_backend, precision=precision, device=device, workspace_gb=workspace_gb, dynamic_batch=dynamic_batch, min_batch=min_batch, opt_batch=opt_batch, max_batch=max_batch, output_dir_in=output_dir, end2end=end2end, calib_images_dir=calib_images_dir)
    if dlhub_params:
        current_params = dlhub_params.get_all()
        current_params['conversion'] = {'target_backend': target_backend, 'precision': precision, 'device': device, 'workspace_gb': workspace_gb, 'dynamic_batch': dynamic_batch, 'min_batch': min_batch, 'opt_batch': opt_batch, 'max_batch': max_batch, 'output_dir': output_dir}
        dlhub_params.save(current_params)
    if precision in ['int8', 'mixed']:
        if not calib_images_dir or not calib_images_dir.strip():
            return ('❌ INT8/Mixed精度需要提供校准图像文件夹路径', '')
        calib_images_dir = calib_images_dir.strip()
        if not os.path.exists(calib_images_dir):
            return (f'❌ 校准图像路径不存在: {calib_images_dir}', '')
        validation_result = validate_calib_data_seg(calib_images_dir)
        if not validation_result.startswith('✅'):
            return (f'❌ 校准数据集验证失败: {validation_result}', '')
    conversion_state.reset()
    conversion_state.is_running = True
    conversion_state.start_time = time.time()
    if not output_dir:
        output_dir = str(Path(full_model_path).parent.parent / 'converted')
    thread = threading.Thread(target=run_conversion_seg, args=(full_model_path, target_backend, precision, device, workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch, output_dir, calib_images_dir), daemon=True)
    thread.start()
    return ('🚀 转换已启动...', '')

def run_conversion_seg(model_path: str, target_backend: str, precision: str, device: str, workspace_gb: int, dynamic_batch: bool, min_batch: int, opt_batch: int, max_batch: int, output_dir: str, calib_images_dir: str=''):
    """
    后台转换线程 - 分割模型版 V2
    
    优化改进：
    1. 不同后端输出到不同子目录 (tensorrt/openvino/onnxruntime)
    2. 每次转换创建日期时间子目录
    3. 始终使用独立子进程导出ONNX，完全避免进程污染
    4. 移除ONNX复用逻辑，每次都重新导出确保参数一致
    """
    global conversion_state
    backend_dir_map = {'tensorrt': 'tensorrt', 'trt': 'tensorrt', 'openvino': 'openvino', 'ov': 'openvino', 'ort': 'onnxruntime', 'onnxruntime': 'onnxruntime'}
    backend_subdir = backend_dir_map.get(target_backend.lower(), target_backend.lower())
    date_subdir = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = output_dir
    output_dir = str(Path(base_output_dir) / backend_subdir / date_subdir)
    ctx = None
    conversion_tool_path = str(Path(__file__).parent.parent / 'model_conversion')
    try:
        conversion_state.add_log('🧹 清理资源...')
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception as e:
            conversion_state.add_log(f'⚠️ 初始清理警告: {e}')
        modules_to_remove = []
        for mod_name in list(sys.modules.keys()):
            if any((target in mod_name for target in ['model_importer', 'model_analyzer', 'model_optimizer', 'model_exporter', 'model_converter', 'config_generator', 'converter_tensorrt', 'converter_openvino', 'converter_ort', 'conversion_validator', 'unified_logger', 'cuda_utils', 'symbolic', 'constants', 'exceptions', 'config_templates', 'onnxsim', 'onnxoptimizer', 'onnxconverter_common'])):
                if not mod_name.startswith('_'):
                    modules_to_remove.append(mod_name)
        openvino_modules = [mod for mod in sys.modules.keys() if 'openvino' in mod.lower()]
        modules_to_remove.extend(openvino_modules)
        for mod_name in modules_to_remove:
            try:
                del sys.modules[mod_name]
            except Exception:
                logger.debug('GUI exception ignored @ L1057', exc_info=True)
        if conversion_tool_path not in sys.path:
            sys.path.insert(0, conversion_tool_path)
        from main import PipelineContext, run_stage1_import, run_stage2_analyze, run_stage3_optimize, run_stage4_export, run_stage5_convert, run_stage8_generate_config
        conversion_state.add_log('━' * 50)
        conversion_state.add_log('🚀 开始语义分割模型转换')
        conversion_state.add_log('━' * 50)
        conversion_state.add_log(f'📂 模型: {model_path}')
        conversion_state.add_log(f'🎯 目标: {target_backend} ({precision})')
        conversion_state.add_log(f'📁 输出: {output_dir}')
        if precision in ['int8', 'mixed']:
            if calib_images_dir:
                conversion_state.add_log(f'📊 校准数据: {calib_images_dir}')
            else:
                conversion_state.add_log('❌ INT8/Mixed精度需要校准数据，但未提供')
                return
        ctx = PipelineContext()
        ctx.target_backend = target_backend
        ctx.precision = precision
        ctx.device = 'cuda' if 'GPU' in device else 'cpu'
        conversion_state.add_log('\n📥 Stage 1: 导入模型...')
        if not run_stage1_import(ctx, model_path, 'seg', device=ctx.device):
            conversion_state.add_log('❌ 模型导入失败')
            return
        conversion_state.add_log(f'   ✅ 模型: {ctx.model_name}')
        conversion_state.add_log(f'   ✅ 输入形状: {ctx.input_shape}')
        conversion_state.add_log('\n🔍 Stage 2: 分析模型...')
        if not run_stage2_analyze(ctx, [target_backend]):
            conversion_state.add_log('❌ 模型分析失败')
            return
        conversion_state.add_log('   ✅ 分析完成')
        conversion_state.add_log('\n⚡ Stage 3: 优化模型...')
        if not run_stage3_optimize(ctx):
            conversion_state.add_log('⚠️ 优化跳过，使用原始模型')
        else:
            conversion_state.add_log('   ✅ 优化完成')
        conversion_state.add_log('\n📦 Stage 4: 导出 ONNX...')
        os.makedirs(output_dir, exist_ok=True)
        onnx_filename = f'{ctx.model_name}.onnx'
        onnx_path = str(Path(output_dir) / onnx_filename)
        onnx_dynamic_batch = dynamic_batch
        if onnx_dynamic_batch:
            if target_backend == 'tensorrt':
                conversion_state.add_log(f'   启用动态Batch: min={min_batch}, opt={opt_batch}, max={max_batch}')
            else:
                conversion_state.add_log(f'   启用动态Batch: min={min_batch}, max={max_batch}')
        conversion_state.add_log(f'   ONNX输出: {os.path.basename(onnx_path)}')
        export_success = False
        if ctx.framework in ['mmseg', 'mmsegmentation']:
            conversion_state.add_log('   使用独立子进程导出 SegFormer 模型...')
            try:
                import subprocess
                import tempfile
                input_shape = list(ctx.input_shape) if ctx.input_shape else [1, 3, 512, 512]
                export_config = {'model_path': ctx.model_path, 'output_path': onnx_path, 'opset': 17, 'dynamic_batch': onnx_dynamic_batch, 'input_shape': input_shape, 'model_name': ctx.model_name}
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(export_config, f)
                    config_path = f.name
                worker_script = os.path.join(os.path.dirname(__file__), 'segformer_export_worker.py')
                if not os.path.exists(worker_script):
                    worker_script = os.path.join(conversion_tool_path, 'segformer_export_worker.py')
                if os.path.exists(worker_script):
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    result = subprocess.run([sys.executable, worker_script, config_path], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300, env=env)
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                print(f'  {line}', flush=True)
                    result_path = config_path + '.result'
                    if os.path.exists(result_path):
                        with open(result_path, 'r', encoding='utf-8') as f:
                            export_result = json.load(f)
                        if export_result.get('success'):
                            ctx.onnx_path = export_result.get('output_path', onnx_path)
                            try:
                                import onnx
                                ctx.onnx_model = onnx.load(ctx.onnx_path)
                            except Exception:
                                logger.debug('GUI exception ignored @ L1195', exc_info=True)
                            export_success = True
                            conversion_state.add_log(f'   ✅ 子进程导出成功')
                        else:
                            conversion_state.add_log(f'   ⚠️ 子进程导出失败: {export_result.get('message', '')}')
                    try:
                        os.unlink(config_path)
                        if os.path.exists(result_path):
                            os.unlink(result_path)
                    except Exception:
                        logger.debug('GUI exception ignored @ L1206', exc_info=True)
                else:
                    conversion_state.add_log('   ⚠️ 未找到子进程脚本，尝试进程内导出')
            except subprocess.TimeoutExpired:
                conversion_state.add_log('   ⚠️ 子进程导出超时，尝试进程内导出')
            except Exception as e:
                conversion_state.add_log(f'   ⚠️ 子进程导出异常: {e}')
        if not export_success:
            conversion_state.add_log('   使用进程内导出...')
            if not run_stage4_export(ctx, output_path=onnx_path, opset=17, enable_dynamic_batch=onnx_dynamic_batch, enable_dynamic_hw=False, enable_simplify=False):
                conversion_state.add_log('❌ ONNX导出失败')
                return
            conversion_state.add_log(f'   ✅ ONNX: {ctx.onnx_path}')
        else:
            conversion_state.add_log(f'   ✅ ONNX: {ctx.onnx_path}')
        conversion_state.add_log(f'\n🔧 Stage 5: 转换到 {target_backend}...')
        backend_kwargs = {}
        if target_backend == 'tensorrt':
            backend_kwargs = {'trt_workspace_gb': workspace_gb, 'trt_dynamic_batch_enabled': dynamic_batch, 'trt_min_batch': min_batch, 'trt_opt_batch': opt_batch, 'trt_max_batch': max_batch}
            conversion_state.add_log(f'   TensorRT 配置:')
            conversion_state.add_log(f'     - Workspace: {workspace_gb} GB')
            conversion_state.add_log(f'     - Dynamic Batch: {dynamic_batch}')
            if dynamic_batch:
                conversion_state.add_log(f'     - Batch Range: [{min_batch}, {opt_batch}, {max_batch}]')
        elif target_backend == 'openvino':
            backend_kwargs = {'ov_dynamic_batch_enabled': dynamic_batch, 'ov_min_batch': min_batch, 'ov_max_batch': max_batch}
            if dynamic_batch:
                conversion_state.add_log(f'   OpenVINO 配置:')
                conversion_state.add_log(f'     - Dynamic Batch: {dynamic_batch}')
                conversion_state.add_log(f'     - Batch Range: [{min_batch}, {max_batch}]')
        elif target_backend in ['ort', 'onnxruntime']:
            backend_kwargs = {'ort_dynamic_batch_enabled': dynamic_batch, 'ort_min_batch': min_batch, 'ort_max_batch': max_batch}
            if dynamic_batch:
                conversion_state.add_log(f'   ONNX Runtime 配置:')
                conversion_state.add_log(f'     - Dynamic Batch: {dynamic_batch}')
        if not run_stage5_convert(ctx, output_dir=output_dir, target_backend=target_backend, precision=precision, calib_data_path=calib_images_dir, enable_validation=False, **backend_kwargs):
            conversion_state.add_log('❌ 转换失败')
            return
        conversion_state.add_log(f'   ✅ 转换完成')
        conversion_state.add_log('\n📝 Stage 8: 生成配置...')
        run_stage8_generate_config(ctx, output_dir)
        conversion_state.add_log('   ✅ 配置已生成')
        conversion_state.add_log('\n📦 Stage 9: 打包 .dlhub...')
        try:
            conversion_tool_path_pack = str(Path(__file__).parent.parent / 'model_conversion')
            if conversion_tool_path_pack not in sys.path:
                sys.path.insert(0, conversion_tool_path_pack)
            from dlhub_packager import DLHubPackager
            packager = DLHubPackager()
            dlhub_path = packager.pack(output_dir=output_dir, task_type='seg', backend=target_backend, precision=precision)
            if dlhub_path:
                conversion_state.add_log(f'   ✅ 已打包: {os.path.basename(dlhub_path)}')
            else:
                conversion_state.add_log('   ⚠️ 打包跳过（无模型文件或打包失败）')
        except Exception as pack_err:
            conversion_state.add_log(f'   ⚠️ 打包失败（非致命）: {pack_err}')
        conversion_state.output_path = output_dir
        elapsed = time.time() - conversion_state.start_time
        conversion_state.add_log('\n' + '━' * 50)
        conversion_state.add_log(f'✅ 转换完成！用时 {elapsed:.1f} 秒')
        conversion_state.add_log(f'📁 输出目录: {output_dir}')
        conversion_state.add_log('━' * 50)
    except Exception as e:
        import traceback
        conversion_state.add_log(f'\n❌ 错误: {str(e)}')
        conversion_state.add_log(traceback.format_exc())
    finally:
        conversion_state.is_running = False
        try:
            if ctx is not None:
                ctx.model = None
                ctx.optimized_model = None
                ctx.onnx_model = None
        except Exception:
            logger.debug('GUI exception ignored @ L1336', exc_info=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug('GUI exception ignored @ L1343', exc_info=True)
        try:
            import gc
            gc.collect()
        except Exception:
            logger.debug('GUI exception ignored @ L1349', exc_info=True)
        try:
            if dlhub_params:
                dlhub_params.save_logs(conversion_state.logs, 'conversion', auto_save=True)
        except Exception:
            logger.debug('GUI exception ignored @ L1356', exc_info=True)

def validate_data_paths(images_dir: str, jsons_dir: str) -> str:
    """验证数据集路径"""
    print(f'🔍 validate_data_paths 被调用')
    print(f"   images_dir: '{images_dir}'")
    print(f"   jsons_dir: '{jsons_dir}'")
    if not images_dir or not images_dir.strip():
        return '⏳ 请输入图像文件夹路径'
    if not jsons_dir or not jsons_dir.strip():
        return '⏳ 请输入JSON文件夹路径'
    try:
        from utils.data_validator import validate_labelme_dataset
        result = validate_labelme_dataset(images_dir.strip(), jsons_dir.strip())
        return result.message
    except Exception as e:
        return f'❌ 验证出错: {str(e)}'

def update_model_info(scale: str) -> str:
    """更新模型信息显示"""
    if not scale:
        return '请选择模型规模'
    try:
        from config.model_registry import get_model_display_info
        return get_model_display_info(scale)
    except Exception as e:
        return f'获取模型信息失败: {str(e)}'

def check_pretrained_weights(scale: str) -> str:
    """检查预训练权重状态"""
    if not scale:
        return ''
    try:
        from models.model_factory import check_pretrained_status
        return check_pretrained_status(scale)
    except Exception as e:
        return f'检查失败: {str(e)}'

def get_data_preview(images_dir: str, jsons_dir: str):
    """获取数据集预览图（高分辨率版本）- 完全按照目标检测的实现"""
    if not images_dir or not jsons_dir:
        return None
    try:
        from data.visualizer import preview_dataset
        preview = preview_dataset(images_dir.strip(), jsons_dir.strip(), num_samples=4, seed=preview_seed[0], high_resolution=True)
        return preview
    except Exception as e:
        print(f'预览失败: {e}')
        import traceback
        traceback.print_exc()
        return None

def refresh_data_preview(images_dir: str, jsons_dir: str):
    """换一批预览图（高分辨率版本）"""
    import time
    preview_seed[0] = int(time.time() * 1000) % 100000
    return get_data_preview(images_dir, jsons_dir)

def get_logs() -> str:
    """获取日志"""
    return '\n'.join(state.logs[-100:]) if state.logs else '暂无日志'

def create_loss_plot():
    """创建Loss曲线图"""
    import plotly.graph_objects as go
    fig = go.Figure()
    if state.train_losses and len(state.train_losses) > 0:
        fig.add_trace(go.Scatter(x=list(range(1, len(state.train_losses) + 1)), y=state.train_losses, mode='lines+markers', name='Train Loss', line=dict(color='#10b981', width=2), marker=dict(size=4)))
    fig.update_layout(title=dict(text=f'训练Loss曲线 ({len(state.train_losses)} epochs)', font=dict(size=14)), xaxis_title='Epoch', yaxis_title='Loss', height=280, margin=dict(l=50, r=20, t=40, b=40), showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    return fig

def create_miou_plot():
    """创建mIoU曲线图"""
    import plotly.graph_objects as go
    fig = go.Figure()
    if state.val_mIoU and len(state.val_mIoU) > 0:
        fig.add_trace(go.Scatter(x=list(range(1, len(state.val_mIoU) + 1)), y=state.val_mIoU, mode='lines+markers', name='mIoU', line=dict(color='#3b82f6', width=2), marker=dict(size=4)))
    if state.val_mDice and len(state.val_mDice) > 0:
        fig.add_trace(go.Scatter(x=list(range(1, len(state.val_mDice) + 1)), y=state.val_mDice, mode='lines+markers', name='mDice', line=dict(color='#f59e0b', width=2), marker=dict(size=4)))
    if state.best_epoch > 0 and state.val_mIoU and (len(state.val_mIoU) >= state.best_epoch):
        fig.add_trace(go.Scatter(x=[state.best_epoch], y=[state.best_mIoU], mode='markers', name=f'Best: {state.best_mIoU:.2f}%', marker=dict(color='#ef4444', size=12, symbol='star')))
    fig.update_layout(title=dict(text=f'验证指标曲线 ({len(state.val_mIoU)} epochs)', font=dict(size=14)), xaxis_title='Epoch', yaxis_title='Score (%)', height=280, margin=dict(l=50, r=20, t=40, b=40), showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', range=[0, 105])
    return fig

def get_progress_html() -> str:
    """生成进度条HTML"""
    if not state.is_training:
        if state.best_mIoU > 0:
            return f'\n            <div style="padding: 15px; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 10px; text-align: center;">\n                <div style="font-size: 18px; font-weight: bold; color: #047857;">✅ 训练完成</div>\n                <div style="margin-top: 8px; color: #065f46;">\n                    🏆 最佳 mIoU: {state.best_mIoU:.4f} (Epoch {state.best_epoch})\n                </div>\n            </div>\n            '
        return '\n        <div style="padding: 15px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 10px; text-align: center;">\n            <div style="font-size: 16px; color: #0369a1;">⏳ 等待开始训练...</div>\n        </div>\n        '
    progress = state.current_epoch / state.total_epochs * 100 if state.total_epochs > 0 else 0
    elapsed = time.time() - state.start_time if state.start_time else 0
    elapsed_str = f'{int(elapsed // 60)}分{int(elapsed % 60)}秒'
    if state.current_epoch > 0:
        eta = elapsed / state.current_epoch * (state.total_epochs - state.current_epoch)
        eta_str = f'{int(eta // 60)}分{int(eta % 60)}秒'
    else:
        eta_str = '计算中...'
    current_loss = state.train_losses[-1] if state.train_losses else 0
    current_miou = state.val_mIoU[-1] if state.val_mIoU else 0
    return f'\n    <div style="padding: 15px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 10px;">\n        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">\n            <span style="font-weight: bold; color: #065f46;">🏋️ 训练进行中</span>\n            <span style="color: #047857;">Epoch {state.current_epoch}/{state.total_epochs}</span>\n        </div>\n        <div style="background: #d1d5db; border-radius: 10px; height: 20px; overflow: hidden;">\n            <div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); height: 100%; width: {progress:.1f}%; \n                        transition: width 0.3s; display: flex; align-items: center; justify-content: center;">\n                <span style="color: white; font-size: 12px; font-weight: bold;">{progress:.1f}%</span>\n            </div>\n        </div>\n        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 13px; color: #374151;">\n            <span>⏱️ 已用: {elapsed_str}</span>\n            <span>📊 Loss: {current_loss:.4f}</span>\n            <span>🎯 mIoU: {current_miou:.4f}</span>\n            <span>⏳ 剩余: {eta_str}</span>\n        </div>\n        <div style="margin-top: 8px; font-size: 13px; color: #059669;">\n            🏆 最佳 mIoU: {state.best_mIoU:.4f} (Epoch {state.best_epoch})\n        </div>\n    </div>\n    '

def get_per_class_iou_html() -> str:
    """生成逐类别IoU显示HTML"""
    if not state.per_class_IoU:
        return "<div style='padding: 10px; color: #6b7280;'>暂无逐类别IoU数据</div>"
    html = "<div style='padding: 10px;'><h4 style='margin-bottom: 10px; color: #374151;'>📊 逐类别 IoU</h4>"
    from config.model_registry import get_color_for_class
    for cls_id, iou in sorted(state.per_class_IoU.items()):
        color = get_color_for_class(cls_id)
        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        cls_name = state.class_names[cls_id] if cls_id < len(state.class_names) else str(cls_id)
        html += f'\n        <div style="margin-bottom: 8px;">\n            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">\n                <span style="display: flex; align-items: center;">\n                    <span style="width: 12px; height: 12px; background: {color_hex}; border-radius: 2px; margin-right: 6px;"></span>\n                    类别 {cls_name}\n                </span>\n                <span style="font-weight: bold;">{iou:.4f}</span>\n            </div>\n            <div style="background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">\n                <div style="background: {color_hex}; height: 100%; width: {iou * 100:.1f}%;"></div>\n            </div>\n        </div>\n        '
    html += '</div>'
    return html

def run_training(images_dir: str, jsons_dir: str, num_classes: int, model_scale: str, epochs: int, batch_size: int, img_size: int, learning_rate: float, val_split: float, include_negative: bool, patience: int, save_period: int, use_fp16: bool, use_class_weight: bool, loss_ce_weight: float, loss_dice_weight: float, flip_horizontal: bool, flip_vertical: bool, random_rotate: bool, rotate_degree: float, color_jitter: bool, device_choice: str):
    """后台训练线程"""
    from config.model_registry import get_model_config
    from data.converter import LabelMeToMMSegConverter
    from engine.trainer import SegFormerTrainer, TrainingCallback
    from utils.env_validator import parse_gpu_choice
    try:
        state.add_log('=' * 50)
        state.add_log('🚀 开始语义分割训练')
        state.add_log('=' * 50)
        if dlhub_params:
            current_params = dlhub_params.get_all()
            current_params['data'] = {'images_dir': images_dir, 'jsons_dir': jsons_dir, 'image_size': img_size, 'val_split': val_split, 'include_negative': include_negative, 'num_classes': num_classes}
            current_params['model'] = {'scale': model_scale}
            current_params['training'] = {'batch_size': batch_size, 'epochs': epochs, 'learning_rate': learning_rate, 'patience': patience, 'save_period': save_period, 'use_fp16': use_fp16, 'use_class_weight': use_class_weight, 'loss_ce_weight': loss_ce_weight, 'loss_dice_weight': loss_dice_weight}
            current_params['augmentation'] = {'flip_horizontal': flip_horizontal, 'flip_vertical': flip_vertical, 'random_rotate': random_rotate, 'rotate_degree': rotate_degree, 'color_jitter': color_jitter}
            current_params['device'] = device_choice
            dlhub_params.save(current_params)
        device_str, gpu_ids = parse_gpu_choice(device_choice)
        state.add_log(f'🖥️ 设备: {device_choice} -> {device_str}')
        model_config = get_model_config(model_scale)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'SegFormer_{model_config['backbone']}_{timestamp}'
        output_base = get_output_dir()
        output_dir = output_base / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        state.output_dir = str(output_dir)
        state.add_log(f'📁 输出目录: {output_dir}')
        if dlhub_params:
            dlhub_params.clear_history(auto_save=False)
            dlhub_params.clear_logs('training', auto_save=False)
            dlhub_params.save_history({'total_epochs': epochs, 'current_epoch': 0, 'output_dir': str(output_dir), 'completed': False}, auto_save=True)
        state.add_log('📦 开始数据转换...')
        dataset_dir = output_dir / 'dataset'
        converter = LabelMeToMMSegConverter(images_dir=images_dir.strip(), jsons_dir=jsons_dir.strip(), output_dir=str(dataset_dir), val_split=val_split, num_classes=int(num_classes), include_negative_samples=include_negative)
        result = converter.convert()
        if not result.success:
            state.add_log(f'❌ 数据转换失败: {result.message}')
            state.is_training = False
            return
        state.add_log(f'✅ 数据转换成功:')
        state.add_log(f'   训练集: {result.train_images} 张')
        state.add_log(f'   验证集: {result.val_images} 张')
        state.add_log(f'   类别数: {result.num_classes}')
        state.add_log(f'   负样本: {result.negative_samples} 张')
        state.dataset_dir = str(dataset_dir)
        state.class_names = result.class_names
        state.total_epochs = epochs

        def on_epoch_end(epoch: int, metrics: Dict):
            state.current_epoch = epoch
            if 'train_loss' in metrics:
                state.train_losses.append(metrics['train_loss'])
            if 'mIoU' in metrics:
                state.val_mIoU.append(metrics['mIoU'])
            if 'mDice' in metrics:
                state.val_mDice.append(metrics['mDice'])
            if 'per_class_IoU' in metrics:
                state.per_class_IoU = metrics['per_class_IoU']
            if 'best_mIoU' in metrics:
                state.best_mIoU = metrics['best_mIoU']
            if 'best_epoch' in metrics:
                state.best_epoch = metrics['best_epoch']
            if dlhub_params:
                epoch_data = {'current_epoch': epoch, 'best_mIoU': state.best_mIoU, 'best_epoch': state.best_epoch}
                if 'train_loss' in metrics:
                    epoch_data['train_loss'] = metrics['train_loss']
                if 'mIoU' in metrics:
                    epoch_data['mIoU'] = metrics['mIoU']
                if 'mDice' in metrics:
                    epoch_data['mDice'] = metrics['mDice']
                dlhub_params.update_history_epoch(epoch_data, auto_save=epoch % 5 == 0)

        def on_log(message: str):
            state.add_log(message)
            if dlhub_params:
                dlhub_params.append_log(message, 'training', auto_save=False)

        def should_stop():
            return state.should_stop
        callback = TrainingCallback(on_epoch_end=on_epoch_end, on_log=on_log, should_stop=should_stop)
        state.add_log('🔧 创建训练器...')
        trainer = SegFormerTrainer(model_scale=model_scale, dataset_dir=str(dataset_dir), output_dir=str(output_dir), num_classes=int(num_classes), epochs=int(epochs), batch_size=int(batch_size), img_size=int(img_size), learning_rate=float(learning_rate), patience=int(patience), device=device_str, use_fp16=use_fp16, use_class_weight=use_class_weight, loss_ce_weight=float(loss_ce_weight), loss_dice_weight=float(loss_dice_weight), flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, random_rotate=random_rotate, rotate_degree=float(rotate_degree), color_jitter=color_jitter, save_period=int(save_period), callback=callback, class_names=result.class_names)
        state.trainer = trainer
        trainer.train()
        if dlhub_params:
            dlhub_params.mark_training_complete(best_metric=state.best_mIoU, best_epoch=state.best_epoch)
    except Exception as e:
        import traceback
        state.add_log(f'❌ 训练出错: {str(e)}')
        state.add_log(traceback.format_exc())
    finally:
        state.is_training = False
        state.trainer = None
        if dlhub_params:
            dlhub_params.save_logs(state.logs, 'training', auto_save=True)

def on_start_training(images_dir, jsons_dir, num_classes, model_scale, epochs, batch_size, img_size, learning_rate, val_split, include_negative, patience, save_period, use_fp16, use_class_weight, loss_ce_weight, loss_dice_weight, flip_horizontal, flip_vertical, random_rotate, rotate_degree, color_jitter, device_choice):
    """开始训练按钮回调"""
    try:
        from engine.trainer import check_mmseg_installation
        check_mmseg_installation()
    except ImportError as e:
        return (f'❌ 环境检查失败:\n{str(e)}', gr.update(), gr.update())
    if not images_dir or not images_dir.strip():
        return ('❌ 请输入图像文件夹路径', gr.update(), gr.update())
    if not jsons_dir or not jsons_dir.strip():
        return ('❌ 请输入JSON文件夹路径', gr.update(), gr.update())
    if not model_scale:
        return ('❌ 请选择模型规模', gr.update(), gr.update())
    if num_classes <= 0:
        return ('❌ 请设置正确的类别数量', gr.update(), gr.update())
    if state.is_training:
        return ('⚠️ 训练已在进行中', gr.update(), gr.update())
    state.reset()
    state.is_training = True
    state.start_time = time.time()
    state.total_epochs = epochs
    training_thread = threading.Thread(target=run_training, args=(images_dir, jsons_dir, num_classes, model_scale, epochs, batch_size, img_size, learning_rate, val_split, include_negative, patience, save_period, use_fp16, use_class_weight, loss_ce_weight, loss_dice_weight, flip_horizontal, flip_vertical, random_rotate, rotate_degree, color_jitter, device_choice), daemon=True)
    training_thread.start()
    return ('✅ 训练已启动（使用MMSegmentation）', gr.update(interactive=False), gr.update(interactive=True))

def yoloseg_update_scales(family: str):
    """系列变更时刷新规模下拉。"""
    try:
        from config.yolo_seg_registry import get_model_scales
        scales = get_model_scales(family)
        return gr.update(choices=scales, value=scales[0] if scales else None)
    except Exception:
        return gr.update()

def yoloseg_update_model_info(family: str, scale: str):
    """刷新 YOLO-seg 模型信息。"""
    try:
        from config.yolo_seg_registry import get_model_display_info
        return get_model_display_info(family, scale)
    except Exception as e:
        return f'未找到模型配置: {e}'

def on_start_training_yoloseg(images_dir, jsons_dir, num_classes, family, model_scale, epochs, batch_size, img_size, learning_rate, val_split, include_negative, patience, save_period, optimizer, weight_decay, momentum, warmup_epochs, cos_lr, mosaic, mixup, fliplr, freeze_backbone, device_choice):
    """YOLO-seg 训练启动 (方案 X: 独立于 SegFormer 的 on_start_training)。"""
    if not _YOLO_SEG_AVAILABLE or yolo_seg is None:
        return ('❌ YOLO-seg 集成不可用', gr.update(), gr.update())
    return yolo_seg.start_training(images_dir=images_dir, jsons_dir=jsons_dir, num_classes=num_classes, family=family, model_scale=model_scale, epochs=epochs, batch_size=batch_size, img_size=img_size, learning_rate=learning_rate, val_split=val_split, include_negative=include_negative, patience=patience, save_period=save_period, optimizer=optimizer, weight_decay=weight_decay, momentum=momentum, warmup_epochs=warmup_epochs, cos_lr=cos_lr, mosaic=mosaic, mixup=mixup, fliplr=fliplr, freeze_backbone=freeze_backbone, device_choice=device_choice)

def on_stop_training():
    """停止训练按钮回调"""
    if not state.is_training:
        return '⚠️ 当前没有正在进行的训练'
    state.should_stop = True
    state.add_log('⏹️ 收到停止训练请求...')
    return '⏹️ 正在停止训练...'

def on_open_output():
    """打开输出目录"""
    if not state.output_dir:
        return '⚠️ 没有可用的输出目录'
    import subprocess
    import platform
    try:
        if platform.system() == 'Windows':
            os.startfile(state.output_dir)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', state.output_dir])
        else:
            subprocess.run(['xdg-open', state.output_dir])
        return f'📂 已打开: {state.output_dir}'
    except Exception as e:
        return f'❌ 无法打开目录: {e}\n路径: {state.output_dir}'

def get_training_status():
    """获取训练状态（定时刷新）"""
    progress_html = get_progress_html()
    loss_plot = create_loss_plot()
    miou_plot = create_miou_plot()
    logs = get_logs()
    per_class_html = get_per_class_iou_html()
    preview = state.segmentation_preview
    return (progress_html, loss_plot, miou_plot, logs, per_class_html, preview)

def on_test_inference(conf_threshold: float=0.5):
    """测试推理（带置信度过滤）"""
    if not state.output_dir:
        return (None, '⚠️ 请先完成训练')
    if _YOLO_SEG_AVAILABLE and yolo_seg is not None and yolo_seg.is_yoloseg_output_dir(state.output_dir):
        return yolo_seg.test_inference_preview(conf_threshold=float(conf_threshold))
    weights_dir = Path(state.output_dir) / 'weights'
    best_model = weights_dir / 'best_model.pth'
    if not best_model.exists():
        work_dir = Path(state.output_dir) / 'work_dirs'
        for ckpt in work_dir.glob('best_*.pth'):
            best_model = ckpt
            break
        for ckpt in work_dir.glob('iter_*.pth'):
            best_model = ckpt
    if not best_model.exists():
        return (None, '⚠️ 未找到训练好的模型')
    if not state.dataset_dir:
        return (None, '⚠️ 未找到数据集目录')
    try:
        from data.dataset import get_val_images
        from data.visualizer import visualize_mask_overlay, create_preview_grid
        from inference.batch_inference import SegFormerInference
        from PIL import Image
        val_images = get_val_images(state.dataset_dir, num_images=4)
        if not val_images:
            return (None, '⚠️ 验证集中没有图像')
        state.add_log(f'🔍 测试推理: 加载 {best_model.name} (置信度阈值: {conf_threshold})')
        inferencer = SegFormerInference(checkpoint_path=str(best_model), device=get_device_string())
        preview_images = []
        total_detections = 0
        for img_path in val_images:
            try:
                if conf_threshold > 0:
                    pred_mask, confidence_map = inferencer.predict_with_confidence(img_path, conf_threshold=float(conf_threshold))
                else:
                    result = inferencer.predict_file(img_path)
                    pred_mask = result.pred_mask
                img = np.array(Image.open(img_path).convert('RGB'))
                vis_img = visualize_mask_overlay(img, pred_mask, alpha=0.5, ignore_index=255)
                preview_images.append(vis_img)
                unique_classes = np.unique(pred_mask)
                for cls_id in unique_classes:
                    if cls_id != 0 and cls_id != 255:
                        pixel_count = np.sum(pred_mask == cls_id)
                        if pixel_count > 100:
                            total_detections += 1
            except Exception as e:
                state.add_log(f'⚠️ 推理失败 {Path(img_path).name}: {e}')
        if not preview_images:
            return (None, '⚠️ 推理失败')
        grid = create_preview_grid(preview_images, grid_size=(2, 2), preserve_resolution=True, max_cell_size=(1500, 1500))
        state.segmentation_preview = grid
        state.add_log(f'✅ 推理完成: 检测到 {total_detections} 个区域 (阈值: {conf_threshold})')
        return (grid, f'✅ 推理完成: {len(preview_images)} 张图像, 阈值: {conf_threshold}')
    except Exception as e:
        import traceback
        state.add_log(f'❌ 推理出错: {e}')
        state.add_log(traceback.format_exc())
        return (None, f'❌ 推理失败: {str(e)}')
try:
    import torch
    _torch_available = True
except ImportError:
    torch = None
    _torch_available = False

def is_cuda_available() -> bool:
    """安全检查CUDA是否可用"""
    if not _torch_available or torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def get_device_string() -> str:
    """获取设备字符串"""
    return 'cuda:0' if is_cuda_available() else 'cpu'

def on_batch_inference(model_path: str, images_dir: str, output_dir: str, device_choice: str, min_area: int, conf_threshold: float=0.0, use_sliding_window: bool=False):
    """批量推理生成LabelMe JSON"""
    if not model_path or not model_path.strip():
        return '❌ 请输入模型文件路径'
    if not images_dir or not images_dir.strip():
        return '❌ 请输入图像文件夹路径'
    if not output_dir or not output_dir.strip():
        return '❌ 请输入输出文件夹路径'
    full_model_path = get_full_model_path_seg(model_path)
    images_dir = images_dir.strip()
    output_dir = output_dir.strip()
    if not Path(full_model_path).exists():
        return f'❌ 模型文件不存在: {full_model_path}'
    if not Path(images_dir).exists():
        return f'❌ 图像文件夹不存在: {images_dir}'
    if _YOLO_SEG_AVAILABLE and yolo_seg is not None and yolo_seg.is_yolo_seg_model(full_model_path):
        if dlhub_params:
            cur = dlhub_params.get_all()
            cur['inference'] = {'images_dir': images_dir, 'output_dir': output_dir, 'device': device_choice, 'conf_threshold': conf_threshold}
            dlhub_params.save(cur)
        return yolo_seg.run_batch_inference(model_path=full_model_path, images_dir=images_dir, output_dir=output_dir, device=device_choice, conf_threshold=float(conf_threshold), save_images=True, save_json=True)
    if dlhub_params:
        current_params = dlhub_params.get_all()
        current_params['inference'] = {'images_dir': images_dir, 'output_dir': output_dir, 'device': device_choice, 'min_area': min_area, 'conf_threshold': conf_threshold, 'use_sliding_window': use_sliding_window}
        dlhub_params.save(current_params)
    try:
        from inference.batch_inference import batch_inference_to_json
        from utils.env_validator import parse_gpu_choice
        device_str, _ = parse_gpu_choice(device_choice)
        device = f'cuda:{device_str}' if device_str != 'cpu' else 'cpu'
        result = batch_inference_to_json(checkpoint_path=full_model_path, images_dir=images_dir, output_dir=output_dir, device=device, use_sliding_window=use_sliding_window, window_size=512, stride=256, min_area=int(min_area), simplify_tolerance=2.0, conf_threshold=float(conf_threshold))
        state.batch_images_dir = images_dir
        state.batch_output_dir = output_dir
        return result.message
    except Exception as e:
        import traceback
        return f'❌ 批量推理失败: {str(e)}\n{traceback.format_exc()}'

def get_batch_preview_images(images_dir: str, output_dir: str, num_images: int=4) -> List[Tuple[str, str]]:
    """获取批量推理预览图像（图像路径和对应JSON路径）"""
    import random
    if not images_dir or not output_dir:
        return []
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    if not images_path.exists() or not output_path.exists():
        return []
    pairs = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    for ext in image_extensions:
        for img_file in images_path.glob(f'*{ext}'):
            json_file = output_path / f'{img_file.stem}.json'
            if json_file.exists():
                pairs.append((str(img_file), str(json_file)))
        for img_file in images_path.glob(f'*{ext.upper()}'):
            json_file = output_path / f'{img_file.stem}.json'
            if json_file.exists():
                pairs.append((str(img_file), str(json_file)))
    if len(pairs) > num_images:
        pairs = random.sample(pairs, num_images)
    return pairs

def visualize_json_on_image(image_path: str, json_path: str) -> np.ndarray:
    """在图像上可视化JSON标注"""
    import json
    import cv2
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法读取图像: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]
    label_colors = {}
    color_idx = 0
    for shape in data.get('shapes', []):
        label = shape.get('label', 'unknown')
        points = shape.get('points', [])
        if len(points) < 3:
            continue
        if label not in label_colors:
            label_colors[label] = colors[color_idx % len(colors)]
            color_idx += 1
        color = label_colors[label]
        pts = np.array(points, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.polylines(img, [pts], True, color, 2)
        if len(pts) > 0:
            x, y = pts[0]
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def on_refresh_batch_preview(images_dir: str, output_dir: str):
    """刷新批量推理预览"""
    from data.visualizer import create_preview_grid
    from PIL import Image
    if not images_dir or not output_dir:
        images_dir = getattr(state, 'batch_images_dir', '')
        output_dir = getattr(state, 'batch_output_dir', '')
    if not images_dir or not output_dir:
        return (None, '⚠️ 请先完成批量推理')
    try:
        pairs = get_batch_preview_images(images_dir, output_dir, num_images=4)
        if not pairs:
            return (None, '⚠️ 没有找到推理结果')
        preview_images = []
        for img_path, json_path in pairs:
            try:
                vis_img = visualize_json_on_image(img_path, json_path)
                preview_images.append(Image.fromarray(vis_img))
            except Exception as e:
                print(f'可视化失败 {img_path}: {e}')
        if not preview_images:
            return (None, '⚠️ 可视化失败')
        grid = create_preview_grid(preview_images, grid_size=(2, 2), preserve_resolution=True, max_cell_size=(800, 800))
        return (grid, f'✅ 显示 {len(preview_images)} 张预览')
    except Exception as e:
        return (None, f'❌ 预览失败: {e}')

def create_ui():
    """创建Gradio界面"""
    global state
    from config.model_registry import get_all_scales, SUPPORTED_IMG_SIZES, DEFAULT_CONFIG
    model_scales = get_all_scales()
    yolo_seg_families = ['YOLOv8', 'YOLOv11', 'YOLO26']
    if _YOLO_SEG_AVAILABLE:
        try:
            from config.yolo_seg_registry import get_all_families
            yolo_seg_families = get_all_families()
        except Exception:
            logger.debug('GUI exception ignored @ L2361', exc_info=True)
    saved_data = {} if not dlhub_params else dlhub_params.get_section('data')
    saved_model = {} if not dlhub_params else dlhub_params.get_section('model')
    saved_training = {} if not dlhub_params else dlhub_params.get_section('training')
    saved_conversion = {} if not dlhub_params else dlhub_params.get_section('conversion')
    saved_inference = {} if not dlhub_params else dlhub_params.get_section('inference')
    saved_aug = {} if not dlhub_params else dlhub_params.get_section('augmentation')
    saved_device = get_saved_param('device', 'cuda:0')
    saved_history = {} if not dlhub_params else dlhub_params.get_history()
    saved_logs = [] if not dlhub_params else dlhub_params.get_logs('training')
    saved_conv_logs = [] if not dlhub_params else dlhub_params.get_logs('conversion')
    if saved_history:
        if saved_history.get('train_losses') or saved_history.get('val_mIoU'):
            state.train_losses = saved_history.get('train_losses', [])
            state.val_mIoU = saved_history.get('val_mIoU', [])
            state.val_mDice = saved_history.get('val_mDice', [])
            state.best_mIoU = saved_history.get('best_mIoU', 0.0)
            state.best_epoch = saved_history.get('best_epoch', 0)
            state.current_epoch = saved_history.get('current_epoch', 0)
            state.total_epochs = saved_history.get('total_epochs', 0)
            state.output_dir = saved_history.get('output_dir', '')
            print(f'[DL-Hub] ✓ 已恢复训练历史: {len(state.train_losses)} epochs, 最佳mIoU: {state.best_mIoU:.4f}')
    if saved_logs:
        state.logs = saved_logs
        print(f'[DL-Hub] ✓ 已恢复训练日志: {len(saved_logs)} 行')
    if saved_conv_logs:
        conversion_state.logs = saved_conv_logs
        print(f'[DL-Hub] ✓ 已恢复转换日志: {len(saved_conv_logs)} 行')
    with gr.Blocks(title='SegFormer 语义分割训练工具', css=CUSTOM_CSS, theme=gr.themes.Soft(), head=IMAGE_VIEWER_HEAD) as app:
        gr.HTML(IMAGE_VIEWER_HTML)
        gr.HTML('\n        <div style="text-align: center; margin-bottom: 20px;">\n            <h1 style="margin: 0; font-size: 2em;">🎨 SegFormer 语义分割训练工具</h1>\n            <p style="margin: 10px 0 0 0; color: #666; font-size: 1.1em;">\n                <b>基于 MMSegmentation + SegFormer 的工业缺陷分割训练平台</b> | \n                支持 LabelMe 多边形标注 | 自动数据转换 | 实时监控 | 批量推理\n            </p>\n        </div>\n        ')
        env_status = check_environment()
        with gr.Accordion('📋 环境信息', open=False):
            gr.Textbox(value=env_status, lines=8, interactive=False, show_label=False)
        with gr.Tabs():
            with gr.Tab('🏋️ 模型训练'):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group():
                            gr.Markdown('### 📂 数据集配置')
                            images_dir = gr.Textbox(label='图像文件夹路径', placeholder='/path/to/images', info='包含训练图像的文件夹', value=saved_data.get('images_dir', ''))
                            jsons_dir = gr.Textbox(label='JSON标注文件夹路径', placeholder='/path/to/jsons', info='包含LabelMe多边形标注的文件夹', value=saved_data.get('jsons_dir', ''))
                            with gr.Row():
                                num_classes = gr.Number(label='类别数量', value=saved_data.get('num_classes', 2), minimum=1, maximum=100, step=1, info='不含背景的缺陷类别数')
                                val_split = gr.Slider(label='验证集比例', value=saved_data.get('val_split', DEFAULT_CONFIG['val_split']), minimum=0, maximum=0.5, step=0.05)
                            include_negative = gr.Checkbox(label='包含负样本（无标注图像作为良品）', value=saved_data.get('include_negative', True))
                            with gr.Row():
                                validate_btn = gr.Button('🔍 验证数据', variant='secondary')
                                preview_btn = gr.Button('👁️ 预览数据', variant='secondary')
                                refresh_preview_btn = gr.Button('🔄 换一批', variant='secondary')
                            validation_output = gr.Textbox(label='验证结果', lines=6, interactive=False)
                            data_preview = gr.Image(label='数据预览（点击放大）', type='pil', height=250)
                        seg_type = gr.Radio(label='🧩 分割类型', choices=['语义分割 (SegFormer)', '实例分割 (YOLO-seg)'], value='语义分割 (SegFormer)', info='语义分割: 逐像素类别图 (SegFormer) | 实例分割: 逐实例mask+框 (YOLO-seg)')
                        with gr.Group(visible=False) as yoloseg_group:
                            gr.Markdown('### 🧠 YOLO-seg 模型配置')
                            with gr.Row():
                                ys_family = gr.Dropdown(label='模型系列', choices=yolo_seg_families if _YOLO_SEG_AVAILABLE else ['YOLO26'], value='YOLO26', info='v8/v11/v26，推荐 v26 (mask 质量最佳)')
                                ys_scale = gr.Dropdown(label='模型规模', choices=['超小', '小', '中', '大', '超大'], value='中', info='超小→超大 对应 n/s/m/l/x')
                            ys_model_info = gr.Textbox(label='模型信息', lines=5, interactive=False)
                            gr.Markdown('### ⚙️ YOLO-seg 训练参数')
                            with gr.Row():
                                ys_epochs = gr.Number(label='训练轮数', value=100, minimum=1, maximum=1000, step=1)
                                ys_batch_size = gr.Number(label='批次大小', value=16, minimum=1, maximum=128, step=1)
                            with gr.Row():
                                ys_img_size = gr.Dropdown(label='输入尺寸', choices=[512, 640, 768, 1024, 1280], value=640)
                                ys_lr = gr.Number(label='学习率(lr0)', value=0.01, minimum=1e-06, maximum=1.0)
                            with gr.Accordion('🔧 YOLO-seg 高级参数', open=False):
                                with gr.Row():
                                    ys_optimizer = gr.Dropdown(label='优化器', choices=['SGD', 'Adam', 'AdamW', 'auto'], value='SGD')
                                    ys_patience = gr.Number(label='早停耐心值', value=50, minimum=0, maximum=300, step=1)
                                with gr.Row():
                                    ys_weight_decay = gr.Number(label='权重衰减', value=0.0005, minimum=0.0, maximum=0.1)
                                    ys_momentum = gr.Number(label='动量', value=0.937, minimum=0.0, maximum=1.0)
                                with gr.Row():
                                    ys_warmup = gr.Number(label='预热轮数', value=3.0, minimum=0.0, maximum=20.0)
                                    ys_save_period = gr.Number(label='保存周期(epoch,0=仅best/last)', value=0, minimum=0, maximum=100, step=1)
                                ys_cos_lr = gr.Checkbox(label='余弦学习率调度', value=True)
                                ys_freeze = gr.Checkbox(label='冻结 backbone (前10层)', value=False)
                                gr.Markdown('**数据增强**')
                                with gr.Row():
                                    ys_mosaic = gr.Slider(label='Mosaic', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
                                    ys_mixup = gr.Slider(label='MixUp', minimum=0.0, maximum=1.0, step=0.1, value=0.0)
                                    ys_fliplr = gr.Slider(label='水平翻转', minimum=0.0, maximum=1.0, step=0.1, value=0.5)
                            ys_start_btn = gr.Button('🚀 开始 YOLO-seg 训练', variant='primary', elem_classes=['primary-btn'])
                        with gr.Group(visible=True) as segformer_model_group:
                            model_info = gr.Textbox(label='模型信息', lines=4, interactive=False)
                            pretrained_status = gr.Textbox(label='预训练权重状态', lines=1, interactive=False)
                        with gr.Group(visible=True) as segformer_param_group:
                            gr.Markdown('### ⚙️ 训练参数')
                            with gr.Row():
                                epochs = gr.Number(label='训练轮数', value=saved_training.get('epochs', DEFAULT_CONFIG['epochs']), minimum=1, maximum=500, step=1)
                                batch_size = gr.Number(label='批次大小', value=saved_training.get('batch_size', DEFAULT_CONFIG['batch_size']), minimum=1, maximum=64, step=1)
                            with gr.Row():
                                img_size = gr.Dropdown(label='输入尺寸', choices=SUPPORTED_IMG_SIZES, value=saved_data.get('image_size', DEFAULT_CONFIG['img_size']))
                                learning_rate = gr.Number(label='学习率', value=saved_training.get('learning_rate', DEFAULT_CONFIG['learning_rate']), minimum=1e-07, maximum=0.01)
                            with gr.Accordion('🔧 高级参数', open=False):
                                with gr.Row():
                                    patience = gr.Number(label='早停耐心值', value=saved_training.get('patience', DEFAULT_CONFIG['patience']), minimum=5, maximum=100, step=1)
                                    save_period = gr.Number(label='保存周期(epoch)', value=saved_training.get('save_period', DEFAULT_CONFIG['save_period']), minimum=1, maximum=50, step=1)
                                with gr.Row():
                                    use_fp16 = gr.Checkbox(label='混合精度(FP16)', value=True)
                                    use_class_weight = gr.Checkbox(label='自动类别权重', value=True)
                                gr.Markdown('**损失函数权重**')
                                with gr.Row():
                                    loss_ce_weight = gr.Slider(label='CrossEntropy权重', value=0.4, minimum=0, maximum=1, step=0.1)
                                    loss_dice_weight = gr.Slider(label='Dice权重', value=0.6, minimum=0, maximum=1, step=0.1)
                                gr.Markdown('**数据增强**')
                                with gr.Row():
                                    flip_horizontal = gr.Checkbox(label='水平翻转', value=saved_aug.get('flip_horizontal', True))
                                    flip_vertical = gr.Checkbox(label='垂直翻转', value=saved_aug.get('flip_vertical', True))
                                    random_rotate = gr.Checkbox(label='随机旋转', value=saved_aug.get('random_rotate', True))
                                    color_jitter = gr.Checkbox(label='颜色抖动', value=saved_aug.get('color_jitter', True))
                                rotate_degree = gr.Slider(label='旋转角度范围(±)', value=saved_aug.get('rotate_degree', 30), minimum=0, maximum=90, step=5)
                        with gr.Group():
                            gr.Markdown('### 🖥️ 设备选择')
                            device_choice = gr.Dropdown(label='训练设备', choices=get_gpu_options(), value=get_gpu_options()[0] if get_gpu_options() else 'CPU')
                        with gr.Row():
                            start_btn = gr.Button('🚀 开始训练', variant='primary', elem_classes=['primary-btn'])
                            stop_btn = gr.Button('⏹️ 停止训练', variant='stop', elem_classes=['stop-btn'], interactive=False)
                            open_output_btn = gr.Button('📂 打开输出目录', variant='secondary')
                        control_status = gr.Textbox(label='控制状态', lines=1, interactive=False)
                    with gr.Column(scale=6):
                        progress_html = gr.HTML(value=get_progress_html(), label='训练进度')
                        with gr.Row():
                            loss_plot = gr.Plot(label='Loss曲线')
                            miou_plot = gr.Plot(label='mIoU曲线')
                        per_class_iou_html = gr.HTML(value="<div style='padding: 10px; color: #6b7280;'>暂无逐类别IoU数据</div>", label='逐类别IoU')
                        logs_box = gr.Textbox(label='📋 训练日志', lines=10, max_lines=20, interactive=False, elem_classes=['log-box'])
                        with gr.Row():
                            test_inference_btn = gr.Button('🔍 测试推理（验证集）', variant='secondary')
                            test_conf_threshold = gr.Slider(label='置信度阈值', minimum=0.0, maximum=1.0, step=0.05, value=0.5, info='低于此值的像素视为背景')
                        inference_status = gr.Textbox(label='推理状态', lines=1, interactive=False)
                        segmentation_preview = gr.Image(label='分割预览（点击放大）', type='pil', height=350)
                refresh_timer = gr.Timer(value=2)
            with gr.Tab('🔄 模型转换'):
                gr.Markdown('\n                ### 📝 说明\n                将训练好的分割模型转换为不同的推理格式，支持 TensorRT、ONNX Runtime、OpenVINO 等后端。\n                - **TensorRT**: NVIDIA GPU 高性能部署\n                - **ONNX Runtime**: 通用跨平台推理\n                - **OpenVINO**: Intel 硬件优化\n                ')
                gpu_options = get_gpu_options()
                trained_models_seg = get_trained_models_seg()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=350):
                        gr.Markdown('### ⚙️ 转换配置')
                        with gr.Group():
                            gr.Markdown('**📂 选择模型**')
                            conv_model_folder = gr.Dropdown(choices=[], value=None, label='选择训练文件夹', info='从output目录选择训练结果文件夹')
                            conv_model_dropdown = gr.Dropdown(choices=[], value=None, label='选择模型文件', allow_custom_value=True)
                        with gr.Group():
                            gr.Markdown('**🎯 转换目标**')
                            target_backend = gr.Dropdown(choices=['tensorrt', 'onnxruntime', 'openvino'], value=saved_conversion.get('target_backend', 'tensorrt'), label='目标后端', info='TensorRT: NVIDIA GPU | ONNX Runtime: 通用 | OpenVINO: Intel')
                            precision = gr.Dropdown(choices=['fp16', 'fp32', 'int8', 'mixed'], value=saved_conversion.get('precision', 'fp16'), label='精度模式', info='fp16推荐 | int8需要校准数据 | mixed混合精度')
                            conv_device = gr.Dropdown(choices=gpu_options, value=saved_conversion.get('device', gpu_options[0] if gpu_options else 'CPU'), label='设备')
                        with gr.Group(visible=False) as calib_group:
                            gr.Markdown('**📊 校准数据集**')
                            calib_data_path = gr.Textbox(label='校准图片文件夹路径', placeholder='输入包含图片的文件夹路径...', info='用于INT8量化校准，无需标注文件')
                            calib_status = gr.Textbox(label='验证状态', interactive=False, value='⏳ 请输入校准数据集路径', lines=2)
                        with gr.Group(visible=True) as trt_group:
                            gr.Markdown('**🔧 TensorRT 专用选项**')
                            workspace_gb = gr.Slider(1, 16, value=saved_conversion.get('workspace_gb', 4), step=1, label='Workspace (GB)', info='更大空间可能找到更优算法')
                        with gr.Group(visible=True) as dynamic_batch_group:
                            gr.Markdown('**⚡ 动态Batch配置**')
                            dynamic_batch = gr.Checkbox(value=saved_conversion.get('dynamic_batch', False), label='启用动态批处理', info='允许运行时使用不同batch size')
                            with gr.Row():
                                min_batch = gr.Slider(1, 64, value=saved_conversion.get('min_batch', 1), step=1, label='最小Batch')
                                opt_batch = gr.Slider(1, 64, value=saved_conversion.get('opt_batch', 1), step=1, label='最优Batch (TensorRT)', visible=True)
                                max_batch = gr.Slider(1, 64, value=saved_conversion.get('max_batch', 8), step=1, label='最大Batch')
                        with gr.Group():
                            gr.Markdown('**🧩 YOLO-seg 选项** (仅实例分割模型生效)')
                            conv_end2end = gr.Checkbox(value=saved_conversion.get('end2end', True), label='end2end (NMS-free)', info='YOLO26-seg: 开=NMS-free单检测输出+proto; 关=原始双输出(部署侧自行NMS)。SegFormer 模型忽略此项')
                        with gr.Group():
                            gr.Markdown('**📁 输出设置**')
                            conv_output_dir = gr.Textbox(label='输出目录', value=saved_conversion.get('output_dir', ''), placeholder='留空则输出到模型目录下的 converted 文件夹')
                        conv_btn = gr.Button('🚀 开始转换', variant='primary', elem_classes='primary-btn')
                        conv_status_text = gr.Textbox(label='状态', interactive=False, lines=1)
                    with gr.Column(scale=2):
                        gr.Markdown('### 📝 转换日志')
                        conv_logs = gr.Textbox(label='日志', lines=30, interactive=False, elem_classes='log-box')
                conv_timer = gr.Timer(value=2)
                conv_timer.tick(fn=get_conversion_status, outputs=[conv_status_text, conv_logs])
                conv_model_folder.focus(fn=scan_model_folders_seg, outputs=conv_model_folder)
                conv_model_folder.change(fn=scan_models_in_selected_folder_seg, inputs=conv_model_folder, outputs=conv_model_dropdown)
                target_backend.change(fn=update_backend_options, inputs=target_backend, outputs=[trt_group, workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch, dynamic_batch_group])
                precision.change(fn=update_calib_visibility, inputs=precision, outputs=calib_group)
                calib_data_path.change(fn=validate_calib_data_seg, inputs=calib_data_path, outputs=calib_status)
                calib_data_path.submit(fn=validate_calib_data_seg, inputs=calib_data_path, outputs=calib_status)
                conv_btn.click(fn=on_start_conversion_seg, inputs=[conv_model_dropdown, target_backend, precision, conv_device, workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch, conv_output_dir, calib_data_path, conv_end2end], outputs=[conv_status_text, conv_logs])
            with gr.Tab('🔍 批量推理'):
                gr.Markdown('\n                ### 批量推理\n                使用训练好的模型对图像文件夹进行批量分割，输出 **LabelMe格式JSON标注文件**。\n                ')
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown('**📂 选择模型**')
                            inf_model_folder = gr.Dropdown(choices=[], value=None, label='选择训练文件夹', info='从output目录选择训练结果文件夹')
                            batch_model_path = gr.Dropdown(choices=[], value=None, label='选择模型文件 (.pth)', allow_custom_value=True)
                        batch_images_dir = gr.Textbox(label='📁 图像文件夹路径', value=saved_inference.get('images_dir', ''), placeholder='例如: G:/test_images', info='包含待预测图像的文件夹')
                        batch_output_dir = gr.Textbox(label='📂 输出文件夹路径', value=saved_inference.get('output_dir', ''), placeholder='例如: G:/predictions', info='保存JSON标注文件的目录')
                        with gr.Row():
                            batch_device = gr.Dropdown(label='🖥️ 推理设备', choices=get_gpu_options(), value=saved_inference.get('device', get_gpu_options()[0] if get_gpu_options() else 'CPU'), scale=2)
                            batch_min_area = gr.Number(label='最小面积', value=saved_inference.get('min_area', 100), info='过滤小于此面积的区域', scale=1)
                        batch_conf_threshold = gr.Slider(label='🎯 置信度阈值', minimum=0.0, maximum=1.0, step=0.05, value=saved_inference.get('conf_threshold', 0.0), info='低于此值的像素视为背景（0=不过滤）')
                        batch_use_sliding_window = gr.Checkbox(label='🔲 使用滑动窗口（大图推荐开启）', value=saved_inference.get('use_sliding_window', False), info='关闭=直接Resize到训练尺寸（与训练一致），开启=滑动窗口切分推理（大图推荐）')
                        batch_start_btn = gr.Button('🚀 开始批量推理', variant='primary', size='lg')
                        batch_result = gr.Textbox(label='📋 推理结果', lines=8, interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown('### 📷 推理结果预览')
                        batch_refresh_btn = gr.Button('🔄 换一批查看', variant='secondary')
                        batch_preview_status = gr.Textbox(label='状态', interactive=False, lines=1)
                        batch_preview_image = gr.Image(label='预览（点击图像可放大）', type='pil', height=500)
        validate_btn.click(fn=validate_data_paths, inputs=[images_dir, jsons_dir], outputs=[validation_output])
        preview_btn.click(fn=get_data_preview, inputs=[images_dir, jsons_dir], outputs=[data_preview])
        refresh_preview_btn.click(fn=refresh_data_preview, inputs=[images_dir, jsons_dir], outputs=[data_preview])
        model_scale.change(fn=update_model_info, inputs=[model_scale], outputs=[model_info])
        model_scale.change(fn=check_pretrained_weights, inputs=[model_scale], outputs=[pretrained_status])
        start_btn.click(fn=on_start_training, inputs=[images_dir, jsons_dir, num_classes, model_scale, epochs, batch_size, img_size, learning_rate, val_split, include_negative, patience, save_period, use_fp16, use_class_weight, loss_ce_weight, loss_dice_weight, flip_horizontal, flip_vertical, random_rotate, rotate_degree, color_jitter, device_choice], outputs=[control_status, start_btn, stop_btn])
        stop_btn.click(fn=on_stop_training, outputs=[control_status])

        def _on_seg_type_change(stype):
            is_inst = stype == '实例分割 (YOLO-seg)'
            return (gr.update(visible=not is_inst), gr.update(visible=not is_inst), gr.update(visible=is_inst), gr.update(visible=not is_inst))
        seg_type.change(fn=_on_seg_type_change, inputs=[seg_type], outputs=[segformer_model_group, segformer_param_group, yoloseg_group, start_btn])
        ys_family.change(fn=yoloseg_update_scales, inputs=[ys_family], outputs=[ys_scale])
        ys_family.change(fn=yoloseg_update_model_info, inputs=[ys_family, ys_scale], outputs=[ys_model_info])
        ys_scale.change(fn=yoloseg_update_model_info, inputs=[ys_family, ys_scale], outputs=[ys_model_info])
        ys_start_btn.click(fn=on_start_training_yoloseg, inputs=[images_dir, jsons_dir, num_classes, ys_family, ys_scale, ys_epochs, ys_batch_size, ys_img_size, ys_lr, val_split, include_negative, ys_patience, ys_save_period, ys_optimizer, ys_weight_decay, ys_momentum, ys_warmup, ys_cos_lr, ys_mosaic, ys_mixup, ys_fliplr, ys_freeze, device_choice], outputs=[control_status, ys_start_btn, stop_btn])
        open_output_btn.click(fn=on_open_output, outputs=[control_status])
        test_inference_btn.click(fn=on_test_inference, inputs=[test_conf_threshold], outputs=[segmentation_preview, inference_status])
        refresh_timer.tick(fn=get_training_status, outputs=[progress_html, loss_plot, miou_plot, logs_box, per_class_iou_html, segmentation_preview])
        app.load(fn=update_model_info, inputs=[model_scale], outputs=[model_info])
        app.load(fn=check_pretrained_weights, inputs=[model_scale], outputs=[pretrained_status])
        if _YOLO_SEG_AVAILABLE:
            app.load(fn=yoloseg_update_model_info, inputs=[ys_family, ys_scale], outputs=[ys_model_info])
        inf_model_folder.focus(fn=scan_model_folders_seg, outputs=inf_model_folder)
        inf_model_folder.change(fn=scan_models_in_selected_folder_seg, inputs=inf_model_folder, outputs=batch_model_path)
        batch_start_btn.click(fn=on_batch_inference, inputs=[batch_model_path, batch_images_dir, batch_output_dir, batch_device, batch_min_area, batch_conf_threshold, batch_use_sliding_window], outputs=[batch_result]).then(fn=on_refresh_batch_preview, inputs=[batch_images_dir, batch_output_dir], outputs=[batch_preview_image, batch_preview_status])
        batch_refresh_btn.click(fn=on_refresh_batch_preview, inputs=[batch_images_dir, batch_output_dir], outputs=[batch_preview_image, batch_preview_status])
        return app

def main():
    """主函数"""
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
    os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    parser = argparse.ArgumentParser(description='SegFormer 语义分割训练工具')
    parser.add_argument('--task-dir', type=str, default=None, help='DL-Hub 任务目录')
    parser.add_argument('--port', type=int, default=7862, help='Gradio 服务端口')
    args, _ = parser.parse_known_args()
    print('=' * 60)
    print('🎨 SegFormer 语义分割训练工具')
    print('=' * 60)
    env_msg = check_environment()
    print(env_msg)
    print()
    app = create_ui()
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode:
        launch_port = dlhub_adapter.port
        launch_inbrowser = False
        server_name = '127.0.0.1'
        print(f'[DL-Hub] 以 DL-Hub 模式启动，端口: {launch_port}')
    else:
        launch_port = args.port
        launch_inbrowser = True
        server_name = '127.0.0.1'
        print(f'[独立模式] 启动端口: {launch_port}')
    print(f'🌐 访问地址: http://{server_name}:{launch_port}')
    app.launch(server_name=server_name, server_port=launch_port, share=False, show_error=True, inbrowser=launch_inbrowser)
if __name__ == '__main__':
    main()