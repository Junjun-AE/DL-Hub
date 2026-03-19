"""
YOLO 目标检测训练工具 - Gradio Web界面
功能：环境检查、模型训练、模型转换、批量推理

修复版本 - 解决连续转换中止问题

已集成 DL-Hub 支持：
- 支持 --task-dir 参数指定任务目录
- 支持 --port 参数指定端口
- 自动保存/加载UI参数
"""

# ============ 关键修复：必须在最开始设置 Matplotlib 后端 ============
# 这必须在任何其他 matplotlib 导入之前执行！
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免线程安全问题
# ==================================================================

import os
import sys
import threading
import time
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import gradio as gr
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


# ==================== DL-Hub 集成 ====================
# 添加父目录到路径以便导入dlhub_params
sys.path.insert(0, str(Path(__file__).parent.parent))

def init_dlhub_adapter():
    """初始化 DL-Hub 适配器"""
    try:
        dlhub_path = Path(__file__).parent.parent / 'dlhub_project' / 'dlhub'
        if dlhub_path.exists():
            sys.path.insert(0, str(dlhub_path.parent))
        
        from dlhub.app_adapters.base_adapter import get_adapter
        adapter = get_adapter(default_port=7861)
        print(f"[DL-Hub] 适配器已初始化，模式: {'DL-Hub' if adapter.is_dlhub_mode else '独立'}")
        return adapter
    except ImportError:
        print("[DL-Hub] 适配器未找到，以独立模式运行")
        return None
    except Exception as e:
        print(f"[DL-Hub] 初始化失败: {e}，以独立模式运行")
        return None


def init_dlhub_params():
    """初始化 DL-Hub 参数管理器（使用单例模式）"""
    try:
        from dlhub_params import get_dlhub_params
        params = get_dlhub_params()
        # 注意：日志已在get_dlhub_params()中打印，这里不重复
        return params
    except ImportError:
        print("[DL-Hub] 参数管理器未找到，参数不会持久化")
        return None
    except Exception as e:
        print(f"[DL-Hub] 参数管理器初始化失败: {e}")
        return None


dlhub_adapter = init_dlhub_adapter()
dlhub_params = init_dlhub_params()


def get_output_dir(default: str = './output') -> Path:
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


# ==================== 自定义CSS样式 ====================
CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
    margin: 0 !important;
}
.main { max-width: 100% !important; }
.contain { max-width: 100% !important; }
.row { gap: 20px !important; }
.group { padding: 15px !important; }

.log-box textarea {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    font-size: 13px !important;
    background: #1a1a2e !important;
    color: #eee !important;
    border-radius: 8px !important;
    line-height: 1.5 !important;
}

.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    padding: 12px 24px !important;
}

.stop-btn {
    background: linear-gradient(135deg, #f43f5e 0%, #ec4899 100%) !important;
    border: none !important;
}
"""


# ==================== 全局状态管理 ====================
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
        self.box_losses = []
        self.cls_losses = []
        self.map50_list = []
        self.map50_95_list = []
        self.logs = []
        self.best_map50 = 0.0
        self.best_map50_95 = 0.0
        self.best_epoch = 0
        self.output_dir = ""
        self.start_time = None
        self.data_yaml = ""
        self.class_names = []
        self.detection_preview = None
        self.images_dir = ""
        self.jsons_dir = ""
    
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
        self.output_path = ""
        self.start_time = None
    
    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]


class InferenceState:
    """推理状态管理"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.is_running = False
        self.logs = []
        self.results = []
        self.output_dir = ""
        self.start_time = None
        self.total_images = 0
        self.processed_images = 0
        self.preview_images = []
        self.preview_jsons = []
        self.current_preview_idx = 0
    
    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]


# 全局状态实例
training_state = TrainingState()
conversion_state = ConversionState()
inference_state = InferenceState()
preview_seed = [42]
_cached_figures = {'fig_loss': None, 'ax_loss': None, 'fig_map': None, 'ax_map': None}

# ============ 关键修复：训练状态UI的图表缓存 ============
# 用于 get_training_status_ui() 函数，避免重复创建 figure 导致内存泄漏
_training_ui_figures = {
    'fig_loss': None,
    'ax_loss': None,
    'fig_map': None,
    'ax_map': None
}
# =========================================================


# ==================== 通用工具函数 ====================
def check_environment() -> str:
    try:
        from utils.env_validator import validate_environment
        success, message, info = validate_environment()
        return message
    except Exception as e:
        return f"❌ 环境验证出错: {str(e)}"


def get_gpu_options() -> list:
    try:
        from utils.env_validator import get_gpu_choices
        return get_gpu_choices()
    except Exception:
        return ["CPU"]


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}时"


def get_trained_models() -> List[str]:
    models = []
    output_dir = Path("./output")
    if output_dir.exists():
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                importer_dir = model_dir / "weights_model_importer"
                if importer_dir.exists():
                    for model_file in ["best_model.pt", "last_model.pt"]:
                        model_path = importer_dir / model_file
                        if model_path.exists():
                            models.append(str(model_path))
    return sorted(models, reverse=True)


def scan_models_in_folder(folder_path: str) -> Dict:
    if not folder_path or not folder_path.strip():
        return gr.update(choices=[], value=None)
    folder_path = folder_path.strip()
    if os.path.isfile(folder_path):
        if folder_path.endswith('.pt'):
            return gr.update(choices=[folder_path], value=folder_path)
        return gr.update(choices=[], value=None)
    if not os.path.exists(folder_path):
        return gr.update(choices=[], value=None)
    models = list(Path(folder_path).rglob('*.pt'))
    model_paths = sorted([str(m) for m in models], reverse=True)
    if model_paths:
        return gr.update(choices=model_paths, value=model_paths[0])
    return gr.update(choices=[], value=None)


def on_start_training(
    images_dir: str, jsons_dir: str, model_family: str, model_scale: str,
    epochs: int, batch_size: int, learning_rate: float, optimizer: str, img_size: int, patience: int,
    freeze_backbone: bool, val_split: float, save_period: int, include_negative: bool, num_classes: int,
    gpu_choice: str, weight_decay: float, momentum: float, warmup_epochs: float,
    cos_lr: bool, mosaic: float, mixup: float, fliplr: float,
) -> Tuple[str, str]:
    global training_state, conversion_state
    if conversion_state.is_running:
        return "⚠️ 模型转换正在进行中，请等待转换完成后再开始训练", ""
    if training_state.is_training:
        return "⚠️ 训练正在进行中...", "\n".join(training_state.logs[-50:])
    if not images_dir or not images_dir.strip():
        return "❌ 请输入图像文件夹路径", ""
    if not jsons_dir or not jsons_dir.strip():
        return "❌ 请输入JSON文件夹路径", ""
    if not model_family or not model_scale:
        return "❌ 请选择模型系列和规模", ""
    training_state.reset()
    training_state.is_training = True
    training_state.total_epochs = int(epochs)
    training_state.images_dir = images_dir.strip()
    training_state.jsons_dir = jsons_dir.strip()
    training_state.start_time = time.time()
    thread = threading.Thread(target=run_training, args=(
        images_dir.strip(), jsons_dir.strip(), model_family, model_scale,
        int(epochs), int(batch_size), float(learning_rate), optimizer, int(img_size), int(patience),
        freeze_backbone, float(val_split), int(save_period), include_negative, int(num_classes),
        gpu_choice, float(weight_decay), float(momentum), float(warmup_epochs),
        cos_lr, float(mosaic), float(mixup), float(fliplr),
    ), daemon=True)
    thread.start()
    return "🚀 训练已启动，请查看下方进度...", ""


def run_training(
    images_dir: str, jsons_dir: str, model_family: str, model_scale: str,
    epochs: int, batch_size: int, learning_rate: float, optimizer: str, img_size: int, patience: int,
    freeze_backbone: bool, val_split: float, save_period: int, include_negative: bool, num_classes: int,
    gpu_choice: str, weight_decay: float, momentum: float, warmup_epochs: float,
    cos_lr: bool, mosaic: float, mixup: float, fliplr: float,
):
    global training_state
    try:
        from config.model_registry import get_model_config
        from data.converter import convert_labelme_to_yolo
        from engine.trainer import YOLOTrainer, TrainingCallback
        from utils.env_validator import parse_gpu_choice
        training_state.add_log("━" * 50)
        training_state.add_log("🚀 开始目标检测训练")
        training_state.add_log("━" * 50)
        device_type, gpu_ids = parse_gpu_choice(gpu_choice)
        # 修复: device_type返回的是"0"或"cpu"，不是"cuda"
        device_str = device_type if device_type != "cpu" else 'cpu'
        training_state.add_log(f"🖥️ 设备: {'cuda:' + device_str if device_str != 'cpu' else 'cpu'}")
        config = get_model_config(model_family, model_scale)
        model_name = config['ultralytics_name']
        training_state.add_log(f"🧠 模型: {model_name} ({config['params']}M 参数)")
        training_state.add_log("📂 转换数据集为YOLO格式...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / f"{config['name']}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        training_state.output_dir = str(output_dir)
        data_yaml, class_names = convert_labelme_to_yolo(
            images_dir=images_dir, jsons_dir=jsons_dir,
            output_dir=str(output_dir / "dataset"), val_split=val_split,
            num_classes=num_classes, include_negative_samples=include_negative,
        )
        training_state.data_yaml = data_yaml
        training_state.class_names = class_names
        training_state.add_log(f"   ✅ 类别: {class_names}")
        
        def on_epoch_end(epoch: int, metrics: Dict):
            training_state.current_epoch = epoch
            training_state.box_losses.append(float(metrics.get('box_loss', 0)))
            training_state.cls_losses.append(float(metrics.get('cls_loss', 0)))
            training_state.map50_list.append(float(metrics.get('map50', 0)))
            training_state.map50_95_list.append(float(metrics.get('map50_95', 0)))
            training_state.best_map50 = float(metrics.get('best_map50', 0))
            training_state.best_map50_95 = float(metrics.get('best_map50_95', 0))
            training_state.best_epoch = int(metrics.get('best_epoch', 0))
        
        def on_log(message: str):
            training_state.add_log(message)
        
        def should_stop():
            return training_state.should_stop
        
        callback = TrainingCallback(on_epoch_end=on_epoch_end, on_log=on_log, should_stop=should_stop)
        trainer = YOLOTrainer(
            model_name=model_name, data_yaml=data_yaml, output_dir=str(output_dir),
            epochs=epochs, batch_size=batch_size, img_size=img_size, learning_rate=learning_rate,
            optimizer=optimizer, patience=patience, freeze_backbone=freeze_backbone,
            device=device_str, workers=4, weight_decay=weight_decay, momentum=momentum,
            warmup_epochs=warmup_epochs, cos_lr=cos_lr, mosaic=mosaic, mixup=mixup, fliplr=fliplr,
            save_period=save_period, callback=callback, class_names=class_names,
        )
        training_state.trainer = trainer
        trainer.train()
        elapsed = time.time() - training_state.start_time
        training_state.add_log("━" * 50)
        training_state.add_log(f"✅ 训练完成！用时 {elapsed/60:.1f} 分钟")
        training_state.add_log(f"🏆 最佳 mAP@50: {training_state.best_map50:.4f} (Epoch {training_state.best_epoch})")
        training_state.add_log("━" * 50)
    except Exception as e:
        import traceback
        training_state.add_log(f"❌ 错误: {str(e)}")
        training_state.add_log(traceback.format_exc())
    finally:
        training_state.is_training = False


def on_stop_training() -> str:
    global training_state
    if not training_state.is_training:
        return "⚠️ 当前没有正在进行的训练"
    training_state.should_stop = True
    training_state.add_log("⏹️ 正在停止训练...")
    return "⏹️ 正在停止，等待当前epoch完成..."


def on_open_output() -> str:
    if training_state.output_dir and os.path.exists(training_state.output_dir):
        try:
            if sys.platform == 'win32':
                os.startfile(training_state.output_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{training_state.output_dir}"')
            else:
                os.system(f'xdg-open "{training_state.output_dir}"')
            return f"📁 已打开: {training_state.output_dir}"
        except Exception:
            return f"📁 输出目录: {training_state.output_dir}"
    return "⚠️ 输出目录不存在"


def on_test_inference():
    global training_state
    if not training_state.data_yaml:
        return None, "❌ 请先开始训练"
    try:
        from data.dataset import get_val_images
        from data.visualizer import visualize_ultralytics_result
        val_images = get_val_images(training_state.data_yaml, num_images=1)
        if not val_images:
            return None, "❌ 未找到验证集图像"
        if training_state.trainer and training_state.trainer.yolo:
            import torch
            with torch.no_grad():
                results = training_state.trainer.yolo.predict(source=val_images[0], conf=0.25, save=False, verbose=False)
            if results:
                vis_img = visualize_ultralytics_result(results[0], class_names=training_state.class_names)
                return vis_img, f"✅ 检测完成: {Path(val_images[0]).name}"
        return None, "⚠️ 模型未加载"
    except Exception as e:
        return None, f"❌ 推理失败: {str(e)}"


def get_training_status() -> Tuple[str, str, str, Any, Any, Any]:
    global training_state, _cached_figures
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    progress_html = get_progress_html()
    if training_state.is_training:
        status_text = f"🏃 训练中 - Epoch {training_state.current_epoch}/{training_state.total_epochs}"
    elif training_state.current_epoch > 0:
        status_text = f"✅ 训练完成 - 最佳mAP@50: {training_state.best_map50:.4f}"
    else:
        status_text = "⏳ 等待开始..."
    logs = "\n".join(training_state.logs[-80:]) if training_state.logs else "暂无日志"
    
    # Loss图
    if _cached_figures['fig_loss'] is None:
        _cached_figures['fig_loss'], _cached_figures['ax_loss'] = plt.subplots(figsize=(5, 3.5), dpi=100)
    else:
        _cached_figures['ax_loss'].clear()
    ax_loss = _cached_figures['ax_loss']
    fig_loss = _cached_figures['fig_loss']
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.set_title('Loss Curve', fontsize=11, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    if training_state.box_losses:
        epochs = list(range(1, len(training_state.box_losses) + 1))
        ax_loss.plot(epochs, training_state.box_losses, 'b-o', label='Box Loss', linewidth=2, markersize=3)
        if training_state.cls_losses:
            ax_loss.plot(epochs[:len(training_state.cls_losses)], training_state.cls_losses, 'r-s', label='Cls Loss', linewidth=2, markersize=3)
        ax_loss.legend(loc='upper right', fontsize=9)
    else:
        ax_loss.text(0.5, 0.5, 'Waiting...', ha='center', va='center', transform=ax_loss.transAxes, fontsize=11, color='gray')
    fig_loss.tight_layout()
    
    # mAP图
    if _cached_figures['fig_map'] is None:
        _cached_figures['fig_map'], _cached_figures['ax_map'] = plt.subplots(figsize=(5, 3.5), dpi=100)
    else:
        _cached_figures['ax_map'].clear()
    ax_map = _cached_figures['ax_map']
    fig_map = _cached_figures['fig_map']
    ax_map.set_xlabel('Epoch', fontsize=10)
    ax_map.set_ylabel('mAP', fontsize=10)
    ax_map.set_title('mAP Curve', fontsize=11, fontweight='bold')
    ax_map.grid(True, alpha=0.3)
    if training_state.map50_list:
        epochs = list(range(1, len(training_state.map50_list) + 1))
        ax_map.plot(epochs, training_state.map50_list, 'g-o', label='mAP@50', linewidth=2, markersize=3)
        if training_state.map50_95_list:
            ax_map.plot(epochs[:len(training_state.map50_95_list)], training_state.map50_95_list, 'm-s', label='mAP@50:95', linewidth=2, markersize=3)
        ax_map.legend(loc='lower right', fontsize=9)
        if training_state.best_epoch > 0 and training_state.best_epoch <= len(training_state.map50_list):
            ax_map.scatter([training_state.best_epoch], [training_state.best_map50], color='red', s=100, zorder=5, marker='*')
    else:
        ax_map.text(0.5, 0.5, 'Waiting...', ha='center', va='center', transform=ax_map.transAxes, fontsize=11, color='gray')
    fig_map.tight_layout()
    return progress_html, status_text, logs, fig_loss, fig_map, training_state.detection_preview


# ==================== Tab 1: 训练相关函数 ====================
def validate_data_paths(images_dir: str, jsons_dir: str) -> str:
    if not images_dir or not images_dir.strip():
        return "⏳ 请输入图像文件夹路径"
    if not jsons_dir or not jsons_dir.strip():
        return "⏳ 请输入JSON文件夹路径"
    try:
        from utils.data_validator import validate_labelme_dataset
        result = validate_labelme_dataset(images_dir.strip(), jsons_dir.strip())
        return result.message
    except Exception as e:
        return f"❌ 验证出错: {str(e)}"


def update_model_info(family: str, scale: str) -> str:
    if not family or not scale:
        return "请选择模型系列和规模"
    try:
        from config.model_registry import get_model_display_info
        return get_model_display_info(family, scale)
    except Exception as e:
        return f"获取模型信息失败: {str(e)}"


def check_pretrained_weights(family: str, scale: str) -> str:
    if not family or not scale:
        return ""
    try:
        from models.model_factory import check_pretrained_status
        return check_pretrained_status(family, scale)
    except Exception as e:
        return f"检查失败: {str(e)}"


def get_data_preview(images_dir: str, jsons_dir: str):
    """获取数据预览，返回图片列表供Gallery显示"""
    if not images_dir or not jsons_dir:
        return []
    try:
        from data.visualizer import preview_dataset
        result = preview_dataset(images_dir.strip(), jsons_dir.strip(), num_samples=4, seed=preview_seed[0], high_resolution=True)
        # Gallery组件需要一个图片列表，而不是单个图片
        if result is None:
            return []
        # 如果preview_dataset返回单个Image对象，包装成列表
        from PIL import Image
        if isinstance(result, Image.Image):
            return [result]
        # 如果已经是列表，直接返回
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]
    except Exception as e:
        print(f"预览失败: {e}")
        return []


def refresh_data_preview(images_dir: str, jsons_dir: str):
    import random
    preview_seed[0] = random.randint(0, 10000)
    return get_data_preview(images_dir, jsons_dir)


def get_progress_html() -> str:
    global training_state
    progress_pct = (training_state.current_epoch / training_state.total_epochs) * 100 if training_state.total_epochs > 0 else 0
    elapsed = time.time() - training_state.start_time if training_state.start_time else 0
    if training_state.current_epoch > 0 and training_state.is_training:
        per_epoch = elapsed / training_state.current_epoch
        estimated_remaining = per_epoch * (training_state.total_epochs - training_state.current_epoch)
    else:
        estimated_remaining = 0
    if training_state.is_training:
        status_color, status_text = "#3b82f6", "训练中"
    elif training_state.current_epoch > 0:
        status_color, status_text = "#10b981", "已完成"
    else:
        status_color, status_text = "#6b7280", "等待中"
    return f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); padding: 15px; border-radius: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="font-weight: bold;">训练进度</span>
            <span style="background: {status_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.85em;">{status_text}</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 15px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {progress_pct:.1f}%; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-size: 0.85em; font-weight: bold;">{progress_pct:.1f}%</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <div style="text-align: center; flex: 1;"><div style="font-size: 0.85em; color: #6c757d;">当前轮次</div><div style="font-size: 1.4em; font-weight: bold;">{training_state.current_epoch}/{training_state.total_epochs}</div></div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #ddd;"><div style="font-size: 0.85em; color: #6c757d;">最佳 mAP@50</div><div style="font-size: 1.4em; font-weight: bold; color: #10b981;">{training_state.best_map50:.4f}</div></div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #ddd;"><div style="font-size: 0.85em; color: #6c757d;">最佳轮次</div><div style="font-size: 1.4em; font-weight: bold;">{f'Epoch {training_state.best_epoch}' if training_state.best_epoch > 0 else '--'}</div></div>
        </div>
        <div style="display: flex; justify-content: space-between; background: white; padding: 12px; border-radius: 8px;">
            <div style="text-align: center; flex: 1;"><div style="font-size: 0.8em; color: #6c757d;">⏱️ 已用时间</div><div style="font-weight: bold;">{format_time(elapsed)}</div></div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #eee;"><div style="font-size: 0.8em; color: #6c757d;">⏳ 预计剩余</div><div style="font-weight: bold;">{format_time(estimated_remaining) if training_state.is_training and training_state.current_epoch > 0 else '--'}</div></div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #eee;"><div style="font-size: 0.8em; color: #6c757d;">📊 每轮耗时</div><div style="font-weight: bold;">{format_time(elapsed / training_state.current_epoch) if training_state.current_epoch > 0 else '--'}</div></div>
        </div>
    </div>
    """


def on_start_training(images_dir, jsons_dir, model_family, model_scale, epochs, batch_size, learning_rate,
                      optimizer_name, img_size, patience, freeze_backbone, val_split, save_period,
                      include_negative, num_classes, gpu_choice, weight_decay, momentum, warmup_epochs,
                      cos_lr, mosaic, mixup, fliplr) -> Tuple[str, str]:
    global training_state, conversion_state
    if conversion_state.is_running:
        return "⚠️ 模型转换正在进行中", ""
    if training_state.is_training:
        return "⚠️ 训练正在进行中...", ""
    if not images_dir or not images_dir.strip():
        return "❌ 请输入图像文件夹路径", ""
    if not jsons_dir or not jsons_dir.strip():
        return "❌ 请输入JSON文件夹路径", ""
    if not model_family or not model_scale:
        return "❌ 请选择模型系列和规模", ""
    
    training_state.reset()
    training_state.is_training = True
    training_state.total_epochs = int(epochs)
    training_state.images_dir = images_dir.strip()
    training_state.jsons_dir = jsons_dir.strip()
    training_state.start_time = time.time()
    
    thread = threading.Thread(target=run_training, args=(
        images_dir.strip(), jsons_dir.strip(), model_family, model_scale,
        int(epochs), int(batch_size), float(learning_rate), optimizer_name, int(img_size), int(patience),
        freeze_backbone, float(val_split), int(save_period), include_negative, int(num_classes), gpu_choice,
        float(weight_decay), float(momentum), float(warmup_epochs), cos_lr, float(mosaic), float(mixup), float(fliplr)
    ), daemon=True)
    thread.start()
    return "🚀 训练已启动...", ""


def run_training(images_dir, jsons_dir, model_family, model_scale, epochs, batch_size, learning_rate,
                 optimizer_name, img_size, patience, freeze_backbone, val_split, save_period,
                 include_negative, num_classes, gpu_choice, weight_decay, momentum, warmup_epochs,
                 cos_lr, mosaic, mixup, fliplr):
    global training_state
    try:
        from config.model_registry import get_model_config
        from data.converter import convert_labelme_to_yolo
        from engine.trainer import YOLOTrainer, TrainingCallback
        from utils.env_validator import parse_gpu_choice
        
        training_state.add_log("━" * 50)
        training_state.add_log("🚀 开始目标检测训练")
        training_state.add_log("━" * 50)
        
        # 保存训练参数到DL-Hub - 使用合并方式避免覆盖其他参数
        if dlhub_params:
            current_params = dlhub_params.get_all()
            current_params['data'] = {
                'images_dir': images_dir,
                'jsons_dir': jsons_dir,
                'image_size': img_size,
                'val_split': val_split,
                'include_negative': include_negative,
                'num_classes': num_classes,
            }
            current_params['model'] = {
                'family': model_family,
                'scale': model_scale,
                'freeze_backbone': freeze_backbone,
            }
            current_params['training'] = {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'optimizer': optimizer_name,
                'patience': patience,
                'save_period': save_period,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'warmup_epochs': warmup_epochs,
                'cos_lr': cos_lr,
            }
            current_params['augmentation'] = {
                'mosaic': mosaic,
                'mixup': mixup,
                'fliplr': fliplr,
            }
            current_params['device'] = gpu_choice
            dlhub_params.save(current_params)
        
        device_type, gpu_ids = parse_gpu_choice(gpu_choice)
        # 修复: device_type返回的是"0"或"cpu"，不是"cuda"
        device = device_type if device_type != "cpu" else 'cpu'
        training_state.add_log(f"🖥️ 设备: {'cuda:' + device if device != 'cpu' else 'cpu'}")
        
        config = get_model_config(model_family, model_scale)
        model_name = config['ultralytics_name']
        training_state.add_log(f"🧠 模型: {model_name}")
        
        training_state.add_log("📊 转换数据集为YOLO格式...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 关键修复：使用get_output_dir()获取正确的输出目录
        output_base = get_output_dir()
        output_dir = output_base / f"{config['name']}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        training_state.output_dir = str(output_dir)
        training_state.add_log(f"📁 输出目录: {output_dir}")
        
        # 【增强】清空之前的历史数据，开始新训练
        if dlhub_params:
            dlhub_params.clear_history(auto_save=False)
            dlhub_params.clear_logs('training', auto_save=False)
            dlhub_params.save_history({
                'total_epochs': epochs,
                'current_epoch': 0,
                'output_dir': str(output_dir),
                'completed': False
            }, auto_save=True)
        
        conversion_result = convert_labelme_to_yolo(
            images_dir=images_dir, jsons_dir=jsons_dir, output_dir=str(output_dir / "dataset"),
            val_split=val_split, num_classes=num_classes, include_negative_samples=include_negative)
        
        # 检查转换是否成功
        if not conversion_result.success:
            raise Exception(f"数据集转换失败: {conversion_result.message}")
        
        data_yaml = conversion_result.data_yaml
        class_names = conversion_result.class_names
        
        training_state.data_yaml = data_yaml
        training_state.class_names = class_names
        training_state.add_log(f"   ├─ 类别数: {len(class_names)}")
        training_state.add_log(f"   ├─ 训练图像: {conversion_result.train_images}")
        training_state.add_log(f"   ├─ 验证图像: {conversion_result.val_images}")
        training_state.add_log(f"   └─ 类别: {class_names}")
        
        def on_epoch_end(epoch, metrics):
            training_state.current_epoch = epoch
            training_state.box_losses.append(float(metrics.get('box_loss', 0)))
            training_state.cls_losses.append(float(metrics.get('cls_loss', 0)))
            training_state.map50_list.append(float(metrics.get('map50', 0)))
            training_state.map50_95_list.append(float(metrics.get('map50_95', 0)))
            training_state.best_map50 = float(metrics.get('best_map50', 0))
            training_state.best_map50_95 = float(metrics.get('best_map50_95', 0))
            training_state.best_epoch = int(metrics.get('best_epoch', 0))
            
            # 【增强】保存历史数据到DL-Hub
            if dlhub_params:
                dlhub_params.update_history_epoch({
                    'box_loss': float(metrics.get('box_loss', 0)),
                    'cls_loss': float(metrics.get('cls_loss', 0)),
                    'map50': float(metrics.get('map50', 0)),
                    'map50_95': float(metrics.get('map50_95', 0)),
                    'current_epoch': epoch,
                    'best_map50': float(metrics.get('best_map50', 0)),
                    'best_epoch': int(metrics.get('best_epoch', 0))
                }, auto_save=(epoch % 5 == 0))
        
        def on_log_callback(message):
            training_state.add_log(message)
            # 【增强】追加日志到DL-Hub
            if dlhub_params:
                dlhub_params.append_log(message, 'training', auto_save=False)
        
        callback = TrainingCallback(on_epoch_end=on_epoch_end, on_log=on_log_callback, should_stop=lambda: training_state.should_stop)
        
        trainer = YOLOTrainer(
            model_name=model_name, data_yaml=data_yaml, output_dir=str(output_dir),
            epochs=epochs, batch_size=batch_size, img_size=img_size, learning_rate=learning_rate,
            optimizer=optimizer_name, patience=patience, freeze_backbone=freeze_backbone, device=device,
            workers=4, weight_decay=weight_decay, momentum=momentum, warmup_epochs=warmup_epochs,
            cos_lr=cos_lr, mosaic=mosaic, mixup=mixup, fliplr=fliplr, save_period=save_period,
            callback=callback, class_names=class_names)
        
        training_state.trainer = trainer
        trainer.train()
        
        elapsed = time.time() - training_state.start_time
        training_state.add_log("━" * 50)
        training_state.add_log(f"✅ 训练完成！用时 {elapsed/60:.1f} 分钟")
        training_state.add_log(f"🏆 最佳 mAP@50: {training_state.best_map50:.4f}")
        training_state.add_log("━" * 50)
        
        # 【增强】标记训练完成
        if dlhub_params:
            dlhub_params.mark_training_complete(
                best_metric=training_state.best_map50,
                best_epoch=training_state.best_epoch
            )
    except Exception as e:
        import traceback
        training_state.add_log(f"❌ 错误: {str(e)}")
        training_state.add_log(traceback.format_exc())
    finally:
        training_state.is_training = False
        # 【增强】保存日志到DL-Hub
        if dlhub_params:
            dlhub_params.save_logs(training_state.logs, 'training', auto_save=True)


def on_stop_training() -> str:
    global training_state
    if not training_state.is_training:
        return "⚠️ 当前没有正在进行的训练"
    training_state.should_stop = True
    training_state.add_log("⏹️ 正在停止训练...")
    return "⏹️ 正在停止..."


def on_open_output() -> str:
    global training_state
    if not training_state.output_dir or not os.path.exists(training_state.output_dir):
        return "⚠️ 输出目录不存在"
    try:
        import subprocess
        if sys.platform == 'win32':
            os.startfile(training_state.output_dir)
        elif sys.platform == 'darwin':
            subprocess.run(['open', training_state.output_dir])
        else:
            subprocess.run(['xdg-open', training_state.output_dir])
        return f"✅ 已打开: {training_state.output_dir}"
    except Exception as e:
        return f"⚠️ 无法打开: {e}"


def on_test_inference():
    global training_state
    if not training_state.trainer or not training_state.trainer.yolo:
        return None, "⚠️ 模型未加载"
    try:
        import torch
        from data.dataset import get_val_images
        from data.visualizer import visualize_ultralytics_result, create_preview_grid
        
        val_images = get_val_images(training_state.data_yaml, num_images=4)
        if not val_images:
            return None, "⚠️ 未找到验证集图像"
        
        with torch.no_grad():
            results = training_state.trainer.yolo.predict(source=val_images, conf=0.25, save=False, verbose=False)
        
        preview_images = []
        result_texts = []
        for i, result in enumerate(results):
            vis_img = visualize_ultralytics_result(result, conf_threshold=0.25)
            if vis_img:
                preview_images.append(vis_img)
            num_det = len(result.boxes) if result.boxes is not None else 0
            result_texts.append(f"图{i+1}: {num_det}个检测")
        
        if preview_images:
            grid = preview_images[0] if len(preview_images) == 1 else create_preview_grid(preview_images, grid_size=(2, 2))
            return grid, " | ".join(result_texts)
        return None, "⚠️ 生成预览失败"
    except Exception as e:
        return None, f"⚠️ 推理失败: {str(e)}"


def get_training_status() -> Tuple[str, str, str, Any, Any, Any]:
    global training_state, _cached_figures
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    progress_html = get_progress_html()
    if training_state.is_training:
        status_text = f"🏃 训练中 - Epoch {training_state.current_epoch}/{training_state.total_epochs}"
    elif training_state.current_epoch > 0:
        status_text = f"✅ 训练完成 - 最佳 mAP@50: {training_state.best_map50:.4f}"
    else:
        status_text = "⏳ 等待开始..."
    logs = "\n".join(training_state.logs[-80:]) if training_state.logs else "暂无日志"
    
    # Loss图
    if _cached_figures['fig_loss'] is None:
        _cached_figures['fig_loss'], _cached_figures['ax_loss'] = plt.subplots(figsize=(5, 3.5), dpi=100)
    else:
        _cached_figures['ax_loss'].clear()
    ax_loss = _cached_figures['ax_loss']
    fig_loss = _cached_figures['fig_loss']
    ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss'); ax_loss.set_title('Loss Curve', fontweight='bold'); ax_loss.grid(True, alpha=0.3)
    if training_state.box_losses:
        epochs = list(range(1, len(training_state.box_losses) + 1))
        ax_loss.plot(epochs, training_state.box_losses, 'b-o', label='Box Loss', linewidth=2, markersize=3)
        if training_state.cls_losses:
            ax_loss.plot(epochs[:len(training_state.cls_losses)], training_state.cls_losses, 'r-s', label='Cls Loss', linewidth=2, markersize=3)
        ax_loss.legend(loc='upper right', fontsize=9)
    else:
        ax_loss.text(0.5, 0.5, 'Waiting...', ha='center', va='center', transform=ax_loss.transAxes, color='gray')
        ax_loss.set_xlim(0, 10); ax_loss.set_ylim(0, 1)
    fig_loss.tight_layout()
    
    # mAP图
    if _cached_figures['fig_map'] is None:
        _cached_figures['fig_map'], _cached_figures['ax_map'] = plt.subplots(figsize=(5, 3.5), dpi=100)
    else:
        _cached_figures['ax_map'].clear()
    ax_map = _cached_figures['ax_map']
    fig_map = _cached_figures['fig_map']
    ax_map.set_xlabel('Epoch'); ax_map.set_ylabel('mAP'); ax_map.set_title('mAP Curve', fontweight='bold'); ax_map.grid(True, alpha=0.3)
    if training_state.map50_list:
        epochs = list(range(1, len(training_state.map50_list) + 1))
        ax_map.plot(epochs, training_state.map50_list, 'g-o', label='mAP@50', linewidth=2, markersize=3)
        if training_state.map50_95_list:
            ax_map.plot(epochs[:len(training_state.map50_95_list)], training_state.map50_95_list, 'm-s', label='mAP@50:95', linewidth=2, markersize=3)
        ax_map.legend(loc='lower right', fontsize=9)
        if training_state.best_epoch > 0 and training_state.best_epoch <= len(training_state.map50_list):
            ax_map.scatter([training_state.best_epoch], [training_state.best_map50], color='red', s=100, zorder=5, marker='*')
    else:
        ax_map.text(0.5, 0.5, 'Waiting...', ha='center', va='center', transform=ax_map.transAxes, color='gray')
        ax_map.set_xlim(0, 10); ax_map.set_ylim(0, 1)
    fig_map.tight_layout()
    
    return progress_html, status_text, logs, fig_loss, fig_map, training_state.detection_preview


# ==================== Tab 2: 模型转换相关函数 ====================
def get_output_base_dir() -> Path:
    """获取output基础目录"""
    if dlhub_params and dlhub_params.is_dlhub_mode:
        return dlhub_params.get_output_dir()
    return Path('./output')


# 全局变量：存储文件夹名到完整路径的映射
_folder_path_map = {}
_model_path_map = {}


def scan_model_folders() -> Dict:
    """扫描output目录下的训练文件夹，只显示文件夹名"""
    global _folder_path_map
    _folder_path_map = {}
    
    base_path = get_output_base_dir()
    if not base_path.exists():
        return gr.update(choices=[], value=None)
    
    folders = []
    for item in base_path.iterdir():
        if item.is_dir():
            has_models = any(item.rglob('*.pt')) or any(item.rglob('*.pth'))
            if has_models:
                folder_name = item.name
                _folder_path_map[folder_name] = str(item)
                folders.append(folder_name)
    
    folders = sorted(folders, reverse=True)
    if folders:
        return gr.update(choices=folders, value=folders[0])
    return gr.update(choices=[], value=None)


def scan_models_in_selected_folder(folder_name: str) -> Dict:
    """扫描选定文件夹下的模型文件，只显示相对路径"""
    global _model_path_map
    _model_path_map = {}
    
    if not folder_name:
        return gr.update(choices=[], value=None)
    
    # 从映射获取完整路径
    folder_path = _folder_path_map.get(folder_name)
    if not folder_path:
        return gr.update(choices=[], value=None)
    
    folder = Path(folder_path)
    if not folder.exists():
        return gr.update(choices=[], value=None)
    
    models = []
    models.extend(folder.rglob('*.pt'))
    models.extend(folder.rglob('*.pth'))
    
    model_items = []
    for m in models:
        rel_path = str(m.relative_to(folder))
        _model_path_map[rel_path] = str(m)
        model_items.append(rel_path)
    
    model_items = sorted(model_items, reverse=True)
    if model_items:
        return gr.update(choices=model_items, value=model_items[0])
    return gr.update(choices=[], value=None)


def get_full_model_path(rel_path: str) -> str:
    """根据相对路径获取完整模型路径"""
    if not rel_path:
        return ""
    # 先检查映射
    if rel_path in _model_path_map:
        return _model_path_map[rel_path]
    # 如果是完整路径则直接返回
    if os.path.isabs(rel_path) or os.path.exists(rel_path):
        return rel_path
    return rel_path


def scan_models_in_folder(folder_path: str) -> Dict:
    """扫描文件夹中的模型文件 - 用于推理Tab"""
    global _model_path_map
    _model_path_map = {}
    
    if not folder_path or not folder_path.strip():
        # 默认扫描output目录
        base_path = get_output_base_dir()
        if not base_path.exists():
            return gr.update(choices=[], value=None)
        folder_path = str(base_path)
    else:
        folder_path = folder_path.strip()
    
    if os.path.isfile(folder_path):
        if folder_path.endswith(('.pth', '.pt')):
            file_name = os.path.basename(folder_path)
            _model_path_map[file_name] = folder_path
            return gr.update(choices=[file_name], value=file_name)
        return gr.update(choices=[], value=None)
    
    if not os.path.exists(folder_path):
        return gr.update(choices=[], value=None)
    
    folder = Path(folder_path)
    models = []
    models.extend(folder.rglob('*.pth'))
    models.extend(folder.rglob('*.pt'))
    
    model_items = []
    for m in models:
        try:
            rel_path = str(m.relative_to(folder))
        except ValueError:
            rel_path = m.name
        _model_path_map[rel_path] = str(m)
        model_items.append(rel_path)
    
    model_items = sorted(model_items, reverse=True)
    if model_items:
        return gr.update(choices=model_items, value=model_items[0])
    return gr.update(choices=[], value=None)


def update_backend_options(backend: str) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    is_tensorrt = (backend == "tensorrt")
    return (
        gr.update(visible=is_tensorrt),
        gr.update(visible=is_tensorrt),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=is_tensorrt),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def update_calib_visibility(precision: str) -> Dict:
    needs_calib = precision in ['int8', 'mixed']
    return gr.update(visible=needs_calib)


def validate_calib_data_det(images_dir: str) -> str:
    if not images_dir or not images_dir.strip():
        return "⏳ 请输入校准图像文件夹路径"
    images_dir = images_dir.strip()
    if not os.path.exists(images_dir):
        return f"❌ 路径不存在: {images_dir}"
    if not os.path.isdir(images_dir):
        return "❌ 请输入文件夹路径，不是文件"
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    total_images = 0
    for ext in image_extensions:
        total_images += len(list(Path(images_dir).rglob(f"*{ext}")))
        total_images += len(list(Path(images_dir).rglob(f"*{ext.upper()}")))
    if total_images == 0:
        return "❌ 未找到图片文件"
    if total_images < 100:
        warning = "⚠️ 建议至少100张图片\n"
    else:
        warning = ""
    return f"✅ 验证通过\n{warning}📊 共 {total_images} 张图片"


def get_conversion_logs() -> str:
    return "\n".join(conversion_state.logs[-100:]) if conversion_state.logs else "暂无日志"


def get_conversion_status() -> Tuple[str, str]:
    if conversion_state.is_running:
        elapsed = time.time() - conversion_state.start_time if conversion_state.start_time else 0
        status = f"🔄 转换中... ({format_time(elapsed)})"
    elif conversion_state.output_path:
        status = f"✅ 转换完成: {conversion_state.output_path}"
    else:
        status = "⏳ 等待开始..."
    logs = get_conversion_logs()
    return status, logs


def on_start_conversion(
    model_path: str, target_backend: str, precision: str, device: str,
    workspace_gb: int, dynamic_batch: bool, min_batch: int, opt_batch: int, max_batch: int,
    output_dir: str, calib_images_dir: str = "",
) -> Tuple[str, str]:
    global conversion_state, training_state
    if training_state.is_training:
        return "⚠️ 训练正在进行中，请等待训练完成后再进行转换", ""
    if conversion_state.is_running:
        return "⚠️ 转换正在进行中...", get_conversion_logs()
    if not model_path:
        return "❌ 请选择要转换的模型", ""
    
    # 获取完整模型路径
    full_model_path = get_full_model_path(model_path)
    
    if not os.path.exists(full_model_path):
        return f"❌ 模型文件不存在: {full_model_path}", ""
    
    # 保存转换参数到DL-Hub
    if dlhub_params:
        current_params = dlhub_params.get_all()
        current_params['conversion'] = {
            'target_backend': target_backend,
            'precision': precision,
            'device': device,
            'workspace_gb': workspace_gb,
            'dynamic_batch': dynamic_batch,
            'min_batch': min_batch,
            'opt_batch': opt_batch,
            'max_batch': max_batch,
            'output_dir': output_dir,
        }
        dlhub_params.save(current_params)
    
    if precision in ['int8', 'mixed']:
        if not calib_images_dir or not calib_images_dir.strip():
            return "❌ INT8/Mixed精度需要提供校准图像文件夹路径", ""
        calib_images_dir = calib_images_dir.strip()
        if not os.path.exists(calib_images_dir):
            return f"❌ 校准图像路径不存在: {calib_images_dir}", ""
        validation_result = validate_calib_data_det(calib_images_dir)
        if not validation_result.startswith("✅"):
            return f"❌ 校准数据集验证失败: {validation_result}", ""
    conversion_state.reset()
    conversion_state.is_running = True
    conversion_state.start_time = time.time()
    if not output_dir:
        output_dir = str(Path(full_model_path).parent.parent / "converted")
    thread = threading.Thread(target=run_conversion, args=(
        full_model_path, target_backend, precision, device,
        workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch,
        output_dir, calib_images_dir,
    ), daemon=True)
    thread.start()
    return "🚀 转换已启动...", ""


def run_conversion(
    model_path: str,
    target_backend: str,
    precision: str,
    device: str,
    workspace_gb: int,
    dynamic_batch: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    output_dir: str,
    calib_images_dir: str = "",
):
    """
    后台转换线程 - 优化版 V2
    
    优化改进：
    1. 不同后端输出到不同子目录 (tensorrt/openvino/onnxruntime)
    2. 每次转换创建日期时间子目录
    3. 始终使用独立子进程导出ONNX，完全避免进程污染
    4. 移除ONNX复用逻辑，每次都重新导出确保参数一致
    5. 完善的资源清理机制（CUDA + 内存）
    6. 禁用onnxsim简化避免卡死
    """
    global conversion_state
    
    # ========== 步骤0：准备工作 ==========
    ctx = None
    conversion_tool_path = str(Path(__file__).parent.parent / "model_conversion")
    
    # ========== 统一输出路径结构 ==========
    # 标准化后端名称用于目录
    backend_dir_map = {
        'tensorrt': 'tensorrt',
        'trt': 'tensorrt',
        'openvino': 'openvino',
        'ov': 'openvino',
        'ort': 'onnxruntime',
        'onnxruntime': 'onnxruntime',
    }
    backend_subdir = backend_dir_map.get(target_backend.lower(), target_backend.lower())
    
    # 生成日期时间子目录
    from datetime import datetime
    date_subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 修改输出目录：base_output_dir/后端/日期时间/
    base_output_dir = output_dir
    output_dir = str(Path(base_output_dir) / backend_subdir / date_subdir)
    
    try:
        # ========== 关键修复1：彻底清理CUDA和内存 ==========
        conversion_state.add_log("🧹 清理资源...")
        print("[DEBUG] 开始资源清理...", flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                # 同步所有CUDA流
                torch.cuda.synchronize()
                # 清空缓存
                torch.cuda.empty_cache()
                # 重置峰值内存统计
                torch.cuda.reset_peak_memory_stats()
                print("[DEBUG] CUDA 缓存已清理", flush=True)
            
            # 强制垃圾回收
            import gc
            gc.collect()
            gc.collect()  # 调用两次确保清理
            print("[DEBUG] 垃圾回收完成", flush=True)
            
            # ============ 关键修复：转换前清理 Matplotlib ============
            import matplotlib.pyplot as plt
            plt.close('all')
            print("[DEBUG] Matplotlib figure 已清理", flush=True)
            # =========================================================
            
            # ============ 关键修复：清理 Ultralytics YOLO 缓存 ============
            try:
                # 清理 ultralytics 的内部缓存
                if 'ultralytics' in sys.modules:
                    # 尝试清理 YOLO 的 hub 缓存
                    try:
                        from ultralytics.utils import SETTINGS
                        # 重置一些可能导致问题的设置
                    except Exception:
                        pass
                print("[DEBUG] Ultralytics 缓存检查完成", flush=True)
            except Exception as e:
                print(f"[DEBUG] Ultralytics 清理警告: {e}", flush=True)
            # ===========================================================
            
        except Exception as e:
            conversion_state.add_log(f"⚠️ 初始清理警告: {e}")
            print(f"[DEBUG] 清理警告: {e}", flush=True)
        
        # ========== 关键修复2：彻底卸载所有转换相关模块 ==========
        modules_to_remove = []
        for mod_name in list(sys.modules.keys()):
            # 卸载所有转换相关模块
            if any(target in mod_name for target in [
                # 转换工具模块
                'model_importer', 'model_analyzer', 'model_optimizer',
                'model_exporter', 'model_converter', 'config_generator',
                'converter_tensorrt', 'converter_openvino', 'converter_ort',
                'conversion_validator', 'unified_logger', 'cuda_utils',
                'symbolic', 'constants', 'exceptions', 'config_templates',
                # ONNX 相关模块
                'onnxsim', 'onnxoptimizer', 'onnxconverter_common',
                # 注意：不要卸载 'main' 因为太通用了
            ]):
                # 排除系统模块
                if not mod_name.startswith('_') and 'site-packages' not in str(sys.modules.get(mod_name, '')):
                    modules_to_remove.append(mod_name)
        
        # 特别处理：卸载转换工具的 main 模块（精确匹配）
        if 'main' in sys.modules:
            main_mod = sys.modules.get('main')
            if main_mod and hasattr(main_mod, 'PipelineContext'):
                modules_to_remove.append('main')
        
        # ============ 关键修复：清理 OpenVINO 相关模块 ============
        # OpenVINO 在连续转换时可能导致问题
        openvino_modules = [mod for mod in sys.modules.keys() if 'openvino' in mod.lower()]
        print(f"[DEBUG] 发现 {len(openvino_modules)} 个 OpenVINO 模块需要清理", flush=True)
        modules_to_remove.extend(openvino_modules)
        # =========================================================
        
        for mod_name in modules_to_remove:
            try:
                del sys.modules[mod_name]
            except Exception:
                pass
        
        print(f"[DEBUG] 已清理 {len(modules_to_remove)} 个模块", flush=True)
        
        # 添加转换工具路径
        if conversion_tool_path not in sys.path:
            sys.path.insert(0, conversion_tool_path)
        
        # 重新导入（这会加载全新的模块实例）
        from main import (
            PipelineContext, 
            run_stage1_import, 
            run_stage2_analyze,
            run_stage3_optimize,
            run_stage4_export,
            run_stage5_convert,
            run_stage8_generate_config,
        )
        
        conversion_state.add_log("━" * 50)
        conversion_state.add_log("🚀 开始目标检测模型转换")
        conversion_state.add_log("━" * 50)
        conversion_state.add_log(f"📂 模型: {model_path}")
        conversion_state.add_log(f"🎯 目标: {target_backend} ({precision})")
        conversion_state.add_log(f"📁 输出: {output_dir}")
        
        # 校准数据路径（由用户提供）
        if precision in ['int8', 'mixed']:
            if calib_images_dir:
                conversion_state.add_log(f"📊 校准数据: {calib_images_dir}")
            else:
                conversion_state.add_log("❌ INT8/Mixed精度需要校准数据，但未提供")
                return
        
        # 创建上下文
        ctx = PipelineContext()
        ctx.target_backend = target_backend
        ctx.precision = precision
        ctx.device = 'cuda' if 'GPU' in device else 'cpu'
        
        # Stage 1: 导入模型
        conversion_state.add_log("\n📥 Stage 1: 导入模型...")
        if not run_stage1_import(ctx, model_path, 'det', device=ctx.device):
            conversion_state.add_log("❌ 模型导入失败")
            return
        conversion_state.add_log(f"   ✅ 模型: {ctx.model_name}")
        conversion_state.add_log(f"   ✅ 输入形状: {ctx.input_shape}")
        
        # Stage 2: 分析模型
        conversion_state.add_log("\n🔍 Stage 2: 分析模型...")
        if not run_stage2_analyze(ctx, [target_backend]):
            conversion_state.add_log("❌ 模型分析失败")
            return
        conversion_state.add_log("   ✅ 分析完成")
        
        # Stage 3: 优化模型
        conversion_state.add_log("\n⚡ Stage 3: 优化模型...")
        if not run_stage3_optimize(ctx):
            conversion_state.add_log("⚠️ 优化跳过，使用原始模型")
        else:
            conversion_state.add_log("   ✅ 优化完成")
        
        # Stage 4: 导出ONNX
        conversion_state.add_log("\n📦 Stage 4: 导出 ONNX...")
        os.makedirs(output_dir, exist_ok=True)
        
        # ONNX文件名简化（日期已在目录路径中）
        onnx_filename = f"{ctx.model_name}.onnx"
        onnx_path = str(Path(output_dir) / onnx_filename)
        
        # 所有后端都可以使用动态batch，根据用户选择决定
        onnx_dynamic_batch = dynamic_batch
        
        if onnx_dynamic_batch:
            if target_backend == 'tensorrt':
                conversion_state.add_log(f"   启用动态Batch: min={min_batch}, opt={opt_batch}, max={max_batch}")
            else:
                conversion_state.add_log(f"   启用动态Batch: min={min_batch}, max={max_batch}")
        
        conversion_state.add_log(f"   ONNX输出: {os.path.basename(onnx_path)}")
        
        # ========== 优化：始终使用子进程导出 YOLO 模型 ==========
        # 使用独立子进程完全隔离每次导出，彻底避免状态污染
        export_success = False
        
        if ctx.framework == 'ultralytics':
            conversion_state.add_log("   使用独立子进程导出 YOLO 模型...")
            print("[DEBUG] 使用独立子进程导出 YOLO 模型...", flush=True)
            
            try:
                import subprocess
                import tempfile
                
                # 准备配置文件
                export_config = {
                    'model_path': ctx.model_path,
                    'output_path': onnx_path,
                    'opset': 17,
                    'dynamic_batch': onnx_dynamic_batch,
                    'simplify': False
                }
                    
                # 写入临时配置文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(export_config, f)
                    config_path = f.name
                
                # 查找 yolo_export_worker.py
                worker_script = os.path.join(conversion_tool_path, 'yolo_export_worker.py')
                if not os.path.exists(worker_script):
                    # 尝试其他可能的位置
                    possible_paths = [
                        os.path.join(os.path.dirname(model_path), '..', '..', 'model_conversion', 'yolo_export_worker.py'),
                        os.path.join(os.path.dirname(__file__), 'yolo_export_worker.py'),
                    ]
                    for p in possible_paths:
                        if os.path.exists(p):
                            worker_script = p
                            break
                
                if os.path.exists(worker_script):
                    print(f"[DEBUG] 调用独立子进程: {worker_script}", flush=True)
                    
                    # 设置环境变量确保子进程使用UTF-8编码输出
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    
                    # 调用子进程 - 完全隔离的新进程
                    result = subprocess.run(
                        [sys.executable, worker_script, config_path],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',  # 修复Windows中文编码问题
                        errors='replace',  # 遇到无法解码的字符用?替换
                        timeout=300,  # 5 分钟超时
                        env=env,  # 使用修改后的环境变量
                    )
                    
                    # 输出子进程日志
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                print(f"  {line}", flush=True)
                    
                    if result.stderr:
                        for line in result.stderr.strip().split('\n'):
                            if line:
                                print(f"  [STDERR] {line}", flush=True)
                    
                    # 读取结果
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
                                pass
                            export_success = True
                            conversion_state.add_log(f"   ✅ 子进程导出成功")
                        else:
                            conversion_state.add_log(f"   ❌ 子进程导出失败: {export_result.get('message', '未知错误')}")
                    else:
                        conversion_state.add_log(f"   ❌ 子进程未返回结果")
                    
                    # 清理临时文件
                    try:
                        os.unlink(config_path)
                        if os.path.exists(result_path):
                            os.unlink(result_path)
                    except Exception:
                        pass
                else:
                    conversion_state.add_log("   ❌ 未找到子进程脚本 yolo_export_worker.py")
                    conversion_state.add_log("❌ ONNX导出失败")
                    return
                    
            except subprocess.TimeoutExpired:
                conversion_state.add_log("   ❌ 子进程导出超时（5分钟）")
                conversion_state.add_log("❌ ONNX导出失败")
                return
            except Exception as e:
                conversion_state.add_log(f"   ❌ 子进程导出异常: {e}")
                print(f"[DEBUG] 子进程导出异常: {e}", flush=True)
                conversion_state.add_log("❌ ONNX导出失败")
                return
            
            if not export_success:
                conversion_state.add_log("❌ ONNX导出失败")
                return
            
            conversion_state.add_log(f"   ✅ ONNX: {ctx.onnx_path}")
        
        else:
            # 非 YOLO 模型使用进程内导出
            conversion_state.add_log("   使用进程内导出（非YOLO模型）...")
            if not run_stage4_export(
                ctx, 
                output_path=onnx_path,
                opset=17,
                enable_dynamic_batch=onnx_dynamic_batch,
                enable_dynamic_hw=False,
                enable_simplify=False,
            ):
                conversion_state.add_log("❌ ONNX导出失败")
                return
            
            conversion_state.add_log(f"   ✅ ONNX: {ctx.onnx_path}")
        
        # Stage 5: 转换到目标后端
        conversion_state.add_log(f"\n🔧 Stage 5: 转换到 {target_backend}...")
        
        # 后端特定参数
        backend_kwargs = {}
        
        if target_backend == 'tensorrt':
            backend_kwargs = {
                'trt_workspace_gb': workspace_gb,
                'trt_dynamic_batch_enabled': dynamic_batch,
                'trt_min_batch': min_batch,
                'trt_opt_batch': opt_batch,
                'trt_max_batch': max_batch,
            }
            conversion_state.add_log(f"   TensorRT 配置:")
            conversion_state.add_log(f"     - Workspace: {workspace_gb} GB")
            conversion_state.add_log(f"     - Dynamic Batch: {dynamic_batch}")
            if dynamic_batch:
                conversion_state.add_log(f"     - Batch Range: [{min_batch}, {opt_batch}, {max_batch}]")
        
        elif target_backend == 'openvino':
            backend_kwargs = {
                'ov_dynamic_batch_enabled': dynamic_batch,
                'ov_min_batch': min_batch,
                'ov_max_batch': max_batch,
            }
            if dynamic_batch:
                conversion_state.add_log(f"   OpenVINO 配置:")
                conversion_state.add_log(f"     - Dynamic Batch: {dynamic_batch}")
                conversion_state.add_log(f"     - Batch Range: [{min_batch}, {max_batch}]")
        
        elif target_backend in ['ort', 'onnxruntime']:
            backend_kwargs = {
                'ort_dynamic_batch_enabled': dynamic_batch,
                'ort_min_batch': min_batch,
                'ort_max_batch': max_batch,
            }
            if dynamic_batch:
                conversion_state.add_log(f"   ONNX Runtime 配置:")
                conversion_state.add_log(f"     - Dynamic Batch: {dynamic_batch} (ONNX已包含动态轴，ORT自动支持)")
        
        if not run_stage5_convert(
            ctx,
            output_dir=output_dir,
            target_backend=target_backend,
            precision=precision,
            calib_data_path=calib_images_dir,
            enable_validation=False,  # 禁用验证，避免 OpenVINO 导入导致崩溃
            **backend_kwargs,
        ):
            conversion_state.add_log("❌ 转换失败")
            return
        
        conversion_state.add_log(f"   ✅ 转换完成")
        
        # Stage 8: 生成配置
        conversion_state.add_log("\n📝 Stage 8: 生成配置...")
        run_stage8_generate_config(ctx, output_dir)
        conversion_state.add_log("   ✅ 配置已生成")
        
        # Stage 9: 打包为 .dlhub
        conversion_state.add_log("\n📦 Stage 9: 打包 .dlhub...")
        try:
            conversion_tool_path_pack = str(Path(__file__).parent.parent / "model_conversion")
            if conversion_tool_path_pack not in sys.path:
                sys.path.insert(0, conversion_tool_path_pack)
            from dlhub_packager import DLHubPackager
            packager = DLHubPackager()
            dlhub_path = packager.pack(
                output_dir=output_dir,
                task_type='det',
                backend=target_backend,
                precision=precision,
            )
            if dlhub_path:
                conversion_state.add_log(f"   ✅ 已打包: {os.path.basename(dlhub_path)}")
            else:
                conversion_state.add_log("   ⚠️ 打包跳过（无模型文件或打包失败）")
        except Exception as pack_err:
            conversion_state.add_log(f"   ⚠️ 打包失败（非致命）: {pack_err}")
        
        # 完成
        conversion_state.output_path = output_dir
        elapsed = time.time() - conversion_state.start_time
        conversion_state.add_log("\n" + "━" * 50)
        conversion_state.add_log(f"✅ 转换完成！用时 {elapsed:.1f} 秒")
        conversion_state.add_log(f"📁 输出目录: {output_dir}")
        conversion_state.add_log("━" * 50)
        
    except Exception as e:
        import traceback
        conversion_state.add_log(f"\n❌ 错误: {str(e)}")
        conversion_state.add_log(traceback.format_exc())
        
    finally:
        conversion_state.is_running = False
        
        # ========== 完善的资源清理 ==========
        try:
            # 1. 清理 Pipeline 上下文中的模型引用
            if ctx is not None:
                ctx.model = None
                ctx.optimized_model = None
                ctx.onnx_model = None
                ctx.analysis_report = None
                ctx.conversion_result = None
        except Exception:
            pass
        
        try:
            # 2. 清理 PyTorch CUDA 缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        
        try:
            # 3. 清理 PyCUDA 上下文
            if conversion_tool_path in sys.path:
                from cuda_utils import cleanup_cuda_context
                cleanup_cuda_context()
        except Exception:
            pass
        
        try:
            # 4. 强制垃圾回收
            import gc
            gc.collect()
        except Exception:
            pass
        
        try:
            # 5. 清理 ONNX Runtime 会话（如果有）
            import onnxruntime as ort
            ort.get_available_providers()  # 触发清理
        except Exception:
            pass
        
        # ============ 关键修复：清理 Matplotlib 资源 ============
        try:
            import matplotlib.pyplot as plt
            # 关闭所有打开的 figure，释放内存
            plt.close('all')
            
            # 重置 UI 图表缓存
            global _training_ui_figures
            _training_ui_figures = {
                'fig_loss': None,
                'ax_loss': None,
                'fig_map': None,
                'ax_map': None
            }
        except Exception as e:
            print(f"Matplotlib 清理警告: {e}")
        # =========================================================
        
        # 6. 保存转换日志到DL-Hub
        try:
            if dlhub_params:
                dlhub_params.save_logs(conversion_state.logs, 'conversion', auto_save=True)
        except Exception:
            pass


# ==================== Tab 3: 批量推理相关函数 ====================
def get_inference_logs() -> str:
    return "\n".join(inference_state.logs[-100:]) if inference_state.logs else "暂无日志"


def get_inference_status() -> Tuple[str, str]:
    if inference_state.is_running:
        elapsed = time.time() - inference_state.start_time if inference_state.start_time else 0
        progress = f"{inference_state.processed_images}/{inference_state.total_images}"
        status = f"🔄 推理中... {progress} ({format_time(elapsed)})"
    elif inference_state.output_dir:
        status = f"✅ 推理完成: {len(inference_state.results)} 张图片"
    else:
        status = "⏳ 等待开始..."
    logs = get_inference_logs()
    return status, logs


def get_inference_status_with_preview() -> Tuple[str, str, Any, str]:
    """获取推理状态并自动刷新预览（推理完成时）"""
    status, logs = get_inference_status()
    
    # 获取预览图像
    preview_img, preview_info = None, "暂无预览图像"
    
    # 只有在推理完成且有预览图像时才更新预览
    if not inference_state.is_running and inference_state.preview_images:
        preview_img, preview_info = get_current_preview()
    
    return status, logs, preview_img, preview_info


def on_start_inference(
    model_path: str, input_path: str, output_dir: str, device: str,
    conf_threshold: float, save_images: bool, save_json: bool,
) -> Tuple[str, str]:
    global inference_state, training_state, conversion_state
    if training_state.is_training:
        return "⚠️ 训练正在进行中，请等待完成", ""
    if conversion_state.is_running:
        return "⚠️ 模型转换正在进行中，请等待完成", ""
    if inference_state.is_running:
        return "⚠️ 推理正在进行中...", get_inference_logs()
    if not model_path:
        return "❌ 请选择模型", ""
    
    # 获取完整模型路径
    full_model_path = get_full_model_path(model_path)
    
    if not os.path.exists(full_model_path):
        return f"❌ 模型文件不存在: {full_model_path}", ""
    if not input_path or not os.path.exists(input_path):
        return f"❌ 输入路径不存在: {input_path}", ""
    
    # 保存推理参数到DL-Hub
    if dlhub_params:
        current_params = dlhub_params.get_all()
        current_params['inference'] = {
            'input_path': input_path,
            'output_dir': output_dir,
            'device': device,
            'conf_threshold': conf_threshold,
            'save_images': save_images,
            'save_json': save_json,
        }
        dlhub_params.save(current_params)
    
    inference_state.reset()
    inference_state.is_running = True
    inference_state.start_time = time.time()
    if not output_dir:
        output_dir = "./inference_output"
    thread = threading.Thread(target=run_inference, args=(full_model_path, input_path, output_dir, device, conf_threshold, save_images, save_json), daemon=True)
    thread.start()
    return "🚀 推理已启动...", ""


def run_inference(model_path: str, input_path: str, output_dir: str, device: str, conf_threshold: float, save_images: bool, save_json: bool):
    global inference_state
    try:
        import torch
        import cv2
        from ultralytics import YOLO
        inference_state.add_log("━" * 50)
        inference_state.add_log("🚀 开始批量推理")
        inference_state.add_log("━" * 50)
        use_cuda = 'GPU' in device and torch.cuda.is_available()
        dev = '0' if use_cuda else 'cpu'
        inference_state.add_log(f"🖥️ 设备: {'GPU' if use_cuda else 'CPU'}")
        inference_state.add_log(f"📂 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        class_names = []
        if 'names' in checkpoint:
            names = checkpoint['names']
            if isinstance(names, dict):
                class_names = [names.get(i, str(i)) for i in range(len(names))]
            elif isinstance(names, list):
                class_names = names
        if 'custom_metadata' in checkpoint:
            meta = checkpoint['custom_metadata']
            if 'class_names' in meta and meta['class_names']:
                class_names = meta['class_names']
        inference_state.add_log(f"   类别: {class_names if class_names else '未知'}")
        model = YOLO(model_path)
        inference_state.add_log("   ✅ 模型加载成功")
        inference_state.add_log(f"\n📂 扫描图片: {input_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        images = []
        input_path = Path(input_path)
        if input_path.is_file():
            if input_path.suffix.lower() in image_extensions:
                images = [input_path]
        else:
            for ext in image_extensions:
                images.extend(input_path.rglob(f"*{ext}"))
                images.extend(input_path.rglob(f"*{ext.upper()}"))
        images = sorted(list(set(images)))
        inference_state.total_images = len(images)
        inference_state.add_log(f"   找到 {len(images)} 张图片")
        if not images:
            inference_state.add_log("❌ 未找到图片")
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_out_dir = output_dir / "images"
        jsons_out_dir = output_dir / "jsons"
        if save_images:
            images_out_dir.mkdir(exist_ok=True)
        if save_json:
            jsons_out_dir.mkdir(exist_ok=True)
        inference_state.add_log("\n🔍 开始推理...")
        results_list = []
        preview_images = []
        for i, img_path in enumerate(images):
            try:
                results = model.predict(source=str(img_path), conf=conf_threshold, device=dev, save=False, verbose=False)
                if not results:
                    continue
                result = results[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    for j in range(len(boxes)):
                        box = boxes.xyxy[j].cpu().numpy()
                        conf = float(boxes.conf[j].cpu().numpy())
                        cls_id = int(boxes.cls[j].cpu().numpy())
                        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                        detections.append({'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])], 'confidence': conf, 'class_id': cls_id, 'class_name': cls_name})
                results_list.append({'image': str(img_path), 'detections': detections, 'num_detections': len(detections)})
                if save_images:
                    annotated_img = result.plot()
                    out_img_path = images_out_dir / img_path.name
                    cv2.imwrite(str(out_img_path), annotated_img)
                    preview_images.append(str(out_img_path))
                if save_json:
                    json_result = {'image': img_path.name, 'image_path': str(img_path), 'detections': detections}
                    json_path = jsons_out_dir / f"{img_path.stem}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_result, f, indent=2, ensure_ascii=False)
                inference_state.processed_images = i + 1
                if (i + 1) % 10 == 0 or (i + 1) == len(images):
                    inference_state.add_log(f"   进度: {i+1}/{len(images)}")
            except Exception as e:
                inference_state.add_log(f"   ⚠️ 跳过 {img_path.name}: {str(e)}")
        inference_state.results = results_list
        inference_state.output_dir = str(output_dir)
        inference_state.preview_images = preview_images
        inference_state.current_preview_index = 0
        summary_path = output_dir / "detection_results.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({'total_images': len(images), 'total_detections': sum(r['num_detections'] for r in results_list), 'results': results_list}, f, indent=2, ensure_ascii=False)
        inference_state.add_log(f"\n📊 结果已保存: {summary_path}")
        total_detections = sum(r['num_detections'] for r in results_list)
        inference_state.add_log(f"\n📈 检测统计:")
        inference_state.add_log(f"   总图片数: {len(images)}")
        inference_state.add_log(f"   总检测数: {total_detections}")
        inference_state.add_log(f"   平均每张: {total_detections/len(images):.1f}")
        elapsed = time.time() - inference_state.start_time
        inference_state.add_log("\n" + "━" * 50)
        inference_state.add_log(f"✅ 推理完成！处理 {len(results_list)} 张图片，用时 {elapsed:.1f} 秒")
        inference_state.add_log("━" * 50)
    except Exception as e:
        import traceback
        inference_state.add_log(f"\n❌ 错误: {str(e)}")
        inference_state.add_log(traceback.format_exc())
    finally:
        inference_state.is_running = False
        
        # 保存推理日志到DL-Hub
        try:
            if dlhub_params:
                dlhub_params.save_logs(inference_state.logs, 'inference', auto_save=True)
        except Exception:
            pass


def get_current_preview() -> Tuple[Any, str]:
    if not inference_state.preview_images:
        return None, "暂无预览图像"
    idx = inference_state.current_preview_index
    total = len(inference_state.preview_images)
    if idx < 0 or idx >= total:
        idx = 0
        inference_state.current_preview_index = 0
    img_path = inference_state.preview_images[idx]
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, f"图片 {idx + 1}/{total}: {Path(img_path).name}"
    except Exception:
        pass
    return None, f"无法加载图片: {img_path}"


def on_prev_preview() -> Tuple[Any, str]:
    if inference_state.preview_images:
        inference_state.current_preview_index -= 1
        if inference_state.current_preview_index < 0:
            inference_state.current_preview_index = len(inference_state.preview_images) - 1
    return get_current_preview()


def on_next_preview() -> Tuple[Any, str]:
    if inference_state.preview_images:
        inference_state.current_preview_index += 1
        if inference_state.current_preview_index >= len(inference_state.preview_images):
            inference_state.current_preview_index = 0
    return get_current_preview()


def on_refresh_preview() -> Tuple[Any, str]:
    return get_current_preview()



# ==================== 辅助函数 ====================
def get_model_families():
    """获取模型系列列表"""
    try:
        from config.model_registry import get_all_families
        return get_all_families()
    except Exception:
        return ["YOLOv8"]


def get_model_scales_for_family(family: str):
    """获取指定系列的模型规模列表"""
    try:
        from config.model_registry import get_all_scales
        return get_all_scales()
    except Exception:
        return ["超小", "小", "中", "大", "超大"]


def update_model_scales(family: str):
    """更新模型规模下拉框"""
    scales = get_model_scales_for_family(family)
    return gr.update(choices=scales, value=scales[0] if scales else None)


def update_model_info_display(family: str, scale: str):
    """更新模型信息和权重状态"""
    info = update_model_info(family, scale)
    weights = check_pretrained_weights(family, scale)
    return info, weights


def make_progress_html(current_epoch, total_epochs, progress_pct, box_loss, cls_loss, elapsed_str):
    """创建进度条HTML"""
    return f"""
    <div style="padding: 15px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #fff; font-weight: bold;">训练进度</span>
            <span style="color: #8be9fd;">Epoch {current_epoch}/{total_epochs}</span>
        </div>
        <div style="background: #2d2d44; border-radius: 8px; height: 24px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {progress_pct:.1f}%; transition: width 0.3s;"></div>
        </div>
        <div style="display: flex; justify-content: space-around; margin-top: 15px;">
            <div style="text-align: center;">
                <div style="color: #8be9fd; font-size: 1.5em; font-weight: bold;">{box_loss:.4f}</div>
                <div style="color: #888; font-size: 0.9em;">Box Loss</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #50fa7b; font-size: 1.5em; font-weight: bold;">{cls_loss:.4f}</div>
                <div style="color: #888; font-size: 0.9em;">Cls Loss</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #ff79c6; font-size: 1.5em; font-weight: bold;">{elapsed_str or '--'}</div>
                <div style="color: #888; font-size: 0.9em;">用时</div>
            </div>
        </div>
    </div>
    """


def run_test_inference(test_img_path):
    """执行测试推理"""
    global training_state
    if not test_img_path:
        return None
    
    if not training_state.output_dir:
        return None
    
    try:
        from ultralytics import YOLO
        import cv2
        
        # 查找最佳模型
        model_path = Path(training_state.output_dir) / "weights_model_importer" / "best_model.pt"
        if not model_path.exists():
            model_path = Path(training_state.output_dir) / "weights" / "best.pt"
        
        if not model_path.exists():
            return None
        
        model = YOLO(str(model_path))
        results = model.predict(test_img_path, conf=0.25)
        
        if results and len(results) > 0:
            annotated = results[0].plot()
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        print(f"测试推理失败: {e}")
        return None


def refresh_model_list():
    """刷新模型列表"""
    models = get_trained_models()
    return gr.update(choices=models, value=models[0] if models else None)


def on_start_training_simple(
    images_dir, jsons_dir, model_family, model_scale,
    epochs, batch_size, learning_rate, optimizer_name,
    img_size, num_classes, val_split, include_negative, save_period, gpu_choice
):
    """简化的训练启动函数（使用默认高级参数，但从保存的参数恢复增强配置）"""
    # 从保存的参数读取增强配置
    saved_aug = {} if not dlhub_params else dlhub_params.get_section('augmentation')
    
    return on_start_training(
        images_dir, jsons_dir, model_family, model_scale,
        epochs, batch_size, learning_rate, optimizer_name,
        img_size, 
        patience=50,           # 默认早停轮数
        freeze_backbone=False, # 不冻结backbone
        val_split=float(val_split),  # 使用传入的验证集比例
        save_period=save_period,
        include_negative=include_negative,
        num_classes=int(num_classes) if num_classes else 0,
        gpu_choice=gpu_choice,
        weight_decay=0.0005,   # 默认权重衰减
        momentum=0.937,        # 默认动量
        warmup_epochs=3.0,     # 默认warmup
        cos_lr=True,           # 使用余弦学习率
        mosaic=saved_aug.get('mosaic', 1.0),    # 从保存参数恢复，默认1.0
        mixup=saved_aug.get('mixup', 0.0),      # 从保存参数恢复，默认0.0
        fliplr=saved_aug.get('fliplr', 0.5)     # 从保存参数恢复，默认0.5
    )


def get_training_status_ui():
    """获取训练状态（用于UI刷新）- 修复版本
    
    修复内容：
    1. 使用缓存机制重用 figure，避免内存泄漏
    2. 移除重复的 matplotlib.use('Agg') 调用（已在文件开头设置）
    3. 添加异常处理确保稳定性
    """
    import matplotlib.pyplot as plt
    
    global training_state, _training_ui_figures
    
    try:
        # 状态文本
        if training_state.is_training:
            status = f"🔄 训练中... Epoch {training_state.current_epoch}/{training_state.total_epochs}"
        elif training_state.output_dir:
            status = f"✅ 训练完成 - {training_state.output_dir}"
        else:
            status = "⏳ 等待开始训练"
        
        # 进度HTML
        progress_pct = (training_state.current_epoch / training_state.total_epochs * 100) if training_state.total_epochs > 0 else 0
        box_loss = training_state.box_losses[-1] if training_state.box_losses else 0
        cls_loss = training_state.cls_losses[-1] if training_state.cls_losses else 0
        elapsed = time.time() - training_state.start_time if training_state.start_time else 0
        elapsed_str = format_time(elapsed) if elapsed > 0 else None
        
        progress_html = make_progress_html(
            training_state.current_epoch, training_state.total_epochs,
            progress_pct, box_loss, cls_loss, elapsed_str
        )
        
        # ============ 关键修复：使用缓存机制避免重复创建 figure ============
        # 损失曲线
        if _training_ui_figures['fig_loss'] is None:
            _training_ui_figures['fig_loss'], _training_ui_figures['ax_loss'] = plt.subplots(figsize=(6, 4))
        else:
            _training_ui_figures['ax_loss'].clear()
        
        ax_loss = _training_ui_figures['ax_loss']
        fig_loss = _training_ui_figures['fig_loss']
        
        if training_state.box_losses:
            epochs = range(1, len(training_state.box_losses) + 1)
            ax_loss.plot(epochs, training_state.box_losses, 'b-', label='Box Loss')
            if training_state.cls_losses:
                ax_loss.plot(epochs[:len(training_state.cls_losses)], 
                           training_state.cls_losses, 'r-', label='Cls Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
        else:
            ax_loss.text(0.5, 0.5, 'Waiting...', ha='center', va='center', 
                        transform=ax_loss.transAxes, fontsize=11, color='gray')
        ax_loss.set_title('Training Loss')
        fig_loss.tight_layout()
        
        # mAP曲线
        if _training_ui_figures['fig_map'] is None:
            _training_ui_figures['fig_map'], _training_ui_figures['ax_map'] = plt.subplots(figsize=(6, 4))
        else:
            _training_ui_figures['ax_map'].clear()
        
        ax_map = _training_ui_figures['ax_map']
        fig_map = _training_ui_figures['fig_map']
        
        if training_state.map50_list:
            epochs = range(1, len(training_state.map50_list) + 1)
            ax_map.plot(epochs, training_state.map50_list, 'g-', label='mAP@50')
            if training_state.map50_95_list:
                ax_map.plot(epochs[:len(training_state.map50_95_list)], 
                           training_state.map50_95_list, 'm-', label='mAP@50:95')
            ax_map.set_xlabel('Epoch')
            ax_map.set_ylabel('mAP')
            ax_map.legend()
            ax_map.grid(True, alpha=0.3)
        else:
            ax_map.text(0.5, 0.5, 'Waiting...', ha='center', va='center',
                       transform=ax_map.transAxes, fontsize=11, color='gray')
        ax_map.set_title('Validation mAP')
        fig_map.tight_layout()
        # ==================================================================
        
        # 日志
        logs = '\n'.join(training_state.logs[-100:])
        
        return progress_html, fig_loss, fig_map, status, logs
        
    except Exception as e:
        # 异常处理：返回空白图表和错误信息，避免整个UI崩溃
        import traceback
        error_msg = f"图表更新失败: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        
        # 尝试创建空白图表
        try:
            fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
            ax_loss.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')
            fig_map, ax_map = plt.subplots(figsize=(6, 4))
            ax_map.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')
        except Exception:
            fig_loss = None
            fig_map = None
        
        return "", fig_loss, fig_map, f"⚠️ UI更新错误: {str(e)}", ""


# ==================== UI 构建 ====================
def create_ui():
    """创建并返回Gradio应用"""
    global training_state
    
    # 初始化数据
    env_status = check_environment()
    gpu_options = get_gpu_options()
    trained_models = get_trained_models()
    families = get_model_families()
    default_scales = get_model_scales_for_family(families[0]) if families else []
    
    # 加载保存的参数
    saved_data = {} if not dlhub_params else dlhub_params.get_section('data')
    saved_model = {} if not dlhub_params else dlhub_params.get_section('model')
    saved_training = {} if not dlhub_params else dlhub_params.get_section('training')
    saved_conversion = {} if not dlhub_params else dlhub_params.get_section('conversion')
    saved_inference = {} if not dlhub_params else dlhub_params.get_section('inference')
    saved_aug = {} if not dlhub_params else dlhub_params.get_section('augmentation')
    saved_device = get_saved_param('device', gpu_options[0] if gpu_options else 'CPU')
    
    # 【增强】加载训练历史数据
    saved_history = {} if not dlhub_params else dlhub_params.get_history()
    saved_logs = [] if not dlhub_params else dlhub_params.get_logs('training')
    saved_conv_logs = [] if not dlhub_params else dlhub_params.get_logs('conversion')
    saved_inf_logs = [] if not dlhub_params else dlhub_params.get_logs('inference')
    
    # 【增强】恢复历史数据到training_state
    if saved_history:
        if saved_history.get('box_losses') or saved_history.get('map50_list'):
            training_state.box_losses = saved_history.get('box_losses', [])
            training_state.cls_losses = saved_history.get('cls_losses', [])
            training_state.map50_list = saved_history.get('map50_list', [])
            training_state.map50_95_list = saved_history.get('map50_95_list', [])
            training_state.best_map50 = saved_history.get('best_map50', 0.0)
            training_state.best_map50_95 = saved_history.get('best_map50_95', 0.0)
            training_state.best_epoch = saved_history.get('best_epoch', 0)
            training_state.current_epoch = saved_history.get('current_epoch', 0)
            training_state.total_epochs = saved_history.get('total_epochs', 0)
            training_state.output_dir = saved_history.get('output_dir', '')
            print(f"[DL-Hub] ✓ 已恢复训练历史: {len(training_state.map50_list)} epochs, 最佳mAP@50: {training_state.best_map50:.4f}")
    
    # 【增强】恢复日志到各个state
    if saved_logs:
        training_state.logs = saved_logs
        print(f"[DL-Hub] ✓ 已恢复训练日志: {len(saved_logs)} 行")
    if saved_conv_logs:
        conversion_state.logs = saved_conv_logs
        print(f"[DL-Hub] ✓ 已恢复转换日志: {len(saved_conv_logs)} 行")
    if saved_inf_logs:
        inference_state.logs = saved_inf_logs
        print(f"[DL-Hub] ✓ 已恢复推理日志: {len(saved_inf_logs)} 行")
    
    with gr.Blocks(
        title="YOLO 目标检测训练工具",
        css=CUSTOM_CSS,
        
    ) as app:
                
        # 标题
        gr.HTML("""
        <div style="text-align: center; padding: 15px 0 10px 0;">
            <h1 style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2em;
                font-weight: bold;
                margin-bottom: 5px;
            ">🎯 YOLO 目标检测工具</h1>
            <p style="color: #666; font-size: 1em; margin: 0;">
                模型训练 · 模型转换 · 批量推理
            </p>
        </div>
        """)
        
        # 环境信息
        with gr.Accordion("📋 环境信息", open=False):
            gr.Textbox(value=env_status, lines=5, interactive=False, show_label=False)
        
        # ==================== 选项卡 ====================
        with gr.Tabs():
            
            # ========== Tab 1: 模型训练 ==========
            with gr.TabItem("🎯 模型训练"):
                with gr.Row(equal_height=False):
                    # 左侧：配置面板
                    with gr.Column(scale=1, min_width=350):
                        gr.Markdown("### ⚙️ 训练配置")
                        
                        # 数据集
                        with gr.Group():
                            gr.Markdown("**📂 数据集 (LabelMe格式)**")
                            images_dir = gr.Textbox(
                                label="图像文件夹路径",
                                placeholder="输入图像文件夹路径...",
                                info="包含训练图像的文件夹",
                                value=saved_data.get('images_dir', '')
                            )
                            jsons_dir = gr.Textbox(
                                label="JSON标注文件夹路径",
                                placeholder="输入JSON标注文件夹路径...",
                                info="包含LabelMe JSON标注的文件夹",
                                value=saved_data.get('jsons_dir', '')
                            )
                            data_status = gr.Textbox(
                                label="验证状态",
                                interactive=False,
                                value="⏳ 请输入数据集路径",
                                lines=4,
                                max_lines=8
                            )
                            with gr.Row():
                                validate_btn = gr.Button("🔍 验证数据集", size="sm")
                                preview_btn = gr.Button("🔄 换一批预览", size="sm")
                            preview_gallery = gr.Gallery(
                                label="数据预览",
                                show_label=True,
                                columns=4,
                                rows=2,
                                height=250,
                                object_fit="contain",
                            )
                        
                        # 模型选择
                        with gr.Group():
                            gr.Markdown("**🧠 模型选择**")
                            with gr.Row():
                                model_family = gr.Dropdown(
                                    choices=families,
                                    value=saved_model.get('family', families[0] if families else None),
                                    label="模型系列"
                                )
                                model_scale = gr.Dropdown(
                                    choices=default_scales,
                                    value=saved_model.get('scale', default_scales[0] if default_scales else None),
                                    label="模型规模"
                                )
                            
                            model_info = gr.Textbox(
                                label="模型信息",
                                interactive=False,
                                lines=3
                            )
                            weights_status = gr.Textbox(
                                label="权重状态",
                                interactive=False
                            )
                        
                        # 训练参数
                        with gr.Group():
                            gr.Markdown("**📊 训练参数**")
                            with gr.Row():
                                epochs = gr.Slider(1, 300, value=saved_training.get('epochs', 100), step=1, label="训练轮数")
                                batch_size = gr.Slider(2, 64, value=saved_training.get('batch_size', 16), step=2, label="批量大小")
                            with gr.Row():
                                learning_rate = gr.Number(value=saved_training.get('learning_rate', 0.01), label="学习率")
                                optimizer = gr.Dropdown(
                                    choices=["SGD", "Adam", "AdamW"],
                                    value=saved_training.get('optimizer', 'SGD'),
                                    label="优化器"
                                )
                            with gr.Row():
                                img_size = gr.Slider(320, 1280, value=saved_data.get('image_size', 640), step=32, label="图像尺寸")
                                num_classes = gr.Number(
                                    value=saved_data.get('num_classes', 0),
                                    label="类别数",
                                    info="0=自动检测，或手动指定（必须>=最大类别ID+1）",
                                    precision=0,
                                    minimum=0
                                )
                            with gr.Row():
                                val_split = gr.Slider(
                                    0.1, 0.4,
                                    value=saved_data.get('val_split', 0.2),
                                    step=0.05,
                                    label="验证集比例",
                                    info="从训练数据中划分的验证集比例"
                                )
                                include_negative = gr.Checkbox(
                                    value=saved_data.get('include_negative', True),
                                    label="包含无标注图像作为负样本",
                                    info="将没有JSON标注的图像加入训练（用于减少误检）"
                                )
                        
                        # 高级设置
                        with gr.Accordion("🔧 高级设置", open=False):
                            with gr.Row():
                                save_freq = gr.Slider(1, 50, value=saved_training.get('save_period', 10), step=1, label="保存频率(epochs)")
                            gpu_choice = gr.Dropdown(
                                choices=gpu_options,
                                value=saved_device if saved_device in gpu_options else (gpu_options[0] if gpu_options else "CPU"),
                                label="计算设备"
                            )
                        
                        # 控制按钮
                        with gr.Row():
                            train_btn = gr.Button("🚀 开始训练", variant="primary", elem_classes="primary-btn")
                            stop_btn = gr.Button("⏹️ 停止训练", variant="stop", elem_classes="stop-btn")
                        
                        train_status = gr.Textbox(label="状态", interactive=False, lines=1)
                    
                    # 右侧：训练监控
                    with gr.Column(scale=2):
                        gr.Markdown("### 📈 训练监控")
                        progress_html = gr.HTML(value=make_progress_html(0, 0, 0, 0.0, 0.0, None))
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**损失曲线**")
                                loss_plot = gr.Plot(label="Loss")
                            with gr.Column():
                                gr.Markdown("**mAP曲线**")
                                map_plot = gr.Plot(label="mAP")
                        
                        # 测试推理
                        with gr.Accordion("🔬 测试推理", open=False):
                            with gr.Row():
                                test_img = gr.Image(label="测试图片", type="filepath")
                                test_result = gr.Image(label="检测结果")
                            test_btn = gr.Button("🔍 执行检测", size="sm")
                        
                        # 训练日志
                        with gr.Accordion("📝 训练日志", open=True):
                            train_logs = gr.Textbox(
                                label="日志输出",
                                lines=25,  # 增加高度
                                max_lines=50,
                                interactive=False,
                                elem_classes="log-box"
                            )
                
                # 定时刷新
                train_timer = gr.Timer(value=2)
                train_timer.tick(
                    fn=get_training_status_ui,
                    outputs=[progress_html, loss_plot, map_plot, train_status, train_logs]
                )
                
                # 事件绑定
                validate_btn.click(fn=validate_data_paths, inputs=[images_dir, jsons_dir], outputs=data_status)
                images_dir.submit(fn=validate_data_paths, inputs=[images_dir, jsons_dir], outputs=data_status)
                jsons_dir.submit(fn=validate_data_paths, inputs=[images_dir, jsons_dir], outputs=data_status)
                
                preview_btn.click(fn=refresh_data_preview, inputs=[images_dir, jsons_dir], outputs=preview_gallery)
                
                model_family.change(fn=update_model_scales, inputs=model_family, outputs=model_scale)
                model_family.change(fn=update_model_info_display, inputs=[model_family, model_scale], outputs=[model_info, weights_status])
                model_scale.change(fn=update_model_info_display, inputs=[model_family, model_scale], outputs=[model_info, weights_status])
                
                train_btn.click(
                    fn=on_start_training_simple,
                    inputs=[
                        images_dir, jsons_dir, model_family, model_scale,
                        epochs, batch_size, learning_rate, optimizer,
                        img_size, num_classes, val_split, include_negative, save_freq, gpu_choice
                    ],
                    outputs=[train_status, train_logs]
                )
                stop_btn.click(fn=on_stop_training, outputs=train_status)
                
                test_btn.click(
                    fn=run_test_inference,
                    inputs=test_img,
                    outputs=test_result
                )
            
            # ========== Tab 2: 模型转换 ==========
            with gr.TabItem("🔄 模型转换"):
                gr.Markdown("""
                ### 📝 说明
                将训练好的 YOLO 模型转换为高性能推理引擎格式（TensorRT/ONNX Runtime/OpenVINO）。
                - **INT8/Mixed 精度**：需要提供校准数据集（图片文件夹，无需标注）
                - **动态批处理**：TensorRT 支持配置动态 batch size
                """)
                
                with gr.Row(equal_height=False):
                    # 左侧：配置
                    with gr.Column(scale=1, min_width=350):
                        gr.Markdown("### ⚙️ 转换配置")
                        
                        with gr.Group():
                            gr.Markdown("**📂 选择模型**")
                            conv_model_folder = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="选择训练文件夹",
                                info="从output目录选择训练结果文件夹"
                            )
                            conv_model_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="选择模型文件",
                                allow_custom_value=True
                            )
                        
                        with gr.Group():
                            gr.Markdown("**🎯 转换目标**")
                            target_backend = gr.Dropdown(
                                choices=["tensorrt", "onnxruntime", "openvino"],
                                value=saved_conversion.get('target_backend', 'tensorrt'),
                                label="目标后端",
                                info="TensorRT: NVIDIA GPU | ONNX Runtime: 通用 | OpenVINO: Intel"
                            )
                            precision = gr.Dropdown(
                                choices=["fp16", "fp32", "int8", "mixed"],
                                value=saved_conversion.get('precision', 'fp16'),
                                label="精度模式",
                                info="fp16推荐 | int8需要校准数据 | mixed混合精度"
                            )
                            conv_device = gr.Dropdown(
                                choices=gpu_options,
                                value=saved_conversion.get('device', gpu_options[0] if gpu_options else "CPU"),
                                label="设备"
                            )
                        
                        # 校准数据集（仅int8/mixed时显示）
                        with gr.Group(visible=False) as calib_group:
                            gr.Markdown("**📊 校准数据集**")
                            calib_data_path = gr.Textbox(
                                label="校准图片文件夹路径",
                                placeholder="输入包含图片的文件夹路径...",
                                info="用于INT8量化校准，无需标注文件"
                            )
                            calib_status = gr.Textbox(
                                label="验证状态",
                                interactive=False,
                                value="⏳ 请输入校准数据集路径",
                                lines=2
                            )
                        
                        # TensorRT选项
                        with gr.Group(visible=True) as trt_group:
                            gr.Markdown("**🔧 TensorRT 专用选项**")
                            workspace_gb = gr.Slider(
                                1, 16, value=saved_conversion.get('workspace_gb', 4), step=1, 
                                label="Workspace (GB)",
                                info="更大空间可能找到更优算法"
                            )
                        
                        # 动态Batch配置
                        with gr.Group(visible=True) as dynamic_batch_group:
                            gr.Markdown("**⚡ 动态Batch配置**")
                            dynamic_batch = gr.Checkbox(
                                value=saved_conversion.get('dynamic_batch', False), 
                                label="启用动态批处理",
                                info="允许运行时使用不同batch size"
                            )
                            with gr.Row():
                                min_batch = gr.Slider(1, 64, value=saved_conversion.get('min_batch', 1), step=1, label="最小Batch")
                                opt_batch = gr.Slider(1, 64, value=saved_conversion.get('opt_batch', 1), step=1, label="最优Batch (TensorRT)", visible=True)
                                max_batch = gr.Slider(1, 64, value=saved_conversion.get('max_batch', 8), step=1, label="最大Batch")
                        
                        # 输出设置
                        with gr.Group():
                            gr.Markdown("**📁 输出设置**")
                            conv_output_dir = gr.Textbox(
                                label="输出目录",
                                value=saved_conversion.get('output_dir', ''),
                                placeholder="留空则输出到模型目录下的 converted 文件夹"
                            )
                        
                        conv_btn = gr.Button("🚀 开始转换", variant="primary", elem_classes="primary-btn")
                        conv_status_text = gr.Textbox(label="状态", interactive=False, lines=1)
                    
                    # 右侧：日志
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 转换日志")
                        conv_logs = gr.Textbox(
                            label="日志",
                            lines=30,
                            interactive=False,
                            elem_classes="log-box"
                        )
                
                # 定时刷新
                conv_timer = gr.Timer(value=2)
                conv_timer.tick(
                    fn=get_conversion_status,
                    outputs=[conv_status_text, conv_logs]
                )
                
                # 事件绑定
                # 下拉框展开时，扫描文件夹
                conv_model_folder.focus(fn=scan_model_folders, outputs=conv_model_folder)
                # 选择文件夹后，扫描模型文件
                conv_model_folder.change(fn=scan_models_in_selected_folder, inputs=conv_model_folder, outputs=conv_model_dropdown)
                
                target_backend.change(
                    fn=update_backend_options,
                    inputs=target_backend,
                    outputs=[trt_group, workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch, dynamic_batch_group]
                )
                
                precision.change(
                    fn=update_calib_visibility,
                    inputs=precision,
                    outputs=calib_group
                )
                
                calib_data_path.change(fn=validate_calib_data_det, inputs=calib_data_path, outputs=calib_status)
                calib_data_path.submit(fn=validate_calib_data_det, inputs=calib_data_path, outputs=calib_status)
                
                conv_btn.click(
                    fn=on_start_conversion,
                    inputs=[
                        conv_model_dropdown, target_backend, precision, conv_device,
                        workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch,
                        conv_output_dir, calib_data_path
                    ],
                    outputs=[conv_status_text, conv_logs]
                )
            
            # ========== Tab 3: 批量推理 ==========
            with gr.TabItem("🔍 批量推理"):
                gr.Markdown("""
                ### 📝 说明
                使用训练好的模型对图片进行批量目标检测。
                - 支持单张图片或文件夹（递归扫描子目录）
                - 输出可视化图片和JSON格式检测结果
                """)
                
                with gr.Row(equal_height=False):
                    # 左侧：配置
                    with gr.Column(scale=1, min_width=350):
                        gr.Markdown("### ⚙️ 推理配置")
                        
                        with gr.Group():
                            gr.Markdown("**📂 选择模型**")
                            inf_model_folder = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="选择训练文件夹",
                                info="从output目录选择训练结果文件夹"
                            )
                            inf_model_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="选择模型文件",
                                allow_custom_value=True
                            )
                        
                        with gr.Group():
                            gr.Markdown("**🖼️ 输入数据**")
                            inf_input_path = gr.Textbox(
                                label="图片路径",
                                value=saved_inference.get('input_path', ''),
                                placeholder="输入图片文件或文件夹路径..."
                            )
                        
                        with gr.Group():
                            gr.Markdown("**⚙️ 推理参数**")
                            with gr.Row():
                                inf_conf = gr.Slider(0.1, 0.9, value=saved_inference.get('conf_threshold', 0.25), step=0.05, label="置信度阈值")
                                inf_iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU阈值")
                            inf_device = gr.Dropdown(
                                choices=gpu_options,
                                value=saved_inference.get('device', gpu_options[0] if gpu_options else "CPU"),
                                label="设备"
                            )
                        
                        with gr.Group():
                            gr.Markdown("**📁 输出设置**")
                            inf_output_dir = gr.Textbox(
                                label="输出目录",
                                value=saved_inference.get('output_dir', ''),
                                placeholder="留空则输出到 ./inference_output"
                            )
                            with gr.Row():
                                inf_save_images = gr.Checkbox(value=saved_inference.get('save_images', True), label="保存可视化图片")
                                inf_save_json = gr.Checkbox(value=saved_inference.get('save_json', True), label="保存JSON结果")
                        
                        inf_btn = gr.Button("🚀 开始推理", variant="primary", elem_classes="primary-btn")
                        inf_status_text = gr.Textbox(label="状态", interactive=False, lines=1)
                    
                    # 右侧：结果展示
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 推理结果")
                        
                        # 结果预览
                        with gr.Group():
                            gr.Markdown("**🖼️ 检测预览**")
                            preview_image = gr.Image(label="检测结果", height=400)
                            preview_info = gr.Textbox(label="图片信息", interactive=False, lines=1)
                            with gr.Row():
                                prev_btn = gr.Button("◀ 上一张", size="sm")
                                refresh_preview_btn = gr.Button("🔄 刷新", size="sm")
                                next_btn = gr.Button("下一张 ▶", size="sm")
                        
                        # 推理日志
                        with gr.Accordion("📝 推理日志", open=True):
                            inf_logs = gr.Textbox(
                                label="日志",
                                lines=25,  # 增加高度
                                max_lines=50,
                                interactive=False,
                                elem_classes="log-box"
                            )
                
                # 定时刷新（包含自动预览刷新）
                inf_timer = gr.Timer(value=2)
                inf_timer.tick(
                    fn=get_inference_status_with_preview,
                    outputs=[inf_status_text, inf_logs, preview_image, preview_info]
                )
                
                # 事件绑定
                # 下拉框展开时，扫描文件夹
                inf_model_folder.focus(fn=scan_model_folders, outputs=inf_model_folder)
                # 选择文件夹后，扫描模型文件
                inf_model_folder.change(fn=scan_models_in_selected_folder, inputs=inf_model_folder, outputs=inf_model_dropdown)
                
                inf_btn.click(
                    fn=on_start_inference,
                    inputs=[
                        inf_model_dropdown, inf_input_path, inf_output_dir,
                        inf_device, inf_conf,
                        inf_save_images, inf_save_json
                    ],
                    outputs=[inf_status_text, inf_logs]
                )
                
                prev_btn.click(fn=on_prev_preview, outputs=[preview_image, preview_info])
                next_btn.click(fn=on_next_preview, outputs=[preview_image, preview_info])
                refresh_preview_btn.click(fn=on_refresh_preview, outputs=[preview_image, preview_info])
    
    return app


if __name__ == "__main__":
    # 【关键】绕过系统代理，防止Gradio 6.0自检被代理拦截
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
    os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    
    # 解析命令行参数（DL-Hub 兼容）
    parser = argparse.ArgumentParser(description='YOLO 目标检测训练工具')
    parser.add_argument('--task-dir', type=str, default=None, help='DL-Hub 任务目录')
    parser.add_argument('--port', type=int, default=7860, help='Gradio 服务端口')
    args, _ = parser.parse_known_args()
    
    app = create_ui()
    
    # 确定启动配置
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode:
        launch_port = dlhub_adapter.port
        launch_inbrowser = False
        print(f"[DL-Hub] 以 DL-Hub 模式启动，端口: {launch_port}")
    else:
        launch_port = args.port
        launch_inbrowser = True
        print(f"[独立模式] 启动端口: {launch_port}")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=launch_inbrowser
    )