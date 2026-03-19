"""
TIMM 图像分类工具 - Gradio Web界面
功能：模型训练、模型转换、批量推理

已集成 DL-Hub 支持：
- 支持 --task-dir 参数指定任务目录
- 支持 --port 参数指定端口
- 自动保存/加载UI参数
"""

import os
import sys
import threading
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import gradio as gr
import pandas as pd

# 添加父目录到路径以便导入dlhub_params
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== DL-Hub 集成 ====================
def init_dlhub_adapter():
    """初始化 DL-Hub 适配器"""
    try:
        # 尝试导入 DL-Hub 适配器
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


# 全局适配器实例
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

.chart-container {
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.logs = []
        self.best_acc = 0.0
        self.best_epoch = 0
        self.output_dir = ""
        self.data_path = ""  # 保存训练数据路径
        self.start_time = None
    
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
    
    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]


# 全局状态实例
training_state = TrainingState()
conversion_state = ConversionState()
inference_state = InferenceState()

# ==================== Matplotlib Figure 缓存 ====================
# 修复：避免每次调用 get_training_status 都创建新的 figure 导致内存泄漏
_cached_figures = {
    'fig_loss': None,
    'ax_loss': None,
    'fig_acc': None,
    'ax_acc': None,
}


# ==================== 通用工具函数 ====================
def check_environment() -> str:
    """检查运行环境"""
    try:
        from utils.env_validator import validate_environment
        success, message, info = validate_environment()
        return message
    except Exception as e:
        return f"❌ 环境验证出错: {str(e)}"


def get_gpu_options() -> list:
    """获取GPU选项列表"""
    try:
        from utils.env_validator import get_gpu_choices
        return get_gpu_choices()
    except Exception:
        return ["CPU"]


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}时"


def get_trained_models() -> List[str]:
    """获取已训练的模型列表（从 checkpoints 文件夹）"""
    models = []
    
    # 使用get_output_dir()获取正确的输出目录
    output_dir = get_output_dir()
    
    if output_dir.exists():
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                checkpoints_dir = model_dir / "checkpoints"
                if checkpoints_dir.exists():
                    # 检查 checkpoints 文件夹中的模型
                    for model_file in ["best_model.pth", "last_model.pth"]:
                        model_path = checkpoints_dir / model_file
                        if model_path.exists():
                            models.append(str(model_path))
    
    return sorted(models, reverse=True)


def get_training_data_path(model_path: str) -> Optional[str]:
    """从模型路径推断训练数据路径"""
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 1. 从训练配置中获取数据路径
        if 'training_config' in checkpoint:
            data_path = checkpoint['training_config'].get('data_path', '')
            if data_path:
                # 尝试绝对路径
                if os.path.exists(data_path):
                    return data_path
                # 尝试相对于模型目录的路径
                model_dir = Path(model_path).parent.parent
                relative_path = model_dir / data_path
                if relative_path.exists():
                    return str(relative_path)
        
        # 2. 从模型目录中查找
        model_dir = Path(model_path).parent.parent  # checkpoints 的父目录
        
        # 3. 常见的数据目录名（扩展搜索范围）
        search_patterns = [
            'train', 'data', 'dataset', 'images',
            '../train', '../data', '../dataset',
            '../../train', '../../data', '../../dataset',
        ]
        
        for pattern in search_patterns:
            p = model_dir / pattern
            if p.exists() and p.is_dir():
                # 验证是否为有效的ImageFolder结构（至少有一个子目录）
                subdirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if subdirs:
                    return str(p.resolve())
        
        # 4. 尝试从全局训练状态获取（如果刚训练完）
        global training_state
        if training_state.data_path and os.path.exists(training_state.data_path):
            return training_state.data_path
        
        return None
    except Exception:
        return None


# ==================== Tab 1: 训练相关函数 ====================
def validate_data_path(data_path: str) -> str:
    """验证数据集路径"""
    if not data_path or not data_path.strip():
        return "⏳ 请输入数据集路径"
    
    try:
        from utils.data_validator import validate_dataset
        result = validate_dataset(data_path.strip())
        return result.message
    except Exception as e:
        return f"❌ 验证出错: {str(e)}"


def update_model_info(family: str, scale: str) -> str:
    """更新模型信息显示"""
    if not family or not scale:
        return "请选择模型系列和规模"
    
    try:
        from config.model_registry import get_model_display_info
        return get_model_display_info(family, scale)
    except Exception as e:
        return f"获取模型信息失败: {str(e)}"


def check_pretrained_weights(family: str, scale: str) -> str:
    """检查预训练权重状态"""
    if not family or not scale:
        return ""
    
    try:
        from config.model_registry import get_model_config
        from models.model_factory import get_pretrained_path
        
        config = get_model_config(family, scale)
        if config is None:
            return "❌ 未找到模型配置"
        
        model_name = config['name']
        local_path = get_pretrained_path(model_name)
        
        if local_path:
            return f"✅ 已缓存: {os.path.basename(local_path)}"
        else:
            return "📥 首次使用将从timm下载并缓存"
    except Exception as e:
        return f"检查失败: {str(e)}"


def get_training_logs() -> str:
    """获取训练日志"""
    return "\n".join(training_state.logs[-100:]) if training_state.logs else "暂无日志"


def on_start_training(
    data_path: str,
    model_family: str,
    model_scale: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    img_size: int,
    val_split: float,
    save_freq: int,
    early_stopping: bool,
    gpu_choice: str,
) -> Tuple[str, str]:
    """开始训练"""
    global training_state, conversion_state
    
    # 检查是否正在转换
    if conversion_state.is_running:
        return "⚠️ 模型转换正在进行中，请等待转换完成后再开始训练", ""
    
    if training_state.is_training:
        return "⚠️ 训练正在进行中...", get_training_logs()
    
    if not data_path or not data_path.strip():
        return "❌ 请输入数据集路径", ""
    
    if not model_family or not model_scale:
        return "❌ 请选择模型系列和规模", ""
    
    training_state.reset()
    training_state.is_training = True
    training_state.total_epochs = int(epochs)
    training_state.data_path = data_path.strip()  # 保存数据路径
    training_state.start_time = time.time()
    
    thread = threading.Thread(
        target=run_training,
        args=(
            data_path.strip(), model_family, model_scale,
            int(epochs), int(batch_size), float(learning_rate),
            optimizer_name, int(img_size), float(val_split),
            int(save_freq), early_stopping, gpu_choice,
        ),
        daemon=True,
    )
    thread.start()
    
    return "🚀 训练已启动，请查看下方进度...", ""


def run_training(
    data_path: str, model_family: str, model_scale: str,
    epochs: int, batch_size: int, learning_rate: float,
    optimizer_name: str, img_size: int, val_split: float,
    save_freq: int, early_stopping: bool, gpu_choice: str,
):
    """后台训练线程"""
    global training_state
    
    try:
        import torch
        from config.model_registry import get_model_config
        from models import create_model, setup_model_for_training
        from data import create_dataloaders
        from engine import Trainer, TrainingCallback
        from utils.env_validator import parse_gpu_choice
        
        # 保存训练参数到DL-Hub - 使用合并方式避免覆盖其他参数
        if dlhub_params:
            current_params = dlhub_params.get_all()
            current_params['data'] = {
                'data_path': data_path,
                'image_size': img_size,
                'val_split': val_split,
            }
            current_params['model'] = {
                'family': model_family,
                'scale': model_scale,
            }
            current_params['training'] = {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'optimizer': optimizer_name,
                'save_freq': save_freq,
                'early_stopping': early_stopping,
            }
            current_params['device'] = gpu_choice
            dlhub_params.save(current_params)
        
        # GPU设置
        device_type, gpu_ids = parse_gpu_choice(gpu_choice)
        if device_type == "cuda" and gpu_ids:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            torch.cuda.set_device(gpu_ids[0])
        else:
            device = torch.device("cpu")
        
        training_state.add_log(f"🖥️ 设备: {device}")
        
        # 模型配置
        config = get_model_config(model_family, model_scale)
        model_name = config['name']
        training_state.add_log(f"🧠 模型: {model_name} ({config['params']}M 参数)")
        
        # 数据集验证
        from utils.data_validator import validate_dataset
        data_info = validate_dataset(data_path)
        if not data_info.is_valid:
            training_state.add_log(f"❌ 数据集错误: {data_info.message}")
            training_state.is_training = False
            return
        
        num_classes = data_info.num_classes
        training_state.add_log(f"📊 类别数: {num_classes}")
        
        # 数据加载
        training_state.add_log("📂 加载数据集...")
        train_loader, val_loader, loader_info = create_dataloaders(
            data_path=data_path,
            model_name=model_name,
            img_size=img_size,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=4,
        )
        training_state.add_log(f"   ├─ 训练: {loader_info['train_size']} 样本")
        training_state.add_log(f"   └─ 验证: {loader_info['val_size']} 样本")
        
        # 创建模型
        training_state.add_log("🔧 初始化模型...")
        model, model_info = create_model(
            model_family=model_family,
            model_scale=model_scale,
            num_classes=num_classes,
            pretrained=True,
        )
        training_state.add_log(f"   └─ 权重: {model_info['weights_source']}")
        
        # 添加类别映射到 model_info
        model_info['class_to_idx'] = loader_info.get('class_to_idx', {})
        
        # 配置模型
        model = setup_model_for_training(
            model=model,
            device=device,
            gpu_ids=gpu_ids if len(gpu_ids) > 1 else None,
        )
        
        # 优化器
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 输出目录 - 关键修复：使用get_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = get_output_dir()
        output_dir = output_base / f"{model_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        training_state.output_dir = str(output_dir)
        training_state.add_log(f"📁 输出: {output_dir}")
        
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
        
        # 回调
        def on_epoch_end(epoch: int, metrics: Dict):
            training_state.current_epoch = epoch
            training_state.train_losses.append(float(metrics['train_loss']))
            training_state.val_losses.append(float(metrics['val_loss']))
            training_state.val_accs.append(float(metrics['val_acc']))
            training_state.best_acc = float(metrics['best_acc'])
            training_state.best_epoch = int(metrics.get('best_epoch', epoch))
            
            # 【增强】保存历史数据到DL-Hub（每个epoch）
            if dlhub_params:
                dlhub_params.update_history_epoch({
                    'train_loss': float(metrics['train_loss']),
                    'val_loss': float(metrics['val_loss']),
                    'val_acc': float(metrics['val_acc']),
                    'current_epoch': epoch,
                    'best_acc': float(metrics['best_acc']),
                    'best_epoch': int(metrics.get('best_epoch', epoch))
                }, auto_save=(epoch % 5 == 0))  # 每5个epoch保存一次，避免频繁IO
        
        def on_log(message: str):
            training_state.add_log(message)
            # 【增强】追加日志到DL-Hub（不自动保存，训练结束时统一保存）
            if dlhub_params:
                dlhub_params.append_log(message, 'training', auto_save=False)
        
        def should_stop():
            return training_state.should_stop
        
        callback = TrainingCallback(
            on_epoch_end=on_epoch_end,
            on_log=on_log,
            should_stop=should_stop,
        )
        
        # 训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=str(output_dir),
            model_info=model_info,
            epochs=epochs,
            use_amp=True,
            early_stopping=early_stopping,
            patience=10,
            save_freq=save_freq,
            callback=callback,
        )
        
        training_state.trainer = trainer
        training_state.add_log("━" * 50)
        training_state.add_log("🚀 开始训练")
        training_state.add_log("━" * 50)
        
        trainer.train()
        
        elapsed = time.time() - training_state.start_time
        training_state.add_log("━" * 50)
        training_state.add_log(f"✅ 训练完成！用时 {elapsed/60:.1f} 分钟")
        training_state.add_log(f"🏆 最佳准确率: {training_state.best_acc:.2f}% (Epoch {training_state.best_epoch})")
        training_state.add_log("━" * 50)
        
        # 【增强】标记训练完成并保存最终状态
        if dlhub_params:
            dlhub_params.mark_training_complete(
                best_metric=training_state.best_acc,
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
    """停止训练"""
    global training_state
    
    if not training_state.is_training:
        return "⚠️ 当前没有正在进行的训练"
    
    training_state.should_stop = True
    training_state.add_log("⏹️ 正在停止训练...")
    return "⏹️ 正在停止，等待当前epoch完成..."


def get_progress_html() -> str:
    """生成训练进度HTML"""
    global training_state
    
    if training_state.total_epochs == 0:
        progress_pct = 0
    else:
        progress_pct = (training_state.current_epoch / training_state.total_epochs) * 100
    
    elapsed = time.time() - training_state.start_time if training_state.start_time else 0
    
    if training_state.current_epoch > 0 and training_state.is_training:
        per_epoch = elapsed / training_state.current_epoch
        remaining_epochs = training_state.total_epochs - training_state.current_epoch
        estimated_remaining = per_epoch * remaining_epochs
    else:
        estimated_remaining = 0
    
    if training_state.is_training:
        status_color = "#3b82f6"
        status_text = "训练中"
    elif training_state.current_epoch > 0:
        status_color = "#10b981"
        status_text = "已完成"
    else:
        status_color = "#6b7280"
        status_text = "等待中"
    
    return f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); padding: 15px; border-radius: 12px; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: bold; color: #374151;">训练进度</span>
            <span style="background: {status_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.85em;">{status_text}</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 15px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {progress_pct:.1f}%; 
                        transition: width 0.5s ease; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-size: 0.85em; font-weight: bold; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">{progress_pct:.1f}%</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; background: white; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 0.85em; color: #6c757d;">当前轮次</div>
                <div style="font-size: 1.4em; font-weight: bold; color: #4a5568;">{training_state.current_epoch}/{training_state.total_epochs}</div>
            </div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #ddd; border-right: 1px solid #ddd;">
                <div style="font-size: 0.85em; color: #6c757d;">最佳准确率</div>
                <div style="font-size: 1.4em; font-weight: bold; color: #10b981;">{training_state.best_acc:.2f}%</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 0.85em; color: #6c757d;">最佳轮次</div>
                <div style="font-size: 1.4em; font-weight: bold; color: #4a5568;">{f'Epoch {training_state.best_epoch}' if training_state.best_epoch > 0 else '--'}</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; background: white; padding: 12px 15px; border-radius: 8px;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 0.8em; color: #6c757d;">⏱️ 已用时间</div>
                <div style="font-size: 1.1em; font-weight: bold; color: #4a5568;">{format_time(elapsed)}</div>
            </div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #eee;">
                <div style="font-size: 0.8em; color: #6c757d;">⏳ 预计剩余</div>
                <div style="font-size: 1.1em; font-weight: bold; color: {'#f59e0b' if training_state.is_training else '#6c757d'};">{format_time(estimated_remaining) if training_state.is_training and training_state.current_epoch > 0 else '--'}</div>
            </div>
            <div style="text-align: center; flex: 1; border-left: 1px solid #eee;">
                <div style="font-size: 0.8em; color: #6c757d;">📊 每轮耗时</div>
                <div style="font-size: 1.1em; font-weight: bold; color: #4a5568;">{format_time(elapsed / training_state.current_epoch) if training_state.current_epoch > 0 else '--'}</div>
            </div>
        </div>
    </div>
    """


def get_training_status() -> Tuple[str, str, str, Any, Any]:
    """获取训练状态 - 修复版：使用缓存的Figure避免内存泄漏"""
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
        status_text = f"✅ 训练完成 - 最佳准确率: {training_state.best_acc:.2f}%"
    else:
        status_text = "⏳ 等待开始..."
    
    logs = "\n".join(training_state.logs[-80:]) if training_state.logs else "暂无日志"
    
    # ========== 修复：复用或创建 Loss 图表，避免内存泄漏 ==========
    if _cached_figures['fig_loss'] is None:
        _cached_figures['fig_loss'], _cached_figures['ax_loss'] = plt.subplots(figsize=(5, 3.5), dpi=100)
    else:
        _cached_figures['ax_loss'].clear()  # 清除旧数据，复用 figure
    
    ax_loss = _cached_figures['ax_loss']
    fig_loss = _cached_figures['fig_loss']
    
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.set_title('Loss Curve', fontsize=11, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    
    if training_state.train_losses and len(training_state.train_losses) > 0:
        epochs = list(range(1, len(training_state.train_losses) + 1))
        ax_loss.plot(epochs, training_state.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        if training_state.val_losses:
            ax_loss.plot(epochs[:len(training_state.val_losses)], training_state.val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax_loss.legend(loc='upper right', fontsize=9)
    else:
        ax_loss.text(0.5, 0.5, 'Waiting for training data...', ha='center', va='center', 
                     transform=ax_loss.transAxes, fontsize=11, color='gray')
        ax_loss.set_xlim(0, 10)
        ax_loss.set_ylim(0, 1)
    
    fig_loss.tight_layout()
    
    # ========== 修复：复用或创建 Accuracy 图表，避免内存泄漏 ==========
    if _cached_figures['fig_acc'] is None:
        _cached_figures['fig_acc'], _cached_figures['ax_acc'] = plt.subplots(figsize=(5, 3.5), dpi=100)
    else:
        _cached_figures['ax_acc'].clear()  # 清除旧数据，复用 figure
    
    ax_acc = _cached_figures['ax_acc']
    fig_acc = _cached_figures['fig_acc']
    
    ax_acc.set_xlabel('Epoch', fontsize=10)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
    ax_acc.set_title('Accuracy Curve', fontsize=11, fontweight='bold')
    ax_acc.grid(True, alpha=0.3)
    
    if training_state.val_accs and len(training_state.val_accs) > 0:
        epochs = list(range(1, len(training_state.val_accs) + 1))
        ax_acc.plot(epochs, training_state.val_accs, 'g-o', label='Val Accuracy', linewidth=2, markersize=4)
        ax_acc.legend(loc='lower right', fontsize=9)
        
        if training_state.best_epoch > 0 and training_state.best_epoch <= len(training_state.val_accs):
            ax_acc.scatter([training_state.best_epoch], [training_state.best_acc], color='red', s=100, zorder=5, marker='*')
    else:
        ax_acc.text(0.5, 0.5, 'Waiting for training data...', ha='center', va='center',
                    transform=ax_acc.transAxes, fontsize=11, color='gray')
        ax_acc.set_xlim(0, 10)
        ax_acc.set_ylim(0, 100)
    
    fig_acc.tight_layout()
    
    return progress_html, status_text, logs, fig_loss, fig_acc


# ==================== Tab 2: 模型转换相关函数 ====================
def refresh_model_list() -> Dict:
    """刷新模型列表"""
    models = get_trained_models()
    if models:
        return gr.update(choices=models, value=models[0])
    return gr.update(choices=[], value=None)


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
            has_models = any(item.rglob('*.pth'))
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
    
    models = list(folder.rglob('*.pth'))
    
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
    """扫描文件夹中的模型文件（仅.pth）- 用于推理Tab"""
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
    
    # 如果输入的是文件路径而非文件夹，直接返回该文件
    if os.path.isfile(folder_path):
        if folder_path.endswith('.pth'):
            file_name = os.path.basename(folder_path)
            _model_path_map[file_name] = folder_path
            return gr.update(choices=[file_name], value=file_name)
        return gr.update(choices=[], value=None)
    
    if not os.path.exists(folder_path):
        return gr.update(choices=[], value=None)
    
    folder = Path(folder_path)
    models = list(folder.rglob('*.pth'))
    
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
    """根据后端更新选项显示"""
    is_tensorrt = (backend == "tensorrt")
    # 动态batch配置对所有后端都显示，但opt_batch只对TensorRT显示
    return (
        gr.update(visible=is_tensorrt),  # trt_group (workspace)
        gr.update(visible=is_tensorrt),  # workspace_gb
        gr.update(visible=True),         # dynamic_batch - 所有后端都显示
        gr.update(visible=True),         # min_batch - 所有后端都显示
        gr.update(visible=is_tensorrt),  # opt_batch - 只有TensorRT需要
        gr.update(visible=True),         # max_batch - 所有后端都显示
        gr.update(visible=True),         # dynamic_batch_group - 所有后端都显示
    )


def update_calib_visibility(precision: str) -> Dict:
    """根据精度更新校准数据可见性"""
    needs_calib = precision in ['int8', 'mixed']
    return gr.update(visible=needs_calib)


def validate_calib_data(calib_path: str) -> str:
    """验证校准数据集路径"""
    if not calib_path or not calib_path.strip():
        return "⏳ 请输入校准数据集路径"
    
    calib_path = calib_path.strip()
    
    if not os.path.exists(calib_path):
        return f"❌ 路径不存在: {calib_path}"
    
    if not os.path.isdir(calib_path):
        return "❌ 请输入文件夹路径，不是文件"
    
    # 检查是否为有效的ImageFolder结构
    subdirs = [d for d in Path(calib_path).iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not subdirs:
        return "❌ 未找到类别子文件夹，请使用ImageFolder格式"
    
    # 统计图片数量
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    total_images = 0
    class_info = []
    
    for subdir in subdirs:
        images = [f for f in subdir.iterdir() if f.suffix.lower() in image_extensions]
        count = len(images)
        total_images += count
        class_info.append(f"{subdir.name}: {count}张")
    
    if total_images == 0:
        return "❌ 未找到图片文件"
    
    if total_images < 100:
        warning = "⚠️ 建议至少100张图片以获得更好的校准效果\n"
    else:
        warning = ""
    
    return f"✅ 验证通过\n{warning}📊 共 {len(subdirs)} 个类别, {total_images} 张图片"


def get_conversion_logs() -> str:
    """获取转换日志"""
    return "\n".join(conversion_state.logs[-100:]) if conversion_state.logs else "暂无日志"


def get_conversion_status() -> Tuple[str, str]:
    """获取转换状态"""
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
    calib_data_path: str = "",
) -> Tuple[str, str]:
    """开始模型转换"""
    global conversion_state, training_state
    
    # 检查是否正在训练
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
    
    # 检查INT8/Mixed精度是否提供了校准数据
    if precision in ['int8', 'mixed']:
        if not calib_data_path or not calib_data_path.strip():
            return "❌ INT8/Mixed精度需要提供校准数据集路径", ""
        
        calib_data_path = calib_data_path.strip()
        if not os.path.exists(calib_data_path):
            return f"❌ 校准数据集路径不存在: {calib_data_path}", ""
        
        # 验证校准数据集
        validation_result = validate_calib_data(calib_data_path)
        if not validation_result.startswith("✅"):
            return f"❌ 校准数据集验证失败: {validation_result}", ""
    
    conversion_state.reset()
    conversion_state.is_running = True
    conversion_state.start_time = time.time()
    
    # 确定输出目录
    if not output_dir:
        output_dir = str(Path(full_model_path).parent.parent / "converted")
    
    thread = threading.Thread(
        target=run_conversion,
        args=(
            full_model_path, target_backend, precision, device,
            workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch,
            output_dir, calib_data_path,
        ),
        daemon=True,
    )
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
    calib_data_path: str = "",
):
    """
    后台转换线程 - 彻底修复版 V4
    
    关键修复：
    1. 强制重新加载转换模块，避免全局状态污染
    2. 每次转换使用独立的时间戳输出路径
    3. 完善的资源清理机制（CUDA + 内存）
    4. 禁用onnxsim简化避免卡死
    5. 捕获并记录详细错误信息
    6. 统一输出路径结构: converted/后端/日期时间/
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
    
    # 构建最终输出目录: base_output_dir/后端/日期时间/
    base_output_dir = output_dir
    output_dir = str(Path(base_output_dir) / backend_subdir / date_subdir)
    
    try:
        # ========== 关键修复1：彻底清理CUDA和内存 ==========
        conversion_state.add_log("🧹 清理资源...")
        try:
            import torch
            if torch.cuda.is_available():
                # 同步所有CUDA流
                torch.cuda.synchronize()
                # 清空缓存
                torch.cuda.empty_cache()
                # 重置峰值内存统计
                torch.cuda.reset_peak_memory_stats()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            gc.collect()  # 调用两次确保清理
            
        except Exception as e:
            conversion_state.add_log(f"⚠️ 初始清理警告: {e}")
        
        # ========== 关键修复2：彻底卸载所有转换相关模块 ==========
        modules_to_remove = []
        for mod_name in list(sys.modules.keys()):
            # 卸载所有转换相关模块
            if any(target in mod_name for target in [
                'main', 'model_importer', 'model_analyzer', 'model_optimizer',
                'model_exporter', 'model_converter', 'config_generator',
                'converter_tensorrt', 'converter_openvino', 'converter_ort',
                'conversion_validator', 'unified_logger', 'cuda_utils',
                'symbolic', 'constants', 'exceptions', 'config_templates',
                'onnxsim', 'onnxoptimizer'  # 也卸载onnx相关模块
            ]):
                modules_to_remove.append(mod_name)
        
        for mod_name in modules_to_remove:
            try:
                del sys.modules[mod_name]
            except Exception:
                pass
        
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
        conversion_state.add_log("🚀 开始模型转换")
        conversion_state.add_log("━" * 50)
        conversion_state.add_log(f"📂 模型: {model_path}")
        conversion_state.add_log(f"🎯 目标: {target_backend} ({precision})")
        conversion_state.add_log(f"📁 输出: {output_dir}")
        
        # 校准数据路径（由用户提供）
        if precision in ['int8', 'mixed']:
            if calib_data_path:
                conversion_state.add_log(f"📊 校准数据: {calib_data_path}")
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
        if not run_stage1_import(ctx, model_path, 'cls', device=ctx.device):
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
        
        elif target_backend == 'ort':
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
            calib_data_path=calib_data_path,
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
                task_type='cls',
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
        
        # 6. 保存转换日志到DL-Hub
        try:
            if dlhub_params:
                dlhub_params.save_logs(conversion_state.logs, 'conversion', auto_save=True)
        except Exception:
            pass


# ==================== Tab 3: 批量推理相关函数 ====================
def get_inference_logs() -> str:
    """获取推理日志"""
    return "\n".join(inference_state.logs[-100:]) if inference_state.logs else "暂无日志"


def get_inference_status() -> Tuple[str, str]:
    """获取推理状态"""
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


def on_start_inference(
    model_path: str,
    input_path: str,
    output_dir: str,
    device: str,
    copy_images: bool,
) -> Tuple[str, str]:
    """开始批量推理"""
    global inference_state, training_state, conversion_state
    
    # 检查是否正在训练或转换
    if training_state.is_training:
        return "⚠️ 训练正在进行中，请等待训练完成后再进行推理", ""
    
    if conversion_state.is_running:
        return "⚠️ 模型转换正在进行中，请等待转换完成后再进行推理", ""
    
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
            'copy_images': copy_images,
        }
        dlhub_params.save(current_params)
    
    inference_state.reset()
    inference_state.is_running = True
    inference_state.start_time = time.time()
    
    if not output_dir:
        output_dir = "./inference_output"
    
    thread = threading.Thread(
        target=run_inference,
        args=(full_model_path, input_path, output_dir, device, copy_images),
        daemon=True,
    )
    thread.start()
    
    return "🚀 推理已启动...", ""


def run_inference(
    model_path: str,
    input_path: str,
    output_dir: str,
    device: str,
    copy_images: bool,
):
    """后台推理线程"""
    global inference_state
    
    try:
        import torch
        import timm
        from PIL import Image
        from torchvision import transforms
        
        inference_state.add_log("━" * 50)
        inference_state.add_log("🚀 开始批量推理")
        inference_state.add_log("━" * 50)
        
        # 设置设备
        use_cuda = 'GPU' in device and torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
        inference_state.add_log(f"🖥️ 设备: {dev}")
        
        # 加载模型
        inference_state.add_log(f"📂 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=dev)
        
        # 提取模型信息
        model_name = checkpoint.get('model_name', 'efficientnet_b0')
        num_classes = checkpoint.get('num_classes', 1000)
        input_size = checkpoint.get('input_size', 224)
        normalize_mean = checkpoint.get('normalize_mean', (0.485, 0.456, 0.406))
        normalize_std = checkpoint.get('normalize_std', (0.229, 0.224, 0.225))
        
        # 获取类别映射
        class_to_idx = checkpoint.get('class_to_idx', {})
        idx_to_class = checkpoint.get('idx_to_class', {})
        if not idx_to_class and class_to_idx:
            idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        inference_state.add_log(f"   ├─ 模型: {model_name}")
        inference_state.add_log(f"   ├─ 类别数: {num_classes}")
        inference_state.add_log(f"   ├─ 输入尺寸: {input_size}")
        inference_state.add_log(f"   └─ 类别: {list(class_to_idx.keys()) if class_to_idx else '未知'}")
        
        # 创建模型
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
        
        # 加载权重
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        # 处理DataParallel的'module.'前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(dev)
        model.eval()
        inference_state.add_log("   ✅ 模型加载成功")
        
        # 构建transform
        transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
        
        # 收集图片
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
        
        images = list(set(images))  # 去重
        inference_state.total_images = len(images)
        inference_state.add_log(f"   找到 {len(images)} 张图片")
        
        if not images:
            inference_state.add_log("❌ 未找到图片")
            return
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 推理
        inference_state.add_log("\n🔍 开始推理...")
        results = []
        class_counts = {}
        
        with torch.no_grad():
            for i, img_path in enumerate(images):
                try:
                    # 加载图片
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(dev)
                    
                    # 推理
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_idx = probs.argmax(dim=1).item()
                    confidence = probs[0, pred_idx].item()
                    
                    # 获取类别名
                    pred_class = idx_to_class.get(pred_idx, str(pred_idx))
                    
                    results.append({
                        'image': str(img_path),
                        'prediction': pred_class,
                        'confidence': f"{confidence:.4f}",
                        'class_idx': pred_idx,
                    })
                    
                    # 统计
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    
                    # 复制图片到类别文件夹
                    if copy_images:
                        class_dir = output_dir / pred_class
                        class_dir.mkdir(exist_ok=True)
                        dst_path = class_dir / img_path.name
                        # 处理重名
                        if dst_path.exists():
                            stem = img_path.stem
                            suffix = img_path.suffix
                            dst_path = class_dir / f"{stem}_{i}{suffix}"
                        shutil.copy2(img_path, dst_path)
                    
                    inference_state.processed_images = i + 1
                    
                    # 每10张输出一次进度
                    if (i + 1) % 10 == 0 or (i + 1) == len(images):
                        inference_state.add_log(f"   进度: {i+1}/{len(images)}")
                        
                except Exception as e:
                    inference_state.add_log(f"   ⚠️ 跳过 {img_path.name}: {str(e)}")
        
        inference_state.results = results
        inference_state.output_dir = str(output_dir)
        
        # 保存结果
        csv_path = output_dir / "predictions.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        inference_state.add_log(f"\n📊 结果已保存: {csv_path}")
        
        # 输出统计
        inference_state.add_log("\n📈 类别分布:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            inference_state.add_log(f"   {cls}: {count} ({pct:.1f}%)")
        
        # 完成
        elapsed = time.time() - inference_state.start_time
        inference_state.add_log("\n" + "━" * 50)
        inference_state.add_log(f"✅ 推理完成！处理 {len(results)} 张图片，用时 {elapsed:.1f} 秒")
        if copy_images:
            inference_state.add_log(f"📁 图片已按类别整理到: {output_dir}")
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


# ==================== 创建界面 ====================
def create_ui():
    """创建Gradio界面"""
    global training_state
    
    env_status = check_environment()
    gpu_options = get_gpu_options()
    trained_models = get_trained_models()
    
    # 加载保存的参数
    saved_data = {} if not dlhub_params else dlhub_params.get_section('data')
    saved_model = {} if not dlhub_params else dlhub_params.get_section('model')
    saved_training = {} if not dlhub_params else dlhub_params.get_section('training')
    saved_conversion = {} if not dlhub_params else dlhub_params.get_section('conversion')
    saved_inference = {} if not dlhub_params else dlhub_params.get_section('inference')
    saved_device = get_saved_param('device', gpu_options[0] if gpu_options else 'CPU')
    
    # 【增强】加载训练历史数据（用于恢复曲线）
    saved_history = {} if not dlhub_params else dlhub_params.get_history()
    saved_logs = [] if not dlhub_params else dlhub_params.get_logs('training')
    saved_conv_logs = [] if not dlhub_params else dlhub_params.get_logs('conversion')
    saved_inf_logs = [] if not dlhub_params else dlhub_params.get_logs('inference')
    
    # 【增强】如果有历史数据，恢复到training_state
    if saved_history:
        if saved_history.get('train_losses'):
            training_state.train_losses = saved_history.get('train_losses', [])
            training_state.val_losses = saved_history.get('val_losses', [])
            training_state.val_accs = saved_history.get('val_accs', [])
            training_state.best_acc = saved_history.get('best_acc', 0.0)
            training_state.best_epoch = saved_history.get('best_epoch', 0)
            training_state.current_epoch = saved_history.get('current_epoch', 0)
            training_state.total_epochs = saved_history.get('total_epochs', 0)
            training_state.output_dir = saved_history.get('output_dir', '')
            print(f"[DL-Hub] ✓ 已恢复训练历史: {len(training_state.train_losses)} epochs, 最佳 {training_state.best_acc:.2f}%")
    
    # 【增强】如果有日志，恢复到各个state
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
        title="TIMM 图像分类工具",
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
            ">🔬 TIMM 图像分类工具</h1>
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
                            gr.Markdown("**📂 数据集**")
                            data_path = gr.Textbox(
                                label="数据集路径",
                                placeholder="输入数据集文件夹路径...",
                                info="ImageFolder格式: 每个类别一个子文件夹",
                                value=saved_data.get('data_path', '')
                            )
                            data_status = gr.Textbox(
                                label="验证状态",
                                interactive=False,
                                value="⏳ 请输入数据集路径",
                                lines=4,
                                max_lines=6
                            )
                        
                        # 模型选择
                        with gr.Group():
                            gr.Markdown("**🧠 模型选择**")
                            try:
                                from config.model_registry import get_model_families, get_model_scales
                                families = get_model_families()
                                default_scales = get_model_scales(families[0]) if families else []
                            except Exception:
                                families = ["EfficientNet"]
                                default_scales = ["超小", "小", "中", "大", "超大"]
                            
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
                                lines=2
                            )
                            weights_status = gr.Textbox(
                                label="权重状态",
                                interactive=False
                            )
                        
                        # 训练参数
                        with gr.Group():
                            gr.Markdown("**📊 训练参数**")
                            with gr.Row():
                                epochs = gr.Slider(1, 200, value=saved_training.get('epochs', 50), step=1, label="训练轮数")
                                batch_size = gr.Slider(4, 128, value=saved_training.get('batch_size', 32), step=4, label="批量大小")
                            with gr.Row():
                                learning_rate = gr.Number(value=saved_training.get('learning_rate', 1e-3), label="学习率")
                                optimizer = gr.Dropdown(
                                    choices=["AdamW", "Adam", "SGD"],
                                    value=saved_training.get('optimizer', 'AdamW'),
                                    label="优化器"
                                )
                            with gr.Row():
                                img_size = gr.Slider(128, 512, value=saved_data.get('image_size', 224), step=32, label="图像尺寸")
                                val_split = gr.Slider(0.1, 0.4, value=saved_data.get('val_split', 0.2), step=0.05, label="验证集比例")
                        
                        # 高级设置
                        with gr.Accordion("🔧 高级设置", open=False):
                            with gr.Row():
                                save_freq = gr.Slider(1, 50, value=saved_training.get('save_freq', 10), step=1, label="保存频率")
                                early_stopping = gr.Checkbox(value=saved_training.get('early_stopping', True), label="早停")
                            gpu_choice = gr.Dropdown(
                                choices=gpu_options,
                                value=saved_device if saved_device in gpu_options else (gpu_options[0] if gpu_options else "CPU"),
                                label="计算设备"
                            )
                        
                        # 控制按钮
                        with gr.Row():
                            train_btn = gr.Button("🚀 开始训练", variant="primary", elem_classes="primary-btn")
                            stop_btn = gr.Button("⏹️ 停止", variant="stop", elem_classes="stop-btn")
                        
                        train_status = gr.Textbox(label="状态", interactive=False, lines=1)
                    
                    # 右侧：监控面板
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 训练监控")
                        
                        progress_html = gr.HTML(get_progress_html())
                        status_text = gr.Textbox(label="当前状态", interactive=False, lines=1, value="⏳ 等待开始...")
                        
                        with gr.Row():
                            loss_plot = gr.Plot(label="Loss曲线")
                            acc_plot = gr.Plot(label="Accuracy曲线")
                        
                        train_logs = gr.Textbox(
                            label="📝 训练日志",
                            lines=22,
                            max_lines=30,
                            interactive=False,
                            elem_classes="log-box"
                        )
                
                # 定时刷新
                train_timer = gr.Timer(value=2)
                train_timer.tick(
                    fn=get_training_status,
                    outputs=[progress_html, status_text, train_logs, loss_plot, acc_plot]
                )
                
                # 事件绑定
                data_path.change(fn=validate_data_path, inputs=data_path, outputs=data_status)
                
                def update_scales(family):
                    try:
                        from config.model_registry import get_model_scales
                        scales = get_model_scales(family)
                        return gr.update(choices=scales, value=scales[0] if scales else None)
                    except Exception:
                        return gr.update()
                
                model_family.change(fn=update_scales, inputs=model_family, outputs=model_scale)
                model_family.change(fn=update_model_info, inputs=[model_family, model_scale], outputs=model_info)
                model_scale.change(fn=update_model_info, inputs=[model_family, model_scale], outputs=model_info)
                model_family.change(fn=check_pretrained_weights, inputs=[model_family, model_scale], outputs=weights_status)
                model_scale.change(fn=check_pretrained_weights, inputs=[model_family, model_scale], outputs=weights_status)
                
                train_btn.click(
                    fn=on_start_training,
                    inputs=[
                        data_path, model_family, model_scale,
                        epochs, batch_size, learning_rate, optimizer,
                        img_size, val_split, save_freq, early_stopping, gpu_choice
                    ],
                    outputs=[train_status, train_logs]
                )
                stop_btn.click(fn=on_stop_training, outputs=train_status)
            
            # ========== Tab 2: 模型转换 ==========
            with gr.TabItem("🔄 模型转换"):
                gr.Markdown("""
                ### 📝 说明
                将训练好的 PyTorch 模型转换为高性能推理引擎格式（TensorRT/ONNX Runtime/OpenVINO）。
                - **INT8/Mixed 精度**：需要提供校准数据集
                - **固定输入尺寸**：不支持动态尺寸，仅支持 TensorRT 动态批处理
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
                                label="校准数据集路径",
                                placeholder="输入ImageFolder格式的数据集路径...",
                                info="用于INT8量化校准，需要与训练数据分布一致"
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
                        
                        # 动态Batch配置 - 所有后端通用
                        with gr.Group(visible=True) as dynamic_batch_group:
                            gr.Markdown("**⚡ 动态Batch配置**")
                            dynamic_batch = gr.Checkbox(
                                value=saved_conversion.get('dynamic_batch', False), 
                                label="启用动态批处理",
                                info="允许运行时使用不同batch size（TensorRT需要配置，OpenVINO需要显式启用，ORT自动支持）"
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
                
                # 后端切换
                target_backend.change(
                    fn=update_backend_options,
                    inputs=target_backend,
                    outputs=[trt_group, workspace_gb, dynamic_batch, min_batch, opt_batch, max_batch, dynamic_batch_group]
                )
                
                # 精度切换 - 显示/隐藏校准数据集
                precision.change(
                    fn=update_calib_visibility,
                    inputs=precision,
                    outputs=calib_group
                )
                
                # 校准数据集路径验证
                calib_data_path.change(fn=validate_calib_data, inputs=calib_data_path, outputs=calib_status)
                calib_data_path.submit(fn=validate_calib_data, inputs=calib_data_path, outputs=calib_status)
                
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
                使用训练好的模型对图片进行批量分类预测。
                - 支持单张图片或文件夹（递归扫描子目录）
                - 可选择将图片按预测类别复制到对应文件夹
                - 输出 CSV 格式预测结果
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
                            gr.Markdown("**📁 输入输出**")
                            input_path = gr.Textbox(
                                label="输入路径",
                                value=saved_inference.get('input_path', ''),
                                placeholder="图片文件或文件夹路径",
                                info="支持 jpg/jpeg/png/bmp/gif/webp，递归扫描子文件夹"
                            )
                            inf_output_dir = gr.Textbox(
                                label="输出目录",
                                value=saved_inference.get('output_dir', './inference_output'),
                                placeholder="预测结果和分类图片保存路径"
                            )
                        
                        with gr.Group():
                            gr.Markdown("**⚙️ 推理设置**")
                            inf_device = gr.Dropdown(
                                choices=gpu_options,
                                value=saved_inference.get('device', gpu_options[0] if gpu_options else "CPU"),
                                label="设备"
                            )
                            copy_images = gr.Checkbox(
                                value=saved_inference.get('copy_images', True),
                                label="复制图片到类别文件夹",
                                info="按预测类别整理图片，便于人工校验"
                            )
                        
                        inf_btn = gr.Button("🚀 开始推理", variant="primary", elem_classes="primary-btn")
                        inf_status_text = gr.Textbox(label="状态", interactive=False, lines=1)
                    
                    # 右侧：日志
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 推理日志")
                        inf_logs = gr.Textbox(
                            label="日志",
                            lines=30,
                            interactive=False,
                            elem_classes="log-box"
                        )
                
                # 定时刷新
                inf_timer = gr.Timer(value=2)
                inf_timer.tick(
                    fn=get_inference_status,
                    outputs=[inf_status_text, inf_logs]
                )
                
                # 事件绑定
                # 下拉框展开时，扫描文件夹
                inf_model_folder.focus(fn=scan_model_folders, outputs=inf_model_folder)
                # 选择文件夹后，扫描模型文件
                inf_model_folder.change(fn=scan_models_in_selected_folder, inputs=inf_model_folder, outputs=inf_model_dropdown)
                
                inf_btn.click(
                    fn=on_start_inference,
                    inputs=[
                        inf_model_dropdown, input_path, inf_output_dir,
                        inf_device, copy_images
                    ],
                    outputs=[inf_status_text, inf_logs]
                )
    
    return app


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 【关键】绕过系统代理，防止Gradio 6.0自检被代理拦截
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
    os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    
    # 解析命令行参数（DL-Hub 兼容）
    parser = argparse.ArgumentParser(description='TIMM 图像分类训练工具')
    parser.add_argument('--task-dir', type=str, default=None, help='DL-Hub 任务目录')
    parser.add_argument('--port', type=int, default=7860, help='Gradio 服务端口')
    args, _ = parser.parse_known_args()
    
    # 创建应用
    app = create_ui()
    
    # 确定启动配置
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode:
        # DL-Hub 模式
        launch_port = dlhub_adapter.port
        launch_inbrowser = False
        print(f"[DL-Hub] 以 DL-Hub 模式启动，端口: {launch_port}")
    else:
        # 独立模式
        launch_port = args.port
        launch_inbrowser = True
        print(f"[独立模式] 启动端口: {launch_port}")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=launch_inbrowser,
    )
