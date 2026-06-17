"""
YOLO训练引擎 - 封装ultralytics训练流程
提供统一的回调接口，支持实时监控和中断控制
"""

import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np

# 导入本地模型查找函数
try:
    from models.model_factory import get_pretrained_path, get_pretrained_dir
except ImportError:
    # 如果导入失败，定义简单的备用函数
    def get_pretrained_path(model_name, ultralytics_name=None):
        return None
    def get_pretrained_dir():
        return Path("./pretrained")


@dataclass
class TrainingCallback:
    """训练回调函数集合"""
    on_epoch_start: Optional[Callable[[int, int], None]] = None  # (epoch, total_epochs)
    on_epoch_end: Optional[Callable[[int, Dict], None]] = None  # (epoch, metrics)
    on_batch_end: Optional[Callable[[int, int, Dict], None]] = None  # (batch, total, metrics)
    on_train_end: Optional[Callable[[Dict], None]] = None  # (final_metrics)
    on_log: Optional[Callable[[str], None]] = None  # (message)
    on_detection_preview: Optional[Callable[[Any], None]] = None  # 检测预览回调
    should_stop: Optional[Callable[[], bool]] = None  # 检查是否需要停止


@dataclass
class TrainingMetrics:
    """训练指标记录"""
    box_losses: List[float] = field(default_factory=list)
    cls_losses: List[float] = field(default_factory=list)
    dfl_losses: List[float] = field(default_factory=list)
    val_map50: List[float] = field(default_factory=list)
    val_map50_95: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_map50: float = 0.0
    best_map50_95: float = 0.0
    best_epoch: int = 0


class YOLOTrainer:
    """
    YOLO训练器
    封装ultralytics的YOLO训练，提供统一的回调接口
    """
    
    def __init__(
        self,
        model_name: str,
        data_yaml: str,
        output_dir: str,
        # 基础参数
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        learning_rate: float = 0.01,
        optimizer: str = 'SGD',
        patience: int = 50,
        freeze_backbone: bool = False,
        device: str = '0',
        workers: int = 8,
        # 高级参数
        weight_decay: float = 0.0005,
        momentum: float = 0.937,
        warmup_epochs: float = 3.0,
        cos_lr: bool = True,
        # 数据增强
        mosaic: float = 1.0,
        mixup: float = 0.0,
        fliplr: float = 0.5,
        # 保存设置
        save_period: int = 0,  # 每隔多少epoch保存一次，0表示不保存
        # 回调
        callback: TrainingCallback = None,
        # 类别信息
        class_names: List[str] = None,
    ):
        """
        初始化训练器
        
        Args:
            model_name: 模型名称或路径 (如 'yolov8n.pt')
            data_yaml: 数据集配置文件路径
            output_dir: 输出目录
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 输入图像尺寸
            learning_rate: 初始学习率
            optimizer: 优化器 ('SGD', 'Adam', 'AdamW')
            patience: 早停耐心值
            freeze_backbone: 是否冻结backbone
            device: 设备 ('0', '0,1', 'cpu')
            workers: 数据加载线程数
            weight_decay: 权重衰减
            momentum: SGD动量
            warmup_epochs: 预热轮数
            cos_lr: 是否使用余弦学习率
            mosaic: 马赛克增强概率
            mixup: MixUp增强概率
            fliplr: 左右翻转概率
            save_period: 每隔多少epoch保存一次模型，0表示不保存中间模型
            callback: 训练回调
            class_names: 类别名称列表
        """
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.patience = patience
        self.freeze_backbone = freeze_backbone
        self.device = device
        self.workers = workers
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.cos_lr = cos_lr
        self.mosaic = mosaic
        self.mixup = mixup
        self.fliplr = fliplr
        self.save_period = save_period
        self.callback = callback or TrainingCallback()
        self.class_names = class_names or []
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 指标记录
        self.metrics = TrainingMetrics()
        
        # 停止标志
        self._stop_requested = False
        
        # YOLO模型
        self.yolo = None
        
        # 当前epoch（用于回调）
        self.current_epoch = 0
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        
        # 写入日志文件
        log_file = self.output_dir / 'training.log'
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(full_message + '\n')
        except Exception:
            pass
        
        # 回调
        if self.callback.on_log:
            self.callback.on_log(full_message)
    
    def train(self) -> Dict[str, Any]:
        """
        开始训练
        
        Returns:
            训练结果字典
        """
        from ultralytics import YOLO
        
        self.log(f"🚀 开始训练: {self.model_name}")
        self.log(f"📊 数据集: {self.data_yaml}")
        self.log(f"📁 输出目录: {self.output_dir}")
        
        start_time = time.time()
        
        # 加载模型 - 优先使用本地缓存
        model_path = self.model_name
        
        # 从模型名称中提取基础名称（去掉.pt后缀和可能的u后缀）
        base_name = self.model_name.replace('.pt', '').rstrip('u')
        
        # 检查本地缓存
        local_path = get_pretrained_path(base_name, self.model_name)
        if local_path and local_path.exists():
            model_path = str(local_path)
            self.log(f"🔧 加载本地模型: {model_path}")
        else:
            self.log(f"🔧 加载模型: {self.model_name} (将从网络下载)")
            pretrained_dir = get_pretrained_dir()
            self.log(f"   💡 提示: 可将模型文件放入 {pretrained_dir} 避免下载")
        
        self.yolo = YOLO(model_path)
        
        # 构建训练参数
        train_args = {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'lr0': self.learning_rate,
            'optimizer': self.optimizer,
            'patience': self.patience,
            'device': self.device,
            'workers': self.workers,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            'pretrained': True,
            # 高级参数
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'warmup_epochs': self.warmup_epochs,
            'cos_lr': self.cos_lr,
            # 数据增强
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            'fliplr': self.fliplr,
            # 保存设置
            'save': True,
            'save_period': -1,  # 只保存best和last
            'plots': True,
            'val': True,
            # 其他
            'verbose': True,
        }
        
        # 冻结backbone（只在需要时添加freeze参数）
        if self.freeze_backbone:
            train_args['freeze'] = 10  # 冻结前10层（backbone）
        
        self.log(f"⚙️ 训练参数:")
        self.log(f"   epochs={self.epochs}, batch={self.batch_size}")
        self.log(f"   imgsz={self.img_size}, lr={self.learning_rate}")
        self.log(f"   optimizer={self.optimizer}, patience={self.patience}")
        if self.freeze_backbone:
            self.log(f"   freeze_backbone=True (冻结前10层)")
        
        # 注册ultralytics回调
        self._register_callbacks()
        
        try:
            # 开始训练
            results = self.yolo.train(**train_args)
            
            # 训练完成
            total_time = time.time() - start_time
            
            # 获取最佳指标
            best_map50 = self.metrics.best_map50
            best_map50_95 = self.metrics.best_map50_95
            
            self.log("━" * 50)
            self.log(f"✅ 训练完成!")
            self.log(f"🏆 最佳 mAP@50: {best_map50:.4f}")
            self.log(f"🏆 最佳 mAP@50:95: {best_map50_95:.4f}")
            self.log(f"🏆 最佳 Epoch: {self.metrics.best_epoch}")
            self.log(f"⏱️ 总用时: {total_time/60:.1f} 分钟")
            self.log("━" * 50)
            
            # 转换checkpoint格式
            self._convert_checkpoints()
            
            final_result = {
                'best_map50': best_map50,
                'best_map50_95': best_map50_95,
                'best_epoch': self.metrics.best_epoch,
                'total_epochs': self.current_epoch,
                'total_time': total_time,
                'output_dir': str(self.output_dir),
            }
            
            if self.callback.on_train_end:
                self.callback.on_train_end(final_result)
            
            return final_result
            
        except Exception as e:
            self.log(f"❌ 训练出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            raise
    
    def _register_callbacks(self):
        """注册ultralytics回调函数"""
        
        def on_train_epoch_end(trainer):
            """每个epoch训练结束时调用"""
            try:
                self.current_epoch = trainer.epoch + 1
                
                # 检查是否需要停止
                if self.callback.should_stop and self.callback.should_stop():
                    trainer.stop = True
                    self.log("⏹️ 收到停止信号，正在保存模型...")
            except Exception as e:
                pass  # 忽略回调中的错误，不影响训练
        
        def on_fit_epoch_end(trainer):
            """每个epoch验证结束时调用"""
            try:
                epoch = trainer.epoch + 1
                self.current_epoch = epoch
                
                # 获取损失（使用detach()避免影响梯度）
                # YOLO26 移除了DFL，loss_items只有2个元素(box, cls)
                # YOLOv5/v8/v11 有3个元素(box, cls, dfl)
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    try:
                        losses = trainer.loss_items.detach().cpu().numpy() if hasattr(trainer.loss_items, 'detach') else trainer.loss_items
                        if len(losses) >= 3:
                            # YOLOv5/v8/v11: box, cls, dfl
                            self.metrics.box_losses.append(float(losses[0]))
                            self.metrics.cls_losses.append(float(losses[1]))
                            self.metrics.dfl_losses.append(float(losses[2]))
                        elif len(losses) >= 2:
                            # YOLO26: box, cls (no DFL)
                            self.metrics.box_losses.append(float(losses[0]))
                            self.metrics.cls_losses.append(float(losses[1]))
                            self.metrics.dfl_losses.append(0.0)  # YOLO26无DFL
                    except Exception:
                        pass
                
                # 获取验证指标 - 尝试多种方式获取
                map50 = 0.0
                map50_95 = 0.0
                
                # 方式1：从trainer.metrics获取
                if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                    metrics = trainer.metrics
                    if hasattr(metrics, 'box'):
                        map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
                        map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0
                        
                        # 获取P/R
                        if hasattr(metrics.box, 'mp'):
                            self.metrics.precision.append(float(metrics.box.mp))
                        if hasattr(metrics.box, 'mr'):
                            self.metrics.recall.append(float(metrics.box.mr))
                
                # 方式2：从trainer.validator.metrics获取（如果方式1失败）
                if map50 == 0.0 and hasattr(trainer, 'validator') and trainer.validator is not None:
                    validator = trainer.validator
                    if hasattr(validator, 'metrics') and validator.metrics is not None:
                        metrics = validator.metrics
                        if hasattr(metrics, 'box'):
                            map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
                            map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0
                
                # 方式3：直接从fitness获取（如果还是0）
                if map50 == 0.0 and hasattr(trainer, 'fitness') and trainer.fitness is not None:
                    map50_95 = float(trainer.fitness)
                
                # 保存指标
                if map50 > 0 or map50_95 > 0:
                    self.metrics.val_map50.append(map50)
                    self.metrics.val_map50_95.append(map50_95)
                    
                    # 检查是否是最佳
                    if map50_95 > self.metrics.best_map50_95:
                        self.metrics.best_map50 = map50
                        self.metrics.best_map50_95 = map50_95
                        self.metrics.best_epoch = epoch
                else:
                    # 如果获取失败，添加0作为占位
                    self.metrics.val_map50.append(0.0)
                    self.metrics.val_map50_95.append(0.0)
                
                # 获取学习率
                if hasattr(trainer, 'lf') and hasattr(trainer, 'optimizer'):
                    try:
                        lr = trainer.optimizer.param_groups[0]['lr']
                        self.metrics.learning_rates.append(float(lr))
                    except Exception:
                        pass
                
                # 日志
                box_loss = self.metrics.box_losses[-1] if self.metrics.box_losses else 0
                cls_loss = self.metrics.cls_losses[-1] if self.metrics.cls_losses else 0
                map50 = self.metrics.val_map50[-1] if self.metrics.val_map50 else 0
                map50_95 = self.metrics.val_map50_95[-1] if self.metrics.val_map50_95 else 0
                
                self.log(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"box: {box_loss:.4f} | cls: {cls_loss:.4f} | "
                    f"mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}"
                )
                
                # epoch回调
                if self.callback.on_epoch_end:
                    epoch_metrics = {
                        'epoch': epoch,
                        'box_loss': box_loss,
                        'cls_loss': cls_loss,
                        'dfl_loss': self.metrics.dfl_losses[-1] if self.metrics.dfl_losses else 0,
                        'map50': map50,
                        'map50_95': map50_95,
                        'precision': self.metrics.precision[-1] if self.metrics.precision else 0,
                        'recall': self.metrics.recall[-1] if self.metrics.recall else 0,
                        'best_map50': self.metrics.best_map50,
                        'best_map50_95': self.metrics.best_map50_95,
                        'best_epoch': self.metrics.best_epoch,
                        'lr': self.metrics.learning_rates[-1] if self.metrics.learning_rates else self.learning_rate,
                    }
                    self.callback.on_epoch_end(epoch, epoch_metrics)
                
                # 周期性保存模型
                if self.save_period > 0 and epoch % self.save_period == 0:
                    self._save_periodic_checkpoint(epoch, trainer)
                
                # 注意：不在训练过程中生成检测预览，因为会破坏梯度计算图
                # 用户可以通过"测试推理"按钮手动查看检测效果
                        
            except Exception as e:
                pass  # 忽略回调中的错误，不影响训练
        
        # 添加回调
        self.yolo.add_callback('on_train_epoch_end', on_train_epoch_end)
        self.yolo.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    
    def _generate_detection_preview(self):
        """生成检测预览图"""
        import torch
        from data.dataset import get_val_images
        from data.visualizer import visualize_ultralytics_result, create_preview_grid
        from PIL import Image
        
        try:
            # 获取验证集图像
            val_images = get_val_images(self.data_yaml, num_images=4)
            if not val_images:
                return None
            
            # 保存当前训练状态
            was_training = self.yolo.model.training if hasattr(self.yolo, 'model') else True
            
            # 运行推理（确保不影响训练梯度）
            with torch.no_grad():
                results = self.yolo.predict(
                    source=val_images,
                    conf=0.25,
                    save=False,
                    verbose=False,
                )
            
            # 恢复训练模式
            if was_training and hasattr(self.yolo, 'model'):
                self.yolo.model.train()
            
            # 可视化每个结果
            preview_images = []
            for result in results:
                vis_img = visualize_ultralytics_result(result, conf_threshold=0.25)
                if vis_img:
                    preview_images.append(vis_img)
            
            if not preview_images:
                return None
            
            # 创建网格
            if len(preview_images) == 1:
                return preview_images[0]
            else:
                return create_preview_grid(preview_images, grid_size=(2, 2))
                
        except Exception as e:
            # 确保即使出错也恢复训练模式
            try:
                if hasattr(self.yolo, 'model'):
                    self.yolo.model.train()
            except Exception:
                pass
            return None
    
    def _convert_checkpoints(self):
        """
        转换checkpoint为兼容格式
        
        最终目录结构 (只保留 weights_model_importer):
        output/YOLOv8n_xxx/
        ├── weights_model_importer/       # 兼容 model_importer
        │   ├── best_model.pt
        │   ├── last_model.pt
        │   └── {model}_epoch{N}.pt       # 周期性保存的模型
        └── model_metadata.json
        
        注：weights 文件夹在处理完成后会被删除
        """
        import shutil
        
        weights_dir = self.output_dir / 'weights'
        
        # 创建 model_importer 兼容目录
        importer_dir = self.output_dir / 'weights_model_importer'
        importer_dir.mkdir(parents=True, exist_ok=True)
        
        # ============================================
        # 处理 best.pt
        # ============================================
        best_pt = weights_dir / 'best.pt'
        if best_pt.exists():
            best_output = importer_dir / 'best_model.pt'
            try:
                self._save_with_metadata(best_pt, best_output, is_best=True)
                self.log(f"💾 已保存: weights_model_importer/best_model.pt")
            except Exception as e:
                self.log(f"⚠️ 保存best_model.pt失败: {e}")
            
            # 保存元数据 JSON 文件
            self._save_metadata_json(self.output_dir / 'model_metadata.json', is_best=True)
        
        # ============================================
        # 处理 last.pt
        # ============================================
        last_pt = weights_dir / 'last.pt'
        if last_pt.exists():
            last_output = importer_dir / 'last_model.pt'
            try:
                self._save_with_metadata(last_pt, last_output, is_best=False)
                self.log(f"💾 已保存: weights_model_importer/last_model.pt")
            except Exception as e:
                self.log(f"⚠️ 保存last_model.pt失败: {e}")
        
        # ============================================
        # 移动周期性保存的模型到 weights_model_importer
        # ============================================
        model_base_name = self.model_name.replace('.pt', '').replace('.yaml', '')
        for pt_file in self.output_dir.glob(f"{model_base_name}_epoch*.pt"):
            try:
                dest = importer_dir / pt_file.name
                shutil.move(str(pt_file), str(dest))
                self.log(f"💾 已移动: {pt_file.name} → weights_model_importer/")
            except Exception as e:
                self.log(f"⚠️ 移动{pt_file.name}失败: {e}")
        
        # ============================================
        # 删除 weights 文件夹（只保留 weights_model_importer）
        # ============================================
        best_ok = (importer_dir / 'best_model.pt').exists() if best_pt.exists() else True
        last_ok = (importer_dir / 'last_model.pt').exists() if last_pt.exists() else True

        if weights_dir.exists() and best_ok and last_ok:
            try:
                shutil.rmtree(weights_dir)
                self.log(f"🗑️ 已删除: weights/ (只保留 weights_model_importer/)")
            except Exception as e:
                self.log(f"⚠️ 删除weights文件夹失败: {e}")
        elif weights_dir.exists():
            self.log(f"⚠️ 跳过删除 weights/（复制未完全成功，保留原始文件作为备份）")
    
    def _save_with_metadata(self, src_pt: Path, dst_pt: Path, is_best: bool = True):
        """
        保存完全兼容 model_importer.py 的 checkpoint
        
        model_importer.py 的 YOLODetectionHandler 需要以下字段:
        
        1. can_handle() 检查:
           - framework == 'ultralytics'  ← 最简单的识别方式
           - 或 '_original_model' in checkpoint
        
        2. get_model_name() 需要:
           - checkpoint['yaml'] 推断版本 (yolov5/yolov8/yolov11)
           - 或 checkpoint['_original_model'] 类名推断
           - 或从文件名推断
        
        3. rebuild() 需要:
           - checkpoint['_original_model'] 直接使用原始模型
        
        4. get_num_classes() 需要:
           - checkpoint['nc']
           - 或 checkpoint['names']
           - 或 checkpoint['_original_model'].nc
           - 或 checkpoint['yaml']['nc']
        """
        import torch
        
        # 加载原始 ultralytics checkpoint
        original_ckpt = torch.load(src_pt, map_location='cpu', weights_only=False)
        
        # 获取类别信息
        nc = len(self.class_names) if self.class_names else 80
        names_dict = {i: name for i, name in enumerate(self.class_names)} if self.class_names else {}
        
        # ============================================
        # 1. framework 字段 - can_handle() 识别用
        # ============================================
        original_ckpt['framework'] = 'ultralytics'
        
        # ============================================
        # 2. _original_model 字段 - rebuild() 直接使用
        # ============================================
        # 从原始checkpoint的model字段获取模型对象
        if 'model' in original_ckpt and original_ckpt['model'] is not None:
            model_obj = original_ckpt['model']
            # 确保是模型对象而不是state_dict
            if hasattr(model_obj, 'state_dict'):
                original_ckpt['_original_model'] = model_obj
        
        # ============================================
        # 3. nc 和 names 字段 - get_num_classes() 使用
        # ============================================
        original_ckpt['nc'] = nc
        original_ckpt['names'] = names_dict
        
        # ============================================
        # 4. yaml 字段 - get_model_name() 推断版本用
        # ============================================
        # 构建yaml配置，包含模型版本信息
        model_name_lower = self.model_name.lower().replace('.pt', '')
        yaml_config = {
            'nc': nc,
            'yaml_file': f"{model_name_lower}.yaml",  # 用于版本推断
        }
        
        # 如果原始checkpoint有yaml，合并
        if 'yaml' in original_ckpt and isinstance(original_ckpt['yaml'], dict):
            yaml_config.update(original_ckpt['yaml'])
            yaml_config['nc'] = nc  # 确保nc正确
        
        original_ckpt['yaml'] = yaml_config
        
        # ============================================
        # 5. train_args 字段 - 训练参数
        # ============================================
        if 'train_args' not in original_ckpt or original_ckpt['train_args'] is None:
            original_ckpt['train_args'] = {}
        
        if not isinstance(original_ckpt['train_args'], dict):
            original_ckpt['train_args'] = {}
        
        original_ckpt['train_args'].update({
            'nc': nc,
            'imgsz': self.img_size,
            'model': f"{self.model_name}",
            'epochs': self.epochs,
            'batch': self.batch_size,
            'lr0': self.learning_rate,
            'optimizer': self.optimizer,
        })
        
        # ============================================
        # 6. custom_metadata 字段 - 预处理信息 (额外)
        # ============================================
        original_ckpt['custom_metadata'] = {
            'model_name': self.model_name.replace('.pt', ''),
            'num_classes': nc,
            'class_names': self.class_names,
            'input_size': self.img_size,
            'input_spec': {
                'shape': (1, 3, self.img_size, self.img_size),
                'color_format': 'RGB',
                'pixel_range': (0, 255),
                'normalize_method': 'divide_255',
                'normalize_mean': (0.0, 0.0, 0.0),
                'normalize_std': (1.0, 1.0, 1.0),
                'value_range': (0.0, 1.0),
                'letterbox_color': (114, 114, 114),
            },
            'train_info': {
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'lr0': self.learning_rate,
                'optimizer': self.optimizer,
            },
            'best_epoch': self.metrics.best_epoch if is_best else self.current_epoch,
            'best_map50': self.metrics.best_map50 if is_best else 0,
            'best_map50_95': self.metrics.best_map50_95 if is_best else 0,
        }
        
        # 保存
        torch.save(original_ckpt, dst_pt)
        
        # 打印确认信息
        has_original_model = '_original_model' in original_ckpt
        self.log(f"   ✅ 已添加字段: framework=ultralytics, nc={nc}, names={len(names_dict)}类")
        self.log(f"   ✅ _original_model: {'有' if has_original_model else '无'}, yaml: 有")
    
    def _save_metadata_json(self, json_path: Path, is_best: bool = True):
        """保存元数据为 JSON 文件 (方便查看和调试)"""
        import json
        
        metadata = {
            'model_name': self.model_name.replace('.pt', ''),
            'num_classes': len(self.class_names) if self.class_names else 80,
            'class_names': self.class_names,
            'input_size': self.img_size,
            'input_spec': {
                'shape': [1, 3, self.img_size, self.img_size],
                'color_format': 'RGB',
                'pixel_range': [0, 255],
                'normalize_method': 'divide_255',
                'normalize_mean': [0.0, 0.0, 0.0],
                'normalize_std': [1.0, 1.0, 1.0],
                'value_range': [0.0, 1.0],
                'letterbox_color': [114, 114, 114],
            },
            'train_args': {
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'lr0': self.learning_rate,
                'optimizer': self.optimizer,
            },
            'metrics': {
                'best_epoch': self.metrics.best_epoch,
                'best_map50': self.metrics.best_map50,
                'best_map50_95': self.metrics.best_map50_95,
            },
            'usage': {
                'onnx_export': '使用 weights/best.pt 进行 ONNX 导出: yolo export model=weights/best.pt format=onnx',
                'model_importer': '使用 weights_model_importer/best_model.pt',
                'ultralytics_load': "from ultralytics import YOLO; model = YOLO('weights/best.pt')",
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _save_periodic_checkpoint(self, epoch: int, trainer):
        """
        周期性保存模型 checkpoint
        
        保存格式与 best_model.pt 相同，完全兼容 model_importer.py
        文件名格式: {model_name}_epoch{epoch}.pt
        """
        import torch
        
        try:
            # 获取当前模型
            model = trainer.model if hasattr(trainer, 'model') else None
            if model is None:
                self.log(f"⚠️ Epoch {epoch}: 无法获取模型，跳过保存")
                return
            
            # 生成文件名
            model_base_name = self.model_name.replace('.pt', '').replace('.yaml', '')
            save_name = f"{model_base_name}_epoch{epoch}.pt"
            save_path = self.output_dir / save_name
            
            # 获取类别信息
            nc = len(self.class_names) if self.class_names else 80
            names_dict = {i: name for i, name in enumerate(self.class_names)} if self.class_names else {}
            
            # 构建 checkpoint
            ckpt = {
                # model_importer.py 识别字段
                'framework': 'ultralytics',
                '_original_model': model,
                'nc': nc,
                'names': names_dict,
                
                # yaml 字段 - get_model_name() 推断版本
                'yaml': {
                    'nc': nc,
                    'yaml_file': f"{model_base_name}.yaml",
                },
                
                # 训练参数
                'train_args': {
                    'nc': nc,
                    'imgsz': self.img_size,
                    'model': self.model_name,
                    'epochs': self.epochs,
                    'batch': self.batch_size,
                    'lr0': self.learning_rate,
                    'optimizer': self.optimizer,
                },
                
                # 训练状态
                'epoch': epoch,
                'model': model,
                
                # 当前指标
                'metrics': {
                    'map50': self.metrics.val_map50[-1] if self.metrics.val_map50 else 0,
                    'map50_95': self.metrics.val_map50_95[-1] if self.metrics.val_map50_95 else 0,
                },
                
                # 预处理信息
                'custom_metadata': {
                    'model_name': model_base_name,
                    'num_classes': nc,
                    'class_names': self.class_names,
                    'input_size': self.img_size,
                    'input_spec': {
                        'shape': (1, 3, self.img_size, self.img_size),
                        'color_format': 'RGB',
                        'pixel_range': (0, 255),
                        'normalize_method': 'divide_255',
                        'normalize_mean': (0.0, 0.0, 0.0),
                        'normalize_std': (1.0, 1.0, 1.0),
                        'value_range': (0.0, 1.0),
                        'letterbox_color': (114, 114, 114),
                    },
                    'epoch': epoch,
                    'map50': self.metrics.val_map50[-1] if self.metrics.val_map50 else 0,
                    'map50_95': self.metrics.val_map50_95[-1] if self.metrics.val_map50_95 else 0,
                },
            }
            
            # 保存
            torch.save(ckpt, save_path)
            self.log(f"💾 Epoch {epoch}: 已保存 {save_name}")
            
        except Exception as e:
            self.log(f"⚠️ Epoch {epoch}: 保存失败 - {e}")
    
    def request_stop(self):
        """请求停止训练"""
        self._stop_requested = True
    
    def get_metrics(self) -> TrainingMetrics:
        """获取训练指标"""
        return self.metrics
    
    def predict(self, source, conf: float = 0.25, use_tta: bool = False):
        """
        使用当前模型进行推理
        
        [优化-A4] 支持TTA测试时增强
        
        Args:
            source: 图像路径或图像列表
            conf: 置信度阈值
            use_tta: 是否使用测试时增强
        
        Returns:
            ultralytics Results对象
        """
        if self.yolo is None:
            raise RuntimeError("模型未加载，请先运行train()或加载模型")
        
        if use_tta:
            return self.predict_with_tta(source, conf)
        
        return self.yolo.predict(source=source, conf=conf, save=False)
    
    def predict_with_tta(self, source, conf: float = 0.25, 
                         augments: list = None,
                         nms_iou: float = 0.5):
        """
        [优化-A4] 使用TTA（测试时增强）进行推理
        
        支持的增强方式：
        - flip_horizontal: 水平翻转
        - flip_vertical: 垂直翻转
        - scale_0.8: 缩放到0.8倍
        - scale_1.2: 缩放到1.2倍
        
        Args:
            source: 图像路径或图像
            conf: 置信度阈值
            augments: 要使用的增强列表，默认['flip_horizontal']
            nms_iou: NMS的IoU阈值
        
        Returns:
            合并后的检测结果
        """
        import cv2
        import numpy as np
        from pathlib import Path
        
        if augments is None:
            augments = ['flip_horizontal']
        
        # 加载原始图像
        if isinstance(source, str):
            image = cv2.imread(source)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            image = source.copy()
        else:
            raise ValueError("source必须是图像路径或numpy数组")
        
        original_h, original_w = image.shape[:2]
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # 原始图像预测
        results = self.yolo.predict(source=image, conf=conf, save=False, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
        
        # 应用各种增强并预测
        for aug in augments:
            aug_image, inverse_fn = self._apply_tta_augment(image, aug)
            
            results = self.yolo.predict(source=aug_image, conf=conf, save=False, verbose=False)
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                # 将检测框逆变换回原始坐标
                aug_h, aug_w = aug_image.shape[:2]
                boxes = inverse_fn(boxes, original_w, original_h, aug_w, aug_h)
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
        
        # 合并所有检测结果
        if not all_boxes:
            return results  # 返回空结果
        
        merged_boxes = np.vstack(all_boxes)
        merged_scores = np.concatenate(all_scores)
        merged_classes = np.concatenate(all_classes)
        
        # 应用NMS合并重叠框
        final_boxes, final_scores, final_classes = self._tta_nms(
            merged_boxes, merged_scores, merged_classes, nms_iou
        )
        
        # 构建返回结果（修改原始results对象）
        if results and len(results) > 0:
            # 更新boxes数据
            import torch
            results[0].boxes.data = torch.tensor(
                np.column_stack([final_boxes, final_scores, final_classes]),
                device=results[0].boxes.data.device
            )
        
        self.log(f"   TTA: 原始 {len(merged_boxes)} 个框 -> NMS后 {len(final_boxes)} 个框")
        
        return results
    
    def _apply_tta_augment(self, image, aug_type):
        """
        [优化-A4] 应用TTA增强并返回逆变换函数
        
        Returns:
            (augmented_image, inverse_function)
        """
        import cv2
        import numpy as np
        
        if aug_type == 'flip_horizontal':
            aug_image = cv2.flip(image, 1)  # 水平翻转
            
            def inverse_fn(boxes, orig_w, orig_h, aug_w, aug_h):
                # 水平翻转boxes: x1 = w - x2, x2 = w - x1
                new_boxes = boxes.copy()
                new_boxes[:, 0] = orig_w - boxes[:, 2]  # new x1
                new_boxes[:, 2] = orig_w - boxes[:, 0]  # new x2
                return new_boxes
            
        elif aug_type == 'flip_vertical':
            aug_image = cv2.flip(image, 0)  # 垂直翻转
            
            def inverse_fn(boxes, orig_w, orig_h, aug_w, aug_h):
                new_boxes = boxes.copy()
                new_boxes[:, 1] = orig_h - boxes[:, 3]  # new y1
                new_boxes[:, 3] = orig_h - boxes[:, 1]  # new y2
                return new_boxes
            
        elif aug_type.startswith('scale_'):
            scale = float(aug_type.split('_')[1])
            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            aug_image = cv2.resize(image, (new_w, new_h))
            
            def inverse_fn(boxes, orig_w, orig_h, aug_w, aug_h):
                # 缩放boxes回原始尺寸
                scale_x = orig_w / aug_w
                scale_y = orig_h / aug_h
                new_boxes = boxes.copy()
                new_boxes[:, [0, 2]] *= scale_x
                new_boxes[:, [1, 3]] *= scale_y
                return new_boxes
        else:
            # 未知增强类型，返回原图
            aug_image = image
            def inverse_fn(boxes, orig_w, orig_h, aug_w, aug_h):
                return boxes
        
        return aug_image, inverse_fn
    
    def _tta_nms(self, boxes, scores, classes, iou_threshold=0.5):
        """
        [优化-A4] TTA结果的NMS合并
        
        对每个类别分别进行NMS
        """
        import numpy as np
        
        final_boxes = []
        final_scores = []
        final_classes = []
        
        unique_classes = np.unique(classes)
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            if len(cls_boxes) == 0:
                continue
            
            # 执行NMS
            keep = self._nms(cls_boxes, cls_scores, iou_threshold)
            
            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_classes.append(np.full(len(keep), cls))
        
        if not final_boxes:
            return np.array([]), np.array([]), np.array([])
        
        return (
            np.vstack(final_boxes),
            np.concatenate(final_scores),
            np.concatenate(final_classes)
        )
    
    def _nms(self, boxes, scores, iou_threshold):
        """
        [优化-A4] 非极大值抑制
        """
        import numpy as np
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep