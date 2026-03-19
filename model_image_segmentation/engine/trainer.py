"""
SegFormer训练引擎 - 基于MMSegmentation
工业级语义分割训练，无模拟，无回退
【修复版】- 参考目标检测代码，训练中途即时保存带元数据的模型
【修复2】- 添加meta字段兼容MMSegmentation的init_model
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np

# 在模块加载时检查MMSegmentation
try:
    import mmseg
    from mmseg.apis import init_model
    from mmseg.models import build_segmentor
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import HOOKS
    from mmengine.hooks import Hook
    MMSEG_AVAILABLE = True
    MMSEG_VERSION = mmseg.__version__
except ImportError as e:
    MMSEG_AVAILABLE = False
    MMSEG_VERSION = None
    MMSEG_IMPORT_ERROR = str(e)


def check_mmseg_installation():
    """检查MMSegmentation安装状态"""
    if not MMSEG_AVAILABLE:
        raise ImportError(
            f"❌ MMSegmentation 未正确安装!\n"
            f"   错误: {MMSEG_IMPORT_ERROR}\n"
            f"\n"
            f"   请按以下步骤安装:\n"
            f"   1. pip install -U openmim\n"
            f"   2. mim install mmengine\n"
            f"   3. mim install 'mmcv>=2.0.0'\n"
            f"   4. mim install mmsegmentation\n"
            f"\n"
            f"   或者使用pip:\n"
            f"   pip install mmengine mmcv mmsegmentation"
        )
    return True


@dataclass
class TrainingCallback:
    """训练回调函数集合"""
    on_epoch_start: Optional[Callable[[int, int], None]] = None
    on_epoch_end: Optional[Callable[[int, Dict], None]] = None
    on_train_end: Optional[Callable[[Dict], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    should_stop: Optional[Callable[[], bool]] = None


@dataclass
class TrainingMetrics:
    """训练指标记录"""
    train_losses: List[float] = field(default_factory=list)
    val_mIoU: List[float] = field(default_factory=list)
    val_mDice: List[float] = field(default_factory=list)
    val_mAcc: List[float] = field(default_factory=list)
    per_class_IoU: Dict[int, float] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    best_mIoU: float = 0.0
    best_epoch: int = 0


class SegFormerTrainer:
    """
    SegFormer训练器 - 基于MMSegmentation
    
    【修复版】参考目标检测代码，训练中途即时保存带元数据的模型
    【修复2】添加meta字段兼容MMSegmentation的init_model
    """
    
    def __init__(
        self,
        model_scale: str,
        dataset_dir: str,
        output_dir: str,
        num_classes: int,
        # 基础参数
        epochs: int = 100,
        batch_size: int = 8,
        img_size: int = 512,
        learning_rate: float = 6e-5,
        optimizer: str = 'AdamW',
        patience: int = 20,
        device: str = '0',
        workers: int = 4,
        # 高级参数
        weight_decay: float = 0.01,
        warmup_iters: int = 500,
        use_poly_lr: bool = True,
        poly_power: float = 0.9,
        # 损失函数
        loss_ce_weight: float = 0.4,
        loss_dice_weight: float = 0.6,
        use_class_weight: bool = True,
        # 数据增强
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        random_rotate: bool = True,
        rotate_degree: float = 30.0,
        color_jitter: bool = True,
        scale_range: tuple = (0.5, 2.0),
        # 混合精度
        use_fp16: bool = True,
        # 保存设置
        save_period: int = 10,
        # 回调
        callback: TrainingCallback = None,
        class_names: List[str] = None,
    ):
        """初始化训练器"""
        check_mmseg_installation()
        
        self.model_scale = model_scale
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.patience = patience
        self.device = device
        self.workers = workers
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.use_poly_lr = use_poly_lr
        self.poly_power = poly_power
        self.loss_ce_weight = loss_ce_weight
        self.loss_dice_weight = loss_dice_weight
        self.use_class_weight = use_class_weight
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.random_rotate = random_rotate
        self.rotate_degree = rotate_degree
        self.color_jitter = color_jitter
        self.scale_range = scale_range
        self.use_fp16 = use_fp16
        self.save_period = save_period
        self.callback = callback or TrainingCallback()
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.output_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)
        self.work_dir = self.output_dir / 'work_dirs'
        self.work_dir.mkdir(exist_ok=True)
        
        # 指标记录
        self.metrics = TrainingMetrics()
        
        # 当前状态
        self.current_epoch = 0
        self.iters_per_epoch = 1
        self.start_time = None
        
        self.log(f"✅ MMSegmentation {MMSEG_VERSION} 已加载")
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        
        log_file = self.output_dir / 'training.log'
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(full_message + '\n')
        except Exception:
            pass
        
        if self.callback.on_log:
            self.callback.on_log(full_message)
    
    def _get_model_config(self):
        """获取模型配置"""
        from config.model_registry import get_model_config
        config = get_model_config(self.model_scale)
        if config is None:
            raise ValueError(f"❌ 未知的模型规模: {self.model_scale}")
        return config
    
    def _validate_dataset(self):
        """验证数据集"""
        train_img_dir = self.dataset_dir / 'images' / 'train'
        train_mask_dir = self.dataset_dir / 'masks' / 'train'
        val_img_dir = self.dataset_dir / 'images' / 'val'
        val_mask_dir = self.dataset_dir / 'masks' / 'val'
        
        errors = []
        
        if not train_img_dir.exists():
            errors.append(f"训练集图像目录不存在: {train_img_dir}")
        if not train_mask_dir.exists():
            errors.append(f"训练集mask目录不存在: {train_mask_dir}")
        if not val_img_dir.exists():
            errors.append(f"验证集图像目录不存在: {val_img_dir}")
        if not val_mask_dir.exists():
            errors.append(f"验证集mask目录不存在: {val_mask_dir}")
        
        if errors:
            raise FileNotFoundError(
                "❌ 数据集目录结构错误:\n" + "\n".join(f"   - {e}" for e in errors)
            )
        
        train_images = len(list(train_img_dir.glob('*.*')))
        val_images = len(list(val_img_dir.glob('*.*')))
        
        if train_images == 0:
            raise ValueError(f"❌ 训练集图像为空: {train_img_dir}")
        if val_images == 0:
            raise ValueError(f"❌ 验证集图像为空: {val_img_dir}")
        
        return train_images, val_images
    
    def _detect_image_suffix(self) -> str:
        """检测图像文件后缀"""
        train_img_dir = self.dataset_dir / 'images' / 'train'
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']:
            if list(train_img_dir.glob(f'*{ext}')):
                return ext.lower()
        
        raise ValueError(f"❌ 未找到支持的图像文件: {train_img_dir}")
    
    def _generate_palette(self, num_classes: int) -> list:
        """生成类别颜色调色板"""
        predefined_colors = [
            [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0],
            [0, 128, 0], [0, 0, 128], [128, 128, 0], [128, 0, 128],
            [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0],
        ]
        
        palette = []
        for i in range(num_classes):
            if i < len(predefined_colors):
                palette.append(predefined_colors[i])
            else:
                np.random.seed(i)
                palette.append([np.random.randint(0, 255) for _ in range(3)])
        
        return palette
    
    def _save_periodic_checkpoint(self, epoch: int, runner):
        """
        周期性保存模型 checkpoint
        
        【参考目标检测】保存格式完全兼容部署
        【修复】添加meta字段兼容MMSegmentation的init_model
        文件名格式: {model_name}_epoch{epoch}.pth
        保存到: weights 目录
        """
        try:
            from config.model_registry import IMAGENET_MEAN, IMAGENET_STD
            
            # 获取当前模型
            model = runner.model if hasattr(runner, 'model') else None
            if model is None:
                self.log(f"⚠️ Epoch {epoch}: 无法获取模型，跳过保存")
                return
            
            # 生成文件名
            model_config = self._get_model_config()
            model_base_name = model_config['name']
            save_name = f"{model_base_name}_epoch{epoch}.pth"
            save_path = self.weights_dir / save_name
            
            # 获取类别信息
            nc = self.num_classes
            names_dict = {i: name for i, name in enumerate(self.class_names)}
            palette = self._generate_palette(nc)
            
            # 获取当前mIoU
            current_mIoU = self.metrics.val_mIoU[-1] if self.metrics.val_mIoU else 0
            
            # 获取state_dict
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            else:
                state_dict = {}
            
            # ============================================
            # 【关键修复】构建 checkpoint - 添加meta字段
            # MMSegmentation的init_model需要meta.dataset_meta
            # ============================================
            ckpt = {
                # 框架识别字段
                'framework': 'mmsegmentation',
                'model_type': 'segformer',
                'model_name': model_base_name,
                
                # 模型权重
                'state_dict': state_dict,
                
                # ============================================
                # 【关键】MMSegmentation需要的meta字段
                # ============================================
                'meta': {
                    'dataset_meta': {
                        'classes': tuple(self.class_names),
                        'palette': palette,
                    },
                    'CLASSES': tuple(self.class_names),
                    'PALETTE': palette,
                    'epoch': epoch,
                    'iter': epoch * self.iters_per_epoch,
                },
                
                # 类别信息 (关键字段)
                'nc': nc,
                'num_classes': nc,
                'names': names_dict,
                'class_names': self.class_names,
                
                # 训练状态
                'epoch': epoch,
                'save_type': 'periodic',
                
                # 当前指标
                'mIoU': current_mIoU,
                'best_mIoU': self.metrics.best_mIoU,
                'best_epoch': self.metrics.best_epoch,
                
                # 预处理信息 (关键！用于部署)
                'model_metadata': {
                    'model_name': model_base_name,
                    'num_classes': nc,
                    'class_names': self.class_names,
                    'input_size': self.img_size,
                    'input_spec': {
                        'shape': (1, 3, self.img_size, self.img_size),
                        'color_format': 'RGB',
                        'pixel_range': (0, 255),
                        'normalize_method': 'imagenet',
                        'normalize_mean': list(IMAGENET_MEAN),
                        'normalize_std': list(IMAGENET_STD),
                        'value_range': (-2.5, 2.5),
                    },
                    'ignore_index': 255,
                    'task': 'semantic_segmentation',
                },
                
                # 训练参数
                'train_args': {
                    'epochs': self.epochs,
                    'batch': self.batch_size,
                    'imgsz': self.img_size,
                    'lr0': self.learning_rate,
                    'optimizer': self.optimizer_name,
                },
            }
            
            # 保存
            torch.save(ckpt, save_path)
            self.log(f"💾 Epoch {epoch}: 已保存 {save_name} (mIoU: {current_mIoU:.4f})")
            
        except Exception as e:
            self.log(f"⚠️ Epoch {epoch}: 保存失败 - {e}")
    
    def _save_best_checkpoint(self, epoch: int, runner):
        """
        保存最佳模型 checkpoint
        【修复】添加meta字段兼容MMSegmentation的init_model
        """
        try:
            from config.model_registry import IMAGENET_MEAN, IMAGENET_STD
            
            model = runner.model if hasattr(runner, 'model') else None
            if model is None:
                return
            
            model_config = self._get_model_config()
            model_base_name = model_config['name']
            save_path = self.weights_dir / 'best_model.pth'
            
            nc = self.num_classes
            names_dict = {i: name for i, name in enumerate(self.class_names)}
            palette = self._generate_palette(nc)
            
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            else:
                state_dict = {}
            
            # ============================================
            # 【关键修复】构建 checkpoint - 添加meta字段
            # ============================================
            ckpt = {
                'framework': 'mmsegmentation',
                'model_type': 'segformer',
                'model_name': model_base_name,
                'state_dict': state_dict,
                
                # ============================================
                # 【关键】MMSegmentation需要的meta字段
                # ============================================
                'meta': {
                    'dataset_meta': {
                        'classes': tuple(self.class_names),
                        'palette': palette,
                    },
                    'CLASSES': tuple(self.class_names),
                    'PALETTE': palette,
                    'epoch': epoch,
                    'iter': epoch * self.iters_per_epoch,
                },
                
                'nc': nc,
                'num_classes': nc,
                'names': names_dict,
                'class_names': self.class_names,
                'epoch': epoch,
                'save_type': 'best',
                'mIoU': self.metrics.best_mIoU,
                'best_mIoU': self.metrics.best_mIoU,
                'best_epoch': self.metrics.best_epoch,
                'model_metadata': {
                    'model_name': model_base_name,
                    'num_classes': nc,
                    'class_names': self.class_names,
                    'input_size': self.img_size,
                    'input_spec': {
                        'shape': (1, 3, self.img_size, self.img_size),
                        'color_format': 'RGB',
                        'pixel_range': (0, 255),
                        'normalize_method': 'imagenet',
                        'normalize_mean': list(IMAGENET_MEAN),
                        'normalize_std': list(IMAGENET_STD),
                        'value_range': (-2.5, 2.5),
                    },
                    'ignore_index': 255,
                    'task': 'semantic_segmentation',
                },
                'train_args': {
                    'epochs': self.epochs,
                    'batch': self.batch_size,
                    'imgsz': self.img_size,
                    'lr0': self.learning_rate,
                    'optimizer': self.optimizer_name,
                },
            }
            
            torch.save(ckpt, save_path)
            self.log(f"🏆 最佳模型已更新: best_model.pth (mIoU: {self.metrics.best_mIoU:.4f})")
            
            # 【关键】同时保存config.py，确保推理时有配置文件
            self._save_config_py()
            
        except Exception as e:
            self.log(f"⚠️ 保存最佳模型失败: {e}")
    
    def _build_config(self) -> Config:
        """构建MMSegmentation配置"""
        from config.model_registry import IMAGENET_MEAN, IMAGENET_STD
        
        model_config = self._get_model_config()
        num_train, num_val = self._validate_dataset()
        img_suffix = self._detect_image_suffix()
        
        self.log(f"📊 数据集: 训练 {num_train} 张, 验证 {num_val} 张")
        self.log(f"📷 图像格式: {img_suffix}")
        
        self.iters_per_epoch = max(1, num_train // self.batch_size)
        max_iters = self.epochs * self.iters_per_epoch
        val_interval = self.iters_per_epoch
        save_interval = self.iters_per_epoch * self.save_period
        
        # ============================================
        # 【关键修复】确保warmup_iters不超过max_iters
        # ============================================
        actual_warmup_iters = min(self.warmup_iters, max_iters // 2)
        if actual_warmup_iters < self.warmup_iters:
            self.log(f"⚠️ warmup_iters从{self.warmup_iters}调整为{actual_warmup_iters}（总迭代数{max_iters}的一半）")
        
        self.log(f"📈 每epoch {self.iters_per_epoch} 次迭代, 共 {max_iters} 次")
        
        # 训练数据管道
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(self.img_size, self.img_size), keep_ratio=False),
        ]
        
        if self.flip_horizontal:
            train_pipeline.append(dict(type='RandomFlip', prob=0.5, direction='horizontal'))
        
        if self.flip_vertical:
            train_pipeline.append(dict(type='RandomFlip', prob=0.5, direction='vertical'))
        
        if self.random_rotate:
            train_pipeline.append(dict(
                type='RandomRotate',
                prob=0.5,
                degree=(-self.rotate_degree, self.rotate_degree),
                pad_val=0,
                seg_pad_val=255,
            ))
        
        if self.color_jitter:
            train_pipeline.append(dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
            ))
        
        train_pipeline.append(dict(type='PackSegInputs'))
        
        # 测试/验证数据管道
        # 【关键】LoadAnnotations 必须在 Resize 之后，这样标注也会被 resize
        val_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(self.img_size, self.img_size), keep_ratio=False),
            dict(type='LoadAnnotations'),  # 在Resize之后！
            dict(type='PackSegInputs'),
        ]
        
        # 纯推理管道（不需要标注）- 用于config.py
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(self.img_size, self.img_size), keep_ratio=False),
            dict(type='PackSegInputs'),
        ]
        
        # 调色板
        palette = self._generate_palette(self.num_classes)
        
        # 损失函数配置
        if self.loss_dice_weight > 0:
            loss_decode = [
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=self.loss_ce_weight,
                ),
                dict(
                    type='DiceLoss',
                    use_sigmoid=False,
                    loss_weight=self.loss_dice_weight,
                ),
            ]
        else:
            loss_decode = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            )
        
        # 学习率调度
        if self.use_poly_lr:
            param_scheduler = [
                dict(
                    type='LinearLR',
                    start_factor=1e-6,
                    by_epoch=False,
                    begin=0,
                    end=actual_warmup_iters,
                ),
                dict(
                    type='PolyLR',
                    eta_min=0.0,
                    power=self.poly_power,
                    begin=actual_warmup_iters,
                    end=max_iters,
                    by_epoch=False,
                ),
            ]
        else:
            param_scheduler = [
                dict(
                    type='LinearLR',
                    start_factor=1e-6,
                    by_epoch=False,
                    begin=0,
                    end=actual_warmup_iters,
                ),
            ]
        
        # 优化器配置
        if self.optimizer_name == 'AdamW':
            optim_wrapper = dict(
                type='OptimWrapper',
                optimizer=dict(
                    type='AdamW',
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=self.weight_decay,
                ),
            )
        elif self.optimizer_name == 'SGD':
            optim_wrapper = dict(
                type='OptimWrapper',
                optimizer=dict(
                    type='SGD',
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                ),
            )
        else:
            optim_wrapper = dict(
                type='OptimWrapper',
                optimizer=dict(
                    type='Adam',
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                ),
            )
        
        # 混合精度
        if self.use_fp16:
            optim_wrapper['type'] = 'AmpOptimWrapper'
            optim_wrapper['loss_scale'] = 'dynamic'
        
        # 获取预训练权重路径
        from models.model_factory import get_pretrained_path, download_pretrained_weights, get_pretrained_dir
        backbone_name = model_config['backbone']
        local_path = get_pretrained_path(backbone_name)
        
        if local_path is None:
            pretrained_dir = get_pretrained_dir()
            save_path = pretrained_dir / f"{backbone_name}.pth"
            if download_pretrained_weights(model_config['checkpoint'], save_path):
                local_path = save_path
        
        init_cfg = None
        if local_path and local_path.exists():
            init_cfg = dict(type='Pretrained', checkpoint=str(local_path))
            self.log(f"📦 使用预训练权重: {local_path.name}")
        
        # 构建配置字典
        cfg_dict = dict(
            # 工作目录
            work_dir=str(self.work_dir),
            
            # 模型配置
            model=dict(
                type='EncoderDecoder',
                data_preprocessor=dict(
                    type='SegDataPreProcessor',
                    mean=list(IMAGENET_MEAN),
                    std=list(IMAGENET_STD),
                    bgr_to_rgb=True,
                    pad_val=0,
                    seg_pad_val=255,
                    size=(self.img_size, self.img_size),
                ),
                backbone=dict(
                    type='MixVisionTransformer',
                    in_channels=3,
                    embed_dims=model_config['embed_dims'][0],
                    num_stages=4,
                    num_layers=model_config['depths'],
                    num_heads=model_config['num_heads'],
                    patch_sizes=[7, 3, 3, 3],
                    sr_ratios=[8, 4, 2, 1],
                    out_indices=(0, 1, 2, 3),
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    drop_path_rate=0.1,
                    init_cfg=init_cfg,
                ),
                decode_head=dict(
                    type='SegformerHead',
                    in_channels=model_config['embed_dims'],
                    in_index=[0, 1, 2, 3],
                    channels=256,
                    dropout_ratio=0.1,
                    num_classes=self.num_classes,
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    align_corners=False,
                    loss_decode=loss_decode,
                ),
                train_cfg=dict(),
                test_cfg=dict(mode='whole'),
            ),
            
            # 数据配置
            train_dataloader=dict(
                batch_size=self.batch_size,
                num_workers=self.workers,
                persistent_workers=True if self.workers > 0 else False,
                sampler=dict(type='InfiniteSampler', shuffle=True),
                dataset=dict(
                    type='BaseSegDataset',
                    data_root=str(self.dataset_dir),
                    data_prefix=dict(
                        img_path='images/train',
                        seg_map_path='masks/train',
                    ),
                    img_suffix=img_suffix,
                    seg_map_suffix='.png',
                    metainfo=dict(
                        classes=self.class_names,
                        palette=palette,
                    ),
                    pipeline=train_pipeline,
                    reduce_zero_label=False,
                ),
            ),
            
            val_dataloader=dict(
                batch_size=1,
                num_workers=self.workers,
                persistent_workers=True if self.workers > 0 else False,
                sampler=dict(type='DefaultSampler', shuffle=False),
                dataset=dict(
                    type='BaseSegDataset',
                    data_root=str(self.dataset_dir),
                    data_prefix=dict(
                        img_path='images/val',
                        seg_map_path='masks/val',
                    ),
                    img_suffix=img_suffix,
                    seg_map_suffix='.png',
                    metainfo=dict(
                        classes=self.class_names,
                        palette=palette,
                    ),
                    pipeline=val_pipeline,  # 使用val_pipeline（包含LoadAnnotations）
                    reduce_zero_label=False,
                ),
            ),
            
            test_dataloader=dict(
                batch_size=1,
                num_workers=self.workers,
                persistent_workers=True if self.workers > 0 else False,
                sampler=dict(type='DefaultSampler', shuffle=False),
                dataset=dict(
                    type='BaseSegDataset',
                    data_root=str(self.dataset_dir),
                    data_prefix=dict(
                        img_path='images/val',
                        seg_map_path='masks/val',
                    ),
                    img_suffix=img_suffix,
                    seg_map_suffix='.png',
                    metainfo=dict(
                        classes=self.class_names,
                        palette=palette,
                    ),
                    pipeline=val_pipeline,  # 也使用val_pipeline
                    reduce_zero_label=False,
                ),
            ),
            
            # 验证评估器
            val_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice']),
            test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice']),
            
            # 优化器
            optim_wrapper=optim_wrapper,
            
            # 学习率调度
            param_scheduler=param_scheduler,
            
            # 训练配置
            train_cfg=dict(
                type='IterBasedTrainLoop',
                max_iters=max_iters,
                val_interval=val_interval,
            ),
            val_cfg=dict(type='ValLoop'),
            test_cfg=dict(type='TestLoop'),
            
            # 日志配置
            default_hooks=dict(
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook',
                    by_epoch=False,
                    interval=save_interval,
                    max_keep_ckpts=3,
                    save_best='mIoU',
                    rule='greater',
                ),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                visualization=dict(type='SegVisualizationHook'),
            ),
            
            # 环境配置
            default_scope='mmseg',
            env_cfg=dict(
                cudnn_benchmark=True,
                mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                dist_cfg=dict(backend='nccl'),
            ),
            
            vis_backends=[dict(type='LocalVisBackend')],
            visualizer=dict(
                type='SegLocalVisualizer',
                vis_backends=[dict(type='LocalVisBackend')],
                name='visualizer',
            ),
            
            log_processor=dict(by_epoch=False),
            log_level='INFO',
            load_from=None,
            resume=False,
            
            # 随机种子
            randomness=dict(seed=42),
        )
        
        cfg = Config(cfg_dict)
        self.start_time = time.time()
        
        return cfg
    
    def _create_monitor_hook(self):
        """创建监控Hook"""
        trainer = self
        
        class SegFormerMonitorHook(Hook):
            """训练监控Hook"""
            
            def before_train_iter(self, runner, batch_idx, data_batch=None):
                """训练迭代前"""
                current_iter = runner.iter + 1
                if current_iter % trainer.iters_per_epoch == 1:
                    trainer.current_epoch = (current_iter - 1) // trainer.iters_per_epoch + 1
                    if trainer.callback.on_epoch_start:
                        trainer.callback.on_epoch_start(trainer.current_epoch, trainer.epochs)
                
                if trainer.callback.should_stop and trainer.callback.should_stop():
                    raise KeyboardInterrupt("用户停止训练")
            
            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                """训练迭代后"""
                if outputs is not None and 'loss' in outputs:
                    loss = outputs['loss']
                    if hasattr(loss, 'item'):
                        loss = loss.item()
                    if not hasattr(trainer, '_epoch_losses'):
                        trainer._epoch_losses = []
                    trainer._epoch_losses.append(float(loss))
            
            def after_val_epoch(self, runner, metrics=None):
                """验证epoch后"""
                if metrics is None:
                    return
                
                mIoU = metrics.get('mIoU', 0)
                mDice = metrics.get('mDice', 0)
                mAcc = metrics.get('aAcc', 0)
                
                epoch_avg_loss = 0.0
                if hasattr(trainer, '_epoch_losses') and trainer._epoch_losses:
                    epoch_avg_loss = sum(trainer._epoch_losses) / len(trainer._epoch_losses)
                    trainer.metrics.train_losses.append(epoch_avg_loss)
                    trainer._epoch_losses = []
                
                trainer.metrics.val_mIoU.append(float(mIoU))
                trainer.metrics.val_mDice.append(float(mDice))
                trainer.metrics.val_mAcc.append(float(mAcc))
                
                # 检查是否是最佳
                is_best = mIoU > trainer.metrics.best_mIoU
                if is_best:
                    trainer.metrics.best_mIoU = float(mIoU)
                    trainer.metrics.best_epoch = trainer.current_epoch
                
                trainer.log(
                    f"Epoch {trainer.current_epoch}/{trainer.epochs} | "
                    f"Loss: {epoch_avg_loss:.4f} | "
                    f"mIoU: {mIoU:.4f} | mDice: {mDice:.4f} | "
                    f"Best: {trainer.metrics.best_mIoU:.4f} @ E{trainer.metrics.best_epoch}"
                )
                
                # ============================================
                # 【关键修复】周期性保存模型 - 参考目标检测
                # ============================================
                if trainer.save_period > 0 and trainer.current_epoch % trainer.save_period == 0:
                    trainer._save_periodic_checkpoint(trainer.current_epoch, runner)
                
                # 保存最佳模型
                if is_best:
                    trainer._save_best_checkpoint(trainer.current_epoch, runner)
                
                # 回调
                if trainer.callback.on_epoch_end:
                    trainer.callback.on_epoch_end(trainer.current_epoch, {
                        'train_loss': epoch_avg_loss,  # 修复：使用train_loss而不是loss
                        'loss': epoch_avg_loss,  # 保留兼容
                        'mIoU': mIoU,
                        'mDice': mDice,
                        'mAcc': mAcc,
                        'best_mIoU': trainer.metrics.best_mIoU,
                        'best_epoch': trainer.metrics.best_epoch,
                    })
        
        return SegFormerMonitorHook()
    
    def train(self) -> Dict[str, Any]:
        """开始训练"""
        self.log("=" * 60)
        self.log("🎨 SegFormer 语义分割训练")
        self.log("=" * 60)
        
        if self.device != 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        
        cfg = self._build_config()
        
        runner = Runner.from_cfg(cfg)
        runner._work_dir = str(self.work_dir)
        
        monitor_hook = self._create_monitor_hook()
        runner.register_hook(monitor_hook, priority='LOW')
        
        self.log("🏋️ 开始训练...")
        try:
            runner.train()
        except KeyboardInterrupt:
            self.log("⏹️ 训练被用户中断")
        except Exception as e:
            self.log(f"⚠️ 训练过程中出现异常: {e}")
        finally:
            # 【关键】无论训练是否正常完成，都保存config.py
            try:
                self._save_config_py()
            except Exception as e:
                self.log(f"⚠️ 保存config.py失败: {e}")
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.log("=" * 60)
        self.log("✅ 训练完成!")
        self.log(f"   🏆 最佳 mIoU: {self.metrics.best_mIoU:.4f}")
        self.log(f"   🏆 最佳 Epoch: {self.metrics.best_epoch}")
        self.log(f"   ⏱️ 总用时: {total_time/60:.1f} 分钟")
        self.log("=" * 60)
        
        # 保存last模型和元数据
        self._save_final_outputs()
        
        result = {
            'best_mIoU': self.metrics.best_mIoU,
            'best_epoch': self.metrics.best_epoch,
            'total_time': total_time,
            'output_dir': str(self.output_dir),
        }
        
        if self.callback.on_train_end:
            self.callback.on_train_end(result)
        
        return result
    
    def _save_final_outputs(self):
        """保存最终输出"""
        from config.model_registry import IMAGENET_MEAN, IMAGENET_STD
        
        # 保存元数据JSON
        metadata = {
            'framework': 'mmsegmentation',
            'model_type': 'segformer',
            'model_name': self._get_model_config()['name'],
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'input_spec': {
                'shape': [1, 3, self.img_size, self.img_size],
                'color_format': 'RGB',
                'normalize_method': 'imagenet',
                'normalize_mean': list(IMAGENET_MEAN),
                'normalize_std': list(IMAGENET_STD),
            },
            'metrics': {
                'best_mIoU': self.metrics.best_mIoU,
                'best_epoch': self.metrics.best_epoch,
            },
            'train_args': {
                'epochs': self.epochs,
                'batch': self.batch_size,
                'lr0': self.learning_rate,
            },
        }
        
        metadata_path = self.output_dir / 'model_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.log(f"📝 元数据已保存: {metadata_path}")
        
        # ============================================
        # 【关键】保存config.py供推理使用
        # ============================================
        self._save_config_py()
        
        # 列出已保存的模型
        saved_models = list(self.weights_dir.glob('*.pth'))
        if saved_models:
            self.log(f"📁 weights目录共 {len(saved_models)} 个模型:")
            for m in sorted(saved_models):
                self.log(f"   - {m.name}")
    
    def _save_config_py(self):
        """保存config.py配置文件供推理使用"""
        from config.model_registry import IMAGENET_MEAN, IMAGENET_STD
        
        model_config = self._get_model_config()
        palette = self._generate_palette(self.num_classes)
        
        config_content = f'''# 自动生成的模型配置文件
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# 数据集配置
dataset_type = 'BaseSegDataset'
# 注意: data_root 为训练时的路径，部署时请根据实际环境修改
# 训练时路径: {str(self.dataset_dir)}
data_root = ''  # 推理时按需设置，不硬编码绝对路径

# 类别配置
num_classes = {self.num_classes}
class_names = {self.class_names}
palette = {palette}
metainfo = dict(classes=class_names, palette=palette)

# 图像配置
img_size = {self.img_size}
crop_size = ({self.img_size}, {self.img_size})

# 归一化配置
img_norm_cfg = dict(
    mean={list(IMAGENET_MEAN)},
    std={list(IMAGENET_STD)},
    to_rgb=True,
)

# 数据处理管道
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=({self.img_size}, {self.img_size}), keep_ratio=False),
    dict(type='PackSegInputs'),
]

# 模型配置
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean={list(IMAGENET_MEAN)},
        std={list(IMAGENET_STD)},
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=({self.img_size}, {self.img_size}),
    ),
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims={model_config['embed_dims'][0]},
        num_stages=4,
        num_layers={model_config['depths']},
        num_heads={model_config['num_heads']},
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels={model_config['embed_dims']},
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes={self.num_classes},
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
'''
        
        # 保存到输出目录根目录
        config_path = self.output_dir / 'config.py'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        self.log(f"📝 配置文件已保存: {config_path}")
        
        # 同时保存到weights目录
        config_path_weights = self.weights_dir / 'config.py'
        with open(config_path_weights, 'w', encoding='utf-8') as f:
            f.write(config_content)
        self.log(f"📝 配置文件已保存: {config_path_weights}")
    
    def request_stop(self):
        """请求停止训练"""
        self.log("⏹️ 收到停止请求...")
    
    def get_metrics(self) -> TrainingMetrics:
        """获取训练指标"""
        return self.metrics