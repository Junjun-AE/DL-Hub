"""
训练引擎 - 核心训练逻辑
支持回调函数更新UI、早停、断点续训

[优化-A1] 添加学习率预热调度器支持
[优化-A3] 添加CutMix/MixUp数据增强
[Bug-3修复] 改进训练中断处理
"""

import os
import time
import yaml
import random
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 兼容不同PyTorch版本的AMP导入
try:
    from torch.amp import GradScaler, autocast
    AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_DEVICE = None


# ==================== [优化-A3] CutMix/MixUp 数据增强 ====================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp数据增强
    
    Args:
        x: 输入图像 [B, C, H, W]
        y: 标签 [B]
        alpha: Beta分布参数
        
    Returns:
        mixed_x: 混合后的图像
        y_a: 原始标签
        y_b: 混合标签
        lam: 混合比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix数据增强
    
    Args:
        x: 输入图像 [B, C, H, W]
        y: 标签 [B]
        alpha: Beta分布参数
        
    Returns:
        mixed_x: CutMix后的图像
        y_a: 原始标签
        y_b: 混合标签
        lam: 混合比例（基于面积）
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    
    # 计算裁剪区域
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 边界框
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 调整lam为实际面积比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """MixUp/CutMix的损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== [优化-A1] 学习率预热调度器 ====================

def create_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建带预热的学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine', 'step', 'plateau', 'onecycle')
        epochs: 总训练轮数
        warmup_epochs: 预热轮数
        min_lr: 最小学习率
        
    Returns:
        学习率调度器
    """
    # 创建主调度器
    if scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'onecycle':
        main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=epochs - warmup_epochs,
            steps_per_epoch=kwargs.get('steps_per_epoch', 100)
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
        )
    
    # 如果不需要预热，直接返回主调度器
    if warmup_epochs <= 0:
        return main_scheduler
    
    # 创建预热调度器
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # 组合调度器
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


@dataclass
class TrainingCallback:
    """训练回调函数集合"""
    on_epoch_start: Optional[Callable[[int, int], None]] = None  # (epoch, total_epochs)
    on_epoch_end: Optional[Callable[[int, Dict], None]] = None  # (epoch, metrics)
    on_batch_end: Optional[Callable[[int, int, Dict], None]] = None  # (batch, total_batches, metrics)
    on_train_end: Optional[Callable[[Dict], None]] = None  # (final_metrics)
    on_log: Optional[Callable[[str], None]] = None  # (message)
    should_stop: Optional[Callable[[], bool]] = None  # 检查是否需要停止


@dataclass
class TrainingMetrics:
    """训练指标记录"""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_acc: float = 0.0
    best_epoch: int = 0


class Trainer:
    """
    训练器
    
    [优化-A1] 支持学习率预热
    [优化-A3] 支持CutMix/MixUp数据增强
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        output_dir: str,
        model_info: Dict[str, Any],
        epochs: int = 100,
        use_amp: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        save_freq: int = 5,
        label_smoothing: float = 0.1,
        callback: TrainingCallback = None,
        # [优化-A3] CutMix/MixUp参数
        use_mixup: bool = False,
        use_cutmix: bool = False,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mix_prob: float = 0.5,  # 使用混合增强的概率
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            output_dir: 输出目录
            model_info: 模型信息字典
            epochs: 训练轮数
            use_amp: 是否使用混合精度
            early_stopping: 是否启用早停
            patience: 早停耐心值
            save_freq: 保存频率
            label_smoothing: 标签平滑
            callback: 回调函数
            use_mixup: 是否使用MixUp [优化-A3]
            use_cutmix: 是否使用CutMix [优化-A3]
            mixup_alpha: MixUp的alpha参数 [优化-A3]
            cutmix_alpha: CutMix的alpha参数 [优化-A3]
            mix_prob: 使用混合增强的概率 [优化-A3]
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_info = model_info
        self.epochs = epochs
        self.use_amp = use_amp and device.type == 'cuda'
        self.early_stopping = early_stopping
        self.patience = patience
        self.save_freq = save_freq
        self.callback = callback or TrainingCallback()
        
        # [优化-A3] CutMix/MixUp参数
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 混合精度
        if self.use_amp:
            if AMP_DEVICE:
                self.scaler = GradScaler(device=AMP_DEVICE)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 指标记录
        self.metrics = TrainingMetrics()
        
        # 早停计数器
        self.no_improve_count = 0
        
        # 停止标志
        self._stop_requested = False
        
        # 日志增强信息
        if self.use_mixup or self.use_cutmix:
            augment_info = []
            if self.use_mixup:
                augment_info.append(f"MixUp(α={self.mixup_alpha})")
            if self.use_cutmix:
                augment_info.append(f"CutMix(α={self.cutmix_alpha})")
            print(f"[Trainer] 启用数据增强: {', '.join(augment_info)}, prob={self.mix_prob}")
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        
        # 写入文件
        log_file = self.output_dir / 'logs' / 'training.log'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')
        
        # 回调
        if self.callback.on_log:
            self.callback.on_log(full_message)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        [优化-A3] 支持CutMix/MixUp数据增强
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # 检查是否需要停止
            if self._should_stop():
                break
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # [优化-A3] 应用CutMix/MixUp
            use_mix = (self.use_mixup or self.use_cutmix) and random.random() < self.mix_prob
            
            if use_mix:
                # 随机选择使用CutMix还是MixUp
                if self.use_cutmix and self.use_mixup:
                    use_cutmix = random.random() < 0.5
                elif self.use_cutmix:
                    use_cutmix = True
                else:
                    use_cutmix = False
                
                if use_cutmix:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                else:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # 兼容不同PyTorch版本
                if AMP_DEVICE:
                    with autocast(device_type=AMP_DEVICE):
                        outputs = self.model(images)
                        # [优化-A3] 使用混合损失
                        if use_mix:
                            loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                        else:
                            loss = self.criterion(outputs, labels)
                else:
                    with autocast():
                        outputs = self.model(images)
                        if use_mix:
                            loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                        else:
                            loss = self.criterion(outputs, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mix:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            # [优化-A3] 使用混合时的准确率计算（使用原始labels或labels_a）
            if use_mix:
                # 对于混合样本，使用加权准确率
                correct_a = predicted.eq(labels_a).float()
                correct_b = predicted.eq(labels_b).float()
                total_correct += (lam * correct_a + (1 - lam) * correct_b).sum().item()
            else:
                total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)
            
            # 批次回调
            if self.callback.on_batch_end and (batch_idx + 1) % 10 == 0:
                batch_metrics = {
                    'loss': loss.item(),
                    'acc': 100.0 * total_correct / total_samples,
                }
                self.callback.on_batch_end(batch_idx + 1, num_batches, batch_metrics)
        
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'acc': 100.0 * total_correct / total_samples if total_samples > 0 else 0,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                # 兼容不同PyTorch版本
                if AMP_DEVICE:
                    with autocast(device_type=AMP_DEVICE):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'acc': 100.0 * total_correct / total_samples,
        }
    
    def _should_stop(self) -> bool:
        """检查是否需要停止"""
        if self._stop_requested:
            return True
        if self.callback.should_stop and self.callback.should_stop():
            self._stop_requested = True
            return True
        return False
    
    def request_stop(self):
        """请求停止训练"""
        self._stop_requested = True
    
    def train(self, start_epoch: int = 0) -> Dict[str, Any]:
        """
        开始训练
        
        [Bug-3修复]: 改进中断处理逻辑，确保正确保存当前epoch状态
        
        Args:
            start_epoch: 起始epoch（用于续训）
        
        Returns:
            训练结果字典
        """
        self.log(f"🚀 开始训练: {self.epochs} epochs")
        self.log(f"📁 输出目录: {self.output_dir}")
        
        start_time = time.time()
        current_epoch = start_epoch  # [Bug-3修复] 跟踪当前epoch
        interrupted = False  # [Bug-3修复] 跟踪是否被中断
        
        try:
            for epoch in range(start_epoch, self.epochs):
                current_epoch = epoch  # [Bug-3修复] 更新当前epoch
                
                # 检查是否需要停止（在epoch开始前检查）
                if self._should_stop():
                    self.log("⏹️ 训练被中断（epoch开始前），正在保存当前状态...")
                    interrupted = True
                    break
                
                epoch_start = time.time()
                
                # 回调
                if self.callback.on_epoch_start:
                    self.callback.on_epoch_start(epoch + 1, self.epochs)
                
                # 训练一个epoch
                train_metrics = self.train_epoch(epoch)
                
                # [Bug-3修复] 检查train_epoch是否被中断
                if self._should_stop():
                    self.log("⏹️ 训练被中断（epoch训练中），正在保存当前状态...")
                    interrupted = True
                    break
                
                # 验证
                val_metrics = self.validate()
                
                # 更新学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler:
                    self.scheduler.step()
                
                # 记录指标
                self.metrics.train_losses.append(train_metrics['loss'])
                self.metrics.val_losses.append(val_metrics['loss'])
                self.metrics.val_accs.append(val_metrics['acc'])
                self.metrics.learning_rates.append(current_lr)
                
                epoch_time = time.time() - epoch_start
                
                # 日志
                self.log(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['acc']:.2f}% | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                # 检查是否是最佳模型
                is_best = val_metrics['acc'] > self.metrics.best_acc
                if is_best:
                    self.metrics.best_acc = val_metrics['acc']
                    self.metrics.best_epoch = epoch + 1
                    self.no_improve_count = 0
                    self._save_checkpoint(epoch, is_best=True)
                    self.log(f"🎉 新的最佳模型! Acc: {val_metrics['acc']:.2f}%")
                else:
                    self.no_improve_count += 1
                
                # 定期保存
                if (epoch + 1) % self.save_freq == 0:
                    self._save_checkpoint(epoch)
                
                # epoch回调
                if self.callback.on_epoch_end:
                    epoch_metrics = {
                        'epoch': epoch + 1,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['acc'],
                        'best_acc': self.metrics.best_acc,
                        'best_epoch': self.metrics.best_epoch,
                        'lr': current_lr,
                    }
                    self.callback.on_epoch_end(epoch + 1, epoch_metrics)
                
                # 早停检查
                if self.early_stopping and self.no_improve_count >= self.patience:
                    self.log(f"⏹️ 早停触发: {self.patience} epochs 无改善")
                    break
            
            # [Bug-3修复] 根据中断状态保存检查点
            if interrupted:
                try:
                    self._save_checkpoint(current_epoch, is_interrupted=True)
                    self.log(f"✅ 中断检查点已保存 (epoch {current_epoch + 1})")
                except Exception as e:
                    self.log(f"⚠️ 保存中断检查点失败: {e}")
            else:
                # 保存最终模型
                self._save_checkpoint(current_epoch, is_last=True)
            
            # 保存训练配置
            self._save_config()
            
        except Exception as e:
            # [Bug-3修复] 捕获任何异常，尝试保存检查点
            self.log(f"❌ 训练出错: {e}")
            try:
                self._save_checkpoint(current_epoch, is_interrupted=True)
                self.log(f"✅ 错误恢复检查点已保存 (epoch {current_epoch + 1})")
            except Exception as save_error:
                self.log(f"⚠️ 保存错误恢复检查点失败: {save_error}")
            raise
        
        total_time = time.time() - start_time
        
        final_result = {
            'best_acc': self.metrics.best_acc,
            'best_epoch': self.metrics.best_epoch,
            'final_epoch': current_epoch + 1,
            'total_time': total_time,
            'output_dir': str(self.output_dir),
            'interrupted': interrupted,  # [Bug-3修复] 返回是否被中断
        }
        
        if interrupted:
            self.log(f"⏸️ 训练中断于 Epoch {current_epoch + 1}")
        else:
            self.log(f"✅ 训练完成! 最佳准确率: {self.metrics.best_acc:.2f}% (Epoch {self.metrics.best_epoch})")
        self.log(f"⏱️ 总用时: {total_time/60:.1f} 分钟")
        
        # 结束回调
        if self.callback.on_train_end:
            self.callback.on_train_end(final_result)
        
        return final_result
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False, is_interrupted: bool = False):
        """
        保存检查点
        
        所有模型都保存到 checkpoints 文件夹：
        - best_model.pth: 最佳模型
        - last_model.pth: 最终模型
        - interrupted_model.pth: 中断时保存的模型
        - epoch_N.pth: 定期保存的模型
        """
        from models.model_factory import save_model_checkpoint
        
        checkpoints_dir = self.output_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定保存路径 - 所有模型都放到 checkpoints 文件夹
        if is_best:
            save_path = checkpoints_dir / 'best_model.pth'
        elif is_last:
            save_path = checkpoints_dir / 'last_model.pth'
        elif is_interrupted:
            save_path = checkpoints_dir / 'interrupted_model.pth'
        else:
            save_path = checkpoints_dir / f'epoch_{epoch + 1}.pth'
        
        # 完整训练配置
        training_config = {
            'epochs': self.epochs,
            'batch_size': self.train_loader.batch_size,
            'optimizer': self.optimizer.__class__.__name__,
            'lr': self.optimizer.param_groups[0]['lr'],
            'initial_lr': self.optimizer.defaults.get('lr', self.optimizer.param_groups[0]['lr']),
            'weight_decay': self.optimizer.defaults.get('weight_decay', 0),
            'use_amp': self.use_amp,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'label_smoothing': self.criterion.label_smoothing if hasattr(self.criterion, 'label_smoothing') else 0,
            'data_path': str(self.train_loader.dataset.root_path) if hasattr(self.train_loader.dataset, 'root_path') else '',
        }
        
        # 添加数据集信息到 model_info
        model_info_with_data = self.model_info.copy()
        if hasattr(self.train_loader.dataset, 'class_to_idx'):
            model_info_with_data['class_to_idx'] = self.train_loader.dataset.class_to_idx
        elif hasattr(self.train_loader.dataset, 'dataset') and hasattr(self.train_loader.dataset.dataset, 'class_to_idx'):
            model_info_with_data['class_to_idx'] = self.train_loader.dataset.dataset.class_to_idx
        
        save_model_checkpoint(
            model=self.model,
            model_info=model_info_with_data,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_acc=self.metrics.best_acc,
            save_path=str(save_path),
            is_best=is_best,
            training_config=training_config,
        )
    
    def _save_config(self):
        """保存训练配置到YAML"""
        config = {
            'model': {
                'name': self.model_info.get('model_name'),
                'family': self.model_info.get('model_family'),
                'scale': self.model_info.get('model_scale'),
                'num_classes': self.model_info.get('num_classes'),
            },
            'training': {
                'epochs': self.epochs,
                'batch_size': self.train_loader.batch_size,
                'optimizer': self.optimizer.__class__.__name__,
                'use_amp': self.use_amp,
                'early_stopping': self.early_stopping,
                'patience': self.patience,
            },
            'results': {
                'best_acc': self.metrics.best_acc,
                'best_epoch': self.metrics.best_epoch,
                'final_epoch': len(self.metrics.val_accs),
            },
            'input_spec': {
                'input_size': self.model_info.get('input_size'),
                'mean': self.model_info.get('mean'),
                'std': self.model_info.get('std'),
            },
        }
        
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def get_metrics(self) -> TrainingMetrics:
        """获取训练指标"""
        return self.metrics
