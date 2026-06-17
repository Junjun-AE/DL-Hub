"""
YOLO-seg 实例分割训练器
=======================
封装 Ultralytics YOLO 的实例分割训练 (task='segment')，提供与图像分割模块
统一的回调接口。改编自检测模块的 YOLOTrainer，差异点：

- task 固定为 'segment'
- loss_items 为 (box, seg, cls[, dfl])，需解析 seg_loss
- 验证指标同时含 box mAP 与 mask mAP，主指标用 mask mAP50-95
- 支持 YOLOv8-seg / YOLOv11-seg / YOLO26-seg

预训练权重查找: pretrained_model/image_segmentation_yolo/ 优先, 否则联网下载。
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


def get_pretrained_dir() -> Path:
    """YOLO-seg 预训练权重缓存目录。"""
    root = Path(__file__).parent.parent.parent  # Deep_learning_tools/
    d = root / 'pretrained_model' / 'image_segmentation_yolo'
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_pretrained_path(ultralytics_name: str) -> Optional[Path]:
    """在本地缓存目录查找权重文件 (如 yolo26m-seg.pt)。"""
    p = get_pretrained_dir() / ultralytics_name
    return p if p.exists() else None


@dataclass
class TrainingCallback:
    """训练回调。"""
    on_epoch_end: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    should_stop: Optional[Callable[[], bool]] = None
    on_train_end: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class TrainingMetrics:
    """训练指标记录。"""
    box_losses: List[float] = field(default_factory=list)
    seg_losses: List[float] = field(default_factory=list)
    cls_losses: List[float] = field(default_factory=list)
    dfl_losses: List[float] = field(default_factory=list)
    # box 指标
    box_map50: List[float] = field(default_factory=list)
    box_map50_95: List[float] = field(default_factory=list)
    # mask 指标 (主指标)
    mask_map50: List[float] = field(default_factory=list)
    mask_map50_95: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_mask_map50: float = 0.0
    best_mask_map50_95: float = 0.0
    best_epoch: int = 0


class YOLOSegTrainer:
    """YOLO 实例分割训练器。"""

    def __init__(
        self,
        model_name: str,             # 预训练权重文件名, 如 'yolo26m-seg.pt'
        data_yaml: str,
        output_dir: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        learning_rate: float = 0.01,
        optimizer: str = 'SGD',
        patience: int = 50,
        freeze_backbone: bool = False,
        device: str = '0',
        workers: int = 8,
        weight_decay: float = 0.0005,
        momentum: float = 0.937,
        warmup_epochs: float = 3.0,
        cos_lr: bool = True,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        fliplr: float = 0.5,
        save_period: int = 0,
        callback: TrainingCallback = None,
        class_names: List[str] = None,
    ):
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = TrainingMetrics()
        self._stop_requested = False
        self.yolo = None
        self.current_epoch = 0

    def log(self, message: str):
        if self.callback.on_log:
            self.callback.on_log(message)

    def request_stop(self):
        self._stop_requested = True

    def get_metrics(self) -> TrainingMetrics:
        return self.metrics

    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, Any]:
        from ultralytics import YOLO

        self.log(f"🚀 开始实例分割训练: {self.model_name}")
        self.log(f"📊 数据集: {self.data_yaml}")
        self.log(f"📁 输出目录: {self.output_dir}")

        start_time = time.time()

        # 优先本地缓存
        model_path = self.model_name
        local = get_pretrained_path(self.model_name)
        if local and local.exists():
            model_path = str(local)
            self.log(f"🔧 加载本地模型: {model_path}")
        else:
            self.log(f"🔧 加载模型: {self.model_name} (将从网络下载)")
            self.log(f"   💡 提示: 可将 {self.model_name} 放入 {get_pretrained_dir()} 避免下载")

        self.yolo = YOLO(model_path)

        train_args = {
            'task': 'segment',          # 关键: 明确指定实例分割任务
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
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'warmup_epochs': self.warmup_epochs,
            'cos_lr': self.cos_lr,
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            'fliplr': self.fliplr,
            'save': True,
            'save_period': -1,
            'plots': True,
            'val': True,
            'verbose': True,
        }
        if self.freeze_backbone:
            train_args['freeze'] = 10

        self.log("⚙️ 训练参数:")
        self.log(f"   epochs={self.epochs}, batch={self.batch_size}, imgsz={self.img_size}")
        self.log(f"   lr={self.learning_rate}, optimizer={self.optimizer}, patience={self.patience}")

        self._register_callbacks()

        try:
            results = self.yolo.train(**train_args)
            total_time = time.time() - start_time

            self.log("━" * 50)
            self.log("✅ 训练完成!")
            self.log(f"🏆 最佳 mask mAP@50: {self.metrics.best_mask_map50:.4f}")
            self.log(f"🏆 最佳 mask mAP@50:95: {self.metrics.best_mask_map50_95:.4f}")
            self.log(f"🏆 最佳 Epoch: {self.metrics.best_epoch}")
            self.log(f"⏱️ 总用时: {total_time/60:.1f} 分钟")
            self.log("━" * 50)

            self._convert_checkpoints()

            final_result = {
                'best_mask_map50': self.metrics.best_mask_map50,
                'best_mask_map50_95': self.metrics.best_mask_map50_95,
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

    # ------------------------------------------------------------------ #
    def _register_callbacks(self):
        def on_train_epoch_end(trainer):
            try:
                self.current_epoch = trainer.epoch + 1
                if self.callback.should_stop and self.callback.should_stop():
                    trainer.stop = True
                    self.log("⏹️ 收到停止信号，正在保存模型...")
            except Exception:
                pass

        def on_fit_epoch_end(trainer):
            try:
                epoch = trainer.epoch + 1
                self.current_epoch = epoch

                # ---- 解析 loss_items ----
                # segment: (box, seg, cls[, dfl])
                #   YOLOv8/11-seg: box, seg, cls, dfl  (4)
                #   YOLO26-seg:    box, seg, cls        (3, 无 DFL)
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    try:
                        li = trainer.loss_items
                        losses = li.detach().cpu().numpy() if hasattr(li, 'detach') else li
                        n = len(losses)
                        if n >= 4:
                            self.metrics.box_losses.append(float(losses[0]))
                            self.metrics.seg_losses.append(float(losses[1]))
                            self.metrics.cls_losses.append(float(losses[2]))
                            self.metrics.dfl_losses.append(float(losses[3]))
                        elif n == 3:
                            self.metrics.box_losses.append(float(losses[0]))
                            self.metrics.seg_losses.append(float(losses[1]))
                            self.metrics.cls_losses.append(float(losses[2]))
                            self.metrics.dfl_losses.append(0.0)
                        elif n == 2:
                            self.metrics.box_losses.append(float(losses[0]))
                            self.metrics.seg_losses.append(0.0)
                            self.metrics.cls_losses.append(float(losses[1]))
                            self.metrics.dfl_losses.append(0.0)
                    except Exception:
                        pass

                # ---- 解析验证指标 (box + mask) ----
                box_map50 = box_map = mask_map50 = mask_map = 0.0
                metrics_obj = None
                if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                    metrics_obj = trainer.metrics
                elif hasattr(trainer, 'validator') and trainer.validator is not None:
                    metrics_obj = getattr(trainer.validator, 'metrics', None)

                if metrics_obj is not None:
                    # box
                    if hasattr(metrics_obj, 'box') and metrics_obj.box is not None:
                        b = metrics_obj.box
                        box_map50 = float(getattr(b, 'map50', 0.0) or 0.0)
                        box_map = float(getattr(b, 'map', 0.0) or 0.0)
                        if hasattr(b, 'mp'):
                            self.metrics.precision.append(float(b.mp))
                        if hasattr(b, 'mr'):
                            self.metrics.recall.append(float(b.mr))
                    # mask (seg)
                    seg_metric = getattr(metrics_obj, 'seg', None)
                    if seg_metric is not None:
                        mask_map50 = float(getattr(seg_metric, 'map50', 0.0) or 0.0)
                        mask_map = float(getattr(seg_metric, 'map', 0.0) or 0.0)

                # fitness 兜底 (主要用于排名, 不区分 box/mask)
                if mask_map == 0.0 and box_map == 0.0 and hasattr(trainer, 'fitness'):
                    try:
                        mask_map = float(trainer.fitness or 0.0)
                    except Exception:
                        pass

                self.metrics.box_map50.append(box_map50)
                self.metrics.box_map50_95.append(box_map)
                self.metrics.mask_map50.append(mask_map50)
                self.metrics.mask_map50_95.append(mask_map)

                # 以 mask mAP50-95 作为最佳判定 (没有则退回 box)
                rank_metric = mask_map if mask_map > 0 else box_map
                best_rank = self.metrics.best_mask_map50_95
                if rank_metric > best_rank:
                    self.metrics.best_mask_map50 = mask_map50 if mask_map50 > 0 else box_map50
                    self.metrics.best_mask_map50_95 = rank_metric
                    self.metrics.best_epoch = epoch

                # 学习率
                if hasattr(trainer, 'optimizer'):
                    try:
                        self.metrics.learning_rates.append(
                            float(trainer.optimizer.param_groups[0]['lr'])
                        )
                    except Exception:
                        pass

                box_loss = self.metrics.box_losses[-1] if self.metrics.box_losses else 0
                seg_loss = self.metrics.seg_losses[-1] if self.metrics.seg_losses else 0
                cls_loss = self.metrics.cls_losses[-1] if self.metrics.cls_losses else 0

                self.log(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"box: {box_loss:.4f} | seg: {seg_loss:.4f} | cls: {cls_loss:.4f} | "
                    f"mask mAP50: {mask_map50:.4f} | mask mAP50-95: {mask_map:.4f}"
                )

                if self.callback.on_epoch_end:
                    self.callback.on_epoch_end(epoch, {
                        'epoch': epoch,
                        'box_loss': box_loss,
                        'seg_loss': seg_loss,
                        'cls_loss': cls_loss,
                        'dfl_loss': self.metrics.dfl_losses[-1] if self.metrics.dfl_losses else 0,
                        'box_map50': box_map50,
                        'box_map50_95': box_map,
                        'mask_map50': mask_map50,
                        'mask_map50_95': mask_map,
                        'precision': self.metrics.precision[-1] if self.metrics.precision else 0,
                        'recall': self.metrics.recall[-1] if self.metrics.recall else 0,
                        'best_mask_map50': self.metrics.best_mask_map50,
                        'best_mask_map50_95': self.metrics.best_mask_map50_95,
                        'best_epoch': self.metrics.best_epoch,
                        'lr': self.metrics.learning_rates[-1] if self.metrics.learning_rates else self.learning_rate,
                    })

                if self.save_period > 0 and epoch % self.save_period == 0:
                    self._save_periodic_checkpoint(epoch, trainer)

            except Exception:
                pass

        self.yolo.add_callback('on_train_epoch_end', on_train_epoch_end)
        self.yolo.add_callback('on_fit_epoch_end', on_fit_epoch_end)

    # ------------------------------------------------------------------ #
    def _find_weights(self) -> Optional[Path]:
        """定位 ultralytics 训练产物 weights 目录。"""
        candidates = [
            self.output_dir / 'weights',
            self.output_dir / self.output_dir.name / 'weights',
        ]
        for c in candidates:
            if c.exists():
                return c
        # 兜底: 递归找 best.pt 所在目录
        for best in self.output_dir.rglob('best.pt'):
            return best.parent
        return None

    def _save_metadata_into(self, pt_path: Path):
        """把类别名/任务等元数据写入 checkpoint 的 custom_metadata。"""
        try:
            import torch
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                ckpt['custom_metadata'] = {
                    'task': 'segment',
                    'model_name': self.model_name.replace('.pt', ''),
                    'class_names': self.class_names,
                    'num_classes': len(self.class_names),
                    'img_size': self.img_size,
                    'best_mask_mAP50': self.metrics.best_mask_map50,
                    'best_mask_mAP50_95': self.metrics.best_mask_map50_95,
                }
                torch.save(ckpt, pt_path)
        except Exception as e:
            self.log(f"   ⚠️ 写入元数据失败(非致命): {e}")

    def _save_metadata_json(self):
        """在输出目录写 model_metadata.json (供状态监控读取 mask mAP)。"""
        import json
        meta = {
            'task': 'segment',
            'model_name': self.model_name.replace('.pt', ''),
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'img_size': self.img_size,
            'best_mask_mAP50': self.metrics.best_mask_map50,
            'best_mask_mAP50_95': self.metrics.best_mask_map50_95,
            'best_box_mAP50': self.metrics.box_map50[-1] if self.metrics.box_map50 else 0.0,
            'best_epoch': self.metrics.best_epoch,
        }
        try:
            with open(self.output_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"   ⚠️ 写入 metadata.json 失败(非致命): {e}")

    def _convert_checkpoints(self):
        """训练完成后整理 checkpoint, 写入元数据。"""
        weights_dir = self._find_weights()
        if weights_dir is None:
            self.log("   ⚠️ 未找到 weights 目录, 跳过 checkpoint 整理")
            self._save_metadata_json()
            return
        for fname in ('best.pt', 'last.pt'):
            p = weights_dir / fname
            if p.exists():
                self._save_metadata_into(p)
        self._save_metadata_json()
        self.log(f"   ✅ 模型权重: {weights_dir}")

    def _save_periodic_checkpoint(self, epoch: int, trainer):
        try:
            weights_dir = self._find_weights()
            if weights_dir is None:
                return
            best = weights_dir / 'best.pt'
            if best.exists():
                import shutil
                dst = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
                shutil.copy(best, dst)
                self.log(f"   💾 已保存周期 checkpoint: {dst.name}")
        except Exception:
            pass
