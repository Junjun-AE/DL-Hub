"""
SevSeg-YOLO 训练引擎
封装 ScoreDetectionTrainer 训练流程，提供统一回调接口
"""

import os
import sys
import time
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class TrainingCallback:
    """训练回调函数集合"""
    on_epoch_end: Optional[Callable[[int, Dict], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    should_stop: Optional[Callable[[], bool]] = None


@dataclass
class TrainingMetrics:
    """训练指标记录"""
    box_losses: List[float] = field(default_factory=list)
    cls_losses: List[float] = field(default_factory=list)
    score_losses: List[float] = field(default_factory=list)
    map50_list: List[float] = field(default_factory=list)
    map50_95_list: List[float] = field(default_factory=list)
    score_mae_list: List[float] = field(default_factory=list)
    best_map50: float = 0.0
    best_epoch: int = 0


class SevSegTrainer:
    """
    SevSeg-YOLO 训练器
    封装 ultralytics 的 ScoreDetectionTrainer
    """

    def __init__(
        self,
        model_yaml: str,
        pretrained_weights: str,
        data_yaml: str,
        output_dir: str,
        epochs: int = 105,
        batch_size: int = 32,
        img_size: int = 640,
        learning_rate: float = 0.01,
        optimizer: str = 'SGD',
        patience: int = 50,
        device: str = '0',
        workers: int = 4,
        cos_lr: bool = True,
        mosaic: float = 1.0,
        fliplr: float = 0.5,
        save_period: int = 0,
        callback: TrainingCallback = None,
    ):
        self.model_yaml = model_yaml
        self.pretrained_weights = pretrained_weights
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.patience = patience
        self.device = device
        self.workers = workers
        self.cos_lr = cos_lr
        self.mosaic = mosaic
        self.fliplr = fliplr
        self.save_period = save_period
        self.callback = callback or TrainingCallback()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = TrainingMetrics()
        self._stop_requested = False
        self.yolo = None
        self.current_epoch = 0

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

    def train(self) -> Dict[str, Any]:
        """开始训练"""
        # 确保使用本地修改版 ultralytics
        sevseg_dir = str(Path(__file__).parent.parent)
        if sevseg_dir not in sys.path:
            sys.path.insert(0, sevseg_dir)

        from ultralytics import YOLO

        self.log(f"🚀 开始 SevSeg-YOLO 训练")
        self.log(f"📊 数据集: {self.data_yaml}")
        self.log(f"📁 输出目录: {self.output_dir}")

        start_time = time.time()

        # 加载模型
        from models.model_factory import get_pretrained_path

        # 优先使用本地缓存的基础权重
        base_name = Path(self.pretrained_weights).stem  # yolo26n
        local_path = get_pretrained_path(base_name, self.pretrained_weights)

        if local_path and local_path.exists():
            self.log(f"🔧 使用本地基础权重: {local_path}")
            # 用model_yaml构建结构，用pretrained权重初始化
            self.yolo = YOLO(str(self.model_yaml))
        else:
            self.log(f"🔧 加载模型配置: {self.model_yaml}")
            self.log(f"   基础权重将从网络下载: {self.pretrained_weights}")
            self.yolo = YOLO(str(self.model_yaml))

        # 构建训练参数
        train_args = {
            'task': 'score_detect',
            'data': str(self.data_yaml),
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
            'pretrained': str(local_path) if local_path else self.pretrained_weights,
            # 固定的Score配置
            'mixup': 0.0,  # 必须关闭
            # 数据增强
            'mosaic': self.mosaic,
            'fliplr': self.fliplr,
            'cos_lr': self.cos_lr,
            # 保存
            'save': True,
            'save_period': self.save_period if self.save_period > 0 else -1,
            'plots': True,
            'val': True,
            'verbose': True,
        }

        # 合并额外增强参数 (flipud, hsv_h, hsv_s, hsv_v, translate, scale)
        if hasattr(self, 'extra_aug') and self.extra_aug:
            train_args.update(self.extra_aug)

        self.log(f"⚙️ 训练参数:")
        self.log(f"   task=score_detect, epochs={self.epochs}, batch={self.batch_size}")
        self.log(f"   imgsz={self.img_size}, lr={self.learning_rate}")
        self.log(f"   optimizer={self.optimizer}, patience={self.patience}")
        self.log(f"   mixup=0.0 (强制关闭)")

        # 注册 ultralytics 回调
        self._register_callbacks()

        try:
            results = self.yolo.train(**train_args)

            elapsed = time.time() - start_time
            self.log("━" * 50)
            self.log(f"✅ 训练完成！用时 {elapsed/60:.1f} 分钟")
            self.log(f"🏆 最佳 mAP@50: {self.metrics.best_map50:.4f} (Epoch {self.metrics.best_epoch})")
            self.log("━" * 50)

            # 尝试从results.csv读取完整指标
            self._read_results_csv()

            # 保存模型元数据和转换checkpoint格式（对齐Detection任务）
            self._convert_checkpoints()

            return {
                'success': True,
                'best_map50': self.metrics.best_map50,
                'best_epoch': self.metrics.best_epoch,
                'elapsed_minutes': elapsed / 60,
                'output_dir': str(self.output_dir),
            }

        except Exception as e:
            import traceback
            self.log(f"❌ 训练错误: {str(e)}")
            self.log(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    def _register_callbacks(self):
        """注册ultralytics回调"""

        def on_train_epoch_end(trainer):
            """每个epoch结束时触发"""
            self.current_epoch = trainer.epoch + 1

            # 从trainer获取loss - 健壮处理不同格式
            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                try:
                    tloss = trainer.tloss
                    # 处理不同的tensor类型
                    if hasattr(tloss, 'cpu'):
                        loss_items = tloss.cpu().numpy()
                    elif hasattr(tloss, '__len__'):
                        loss_items = [float(x) for x in tloss]
                    else:
                        loss_items = []

                    # ScoreDetectionTrainer 有4个loss: box, cls, dfl, score
                    if len(loss_items) >= 4:
                        self.metrics.box_losses.append(float(loss_items[0]))
                        self.metrics.cls_losses.append(float(loss_items[1]))
                        self.metrics.score_losses.append(float(loss_items[3]))
                    elif len(loss_items) >= 3:
                        # 回退: 标准检测只有3个loss
                        self.metrics.box_losses.append(float(loss_items[0]))
                        self.metrics.cls_losses.append(float(loss_items[1]))
                except Exception as e:
                    self.log(f"⚠️ 读取loss失败: {e}")

            # 检查是否需要停止
            if self.callback.should_stop and self.callback.should_stop():
                self._stop_requested = True
                trainer.stop = True
                self.log("⏹️ 用户请求停止训练")

        def on_fit_epoch_end(trainer):
            """验证结束后触发 - 获取验证指标"""
            epoch = trainer.epoch + 1

            # trainer.metrics 在训练期间是 dict 格式
            # keys: "metrics/mAP50(B)", "metrics/mAP50-95(B)", "score_mae", "score_spearman" 等
            metrics_dict = {}
            m = trainer.metrics
            if isinstance(m, dict):
                # 从 dict 中提取 mAP (key 格式: "metrics/mAP50(B)")
                for k, v in m.items():
                    if 'mAP50(' in k and 'mAP50-95' not in k:
                        metrics_dict['map50'] = float(v)
                    elif 'mAP50-95(' in k:
                        metrics_dict['map50_95'] = float(v)
                # score 指标 (直接用 key)
                if 'score_mae' in m:
                    metrics_dict['score_mae'] = float(m['score_mae'])
                if 'score_spearman' in m:
                    metrics_dict['score_spearman'] = float(m['score_spearman'])
            elif hasattr(m, 'box'):
                # 回退: 如果是 DetMetrics 对象 (final_eval 等场景)
                metrics_dict['map50'] = float(getattr(m.box, 'map50', 0))
                metrics_dict['map50_95'] = float(getattr(m.box, 'map', 0))

            # 追加到列表
            map50 = metrics_dict.get('map50', 0)
            map50_95 = metrics_dict.get('map50_95', 0)
            score_mae = metrics_dict.get('score_mae', float('nan'))

            self.metrics.map50_list.append(map50)
            self.metrics.map50_95_list.append(map50_95)
            if not np.isnan(score_mae):
                self.metrics.score_mae_list.append(score_mae)

            # 更新最佳
            if map50 > self.metrics.best_map50:
                self.metrics.best_map50 = map50
                self.metrics.best_epoch = epoch

            # 触发回调
            if self.callback.on_epoch_end:
                self.callback.on_epoch_end(epoch, {
                    'box_loss': self.metrics.box_losses[-1] if self.metrics.box_losses else 0,
                    'cls_loss': self.metrics.cls_losses[-1] if self.metrics.cls_losses else 0,
                    'score_loss': self.metrics.score_losses[-1] if self.metrics.score_losses else 0,
                    'map50': map50,
                    'map50_95': map50_95,
                    'score_mae': score_mae,
                    'best_map50': self.metrics.best_map50,
                    'best_epoch': self.metrics.best_epoch,
                })

            # 日志
            score_str = f", MAE={score_mae:.3f}" if not np.isnan(score_mae) else ""
            self.log(f"📊 Epoch {epoch}: mAP@50={map50:.4f}, mAP@50:95={map50_95:.4f}{score_str}")

        # 注册回调到YOLO实例
        self.yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.yolo.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    def _read_results_csv(self):
        """从results.csv读取完整训练指标"""
        csv_path = self.output_dir / 'results.csv'
        if not csv_path.exists():
            return

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return

            # 补充可能遗漏的数据
            for row in rows:
                # 读取score_loss
                score_loss_key = None
                for key in row:
                    if 'score_loss' in key.strip():
                        score_loss_key = key
                        break

            self.log(f"📈 results.csv: {len(rows)} epochs 记录")

        except Exception as e:
            self.log(f"⚠️ 读取results.csv失败: {e}")

    def request_stop(self):
        """请求停止训练"""
        self._stop_requested = True

    def get_metrics(self) -> TrainingMetrics:
        """获取训练指标"""
        return self.metrics

    # ==================== 模型元数据与checkpoint转换 ====================

    def _get_class_names(self) -> list:
        """从data.yaml获取类别名称"""
        try:
            import yaml
            with open(self.data_yaml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            names = data.get('names', {})
            if isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
            elif isinstance(names, list):
                return names
        except Exception as e:
            self.log(f"⚠️ 读取类别名称失败: {e}")
        return []

    def _convert_checkpoints(self):
        """
        转换checkpoint为兼容格式，保存元数据（对齐Detection任务）

        最终目录结构:
        output/sevseg_xxx/
        ├── weights_model_importer/
        │   ├── best_model.pt
        │   └── last_model.pt
        └── model_metadata.json
        """
        import shutil
        import json
        import torch

        weights_dir = self.output_dir / 'weights'
        if not weights_dir.exists():
            self.log("⚠️ weights目录不存在，跳过checkpoint转换")
            return

        class_names = self._get_class_names()
        nc = len(class_names) if class_names else 0

        # 创建 model_importer 兼容目录
        importer_dir = self.output_dir / 'weights_model_importer'
        importer_dir.mkdir(parents=True, exist_ok=True)

        # 处理 best.pt
        best_pt = weights_dir / 'best.pt'
        if best_pt.exists():
            best_output = importer_dir / 'best_model.pt'
            try:
                self._save_with_metadata(best_pt, best_output, class_names, nc, is_best=True)
                self.log(f"💾 已保存: weights_model_importer/best_model.pt")
            except Exception as e:
                self.log(f"⚠️ 保存best_model.pt失败: {e}")

        # 处理 last.pt
        last_pt = weights_dir / 'last.pt'
        if last_pt.exists():
            last_output = importer_dir / 'last_model.pt'
            try:
                self._save_with_metadata(last_pt, last_output, class_names, nc, is_best=False)
                self.log(f"💾 已保存: weights_model_importer/last_model.pt")
            except Exception as e:
                self.log(f"⚠️ 保存last_model.pt失败: {e}")

        # 保存元数据JSON
        self._save_metadata_json(self.output_dir / 'model_metadata.json', class_names, nc)

        # 验证复制成功后才删除原始 weights 文件夹
        best_ok = (importer_dir / 'best_model.pt').exists() if best_pt.exists() else True
        last_ok = (importer_dir / 'last_model.pt').exists() if last_pt.exists() else True

        if best_ok and last_ok:
            try:
                shutil.rmtree(weights_dir)
                self.log(f"🗑️ 已删除: weights/ (只保留 weights_model_importer/)")
            except Exception as e:
                self.log(f"⚠️ 删除weights文件夹失败: {e}")
        else:
            self.log(f"⚠️ 跳过删除 weights/（复制未完全成功，保留原始文件作为备份）")

    def _save_with_metadata(self, src_pt, dst_pt, class_names, nc, is_best=True):
        """保存带元数据的checkpoint（兼容model_importer.py）"""
        import torch

        original_ckpt = torch.load(src_pt, map_location='cpu', weights_only=False)
        names_dict = {i: name for i, name in enumerate(class_names)} if class_names else {}

        # 添加兼容字段
        original_ckpt['framework'] = 'ultralytics'
        original_ckpt['nc'] = nc
        original_ckpt['names'] = names_dict

        # _original_model 字段
        if 'model' in original_ckpt and original_ckpt['model'] is not None:
            model_obj = original_ckpt['model']
            if hasattr(model_obj, 'state_dict'):
                original_ckpt['_original_model'] = model_obj

        # yaml 字段
        model_name_lower = Path(self.pretrained_weights).stem.lower()
        yaml_config = {'nc': nc, 'yaml_file': f"{model_name_lower}.yaml"}
        if 'yaml' in original_ckpt and isinstance(original_ckpt['yaml'], dict):
            yaml_config.update(original_ckpt['yaml'])
            yaml_config['nc'] = nc
        original_ckpt['yaml'] = yaml_config

        # train_args 字段
        if 'train_args' not in original_ckpt or not isinstance(original_ckpt.get('train_args'), dict):
            original_ckpt['train_args'] = {}
        original_ckpt['train_args'].update({
            'nc': nc, 'imgsz': self.img_size, 'model': self.pretrained_weights,
            'epochs': self.epochs, 'batch': self.batch_size, 'lr0': self.learning_rate,
            'optimizer': self.optimizer, 'task': 'score_detect',
        })

        # custom_metadata 字段 - 预处理信息（部署必需）
        original_ckpt['custom_metadata'] = {
            'model_name': model_name_lower,
            'task_type': 'sevseg',
            'num_classes': nc,
            'class_names': class_names,
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
                'epochs': self.epochs, 'batch': self.batch_size,
                'imgsz': self.img_size, 'lr0': self.learning_rate,
                'optimizer': self.optimizer,
            },
            'best_epoch': self.metrics.best_epoch if is_best else self.current_epoch,
            'best_map50': self.metrics.best_map50 if is_best else 0,
        }

        torch.save(original_ckpt, dst_pt)
        self.log(f"   ✅ 已添加字段: framework=ultralytics, nc={nc}, names={len(names_dict)}类")

    def _save_metadata_json(self, json_path, class_names, nc):
        """保存元数据JSON文件（方便查看和供status_monitor读取）"""
        import json

        metadata = {
            'model_name': Path(self.pretrained_weights).stem,
            'task_type': 'sevseg',
            'num_classes': nc,
            'class_names': class_names,
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
                'epochs': self.epochs, 'batch': self.batch_size,
                'imgsz': self.img_size, 'lr0': self.learning_rate,
                'optimizer': self.optimizer,
            },
            'best_mAP50': self.metrics.best_map50,
            'best_epoch': self.metrics.best_epoch,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.log(f"📝 元数据已保存: {json_path}")
