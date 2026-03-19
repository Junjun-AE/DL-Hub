# -*- coding: utf-8 -*-
"""
状态监控服务
===========
监控任务状态和训练指标
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import os


def is_process_running(pid: int) -> bool:
    """
    检查进程是否在运行
    
    Args:
        pid: 进程ID
        
    Returns:
        是否在运行
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_task_status(task_dir: Path, running_task_id: str = None, current_task_id: str = None) -> str:
    """
    根据任务目录内容判断任务状态
    
    改进：增加对正在运行任务的检测
    
    Args:
        task_dir: 任务目录路径
        running_task_id: 当前正在运行的任务ID（从ProcessService获取）
        current_task_id: 此任务的ID
        
    Returns:
        状态字符串: idle/training/completed/interrupted/error
    """
    task_dir = Path(task_dir)
    dlhub_dir = task_dir / ".dlhub"
    output_dir = task_dir / "output"
    
    # 0. 检查是否是当前正在运行的任务
    if running_task_id and current_task_id and running_task_id == current_task_id:
        return "training"
    
    # 1. 检查是否有运行中的进程
    pid_file = dlhub_dir / "running.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if is_process_running(pid):
                return "training"
            else:
                # 进程已结束，清理PID文件
                try:
                    pid_file.unlink()
                except Exception:
                    pass
        except (ValueError, IOError):
            pass
    
    # 2. 检查task.json中的状态
    task_json = dlhub_dir / "task.json"
    if task_json.exists():
        try:
            with open(task_json, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            saved_status = task_data.get('status', 'idle')
            # 如果明确标记为训练中，返回训练中
            if saved_status == 'training':
                return "training"
            # 如果明确标记为完成或错误，返回对应状态
            if saved_status in ['completed', 'error']:
                return saved_status
        except Exception:
            pass
    
    # 3. 检查输出目录
    if not output_dir.exists():
        return "idle"
    
    # 4. 查找训练输出
    train_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not train_dirs:
        return "idle"
    
    # 获取最新的训练目录
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    
    # 5. 检查是否有best_model
    best_model_patterns = [
        "best_model.pth",
        "best_model.pt", 
        "best.pt",
        "weights/best.pt",
        "weights/best_model.pt",
        "weights/best_model.pth",
        "checkpoints/best_model.pth",
        "checkpoints/best.pth",
        "weights_model_importer/best_model.pt",
        "weights_model_importer/best_model.pth",
    ]
    
    has_best_model = False
    for pattern in best_model_patterns:
        if (latest_dir / pattern).exists():
            has_best_model = True
            break
    
    # 6. 检查是否有错误日志
    error_indicators = ["error.log", "crash.log"]
    for indicator in error_indicators:
        if (latest_dir / indicator).exists():
            return "error"
    
    # 7. 检查是否有last_model（训练完成的标志）
    last_model_patterns = [
        "last_model.pth",
        "last_model.pt",
        "checkpoints/last_model.pth",
        "weights/last.pt",
        "weights_model_importer/last_model.pt",
        "weights_model_importer/last_model.pth",
    ]
    
    has_last_model = False
    for pattern in last_model_patterns:
        if (latest_dir / pattern).exists():
            has_last_model = True
            break
    
    # 如果有last_model，说明训练已完成
    if has_last_model:
        return "completed"
    
    # 如果只有best_model但没有last_model，可能还在训练中或中断了
    if has_best_model:
        # 检查目录的修改时间，如果最近5分钟内有更新，可能还在训练
        import time
        try:
            mtime = latest_dir.stat().st_mtime
            if time.time() - mtime < 300:  # 5分钟内有更新
                return "training"
        except Exception:
            pass
        return "interrupted"
    
    # 8. 有输出目录但没有best_model，可能是中断了
    checkpoint_patterns = ["*.pth", "*.pt", "weights/*.pt", "checkpoints/*.pth", "weights_model_importer/*.pt"]
    has_checkpoint = False
    for pattern in checkpoint_patterns:
        if list(latest_dir.glob(pattern)):
            has_checkpoint = True
            break
    
    if has_checkpoint:
        return "interrupted"
    
    return "idle"


def get_task_metrics(task_dir: Path, task_type: str) -> Optional[Dict[str, Any]]:
    """
    获取任务的最佳指标
    
    Args:
        task_dir: 任务目录路径
        task_type: 任务类型
        
    Returns:
        指标字典 {name: str, value: float} 或 None
    """
    task_dir = Path(task_dir)
    output_dir = task_dir / "output"
    
    if not output_dir.exists():
        return None
    
    # 获取最新的训练目录
    train_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not train_dirs:
        return None
    
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    
    # 根据任务类型解析指标
    metrics = None
    
    if task_type == 'classification':
        metrics = _get_classification_metrics(latest_dir)
    elif task_type == 'detection':
        metrics = _get_detection_metrics(latest_dir)
    elif task_type == 'segmentation':
        metrics = _get_segmentation_metrics(latest_dir)
    elif task_type == 'anomaly':
        metrics = _get_anomaly_metrics(latest_dir)
    elif task_type == 'ocr':
        metrics = _get_ocr_metrics(latest_dir)
    elif task_type == 'sevseg':
        metrics = _get_detection_metrics(latest_dir)  # SevSeg使用与Detection相同的mAP指标
    
    return metrics


def _get_classification_metrics(train_dir: Path) -> Optional[Dict[str, Any]]:
    """获取分类任务指标
    
    BUG修复: trainer.py保存的best_acc已经是百分比形式(如97.68)，不需要再乘100
    """
    # 尝试从checkpoint读取
    best_model = train_dir / "checkpoints" / "best_model.pth"
    if not best_model.exists():
        best_model = train_dir / "best_model.pth"
    
    if best_model.exists():
        try:
            import torch
            ckpt = torch.load(best_model, map_location='cpu', weights_only=False)
            if 'training_state' in ckpt:
                best_acc = ckpt['training_state'].get('best_acc')
                if best_acc is not None:
                    # 修复：best_acc已经是百分比，不需要再乘100
                    # 如果值小于1，说明是小数形式需要乘100；否则已经是百分比
                    if best_acc <= 1.0:
                        return {"name": "Accuracy", "value": round(best_acc * 100, 2)}
                    else:
                        return {"name": "Accuracy", "value": round(best_acc, 2)}
        except Exception as e:
            print(f"读取分类指标失败: {e}")
    
    return None


def _get_detection_metrics(train_dir: Path) -> Optional[Dict[str, Any]]:
    """获取检测任务指标
    
    修复：统一处理百分比格式
    """
    # 尝试从model_metadata.json读取
    metadata_file = train_dir / "model_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 检查各种可能的指标字段
            for key in ['best_mAP50', 'mAP50', 'metrics/mAP50(B)']:
                if key in metadata:
                    value = metadata[key]
                    # 如果值小于等于1，说明是小数形式需要乘100
                    if value <= 1.0:
                        return {"name": "mAP@50", "value": round(value * 100, 2)}
                    else:
                        return {"name": "mAP@50", "value": round(value, 2)}
        except Exception as e:
            print(f"读取检测指标失败: {e}")
    
    return None


def _get_segmentation_metrics(train_dir: Path) -> Optional[Dict[str, Any]]:
    """获取分割任务指标
    
    修复：统一处理百分比格式
    """
    # 尝试从model_metadata.json读取
    metadata_file = train_dir / "model_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if 'best_mIoU' in metadata:
                value = metadata['best_mIoU']
                # 如果值小于等于1，说明是小数形式需要乘100
                if value <= 1.0:
                    return {"name": "mIoU", "value": round(value * 100, 2)}
                else:
                    return {"name": "mIoU", "value": round(value, 2)}
        except Exception as e:
            print(f"读取分割指标失败: {e}")
    
    return None


def _get_anomaly_metrics(train_dir: Path) -> Optional[Dict[str, Any]]:
    """获取异常检测任务指标
    
    修复：统一处理百分比格式
    """
    # 查找.pkg文件中的config.json
    pkg_files = list(train_dir.glob("*.pkg"))
    if not pkg_files:
        exports_dir = train_dir.parent / "exports"
        if exports_dir.exists():
            pkg_files = list(exports_dir.glob("*.pkg"))
    
    for pkg_file in pkg_files:
        try:
            import zipfile
            with zipfile.ZipFile(pkg_file, 'r') as zf:
                if 'config.json' in zf.namelist():
                    config = json.loads(zf.read('config.json'))
                    threshold_info = config.get('threshold', {})
                    if 'auroc' in threshold_info:
                        value = threshold_info['auroc']
                        # 如果值小于等于1，说明是小数形式需要乘100
                        if value <= 1.0:
                            return {"name": "AUROC", "value": round(value * 100, 2)}
                        else:
                            return {"name": "AUROC", "value": round(value, 2)}
        except Exception as e:
            print(f"读取异常检测指标失败: {e}")
    
    return None


def _get_ocr_metrics(train_dir: Path) -> Optional[Dict[str, Any]]:
    """获取OCR任务指标"""
    # OCR通常是预训练模型，没有训练指标
    return None
