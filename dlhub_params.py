# -*- coding: utf-8 -*-
"""
DL-Hub 参数管理器 (增强版 v2.2 - 基于文件的单例模式)
=====================================================
用于在Gradio App中加载和保存UI参数、训练历史、日志

功能：
- 保存/加载配置参数 (params)
- 保存/加载训练历史数据 (history) - 用于恢复曲线图
- 保存/加载训练日志 (logs)
- 基于文件路径的单例模式：确保所有模块共享同一个实例

使用方法：
    from dlhub_params import get_dlhub_params
    
    # 获取单例实例（推荐方式）
    params = get_dlhub_params()
    
    # 获取参数
    batch_size = params.get('training.batch_size', 32)
    
    # 获取历史数据（用于恢复曲线）
    history = params.get_history()
    train_losses = history.get('train_losses', [])
    
    # 保存训练历史
    params.save_history({
        'train_losses': [0.5, 0.3, 0.2],
        'val_accs': [80, 85, 90],
        'best_acc': 90,
        'best_epoch': 3
    })
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading


# ==================== 基于文件路径的单例实例管理 ====================
# 使用字典存储不同task_dir对应的实例
_instances: Dict[str, 'DLHubParams'] = {}
_instance_lock = threading.Lock()
_initialized_paths: set = set()  # 跟踪已初始化的路径


def get_dlhub_params(task_dir: str = None) -> Optional['DLHubParams']:
    """
    获取DLHubParams单例实例（推荐使用此函数）
    
    基于任务目录实现单例，确保同一个任务目录只有一个实例。
    
    Args:
        task_dir: 任务目录，默认从环境变量获取
        
    Returns:
        DLHubParams实例，如果初始化失败返回None
    """
    global _instances, _initialized_paths
    
    # 确定实际的任务目录
    actual_task_dir = task_dir or os.environ.get('DLHUB_TASK_DIR', '.')
    actual_task_dir = str(Path(actual_task_dir).resolve())  # 标准化路径
    
    with _instance_lock:
        # 检查是否已有该路径的实例
        if actual_task_dir in _instances:
            return _instances[actual_task_dir]
        
        # 创建新实例
        try:
            instance = DLHubParams(actual_task_dir)
            instance.load()
            _instances[actual_task_dir] = instance
            
            # 只在首次初始化时打印日志
            if actual_task_dir not in _initialized_paths:
                _initialized_paths.add(actual_task_dir)
                print(f"[DL-Hub] ✓ 参数管理器已初始化 (单例)")
                print(f"[DL-Hub]   任务目录: {actual_task_dir}")
            
            return instance
        except Exception as e:
            print(f"[DL-Hub] 参数管理器初始化失败: {e}")
            return None


def reset_dlhub_params(task_dir: str = None):
    """重置单例实例（主要用于测试）"""
    global _instances, _initialized_paths
    with _instance_lock:
        if task_dir:
            actual_task_dir = str(Path(task_dir).resolve())
            _instances.pop(actual_task_dir, None)
            _initialized_paths.discard(actual_task_dir)
        else:
            _instances.clear()
            _initialized_paths.clear()


class DLHubParams:
    """
    DL-Hub参数管理器 (增强版)
    
    支持保存：
    - params: UI配置参数
    - history: 训练历史数据（曲线、最佳结果）
    - logs: 训练/转换/推理日志
    """
    
    # 历史数据最大长度限制
    MAX_HISTORY_LENGTH = 500
    MAX_LOG_LINES = 200
    
    def __init__(self, task_dir: str = None):
        """
        初始化参数管理器
        
        Args:
            task_dir: 任务目录，默认从环境变量 DLHUB_TASK_DIR 获取
        """
        self.task_dir = Path(task_dir or os.environ.get('DLHUB_TASK_DIR', '.'))
        self.task_id = os.environ.get('DLHUB_TASK_ID', 'unknown')
        self.params_file = self.task_dir / '.dlhub' / 'ui_params.json'
        self._params: Dict[str, Any] = {}
        self._history: Dict[str, Any] = {}
        self._logs: Dict[str, List[str]] = {}
        self._loaded = False
        self._task_type = self._read_task_type()
        self._lock = threading.Lock()
    
    def _read_task_type(self) -> str:
        """从task.json读取任务类型"""
        task_json = self.task_dir / '.dlhub' / 'task.json'
        if task_json.exists():
            try:
                with open(task_json, 'r', encoding='utf-8') as f:
                    return json.load(f).get('task_type', 'unknown')
            except Exception:
                pass
        return 'unknown'
    
    @property
    def is_dlhub_mode(self) -> bool:
        """是否在DL-Hub模式下运行"""
        return bool(os.environ.get('DLHUB_TASK_DIR'))
    
    @property
    def task_type(self) -> str:
        """获取任务类型"""
        return self._task_type
    
    def load(self) -> Dict[str, Any]:
        """
        从文件加载所有数据（参数、历史、日志）
        
        Returns:
            参数字典
        """
        with self._lock:
            if self.params_file.exists():
                try:
                    with open(self.params_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._params = data.get('params', {})
                    self._history = data.get('history', {})
                    self._logs = data.get('logs', {})
                    self._loaded = True
                    print(f"[DL-Hub] ✓ 已加载数据: {self.params_file}")
                    
                    # 打印加载的历史数据信息
                    if self._history:
                        epochs = len(self._history.get('train_losses', []))
                        best = self._history.get('best_acc') or self._history.get('best_map50')
                        if epochs > 0:
                            print(f"[DL-Hub]   ├─ 训练历史: {epochs} epochs")
                        if best:
                            print(f"[DL-Hub]   └─ 最佳结果: {best}")
                except Exception as e:
                    print(f"[DL-Hub] ✗ 加载数据失败: {e}")
                    self._params = {}
                    self._history = {}
                    self._logs = {}
            else:
                print(f"[DL-Hub] 参数文件不存在，将使用默认值")
                self._params = {}
                self._history = {}
                self._logs = {}
            
            return self._params
    
    def save(self, params: Dict[str, Any] = None) -> bool:
        """
        保存参数到文件（同时保存历史和日志）
        
        Args:
            params: 要保存的参数字典，None则保存当前内存中的参数
            
        Returns:
            是否保存成功
        """
        with self._lock:
            if params is not None:
                self._params = params
            
            return self._save_all()
    
    def _save_all(self) -> bool:
        """内部方法：保存所有数据"""
        # 确保目录存在
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 限制历史数据长度
        history_to_save = self._limit_history(self._history.copy())
        
        # 限制日志长度
        logs_to_save = {}
        for key, lines in self._logs.items():
            logs_to_save[key] = lines[-self.MAX_LOG_LINES:] if len(lines) > self.MAX_LOG_LINES else lines
        
        now = datetime.now().isoformat()
        
        data = {
            'version': '2.0',
            'task_type': self._task_type,
            'task_id': self.task_id,
            'saved_at': now,
            'params': self._params,
            'history': history_to_save,
            'logs': logs_to_save
        }
        
        try:
            # 原子写入：先写临时文件再重命名
            temp_file = self.params_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_file.replace(self.params_file)
            
            # 【关键修复】同时更新 task.json 的 updated_at
            # 这样前端显示的更新时间才会正确更新
            self._update_task_json_timestamp(now)
            
            return True
        except Exception as e:
            print(f"[DL-Hub] ✗ 保存数据失败: {e}")
            return False
    
    def _update_task_json_timestamp(self, timestamp: str) -> None:
        """更新 task.json 中的 updated_at 时间戳"""
        task_json = self.task_dir / '.dlhub' / 'task.json'
        if not task_json.exists():
            return
        
        try:
            with open(task_json, 'r', encoding='utf-8') as f:
                task_meta = json.load(f)
            
            task_meta['updated_at'] = timestamp
            
            # 原子写入
            temp_file = task_json.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(task_meta, f, ensure_ascii=False, indent=2)
            temp_file.replace(task_json)
        except Exception as e:
            # 更新失败不影响主流程，只打印警告
            print(f"[DL-Hub] ⚠ 更新task.json时间戳失败: {e}")
    
    def _limit_history(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """限制历史数据数组长度"""
        array_keys = ['train_losses', 'val_losses', 'val_accs', 
                      'box_losses', 'cls_losses', 'map50_list', 'map50_95_list',
                      'learning_rates', 'score_losses', 'score_mae_list']
        
        for key in array_keys:
            if key in history and isinstance(history[key], list):
                if len(history[key]) > self.MAX_HISTORY_LENGTH:
                    history[key] = history[key][-self.MAX_HISTORY_LENGTH:]
        
        return history
    
    # ==================== 参数相关方法 ====================
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取参数值
        
        支持点号分隔的嵌套键，如 'training.batch_size'
        """
        if not self._loaded:
            self.load()
        
        keys = key.split('.')
        value = self._params
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置参数值，支持点号分隔的嵌套键"""
        # 确保先加载已有数据
        if not self._loaded:
            self.load()
        
        keys = key.split('.')
        target = self._params
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取参数的某个部分（如 'data', 'model', 'training'）"""
        if not self._loaded:
            self.load()
        return self._params.get(section, {})
    
    def set_section(self, section: str, values: Dict[str, Any]) -> None:
        """设置参数的某个部分"""
        self._params[section] = values
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有参数"""
        if not self._loaded:
            self.load()
        return self._params.copy()
    
    # ==================== 历史数据相关方法 ====================
    
    def get_history(self) -> Dict[str, Any]:
        """
        获取训练历史数据
        
        Returns:
            历史数据字典，包含:
            - train_losses: 训练损失列表
            - val_losses: 验证损失列表
            - val_accs: 验证准确率列表 (分类任务)
            - map50_list: mAP@50列表 (检测任务)
            - best_acc/best_map50: 最佳指标
            - best_epoch: 最佳epoch
            - current_epoch: 当前epoch
            - total_epochs: 总epochs
            - output_dir: 输出目录
            - completed: 是否完成
        """
        if not self._loaded:
            self.load()
        return self._history.copy()
    
    def save_history(self, history: Dict[str, Any], auto_save: bool = True) -> bool:
        """
        保存训练历史数据
        
        Args:
            history: 历史数据字典
            auto_save: 是否自动保存到文件
            
        Returns:
            是否保存成功
        """
        with self._lock:
            self._history.update(history)
            if auto_save:
                return self._save_all()
            return True
    
    def update_history_epoch(self, epoch_data: Dict[str, Any], auto_save: bool = True) -> bool:
        """
        更新单个epoch的历史数据（追加到列表）
        
        Args:
            epoch_data: 包含本epoch数据的字典，例如:
                {
                    'train_loss': 0.5,
                    'val_loss': 0.6,
                    'val_acc': 85.0,
                    'current_epoch': 5
                }
            auto_save: 是否自动保存
            
        Returns:
            是否保存成功
        """
        with self._lock:
            # 追加到对应的列表
            if 'train_loss' in epoch_data:
                if 'train_losses' not in self._history:
                    self._history['train_losses'] = []
                self._history['train_losses'].append(epoch_data['train_loss'])
            
            if 'val_loss' in epoch_data:
                if 'val_losses' not in self._history:
                    self._history['val_losses'] = []
                self._history['val_losses'].append(epoch_data['val_loss'])
            
            if 'val_acc' in epoch_data:
                if 'val_accs' not in self._history:
                    self._history['val_accs'] = []
                self._history['val_accs'].append(epoch_data['val_acc'])
            
            if 'map50' in epoch_data:
                if 'map50_list' not in self._history:
                    self._history['map50_list'] = []
                self._history['map50_list'].append(epoch_data['map50'])
            
            if 'map50_95' in epoch_data:
                if 'map50_95_list' not in self._history:
                    self._history['map50_95_list'] = []
                self._history['map50_95_list'].append(epoch_data['map50_95'])
            
            if 'score_loss' in epoch_data:
                if 'score_losses' not in self._history:
                    self._history['score_losses'] = []
                self._history['score_losses'].append(epoch_data['score_loss'])
            
            if 'score_mae' in epoch_data:
                import math
                if not math.isnan(epoch_data['score_mae']):
                    if 'score_mae_list' not in self._history:
                        self._history['score_mae_list'] = []
                    self._history['score_mae_list'].append(epoch_data['score_mae'])
            
            # 更新当前epoch
            if 'current_epoch' in epoch_data:
                self._history['current_epoch'] = epoch_data['current_epoch']
            
            # 更新最佳结果
            if 'best_acc' in epoch_data:
                self._history['best_acc'] = epoch_data['best_acc']
            if 'best_map50' in epoch_data:
                self._history['best_map50'] = epoch_data['best_map50']
            if 'best_epoch' in epoch_data:
                self._history['best_epoch'] = epoch_data['best_epoch']
            
            if auto_save:
                return self._save_all()
            return True
    
    def clear_history(self, auto_save: bool = True) -> bool:
        """清空历史数据（开始新训练时调用）"""
        with self._lock:
            self._history = {}
            if auto_save:
                return self._save_all()
            return True
    
    def has_history(self) -> bool:
        """检查是否有历史数据"""
        if not self._loaded:
            self.load()
        return bool(self._history and (
            self._history.get('train_losses') or 
            self._history.get('val_accs') or
            self._history.get('map50_list') or
            self._history.get('score_losses')
        ))
    
    # ==================== 日志相关方法 ====================
    
    def get_logs(self, log_type: str = 'training') -> List[str]:
        """
        获取日志
        
        Args:
            log_type: 日志类型 ('training', 'conversion', 'inference')
            
        Returns:
            日志行列表
        """
        if not self._loaded:
            self.load()
        return self._logs.get(log_type, []).copy()
    
    def save_logs(self, logs: List[str], log_type: str = 'training', auto_save: bool = True) -> bool:
        """
        保存日志
        
        Args:
            logs: 日志行列表
            log_type: 日志类型
            auto_save: 是否自动保存
            
        Returns:
            是否保存成功
        """
        with self._lock:
            # 限制日志行数
            self._logs[log_type] = logs[-self.MAX_LOG_LINES:] if len(logs) > self.MAX_LOG_LINES else logs
            if auto_save:
                return self._save_all()
            return True
    
    def append_log(self, line: str, log_type: str = 'training', auto_save: bool = False) -> None:
        """
        追加一行日志
        
        Args:
            line: 日志行
            log_type: 日志类型
            auto_save: 是否自动保存（频繁调用时建议False）
        """
        with self._lock:
            if log_type not in self._logs:
                self._logs[log_type] = []
            self._logs[log_type].append(line)
            
            # 限制长度
            if len(self._logs[log_type]) > self.MAX_LOG_LINES:
                self._logs[log_type] = self._logs[log_type][-self.MAX_LOG_LINES:]
            
            if auto_save:
                self._save_all()
    
    def clear_logs(self, log_type: str = None, auto_save: bool = True) -> bool:
        """
        清空日志
        
        Args:
            log_type: 日志类型，None则清空所有
            auto_save: 是否自动保存
        """
        with self._lock:
            if log_type:
                self._logs[log_type] = []
            else:
                self._logs = {}
            if auto_save:
                return self._save_all()
            return True
    
    # ==================== 工具方法 ====================
    
    def get_output_dir(self) -> Path:
        """获取输出目录"""
        output_dir = self.task_dir / 'output'
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def mark_training_complete(self, best_metric: float = None, best_epoch: int = None) -> bool:
        """
        标记训练完成
        
        Args:
            best_metric: 最佳指标值
            best_epoch: 最佳epoch
        """
        with self._lock:
            self._history['completed'] = True
            self._history['completed_at'] = datetime.now().isoformat()
            if best_metric is not None:
                if self._task_type == 'classification':
                    self._history['best_acc'] = best_metric
                elif self._task_type in ('detection', 'sevseg'):
                    self._history['best_map50'] = best_metric
                else:
                    self._history['best_metric'] = best_metric
            if best_epoch is not None:
                self._history['best_epoch'] = best_epoch
            return self._save_all()
    
    def is_training_completed(self) -> bool:
        """检查训练是否已完成"""
        if not self._loaded:
            self.load()
        return self._history.get('completed', False)


# ==================== 便捷函数 ====================

def create_params_manager(task_dir: str = None) -> DLHubParams:
    """创建参数管理器实例"""
    return DLHubParams(task_dir)


def get_task_output_dir() -> Path:
    """获取任务输出目录"""
    task_dir = os.environ.get('DLHUB_TASK_DIR')
    if task_dir:
        output_dir = Path(task_dir) / 'output'
        output_dir.mkdir(exist_ok=True)
        return output_dir
    return Path('./output')


# ==================== 测试 ====================

if __name__ == '__main__':
    # 测试参数管理器
    params = DLHubParams()
    params.load()
    
    print(f"Task type: {params.task_type}")
    print(f"DL-Hub mode: {params.is_dlhub_mode}")
    print(f"Output dir: {params.get_output_dir()}")
    print(f"Has history: {params.has_history()}")
    
    # 测试设置和获取
    params.set('training.batch_size', 32)
    params.set('training.epochs', 100)
    print(f"Batch size: {params.get('training.batch_size')}")
    
    # 测试历史数据
    params.update_history_epoch({
        'train_loss': 0.5,
        'val_loss': 0.6,
        'val_acc': 85.0,
        'current_epoch': 1
    }, auto_save=False)
    
    params.update_history_epoch({
        'train_loss': 0.3,
        'val_loss': 0.4,
        'val_acc': 90.0,
        'current_epoch': 2,
        'best_acc': 90.0,
        'best_epoch': 2
    }, auto_save=False)
    
    # 测试日志
    params.append_log("开始训练...", auto_save=False)
    params.append_log("Epoch 1 完成", auto_save=False)
    
    # 保存
    params.save()
    
    # 打印历史
    history = params.get_history()
    print(f"History: {history}")
    print(f"Logs: {params.get_logs()}")