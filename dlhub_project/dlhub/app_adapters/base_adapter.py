# -*- coding: utf-8 -*-
"""
DL-Hub 基础适配器
================
为5大任务的Gradio应用提供统一的DL-Hub集成支持

使用方法：
在每个任务的app.py中导入并初始化：

```python
from dlhub.app_adapters.base_adapter import DLHubAdapter, get_adapter

# 在app.py开头
adapter = get_adapter()

# 获取输出目录
output_dir = adapter.get_output_dir()

# 加载保存的参数
saved_params = adapter.load_params()

# 保存参数（在参数变化时调用）
adapter.save_params({
    'model': 'resnet50',
    'epochs': 100,
    ...
})

# 训练开始时
adapter.on_training_start()

# 训练结束时
adapter.on_training_end(success=True, metrics={'accuracy': 0.95})
```
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading


class DLHubAdapter:
    """
    DL-Hub适配器基类
    
    为Gradio训练应用提供与DL-Hub平台的集成支持，包括：
    - 命令行参数解析（--task-dir, --port）
    - UI参数的自动保存和加载
    - 训练状态管理（PID文件、状态更新）
    - 输出目录管理
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, default_port: int = 7861):
        """
        初始化适配器
        
        Args:
            default_port: 默认端口号
        """
        if self._initialized:
            return
        
        self._default_port = default_port
        self._task_dir: Optional[Path] = None
        self._port: int = default_port
        self._task_id: Optional[str] = None
        self._params_file: Optional[Path] = None
        self._pid_file: Optional[Path] = None
        self._save_lock = threading.Lock()
        
        # 解析命令行参数
        self._parse_args()
        
        # 从环境变量获取（如果命令行未指定）
        if self._task_dir is None:
            env_task_dir = os.environ.get('DLHUB_TASK_DIR')
            if env_task_dir:
                self._task_dir = Path(env_task_dir)
        
        if self._task_id is None:
            self._task_id = os.environ.get('DLHUB_TASK_ID')
        
        # 设置文件路径
        if self._task_dir:
            dlhub_dir = self._task_dir / '.dlhub'
            dlhub_dir.mkdir(parents=True, exist_ok=True)
            self._params_file = dlhub_dir / 'ui_params.json'
            self._pid_file = dlhub_dir / 'running.pid'
            
            print(f"[DL-Hub] 任务目录: {self._task_dir}")
            print(f"[DL-Hub] 参数文件: {self._params_file}")
        
        self._initialized = True
    
    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--task-dir', type=str, default=None,
                          help='DL-Hub任务目录路径')
        parser.add_argument('--port', type=int, default=self._default_port,
                          help=f'Gradio服务端口 (默认: {self._default_port})')
        
        # 只解析已知参数，忽略其他参数
        args, _ = parser.parse_known_args()
        
        if args.task_dir:
            self._task_dir = Path(args.task_dir)
        self._port = args.port
    
    @property
    def task_dir(self) -> Optional[Path]:
        """任务目录"""
        return self._task_dir
    
    @property
    def port(self) -> int:
        """服务端口"""
        return self._port
    
    @property
    def is_dlhub_mode(self) -> bool:
        """是否在DL-Hub模式下运行"""
        return self._task_dir is not None
    
    def get_output_dir(self, default: str = './output') -> Path:
        """
        获取输出目录
        
        Args:
            default: 非DL-Hub模式时的默认目录
            
        Returns:
            输出目录路径
        """
        if self._task_dir:
            output_dir = self._task_dir / 'output'
        else:
            output_dir = Path(default)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def load_params(self) -> Dict[str, Any]:
        """
        加载保存的UI参数
        
        Returns:
            参数字典，如果没有保存的参数则返回空字典
        """
        if not self._params_file or not self._params_file.exists():
            return {}
        
        try:
            with open(self._params_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('params', {})
        except Exception as e:
            print(f"[DL-Hub] 加载参数失败: {e}")
            return {}
    
    def save_params(self, params: Dict[str, Any]):
        """
        保存UI参数
        
        Args:
            params: 参数字典
        """
        if not self._params_file:
            return
        
        with self._save_lock:
            try:
                self._params_file.parent.mkdir(parents=True, exist_ok=True)
                
                data = {
                    'saved_at': datetime.now().isoformat(),
                    'params': params
                }
                
                with open(self._params_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"[DL-Hub] 保存参数失败: {e}")
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        获取单个参数值
        
        Args:
            key: 参数键名
            default: 默认值
            
        Returns:
            参数值
        """
        params = self.load_params()
        return params.get(key, default)
    
    def on_training_start(self):
        """训练开始时调用"""
        if not self._pid_file:
            return
        
        try:
            # 写入PID文件
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            self._pid_file.write_text(str(os.getpid()))
            
            # 更新task.json状态
            self._update_task_status('training')
            
            print(f"[DL-Hub] 训练开始，PID: {os.getpid()}")
            
        except Exception as e:
            print(f"[DL-Hub] 记录训练开始失败: {e}")
    
    def on_training_end(self, success: bool = True, metrics: Dict[str, Any] = None):
        """
        训练结束时调用
        
        Args:
            success: 是否成功完成
            metrics: 训练指标
        """
        if not self._pid_file:
            return
        
        try:
            # 删除PID文件
            if self._pid_file.exists():
                self._pid_file.unlink()
            
            # 更新task.json状态
            status = 'completed' if success else 'error'
            extra = {}
            if metrics:
                extra['best_metric'] = metrics
            
            self._update_task_status(status, extra)
            
            print(f"[DL-Hub] 训练结束，状态: {status}")
            
        except Exception as e:
            print(f"[DL-Hub] 记录训练结束失败: {e}")
    
    def _update_task_status(self, status: str, extra: Dict[str, Any] = None):
        """更新task.json中的状态"""
        if not self._task_dir:
            return
        
        task_json = self._task_dir / '.dlhub' / 'task.json'
        if not task_json.exists():
            return
        
        try:
            with open(task_json, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            task_data['status'] = status
            task_data['updated_at'] = datetime.now().isoformat()
            
            if extra:
                task_data.update(extra)
            
            with open(task_json, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"[DL-Hub] 更新任务状态失败: {e}")
    
    def create_auto_save_callback(self, param_names: List[str]):
        """
        创建自动保存回调函数，用于Gradio组件的change事件
        
        Args:
            param_names: 参数名称列表，与组件顺序对应
            
        Returns:
            回调函数
        """
        def callback(*values):
            params = dict(zip(param_names, values))
            self.save_params(params)
            return values
        
        return callback
    
    def wrap_gradio_launch(self, app):
        """
        包装Gradio应用的launch方法
        
        Args:
            app: Gradio应用对象
            
        Returns:
            包装后的应用
        """
        original_launch = app.launch
        
        def wrapped_launch(**kwargs):
            # 使用适配器的端口设置
            kwargs.setdefault('server_port', self._port)
            kwargs.setdefault('server_name', '127.0.0.1')
            kwargs.setdefault('share', False)
            kwargs.setdefault('inbrowser', False)  # DL-Hub模式下不自动打开浏览器
            
            return original_launch(**kwargs)
        
        app.launch = wrapped_launch
        return app


# 全局适配器实例
_adapter: Optional[DLHubAdapter] = None


def get_adapter(default_port: int = 7861) -> DLHubAdapter:
    """
    获取全局适配器实例
    
    Args:
        default_port: 默认端口号
        
    Returns:
        适配器实例
    """
    global _adapter
    if _adapter is None:
        _adapter = DLHubAdapter(default_port=default_port)
    return _adapter
