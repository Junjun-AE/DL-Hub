# DL-Hub 参数保存/加载集成指南

本文档详细说明如何在各训练App中实现参数的保存和加载功能。

## 1. 概述

DL-Hub为每个任务维护一个参数文件 `.dlhub/ui_params.json`，用于：
- **保存**：训练前将App界面的参数保存到文件
- **加载**：打开任务时从文件恢复参数到界面

## 2. 参数文件位置

```
{任务目录}/
└── .dlhub/
    └── ui_params.json    <-- 参数文件
```

## 3. 参数文件格式

```json
{
  "version": "1.0",
  "task_type": "detection",
  "task_id": "det_20250129_143052_a1b2c3",
  "saved_at": "2025-01-29T16:45:30.789012",
  "params": {
    "data": {
      "train_dir": "D:/datasets/train",
      "val_dir": "D:/datasets/val",
      "image_size": 640
    },
    "model": {
      "backbone": "yolov8n",
      "pretrained": true
    },
    "training": {
      "batch_size": 16,
      "epochs": 100,
      "learning_rate": 0.01
    },
    "augmentation": {
      "flip": true,
      "mosaic": true
    },
    "export": {
      "format": "onnx"
    }
  }
}
```

## 4. Python参数管理类

在各App中复制使用以下类：

```python
# dlhub_params.py - 放在各App目录下

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class DLHubParams:
    """
    DL-Hub参数管理器
    
    用于在Gradio App中加载和保存UI参数
    
    使用示例:
        params = DLHubParams()
        params.load()
        
        # 获取参数
        batch_size = params.get('training.batch_size', 32)
        
        # 设置并保存
        params.set('training.batch_size', 64)
        params.save()
    """
    
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
        self._loaded = False
        self._task_type = self._read_task_type()
    
    def _read_task_type(self) -> str:
        """从task.json读取任务类型"""
        task_json = self.task_dir / '.dlhub' / 'task.json'
        if task_json.exists():
            try:
                with open(task_json, 'r', encoding='utf-8') as f:
                    return json.load(f).get('task_type', 'unknown')
            except:
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
        从文件加载参数
        
        Returns:
            参数字典
        """
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._params = data.get('params', {})
                self._loaded = True
                print(f"[DL-Hub] ✓ 已加载参数: {self.params_file}")
            except Exception as e:
                print(f"[DL-Hub] ✗ 加载参数失败: {e}")
                self._params = {}
        else:
            print(f"[DL-Hub] 参数文件不存在，将使用默认值")
            self._params = {}
        
        return self._params
    
    def save(self, params: Dict[str, Any] = None) -> bool:
        """
        保存参数到文件
        
        Args:
            params: 要保存的参数字典，None则保存当前内存中的参数
            
        Returns:
            是否保存成功
        """
        if params is not None:
            self._params = params
        
        # 确保目录存在
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '1.0',
            'task_type': self._task_type,
            'task_id': self.task_id,
            'saved_at': datetime.now().isoformat(),
            'params': self._params
        }
        
        try:
            with open(self.params_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[DL-Hub] ✓ 参数已保存")
            return True
        except Exception as e:
            print(f"[DL-Hub] ✗ 保存参数失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取参数值
        
        支持点号分隔的嵌套键，如 'training.batch_size'
        
        Args:
            key: 参数键名
            default: 默认值
            
        Returns:
            参数值
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
        """
        设置参数值
        
        支持点号分隔的嵌套键
        
        Args:
            key: 参数键名
            value: 参数值
        """
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
    
    def get_output_dir(self) -> Path:
        """获取输出目录"""
        output_dir = self.task_dir / 'output'
        output_dir.mkdir(exist_ok=True)
        return output_dir


# 便捷函数
def create_params_manager(task_dir: str = None) -> DLHubParams:
    """创建参数管理器实例"""
    return DLHubParams(task_dir)
```

## 5. 在Gradio App中使用

### 5.1 基本使用模式

```python
import gradio as gr
import argparse
from dlhub_params import DLHubParams

# 初始化参数管理器
params_manager = DLHubParams()

# 加载已保存的参数
params_manager.load()

# 创建Gradio界面
with gr.Blocks(title="训练界面") as app:
    
    # ====== 数据配置 Tab ======
    with gr.Tab("数据配置"):
        train_dir = gr.Textbox(
            label="训练数据目录",
            value=params_manager.get('data.train_dir', ''),
            placeholder="选择或输入训练数据路径"
        )
        val_dir = gr.Textbox(
            label="验证数据目录", 
            value=params_manager.get('data.val_dir', ''),
        )
        image_size = gr.Slider(
            label="图像尺寸",
            minimum=32, maximum=1280, step=32,
            value=params_manager.get('data.image_size', 640),
        )
    
    # ====== 模型配置 Tab ======
    with gr.Tab("模型配置"):
        backbone = gr.Dropdown(
            label="骨干网络",
            choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
            value=params_manager.get('model.backbone', 'yolov8n'),
        )
        pretrained = gr.Checkbox(
            label="使用预训练权重",
            value=params_manager.get('model.pretrained', True),
        )
    
    # ====== 训练配置 Tab ======
    with gr.Tab("训练配置"):
        batch_size = gr.Slider(
            label="批次大小",
            minimum=1, maximum=128, step=1,
            value=params_manager.get('training.batch_size', 16),
        )
        epochs = gr.Slider(
            label="训练轮数",
            minimum=1, maximum=1000, step=1,
            value=params_manager.get('training.epochs', 100),
        )
        learning_rate = gr.Number(
            label="学习率",
            value=params_manager.get('training.learning_rate', 0.01),
        )
    
    # ====== 按钮区域 ======
    with gr.Row():
        save_btn = gr.Button("💾 保存参数", variant="secondary")
        train_btn = gr.Button("🚀 开始训练", variant="primary")
    
    output_log = gr.Textbox(label="输出日志", lines=10)
    
    # ====== 回调函数 ======
    def save_current_params(train_dir, val_dir, image_size, 
                           backbone, pretrained,
                           batch_size, epochs, learning_rate):
        """保存当前UI参数到文件"""
        params = {
            'data': {
                'train_dir': train_dir,
                'val_dir': val_dir,
                'image_size': int(image_size),
            },
            'model': {
                'backbone': backbone,
                'pretrained': pretrained,
            },
            'training': {
                'batch_size': int(batch_size),
                'epochs': int(epochs),
                'learning_rate': float(learning_rate),
            }
        }
        
        success = params_manager.save(params)
        return "✅ 参数已保存到任务目录" if success else "❌ 保存失败"
    
    def start_training(train_dir, val_dir, image_size,
                      backbone, pretrained,
                      batch_size, epochs, learning_rate):
        """开始训练"""
        # 1. 先保存当前参数
        save_result = save_current_params(
            train_dir, val_dir, image_size,
            backbone, pretrained,
            batch_size, epochs, learning_rate
        )
        yield f"{save_result}\n开始训练...\n"
        
        # 2. 获取输出目录
        output_dir = params_manager.get_output_dir()
        yield f"输出目录: {output_dir}\n"
        
        # 3. 执行训练逻辑...
        # TODO: 在这里添加实际的训练代码
        
        yield "训练完成！"
    
    # 绑定按钮事件
    all_inputs = [train_dir, val_dir, image_size, 
                  backbone, pretrained,
                  batch_size, epochs, learning_rate]
    
    save_btn.click(
        fn=save_current_params,
        inputs=all_inputs,
        outputs=output_log
    )
    
    train_btn.click(
        fn=start_training,
        inputs=all_inputs,
        outputs=output_log
    )


# ====== 主函数 ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=str, help='DL-Hub任务目录')
    parser.add_argument('--port', type=int, default=7861, help='服务端口')
    args = parser.parse_args()
    
    # 如果指定了任务目录，设置环境变量
    if args.task_dir:
        import os
        os.environ['DLHUB_TASK_DIR'] = args.task_dir
        # 重新初始化参数管理器
        params_manager = DLHubParams(args.task_dir)
        params_manager.load()
    
    # 启动Gradio应用
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        inbrowser=False  # DL-Hub会在新标签页打开
    )
```

### 5.2 参数自动保存（可选）

如果希望参数实时自动保存，可以为每个组件添加change事件：

```python
# 定义自动保存函数
def auto_save_params(*args):
    # 将参数打包保存
    params_manager.save({
        'data': {'train_dir': args[0], 'val_dir': args[1], 'image_size': int(args[2])},
        'model': {'backbone': args[3], 'pretrained': args[4]},
        'training': {'batch_size': int(args[5]), 'epochs': int(args[6]), 'learning_rate': float(args[7])}
    })
    return args  # 返回原值

# 为所有组件添加change事件
for component in all_inputs:
    component.change(
        fn=auto_save_params,
        inputs=all_inputs,
        outputs=all_inputs
    )
```

## 6. 各任务类型的参数结构

### 6.1 图像分类 (classification)

```python
params = {
    'data': {
        'train_dir': '',        # 训练集目录
        'val_dir': '',          # 验证集目录
        'test_dir': '',         # 测试集目录（可选）
        'image_size': 224,      # 图像尺寸
        'num_classes': 0,       # 类别数
    },
    'model': {
        'backbone': 'resnet50', # 骨干网络
        'pretrained': True,     # 预训练权重
        'freeze_layers': 0,     # 冻结层数
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'weight_decay': 0.0001,
        'early_stopping': True,
        'patience': 10,
    },
    'augmentation': {
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation': 15,
        'color_jitter': True,
    },
    'export': {
        'format': 'onnx',
        'quantize': False,
    }
}
```

### 6.2 目标检测 (detection)

```python
params = {
    'data': {
        'data_yaml': '',        # YOLO格式的data.yaml路径
        'image_size': 640,
    },
    'model': {
        'architecture': 'yolov8',
        'variant': 'n',         # n/s/m/l/x
        'pretrained': 'yolov8n.pt',
    },
    'training': {
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.01,
        'optimizer': 'SGD',
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 50,
    },
    'augmentation': {
        'mosaic': 1.0,
        'mixup': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,
        'flipud': 0.0,
    },
    'export': {
        'format': 'onnx',
        'half': False,
        'simplify': True,
    }
}
```

### 6.3 语义分割 (segmentation)

```python
params = {
    'data': {
        'train_images': '',
        'train_masks': '',
        'val_images': '',
        'val_masks': '',
        'image_size': [512, 512],
        'num_classes': 0,
        'ignore_index': 255,
    },
    'model': {
        'architecture': 'segformer',
        'variant': 'b2',
        'pretrained': True,
    },
    'training': {
        'batch_size': 8,
        'epochs': 200,
        'learning_rate': 0.0001,
        'optimizer': 'adamw',
        'scheduler': 'poly',
        'loss': 'cross_entropy',
    },
    'augmentation': {
        'random_crop': True,
        'random_flip': True,
        'color_jitter': True,
        'random_scale': [0.5, 2.0],
    },
    'export': {
        'format': 'onnx',
    }
}
```

### 6.4 异常检测 (anomaly)

```python
params = {
    'data': {
        'good_dir': '',         # 正常样本目录
        'test_dir': '',         # 测试样本目录
        'mask_dir': '',         # 标注掩码目录（可选）
        'image_size': 224,
        'center_crop': 224,
    },
    'model': {
        'algorithm': 'patchcore',
        'backbone': 'wide_resnet50_2',
        'layers': ['layer2', 'layer3'],
        'coreset_sampling_ratio': 0.01,
        'num_neighbors': 9,
    },
    'threshold': {
        'method': 'auto',       # auto/manual
        'percentile': 99.5,
        'manual_value': None,
    },
    'visualization': {
        'show_heatmap': True,
        'colormap': 'jet',
    },
    'export': {
        'format': 'pkg',
    }
}
```

### 6.5 OCR识别 (ocr)

```python
params = {
    'data': {
        'train_images': '',
        'train_labels': '',
        'val_images': '',
        'val_labels': '',
        'max_text_length': 25,
    },
    'model': {
        'det_model': 'ch_PP-OCRv4_det',
        'rec_model': 'ch_PP-OCRv4_rec',
        'use_angle_cls': True,
        'lang': 'ch',
    },
    'detection': {
        'det_db_thresh': 0.3,
        'det_db_box_thresh': 0.6,
        'det_db_unclip_ratio': 1.5,
    },
    'recognition': {
        'rec_batch_num': 6,
    },
    'export': {
        'format': 'onnx',
    }
}
```

## 7. 注意事项

1. **向后兼容**：修改后的App应该在没有DL-Hub的情况下也能正常运行
2. **默认值**：始终为 `params.get()` 提供合理的默认值
3. **类型转换**：从UI获取的值可能是字符串，保存前注意类型转换
4. **路径处理**：路径使用正斜杠或原生Path，避免跨平台问题
5. **错误处理**：参数加载/保存失败不应中断程序运行

## 8. 测试方法

```bash
# 模拟DL-Hub启动方式
python app.py --task-dir "D:/tasks/my_task_20250129_123456_abc123" --port 7861

# 独立运行（不使用DL-Hub）
python app.py --port 7860
```
