# DL-Hub 应用集成指南
## 如何将现有Gradio应用与DL-Hub集成

本指南说明如何修改5大任务的Gradio应用以支持DL-Hub平台的集成功能。

---

## 需要修改的内容

每个任务的`app.py`需要添加以下功能：
1. 命令行参数支持（`--task-dir`, `--port`）
2. 参数的自动加载和保存
3. 输出目录重定向到任务目录
4. 训练状态上报

---

## 修改模板

### 1. 在文件开头添加导入和适配器初始化

```python
# 在 app.py 文件的开头添加
import sys
from pathlib import Path

# 添加dlhub到路径（如果需要）
DLHUB_PATH = Path(__file__).parent.parent / 'dlhub'
if DLHUB_PATH.exists():
    sys.path.insert(0, str(DLHUB_PATH.parent))

# 导入并初始化适配器
try:
    from dlhub.app_adapters.base_adapter import get_adapter
    adapter = get_adapter(default_port=7861)  # 根据任务类型设置默认端口
except ImportError:
    adapter = None
    print("[Warning] DL-Hub adapter not found, running in standalone mode")
```

### 2. 修改输出目录获取逻辑

```python
# 原来的代码
# output_dir = Path('./output')

# 修改为
def get_output_dir():
    if adapter and adapter.is_dlhub_mode:
        return adapter.get_output_dir()
    return Path('./output')

output_dir = get_output_dir()
```

### 3. 修改UI组件的默认值加载

```python
# 加载保存的参数
saved_params = adapter.load_params() if adapter else {}

# 创建UI组件时使用保存的值
model_dropdown = gr.Dropdown(
    choices=model_list,
    value=saved_params.get('model', 'resnet50'),  # 使用保存的值或默认值
    label="选择模型"
)

epochs_slider = gr.Slider(
    minimum=1,
    maximum=500,
    value=saved_params.get('epochs', 100),  # 使用保存的值或默认值
    label="训练轮数"
)
```

### 4. 添加参数自动保存逻辑

有两种方式实现参数实时保存：

#### 方式A：使用change事件（推荐）

```python
# 定义参数保存函数
def save_all_params(model, epochs, batch_size, lr, ...):
    if adapter:
        adapter.save_params({
            'model': model,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            # ... 其他参数
        })
    return model, epochs, batch_size, lr, ...

# 为每个组件添加change事件
all_components = [model_dropdown, epochs_slider, batch_size_slider, lr_input, ...]

for comp in all_components:
    comp.change(
        fn=save_all_params,
        inputs=all_components,
        outputs=all_components
    )
```

#### 方式B：在训练开始时保存

```python
def start_training(model, epochs, batch_size, lr, ...):
    # 保存当前参数
    if adapter:
        adapter.save_params({
            'model': model,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
        })
    
    # 继续原有的训练逻辑
    ...
```

### 5. 添加训练状态上报

```python
def start_training(...):
    # 训练开始
    if adapter:
        adapter.on_training_start()
    
    try:
        # 原有的训练逻辑
        result = train(...)
        
        # 训练成功结束
        if adapter:
            adapter.on_training_end(
                success=True,
                metrics={'accuracy': best_acc}  # 根据任务类型调整指标
            )
        
        return result
        
    except Exception as e:
        # 训练失败
        if adapter:
            adapter.on_training_end(success=False)
        raise e
```

### 6. 修改Gradio启动配置

```python
# 原来的代码
# app.launch(server_name="0.0.0.0", server_port=7860)

# 修改为
if __name__ == "__main__":
    launch_kwargs = {
        'server_name': '0.0.0.0',
        'share': False,
    }
    
    if adapter:
        launch_kwargs['server_port'] = adapter.port
        launch_kwargs['inbrowser'] = False  # DL-Hub模式下不自动打开浏览器
    else:
        launch_kwargs['server_port'] = 7860
        launch_kwargs['inbrowser'] = True
    
    app.launch(**launch_kwargs)
```

---

## 完整示例：图像分类任务

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图像分类训练应用 - DL-Hub集成版"""

import sys
from pathlib import Path
import gradio as gr

# DL-Hub集成
try:
    from dlhub.app_adapters.base_adapter import get_adapter
    adapter = get_adapter(default_port=7861)
except ImportError:
    adapter = None

# 加载保存的参数
saved_params = adapter.load_params() if adapter else {}

# 获取输出目录
def get_output_dir():
    if adapter and adapter.is_dlhub_mode:
        return adapter.get_output_dir()
    return Path('./output')

# 创建Gradio界面
with gr.Blocks() as app:
    gr.Markdown("# 图像分类训练")
    
    with gr.Row():
        model = gr.Dropdown(
            choices=['resnet18', 'resnet50', 'efficientnet_b0'],
            value=saved_params.get('model', 'resnet50'),
            label="模型"
        )
        epochs = gr.Slider(1, 500, value=saved_params.get('epochs', 100), label="Epochs")
    
    # 参数保存
    def save_params(m, e):
        if adapter:
            adapter.save_params({'model': m, 'epochs': e})
        return m, e
    
    model.change(save_params, [model, epochs], [model, epochs])
    epochs.change(save_params, [model, epochs], [model, epochs])
    
    # 训练逻辑
    def train(m, e):
        if adapter:
            adapter.on_training_start()
        
        try:
            output_dir = get_output_dir()
            # ... 训练代码 ...
            
            if adapter:
                adapter.on_training_end(True, {'accuracy': 0.95})
            return "训练完成"
        except Exception as ex:
            if adapter:
                adapter.on_training_end(False)
            return f"训练失败: {ex}"
    
    train_btn = gr.Button("开始训练")
    output = gr.Textbox(label="输出")
    train_btn.click(train, [model, epochs], output)

# 启动
if __name__ == "__main__":
    port = adapter.port if adapter else 7860
    app.launch(server_name="0.0.0.0", server_port=port, inbrowser=not adapter)
```

---

## 注意事项

1. **向后兼容**：修改后的代码应该在没有DL-Hub的情况下也能正常运行
2. **参数保存频率**：参数会在每次change事件时保存，这是实时的
3. **输出目录**：所有训练输出应该保存到`adapter.get_output_dir()`返回的目录
4. **状态上报**：确保在训练开始和结束时调用相应的方法

---

## 各任务的端口配置

| 任务 | 默认端口 |
|------|---------|
| DL-Hub主界面 | 7860 |
| 分类 | 7861 |
| 检测 | 7861 |
| 分割 | 7861 |
| 异常检测 | 7861 |
| OCR | 7861 |

由于是单任务模式，所有训练应用统一使用7861端口。
