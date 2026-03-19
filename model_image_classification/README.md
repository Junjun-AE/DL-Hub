# 🔬 TIMM 图像分类训练工具

基于 [timm](https://github.com/huggingface/pytorch-image-models) 库的图像分类模型训练工具，提供现代化的 Gradio Web 界面。

## ✨ 功能特性

- 🧠 **15个预置模型**: EfficientNet、MobileNetV3、WideResNet 三大系列
- 🎨 **现代Web界面**: 基于 Gradio，支持浏览器和手机访问
- 📊 **实时训练曲线**: Loss 和 Accuracy 实时可视化
- 🖥️ **多GPU支持**: 自动检测GPU，支持单卡/多卡训练
- 💾 **规范化保存**: 完整元数据，便于后续部署
- ⏹️ **断点续训**: 停止后可继续训练
- 🛑 **早停机制**: 自动防止过拟合
- 🔄 **自动下载权重**: 本地无权重时自动从 timm 下载

## 📦 支持的模型

| 系列 | 规模 | 模型名称 | 参数量 | ImageNet精度 |
|------|------|----------|--------|--------------|
| EfficientNet | 超小 | efficientnet_b0 | 5.3M | 77.1% |
| EfficientNet | 小 | efficientnet_b1 | 7.8M | 79.1% |
| EfficientNet | 中 | efficientnet_b2 | 9.2M | 80.1% |
| EfficientNet | 大 | efficientnet_b3 | 12.0M | 81.6% |
| EfficientNet | 超大 | efficientnet_b4 | 19.0M | 82.9% |
| MobileNetV3 | 超小 | mobilenetv3_small_050 | 1.0M | 57.9% |
| MobileNetV3 | 小 | mobilenetv3_small_100 | 2.5M | 67.7% |
| MobileNetV3 | 中 | mobilenetv3_large_075 | 4.0M | 73.4% |
| MobileNetV3 | 大 | mobilenetv3_large_100 | 5.4M | 75.2% |
| MobileNetV3 | 超大 | tf_mobilenetv3_large_100 | 5.5M | 75.5% |
| WideResNet | 超小 | wide_resnet50_2 | 68.9M | 81.5% |
| WideResNet | 小 | wide_resnet101_2 | 126.9M | 82.5% |
| WideResNet | 中 | resnet50 | 25.6M | 80.4% |
| WideResNet | 大 | resnet101 | 44.5M | 81.9% |
| WideResNet | 超大 | resnet152 | 60.2M | 82.3% |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

数据集需要是 **ImageFolder** 格式：

```
dataset/
├── train/          # 可选，如果只有根目录会自动划分
│   ├── class_a/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_b/
│       └── img3.jpg
└── val/            # 可选
    ├── class_a/
    └── class_b/
```

或者简单格式（自动按比例划分验证集）：

```
dataset/
├── class_a/
│   ├── img1.jpg
│   └── img2.jpg
└── class_b/
    └── img3.jpg
```

### 3. 启动训练界面

```bash
python app.py
```

浏览器会自动打开 `http://localhost:7860`

### 4. 使用本地预训练权重（可选）

将预训练权重文件放入 `models/pretrained/` 目录：

```
models/pretrained/
├── efficientnet_b0.pth
├── efficientnet_b2.pth
└── ...
```

如果没有本地权重，程序会自动从 timm 在线下载。

## 🎛️ 可配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 数据集路径 | 文本 | 必填 | ImageFolder格式数据集 |
| 模型系列 | 选择 | EfficientNet | 三个系列可选 |
| 模型规模 | 选择 | 中 | 5个规模可选 |
| 训练轮数 | 滑动条 | 100 | 10-300 |
| 批次大小 | 滑动条 | 32 | 8-128 |
| 学习率 | 数字 | 0.001 | 初始学习率 |
| 优化器 | 下拉框 | AdamW | Adam/AdamW/SGD |
| 图像尺寸 | 数字 | 224 | 输入图像大小 |
| 验证集比例 | 滑动条 | 0.2 | 自动划分时使用 |
| 保存频率 | 数字 | 5 | 每N轮保存 |
| 启用早停 | 复选框 | ✅ | 防止过拟合 |
| GPU选择 | 下拉框 | 自动 | CPU/GPU0/全部GPU |

## 📁 输出目录结构

```
output/efficientnet_b2_20250101_143052/
├── best_model.pth      # 最佳模型（完整元数据）
├── last_model.pth      # 最后一轮模型
├── interrupted_model.pth # 中断时保存（如有）
├── checkpoints/        # 定期保存的检查点
│   ├── epoch_5.pth
│   └── epoch_10.pth
├── logs/
│   └── training.log    # 训练日志
└── config.yaml         # 训练配置记录
```

## 📦 模型文件格式

保存的模型包含完整元数据，便于后续部署：

```python
checkpoint = {
    "state_dict": 模型权重,
    "model_metadata": {
        "model_name": "efficientnet_b2",
        "model_family": "EfficientNet",
        "model_scale": "中",
        "num_classes": 5,
        "input_spec": {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bilinear",
        },
    },
    "training_state": {
        "epoch": 50,
        "best_acc": 95.2,
        "optimizer_state_dict": ...,
        "scheduler_state_dict": ...,
    },
    "training_config": {...},
}
```

## 🔧 常见问题

### Q: 没有GPU可以用吗？
A: 可以，会自动使用CPU训练，但速度较慢。

### Q: 如何使用多GPU？
A: 在GPU选择下拉框中选择"全部GPU"即可。

### Q: 预训练权重下载失败？
A: 检查网络连接，或手动下载后放入 `models/pretrained/` 目录。

### Q: 训练中断了怎么办？
A: 程序会自动保存 `interrupted_model.pth`，下次可以继续训练。

## 📄 许可证

MIT License
