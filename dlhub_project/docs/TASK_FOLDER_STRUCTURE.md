# DL-Hub 任务文件夹结构详解

## 1. 文件夹创建规则

### 1.1 命名规则

```
{任务名称}_{时间戳}_{唯一ID}/
```

**示例**: `产品缺陷检测_20250129_143052_a1b2c3/`

| 组成部分 | 格式 | 示例 |
|---------|------|------|
| 任务名称 | 用户输入，经过清理 | `产品缺陷检测` |
| 时间戳 | `YYYYMMDD_HHMMSS` | `20250129_143052` |
| 唯一ID | 6位十六进制 | `a1b2c3` |

### 1.2 任务ID规则

```
{类型前缀}_{时间戳}_{唯一ID}
```

| 任务类型 | 前缀 | 示例 |
|---------|------|------|
| classification | `cla` | `cla_20250129_143052_a1b2c3` |
| detection | `det` | `det_20250129_143052_a1b2c3` |
| segmentation | `seg` | `seg_20250129_143052_a1b2c3` |
| anomaly | `ano` | `ano_20250129_143052_a1b2c3` |
| ocr | `ocr` | `ocr_20250129_143052_a1b2c3` |

---

## 2. 当前文件夹结构

```
任务名称_20250129_143052_a1b2c3/
├── .dlhub/                          # DL-Hub 管理目录
│   ├── task.json                    # 任务元数据（必须）
│   ├── ui_params.json               # UI参数（待实现）
│   └── running.pid                  # 运行中的进程ID（临时）
│
├── output/                          # 训练输出目录
│   └── train_20250129_150000/       # 单次训练结果
│       ├── checkpoints/             # 模型检查点
│       │   ├── best_model.pth
│       │   ├── last_model.pth
│       │   └── epoch_*.pth
│       ├── logs/                    # 训练日志
│       │   └── events.out.tfevents.*
│       ├── model_metadata.json      # 模型元数据
│       └── config.yaml              # 训练配置
│
└── data/                            # 数据目录（可选）
    ├── train/
    ├── val/
    └── test/
```

---

## 3. 核心文件说明

### 3.1 task.json（任务元数据）

**位置**: `.dlhub/task.json`

```json
{
  "dlhub_signature": "dlhub_task_v1",
  "dlhub_version": "2.0",
  "task_id": "det_20250129_143052_a1b2c3",
  "task_type": "detection",
  "task_name": "产品缺陷检测",
  "description": "检测PCB板上的焊接缺陷",
  "conda_env": "yolov8",
  "created_at": "2025-01-29T14:30:52.123456",
  "updated_at": "2025-01-29T16:45:30.789012",
  "status": "completed"
}
```

### 3.2 ui_params.json（UI参数 - 新设计）

**位置**: `.dlhub/ui_params.json`

这是**核心新增文件**，用于保存各App的界面参数。

---

## 4. 五大任务的参数结构设计

### 4.1 图像分类 (classification)

```json
{
  "version": "1.0",
  "task_type": "classification",
  "saved_at": "2025-01-29T16:45:30.789012",
  "params": {
    "data": {
      "train_dir": "D:/datasets/cls/train",
      "val_dir": "D:/datasets/cls/val",
      "test_dir": "",
      "image_size": 224,
      "num_classes": 10,
      "class_names": ["cat", "dog", "bird", ...]
    },
    "model": {
      "backbone": "resnet50",
      "pretrained": true,
      "freeze_layers": 0
    },
    "training": {
      "batch_size": 32,
      "epochs": 100,
      "learning_rate": 0.001,
      "optimizer": "adam",
      "scheduler": "cosine",
      "weight_decay": 0.0001,
      "early_stopping": true,
      "patience": 10
    },
    "augmentation": {
      "horizontal_flip": true,
      "vertical_flip": false,
      "rotation": 15,
      "color_jitter": true,
      "random_crop": true,
      "mixup": false,
      "cutout": false
    },
    "export": {
      "format": "onnx",
      "quantize": false,
      "input_shape": [1, 3, 224, 224]
    }
  }
}
```

### 4.2 目标检测 (detection)

```json
{
  "version": "1.0",
  "task_type": "detection",
  "saved_at": "2025-01-29T16:45:30.789012",
  "params": {
    "data": {
      "data_yaml": "D:/datasets/det/data.yaml",
      "train_images": "D:/datasets/det/images/train",
      "val_images": "D:/datasets/det/images/val",
      "image_size": 640,
      "num_classes": 5,
      "class_names": ["person", "car", "bike", ...]
    },
    "model": {
      "architecture": "yolov8",
      "variant": "n",
      "pretrained": "yolov8n.pt"
    },
    "training": {
      "batch_size": 16,
      "epochs": 300,
      "learning_rate": 0.01,
      "optimizer": "SGD",
      "momentum": 0.937,
      "weight_decay": 0.0005,
      "warmup_epochs": 3,
      "patience": 50,
      "workers": 8
    },
    "augmentation": {
      "hsv_h": 0.015,
      "hsv_s": 0.7,
      "hsv_v": 0.4,
      "degrees": 0.0,
      "translate": 0.1,
      "scale": 0.5,
      "shear": 0.0,
      "perspective": 0.0,
      "flipud": 0.0,
      "fliplr": 0.5,
      "mosaic": 1.0,
      "mixup": 0.0
    },
    "export": {
      "format": "onnx",
      "half": false,
      "dynamic": false,
      "simplify": true,
      "opset": 12
    }
  }
}
```

### 4.3 语义分割 (segmentation)

```json
{
  "version": "1.0",
  "task_type": "segmentation",
  "saved_at": "2025-01-29T16:45:30.789012",
  "params": {
    "data": {
      "train_images": "D:/datasets/seg/images/train",
      "train_masks": "D:/datasets/seg/masks/train",
      "val_images": "D:/datasets/seg/images/val",
      "val_masks": "D:/datasets/seg/masks/val",
      "image_size": [512, 512],
      "num_classes": 21,
      "class_names": ["background", "aeroplane", ...]
    },
    "model": {
      "architecture": "segformer",
      "variant": "b2",
      "pretrained": true,
      "encoder_weights": "imagenet"
    },
    "training": {
      "batch_size": 8,
      "epochs": 200,
      "learning_rate": 0.0001,
      "optimizer": "adamw",
      "scheduler": "poly",
      "weight_decay": 0.01,
      "ignore_index": 255,
      "loss": "cross_entropy"
    },
    "augmentation": {
      "random_crop": true,
      "random_flip": true,
      "color_jitter": true,
      "random_scale": [0.5, 2.0],
      "normalize": true
    },
    "export": {
      "format": "onnx",
      "input_shape": [1, 3, 512, 512]
    }
  }
}
```

### 4.4 异常检测 (anomaly)

```json
{
  "version": "1.0",
  "task_type": "anomaly",
  "saved_at": "2025-01-29T16:45:30.789012",
  "params": {
    "data": {
      "good_dir": "D:/datasets/anomaly/good",
      "test_dir": "D:/datasets/anomaly/test",
      "mask_dir": "D:/datasets/anomaly/ground_truth",
      "image_size": 224,
      "center_crop": 224
    },
    "model": {
      "algorithm": "patchcore",
      "backbone": "wide_resnet50_2",
      "layers": ["layer2", "layer3"],
      "coreset_sampling_ratio": 0.01,
      "num_neighbors": 9
    },
    "training": {
      "batch_size": 32,
      "seed": 42
    },
    "threshold": {
      "method": "auto",
      "percentile": 99.5,
      "manual_value": null
    },
    "export": {
      "format": "pkg",
      "include_visualizer": true
    }
  }
}
```

### 4.5 OCR识别 (ocr)

```json
{
  "version": "1.0",
  "task_type": "ocr",
  "saved_at": "2025-01-29T16:45:30.789012",
  "params": {
    "data": {
      "train_images": "D:/datasets/ocr/images",
      "train_labels": "D:/datasets/ocr/labels.txt",
      "val_images": "D:/datasets/ocr/val_images",
      "val_labels": "D:/datasets/ocr/val_labels.txt",
      "max_text_length": 25,
      "character_set": "chinese"
    },
    "model": {
      "det_model": "ch_PP-OCRv4_det",
      "rec_model": "ch_PP-OCRv4_rec",
      "cls_model": "ch_ppocr_mobile_v2.0_cls",
      "use_angle_cls": true,
      "lang": "ch"
    },
    "detection": {
      "det_algorithm": "DB",
      "det_limit_side_len": 960,
      "det_db_thresh": 0.3,
      "det_db_box_thresh": 0.6,
      "det_db_unclip_ratio": 1.5
    },
    "recognition": {
      "rec_algorithm": "SVTR_LCNet",
      "rec_image_shape": "3,48,320",
      "rec_batch_num": 6
    },
    "training": {
      "batch_size": 64,
      "epochs": 500,
      "learning_rate": 0.001,
      "optimizer": "adam"
    },
    "export": {
      "format": "onnx",
      "include_det": true,
      "include_rec": true
    }
  }
}
```

---

## 5. 参数保存/加载API设计

### 5.1 后端API

```python
# 保存参数
POST /api/tasks/{task_id}/params
Body: {
  "params": { ... }
}

# 获取参数
GET /api/tasks/{task_id}/params
Response: {
  "params": { ... },
  "saved_at": "2025-01-29T16:45:30"
}

# 获取默认参数模板
GET /api/tasks/params/template/{task_type}
Response: {
  "params": { ... }  # 该任务类型的默认参数
}
```

### 5.2 App端集成

各训练App需要：
1. **启动时读取参数**: 检查 `DLHUB_TASK_DIR/.dlhub/ui_params.json`
2. **训练前保存参数**: 将当前UI参数保存到文件
3. **参数变更时自动保存**: 可选，实时保存

---

## 6. 实现步骤

### Step 1: 后端API扩展
- 添加 `POST /api/tasks/{task_id}/params` 保存参数
- 添加 `GET /api/tasks/params/template/{task_type}` 获取模板

### Step 2: 前端集成
- 启动任务时传递参数文件路径
- 显示参数保存状态

### Step 3: App适配
- 修改各App读取参数文件
- 训练前自动保存参数
- 提供参数重置功能

---

## 7. 文件夹完整结构（最终版）

```
任务名称_20250129_143052_a1b2c3/
│
├── .dlhub/                          # DL-Hub 管理目录（隐藏）
│   ├── task.json                    # 任务元数据 ✅ 已实现
│   ├── ui_params.json               # UI参数 🆕 待实现
│   ├── training_history.json        # 训练历史 🆕 可选
│   └── running.pid                  # 运行进程ID（临时）
│
├── output/                          # 训练输出
│   ├── train_20250129_150000/       # 训练结果1
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── ...
│   └── train_20250129_170000/       # 训练结果2
│
├── data/                            # 数据（可选，可以是软链接）
│   ├── train/
│   ├── val/
│   └── test/
│
├── exports/                         # 导出的模型
│   ├── model.onnx
│   ├── model.pkg
│   └── ...
│
└── README.md                        # 任务说明（可选）
```
