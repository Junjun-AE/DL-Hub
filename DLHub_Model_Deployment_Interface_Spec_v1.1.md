# DL-Hub 通用模型部署接口规范文档（勘误校订版）

> **版本**: 1.1.0 (勘误校订版)  
> **日期**: 2026-03-19  
> **适用范围**: Deep_learning_tools4_final_v2 全部 6 大任务  
> **目标读者**: 部署软件开发者  
> **校订说明**: 本版对照全部源码逐行校验，修正了 v1.0 中的多项错误

---

## 勘误摘要（相比 v1.0 的重要修正）

1. **分类模型 checkpoint 结构完全重写**: 实际保存格式以 model_importer 兼容为目标，顶层包含 `model`(权重)、`framework`、`task`、`model_name`、`normalize_mean`/`normalize_std`、`class_to_idx`、`idx_to_class`、`class_names` 等字段，而非之前描述的 `model_state_dict` + `model_info` 嵌套结构
2. **OCR 和 PatchCore 没有 .dlhub 打包**: OCR 使用独立的 `.pkg` 格式（ZIP包，内含自己的 `config.json` V2 格式）；PatchCore 使用独立的 `.pkg` 格式（ZIP包，内含 `config.json` + 多组件文件）。两者均**不使用** DLHubPackager
3. **SevSeg severity 原始值在 [0,1]**: 模型 ONNX 输出的第7列 severity 值范围是 [0,1]，后处理需要 `×10` 才得到 [0,10] 的最终分数，v1.0 中说"不需要 denormalize"是错误的
4. **分割任务 letterbox 填充色是 0 不是 114**: 代码中 `pad_val=0`，`seg_pad_val=255`（忽略索引），而非检测任务的 114
5. **分割任务归一化参数**: model_config.yaml 中实际存储的是 MMSeg 像素级参数 `[123.675, 116.28, 103.53]` / `[58.395, 57.12, 57.375]`（基于 [0,255] 范围），deploy_config.json 中也是这些值（从 source_config 读取），而非 v1.0 中暗示的 [0,1] 范围 ImageNet 参数
6. **SevSeg 转换不走通用 Pipeline**: SevSeg 有独立的转换流程（直接调用 `sevseg_yolo.export` 和 `sevseg_yolo.tensorrt_deploy`），不经过 model_conversion 的 8 阶段 Pipeline，仅在最后调用 DLHubPackager 打包
7. **OCR config.json 有独立格式**: OCR 的 config.json 是 V2 格式，结构完全不同于 deploy_config.json，包含 `preprocess`/`postprocess`/`inference` 等 OCR 专属字段
8. **分类模型 input_size**: model_factory 中保存到 checkpoint 时做了处理，如果从 timm 获取的是 `(3,224,224)` 元组，会提取 `input_size_hw = input_size[1]` 保存为整数
9. **OCR 检测模型归一化参数**: 实际是 ImageNet 参数 `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`；OCR 识别模型归一化参数是 `[0.5, 0.5, 0.5]` / `[0.5, 0.5, 0.5]`（v1.0 写的基本正确，但缺少细节）

---

## 第一章 总体概述

### 1.1 打包格式总览

**关键事实**: 6 大任务使用**三种不同的打包格式**，并非统一的 .dlhub：

| 任务 | 打包格式 | 配置文件 | 打包工具 |
|------|---------|---------|---------|
| 分类 (cls) | `.dlhub` (ZIP) | `deploy_config.json` + `model_config.yaml` | DLHubPackager |
| 检测 (det) | `.dlhub` (ZIP) | `deploy_config.json` + `model_config.yaml` | DLHubPackager |
| 分割 (seg) | `.dlhub` (ZIP) | `deploy_config.json` + `model_config.yaml` | DLHubPackager |
| 异常检测 (anomaly) | `.pkg` (ZIP) | `config.json` + 多组件文件 | PatchCoreExporter |
| OCR (ocr) | `.pkg` (ZIP，检测和识别各一个) | `config.json` V2 | 自定义 zipfile 打包 |
| 缺陷检测 (sevseg) | `.dlhub` (ZIP) | `deploy_config.json`（SevSeg专用格式） | DLHubPackager |

### 1.2 转换流程总览

**使用通用 model_conversion Pipeline 的任务** (cls/det/seg):

```
训练产物(.pth/.pt)
  → Stage 1: 模型导入 (model_importer)
  → Stage 2: 模型分析 (model_analyzer)
  → Stage 3: 图优化 (model_optimizer, Conv+BN融合)
  → Stage 4: ONNX导出 (model_exporter / 子进程worker)
  → Stage 5: 后端转换 (model_converter → TRT/OV/ORT)
  → Stage 8: 配置生成 (config_generator → model_config.yaml)
  → Stage 9: 统一打包 (dlhub_packager → .dlhub)
```

**使用独立导出流程的任务**:
- **sevseg**: `sevseg_yolo.export` → ONNX → `sevseg_yolo.tensorrt_deploy` → TRT → `_save_deploy_meta()` → DLHubPackager → `.dlhub`
- **anomaly**: PatchCoreExporter → 多组件文件 → `.pkg`
- **ocr**: `export_panel.py` → paddle2onnx → ONNX → TRT/OV → 自定义打包 → `.pkg` ×2

### 1.3 支持的后端与精度

| 后端 | 输出文件 | 支持精度 | 适用任务 |
|------|---------|---------|---------|
| TensorRT | `.engine` | FP32/FP16/INT8/Mixed | cls/det/seg/sevseg/anomaly(仅backbone)/ocr |
| OpenVINO | `.xml` + `.bin` | FP32/FP16/INT8 | cls/det/seg/ocr |
| ONNX Runtime | `.onnx` | FP32/FP16/INT8 | cls/det/seg/sevseg/ocr |

---

## 第二章 .dlhub 打包格式（cls/det/seg/sevseg 使用）

### 2.1 文件本质

`.dlhub` 文件本质是 **ZIP 压缩包**，可直接用 `unzip` 解压。

### 2.2 内部结构

```
<model_name>_<backend>_<precision>.dlhub (ZIP)
├── model/                      # 模型文件目录
│   ├── <model>.engine          # TensorRT 引擎 (仅TRT)
│   ├── <model>.onnx            # ONNX 模型
│   ├── <model>.xml             # OpenVINO IR (仅OV)
│   ├── <model>.bin             # OpenVINO 权重 (仅OV)
│   ├── model_config.yaml       # YAML 部署配置 (Stage 8生成)
│   └── deploy_config.json      # JSON 部署配置 (仅sevseg的_save_deploy_meta)
├── deploy_config.json          # ← 包根目录的统一部署配置（DLHubPackager生成）
├── manifest.json               # 文件清单 + SHA256 校验
└── README.txt                  # 人可读说明
```

**注意**: 包内有两个层级的 `deploy_config.json`：
- 根目录的是 DLHubPackager 的 `_build_deploy_config()` 生成的**统一格式**
- `model/` 内的是各任务自己生成的（如 sevseg 的 `_save_deploy_meta()`），格式可能不同

### 2.3 deploy_config.json 统一格式（DLHubPackager 生成）

```json
{
    "dlhub_version": "1.0",
    "task_type": "det",
    "backend": "tensorrt",
    "model_name": "yolo26m",
    "precision": "fp16",
    "model_files": ["model/yolo26m.engine", "model/yolo26m.onnx"],
    "input": {
        "shape": [1, 3, 640, 640],
        "dtype": "float32",
        "color_format": "RGB",
        "pixel_range": [0, 255],
        "normalize_method": "divide_255",
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
        "letterbox_color": [114, 114, 114]
    },
    "output": {
        "num_classes": 5,
        "class_names": ["defect_A", "defect_B", "..."]
    },
    "dynamic_batch": false,
    "export_info": {
        "exported_at": "2026-03-18T12:00:00",
        "source_framework": "ultralytics"
    }
}
```

### 2.4 deploy_config.json 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dlhub_version` | string | 是 | 固定 `"1.0"` |
| `task_type` | string | 是 | `cls` / `det` / `seg` / `anomaly` / `ocr` / `sevseg` |
| `backend` | string | 是 | `tensorrt` / `openvino` / `ort` |
| `model_name` | string | 是 | 模型名称 |
| `precision` | string | 是 | `fp32` / `fp16` / `int8` / `mixed` |
| `model_files` | list[str] | 是 | 包内模型文件相对路径 |
| `input.shape` | list[int] | 是 | `[B, C, H, W]` |
| `input.dtype` | string | 是 | 固定 `"float32"` |
| `input.color_format` | string | 是 | `"RGB"` |
| `input.pixel_range` | list | 是 | `[0, 255]` |
| `input.normalize_method` | string | 是 | `"divide_255"` 或 `"imagenet"` |
| `input.normalize_mean` | list[float] | 是 | 3元素均值 |
| `input.normalize_std` | list[float] | 是 | 3元素标准差 |
| `input.letterbox_color` | list[int] | 条件 | 仅 det/sevseg 为 `[114,114,114]` |
| `output.num_classes` | int | 否 | 类别数 |
| `output.class_names` | list[str] | 否 | 类别名列表 |
| `dynamic_batch` | bool | 否 | 是否动态 batch |

### 2.5 DLHubPackager 默认预处理值

DLHubPackager 在 `_build_deploy_config()` 中按 task_type 设置默认值（当 source_config 中无对应字段时使用）：

| task_type | normalize_mean | normalize_std | normalize_method | letterbox |
|-----------|---------------|---------------|------------------|-----------|
| `det`, `sevseg` | [0.0, 0.0, 0.0] | [1.0, 1.0, 1.0] | `divide_255` | [114, 114, 114] |
| `cls`, `seg` | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | `imagenet` | 无 |
| 其他 | [0.0, 0.0, 0.0] | [1.0, 1.0, 1.0] | `divide_255` | 无 |

**重要**: 这些是**回退默认值**。实际写入的值优先从 model_config.yaml (source_config) 中读取。例如 seg 任务，如果 model_config.yaml 中已有 MMSeg 像素级参数 `[123.675, ...]`，则 deploy_config.json 中也会是这些像素级参数。

### 2.6 manifest.json 格式

```json
{
    "version": "1.0",
    "created": "2026-03-18T12:00:00.000000",
    "files": {
        "model/yolo26m.engine": {
            "size": 52428800,
            "sha256": "a1b2c3d4e5f6..."
        }
    }
}
```

### 2.7 打包/解包 API

```python
from model_conversion.dlhub_packager import DLHubPackager

packager = DLHubPackager()

# 打包
dlhub_path = packager.pack(
    output_dir="/path/to/converted/tensorrt/20260318_120000",
    task_type="det",          # cls/det/seg/sevseg
    backend="tensorrt",       # 自动检测
    precision="fp16",         # 自动推断
)

# 解包
config = packager.unpack("/path/to/model.dlhub")
```

---

## 第三章 各任务转换详细规格

### 3.1 图像分类 (task_type: "cls")

#### 3.1.1 训练产物 checkpoint 结构

文件路径: `output/<训练名>/checkpoints/best_model.pth`

```python
checkpoint = {
    # ====== 核心字段（model_importer 必需）======
    "model": state_dict,                    # 模型权重 (注意键名是 'model')
    "framework": "timm",                    # 框架标识
    "task": "cls",                          # 任务类型
    "model_name": "efficientnet_b2",        # timm 模型名
    "num_classes": 10,                      # 类别数
    "input_size": 260,                      # 输入尺寸 (整数 H/W)

    # ====== 类别映射（推理必需）======
    "class_to_idx": {"cat": 0, "dog": 1},  # 类别名 → 索引
    "idx_to_class": {0: "cat", 1: "dog"},  # 索引 → 类别名
    "class_names": ["cat", "dog"],          # 类别名列表

    # ====== 预处理配置（推理必需）======
    "normalize_mean": (0.485, 0.456, 0.406),  # 来自 timm data_config
    "normalize_std": (0.229, 0.224, 0.225),   # 来自 timm data_config
    "interpolation": "bilinear",

    # ====== 兼容字段 ======
    "arch": "efficientnet_b2",              # 架构名称（备用）
    "state_dict": state_dict,               # 兼容旧格式

    # ====== 训练状态 ======
    "training_state": {
        "epoch": 50,
        "best_acc": 97.5,
        "optimizer_state_dict": {...},
        "scheduler_state_dict": {...},
    },

    # ====== 模型元数据 ======
    "model_metadata": {
        "model_name": "efficientnet_b2",
        "model_family": "EfficientNet",
        "model_scale": "中",
        "num_classes": 10,
        "params_m": 9.2,
        "input_size": (3, 260, 260),        # 注意：这里是元组格式
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },

    # ====== 训练配置 ======
    "training_config": {
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "AdamW",
        ...
    },

    # ====== 保存信息 ======
    "save_info": {
        "is_best": true,
        "save_time": "2026-03-18T12:00:00",
        "pytorch_version": "2.1.0",
    },
}
```

#### 3.1.2 转换流程

使用通用 model_conversion Pipeline:
```
best_model.pth
  → Stage 1: run_stage1_import(ctx, model_path, 'cls')
  → Stage 2: run_stage2_analyze(ctx, [target_backend])
  → Stage 3: run_stage3_optimize(ctx)
  → Stage 4: run_stage4_export(ctx, opset=17, enable_simplify=False)
  → Stage 5: run_stage5_convert(ctx, target_backend, precision)
  → Stage 8: run_stage8_generate_config(ctx, output_dir)
  → Stage 9: DLHubPackager().pack(task_type='cls')
```

#### 3.1.3 预处理规格

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 模型相关(224/240/260/300/380) | |
| 颜色空间 | RGB | |
| 先 /255 | 是 | 归一化到 [0, 1] |
| normalize_mean | [0.485, 0.456, 0.406] | ImageNet |
| normalize_std | [0.229, 0.224, 0.225] | ImageNet |
| normalize_method | `"imagenet"` | `(pixel/255 - mean) / std` |
| resize | 直接 resize (bilinear) | **无 letterbox** |

#### 3.1.4 输出

| 输出节点 | 形状 | 后处理 |
|---------|------|--------|
| `output0` | `[B, num_classes]` | `softmax → argmax → class_names[idx]` |

---

### 3.2 目标检测 (task_type: "det")

#### 3.2.1 训练产物 checkpoint 结构

文件路径: `output/<训练名>/weights/best.pt`

```python
checkpoint = {
    "model": nn.Module,                     # 完整模型对象
    "ema": nn.Module,                       # EMA模型
    "framework": "ultralytics",             # 框架标识
    "_original_model": nn.Module,           # 原始模型对象引用

    # 类别信息
    "nc": 5,                                # 类别数
    "names": {0: "cat", 1: "dog", ...},     # 类别名字典

    # YAML配置
    "yaml": {
        "nc": 5,
        "yaml_file": "yolo26m.yaml",
    },

    # 训练参数
    "train_args": {
        "nc": 5,
        "imgsz": 640,
        "model": "yolo26m.pt",
        "epochs": 100,
        "batch": 32,
        "lr0": 0.01,
        "optimizer": "SGD",
    },

    # 预处理元数据
    "custom_metadata": {
        "model_name": "yolo26m",
        "num_classes": 5,
        "class_names": ["cat", "dog", ...],
        "input_size": 640,
        "input_spec": {
            "shape": (1, 3, 640, 640),
            "color_format": "RGB",
            "pixel_range": (0, 255),
            "normalize_method": "divide_255",
            "normalize_mean": (0.0, 0.0, 0.0),
            "normalize_std": (1.0, 1.0, 1.0),
            "value_range": (0.0, 1.0),
            "letterbox_color": (114, 114, 114),
        },
    },
}
```

#### 3.2.2 转换流程

使用通用 Pipeline，但 **ONNX 导出使用独立子进程**:
```
best.pt
  → Stage 1: run_stage1_import(ctx, model_path, 'det')
  → Stage 2-3: 分析+优化
  → Stage 4: 独立子进程 yolo_export_worker.py 导出 ONNX
  → Stage 5: run_stage5_convert(ctx, ...)
  → Stage 8: run_stage8_generate_config(ctx, output_dir)
  → Stage 9: DLHubPackager().pack(task_type='det')
```

#### 3.2.3 预处理规格

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 640×640 |
| 颜色空间 | RGB |
| 先 /255 | 是 |
| normalize_mean | [0.0, 0.0, 0.0] |
| normalize_std | [1.0, 1.0, 1.0] |
| normalize_method | `"divide_255"` |
| resize | **letterbox** (保持宽高比, 填充色 `[114,114,114]`, stride=32) |

#### 3.2.4 输出

| 输出节点 | 形状 | 说明 |
|---------|------|------|
| `output0` | `[B, K, 6]` 或 `[B, K, 4+nc+1]` | 格式视模型版本而异 |

后处理: 置信度过滤(0.25) → NMS(IoU=0.45) → letterbox坐标还原

**注意**: YOLO26 是 NMS-free，输出已是最终结果，**不需要额外 NMS**。

---

### 3.3 语义分割 (task_type: "seg")

#### 3.3.1 训练产物 checkpoint 结构

文件路径: `output/<训练名>/weights/<model>_best.pth`

```python
checkpoint = {
    "framework": "mmsegmentation",
    "model_type": "segformer",
    "model_name": "segformer_b2",

    # 模型权重
    "state_dict": OrderedDict,

    # MMSegmentation 需要的 meta 字段
    "meta": {
        "dataset_meta": {
            "classes": ("background", "defect_A", "defect_B"),
            "palette": [[0,0,0], [255,0,0], [0,255,0]],
        },
        "CLASSES": ("background", "defect_A", "defect_B"),
        "PALETTE": [[0,0,0], [255,0,0], [0,255,0]],
        "epoch": 80,
    },

    # 类别信息
    "nc": 3,
    "num_classes": 3,
    "names": {0: "background", 1: "defect_A", 2: "defect_B"},
    "class_names": ["background", "defect_A", "defect_B"],

    # 预处理元数据
    "model_metadata": {
        "model_name": "segformer_b2",
        "num_classes": 3,
        "class_names": ["background", "defect_A", "defect_B"],
        "input_size": 512,
        "input_spec": {
            "shape": (1, 3, 512, 512),
            "color_format": "RGB",
            "pixel_range": (0, 255),
            "normalize_method": "imagenet",
            "normalize_mean": [123.675, 116.28, 103.53],   # ← MMSeg 像素级
            "normalize_std": [58.395, 57.12, 57.375],       # ← MMSeg 像素级
            "value_range": (-2.5, 2.5),
        },
        "ignore_index": 255,
        "task": "semantic_segmentation",
    },
}
```

#### 3.3.2 转换流程

使用通用 Pipeline，**ONNX 导出使用独立子进程 segformer_export_worker.py**:
```
best_model.pth
  → Stage 1: run_stage1_import(ctx, model_path, 'seg')
  → Stage 2-3: 分析+优化
  → Stage 4: 独立子进程 segformer_export_worker.py 导出 ONNX
      (回退: 进程内 run_stage4_export)
  → Stage 5: run_stage5_convert(ctx, ...)
  → Stage 8: run_stage8_generate_config(ctx, output_dir)
  → Stage 9: DLHubPackager().pack(task_type='seg')
```

#### 3.3.3 预处理规格 (⚠️ 与检测任务不同)

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 512 (默认，可选 512/640/768/1024) | |
| 颜色空间 | RGB | |
| **先 /255** | **否** | ⚠️ 直接在 [0,255] 上归一化 |
| normalize_mean | **[123.675, 116.28, 103.53]** | MMSeg 像素级 |
| normalize_std | **[58.395, 57.12, 57.375]** | MMSeg 像素级 |
| normalize_method | `"imagenet"` | 公式: `(pixel - mean) / std` |
| value_range | **[0, 255]** → 归一化后约 [-2.5, 2.5] | |
| resize | letterbox (保持宽高比) | |
| **letterbox 填充色** | **[0, 0, 0]** | ⚠️ **不是 114**, seg_pad_val=255(忽略索引) |
| letterbox stride | 32 | |

**关键区别**: 分割任务的归一化公式是 `(pixel_value - 123.675) / 58.395`，输入是原始像素值 [0,255]，**不需要先除以 255**。这与分类/检测任务完全不同。

**但注意**: model_config.yaml 中 `value_range` 由 model_importer 设为 `(0.0, 255.0)`。deploy_config.json 中 `pixel_range` 和 `normalize_mean/std` 也会反映这一点。部署软件应根据 `normalize_method` 和实际 mean/std 值范围来判断是否需要先 /255。

#### 3.3.4 输出

| 输出节点 | 形状 | 后处理 |
|---------|------|--------|
| `output0` | `[B, num_classes, H, W]` | `argmax(dim=1)` → resize回原图 → 去除letterbox填充 |

---

### 3.4 异常检测 (task_type: "anomaly")

#### 3.4.1 ⚠️ 不使用 .dlhub 格式

PatchCore 使用独立的 `.pkg` 打包格式。

#### 3.4.2 训练产物结构

```
output/<训练名>/exports/
├── patchcore_<YYYYMMDD_HHMMSS>.pkg   # ZIP打包 (推荐)
└── patchcore_model/                    # 或未打包目录
    ├── config.json                     # 完整配置 (核心)
    ├── manifest.json                   # 文件清单 (MD5校验)
    ├── backbone/
    │   ├── backbone.onnx               # Backbone ONNX (opset=14)
    │   └── backbone_<precision>.engine # Backbone TRT (可选)
    ├── memory_bank/
    │   ├── features.npy                # Memory Bank [N, D] float16
    │   ├── faiss_index.bin             # Faiss 索引
    │   └── pca_model.npz              # PCA: components(float16) + mean(float32)
    ├── normalization/
    │   └── params.json                 # 评分归一化参数
    ├── threshold/
    │   └── config.json                 # 阈值配置
    └── metadata/
        └── info.json                   # 训练元数据
```

#### 3.4.3 config.json 完整结构

```json
{
    "model_info": {
        "name": "PatchCore",
        "version": "1.0.0"
    },
    "backbone": {
        "name": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "feature_dim": 1536,
        "pretrained": "imagenet"
    },
    "preprocessing": {
        "input_size": [256, 256],
        "resize_mode": "bilinear",
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "input_format": "RGB",
        "input_dtype": "float32",
        "input_range": [0.0, 1.0]
    },
    "feature_extraction": {
        "pca": {"enabled": true, "n_components": 256}
    },
    "memory_bank": {
        "sampling_method": "coreset",
        "sampling_ratio": 0.01,
        "size": 5000,
        "feature_dtype": "float16"
    },
    "knn": {"k": 9, "index_type": "IVFFlat", "metric": "L2"},
    "postprocessing": {
        "upsample_mode": "bilinear",
        "gaussian_blur": {"enabled": true, "sigma": 4.0},
        "score_aggregation": "max"
    },
    "normalization": {...},
    "thresholds": {...}
}
```

#### 3.4.4 推理流程

```
原图 → resize(bilinear) → /255 → ImageNet归一化
  → Backbone ONNX/TRT推理 → 多层特征
  → PCA降维(可选) → KNN近邻搜索(Faiss)
  → L2距离评分 → 评分归一化 → 高斯模糊 → 阈值判定
输出: {score(0-100), is_anomaly, anomaly_map[H,W], binary_mask[H,W]}
```

#### 3.4.5 加载 API

```python
from model_image_patchcore.export.exporter import load_patchcore_model

components = load_patchcore_model("patchcore_20260318.pkg")
# {config, features, faiss_index, pca_model, normalization, thresholds, model_dir}

from model_image_patchcore.inference.predictor import PatchCorePredictor
predictor = PatchCorePredictor.from_package("model.pkg")
result = predictor.predict(image)
```

---

### 3.5 OCR识别 (task_type: "ocr")

#### 3.5.1 ⚠️ 不使用 .dlhub 格式

OCR 使用独立的 `.pkg` 格式，且检测模型和识别模型**分别打包**为两个独立的 `.pkg` 文件。

#### 3.5.2 导出产物

```
output/
├── det_<model_name>_<timestamp>.pkg     # 检测模型包 (ZIP)
│   ├── model.onnx                       # 检测 ONNX
│   ├── model.engine                     # 检测 TRT (可选)
│   ├── model.xml + model.bin            # 检测 OpenVINO (可选)
│   └── config.json                      # 检测配置 V2
│
└── rec_<model_name>_<timestamp>.pkg     # 识别模型包 (ZIP)
    ├── model.onnx                       # 识别 ONNX
    ├── model.engine                     # 识别 TRT (可选)
    ├── model.xml + model.bin            # 识别 OpenVINO (可选)
    ├── ppocr_keys_v1.txt                # 字符字典 (必需)
    └── config.json                      # 识别配置 V2
```

#### 3.5.3 检测模型 config.json V2

```json
{
    "config_version": "2.0",
    "model_info": {
        "name": "PP-OCRv4_server_det",
        "task": "ocr_detection",
        "algorithm": "DB",
        "framework": "PaddleOCR"
    },
    "files": {"onnx": "model.onnx", "tensorrt": "model.engine"},
    "input": {
        "name": "x",
        "shape": [-1, 3, 640, 640],
        "dtype": "float32",
        "layout": "NCHW"
    },
    "output": {
        "name": "sigmoid_0.tmp_0",
        "shape": [-1, 1, 640, 640],
        "dtype": "float32"
    },
    "inference": {"max_batch_size": 8, "fp16": true},
    "preprocess": {
        "resize": {
            "target_height": 640, "target_width": 640,
            "keep_ratio": true, "padding": true,
            "pad_value": 0, "interp": "LINEAR"
        },
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "to_rgb": true
        }
    },
    "postprocess": {
        "algorithm": "DB",
        "thresh": 0.3,
        "box_thresh": 0.6,
        "unclip_ratio": 1.5,
        "max_candidates": 1000,
        "use_dilation": false,
        "score_mode": "fast"
    }
}
```

#### 3.5.4 识别模型 config.json V2

```json
{
    "config_version": "2.0",
    "model_info": {
        "name": "PP-OCRv4_server_rec",
        "task": "ocr_recognition",
        "algorithm": "CTC",
        "framework": "PaddleOCR"
    },
    "files": {"onnx": "model.onnx"},
    "character_dict": "ppocr_keys_v1.txt",
    "input": {
        "name": "x",
        "shape": [-1, 3, 48, 320],
        "dtype": "float32",
        "layout": "NCHW"
    },
    "output": {
        "name": "softmax_0.tmp_0",
        "shape": [-1, -1, "num_classes"],
        "dtype": "float32"
    },
    "inference": {"max_batch_size": 32, "fp16": true},
    "preprocess": {
        "resize": {
            "target_height": 48, "target_width": 320,
            "keep_ratio": true, "padding": "right",
            "pad_value": 0, "interp": "LINEAR"
        },
        "normalize": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "to_rgb": true
        }
    },
    "postprocess": {
        "algorithm": "CTC",
        "use_space_char": true,
        "score_thresh": 0.5,
        "remove_duplicates": true
    }
}
```

#### 3.5.5 完整推理流程

```
原图
  → det 模型推理:
      resize(keep_ratio,pad) → /255 → ImageNet归一化 → 模型推理
      → DB后处理(阈值+轮廓) → 文字区域四点坐标列表
  → 逐区域裁剪+透视变换
  → rec 模型推理(批量):
      resize(keep_ratio,right_pad) → /255 → [0.5,0.5,0.5]归一化 → 模型推理
      → CTC解码(使用 ppocr_keys_v1.txt) → 文字字符串
输出: [{box: 四点坐标, text: 文字, score: 置信度}, ...]
```

---

### 3.6 工业缺陷检测 (task_type: "sevseg")

#### 3.6.1 ⚠️ 不走通用 Pipeline，有独立导出流程

SevSeg-YOLO 的转换流程:
```
best.pt
  → 加载: YOLO(model_path, task='score_detect')
  → ONNX导出: sevseg_yolo.export.export_scoreyolo_onnx()
      ├── 可选嵌入 PCA 1×1 Conv (用于 Mask Generator)
      └── 输出节点: "det_output" + ["feat_p3","feat_p4","feat_p5"]
  → (TRT路径) sevseg_yolo.tensorrt_deploy.build_trt_engine()
  → _save_deploy_meta() → deploy_config.json (SevSeg专用格式)
  → DLHubPackager().pack(task_type='sevseg') → .dlhub
```

#### 3.6.2 SevSeg 专用 deploy_config.json

```json
{
    "model": {
        "task": "score_detect",
        "framework": "SevSeg-YOLO",
        "weights": "/path/to/best.pt",
        "exported": "/path/to/best.engine",
        "format": "TensorRT",
        "fp16": true
    },
    "classes": {
        "nc": 5,
        "names": {"0": "defect_A", "1": "defect_B", ...}
    },
    "preprocessing": {
        "input_size": [640, 640],
        "input_format": "BCHW",
        "color_space": "RGB",
        "normalize": "divide_255",
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
        "resize": "letterbox",
        "pad_value": 114
    },
    "output": {
        "det_columns": "x1,y1,x2,y2,conf,cls_id,severity",
        "severity_range": [0, 10]
    },
    "mask_generator": {
        "method": "MaskGeneratorV2",
        "upsample": "canny_edge_guided_blend"
    },
    "exported_at": "2026-03-18T12:00:00"
}
```

#### 3.6.3 ONNX 输出节点

| 输出节点 | 形状 | 说明 |
|---------|------|------|
| `det_output` | `[B, K, 7]` | `[x1,y1,x2,y2,conf,cls_id,severity_01]` |
| `feat_p3` (可选) | `[B, pca_dim, H/8, W/8]` | PCA压缩P3特征 |
| `feat_p4` (可选) | `[B, pca_dim, H/16, W/16]` | PCA压缩P4特征 |
| `feat_p5` (可选) | `[B, pca_dim, H/32, W/32]` | PCA压缩P5特征 |

#### 3.6.4 ⚠️ severity 后处理（v1.0勘误）

模型原始输出的 severity（第7列）范围是 **[0, 1]**，后处理必须 **× 10** 才得到 [0, 10] 的最终严重程度分数：

```python
severity_01 = det_output[:, 6]   # 模型原始输出 [0, 1]
severity_10 = severity_01 * 10.0  # 最终分数 [0, 10]
```

---

## 第四章 6 大任务预处理对照表（校订版）

| 项目 | cls | det | seg | anomaly | ocr-det | ocr-rec | sevseg |
|------|-----|-----|-----|---------|---------|---------|--------|
| 输入尺寸 | 模型相关 | 640 | 512 | 256 | 配置决定 | 48×W(配置) | 640 |
| 颜色空间 | RGB | RGB | RGB | RGB | RGB | RGB | RGB |
| 先 /255 | **是** | **是** | **否** | **是** | **是** | **是** | **是** |
| mean | [.485,.456,.406] | [0,0,0] | **[123.7,116.3,103.5]** | [.485,.456,.406] | [.485,.456,.406] | [.5,.5,.5] | [0,0,0] |
| std | [.229,.224,.225] | [1,1,1] | **[58.4,57.1,57.4]** | [.229,.224,.225] | [.229,.224,.225] | [.5,.5,.5] | [1,1,1] |
| letterbox | 否 | **是(114)** | **是(0)** | 否 | 是(0) | 否(右填充) | **是(114)** |
| stride | - | 32 | 32 | - | - | - | 32 |
| 打包格式 | .dlhub | .dlhub | .dlhub | **.pkg** | **.pkg×2** | **.pkg×2** | .dlhub |

---

## 第五章 6 大任务输出对照表（校订版）

| 任务 | 输出形状 | 含义 | 关键后处理 |
|------|---------|------|-----------|
| cls | `[B, C]` | 类别logits | softmax → argmax |
| det | `[B, K, 6]` | x1,y1,x2,y2,conf,cls | NMS(v5/v8/v11) 或直接使用(v26) |
| seg | `[B, C, H, W]` | 像素级logits | argmax(dim=1) → resize回原图 |
| anomaly | Backbone多层特征 | 中间层特征图 | PCA→KNN→评分→模糊→阈值 |
| ocr-det | `[B, 1, H, W]` | 文字区域概率图 | DB二值化→轮廓→四点坐标 |
| ocr-rec | `[B, T, V]` | 字符概率序列 | CTC解码(需字典文件) |
| sevseg | `[B, K, 7]` | x1,y1,x2,y2,conf,cls,**sev_01** | NMS + **severity×10** + MaskGen |

---

## 附录 A: task_type 规范化映射

```python
TASK_TYPE_MAP = {
    'classification': 'cls', 'cls': 'cls',
    'detection': 'det', 'det': 'det',
    'segmentation': 'seg', 'seg': 'seg',
    'anomaly': 'anomaly', 'patchcore': 'anomaly',
    'ocr': 'ocr',
    'sevseg': 'sevseg', 'score_detect': 'sevseg',
}
```

## 附录 B: 后端文件扩展名

```python
BACKEND_FILE_PATTERNS = {
    'tensorrt': ['*.engine', '*.trt'],
    'openvino': ['*.xml', '*.bin'],
    'ort': ['*.onnx'], 'onnxruntime': ['*.onnx'],
}
```

## 附录 C: 各任务源框架标识

| task_type | framework | 导出方式 |
|-----------|-----------|---------|
| cls | `timm` | 通用Pipeline Stage 4 (进程内) |
| det | `ultralytics` | 独立子进程 yolo_export_worker.py |
| seg | `mmsegmentation` | 独立子进程 segformer_export_worker.py |
| anomaly | `torch` (自定义) | PatchCoreExporter |
| ocr | `paddlepaddle` | paddle2onnx (export_panel.py) |
| sevseg | `SevSeg-YOLO` | sevseg_yolo.export.export_scoreyolo_onnx |

## 附录 D: 输出目录命名规则

**cls/det/seg** (通用Pipeline):
```
<任务output>/converted/<后端名>/<YYYYMMDD_HHMMSS>/
```
后端名映射: `tensorrt` / `openvino` / `onnxruntime`

**sevseg**: 输出在模型文件同级目录（如 `weights/` 下）

**anomaly**: `output/<训练名>/exports/`

**ocr**: `output/` (可配置)
