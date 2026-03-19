<div align="center">

# 🧠 DL-Hub

### 深度学习任务管理平台 · Deep Learning Task Management Platform

**训练 · 转换 · 部署 —— 六大视觉AI任务，一个平台全搞定**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange?logo=gradio)](https://gradio.app)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Code Lines](https://img.shields.io/badge/Code-134K%2B_lines-brightgreen)]()

<br/>

[English](#overview) · [快速开始](#-快速开始) · [功能总览](#-六大任务一览) · [架构设计](#-平台架构) · [模型转换](#-模型转换工具链) · [部署导出](#-部署导出格式) · [许可证](#-许可证)

</div>

---

## Overview

DL-Hub is a full-stack deep learning platform that manages **six production-ready computer vision tasks** through a unified web interface. Each task provides a complete pipeline — from data preparation and model training to format conversion and deployment packaging — with no command-line required.

> **为什么选择 DL-Hub?**
> - 🎯 **零门槛**: 全图形化界面，无需写一行代码即可完成模型训练到部署
> - 🔄 **端到端**: 数据标注验证 → 模型训练 → 格式转换 → 量化优化 → 部署打包
> - 🏭 **工业级**: 支持 TensorRT/OpenVINO/ONNX Runtime 三大部署后端，FP16/INT8 量化
> - 📊 **实时监控**: WebSocket 实时推送训练日志，Loss/mAP/mIoU 曲线动态刷新
> - 💾 **参数记忆**: 所有 UI 参数、训练历史、日志自动持久化，关闭重开状态不丢失

---

## 🎯 六大任务一览

<table>
<tr>
<td width="33%" valign="top">

### 🎯 图像分类
**基于 timm 的迁移学习训练**

- 15 个预训练模型 (3 大系列)
- EfficientNet B0–B4
- MobileNetV3 Small/Large
- ResNet18/34/50/WideResNet
- CutMix / MixUp 数据增强
- AMP 混合精度训练
- 学习率预热 + 余弦退火

</td>
<td width="33%" valign="top">

### 🔍 目标检测
**YOLO 系列一站式训练**

- 20 个预训练模型 (4 大系列)
- YOLOv5 n/s/m/l/x
- YOLOv8 n/s/m/l/x
- YOLOv11 n/s/m/l/x
- **YOLO26 n/s/m/l/x** (NMS-free)
- LabelMe → YOLO 自动转换
- 实时检测预览

</td>
<td width="33%" valign="top">

### 🎨 语义分割
**SegFormer 像素级分割**

- 6 个模型 (B0–B5)
- Mix Transformer 骨干网络
- MMSegmentation 引擎
- 滑动窗口大图推理
- 多类别调色板可视化
- LabelMe → MMSeg 格式转换

</td>
</tr>
<tr>
<td width="33%" valign="top">

### 🔬 异常检测
**PatchCore 无监督检测**

- **仅需良品样本**，无需缺陷标注
- WideResNet / ResNet / EfficientNet
- Coreset 采样 + Faiss 加速 KNN
- PCA 特征压缩
- 像素级异常热力图
- 自动最优阈值搜索 (F1)

</td>
<td width="33%" valign="top">

### 📝 OCR 识别
**PaddleOCR 文字检测识别**

- PP-OCRv4 / v5 Server/Mobile
- 文字检测 (DB 算法)
- 文字识别 (CTC 解码)
- 中英文 + 多语言支持
- 批量 OCR 处理
- ONNX / TensorRT / OpenVINO 导出

</td>
<td width="33%" valign="top">

### 🏭 工业缺陷检测
**SevSeg-YOLO (自研)**

- **三合一**: 检测 + 评分 + 分割
- YOLO26-Score 架构 (5 个规模)
- 严重程度评分 [0–10]
- **零标注近似分割** (MaskGenerator)
- 3 代掩膜生成器 (V1/V2/V3)
- Gaussian NLL 评分损失

</td>
</tr>
</table>

---

## ✨ 平台核心功能

### 🖥️ 统一管理界面

DL-Hub 提供 React 前端 + FastAPI 后端的现代化 Web 管理界面:

- **任务卡片仪表板** — 六大任务一目了然，点击即进入训练界面
- **工作空间管理** — 集中管理所有任务目录、模型文件、训练产物
- **任务生命周期** — 创建 → 配置 → 训练 → 转换 → 导出，全程可视化
- **多任务并行** — 同时管理多个训练任务，互不干扰
- **深色/浅色主题** — 自适应系统主题，支持手动切换

### 👤 用户系统

- 用户注册/登录 (SHA-256 密码哈希 + Salt)
- HttpOnly Cookie 会话管理
- 服务器重启自动失效旧会话
- 用户头像上传 (Base64 本地存储)
- 多用户支持

### 📡 实时监控

- **WebSocket 日志推送** — 训练日志实时显示，无需刷新
- **训练曲线** — Loss / Accuracy / mAP / mIoU 动态图表
- **系统资源** — GPU 显存、CPU、磁盘使用率实时监控
- **Conda 环境检测** — 自动扫描系统中已安装的 Python 环境

### 💾 参数持久化

DL-Hub 的独特设计 — `DLHubParams` 单例管理器:

```python
# 所有 UI 参数自动保存，下次打开完全恢复
params.save({'model': 'resnet50', 'epochs': 100, 'lr': 0.001})

# 训练历史也持久化（用于恢复 Loss 曲线图）
params.save_history({'train_losses': [...], 'val_accs': [...]})

# 训练日志同样持久化
params.save_logs(['Epoch 1/100...', 'Loss: 0.234...'])
```

每个任务的 Gradio 应用同时支持**独立运行**和 **DL-Hub 托管**两种模式。

---

## 🏗️ 平台架构

```
┌─────────────────────────────────────────────────────────────┐
│                  DL-Hub React Frontend                       │
│              http://localhost:7860                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ 任务卡片  │ │ 工作空间  │ │ 用户系统  │ │  系统监控     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
└─────────────────────┬──────────────────┬────────────────────┘
                      │ REST API         │ WebSocket
┌─────────────────────▼──────────────────▼────────────────────┐
│                FastAPI Backend                                │
│  /api/auth · /api/workspace · /api/tasks · /api/app          │
│  /api/system · /ws/logs/{task_id}                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ 子进程启动
┌─────────────────────▼───────────────────────────────────────┐
│          6 大任务 Gradio 应用 (http://localhost:7861)         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────┐ ┌─────┐ ┌──────┐       │
│  │ 分类 │ │ 检测 │ │ 分割 │ │ 异常  │ │ OCR │ │ 缺陷  │       │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬───┘ └──┬──┘ └──┬───┘       │
│     │      │      │      │       │      │              │
│  ┌──▼──────▼──────▼──────▼───────▼──────▼──┐           │
│  │     DLHubAdapter + DLHubParams          │           │
│  │     (统一适配器 + 参数持久化单例)          │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            Model Conversion Pipeline (8 阶段)                │
│  Import → Analyze → Optimize → ONNX → TRT/OV/ORT → Config  │
│                                  → .dlhub / .pkg 打包        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 推荐 3.10+ |
| CUDA | 11.8+ | GPU 训练必需 |
| Node.js | 16+ | 前端构建（可选，已含预编译版本） |
| 显存 | 4GB+ | 视模型大小而定 |

### 安装

```bash
# 克隆项目
git clone https://github.com/YOUR_ORG/DL-Hub.git
cd DL-Hub

# 安装平台基础依赖
pip install -r requirements.txt

# 按需安装任务依赖 ——————————————————————————————
# 图像分类
pip install torch torchvision timm

# 目标检测 / 工业缺陷检测
pip install ultralytics

# 语义分割
pip install -U openmim && mim install mmengine 'mmcv>=2.0.0' mmsegmentation

# 异常检测
pip install scikit-learn faiss-cpu   # GPU版: faiss-gpu

# OCR
pip install paddlepaddle-gpu paddleocr paddle2onnx

# 模型转换（可选）
pip install onnx onnxruntime-gpu     # TensorRT 需单独安装
```

### 启动平台

```bash
python run_dlhub.py
# ✅ 自动打开浏览器 → http://localhost:7860
# ✅ 首次启动自动构建前端（需要 Node.js）
# ✅ 首次使用需要注册账号
```

### 独立运行单个任务

每个任务都可以脱离 DL-Hub 独立使用:

```bash
cd model_image_classification && python app.py   # 分类 → http://localhost:7861
cd model_image_detection && python app.py         # 检测
cd model_image_segmentation && python app.py      # 分割
cd model_image_patchcore && python app.py         # 异常检测
cd model_image_ocr && python app.py               # OCR
cd model_image_sevseg && python app.py            # 缺陷检测
```

### 启动参数

```bash
python run_dlhub.py [选项]
  --port PORT        # 服务端口 (默认 7860)
  --host HOST        # 服务地址 (默认 0.0.0.0)
  --no-browser       # 不自动打开浏览器
  --rebuild          # 强制重新构建前端
  --dev              # 开发模式 (启用热重载)
```

---

## 🔧 模型转换工具链

DL-Hub 内置 **8 阶段工业级模型转换 Pipeline**，将训练产物转换为生产部署格式:

```
PyTorch 模型 (.pth/.pt)
  │
  ├── Stage 1: 模型导入 ── 自动识别 timm / Ultralytics / MMSeg 框架
  ├── Stage 2: 模型分析 ── 兼容性检测、性能分析、转换建议
  ├── Stage 3: 图优化 ─── Conv+BN 融合、冗余层移除
  ├── Stage 4: ONNX 导出 ─ 动态轴、模型简化、精度验证
  ├── Stage 5: 量化转换 ── FP16 / INT8 / Mixed 精度
  ├── Stage 6: 精度验证 ── PyTorch vs ONNX vs 目标后端对比
  ├── Stage 8: 配置生成 ── model_config.yaml (预处理参数、类别信息)
  └── Stage 9: 打包 ───── .dlhub 统一部署包
```

### 支持的部署后端

| 后端 | 输出格式 | 精度 | 适用场景 |
|------|---------|------|---------|
| **TensorRT** | `.engine` | FP32 / FP16 / INT8 / Mixed | NVIDIA GPU 高性能推理 |
| **OpenVINO** | `.xml` + `.bin` | FP32 / FP16 / INT8 | Intel CPU / GPU / VPU |
| **ONNX Runtime** | `.onnx` | FP32 / FP16 / INT8 | 跨平台通用部署 |

### 使用方式

```bash
# 方式 1: GUI — 每个任务 Gradio 界面的"模型转换"标签页

# 方式 2: YAML 配置文件
cd model_conversion
python main.py init -t det              # 生成检测任务配置模板
vim config_det.yaml                     # 编辑配置
python main.py run -c config_det.yaml   # 运行转换

# 方式 3: 命令行
python main.py pipeline -m best.pt -t det -o ./output --target tensorrt --precision fp16
```

---

## 📦 部署导出格式

### .dlhub 格式 (分类 / 检测 / 分割 / 缺陷检测)

`.dlhub` 本质是 ZIP 压缩包，包含模型文件 + 统一部署配置:

```
model_tensorrt_fp16.dlhub (ZIP)
├── model/
│   ├── model.engine          # TensorRT 引擎
│   ├── model.onnx            # ONNX (备用)
│   └── model_config.yaml     # 详细部署配置
├── deploy_config.json        # 统一部署配置
├── manifest.json             # SHA256 校验清单
└── README.txt
```

### .pkg 格式 (异常检测)

PatchCore 多组件打包:

```
patchcore.pkg (ZIP)
├── config.json, manifest.json
├── backbone/backbone.onnx
├── memory_bank/features.npy + faiss_index.bin
├── normalization/params.json
└── threshold/config.json
```

### .pkg 格式 (OCR)

检测模型和识别模型分别打包:

```
det_PP-OCRv4.pkg    # 检测: model.onnx + config.json
rec_PP-OCRv4.pkg    # 识别: model.onnx + config.json + ppocr_keys_v1.txt
```

---

## 📁 项目结构

```
DL-Hub/
├── run_dlhub.py                       # 🚀 平台启动入口
├── dlhub_params.py                    # 💾 参数持久化管理器 (单例)
├── dlhub_user_widget.py               # 👤 用户头像组件
│
├── dlhub_project/                     # 🖥️ 平台核心
│   ├── dlhub/backend/                 #    FastAPI 后端
│   │   ├── main.py                    #    应用入口 + WebSocket + 认证中间件
│   │   ├── routers/                   #    API 路由 (auth/workspace/tasks/app/system)
│   │   └── services/                  #    业务服务 (任务/进程/状态监控/认证)
│   ├── dlhub/frontend/                #    React 前端
│   │   ├── src/App.jsx                #    主应用 (单文件, ~1400行)
│   │   └── dist/                      #    预编译产物 (可直接使用)
│   ├── dlhub/app_adapters/            #    任务适配器接口
│   ├── user_manual/                   #    📖 6 篇 HTML 使用手册
│   └── docs/                          #    📝 集成指南 / 任务结构 / 测试报告
│
├── model_image_classification/        # 🎯 图像分类 (timm)
├── model_image_detection/             # 🔍 目标检测 (YOLO)
├── model_image_segmentation/          # 🎨 语义分割 (SegFormer)
├── model_image_patchcore/             # 🔬 异常检测 (PatchCore)
├── model_image_ocr/                   # 📝 OCR (PaddleOCR)
├── model_image_sevseg/                # 🏭 工业缺陷检测 (SevSeg-YOLO)
│   ├── sevseg_yolo/                   #    核心: 模型/导出/MaskGenerator
│   └── ultralytics/                   #    修改版 Ultralytics (⚠️ AGPL-3.0)
│
├── model_conversion/                  # 🔧 8 阶段模型转换 Pipeline
│   ├── model_importer.py              #    Stage 1: 模型导入
│   ├── model_analyzer.py              #    Stage 2: 模型分析
│   ├── model_optimizer.py             #    Stage 3: 图优化
│   ├── model_exporter.py              #    Stage 4: ONNX 导出
│   ├── model_converter.py             #    Stage 5: 量化转换
│   ├── config_generator.py            #    Stage 8: 配置生成
│   └── dlhub_packager.py              #    Stage 9: .dlhub 打包
│
└── pretrained_model/                  # 📥 预训练权重 (需自行下载)
```

### 每个任务模块统一结构

```
model_image_<task>/
├── app.py              # Gradio Web 应用入口
├── config/             # 模型注册表 + 默认配置
│   └── model_registry.py  # 所有预训练模型信息
├── models/             # 模型工厂 (创建/加载模型)
├── engine/             # 训练引擎 (统一回调接口)
├── data/               # 数据集 + 格式转换器
├── utils/              # 环境检查 + 数据验证
└── README.md           # 模块文档
```

---

## 📖 文档体系

| 文档 | 路径 | 说明 |
|------|------|------|
| **使用手册** (6 篇) | `dlhub_project/user_manual/*.html` | 每个任务的详细图文操作指南 |
| **任务结构规范** | `dlhub_project/docs/TASK_FOLDER_STRUCTURE.md` | 任务目录约定 |
| **参数集成指南** | `dlhub_project/docs/PARAMS_INTEGRATION_GUIDE.md` | DLHubParams 集成方法 |
| **应用适配器指南** | `dlhub_project/dlhub/app_adapters/INTEGRATION_GUIDE.md` | 添加新任务类型的步骤 |
| **测试分析报告** | `dlhub_project/docs/TEST_ANALYSIS_REPORT.md` | 测试结果与已知问题 |

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解:

- 开发环境搭建
- 代码规范与提交约定
- 如何添加新的任务类型
- Pull Request 流程

---

## ⚖️ 许可证

本项目基于 **Apache License 2.0** 许可 — 详见 [LICENSE](LICENSE)。

> ⚠️ **例外**: `model_image_sevseg/ultralytics/` 目录包含修改的 [Ultralytics](https://github.com/ultralytics/ultralytics) 代码，使用 **AGPL-3.0** 许可。如果您使用或分发 SevSeg 模块，必须遵守该组件的 AGPL-3.0 条款。详见 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)。

---

## 🙏 致谢

DL-Hub 站在巨人的肩膀上:

| 项目 | 用途 | 许可证 |
|------|------|--------|
| [timm](https://github.com/huggingface/pytorch-image-models) | 图像分类模型库 | Apache-2.0 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO 检测框架 | AGPL-3.0 |
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) | 语义分割框架 | Apache-2.0 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | OCR 工具集 | Apache-2.0 |
| [Gradio](https://github.com/gradio-app/gradio) | Web UI 框架 | Apache-2.0 |
| [FastAPI](https://github.com/tiangolo/fastapi) | 后端 API 框架 | MIT |
| [React](https://react.dev) | 前端框架 | MIT |

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！⭐**

</div>
