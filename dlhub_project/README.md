# 🧠 DL-Hub - 深度学习任务管理平台

统一管理您的深度学习任务：分类 · 检测 · 分割 · 异常检测 · OCR

![DL-Hub](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ 功能特性

- 🎯 **统一管理** - 一个界面管理5种深度学习任务
- 📁 **工作空间** - 集中存储所有任务和模型
- 💾 **参数保存** - 自动保存UI参数，下次继续
- 🔄 **状态监控** - 实时显示训练状态和指标
- 🚀 **一键启动** - 点击任务直接进入训练界面

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────┐
│           DL-Hub React Frontend                  │
│         http://localhost:7860                    │
└─────────────────────────────────────────────────┘
                      ↓ API
┌─────────────────────────────────────────────────┐
│           FastAPI Backend                        │
│   /api/workspace  /api/tasks  /api/app          │
└─────────────────────────────────────────────────┘
                      ↓ 子进程
┌─────────────────────────────────────────────────┐
│        5大任务 Gradio 应用                       │
│   分类 | 检测 | 分割 | 异常检测 | OCR            │
│         http://localhost:7861                    │
└─────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Node.js 16+（用于构建前端）
- 现有的5大任务训练代码

### 2. 安装依赖

```bash
# Python依赖
pip install -r requirements.txt

# 前端依赖（首次运行会自动安装）
cd dlhub/frontend && npm install
```

### 3. 启动DL-Hub

```bash
python run_dlhub.py
```

首次启动会：
1. 检查并安装依赖
2. 构建前端（如果需要）
3. 启动服务器
4. 自动打开浏览器

### 4. 配置工作空间

首次运行时会弹出工作空间选择窗口，选择一个目录用于存储所有任务。

---

## 📁 目录结构

```
dlhub_project/
├── run_dlhub.py              # 启动脚本
├── requirements.txt          # Python依赖
├── dlhub_config.json         # 全局配置（自动生成）
│
├── dlhub/
│   ├── backend/              # FastAPI后端
│   │   ├── main.py           # 应用入口
│   │   ├── config.py         # 配置管理
│   │   ├── routers/          # API路由
│   │   │   ├── workspace.py  # 工作空间API
│   │   │   ├── tasks.py      # 任务管理API
│   │   │   ├── app_launcher.py # 应用启动API
│   │   │   └── system.py     # 系统信息API
│   │   └── services/         # 业务服务
│   │       ├── task_service.py
│   │       ├── process_service.py
│   │       └── status_monitor.py
│   │
│   ├── frontend/             # React前端
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.jsx       # 主应用
│   │   │   └── api.js        # API客户端
│   │   └── dist/             # 构建产物
│   │
│   └── app_adapters/         # 应用适配器
│       ├── base_adapter.py   # 基础适配器
│       └── INTEGRATION_GUIDE.md # 集成指南
│
└── model_image_xxx/          # 原有的5大任务代码
```

---

## 📋 工作空间结构

```
你的工作空间/
├── .dlhub/
│   └── workspace.json        # 工作空间配置
│
├── classification/           # 分类任务
│   └── 产品分类_20240126/    # 任务目录
│       ├── .dlhub/
│       │   ├── task.json     # 任务元数据
│       │   └── ui_params.json # UI参数
│       └── output/           # 训练输出
│
├── detection/                # 检测任务
├── segmentation/             # 分割任务
├── anomaly/                  # 异常检测任务
└── ocr/                      # OCR任务
```

---

## 🔧 集成现有应用

要将现有的Gradio训练应用与DL-Hub集成，请参考：

[👉 集成指南](dlhub/app_adapters/INTEGRATION_GUIDE.md)

主要修改：
1. 添加命令行参数支持
2. 实现参数自动保存/加载
3. 添加训练状态上报

---

## 🌐 API文档

启动后访问 http://localhost:7860/api/docs 查看完整API文档

### 主要接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/workspace` | GET | 获取工作空间配置 |
| `/api/workspace/setup` | POST | 设置工作空间 |
| `/api/tasks` | GET | 获取任务列表 |
| `/api/tasks` | POST | 创建新任务 |
| `/api/app/launch/{task_id}` | POST | 启动训练应用 |
| `/api/app/stop` | POST | 停止训练应用 |

---

## ⚙️ 配置选项

### 启动参数

```bash
python run_dlhub.py [options]

Options:
  --port PORT        服务端口 (默认: 7860)
  --host HOST        服务地址 (默认: 0.0.0.0)
  --no-browser       不自动打开浏览器
  --rebuild          强制重新构建前端
  --dev              开发模式（启用热重载）
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DLHUB_TASK_DIR` | 任务目录 | - |
| `DLHUB_TASK_ID` | 任务ID | - |

---

## 🔍 常见问题

### Q: 前端构建失败？

确保已安装Node.js 16+：
```bash
node --version  # 应该是 v16 或更高
npm --version
```

### Q: 找不到训练应用？

检查`APP_PATHS`配置是否正确指向您的应用文件：
```python
# dlhub/backend/routers/app_launcher.py
APP_PATHS = {
    'classification': 'model_image_classification/app.py',
    ...
}
```

### Q: 参数没有保存？

确保您的应用已集成DL-Hub适配器，参考集成指南。

---

## 📝 开发说明

### 前端开发

```bash
cd dlhub/frontend
npm run dev  # 启动开发服务器（端口3000）
```

### 后端开发

```bash
python run_dlhub.py --dev  # 启用热重载
```

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交Issue和Pull Request！
