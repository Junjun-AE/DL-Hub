# DL-Hub 测试分析报告

## 一、当前逻辑梳理（从测试工程师角度）

### 1. 任务创建流程

```
用户点击"新建任务" 
    → 填写：任务名称、工作目录、Conda环境
    → 后端 task_service.create_task()
        → 创建文件夹：{工作目录}/{任务名}_{时间戳}_{ID}/
        → 创建子目录：.dlhub/, output/
        → 创建配置：.dlhub/task.json
        → 添加到全局配置：dlhub_config.json
```

### 2. 任务打开流程（关键路径）

```
用户点击任务卡片
    → 前端调用 api.launchApp(taskId)
    → 后端 app_launcher.launch_app()
        → 检查是否有正在运行的App
            → ⚠️ 问题1：当前会自动停止旧App，没有提示用户！
        → 获取App路径：get_app_path(task_type)
            → ⚠️ 问题2：路径计算可能错误
        → 启动进程：process_service.launch_app()
            → 设置环境变量：
                - DLHUB_TASK_DIR = 任务目录路径
                - DLHUB_TASK_ID = 任务ID
            → 启动命令：python app.py --task-dir {路径} --port 7861
    → 返回URL，前端打开新标签页
```

### 3. App内部逻辑（以目标检测为例）

```
app.py 启动
    → init_dlhub_adapter() 初始化适配器
        → 读取 DLHUB_TASK_DIR 环境变量
    → get_output_dir() 获取输出目录
        → ⚠️ 问题3：当前返回 ./output，不是任务目录下的output！
    → 训练开始
        → output_dir = Path("./output") / f"{model}_{timestamp}"
            → ⚠️ 问题4：没有使用 get_output_dir()！输出到了App目录下！
        → 保存模型到 output_dir
```

### 4. 发现的严重问题

| 问题 | 描述 | 影响 |
|-----|------|------|
| **问题1** | 打开新任务会自动停止旧任务，没有提示 | 用户可能丢失训练进度 |
| **问题2** | App路径计算依赖目录结构假设 | 启动失败 |
| **问题3** | 各App的输出目录没有正确使用任务目录 | 模型保存位置错误 |
| **问题4** | 参数保存/加载功能没有实际集成到App | 每次打开参数都是默认值 |
| **问题5** | ui_params.json 没有被App读取 | 参数不能持久化 |

---

## 二、输出目录问题详细分析

### 2.1 目标检测 (app.py 第622-625行)

```python
# 当前代码
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("./output") / f"{config['name']}_{timestamp}"  # ❌ 使用相对路径
output_dir.mkdir(parents=True, exist_ok=True)
training_state.output_dir = str(output_dir)

# 应该是
output_base = get_output_dir()  # 从DL-Hub获取任务目录
output_dir = output_base / f"{config['name']}_{timestamp}"
```

### 2.2 图像分类、语义分割、异常检测、OCR

需要逐一检查并修复相同问题。

---

## 三、需要修复的内容

### 3.1 后端修复

1. **app_launcher.py**: 启动前检查是否有运行中的任务，返回状态供前端弹窗
2. **api.js**: 添加获取运行状态的API调用

### 3.2 前端修复

1. **App.jsx**: 启动任务前先检查运行状态，弹窗提示用户

### 3.3 各App修复（5个）

1. 正确使用 `get_output_dir()` 获取输出目录
2. 实现参数保存：训练前保存到 `.dlhub/ui_params.json`
3. 实现参数加载：启动时从文件恢复UI组件值

---

## 四、正确的数据流

### 4.1 任务目录结构（修复后）

```
D:/projects/缺陷检测_20250129_143052_abc123/
├── .dlhub/
│   ├── task.json           # 任务元数据
│   ├── ui_params.json      # UI参数 ← App读写
│   └── running.pid         # 运行进程ID
│
├── output/                  # 训练输出 ← App写入
│   └── yolov8n_20250129_150000/
│       ├── dataset/        # 转换后的数据集
│       ├── weights/        # 模型权重
│       │   ├── best.pt
│       │   └── last.pt
│       └── results.csv
│
├── data/                    # 用户数据（可选）
└── exports/                 # 导出模型（可选）
```

### 4.2 参数流

```
创建任务 → 打开App → 加载 ui_params.json → 显示在UI
                           ↓
用户修改参数 → 点击训练 → 保存到 ui_params.json → 开始训练
                           ↓
下次打开 → 自动加载上次参数
```
