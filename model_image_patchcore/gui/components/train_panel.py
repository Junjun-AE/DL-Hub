# -*- coding: utf-8 -*-
"""PatchCore 训练面板 - 优化版 v2.1
分离数据配置与训练状态页面，使用v1.9风格进度条
"""

import gradio as gr
import threading
import time
import sys
from pathlib import Path
from config import (BACKBONE_OPTIONS, DEFAULT_BACKBONE, IMAGE_SIZE_OPTIONS, DEFAULT_IMAGE_SIZE,
    DEFAULT_CORESET_RATIO, DEFAULT_PCA_COMPONENTS, DEFAULT_KNN_K, DEVICE_OPTIONS, get_backbone_info)

# 导入DL-Hub参数保存函数
try:
    from gui.app import save_all_params, get_output_dir, get_saved_param, dlhub_params
except ImportError:
    def save_all_params(params): return False
    def get_output_dir(): return Path('./output')
    def get_saved_param(key, default=None): return default
    dlhub_params = None

# 获取保存的参数
def get_saved_section(section: str) -> dict:
    """获取保存的参数section"""
    if dlhub_params:
        return dlhub_params.get_section(section)
    return {}

TRAINING_PHASES = ["初始化", "加载数据", "特征提取", "构建Bank", "优化索引", "导出模型"]
training_state = {
    'trainer': None, 'is_training': False, 'should_stop': False, 'logs': [], 
    'current_phase': '', 'phase_progress': 0, 'total_progress': 0, 
    'current_phase_idx': 0, 'total_phases': 6, 'start_time': None, 'result': None,
    'last_update_time': 0  # 用于控制刷新频率
}

def _alert(t, msg):
    s = {'success':('#e6f4ea','#1e8e3e','✓'), 'error':('#fce8e6','#d93025','✗'), 'info':('#e8f0fe','#1967d2','ℹ'), 'warning':('#fef7e0','#b06000','!')}
    bg,c,i = s.get(t, s['info'])
    return f'<div style="background:{bg};color:{c};padding:18px 22px;border-radius:12px;font-size:17px;margin:14px 0;"><span style="font-weight:700;margin-right:12px;font-size:18px;">{i}</span>{msg}</div>'

def _section(icon, title, desc=""):
    d = f'<span style="color:#5f6368;font-size:16px;margin-left:12px;">{desc}</span>' if desc else ''
    return f'<div style="display:flex;align-items:center;gap:14px;margin:28px 0 18px;padding-bottom:12px;border-bottom:2px solid #e8eaed;"><span style="font-size:28px;">{icon}</span><span style="font-size:22px;font-weight:700;color:#202124;">{title}{d}</span></div>'

def _env_result(results):
    items = ""
    for r in results:
        status = r.get('status', 'ok')
        colors = {'ok':('#e6f4ea','#1e8e3e','✓'), 'warning':('#fef7e0','#b06000','!'), 'error':('#fce8e6','#d93025','✗')}
        bg, color, icon = colors.get(status, colors['ok'])
        ver = f'<span style="color:#5f6368;font-size:14px;margin-left:8px;">{r.get("version","")}</span>' if r.get('version') else ''
        det = f'<div style="font-size:14px;color:#5f6368;">{r.get("detail","")}</div>' if r.get('detail') else ''
        items += f'''<div style="display:flex;align-items:flex-start;gap:12px;padding:14px 18px;background:{bg};border-radius:10px;">
            <span style="width:26px;height:26px;border-radius:50%;background:{color};color:white;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0;">{icon}</span>
            <div><div style="font-size:17px;font-weight:600;color:#202124;">{r.get("name","")}{ver}</div>{det}</div>
        </div>'''
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;">{items}</div>'

def check_environment():
    results = []
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results.append({'name': 'Python', 'status': 'ok', 'version': py_ver})
    try:
        import torch
        if torch.cuda.is_available():
            results.append({'name': 'PyTorch', 'status': 'ok', 'version': torch.__version__, 'detail': 'CUDA'})
            results.append({'name': 'GPU', 'status': 'ok', 'version': f'{torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f}GB', 'detail': torch.cuda.get_device_name(0)[:20]})
        else: results.append({'name': 'PyTorch', 'status': 'warning', 'version': torch.__version__, 'detail': 'CPU'})
    except: results.append({'name': 'PyTorch', 'status': 'error', 'detail': '未安装'})
    for mod, name, req in [('timm','timm',True),('faiss','Faiss',True),('sklearn','sklearn',True),('onnx','ONNX',False),('tensorrt','TensorRT',False)]:
        try: __import__(mod); results.append({'name': name, 'status': 'ok'})
        except: results.append({'name': name, 'status': 'error' if req else 'warning', 'detail': '未安装' + ('' if req else '(可选)')})
    return _env_result(results)

def create_train_panel():
    """创建训练面板 - 使用子Tab分离数据配置和训练状态"""
    
    # 加载保存的参数
    saved_data = get_saved_section('data')
    saved_model = get_saved_section('model')
    saved_training = get_saved_section('training')
    saved_aug = get_saved_section('augmentation')
    saved_export = get_saved_section('export')
    
    with gr.Tabs() as train_tabs:
        # ===== Tab 1: 数据配置 =====
        with gr.Tab("📁 数据配置", id="config_tab"):
            with gr.Accordion("🔧 环境检查", open=False):
                env_status = gr.HTML('<div style="color:#5f6368;font-size:16px;">点击检查按钮查看环境状态</div>')
                check_env_btn = gr.Button("🔍 检查环境", variant="secondary", size="lg")
            
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML(_section("📁", "数据配置"))
                    with gr.Group():
                        data_dir = gr.Textbox(label="数据集目录", placeholder="📂 输入数据集路径（需包含 good 文件夹）", info="支持MVTec格式或简单格式", value=saved_data.get('data_dir', ''))
                        output_dir = gr.Textbox(label="输出目录", value=str(get_output_dir()), info="模型保存位置")
                        data_info = gr.HTML(_alert('info', '请选择数据集目录'))
                    
                    gr.HTML(_section("🧠", "模型配置"))
                    with gr.Group():
                        with gr.Row():
                            backbone = gr.Dropdown(label="骨干网络", choices=[(v['name'], k) for k, v in BACKBONE_OPTIONS.items()], value=saved_model.get('backbone', DEFAULT_BACKBONE), info="WideResNet50精度高")
                            image_size = gr.Dropdown(label="输入尺寸", choices=IMAGE_SIZE_OPTIONS, value=saved_model.get('image_size', DEFAULT_IMAGE_SIZE), info="越大精度越高")
                        backbone_info = gr.HTML(f'<div style="background:#f8f9fa;padding:16px 20px;border-radius:10px;font-size:16px;color:#5f6368;margin-top:8px;">{get_backbone_info(saved_model.get("backbone", DEFAULT_BACKBONE))}</div>')
                
                with gr.Column(scale=1):
                    gr.HTML(_section("💾", "Memory Bank"))
                    with gr.Group():
                        coreset_ratio = gr.Slider(label="CoreSet采样率(%)", minimum=0.1, maximum=20, value=saved_training.get('coreset_ratio', 1.0), step=0.1, info="🏭 工业:1-5% | 高精度:5-10%")
                        with gr.Row():
                            pca_components = gr.Slider(label="PCA维度", minimum=64, maximum=512, value=saved_training.get('pca_components', DEFAULT_PCA_COMPONENTS), step=32)
                            knn_k = gr.Slider(label="KNN近邻数", minimum=1, maximum=50, value=saved_training.get('knn_k', DEFAULT_KNN_K), step=1)
                    
                    gr.HTML(_section("📦", "导出配置"))
                    with gr.Group():
                        with gr.Row():
                            export_tensorrt = gr.Checkbox(label="导出TensorRT", value=saved_export.get('tensorrt', True))
                            tensorrt_precision = gr.Dropdown(label="精度", choices=['fp32','fp16','int8'], value=saved_export.get('precision', 'fp16'))
                    
                    with gr.Accordion("⚡ 高级选项", open=False):
                        gr.HTML('<div style="background:#e8f0fe;padding:18px;border-radius:10px;font-size:16px;color:#1967d2;margin-bottom:16px;">💡 FP16节省内存 | 增量PCA避免溢出 | 随机投影加速10倍+</div>')
                        
                        gr.HTML('<div style="font-size:17px;font-weight:700;color:#202124;margin:16px 0 10px;">🖼️ 图像增强</div>')
                        with gr.Row():
                            aug_hflip = gr.Checkbox(label="水平翻转", value=saved_aug.get('hflip', True))
                            aug_vflip = gr.Checkbox(label="垂直翻转", value=saved_aug.get('vflip', False))
                            aug_rotate = gr.Checkbox(label="随机旋转", value=saved_aug.get('rotate', True))
                        with gr.Row():
                            aug_brightness = gr.Slider(label="亮度变化", minimum=0, maximum=0.5, value=saved_aug.get('brightness', 0.1), step=0.05)
                            aug_contrast = gr.Slider(label="对比度变化", minimum=0, maximum=0.5, value=saved_aug.get('contrast', 0.1), step=0.05)
                        
                        gr.HTML('<div style="font-size:17px;font-weight:700;color:#202124;margin:16px 0 10px;">⚙️ 优化选项</div>')
                        with gr.Row():
                            use_fp16_features = gr.Checkbox(label="FP16特征", value=saved_training.get('fp16_features', True))
                            incremental_pca = gr.Checkbox(label="增量PCA", value=saved_training.get('incremental_pca', True))
                        feature_chunk_size = gr.Slider(label="特征分块", minimum=1000, maximum=50000, value=saved_training.get('feature_chunk_size', 10000), step=1000)
                        random_projection = gr.Checkbox(label="随机投影加速", value=saved_training.get('random_projection', True))
                        with gr.Row():
                            device = gr.Dropdown(label="设备", choices=DEVICE_OPTIONS, value=saved_training.get('device', 'auto'))
                            num_workers = gr.Slider(label="加载进程", minimum=0, maximum=16, value=saved_training.get('num_workers', 4), step=1)
            
            # 开始训练按钮
            gr.HTML('<div style="height:20px;"></div>')
            start_btn = gr.Button("🚀 开始训练", variant="primary", size="lg", scale=1)
            config_status = gr.HTML("")
        
        # ===== Tab 2: 训练状态 =====
        with gr.Tab("📊 训练状态", id="status_tab"):
            with gr.Row():
                stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg")
            
            # 状态显示 (使用v1.9风格的Markdown和Slider)
            status_text = gr.Markdown("**状态**: 未开始")
            time_estimate = gr.Markdown("**⏱️ 预计时间**: -")
            
            # 总进度条 (v1.9风格)
            gr.Markdown("**总进度**")
            progress_slider = gr.Slider(
                label="",
                minimum=0, maximum=100, value=0,
                interactive=False,
            )
            
            # 阶段进度
            phase_info = gr.Markdown("**当前阶段**: 等待开始")
            phase_progress_bar = gr.Slider(
                label="阶段进度",
                minimum=0, maximum=100, value=0,
                interactive=False,
            )
            
            # Memory 使用
            with gr.Row():
                memory_info = gr.Markdown("**内存**: GPU 0MB | CPU 0MB")
                refresh_memory_btn = gr.Button("🔄", size="sm")
            
            # 训练结果摘要
            gr.HTML(_section("📋", "训练结果"))
            result_summary = gr.Markdown("*训练完成后显示*")
            
            # 日志输出
            gr.HTML(_section("📝", "训练日志"))
            log_output = gr.Textbox(
                label="",
                lines=12,
                max_lines=15,
                interactive=False,
                show_label=False,
            )
    
    # ===== 事件处理函数 =====
    
    def validate_dataset(path):
        if not path: return _alert('info', '请选择数据集目录')
        try:
            from data.dataset import validate_dataset_structure
            ok, msg, stats = validate_dataset_structure(path)
            if ok: return _alert('success', f"数据集有效 | 良品: <strong>{stats.get('good_count',0)}</strong> 张 | 异常: {stats.get('defect_count',0)} 张")
            return _alert('error', msg)
        except Exception as e: return _alert('error', f'验证失败: {e}')
    
    def update_backbone_info(name):
        return f'<div style="background:#f8f9fa;padding:16px 20px;border-radius:10px;font-size:16px;color:#5f6368;margin-top:8px;">{get_backbone_info(name)}</div>'
    
    def get_memory_info():
        gpu_mem = cpu_mem = 0
        try:
            import torch
            if torch.cuda.is_available(): gpu_mem = torch.cuda.memory_allocated() / (1024**2)
        except: pass
        try:
            import psutil; cpu_mem = psutil.Process().memory_info().rss / (1024**2)
        except: pass
        return f"**内存**: GPU {gpu_mem:.0f}MB | CPU {cpu_mem:.0f}MB"
    
    def start_training(data_dir_val, output_dir_val, backbone_val, image_size_val, coreset_ratio_val, 
                       pca_components_val, knn_k_val, export_tensorrt_val, tensorrt_precision_val,
                       aug_hflip_val, aug_vflip_val, aug_rotate_val, aug_brightness_val, aug_contrast_val, 
                       use_fp16_features_val, incremental_pca_val, feature_chunk_size_val, 
                       random_projection_val, device_val, num_workers_val):
        global training_state
        
        if training_state['is_training']:
            return (
                _alert('warning', '训练正在进行中...'),
                gr.update(selected="status_tab"),
                "**状态**: 🏃 训练中",
                f"**⏱️ 已用**: 计算中...",
                training_state['total_progress'],
                training_state['phase_progress'],
                f"**当前阶段**: {training_state['current_phase']}",
                "*训练中...*",
                ""
            )
        
        if not data_dir_val:
            return (
                _alert('error', '请先选择数据集目录'),
                gr.update(),
                "**状态**: ❌ 配置错误",
                "-",
                0, 0,
                "**当前阶段**: 等待开始",
                "*请选择数据集*",
                ""
            )
        
        # 重置状态
        training_state.update({
            'is_training': True, 'should_stop': False, 'logs': [],
            'current_phase': '初始化', 'phase_progress': 0, 'total_progress': 0,
            'current_phase_idx': 0, 'start_time': time.time(), 'result': None
        })
        
        try:
            from config import TrainingConfig
            from engine.trainer import PatchCoreTrainer, TrainingCallback
            
            # 保存参数到DL-Hub
            save_all_params({
                'data': {
                    'data_dir': data_dir_val,
                    'image_size': int(image_size_val),
                },
                'model': {
                    'backbone': backbone_val,
                    'coreset_ratio': float(coreset_ratio_val),
                    'pca_components': int(pca_components_val),
                    'knn_k': int(knn_k_val),
                },
                'augmentation': {
                    'hflip': aug_hflip_val,
                    'vflip': aug_vflip_val,
                    'rotate': aug_rotate_val,
                    'brightness': aug_brightness_val,
                    'contrast': aug_contrast_val,
                },
                'optimization': {
                    'use_fp16': use_fp16_features_val,
                    'incremental_pca': incremental_pca_val,
                    'feature_chunk_size': int(feature_chunk_size_val),
                    'random_projection': random_projection_val,
                },
                'export': {
                    'tensorrt_enabled': export_tensorrt_val,
                    'tensorrt_precision': tensorrt_precision_val,
                },
                'device': device_val,
            })
            
            # 使用DL-Hub输出目录
            actual_output_dir = output_dir_val if output_dir_val else str(get_output_dir())
            
            config = TrainingConfig()
            config.dataset_dir = data_dir_val
            config.output_dir = actual_output_dir
            config.image_size = int(image_size_val)
            config.device = device_val
            config.backbone.name = backbone_val
            config.memory_bank.coreset_sampling_ratio = float(coreset_ratio_val) / 100.0
            config.memory_bank.pca_components = int(pca_components_val)
            config.knn.k = int(knn_k_val)
            config.export.tensorrt_enabled = export_tensorrt_val
            config.export.tensorrt_precision = tensorrt_precision_val
            config.augmentation.horizontal_flip = aug_hflip_val
            config.augmentation.vertical_flip = aug_vflip_val
            config.augmentation.random_rotation = aug_rotate_val
            config.augmentation.brightness = aug_brightness_val
            config.augmentation.contrast = aug_contrast_val
            config.optimization.use_fp16_features = use_fp16_features_val
            config.optimization.incremental_pca = incremental_pca_val
            config.optimization.feature_chunk_size = int(feature_chunk_size_val)
            config.optimization.random_projection_enabled = random_projection_val
            config.optimization.num_workers = int(num_workers_val)
            
            trainer = PatchCoreTrainer(config)
            training_state['trainer'] = trainer
            
            def on_log(msg):
                training_state['logs'].append(msg)
                if len(training_state['logs']) > 100:
                    training_state['logs'] = training_state['logs'][-100:]
            
            def on_phase_start(phase_name, current, total):
                training_state['current_phase'] = phase_name
                training_state['current_phase_idx'] = current
                training_state['total_phases'] = total
                training_state['phase_progress'] = 0
                training_state['total_progress'] = int((current - 1) / total * 100)
            
            def on_progress(current, total, message):
                if total > 0:
                    training_state['phase_progress'] = int(current / total * 100)
            
            def should_stop():
                return training_state['should_stop']
            
            callback = TrainingCallback(
                on_log=on_log,
                on_phase_start=on_phase_start,
                on_progress=on_progress,
                should_stop=should_stop,
            )
            trainer.set_callback(callback)
            
            def train_thread():
                try:
                    result = trainer.train()
                    training_state['result'] = result
                    training_state['total_progress'] = 100
                except Exception as e:
                    import traceback
                    training_state['logs'].append(f"❌ 训练错误: {e}")
                    traceback.print_exc()
                finally:
                    training_state['is_training'] = False
            
            threading.Thread(target=train_thread).start()
            
            # 返回成功启动的状态，并切换到训练状态Tab
            return (
                _alert('success', '🚀 训练已启动，正在跳转到训练状态页面...'),
                gr.update(selected="status_tab"),
                "**状态**: 🏃 训练已启动",
                "**⏱️ 预计时间**: 计算中...",
                0, 0,
                f"**当前阶段** [1/{training_state['total_phases']}]: 初始化",
                "*训练中...*",
                "训练启动中..."
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            training_state['is_training'] = False
            return (
                _alert('error', f'启动失败: {e}'),
                gr.update(),
                f"**状态**: ❌ {e}",
                "-",
                0, 0,
                "**当前阶段**: 错误",
                f"*{e}*",
                str(e)
            )
    
    def stop_training():
        global training_state
        if not training_state['is_training']:
            return "**状态**: ⚠️ 当前没有训练任务"
        training_state['should_stop'] = True
        if training_state['trainer']:
            training_state['trainer'].request_stop()
        return "**状态**: ⏳ 正在停止..."
    
    def update_progress():
        """更新进度显示 - 训练完成后停止刷新动画"""
        global training_state
        
        # 检查是否需要跳过更新（控制刷新频率为3秒）
        current_time = time.time()
        if training_state['is_training'] and (current_time - training_state.get('last_update_time', 0)) < 3:
            # 训练中时，每3秒更新一次
            pass
        training_state['last_update_time'] = current_time
        
        if not training_state['is_training'] and training_state['result'] is None:
            return (
                "**状态**: 未开始",
                "**⏱️ 预计时间**: -",
                0, 0,
                "**当前阶段**: 等待开始",
                "*训练完成后显示*",
                ""
            )
        
        # 状态文本
        if training_state['is_training']:
            status = f"**状态**: 🏃 训练中 ({training_state['total_progress']}%)"
        elif training_state['result'] and training_state['result'].success:
            status = "**状态**: ✅ 训练完成"
        elif training_state['result']:
            status = f"**状态**: ❌ {training_state['result'].message}"
        else:
            status = "**状态**: ⏹️ 已停止"
        
        # 时间估计
        if training_state['start_time'] and training_state['is_training']:
            elapsed = time.time() - training_state['start_time']
            progress = training_state['total_progress']
            if progress > 5:
                total_est = elapsed / progress * 100
                remaining = total_est - elapsed
                time_str = f"**⏱️ 已用**: {elapsed/60:.1f}min | **剩余**: ~{remaining/60:.1f}min"
            else:
                time_str = f"**⏱️ 已用**: {elapsed:.0f}s"
        elif training_state['result']:
            time_str = f"**⏱️ 总用时**: {training_state['result'].total_time_seconds/60:.1f}min"
        else:
            time_str = "**⏱️ 预计时间**: -"
        
        # 阶段信息
        phase_idx = training_state['current_phase_idx']
        total_phases = training_state['total_phases']
        phase_str = f"**当前阶段** [{phase_idx}/{total_phases}]: {training_state['current_phase']}"
        
        # 结果摘要
        if training_state['result'] and training_state['result'].success:
            r = training_state['result']
            result_str = f"""
✅ **训练成功完成**

| 指标 | 数值 |
|------|------|
| 图像数量 | {r.num_images} |
| 原始Patch数 | {r.num_patches:,} |
| Memory Bank | {r.memory_bank_size:,} ({r.memory_bank_size/r.num_patches*100:.2f}%) |
| 特征维度 | {r.feature_dim}-D |
| PCA方差 | {r.pca_variance_explained:.2%} |
| 默认阈值 | {r.default_threshold:.1f} |

📦 **模型路径**: `{r.export_path}`
"""
        else:
            result_str = "*训练完成后显示*"
        
        # 日志
        logs = "\n".join(training_state['logs'][-30:])
        
        return (
            status,
            time_str,
            training_state['total_progress'],
            training_state['phase_progress'],
            phase_str,
            result_str,
            logs,
        )
    
    # ===== 绑定事件 =====
    check_env_btn.click(fn=check_environment, outputs=[env_status])
    data_dir.change(fn=validate_dataset, inputs=[data_dir], outputs=[data_info])
    backbone.change(fn=update_backbone_info, inputs=[backbone], outputs=[backbone_info])
    
    # 开始训练 - 切换到训练状态Tab
    start_btn.click(
        fn=start_training,
        inputs=[data_dir, output_dir, backbone, image_size, coreset_ratio, pca_components, knn_k,
                export_tensorrt, tensorrt_precision, aug_hflip, aug_vflip, aug_rotate,
                aug_brightness, aug_contrast, use_fp16_features, incremental_pca,
                feature_chunk_size, random_projection, device, num_workers],
        outputs=[config_status, train_tabs, status_text, time_estimate, progress_slider, 
                 phase_progress_bar, phase_info, result_summary, log_output],
    )
    
    stop_btn.click(fn=stop_training, outputs=[status_text])
    refresh_memory_btn.click(fn=get_memory_info, outputs=[memory_info])
    
    # 定时器 - 训练完成后使用较长间隔
    timer = gr.Timer(value=3.0)  # 改为3秒刷新一次
    timer.tick(
        fn=update_progress,
        outputs=[status_text, time_estimate, progress_slider, phase_progress_bar,
                 phase_info, result_summary, log_output],
    )
