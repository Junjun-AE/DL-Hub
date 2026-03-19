# -*- coding: utf-8 -*-
"""PatchCore 数据配置面板 - v2.8
修复：
- 图像增强默认不使用
- DL-Hub模式下隐藏输出目录选项
- 修复参数保存覆盖问题
- 从app.py统一导入dlhub_params（避免多实例问题）
"""

import gradio as gr
import sys
from pathlib import Path
from config import (BACKBONE_OPTIONS, DEFAULT_BACKBONE, IMAGE_SIZE_OPTIONS, DEFAULT_IMAGE_SIZE,
    DEFAULT_CORESET_RATIO, DEFAULT_PCA_COMPONENTS, DEFAULT_KNN_K, DEVICE_OPTIONS, get_backbone_info)

# 从app.py导入共享的dlhub_params实例（延迟导入，在函数内进行）
dlhub_params = None

def _get_dlhub_params():
    """延迟获取dlhub_params，避免循环导入"""
    global dlhub_params
    if dlhub_params is None:
        try:
            from gui.app import dlhub_params as _params
            dlhub_params = _params
        except ImportError:
            pass
    return dlhub_params


def get_default_output_dir():
    """获取默认输出目录，优先使用DL-Hub任务目录"""
    params = _get_dlhub_params()
    if params and params.is_dlhub_mode:
        return str(params.get_output_dir())
    return './output'

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

def create_config_panel():
    """创建数据配置面板"""
    # 获取共享的dlhub_params实例
    params = _get_dlhub_params()
    
    # 加载保存的参数
    saved_config = {} if not params else params.get_section('config')
    
    with gr.Accordion("🔧 环境检查", open=False):
        env_status = gr.HTML('<div style="color:#5f6368;font-size:16px;">点击检查按钮查看环境状态</div>')
        check_env_btn = gr.Button("🔍 检查环境", variant="secondary", size="lg")
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(_section("📁", "数据配置"))
            with gr.Group():
                data_dir = gr.Textbox(
                    label="数据集目录", 
                    value=saved_config.get('data_dir', ''),
                    placeholder="📂 输入数据集路径（需包含 good 文件夹）", 
                    info="支持MVTec格式或简单格式"
                )
                # DL-Hub模式下输出目录固定，不允许修改
                is_dlhub_mode = params and params.is_dlhub_mode
                output_dir = gr.Textbox(
                    label="输出目录", 
                    value=get_default_output_dir(),  # DL-Hub模式下始终使用任务目录
                    info="模型保存位置" + ("（DL-Hub模式，自动设置）" if is_dlhub_mode else ""),
                    interactive=not is_dlhub_mode,  # DL-Hub模式下只读
                    visible=not is_dlhub_mode  # DL-Hub模式下隐藏
                )
                data_info = gr.HTML(_alert('info', '请选择数据集目录'))
            
            gr.HTML(_section("🧠", "模型配置"))
            with gr.Group():
                with gr.Row():
                    backbone = gr.Dropdown(
                        label="骨干网络", 
                        choices=[(v['name'], k) for k, v in BACKBONE_OPTIONS.items()], 
                        value=saved_config.get('backbone', DEFAULT_BACKBONE), 
                        info="WideResNet50精度高"
                    )
                    image_size = gr.Dropdown(
                        label="输入尺寸", 
                        choices=IMAGE_SIZE_OPTIONS, 
                        value=saved_config.get('image_size', DEFAULT_IMAGE_SIZE), 
                        info="越大精度越高"
                    )
                backbone_info = gr.HTML(f'<div style="background:#f8f9fa;padding:16px 20px;border-radius:10px;font-size:16px;color:#5f6368;margin-top:8px;">{get_backbone_info(DEFAULT_BACKBONE)}</div>')
        
        with gr.Column(scale=1):
            gr.HTML(_section("💾", "Memory Bank"))
            with gr.Group():
                coreset_ratio = gr.Slider(
                    label="CoreSet采样率(%)", 
                    minimum=0.1, maximum=20, 
                    value=saved_config.get('coreset_ratio', 1.0), 
                    step=0.1, 
                    info="🏭 工业:1-5% | 高精度:5-10%"
                )
                with gr.Row():
                    pca_components = gr.Slider(
                        label="PCA维度", 
                        minimum=64, maximum=512, 
                        value=saved_config.get('pca_components', DEFAULT_PCA_COMPONENTS), 
                        step=32
                    )
                    knn_k = gr.Slider(
                        label="KNN近邻数", 
                        minimum=1, maximum=50, 
                        value=saved_config.get('knn_k', DEFAULT_KNN_K), 
                        step=1
                    )
            
            gr.HTML(_section("📦", "导出配置"))
            with gr.Group():
                with gr.Row():
                    export_tensorrt = gr.Checkbox(
                        label="导出TensorRT", 
                        value=saved_config.get('export_tensorrt', True)
                    )
                    tensorrt_precision = gr.Dropdown(
                        label="精度", 
                        choices=['fp32','fp16','int8'], 
                        value=saved_config.get('tensorrt_precision', 'fp16')
                    )
            
            with gr.Accordion("⚡ 高级选项", open=False):
                gr.HTML('<div style="background:#e8f0fe;padding:18px;border-radius:10px;font-size:16px;color:#1967d2;margin-bottom:16px;">💡 FP16节省内存 | 增量PCA避免溢出 | 随机投影加速10倍+</div>')
                
                gr.HTML('<div style="font-size:17px;font-weight:700;color:#202124;margin:16px 0 10px;">🖼️ 图像增强</div>')
                with gr.Row():
                    # 默认全部关闭
                    aug_hflip = gr.Checkbox(label="水平翻转", value=saved_config.get('aug_hflip', False))
                    aug_vflip = gr.Checkbox(label="垂直翻转", value=saved_config.get('aug_vflip', False))
                    aug_rotate = gr.Checkbox(label="随机旋转", value=saved_config.get('aug_rotate', False))
                with gr.Row():
                    # 默认设为0
                    aug_brightness = gr.Slider(label="亮度变化", minimum=0, maximum=0.5, value=saved_config.get('aug_brightness', 0), step=0.05)
                    aug_contrast = gr.Slider(label="对比度变化", minimum=0, maximum=0.5, value=saved_config.get('aug_contrast', 0), step=0.05)
                
                gr.HTML('<div style="font-size:17px;font-weight:700;color:#202124;margin:16px 0 10px;">⚙️ 优化选项</div>')
                with gr.Row():
                    use_fp16_features = gr.Checkbox(label="FP16特征", value=saved_config.get('use_fp16_features', True))
                    incremental_pca = gr.Checkbox(label="增量PCA", value=saved_config.get('incremental_pca', True))
                feature_chunk_size = gr.Slider(label="特征分块", minimum=1000, maximum=50000, value=saved_config.get('feature_chunk_size', 10000), step=1000)
                random_projection = gr.Checkbox(label="随机投影加速", value=saved_config.get('random_projection', True))
                with gr.Row():
                    device = gr.Dropdown(label="设备", choices=DEVICE_OPTIONS, value=saved_config.get('device', 'auto'))
                    num_workers = gr.Slider(label="加载进程", minimum=0, maximum=16, value=saved_config.get('num_workers', 4), step=1)
    
    # 开始训练按钮
    gr.HTML('<div style="height:20px;"></div>')
    with gr.Row():
        start_btn = gr.Button("🚀 开始训练", variant="primary", size="lg", scale=2)
    config_status = gr.HTML("")
    
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
    
    def on_start_training(data_dir_val, output_dir_val, backbone_val, image_size_val, coreset_ratio_val, 
                          pca_components_val, knn_k_val, export_tensorrt_val, tensorrt_precision_val,
                          aug_hflip_val, aug_vflip_val, aug_rotate_val, aug_brightness_val, aug_contrast_val, 
                          use_fp16_features_val, incremental_pca_val, feature_chunk_size_val, 
                          random_projection_val, device_val, num_workers_val):
        """启动训练"""
        if not data_dir_val:
            return _alert('error', '请先选择数据集目录')
        
        # 保存配置参数到DL-Hub（使用闭包中的params变量）
        if params:
            current_params = params.get_all()
            current_params['config'] = {
                'data_dir': data_dir_val,
                'output_dir': output_dir_val,
                'backbone': backbone_val,
                'image_size': image_size_val,
                'coreset_ratio': coreset_ratio_val,
                'pca_components': pca_components_val,
                'knn_k': knn_k_val,
                'export_tensorrt': export_tensorrt_val,
                'tensorrt_precision': tensorrt_precision_val,
                'aug_hflip': aug_hflip_val,
                'aug_vflip': aug_vflip_val,
                'aug_rotate': aug_rotate_val,
                'aug_brightness': aug_brightness_val,
                'aug_contrast': aug_contrast_val,
                'use_fp16_features': use_fp16_features_val,
                'incremental_pca': incremental_pca_val,
                'feature_chunk_size': feature_chunk_size_val,
                'random_projection': random_projection_val,
                'device': device_val,
                'num_workers': num_workers_val,
            }
            params.save(current_params)
        
        # 在函数内部导入以避免循环导入
        from gui.components.train_status_panel import start_training_task
        
        # 调用共享的训练启动函数
        success, msg = start_training_task(
            data_dir_val, output_dir_val, backbone_val, image_size_val, coreset_ratio_val, 
            pca_components_val, knn_k_val, export_tensorrt_val, tensorrt_precision_val,
            aug_hflip_val, aug_vflip_val, aug_rotate_val, aug_brightness_val, aug_contrast_val, 
            use_fp16_features_val, incremental_pca_val, feature_chunk_size_val, 
            random_projection_val, device_val, num_workers_val
        )
        
        if success:
            return _alert('success', '🚀 训练已启动！请切换到「📊 训练状态」页面查看进度')
        else:
            return _alert('error', msg)
    
    # ===== 绑定事件 =====
    check_env_btn.click(fn=check_environment, outputs=[env_status])
    data_dir.change(fn=validate_dataset, inputs=[data_dir], outputs=[data_info])
    backbone.change(fn=update_backbone_info, inputs=[backbone], outputs=[backbone_info])
    
    start_btn.click(
        fn=on_start_training,
        inputs=[data_dir, output_dir, backbone, image_size, coreset_ratio, pca_components, knn_k,
                export_tensorrt, tensorrt_precision, aug_hflip, aug_vflip, aug_rotate,
                aug_brightness, aug_contrast, use_fp16_features, incremental_pca,
                feature_chunk_size, random_projection, device, num_workers],
        outputs=[config_status],
    )
