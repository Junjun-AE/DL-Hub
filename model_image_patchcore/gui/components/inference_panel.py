# -*- coding: utf-8 -*-
"""PatchCore 批量推理面板 - v2.8
修复:
1. 检测完成后自动刷新一次(使用轮询检查完成状态)
2. 不使用高频Timer避免闪动
3. 使用低频Timer(3秒)仅在运行时刷新，完成后停止
"""

import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading, time, json, csv

inference_state = {
    'predictor': None, 'is_running': False, 'should_stop': False, 
    'results': [], 'progress': 0, 'total': 0, 'start_time': None,
    'current_threshold': 50, 'error_log': [], 'last_refresh_count': 0
}

def _sec(icon, title, desc=""):
    d = f'<span style="color:#5f6368;font-size:16px;margin-left:12px;">{desc}</span>' if desc else ''
    return f'<div style="display:flex;align-items:center;gap:14px;margin:28px 0 18px;padding-bottom:12px;border-bottom:2px solid #e8eaed;"><span style="font-size:28px;">{icon}</span><span style="font-size:22px;font-weight:700;color:#202124;">{title}{d}</span></div>'

def _alert(t, msg):
    s = {'success':('#e6f4ea','#1e8e3e','✓'), 'error':('#fce8e6','#d93025','✗'), 'info':('#e8f0fe','#1967d2','ℹ'), 'warning':('#fef7e0','#b06000','!')}
    bg,c,i = s.get(t, s['info'])
    return f'<div style="background:{bg};color:{c};padding:18px 22px;border-radius:12px;font-size:17px;margin:14px 0;"><span style="font-weight:700;margin-right:12px;font-size:18px;">{i}</span>{msg}</div>'

def _status_html(is_running, progress, total_done, total_all):
    if not is_running and total_done == 0:
        return '<div style="background:#f1f3f4;border-radius:12px;padding:20px;text-align:center;"><span style="font-size:18px;color:#5f6368;">⏸️ 等待开始</span></div>'
    elif is_running:
        return f'<div style="background:#e8f0fe;border-radius:12px;padding:20px;text-align:center;"><span style="font-size:18px;color:#1a73e8;font-weight:600;">🔄 检测中... {progress}% ({total_done}/{total_all})</span></div>'
    else:
        return f'<div style="background:#e6f4ea;border-radius:12px;padding:20px;text-align:center;"><span style="font-size:18px;color:#1e8e3e;font-weight:600;">✅ 检测完成 ({total_done} 张)</span></div>'

def _progress_html(progress):
    bar_color = '#1a73e8' if progress < 100 else '#34a853'
    return f'''<div style="background:#f8f9fa;border-radius:12px;padding:16px;border:1px solid #e8eaed;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;"><span style="font-size:16px;font-weight:600;">检测进度</span><span style="font-size:18px;font-weight:700;color:{bar_color};">{progress}%</span></div>
        <div style="height:12px;background:#e8eaed;border-radius:6px;overflow:hidden;"><div style="height:100%;width:{progress}%;background:{bar_color};border-radius:6px;transition:width 0.3s;"></div></div>
    </div>'''

def _model_card(info):
    if not info: return '<div style="background:#f8f9fa;border-radius:14px;padding:30px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:40px;opacity:0.3;margin-bottom:10px;">🧠</div><div style="color:#5f6368;font-size:17px;">请加载模型</div></div>'
    return f'''<div style="background:linear-gradient(135deg,#e8f0fe,#d2e3fc);border-radius:14px;padding:22px;border:1px solid rgba(26,115,232,0.2);"><div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;"><span style="font-size:28px;">🧠</span><span style="font-size:18px;font-weight:700;color:#1967d2;">模型已加载</span></div><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:14px;"><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:14px;color:#5f6368;">骨干网络</div><div style="font-size:18px;font-weight:700;color:#202124;">{info.get('backbone','-')}</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:14px;color:#5f6368;">输入尺寸</div><div style="font-size:18px;font-weight:700;color:#202124;">{info.get('input_size','-')}</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:14px;color:#5f6368;">Memory Bank</div><div style="font-size:18px;font-weight:700;color:#202124;">{info.get('memory_bank',0):,}</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:14px;color:#5f6368;">默认阈值</div><div style="font-size:18px;font-weight:700;color:#202124;">{info.get('threshold',50):.1f}</div></div></div></div>'''

def _stats_html(total, anomaly, normal, avg_time):
    if total == 0: return '<div style="background:#f8f9fa;border-radius:14px;padding:30px;text-align:center;border:1px solid #e8eaed;"><div style="color:#5f6368;font-size:17px;">检测后显示统计</div></div>'
    ap, np_ = (anomaly/total*100, normal/total*100) if total else (0, 0)
    return f'''<div style="background:#f8f9fa;border-radius:14px;padding:22px;border:1px solid #e8eaed;"><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px;"><div style="background:white;padding:18px;border-radius:10px;text-align:center;"><div style="font-size:32px;font-weight:700;color:#1a73e8;">{total}</div><div style="font-size:15px;color:#5f6368;">总计</div></div><div style="background:white;padding:18px;border-radius:10px;text-align:center;"><div style="font-size:32px;font-weight:700;color:#34a853;">{normal}</div><div style="font-size:15px;color:#5f6368;">正常 ({np_:.1f}%)</div></div><div style="background:white;padding:18px;border-radius:10px;text-align:center;"><div style="font-size:32px;font-weight:700;color:#ea4335;">{anomaly}</div><div style="font-size:15px;color:#5f6368;">异常 ({ap:.1f}%)</div></div><div style="background:white;padding:18px;border-radius:10px;text-align:center;"><div style="font-size:32px;font-weight:700;color:#5f6368;">{avg_time:.0f}</div><div style="font-size:15px;color:#5f6368;">耗时(ms)</div></div></div><div style="height:10px;background:#e8eaed;border-radius:5px;overflow:hidden;display:flex;"><div style="width:{np_}%;background:#34a853;"></div><div style="width:{ap}%;background:#ea4335;"></div></div></div>'''

def _result_card(score, is_anomaly, threshold, filename=""):
    if is_anomaly: bg, border, icon, txt, color = "linear-gradient(135deg,#fce8e6,#f8d7da)", "rgba(234,67,53,0.3)", "🚨", "异常", "#d93025"
    else: bg, border, icon, txt, color = "linear-gradient(135deg,#e6f4ea,#d4edda)", "rgba(52,168,83,0.3)", "✅", "正常", "#1e8e3e"
    filename_html = f'<div style="font-size:14px;color:#5f6368;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">📄 {filename}</div>' if filename else ''
    return f'<div style="background:{bg};border-radius:14px;padding:24px;border:1px solid {border};text-align:center;">{filename_html}<div style="font-size:48px;margin-bottom:8px;">{icon}</div><div style="font-size:26px;font-weight:700;color:{color};">{txt}</div><div style="font-size:42px;font-weight:700;color:#202124;margin:12px 0;">{score:.1f}</div><div style="font-size:16px;color:#5f6368;">阈值: {threshold:.1f}</div></div>'

def _setup_chinese_font():
    import matplotlib; matplotlib.use('Agg')
    for font in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']:
        try: matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']; break
        except: continue
    matplotlib.rcParams['axes.unicode_minus'] = False

def create_inference_panel():
    # 导入模型扫描函数
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app import scan_model_folders, scan_models_in_selected_folder, get_full_model_path, dlhub_params
    
    # 加载保存的参数
    saved_inference = {} if not dlhub_params else dlhub_params.get_section('inference')
    
    # 获取默认输出目录
    def get_default_output():
        if dlhub_params and dlhub_params.is_dlhub_mode:
            return str(dlhub_params.get_output_dir() / 'detection_results')
        return './detection_results'
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(_sec("🧠", "模型配置"))
            with gr.Group():
                model_folder = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="选择训练文件夹",
                    info="从output目录选择训练结果"
                )
                model_path = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="选择模型文件",
                    allow_custom_value=True
                )
                model_info = gr.HTML(_model_card(None))
                load_btn = gr.Button("📂 加载模型", variant="secondary", size="lg")
            gr.HTML(_sec("📁", "输入数据"))
            with gr.Group():
                input_mode = gr.Radio(label="输入方式", choices=["文件夹", "上传图片"], value="文件夹")
                input_folder = gr.Textbox(
                    label="图片文件夹", 
                    value=saved_inference.get('input_folder', ''),
                    placeholder="包含待检测图片的目录", visible=True
                )
                input_files = gr.File(label="上传图片", file_count="multiple", file_types=["image"], visible=False)
                input_stats = gr.HTML(_alert('info', '请选择输入数据'))
            gr.HTML(_sec("⚙️", "检测配置"))
            with gr.Group():
                threshold = gr.Slider(
                    label="检测阈值", minimum=0, maximum=100, 
                    value=saved_inference.get('threshold', 50), 
                    step=1, info="调高→更严格 | 调低→更敏感"
                )
                with gr.Row():
                    save_heatmap = gr.Checkbox(label="保存热力图", value=saved_inference.get('save_heatmap', True))
                    save_binary = gr.Checkbox(label="保存二值图", value=saved_inference.get('save_binary', False))
            gr.HTML(_sec("📤", "输出配置"))
            with gr.Group():
                output_folder = gr.Textbox(
                    label="输出目录", 
                    value=saved_inference.get('output_folder', get_default_output())
                )
                export_format = gr.CheckboxGroup(
                    label="导出格式", 
                    choices=["CSV报告", "JSON详情", "Excel表格"], 
                    value=saved_inference.get('export_format', ["CSV报告"])
                )
            with gr.Row():
                start_btn = gr.Button("🚀 开始检测", variant="primary", size="lg", scale=2)
                stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg", scale=1)
        
        with gr.Column(scale=1):
            gr.HTML(_sec("📊", "检测状态"))
            status_display = gr.HTML(_status_html(False, 0, 0, 0))
            progress_display = gr.HTML(_progress_html(0))
            gr.HTML(_sec("📈", "检测统计"))
            stats_display = gr.HTML(_stats_html(0, 0, 0, 0))
            result_plot = gr.Plot(label="分数分布")
            gr.HTML(_sec("🖼️", "结果预览"))
            with gr.Row():
                preview_filter_anomaly = gr.Checkbox(label="仅显示异常", value=False)
                preview_filter_good = gr.Checkbox(label="仅显示良品", value=False)
            preview_filter_info = gr.HTML('<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: 全部样本 (0 张)</div>')
            with gr.Row():
                preview_prev_btn = gr.Button("◀ 上一个", size="lg")
                preview_idx = gr.Slider(label="样本编号", minimum=1, maximum=1, value=1, step=1)
                preview_next_btn = gr.Button("下一个 ▶", size="lg")
            with gr.Row():
                preview_img1 = gr.Image(label="原图", height=180)
                preview_heatmap1 = gr.Image(label="热力图", height=180)
            with gr.Row():
                preview_img2 = gr.Image(label="二值图", height=180)
                preview_contour = gr.Image(label="轮廓图", height=180)
            preview_result = gr.HTML('<div style="background:#f8f9fa;border-radius:10px;padding:20px;text-align:center;color:#5f6368;">检测后显示结果</div>')
            gr.HTML(_sec("📋", "详细结果"))
            results_table = gr.Dataframe(headers=["文件名", "分数", "判定", "耗时"], label="", interactive=False)
            with gr.Row(): export_btn = gr.Button("📥 导出结果", variant="secondary", size="lg")
            export_status = gr.HTML("")
    
    def toggle_mode(m): return gr.update(visible=(m=="文件夹")), gr.update(visible=(m=="上传图片"))
    
    def load_model_fn(path):
        global inference_state
        if not path: return _model_card(None)
        try:
            # 获取完整路径
            full_path = get_full_model_path(path)
            if not Path(full_path).exists(): return _alert('error', f'路径不存在')
            from inference.predictor import PatchCorePredictor
            predictor = PatchCorePredictor.from_package(full_path)
            inference_state['predictor'] = predictor
            cfg = predictor.config
            return _model_card({'backbone': cfg['backbone']['name'], 'input_size': f"{cfg['preprocessing']['input_size'][0]}×{cfg['preprocessing']['input_size'][1]}", 'memory_bank': cfg['memory_bank']['size'], 'threshold': predictor.get_threshold()})
        except Exception as e: return _alert('error', f'加载失败: {e}')
    
    def validate_input_fn(mode, folder, files):
        if mode == "文件夹":
            if not folder: return _alert('info', '请选择文件夹')
            from data.dataset import scan_image_directory
            imgs = scan_image_directory(folder)
            return _alert('success', f'找到 <strong>{len(imgs)}</strong> 张图片') if imgs else _alert('error', '未找到图片')
        elif mode == "上传图片":
            return _alert('success', f'已上传 <strong>{len(files)}</strong> 张图片') if files else _alert('info', '请上传图片')
        return _alert('info', '请选择输入数据')
    
    def get_filtered_results(threshold_val, filter_anomaly, filter_good):
        global inference_state
        results = inference_state['results']
        if not results: return [], 0
        filtered = []
        for r in results:
            is_anomaly = r['score'] >= threshold_val
            if filter_anomaly and filter_good: filtered.append({**r, 'is_anomaly': is_anomaly})
            elif filter_anomaly and is_anomaly: filtered.append({**r, 'is_anomaly': is_anomaly})
            elif filter_good and not is_anomaly: filtered.append({**r, 'is_anomaly': is_anomaly})
            elif not filter_anomaly and not filter_good: filtered.append({**r, 'is_anomaly': is_anomaly})
        return filtered, len(filtered)
    
    def get_filter_info_html(threshold_val, filter_anomaly, filter_good):
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        if filter_anomaly and filter_good: filter_desc = "全部样本"
        elif filter_anomaly: filter_desc = "异常样本"
        elif filter_good: filter_desc = "良品样本"
        else: filter_desc = "全部样本"
        if count == 0: return f'<div style="background:#fef7e0;padding:12px;border-radius:8px;font-size:14px;color:#b06000;">当前筛选: {filter_desc} (0 张)</div>'
        return f'<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: {filter_desc} ({count} 张)</div>'
    
    def show_sample_internal(idx, threshold_val, filter_anomaly, filter_good):
        global inference_state
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        if not filtered: return None, None, None, None, '<div style="background:#f8f9fa;border-radius:10px;padding:20px;text-align:center;color:#5f6368;">无符合条件的样本</div>'
        idx = max(1, min(int(idx), count)); r = filtered[idx - 1]
        preview_orig, preview_heat, preview_bin, preview_cont = r.get('original_img'), r.get('heatmap_img'), r.get('binary_img'), r.get('contour_img')
        is_anomaly = r['score'] >= threshold_val
        if preview_orig is None and 'path' in r:
            try:
                import cv2; from PIL import Image
                orig = np.array(Image.open(r['path']).convert('RGB')); preview_orig = orig
                amap = r.get('anomaly_map')
                if amap is not None:
                    H, W = orig.shape[:2]
                    if amap.shape != (H, W): amap = cv2.resize(amap, (W, H))
                    norm = np.clip(amap / 100, 0, 1)
                    preview_heat = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                    binary = ((amap >= threshold_val) * 255).astype(np.uint8)
                    if binary.shape != (H, W): binary = cv2.resize(binary, (W, H))
                    preview_bin = binary
                    contour_img = orig.copy(); contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2); preview_cont = contour_img
            except: pass
        return preview_orig, preview_heat, preview_bin, preview_cont, _result_card(r['score'], is_anomaly, threshold_val, r['filename'])
    
    def show_sample(idx, threshold_val, filter_anomaly, filter_good):
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        if count == 0: return None, None, None, None, '<div style="background:#f8f9fa;border-radius:10px;padding:20px;text-align:center;color:#5f6368;">无符合条件的样本</div>'
        idx = max(1, min(int(idx), count)); return show_sample_internal(idx, threshold_val, filter_anomaly, filter_good)
    
    def navigate(direction, current_idx, threshold_val, filter_anomaly, filter_good):
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        max_idx = max(1, count); current_idx = max(1, min(int(current_idx), max_idx))
        new_idx = max(1, current_idx - 1) if direction == 'prev' else min(max_idx, current_idx + 1)
        return (new_idx,) + show_sample(new_idx, threshold_val, filter_anomaly, filter_good)
    
    def on_filter_change(threshold_val, filter_anomaly, filter_good):
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        filter_info_html = get_filter_info_html(threshold_val, filter_anomaly, filter_good)
        new_max = max(1, count); imgs = show_sample_internal(1, threshold_val, filter_anomaly, filter_good)
        return gr.update(maximum=new_max, value=1), filter_info_html, imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]
    
    def create_plot(results, threshold_val):
        if not results: return None
        try:
            import matplotlib.pyplot as plt; _setup_chinese_font()
            scores = [r['score'] for r in results]
            fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(scores, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
            ax.axvline(x=threshold_val, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold_val:.0f})')
            ax.set_xlabel('Anomaly Score', fontsize=12); ax.set_ylabel('Count', fontsize=12); ax.legend(fontsize=11)
            ax.set_xlim([0, max(100, max(scores) + 10) if scores else 100]); plt.tight_layout(); return fig
        except: return None
    
    def start_detection(model_path_val, input_mode_val, input_folder_val, input_files_val, threshold_val, save_heatmap_val, save_binary_val, output_folder_val):
        global inference_state
        if inference_state['is_running']: return (_status_html(True, inference_state['progress'], len(inference_state['results']), inference_state['total']), _progress_html(inference_state['progress']), _alert('warning', '检测正在进行中'))
        if not inference_state['predictor']:
            # 使用完整路径加载模型
            full_model_path = get_full_model_path(model_path_val)
            load_model_fn(full_model_path)
            if not inference_state['predictor']: return (_status_html(False, 0, 0, 0), _progress_html(0), _alert('error', '请先加载模型'))
        from data.dataset import scan_image_directory
        image_paths = scan_image_directory(input_folder_val) if input_mode_val == "文件夹" else ([f.name for f in input_files_val] if input_files_val else [])
        if not image_paths: return (_status_html(False, 0, 0, 0), _progress_html(0), _alert('error', '没有找到图片'))
        
        # 保存推理参数到DL-Hub
        if dlhub_params:
            current_params = dlhub_params.get_all()
            current_params['inference'] = {
                'input_folder': input_folder_val or '',
                'threshold': threshold_val,
                'save_heatmap': save_heatmap_val,
                'save_binary': save_binary_val,
                'output_folder': output_folder_val,
            }
            dlhub_params.save(current_params)
        
        inference_state.update({'is_running': True, 'should_stop': False, 'results': [], 'progress': 0, 'total': len(image_paths), 'start_time': time.time(), 'current_threshold': threshold_val, 'error_log': [], 'last_refresh_count': 0})
        predictor = inference_state['predictor']; predictor.set_threshold(threshold_val)
        output_path = Path(output_folder_val); output_path.mkdir(parents=True, exist_ok=True)
        if save_heatmap_val: (output_path / 'heatmaps').mkdir(exist_ok=True)
        if save_binary_val: (output_path / 'binary').mkdir(exist_ok=True)
        
        def detection_thread():
            import cv2; total = len(image_paths)
            for i, img_path in enumerate(image_paths):
                if inference_state['should_stop']: break
                try:
                    result = predictor.predict(img_path, return_visualization=True)
                    record = {'filename': Path(img_path).name, 'path': str(img_path), 'score': result.score, 'is_anomaly': result.is_anomaly, 'inference_time_ms': getattr(result, 'inference_time_ms', 0), 'anomaly_map': result.anomaly_map}
                    if save_heatmap_val and result.anomaly_map is not None:
                        try:
                            from inference.predictor import create_visualization; vis = create_visualization(img_path, result)
                            hp = output_path / 'heatmaps' / f'{Path(img_path).stem}_heatmap.jpg'; cv2.imwrite(str(hp), cv2.cvtColor(vis['heatmap'], cv2.COLOR_RGB2BGR))
                            record['original_img'], record['heatmap_img'], record['binary_img'], record['contour_img'] = vis.get('original'), vis.get('heatmap'), vis.get('binary'), vis.get('contour')
                        except: pass
                    if save_binary_val and result.binary_mask is not None: cv2.imwrite(str(output_path / 'binary' / f'{Path(img_path).stem}_binary.png'), result.binary_mask)
                    inference_state['results'].append(record)
                except Exception as e:
                    inference_state['error_log'].append(f"{Path(img_path).name}: {e}")
                    inference_state['results'].append({'filename': Path(img_path).name, 'path': str(img_path), 'score': 0, 'is_anomaly': False, 'inference_time_ms': 0, 'anomaly_map': None, 'error': str(e)})
                inference_state['progress'] = int((i + 1) / total * 100)
            inference_state['progress'] = 100; inference_state['is_running'] = False
        
        threading.Thread(target=detection_thread, daemon=True).start()
        return (_status_html(True, 0, 0, len(image_paths)), _progress_html(0), _alert('info', f'🚀 开始检测 {len(image_paths)} 张图片'))
    
    def stop_detection(): inference_state['should_stop'] = True; return _alert('warning', '正在停止...')
    
    def auto_refresh(threshold_val, filter_anomaly, filter_good):
        """低频自动刷新 - 只在运行时刷新，完成后显示最终结果"""
        global inference_state
        results = inference_state['results']; is_running = inference_state['is_running']; progress = inference_state['progress']; total = inference_state['total']
        current_count = len(results)
        
        # 关键：如果没有运行且没有结果，不更新任何内容
        if not is_running and current_count == 0:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        
        # 关键：如果已完成且已经刷新过，不再更新
        if not is_running and inference_state['last_refresh_count'] == current_count and current_count > 0:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        
        # 记录本次刷新的数量
        inference_state['last_refresh_count'] = current_count
        
        # 生成更新内容
        status_html = _status_html(is_running, progress, current_count, total)
        progress_html = _progress_html(progress)
        
        if results:
            total_count = len(results); anomaly_count = sum(1 for r in results if r['score'] >= threshold_val); normal_count = total_count - anomaly_count
            avg_time = np.mean([r.get('inference_time_ms', 0) for r in results])
        else:
            total_count = anomaly_count = normal_count = 0; avg_time = 0
        
        stats_html = _stats_html(total_count, anomaly_count, normal_count, avg_time)
        plot = create_plot(results, threshold_val)
        
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        filter_info_html = get_filter_info_html(threshold_val, filter_anomaly, filter_good)
        new_max = max(1, count)
        
        if filtered:
            imgs = show_sample_internal(count, threshold_val, filter_anomaly, filter_good)
        else:
            imgs = (None, None, None, None, '<div style="background:#f8f9fa;border-radius:10px;padding:20px;text-align:center;color:#5f6368;">无结果</div>')
        
        table = [[r['filename'], f"{r['score']:.1f}", "🚨 异常" if r['score'] >= threshold_val else "✅ 正常", f"{r.get('inference_time_ms',0):.0f}ms"] for r in results[-50:]]
        
        return (status_html, progress_html, stats_html, plot, gr.update(maximum=new_max, value=min(count, new_max)), filter_info_html, imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], table)
    
    def on_threshold_change(threshold_val, filter_anomaly, filter_good, current_idx):
        global inference_state
        results = inference_state['results']
        if not results: return (_stats_html(0, 0, 0, 0), None, gr.update(maximum=1, value=1), get_filter_info_html(threshold_val, filter_anomaly, filter_good), None, None, None, None, '<div style="background:#f8f9fa;border-radius:10px;padding:20px;text-align:center;color:#5f6368;">检测后显示结果</div>', [])
        inference_state['current_threshold'] = threshold_val
        total_count = len(results); anomaly_count = sum(1 for r in results if r['score'] >= threshold_val); avg_time = np.mean([r.get('inference_time_ms', 0) for r in results])
        stats_html = _stats_html(total_count, anomaly_count, total_count - anomaly_count, avg_time)
        plot = create_plot(results, threshold_val)
        filtered, count = get_filtered_results(threshold_val, filter_anomaly, filter_good)
        filter_info_html = get_filter_info_html(threshold_val, filter_anomaly, filter_good)
        new_max = max(1, count); new_val = min(current_idx, new_max) if current_idx >= 1 else 1
        imgs = show_sample_internal(new_val, threshold_val, filter_anomaly, filter_good)
        table = [[r['filename'], f"{r['score']:.1f}", "🚨 异常" if r['score'] >= threshold_val else "✅ 正常", f"{r.get('inference_time_ms',0):.0f}ms"] for r in results[-50:]]
        return (stats_html, plot, gr.update(maximum=new_max, value=new_val), filter_info_html, imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], table)
    
    def export_results(output_folder_val, export_format_val, threshold_val):
        results = inference_state['results']
        if not results: return _alert('error', '没有可导出的结果')
        output_path = Path(output_folder_val); output_path.mkdir(parents=True, exist_ok=True); ts = datetime.now().strftime("%Y%m%d_%H%M%S"); exported = []
        try:
            export_data = [{'filename': r['filename'], 'score': r['score'], 'is_anomaly': r['score'] >= threshold_val, 'inference_time_ms': r.get('inference_time_ms', 0)} for r in results]
            if "CSV报告" in export_format_val:
                with open(output_path / f"report_{ts}.csv", 'w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=['filename', 'score', 'is_anomaly', 'inference_time_ms']); w.writeheader(); w.writerows(export_data)
                exported.append("CSV")
            if "JSON详情" in export_format_val:
                with open(output_path / f"details_{ts}.json", 'w', encoding='utf-8') as f: json.dump(export_data, f, indent=2, ensure_ascii=False)
                exported.append("JSON")
            if "Excel表格" in export_format_val:
                try: pd.DataFrame(export_data).to_excel(output_path / f"report_{ts}.xlsx", index=False); exported.append("Excel")
                except: exported.append("Excel(需openpyxl)")
            return _alert('success', f'已导出: {", ".join(exported)}')
        except Exception as e: return _alert('error', f'导出失败: {e}')
    
    input_mode.change(toggle_mode, [input_mode], [input_folder, input_files])
    
    # 模型文件夹选择事件
    model_folder.focus(fn=scan_model_folders, outputs=model_folder)
    model_folder.change(fn=scan_models_in_selected_folder, inputs=model_folder, outputs=model_path)
    
    load_btn.click(load_model_fn, [model_path], [model_info])
    input_folder.change(lambda m, f, _: validate_input_fn(m, f, None), [input_mode, input_folder, input_files], [input_stats])
    input_files.change(lambda m, _, f: validate_input_fn(m, None, f), [input_mode, input_folder, input_files], [input_stats])
    start_btn.click(start_detection, [model_path, input_mode, input_folder, input_files, threshold, save_heatmap, save_binary, output_folder], [status_display, progress_display, export_status])
    stop_btn.click(stop_detection, outputs=[export_status])
    threshold.change(on_threshold_change, [threshold, preview_filter_anomaly, preview_filter_good, preview_idx], [stats_display, result_plot, preview_idx, preview_filter_info, preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result, results_table])
    preview_idx.change(show_sample, [preview_idx, threshold, preview_filter_anomaly, preview_filter_good], [preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result])
    preview_prev_btn.click(lambda idx, t, fa, fg: navigate('prev', idx, t, fa, fg), [preview_idx, threshold, preview_filter_anomaly, preview_filter_good], [preview_idx, preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result])
    preview_next_btn.click(lambda idx, t, fa, fg: navigate('next', idx, t, fa, fg), [preview_idx, threshold, preview_filter_anomaly, preview_filter_good], [preview_idx, preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result])
    preview_filter_anomaly.change(on_filter_change, [threshold, preview_filter_anomaly, preview_filter_good], [preview_idx, preview_filter_info, preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result])
    preview_filter_good.change(on_filter_change, [threshold, preview_filter_anomaly, preview_filter_good], [preview_idx, preview_filter_info, preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result])
    export_btn.click(export_results, [output_folder, export_format, threshold], [export_status])
    
    # 使用3秒低频Timer，只在有变化时才更新
    refresh_timer = gr.Timer(value=3.0)
    refresh_timer.tick(auto_refresh, inputs=[threshold, preview_filter_anomaly, preview_filter_good], outputs=[status_display, progress_display, stats_display, result_plot, preview_idx, preview_filter_info, preview_img1, preview_heatmap1, preview_img2, preview_contour, preview_result, results_table])
