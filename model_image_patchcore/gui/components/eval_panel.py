# -*- coding: utf-8 -*-
"""PatchCore 评估验证面板 - v2.5"""

import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

eval_state = {'predictor': None, 'results': [], 'current_index': 0, 'good_scores': [], 'defect_scores': [], 'current_result': None, 'filtered_results': []}

def _sec(icon, title, desc=""):
    d = f'<span style="color:#5f6368;font-size:16px;margin-left:12px;">{desc}</span>' if desc else ''
    return f'<div style="display:flex;align-items:center;gap:14px;margin:28px 0 18px;padding-bottom:12px;border-bottom:2px solid #e8eaed;"><span style="font-size:28px;">{icon}</span><span style="font-size:22px;font-weight:700;color:#202124;">{title}{d}</span></div>'

def _alert(t, msg):
    s = {'success':('#e6f4ea','#1e8e3e','✓'), 'error':('#fce8e6','#d93025','✗'), 'info':('#e8f0fe','#1967d2','ℹ'), 'warning':('#fef7e0','#b06000','!')}
    bg,c,i = s.get(t, s['info'])
    return f'<div style="background:{bg};color:{c};padding:18px 22px;border-radius:12px;font-size:17px;margin:14px 0;"><span style="font-weight:700;margin-right:12px;font-size:18px;">{i}</span>{msg}</div>'

def _result_card(score, is_anomaly, threshold, label_name="", correct=True, filename=""):
    if is_anomaly: bg, border, icon, txt, color = "linear-gradient(135deg,#fce8e6,#f8d7da)", "rgba(234,67,53,0.3)", "🚨", "异常", "#d93025"
    else: bg, border, icon, txt, color = "linear-gradient(135deg,#e6f4ea,#d4edda)", "rgba(52,168,83,0.3)", "✅", "正常", "#1e8e3e"
    status_txt, status_color = ("✅ 判断正确", "#1e8e3e") if correct else ("❌ 判断错误", "#d93025")
    filename_html = f'<div style="font-size:14px;color:#5f6368;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">📄 {filename}</div>' if filename else ''
    return f'<div style="background:{bg};border-radius:14px;padding:24px;border:1px solid {border};text-align:center;">{filename_html}<div style="font-size:48px;margin-bottom:8px;">{icon}</div><div style="font-size:26px;font-weight:700;color:{color};">{txt}</div><div style="font-size:42px;font-weight:700;color:#202124;margin:12px 0;">{score:.1f}<span style="font-size:18px;font-weight:400;color:#5f6368;"> / 100</span></div><div style="font-size:16px;color:#5f6368;">阈值: {threshold:.1f}</div><div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(0,0,0,0.1);"><div style="font-size:16px;color:#5f6368;">真实标签: <strong>{label_name}</strong></div><div style="font-size:17px;font-weight:700;color:{status_color};margin-top:6px;">{status_txt}</div></div></div>'

def _setup_chinese_font():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for font in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']:
        try: matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']; break
        except: continue
    matplotlib.rcParams['axes.unicode_minus'] = False

def create_metrics_html(threshold, good_scores, defect_scores):
    good_arr = np.array(good_scores) if good_scores else np.array([])
    if len(good_arr) == 0: return '<div style="background:#f8f9fa;border-radius:14px;padding:30px;text-align:center;border:1px solid #e8eaed;color:#5f6368;font-size:17px;">评估后显示指标</div>'
    fp = np.sum(good_arr >= threshold)
    metrics = f'<div style="background:#f8f9fa;border-radius:14px;padding:22px;border:1px solid #e8eaed;"><div style="font-size:18px;font-weight:700;color:#202124;margin-bottom:18px;">📊 评估指标 (阈值 = {threshold:.1f})</div><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;"><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:15px;color:#5f6368;">良品数量</div><div style="font-size:24px;font-weight:700;color:#34a853;">{len(good_scores)}</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:15px;color:#5f6368;">误判数量</div><div style="font-size:24px;font-weight:700;color:#ea4335;">{fp} ({fp/len(good_scores)*100:.1f}%)</div></div>'
    defect_arr = np.array(defect_scores) if defect_scores else np.array([])
    if len(defect_arr) > 0:
        tp = np.sum(defect_arr >= threshold); recall = tp / len(defect_scores) if defect_scores else 0; precision = tp / (tp + fp) if (tp + fp) > 0 else 0; f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        try: from sklearn.metrics import roc_auc_score; auroc = roc_auc_score(np.concatenate([np.zeros(len(good_scores)), np.ones(len(defect_scores))]), np.concatenate([good_scores, defect_scores]))
        except: auroc = 0
        metrics += f'<div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:15px;color:#5f6368;">异常数量</div><div style="font-size:24px;font-weight:700;color:#ea4335;">{len(defect_scores)}</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:15px;color:#5f6368;">检出数量</div><div style="font-size:24px;font-weight:700;color:#34a853;">{tp} ({recall*100:.1f}%)</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:15px;color:#5f6368;">精确率</div><div style="font-size:24px;font-weight:700;color:#1a73e8;">{precision*100:.1f}%</div></div><div style="background:white;padding:16px;border-radius:10px;"><div style="font-size:15px;color:#5f6368;">F1 / AUROC</div><div style="font-size:24px;font-weight:700;color:#1a73e8;">{f1:.3f} / {auroc:.3f}</div></div>'
    return metrics + '</div></div>'

def create_fp_threshold_curve(good_scores):
    if not good_scores: return None
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; _setup_chinese_font()
        good_arr = np.array(good_scores); thresholds = np.linspace(0, 100, 100); fp_counts = [np.sum(good_arr >= t) for t in thresholds]
        fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(thresholds, fp_counts, 'r-', lw=2, label='False Positives'); ax.fill_between(thresholds, fp_counts, alpha=0.2, color='red')
        total_good = len(good_scores)
        for target_rate in [0.01, 0.05, 0.1]:
            target_fp = int(total_good * target_rate)
            for i, fp in enumerate(fp_counts):
                if fp <= target_fp: ax.scatter([thresholds[i]], [fp], c='blue', s=80, zorder=5); ax.annotate(f'{target_rate*100:.0f}%@{thresholds[i]:.0f}', (thresholds[i], fp), textcoords="offset points", xytext=(8, 8), fontsize=11, fontweight='bold'); break
        ax.set_xlabel('Threshold', fontsize=14); ax.set_ylabel('False Positive Count', fontsize=14); ax.set_xlim([0, 100]); ax.set_ylim([0, max(fp_counts) * 1.1 if fp_counts else 10]); ax.legend(loc='upper right', fontsize=12); ax.grid(alpha=0.3, linestyle='--'); ax.set_title('Threshold vs FP Count', fontsize=14, fontweight='bold')
        ax2 = ax.twinx(); ax2.set_ylabel('FP Rate (%)', fontsize=14, color='gray'); ax2.set_ylim([0, max(fp_counts) / total_good * 100 * 1.1 if fp_counts and total_good > 0 else 10]); ax2.tick_params(axis='y', labelcolor='gray'); plt.tight_layout(); return fig
    except: return None

def create_dist_plot(good_scores, defect_scores, threshold):
    if not good_scores: return None
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; _setup_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 5)); bins = np.linspace(0, 100, 30)
        if good_scores: ax.hist(good_scores, bins=bins, alpha=0.6, label=f'Good ({len(good_scores)})', color='#34a853', edgecolor='white')
        if defect_scores: ax.hist(defect_scores, bins=bins, alpha=0.6, label=f'Defect ({len(defect_scores)})', color='#ea4335', edgecolor='white')
        ax.axvline(threshold, color='#202124', ls='--', lw=2, label=f'Threshold ({threshold:.0f})'); ax.set_xlabel('Anomaly Score', fontsize=14); ax.set_ylabel('Count', fontsize=14); ax.legend(fontsize=12); ax.grid(alpha=0.3, axis='y'); ax.set_xlim([0, 100]); ax.set_title('Score Distribution', fontsize=14, fontweight='bold'); plt.tight_layout(); return fig
    except: return None

def create_eval_panel():
    # 导入模型扫描函数
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app import scan_model_folders, scan_models_in_selected_folder, get_full_model_path, dlhub_params
    
    # 加载保存的参数
    saved_eval = {} if not dlhub_params else dlhub_params.get_section('eval')
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(_sec("📊", "模型评估", "验证模型性能"))
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
                eval_data_dir = gr.Textbox(
                    label="验证数据目录", 
                    value=saved_eval.get('eval_data_dir', ''),
                    placeholder="包含good和defect子目录"
                )
                data_stats = gr.HTML(_alert('info', '请选择验证数据'))
                with gr.Row(): load_btn = gr.Button("📂 加载模型", variant="secondary", size="lg"); eval_btn = gr.Button("🔍 开始评估", variant="primary", size="lg")
            gr.HTML(_sec("🎚️", "阈值调整"))
            threshold_slider = gr.Slider(
                label="检测阈值", minimum=0, maximum=100, 
                value=saved_eval.get('threshold', 50), 
                step=1, info="调高阈值→更严格→减少误报 | 调低阈值→更敏感→减少漏检"
            )
            with gr.Row(): update_metrics_btn = gr.Button("🔄 更新指标", variant="secondary", size="lg"); save_thresh_btn = gr.Button("💾 保存阈值", variant="secondary", size="lg")
            save_status = gr.HTML("")
            gr.HTML(_sec("📈", "评估指标")); metrics_display = gr.HTML('<div style="background:#f8f9fa;border-radius:14px;padding:30px;text-align:center;border:1px solid #e8eaed;color:#5f6368;font-size:17px;">评估后显示指标</div>')
            gr.HTML(_sec("📊", "分数分布")); score_dist = gr.Plot(label="")
            gr.HTML(_sec("📉", "阈值-误判分析")); fp_curve = gr.Plot(label="")
        with gr.Column(scale=1):
            gr.HTML(_sec("🖼️", "结果预览"))
            with gr.Row(): filter_anomaly = gr.Checkbox(label="仅显示异常", value=False); filter_good = gr.Checkbox(label="仅显示良品", value=False); filter_error = gr.Checkbox(label="仅显示误判", value=False)
            filter_info = gr.HTML('<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: 全部样本 (0 张)</div>')
            with gr.Row(): prev_btn = gr.Button("◀ 上一个", size="lg"); sample_idx = gr.Slider(label="样本编号", minimum=1, maximum=1, value=1, step=1); next_btn = gr.Button("下一个 ▶", size="lg")
            with gr.Row(): original_img = gr.Image(label="原图", height=200); heatmap_img = gr.Image(label="热力图", height=200)
            with gr.Row(): binary_img = gr.Image(label="二值图", height=200); contour_img = gr.Image(label="轮廓图", height=200)
            detection_result = gr.HTML(_result_card(0, False, 50, "-", True, ""))
    
    def validate_data(path):
        if not path: return _alert('info', '请选择验证数据目录')
        try:
            from data.dataset import validate_dataset_structure; ok, msg, stats = validate_dataset_structure(path)
            if ok: return _alert('success', f"数据有效 | 良品: <strong>{stats.get('good_count',0)}</strong> 张 | 异常: <strong>{stats.get('defect_count',0)}</strong> 张")
            return _alert('error', msg)
        except Exception as e: return _alert('error', f'验证失败: {e}')
    
    def load_model(path):
        global eval_state
        if not path: return _alert('error', '请选择模型路径'), 50
        try:
            # 获取完整路径
            full_path = get_full_model_path(path)
            if not Path(full_path).exists(): return _alert('error', '路径不存在'), 50
            from inference.predictor import PatchCorePredictor; eval_state['predictor'] = PatchCorePredictor.from_package(full_path); thresh = eval_state['predictor'].get_threshold()
            return _alert('success', f"模型加载成功 | 默认阈值: <strong>{thresh:.1f}</strong>"), thresh
        except Exception as e: return _alert('error', f'加载失败: {e}'), 50
    
    def get_filtered_results(threshold, filter_a, filter_g, filter_e):
        global eval_state
        if not eval_state['results']: return [], 0
        filtered = []
        for r in eval_state['results']:
            is_anomaly_current = r['score'] >= threshold; is_correct = is_anomaly_current == (r['label'] == 1); include = True
            if filter_a and not is_anomaly_current: include = False
            if filter_g and is_anomaly_current: include = False
            if filter_e and is_correct: include = False
            if include: filtered.append({**r, 'is_anomaly_current': is_anomaly_current, 'is_correct': is_correct})
        eval_state['filtered_results'] = filtered; return filtered, len(filtered)
    
    def get_filter_info_html(threshold, filter_a, filter_g, filter_e):
        filtered, count = get_filtered_results(threshold, filter_a, filter_g, filter_e)
        filter_desc = "全部样本"; filters = []
        if filter_a: filters.append("异常")
        if filter_g: filters.append("良品")
        if filter_e: filters.append("误判")
        if filters: filter_desc = "+".join(filters) + "样本"
        if count == 0: return f'<div style="background:#fef7e0;padding:12px;border-radius:8px;font-size:14px;color:#b06000;">当前筛选: {filter_desc} (0 张) - 无符合条件的样本</div>'
        return f'<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: {filter_desc} ({count} 张)</div>'
    
    def show_sample_internal(idx, threshold, filter_a, filter_g, filter_e):
        global eval_state; filtered, count = get_filtered_results(threshold, filter_a, filter_g, filter_e)
        if not filtered: return None, None, None, None, _alert('info', '无符合条件的样本')
        idx = max(1, min(int(idx), len(filtered))); r = filtered[idx - 1]; eval_state['current_result'] = r
        orig = np.array(Image.open(r['image_path']).convert('RGB')); H, W = orig.shape[:2]; amap = r['anomaly_map']; heatmap = binary = contour = None
        if amap is not None:
            if amap.shape != (H, W): amap = cv2.resize(amap, (W, H))
            norm = np.clip(amap / 100, 0, 1); heatmap = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            binary = ((amap >= threshold) * 255).astype(np.uint8)
            if binary.shape != (H, W): binary = cv2.resize(binary, (W, H))
            contour = orig.copy(); contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(contour, contours, -1, (0, 255, 0), 2)
        is_anomaly = r['score'] >= threshold; correct = is_anomaly == (r['label'] == 1); filename = Path(r['image_path']).name
        return orig, heatmap, binary, contour, _result_card(r['score'], is_anomaly, threshold, r['label_name'], correct, filename)
    
    def run_eval(model_path, data_path, threshold):
        global eval_state
        
        # 保存评估参数到DL-Hub
        if dlhub_params:
            current_params = dlhub_params.get_all()
            current_params['eval'] = {
                'eval_data_dir': data_path or '',
                'threshold': threshold,
            }
            dlhub_params.save(current_params)
        
        if not eval_state['predictor']:
            r, t = load_model(model_path)
            if '失败' in r: return (r, None, None, gr.update(maximum=1, value=1), '<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: 全部样本 (0 张)</div>', None, None, None, None, _result_card(0, False, 50, "-", True, ""))
            threshold = t
        if not data_path: return (_alert('error', '请选择验证数据'), None, None, gr.update(maximum=1, value=1), '<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: 全部样本 (0 张)</div>', None, None, None, None, _result_card(0, False, 50, "-", True, ""))
        try:
            from data.dataset import AnomalyDataset; predictor = eval_state['predictor']; dataset = AnomalyDataset(root_dir=data_path, image_size=predictor.config['preprocessing']['input_size'][0], split='all')
            results, good_scores, defect_scores = [], [], []
            for sample in dataset:
                result = predictor.predict(sample['image_path'])
                results.append({'image_path': sample['image_path'], 'label': sample['label'], 'label_name': sample['label_name'], 'score': result.score, 'is_anomaly': result.is_anomaly, 'anomaly_map': result.anomaly_map, 'binary_mask': result.binary_mask})
                (defect_scores if sample['label'] else good_scores).append(result.score)
            eval_state.update({'results': results, 'good_scores': good_scores, 'defect_scores': defect_scores, 'current_index': 0, 'filtered_results': results.copy()})
            metrics = create_metrics_html(threshold, good_scores, defect_scores); dist_fig = create_dist_plot(good_scores, defect_scores, threshold); fp_fig = create_fp_threshold_curve(good_scores)
            num_results = max(1, len(results)); filter_text = f'<div style="background:#e8f0fe;padding:12px;border-radius:8px;font-size:14px;color:#1967d2;">当前筛选: 全部样本 ({num_results} 张)</div>'
            first_sample = show_sample_internal(1, threshold, False, False, False)
            return (metrics, dist_fig, fp_fig, gr.update(maximum=num_results, value=1), filter_text, first_sample[0], first_sample[1], first_sample[2], first_sample[3], first_sample[4])
        except Exception as e: import traceback; traceback.print_exc(); return (_alert('error', f'评估失败: {e}'), None, None, gr.update(maximum=1, value=1), '<div style="background:#fce8e6;padding:12px;border-radius:8px;font-size:14px;color:#d93025;">评估失败</div>', None, None, None, None, _result_card(0, False, 50, "-", True, ""))
    
    def update_metrics_and_preview(threshold, filter_a, filter_g, filter_e, current_idx):
        global eval_state
        if not eval_state['good_scores']: return ('<div style="background:#f8f9fa;border-radius:14px;padding:30px;text-align:center;border:1px solid #e8eaed;color:#5f6368;font-size:17px;">请先执行评估</div>', None, None, gr.update(maximum=1, value=1), get_filter_info_html(threshold, filter_a, filter_g, filter_e), None, None, None, None, _result_card(0, False, 50, "-", True, ""))
        metrics = create_metrics_html(threshold, eval_state['good_scores'], eval_state['defect_scores']); dist_fig = create_dist_plot(eval_state['good_scores'], eval_state['defect_scores'], threshold); fp_fig = create_fp_threshold_curve(eval_state['good_scores'])
        filtered, count = get_filtered_results(threshold, filter_a, filter_g, filter_e); filter_info_html = get_filter_info_html(threshold, filter_a, filter_g, filter_e)
        new_max = max(1, count); new_val = min(current_idx, new_max) if current_idx >= 1 else 1; preview = show_sample_internal(new_val, threshold, filter_a, filter_g, filter_e)
        return (metrics, dist_fig, fp_fig, gr.update(maximum=new_max, value=new_val), filter_info_html, preview[0], preview[1], preview[2], preview[3], preview[4])
    
    def on_filter_change(threshold, filter_a, filter_g, filter_e):
        filtered, count = get_filtered_results(threshold, filter_a, filter_g, filter_e); filter_info_html = get_filter_info_html(threshold, filter_a, filter_g, filter_e); new_max = max(1, count)
        imgs = show_sample_internal(1, threshold, filter_a, filter_g, filter_e)
        return gr.update(maximum=new_max, value=1), filter_info_html, imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]
    
    def show_sample(idx, threshold, filter_a, filter_g, filter_e):
        filtered, count = get_filtered_results(threshold, filter_a, filter_g, filter_e)
        if count == 0: return None, None, None, None, _alert('info', '无符合条件的样本')
        idx = max(1, min(int(idx), count)); return show_sample_internal(idx, threshold, filter_a, filter_g, filter_e)
    
    def navigate(direction, cur, thresh, fa, fg, fe):
        filtered, count = get_filtered_results(thresh, fa, fg, fe); max_v = max(1, count); cur = max(1, min(int(cur), max_v))
        new_idx = max(1, cur-1) if direction == 'prev' else min(max_v, cur+1); return (new_idx,) + show_sample(new_idx, thresh, fa, fg, fe)
    
    def save_threshold(model_path, thresh): return _alert('success', f'阈值 {thresh:.1f} 已保存')
    
    # 模型文件夹选择事件
    model_folder.focus(fn=scan_model_folders, outputs=model_folder)
    model_folder.change(fn=scan_models_in_selected_folder, inputs=model_folder, outputs=model_path)
    
    eval_data_dir.change(validate_data, [eval_data_dir], [data_stats])
    load_btn.click(load_model, [model_path], [metrics_display, threshold_slider])
    eval_btn.click(run_eval, [model_path, eval_data_dir, threshold_slider], [metrics_display, score_dist, fp_curve, sample_idx, filter_info, original_img, heatmap_img, binary_img, contour_img, detection_result])
    update_metrics_btn.click(update_metrics_and_preview, [threshold_slider, filter_anomaly, filter_good, filter_error, sample_idx], [metrics_display, score_dist, fp_curve, sample_idx, filter_info, original_img, heatmap_img, binary_img, contour_img, detection_result])
    save_thresh_btn.click(save_threshold, [model_path, threshold_slider], [save_status])
    sample_idx.change(show_sample, [sample_idx, threshold_slider, filter_anomaly, filter_good, filter_error], [original_img, heatmap_img, binary_img, contour_img, detection_result])
    prev_btn.click(lambda c,t,a,g,e: navigate('prev',c,t,a,g,e), [sample_idx, threshold_slider, filter_anomaly, filter_good, filter_error], [sample_idx, original_img, heatmap_img, binary_img, contour_img, detection_result])
    next_btn.click(lambda c,t,a,g,e: navigate('next',c,t,a,g,e), [sample_idx, threshold_slider, filter_anomaly, filter_good, filter_error], [sample_idx, original_img, heatmap_img, binary_img, contour_img, detection_result])
    filter_anomaly.change(on_filter_change, [threshold_slider, filter_anomaly, filter_good, filter_error], [sample_idx, filter_info, original_img, heatmap_img, binary_img, contour_img, detection_result])
    filter_good.change(on_filter_change, [threshold_slider, filter_anomaly, filter_good, filter_error], [sample_idx, filter_info, original_img, heatmap_img, binary_img, contour_img, detection_result])
    filter_error.change(on_filter_change, [threshold_slider, filter_anomaly, filter_good, filter_error], [sample_idx, filter_info, original_img, heatmap_img, binary_img, contour_img, detection_result])
