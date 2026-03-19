# -*- coding: utf-8 -*-
"""批量OCR处理面板 (增强版 v2.1)
增强：
- 支持DL-Hub日志保存
- 使用单例模式共享dlhub_params
"""

import gradio as gr
import numpy as np
from pathlib import Path
import threading, time, json, csv
import sys

# 添加父目录以便导入dlhub_params
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 使用单例模式获取DL-Hub参数管理器
dlhub_params = None
try:
    from dlhub_params import get_dlhub_params
    dlhub_params = get_dlhub_params()
except Exception:
    pass


def get_default_output_dir():
    """获取默认输出目录，优先使用DL-Hub任务目录"""
    if dlhub_params and dlhub_params.is_dlhub_mode:
        return str(dlhub_params.get_output_dir() / 'ocr_results')
    return './ocr_results'


batch_state = {'engine': None, 'is_running': False, 'should_stop': False, 'results': [], 'progress': 0, 'total': 0, 'logs': []}

# 【增强】加载历史日志
def _load_saved_state():
    global batch_state
    if dlhub_params:
        saved_logs = dlhub_params.get_logs('inference')
        if saved_logs:
            batch_state['logs'] = saved_logs
            print(f"[DL-Hub] ✓ 已恢复OCR处理日志: {len(saved_logs)} 行")

_load_saved_state()

def _add_log(msg):
    """添加日志"""
    batch_state['logs'].append(msg)
    if len(batch_state['logs']) > 100:
        batch_state['logs'] = batch_state['logs'][-100:]
    if dlhub_params:
        dlhub_params.append_log(msg, 'inference', auto_save=False)

def _sec(icon, title):
    return f'<div style="display:flex;align-items:center;gap:10px;margin:20px 0 14px;padding-bottom:10px;border-bottom:2px solid #e8eaed;"><span style="font-size:24px;">{icon}</span><span style="font-size:18px;font-weight:700;color:#202124;">{title}</span></div>'

def _alert(t, msg):
    s = {'success':('#e6f4ea','#1e8e3e','✓'), 'error':('#fce8e6','#d93025','✗'), 'info':('#e8f0fe','#1967d2','ℹ'), 'warning':('#fef7e0','#b06000','!')}
    bg, c, i = s.get(t, s['info'])
    return f'<div style="background:{bg};color:{c};padding:14px 18px;border-radius:10px;font-size:15px;margin:10px 0;"><span style="font-weight:700;margin-right:10px;">{i}</span>{msg}</div>'

def _status_html(running, progress, done, total):
    if not running and done == 0:
        return '<div style="background:#f1f3f4;border-radius:10px;padding:16px;text-align:center;color:#5f6368;">⏸️ 等待开始</div>'
    elif running:
        return f'<div style="background:#e8f0fe;border-radius:10px;padding:16px;text-align:center;color:#1a73e8;font-weight:600;">🔄 处理中 {progress}% ({done}/{total})</div>'
    else:
        return f'<div style="background:#e6f4ea;border-radius:10px;padding:16px;text-align:center;color:#1e8e3e;font-weight:600;">✅ 完成 ({done}张)</div>'

def _progress_html(p):
    c = '#1a73e8' if p < 100 else '#34a853'
    return f'<div style="background:#f8f9fa;border-radius:10px;padding:14px;border:1px solid #e8eaed;"><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="font-weight:600;">进度</span><span style="font-weight:700;color:{c};">{p}%</span></div><div style="height:10px;background:#e8eaed;border-radius:5px;overflow:hidden;"><div style="height:100%;width:{p}%;background:{c};border-radius:5px;"></div></div></div>'

def _stats_html(total, texts, avg_ms, total_s):
    return f'''<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
        <div style="background:#f8f9fa;padding:14px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:26px;font-weight:700;color:#1a73e8;">{total}</div><div style="font-size:13px;color:#5f6368;">图片数</div></div>
        <div style="background:#f8f9fa;padding:14px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:26px;font-weight:700;color:#34a853;">{texts}</div><div style="font-size:13px;color:#5f6368;">文本数</div></div>
        <div style="background:#f8f9fa;padding:14px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:26px;font-weight:700;color:#ea4335;">{avg_ms:.0f}</div><div style="font-size:13px;color:#5f6368;">平均(ms)</div></div>
        <div style="background:#f8f9fa;padding:14px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:26px;font-weight:700;color:#5f6368;">{total_s:.1f}</div><div style="font-size:13px;color:#5f6368;">总耗时(s)</div></div>
    </div>'''

def create_batch_ocr_panel():
    # 加载保存的参数
    saved_params = {} if not dlhub_params else dlhub_params.get_section('batch_ocr')
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            gr.HTML(_sec("📁", "输入配置"))
            with gr.Group():
                input_folder = gr.Textbox(
                    label="图片文件夹", 
                    value=saved_params.get('input_folder', ''),
                    placeholder="选择包含图片的文件夹"
                )
                input_stats = gr.HTML(_alert('info', '请选择文件夹'))
            
            gr.HTML(_sec("⚙️", "处理配置"))
            with gr.Group():
                use_gpu = gr.Checkbox(label="GPU加速", value=saved_params.get('use_gpu', True))
                save_vis = gr.Checkbox(label="保存可视化", value=saved_params.get('save_vis', True))
            
            gr.HTML(_sec("📤", "输出配置"))
            with gr.Group():
                output_folder = gr.Textbox(
                    label="输出目录", 
                    value=saved_params.get('output_folder', get_default_output_dir())
                )
                export_format = gr.CheckboxGroup(
                    choices=["CSV", "JSON"], 
                    value=saved_params.get('export_format', ["CSV", "JSON"]), 
                    label="导出格式"
                )
            
            with gr.Row():
                start_btn = gr.Button("🚀 开始处理", variant="primary", size="lg", scale=2)
                stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg", scale=1)
        
        with gr.Column(scale=2, min_width=500):
            gr.HTML(_sec("📊", "处理状态"))
            status_html = gr.HTML(_status_html(False, 0, 0, 0))
            progress_html = gr.HTML(_progress_html(0))
            refresh_btn = gr.Button("🔄 刷新状态", variant="secondary")
            
            gr.HTML(_sec("📈", "处理统计"))
            stats_html = gr.HTML(_stats_html(0, 0, 0, 0))
            
            gr.HTML(_sec("📋", "处理结果"))
            results_table = gr.Dataframe(
                headers=["文件名", "文本数", "耗时(ms)", "状态"],
                datatype=["str", "number", "str", "str"],
                column_count=(4, "fixed"),
                interactive=False,
            )
            export_status = gr.HTML("")
    
    def validate_input(folder):
        if not folder: return _alert('info', '请选择文件夹')
        p = Path(folder)
        if not p.exists(): return _alert('error', '文件夹不存在')
        imgs = list(p.glob('*.jpg')) + list(p.glob('*.png')) + list(p.glob('*.jpeg')) + list(p.glob('*.JPG')) + list(p.glob('*.PNG'))
        return _alert('success', f'找到 {len(imgs)} 张图片') if imgs else _alert('warning', '未找到图片')
    
    def start_batch(folder, gpu, save_v, out_folder, formats):
        global batch_state
        if batch_state['is_running']:
            return _status_html(True, batch_state['progress'], len(batch_state['results']), batch_state['total']), _progress_html(batch_state['progress']), _alert('warning', '正在处理')
        
        p = Path(folder)
        if not p.exists():
            return _status_html(False,0,0,0), _progress_html(0), _alert('error', '文件夹不存在')
        
        imgs = list(p.glob('*.jpg')) + list(p.glob('*.png')) + list(p.glob('*.jpeg')) + list(p.glob('*.JPG')) + list(p.glob('*.PNG'))
        if not imgs:
            return _status_html(False,0,0,0), _progress_html(0), _alert('error', '未找到图片')
        
        # 保存参数到DL-Hub
        if dlhub_params:
            current_params = dlhub_params.get_all()
            current_params['batch_ocr'] = {
                'input_folder': folder,
                'use_gpu': gpu,
                'save_vis': save_v,
                'output_folder': out_folder,
                'export_format': formats,
            }
            dlhub_params.save(current_params)
        
        # 【增强】清空日志，开始新处理
        batch_state['logs'] = []
        if dlhub_params:
            dlhub_params.clear_logs('inference', auto_save=False)
        _add_log(f"━━━ 开始批量OCR处理 ━━━")
        _add_log(f"📁 输入: {folder}")
        _add_log(f"📊 图片数: {len(imgs)}")
        
        try:
            from paddleocr import PaddleOCR
            import paddleocr
            version = getattr(paddleocr, '__version__', '2.0.0')
            major_version = int(version.split('.')[0])
            
            if major_version >= 3:
                # PaddleOCR 3.x: use device parameter, no show_log
                device = 'gpu' if gpu else 'cpu'
                batch_state['engine'] = PaddleOCR(use_angle_cls=False, lang='ch', device=device)
            else:
                # PaddleOCR 2.x: use use_gpu and show_log
                batch_state['engine'] = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=gpu, show_log=False)
            batch_state['ocr_version'] = major_version  # 保存版本信息
        except Exception as e:
            return _status_html(False,0,0,0), _progress_html(0), _alert('error', f'引擎加载失败: {e}')
        
        batch_state.update({'is_running': True, 'should_stop': False, 'results': [], 'progress': 0, 'total': len(imgs)})
        out_path = Path(out_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        
        def process():
            import cv2
            all_results = []
            ocr_v3 = batch_state.get('ocr_version', 2) >= 3  # 获取版本信息
            for i, img_path in enumerate(imgs):
                if batch_state['should_stop']: break
                try:
                    start = time.time()
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # PaddleOCR 3.x: ocr() 不再接受 cls 参数
                    if ocr_v3:
                        if hasattr(batch_state['engine'], 'predict'):
                            results = batch_state['engine'].predict(img_rgb)
                        else:
                            results = batch_state['engine'].ocr(img_rgb)
                    else:
                        results = batch_state['engine'].ocr(img_rgb, cls=False)
                    elapsed = (time.time() - start) * 1000
                    
                    texts = []
                    # ============ 兼容 PaddleOCR 2.x 和 3.x 的结果格式 ============
                    if results:
                        result_list = results[0] if isinstance(results, list) and len(results) > 0 else results
                        if result_list:
                            for item in result_list:
                                try:
                                    # PaddleOCR 3.x dict 格式
                                    if isinstance(item, dict):
                                        box = item.get('dt_polys', item.get('box', item.get('points', [])))
                                        text = item.get('rec_text', item.get('text', ''))
                                        score = item.get('rec_score', item.get('score', 0.0))
                                        if box:
                                            texts.append({
                                                'text': str(text),
                                                'score': float(score) if score else 0.0,
                                                'box': [list(map(int, pt)) for pt in box] if isinstance(box[0], (list, tuple)) else box
                                            })
                                    # PaddleOCR 2.x list 格式
                                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                        box = item[0]
                                        text_info = item[1]
                                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                            text, score = text_info[0], text_info[1]
                                        elif isinstance(text_info, dict):
                                            text = text_info.get('text', text_info.get('rec_text', ''))
                                            score = text_info.get('score', text_info.get('rec_score', 0.0))
                                        else:
                                            continue
                                        texts.append({
                                            'text': str(text),
                                            'score': float(score),
                                            'box': [list(map(int, pt)) for pt in box]
                                        })
                                except (IndexError, TypeError, ValueError):
                                    continue
                        
                        if save_v and texts:
                            vis = img.copy()
                            for t in texts:
                                cv2.polylines(vis, [np.array(t['box'])], True, (0,255,0), 2)
                            cv2.imwrite(str(out_path / f'{img_path.stem}_vis.jpg'), vis)
                    
                    record = {'filename': img_path.name, 'texts': texts, 'num_texts': len(texts), 'time_ms': elapsed, 'status': 'success'}
                    all_results.append(record)
                    batch_state['results'].append(record)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    batch_state['results'].append({'filename': img_path.name, 'texts': [], 'num_texts': 0, 'time_ms': 0, 'status': f'error'})
                batch_state['progress'] = int((i+1) / len(imgs) * 100)
                
                # 每处理10张图片保存一次日志，避免应用关闭时丢失
                if (i + 1) % 10 == 0 and dlhub_params:
                    dlhub_params.save_logs(batch_state['logs'], 'inference', auto_save=True)
            
            if "CSV" in formats:
                with open(out_path / 'results.csv', 'w', newline='', encoding='utf-8-sig') as f:
                    w = csv.writer(f)
                    w.writerow(['文件名', '文本', '置信度'])
                    for r in all_results:
                        for t in r['texts']:
                            w.writerow([r['filename'], t['text'], f"{t['score']:.2%}"])
            
            if "JSON" in formats:
                with open(out_path / 'results.json', 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            batch_state['progress'] = 100
            batch_state['is_running'] = False
            
            # 【增强】保存完成日志
            total_texts = sum(r['num_texts'] for r in all_results)
            total_time = sum(r['time_ms'] for r in all_results) / 1000
            _add_log(f"━━━ 处理完成 ━━━")
            _add_log(f"✅ 处理: {len(all_results)} 张图片")
            _add_log(f"📝 识别: {total_texts} 个文本")
            _add_log(f"⏱️ 耗时: {total_time:.1f} 秒")
            _add_log(f"📤 输出: {out_path}")
            if dlhub_params:
                dlhub_params.save_logs(batch_state['logs'], 'inference', auto_save=True)
        
        threading.Thread(target=process, daemon=True).start()
        return _status_html(True, 0, 0, len(imgs)), _progress_html(0), _alert('info', f'开始处理 {len(imgs)} 张图片')
    
    def stop_batch():
        batch_state['should_stop'] = True
        return _alert('warning', '正在停止...')
    
    def refresh():
        results = batch_state['results']
        running = batch_state['is_running']
        progress = batch_state['progress']
        total = batch_state['total']
        
        if not results and not running:
            return _status_html(False,0,0,0), _progress_html(0), _stats_html(0,0,0,0), []
        
        total_texts = sum(r['num_texts'] for r in results)
        total_time = sum(r['time_ms'] for r in results) / 1000
        avg_ms = np.mean([r['time_ms'] for r in results]) if results else 0
        table = [[r['filename'], r['num_texts'], f"{r['time_ms']:.0f}", r['status']] for r in results[-50:]]
        
        return _status_html(running, progress, len(results), total), _progress_html(progress), _stats_html(len(results), total_texts, avg_ms, total_time), table
    
    input_folder.change(validate_input, [input_folder], [input_stats])
    start_btn.click(start_batch, [input_folder, use_gpu, save_vis, output_folder, export_format], [status_html, progress_html, export_status])
    stop_btn.click(stop_batch, outputs=[export_status])
    refresh_btn.click(refresh, outputs=[status_html, progress_html, stats_html, results_table])
    
    timer = gr.Timer(value=3.0)
    timer.tick(refresh, outputs=[status_html, progress_html, stats_html, results_table])
