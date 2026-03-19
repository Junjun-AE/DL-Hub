# -*- coding: utf-8 -*-
"""快速OCR识别面板 - V6.1
修复：使用单例模式共享dlhub_params
"""

import gradio as gr
import numpy as np
from pathlib import Path
import time
import threading
import random
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

cv2 = None
Image = None
ImageDraw = None

def _lazy_import():
    global cv2, Image, ImageDraw
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if Image is None:
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        Image = _Image
        ImageDraw = _ImageDraw

def get_pretrained_models_dir():
    current_file = Path(__file__).resolve()
    # 从 components 目录往上 4 层到 Deep_learning_tools 根目录
    # components -> gui -> model_image_ocr -> Deep_learning_tools
    root_dir = current_file.parent.parent.parent.parent
    models_dir = root_dir / 'pretrained_model' / 'images_ocr'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

class OCREngineManager:
    _instance = None
    _lock = threading.Lock()
    _engine = None
    _engine_config = None
    _ocr_v3 = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_engine(self, use_angle_cls=False, lang='ch', use_gpu=True, force_reload=False):
        config = (use_angle_cls, lang, use_gpu)
        if self._engine is not None and self._engine_config == config and not force_reload:
            return self._engine
        with self._lock:
            if self._engine is None or force_reload:
                from paddleocr import PaddleOCR
                import paddleocr
                version = getattr(paddleocr, '__version__', '2.0.0')
                major_version = int(version.split('.')[0])
                self._ocr_v3 = major_version >= 3
                print(f"📦 PaddleOCR: {version}")
                if major_version >= 3:
                    self._engine = PaddleOCR(lang=lang, device='gpu' if use_gpu else 'cpu', 
                                            use_doc_orientation_classify=False, use_doc_unwarping=False)
                else:
                    self._engine = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu, show_log=False)
                self._engine_config = config
                print(f"✅ OCR引擎加载成功")
        return self._engine
    
    @property
    def is_loaded(self):
        return self._engine is not None
    
    @property
    def is_v3(self):
        return self._ocr_v3

ocr_manager = OCREngineManager()
ocr_state = {'last_result': None}

def _sec(icon, title):
    return f'<div style="display:flex;align-items:center;gap:10px;margin:20px 0 14px;padding-bottom:10px;border-bottom:2px solid #e8eaed;"><span style="font-size:24px;">{icon}</span><span style="font-size:18px;font-weight:700;color:#202124;">{title}</span></div>'

def _alert(t, msg):
    s = {'success':('#e6f4ea','#1e8e3e','✓'), 'error':('#fce8e6','#d93025','✗'), 'info':('#e8f0fe','#1967d2','ℹ'), 'warning':('#fef7e0','#b06000','!')}
    bg, c, i = s.get(t, s['info'])
    return f'<div style="background:{bg};color:{c};padding:14px 18px;border-radius:10px;font-size:15px;margin:10px 0;"><span style="font-weight:700;margin-right:10px;">{i}</span>{msg}</div>'

def _stats_html(num, total_ms, det_ms, rec_ms):
    return f'''<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
        <div style="background:#f8f9fa;padding:16px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:28px;font-weight:700;color:#1a73e8;">{num}</div><div style="font-size:13px;color:#5f6368;">文本数量</div></div>
        <div style="background:#f8f9fa;padding:16px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:28px;font-weight:700;color:#34a853;">{total_ms:.0f}</div><div style="font-size:13px;color:#5f6368;">总耗时(ms)</div></div>
        <div style="background:#f8f9fa;padding:16px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:28px;font-weight:700;color:#ea4335;">{det_ms:.0f}</div><div style="font-size:13px;color:#5f6368;">检测(ms)</div></div>
        <div style="background:#f8f9fa;padding:16px;border-radius:10px;text-align:center;border:1px solid #e8eaed;"><div style="font-size:28px;font-weight:700;color:#f59f00;">{rec_ms:.0f}</div><div style="font-size:13px;color:#5f6368;">识别(ms)</div></div>
    </div>'''

def _engine_html(status, name="PaddleOCR"):
    if status == 'ready': bg, color, icon = '#e6f4ea', '#1e8e3e', '✅'
    elif status == 'loading': bg, color, icon = '#e8f0fe', '#1a73e8', '⏳'
    else: bg, color, icon = '#f8f9fa', '#5f6368', '⚪'
    return f'<div style="background:{bg};border-radius:10px;padding:14px;display:flex;align-items:center;gap:10px;"><span style="font-size:20px;">{icon}</span><span style="font-size:16px;font-weight:600;color:{color};">{name}</span></div>'

def process_ocr_result_v3(result_obj, img_shape, rec_th=0.5):
    boxes, texts, scores = [], [], []
    img_h, img_w = img_shape[:2]
    polys = None
    for src in ['rec_polys', 'dt_polys']:
        try:
            p = result_obj[src] if hasattr(result_obj, 'get') else getattr(result_obj, src, None)
            if p is not None and len(p) > 0:
                polys = p
                break
        except Exception:
            pass
    if polys is None:
        return boxes, texts, scores
    rec_texts = result_obj.get('rec_texts') if hasattr(result_obj, 'get') else getattr(result_obj, 'rec_texts', None)
    rec_scores = result_obj.get('rec_scores') if hasattr(result_obj, 'get') else getattr(result_obj, 'rec_scores', None)
    for i, poly in enumerate(polys):
        try:
            text = rec_texts[i] if rec_texts and i < len(rec_texts) else ""
            score_val = float(rec_scores[i]) if rec_scores and i < len(rec_scores) else 1.0
            if score_val < rec_th:
                continue
            poly_arr = np.array(poly, dtype=np.float32)
            if poly_arr.ndim == 1:
                if len(poly_arr) == 4:
                    x1, y1, x2, y2 = poly_arr
                    poly_arr = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
                elif len(poly_arr) >= 8:
                    poly_arr = poly_arr[:8].reshape(4, 2)
                else:
                    continue
            elif poly_arr.ndim == 2:
                poly_arr = poly_arr[:4]
            if poly_arr.shape[0] < 4:
                continue
            poly_arr[:, 0] = np.clip(poly_arr[:, 0], 0, img_w - 1)
            poly_arr[:, 1] = np.clip(poly_arr[:, 1], 0, img_h - 1)
            boxes.append(poly_arr.astype(np.int32))
            texts.append(str(text))
            scores.append(score_val)
        except Exception:
            pass
    return boxes, texts, scores

def process_ocr_result_v2(results, img_shape, rec_th=0.5):
    boxes, texts, scores = [], [], []
    img_h, img_w = img_shape[:2]
    if not results or not results[0]:
        return boxes, texts, scores
    for item in results[0]:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box, text_info = item[0], item[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, score = text_info[0], float(text_info[1])
                    if score >= rec_th:
                        box_arr = np.array(box, dtype=np.int32)[:4]
                        box_arr[:, 0] = np.clip(box_arr[:, 0], 0, img_w - 1)
                        box_arr[:, 1] = np.clip(box_arr[:, 1], 0, img_h - 1)
                        boxes.append(box_arr)
                        texts.append(str(text))
                        scores.append(score)
        except Exception:
            pass
    return boxes, texts, scores

def draw_boxes_pil(image, boxes, style='outline'):
    _lazy_import()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    random.seed(0)
    for i, box in enumerate(boxes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pts = [(int(x), int(y)) for x, y in np.array(box, dtype=np.int32).tolist()]
        if style == 'fill':
            draw.polygon(pts, fill=color)
        else:
            pts.append(pts[0])
            draw.line(pts, fill=color, width=2)
        draw.text((pts[0][0], max(0, pts[0][1] - 15)), str(i + 1), fill=color)
    if style == 'fill':
        pil_img = Image.blend(Image.fromarray(image_rgb), pil_img, 0.5)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_quick_ocr_panel():
    # 加载保存的参数
    saved_params = {} if not dlhub_params else dlhub_params.get_section('quick_ocr')
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            gr.HTML(_sec("⚙️", "引擎配置"))
            engine_status = gr.HTML(_engine_html('ready' if ocr_manager.is_loaded else 'none'))
            with gr.Group():
                engine_type = gr.Radio(
                    ["PaddleOCR", "ONNX", "TensorRT"], 
                    value=saved_params.get('engine_type', "PaddleOCR"), 
                    label="推理引擎"
                )
                use_gpu = gr.Checkbox(label="GPU加速", value=saved_params.get('use_gpu', True))
                use_angle = gr.Checkbox(label="方向分类", value=saved_params.get('use_angle', False))
                load_btn = gr.Button("🔄 加载引擎", variant="primary")
            gr.HTML(_sec("🎛️", "识别参数"))
            with gr.Group():
                language = gr.Dropdown(
                    ["中文+英文", "英文", "日文"], 
                    value=saved_params.get('language', "中文+英文"), 
                    label="语言"
                )
                det_thresh = gr.Slider(0.1, 0.9, saved_params.get('det_thresh', 0.3), step=0.05, label="检测阈值")
                rec_thresh = gr.Slider(0.0, 1.0, saved_params.get('rec_thresh', 0.5), step=0.05, label="识别阈值")
                draw_style = gr.Radio(
                    ["轮廓线", "填充混合"], 
                    value=saved_params.get('draw_style', "轮廓线"), 
                    label="绘制样式"
                )
            gr.HTML(_sec("📁", "模型目录"))
            gr.Markdown(f"`{get_pretrained_models_dir()}`")
        with gr.Column(scale=2, min_width=400):
            gr.HTML(_sec("📷", "输入/输出"))
            with gr.Row():
                input_image = gr.Image(label="上传图片", type="numpy", height=320)
                result_image = gr.Image(label="检测结果", height=320)
            with gr.Row():
                ocr_btn = gr.Button("🔤 开始识别", variant="primary", size="lg", scale=3)
                clear_btn = gr.Button("🗑️ 清空", size="lg", scale=1)
            gr.HTML(_sec("📊", "统计"))
            stats_display = gr.HTML(_stats_html(0, 0, 0, 0))
        with gr.Column(scale=1, min_width=320):
            gr.HTML(_sec("📝", "识别结果"))
            result_text = gr.Textbox(label="文本内容", lines=12)
            gr.HTML(_sec("📋", "详细列表"))
            result_table = gr.Dataframe(headers=["序号", "文本", "置信度"], datatype=["number", "str", "str"], column_count=(3, "fixed"), interactive=False)
            with gr.Row():
                export_txt = gr.Button("📄 导出TXT", size="sm")
                export_json = gr.Button("📋 导出JSON", size="sm")
            export_status = gr.HTML("")
    
    def load_engine(engine_val, gpu_val, angle_val, lang_val, det_th_val, rec_th_val, draw_style_val):
        try:
            # 保存所有参数到DL-Hub
            if dlhub_params:
                current_params = dlhub_params.get_all()
                current_params['quick_ocr'] = {
                    'engine_type': engine_val,
                    'use_gpu': gpu_val,
                    'use_angle': angle_val,
                    'language': lang_val,
                    'det_thresh': det_th_val,
                    'rec_thresh': rec_th_val,
                    'draw_style': draw_style_val,
                }
                dlhub_params.save(current_params)
            
            if engine_val == "PaddleOCR":
                engine = ocr_manager.get_engine(use_angle_cls=angle_val, lang='ch', use_gpu=gpu_val, force_reload=True)
                return _engine_html('ready', 'PaddleOCR') if engine else _alert('error', '加载失败')
            return _alert('warning', f'{engine_val}需要先导出模型')
        except Exception as e:
            return _alert('error', f'加载失败: {e}')
    
    def run_ocr(image, det_th, rec_th, draw_style_val):
        global ocr_state
        _lazy_import()
        if image is None:
            return None, _stats_html(0,0,0,0), "", [], _alert('info', '请上传图片')
        try:
            engine = ocr_manager.get_engine()
        except Exception as e:
            return None, _stats_html(0,0,0,0), "", [], _alert('error', f'引擎失败: {e}')
        try:
            start = time.time()
            results = engine.predict(image) if ocr_manager.is_v3 else engine.ocr(image, cls=False)
            total_ms = (time.time() - start) * 1000
            if ocr_manager.is_v3 and results:
                boxes, texts, scores = process_ocr_result_v3(results[0], image.shape, rec_th)
            else:
                boxes, texts, scores = process_ocr_result_v2(results, image.shape, rec_th)
            style = 'fill' if draw_style_val == "填充混合" else 'outline'
            vis = draw_boxes_pil(image, boxes, style=style)
            ocr_state['last_result'] = {'texts': texts, 'scores': scores}
            table = [[i+1, t, f"{s:.2%}"] for i,(t,s) in enumerate(zip(texts, scores))]
            return vis, _stats_html(len(texts), total_ms, total_ms*0.4, total_ms*0.6), "\n".join(texts), table, _alert('success', f'识别完成: {len(texts)}处')
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, _stats_html(0,0,0,0), "", [], _alert('error', f'识别失败: {e}')
    
    def clear_all():
        return None, None, _stats_html(0,0,0,0), "", [], ""
    
    def do_export_txt():
        if not ocr_state.get('last_result'): return _alert('warning', '无结果')
        Path("ocr_result.txt").write_text("\n".join(ocr_state['last_result']['texts']), encoding='utf-8')
        return _alert('success', '已导出: ocr_result.txt')
    
    def do_export_json():
        import json
        if not ocr_state.get('last_result'): return _alert('warning', '无结果')
        r = ocr_state['last_result']
        Path("ocr_result.json").write_text(json.dumps([{'text':t,'score':float(s)} for t,s in zip(r['texts'],r['scores'])], ensure_ascii=False, indent=2), encoding='utf-8')
        return _alert('success', '已导出: ocr_result.json')
    
    load_btn.click(load_engine, [engine_type, use_gpu, use_angle, language, det_thresh, rec_thresh, draw_style], [engine_status])
    ocr_btn.click(run_ocr, [input_image, det_thresh, rec_thresh, draw_style], [result_image, stats_display, result_text, result_table, export_status])
    clear_btn.click(clear_all, outputs=[input_image, result_image, stats_display, result_text, result_table, export_status])
    export_txt.click(do_export_txt, outputs=[export_status])
    export_json.click(do_export_json, outputs=[export_status])
