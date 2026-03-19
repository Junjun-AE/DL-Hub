# -*- coding: utf-8 -*-
"""文本检测面板"""

import gradio as gr

def _sec(icon, title, desc=""):
    d = f'<span style="color:#5f6368;font-size:16px;margin-left:12px;">{desc}</span>' if desc else ''
    return f'<div style="display:flex;align-items:center;gap:14px;margin:28px 0 18px;padding-bottom:12px;border-bottom:2px solid #e8eaed;"><span style="font-size:28px;">{icon}</span><span style="font-size:22px;font-weight:700;color:#202124;">{title}{d}</span></div>'

def create_detection_panel():
    """创建文本检测面板 - 仅检测不识别"""
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(_sec("🔍", "检测配置"))
            
            with gr.Group():
                det_model = gr.Dropdown(
                    label="检测模型",
                    choices=["PP-OCRv4 检测", "PP-OCRv3 检测", "DBNet++"],
                    value="PP-OCRv4 检测"
                )
                
                det_thresh = gr.Slider(label="检测阈值", minimum=0.1, maximum=0.9, value=0.3, step=0.05)
                box_thresh = gr.Slider(label="文本框阈值", minimum=0.1, maximum=0.9, value=0.5, step=0.05)
                min_area = gr.Slider(label="最小面积", minimum=10, maximum=500, value=50, step=10)
                
                use_gpu = gr.Checkbox(label="使用GPU", value=True)
            
            detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.HTML(_sec("📷", "检测结果"))
            
            with gr.Row():
                input_image = gr.Image(label="输入图像", type="numpy", height=350)
                result_image = gr.Image(label="检测结果", height=350)
            
            gr.HTML(_sec("📊", "检测统计"))
            
            stats_html = gr.HTML('''
                <div style="background:#f8f9fa;border-radius:14px;padding:22px;border:1px solid #e8eaed;">
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
                        <div style="background:white;padding:18px;border-radius:10px;text-align:center;">
                            <div style="font-size:32px;font-weight:700;color:#1a73e8;">0</div>
                            <div style="font-size:15px;color:#5f6368;">检测框数量</div>
                        </div>
                        <div style="background:white;padding:18px;border-radius:10px;text-align:center;">
                            <div style="font-size:32px;font-weight:700;color:#34a853;">0</div>
                            <div style="font-size:15px;color:#5f6368;">检测耗时(ms)</div>
                        </div>
                        <div style="background:white;padding:18px;border-radius:10px;text-align:center;">
                            <div style="font-size:32px;font-weight:700;color:#ea4335;">0</div>
                            <div style="font-size:15px;color:#5f6368;">平均面积</div>
                        </div>
                    </div>
                </div>
            ''')
            
            gr.HTML(_sec("📋", "检测框列表"))
            boxes_table = gr.Dataframe(
                headers=["序号", "坐标", "面积", "宽高比"],
                label="",
                interactive=False
            )
