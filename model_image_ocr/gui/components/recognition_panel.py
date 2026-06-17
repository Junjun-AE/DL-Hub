# -*- coding: utf-8 -*-
"""文本识别面板 - 给定文本区域进行识别"""

import gradio as gr

def _sec(icon, title, desc=""):
    d = f'<span style="color:#5f6368;font-size:16px;margin-left:12px;">{desc}</span>' if desc else ''
    return f'<div style="display:flex;align-items:center;gap:14px;margin:28px 0 18px;padding-bottom:12px;border-bottom:2px solid #e8eaed;"><span style="font-size:28px;">{icon}</span><span style="font-size:22px;font-weight:700;color:#202124;">{title}{d}</span></div>'

def create_recognition_panel():
    """创建文本识别面板"""
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(_sec("📝", "识别配置"))
            
            with gr.Group():
                rec_model = gr.Dropdown(
                    label="识别模型",
                    choices=["PP-OCRv4 中英文", "PP-OCRv4 英文", "PP-OCRv3 中英文", "SVTR"],
                    value="PP-OCRv4 中英文"
                )
                
                language = gr.Dropdown(
                    label="识别语言",
                    choices=["中文+英文", "英文", "日文", "韩文"],
                    value="中文+英文"
                )
                
                use_space = gr.Checkbox(label="识别空格", value=True)
                use_gpu = gr.Checkbox(label="使用GPU", value=True)
            
            recognize_btn = gr.Button("📝 开始识别", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.HTML(_sec("📷", "输入文本图像"))
            gr.Markdown("*上传裁剪好的文本行图像，或使用检测面板裁剪*")
            
            input_images = gr.Gallery(label="文本图像", columns=4, height=200)
            
            gr.HTML(_sec("📊", "识别结果"))
            
            results_table = gr.Dataframe(
                headers=["序号", "识别文本", "置信度", "耗时(ms)"],
                label="",
                interactive=False
            )
            
            result_text = gr.Textbox(
                label="合并文本",
                lines=5
            )
