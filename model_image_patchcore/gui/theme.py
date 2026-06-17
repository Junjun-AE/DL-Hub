# -*- coding: utf-8 -*-
"""PatchCore UI 主题系统 - 工业级专业UI设计"""

import gradio as gr

COLORS = {
    'primary': '#1a73e8', 'primary_dark': '#1557b0', 'primary_light': '#4285f4',
    'secondary': '#5f6368', 'success': '#34a853', 'success_light': '#e6f4ea',
    'warning': '#fbbc04', 'warning_light': '#fef7e0', 'danger': '#ea4335', 'danger_light': '#fce8e6',
    'info': '#4285f4', 'info_light': '#e8f0fe', 'bg_primary': '#ffffff', 'bg_secondary': '#f8f9fa',
    'border': '#dadce0', 'text_primary': '#202124', 'text_secondary': '#5f6368',
}

CUSTOM_CSS = """
/* 全局样式 - 自适应宽度 */
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 16px 24px !important; margin: 0 auto !important; }
.main { max-width: 100% !important; }
footer { display: none !important; }

/* 减少容器内边距 */
.contain { padding: 0 !important; }
.block { padding: 8px !important; }
.form { gap: 8px !important; }

/* Tab标签优化 */
.tabs { border: none !important; }
.tab-nav { background: #f8f9fa !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; margin-bottom: 16px !important; }
.tab-nav button { background: transparent !important; border: none !important; padding: 10px 20px !important; border-radius: 8px !important; font-weight: 500 !important; color: #5f6368 !important; transition: all 0.2s !important; }
.tab-nav button.selected { background: white !important; color: #1a73e8 !important; box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important; }
.tab-nav button:hover:not(.selected) { background: rgba(26,115,232,0.08) !important; color: #1a73e8 !important; }
.tabitem { padding: 0 !important; }

/* 按钮样式 */
.primary { background: linear-gradient(135deg, #1a73e8, #1557b0) !important; border: none !important; color: white !important; font-weight: 600 !important; padding: 10px 24px !important; border-radius: 8px !important; box-shadow: 0 2px 6px rgba(26,115,232,0.3) !important; }
.primary:hover { background: linear-gradient(135deg, #1557b0, #0d47a1) !important; box-shadow: 0 4px 12px rgba(26,115,232,0.4) !important; transform: translateY(-1px) !important; }
.secondary { background: white !important; border: 1.5px solid #dadce0 !important; color: #5f6368 !important; font-weight: 500 !important; padding: 10px 20px !important; border-radius: 8px !important; }
.secondary:hover { background: #f8f9fa !important; border-color: #1a73e8 !important; color: #1a73e8 !important; }
.stop { background: linear-gradient(135deg, #ea4335, #c5221f) !important; border: none !important; color: white !important; font-weight: 600 !important; }

/* 输入框优化 */
input[type="text"], textarea, .input-text { border: 1.5px solid #dadce0 !important; border-radius: 8px !important; padding: 10px 14px !important; font-size: 14px !important; transition: all 0.2s !important; }
input[type="text"]:focus, textarea:focus { border-color: #1a73e8 !important; box-shadow: 0 0 0 3px rgba(26,115,232,0.15) !important; outline: none !important; }

/* 滑块优化 */
input[type="range"] { accent-color: #1a73e8 !important; }
.slider { margin: 4px 0 !important; }

/* 下拉框优化 */
select, .dropdown { border: 1.5px solid #dadce0 !important; border-radius: 8px !important; padding: 10px 14px !important; }

/* 复选框优化 */
input[type="checkbox"] { accent-color: #1a73e8 !important; width: 18px !important; height: 18px !important; }

/* 分组容器优化 */
.group, .panel { background: #f8f9fa !important; border: 1px solid #e8eaed !important; border-radius: 10px !important; padding: 16px !important; }

/* Accordion优化 */
.accordion { border: 1px solid #e8eaed !important; border-radius: 10px !important; margin-bottom: 12px !important; }
.accordion > .label-wrap { background: #f8f9fa !important; padding: 12px 16px !important; font-weight: 600 !important; border-radius: 10px !important; }
.accordion[open] > .label-wrap { border-radius: 10px 10px 0 0 !important; }
.accordion > .wrap { padding: 16px !important; }

/* 代码块优化 */
.code-block, pre, .code { background: #1e1e1e !important; color: #d4d4d4 !important; border-radius: 8px !important; padding: 12px 16px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; line-height: 1.5 !important; }

/* 图片容器优化 */
.image-container { border-radius: 8px !important; overflow: hidden !important; border: 1px solid #e8eaed !important; }
.image-container img { object-fit: contain !important; }

/* Gallery优化 */
.gallery { gap: 8px !important; }
.gallery-item { border-radius: 8px !important; border: 1px solid #e8eaed !important; }

/* 数据表格优化 */
.dataframe { border: 1px solid #e8eaed !important; border-radius: 8px !important; overflow: hidden !important; }
.dataframe th { background: #f8f9fa !important; font-weight: 600 !important; padding: 12px !important; }
.dataframe td { padding: 10px 12px !important; border-bottom: 1px solid #e8eaed !important; }
.dataframe tr:hover td { background: #f8f9fa !important; }

/* Row和Column优化 - 减少间距 */
.row { gap: 16px !important; }
.column { gap: 8px !important; }

/* Label优化 */
label { font-weight: 500 !important; color: #202124 !important; font-size: 13px !important; margin-bottom: 4px !important; }
.info { font-size: 11px !important; color: #5f6368 !important; }

/* Plot优化 */
.plot-container { border-radius: 8px !important; border: 1px solid #e8eaed !important; padding: 8px !important; background: white !important; }

/* 滚动条美化 */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }

/* 动画 */
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }

/* 响应式布局 */
@media (max-width: 768px) {
    .gradio-container { padding: 12px !important; }
    .row { flex-direction: column !important; }
    .tab-nav button { padding: 8px 12px !important; font-size: 12px !important; }
}
"""

def create_custom_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#1a1a1a",
        block_background_fill="#f8f9fa",
        block_border_width="1px",
        block_border_color="#e8eaed",
        block_radius="10px",
        block_label_text_size="13px",
        block_title_text_size="14px",
        input_background_fill="#ffffff",
        input_border_color="#dadce0",
        input_border_width="1.5px",
        input_radius="8px",
        button_primary_background_fill="linear-gradient(135deg, #1a73e8, #1557b0)",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_border_color="#dadce0",
        checkbox_background_color="#ffffff",
        checkbox_border_color="#dadce0",
        slider_color="#1a73e8",
    )
