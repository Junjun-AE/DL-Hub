# -*- coding: utf-8 -*-
"""
OCR Toolkit GUI - Gradio 6.0 兼容版

已集成 DL-Hub 支持：
- 支持 --task-dir 参数指定任务目录
- 支持 --port 参数指定端口
- 自动保存/加载UI参数
"""

import gradio as gr
import sys
import os
import argparse
from pathlib import Path

# 添加父目录以便导入dlhub_params
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ==================== DL-Hub 集成 ====================
def init_dlhub_adapter():
    """初始化 DL-Hub 适配器"""
    try:
        dlhub_path = Path(__file__).parent.parent.parent / 'dlhub_project' / 'dlhub'
        if dlhub_path.exists():
            sys.path.insert(0, str(dlhub_path.parent))
        
        from dlhub.app_adapters.base_adapter import get_adapter
        adapter = get_adapter(default_port=7861)
        print(f"[DL-Hub] 适配器已初始化，模式: {'DL-Hub' if adapter.is_dlhub_mode else '独立'}")
        return adapter
    except ImportError:
        print("[DL-Hub] 适配器未找到，以独立模式运行")
        return None
    except Exception as e:
        print(f"[DL-Hub] 初始化失败: {e}，以独立模式运行")
        return None


def init_dlhub_params():
    """初始化 DL-Hub 参数管理器（使用单例模式）"""
    try:
        from dlhub_params import get_dlhub_params
        params = get_dlhub_params()
        # 注意：日志已在get_dlhub_params()中打印，这里不重复
        return params
    except ImportError:
        print("[DL-Hub] 参数管理器未找到，参数不会持久化")
        return None
    except Exception as e:
        print(f"[DL-Hub] 参数管理器初始化失败: {e}")
        return None


dlhub_adapter = init_dlhub_adapter()
dlhub_params = init_dlhub_params()


def get_output_dir(default: str = './output') -> Path:
    """获取输出目录，优先使用 DL-Hub 任务目录"""
    if dlhub_params and dlhub_params.is_dlhub_mode:
        return dlhub_params.get_output_dir()
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode:
        return dlhub_adapter.get_output_dir()
    return Path(default)


def get_saved_param(key: str, default=None):
    """获取保存的参数值"""
    if dlhub_params:
        return dlhub_params.get(key, default)
    return default


def save_all_params(params_dict: dict) -> bool:
    """保存所有参数"""
    if dlhub_params:
        return dlhub_params.save(params_dict)
    return False

# Gradio 6.0: css 和 theme 参数需要传递给 launch() 方法
CUSTOM_CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 10px 20px !important; margin: 0 !important; }
.contain { max-width: 100% !important; }
.gr-button-primary { background: linear-gradient(135deg, #1a73e8, #4285f4) !important; }
"""


def create_app():
    with gr.Blocks(title="🔤 OCR Toolkit") as app:        
        gr.HTML("""
        <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1a73e8,#4285f4);border-radius:12px;margin-bottom:16px;">
            <h1 style="color:white;margin:0;font-size:26px;">🔤 OCR Toolkit</h1>
            <p style="color:rgba(255,255,255,0.9);margin:6px 0 0;font-size:14px;">文本检测与识别 | 支持TensorRT加速部署</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.TabItem("🚀 快速识别"):
                from .components.quick_ocr_panel import create_quick_ocr_panel
                create_quick_ocr_panel()
            
            with gr.TabItem("📁 批量处理"):
                from .components.batch_ocr_panel import create_batch_ocr_panel
                create_batch_ocr_panel()
            
            with gr.TabItem("📦 模型导出 (.pkg)"):
                from .components.export_panel import create_export_panel
                create_export_panel()
            
            with gr.TabItem("⚙️ 设置"):
                from .components.settings_panel import create_settings_panel
                create_settings_panel()
    
    return app

def main():
    # 【关键】绕过系统代理，防止Gradio 6.0自检被代理拦截
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
    os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    
    # 解析命令行参数（DL-Hub 兼容）
    parser = argparse.ArgumentParser(description='OCR Toolkit')
    parser.add_argument('--task-dir', type=str, default=None, help='DL-Hub 任务目录')
    parser.add_argument('--port', type=int, default=7861, help='Gradio 服务端口')
    args, _ = parser.parse_known_args()
    
    app = create_app()
    
    # 确定启动配置
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode:
        launch_port = dlhub_adapter.port
        launch_inbrowser = False
        print(f"[DL-Hub] 以 DL-Hub 模式启动，端口: {launch_port}")
    else:
        launch_port = args.port
        launch_inbrowser = True
        print(f"[独立模式] 启动端口: {launch_port}")
    
    app.launch(
        server_name="127.0.0.1", 
        server_port=launch_port, 
        share=False, 
        inbrowser=launch_inbrowser,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    main()
