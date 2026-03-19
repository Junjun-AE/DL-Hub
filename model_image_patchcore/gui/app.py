# -*- coding: utf-8 -*-
"""
PatchCore 工业级异常检测系统 - 优化版 v2.1

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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 添加父目录以便导入dlhub_params
sys.path.insert(0, str(PROJECT_ROOT.parent))

from gui.theme import CUSTOM_CSS, create_custom_theme


# ==================== DL-Hub 集成 ====================
def init_dlhub_adapter():
    """初始化 DL-Hub 适配器"""
    try:
        dlhub_path = PROJECT_ROOT.parent / 'dlhub_project' / 'dlhub'
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


# 全局变量：存储文件夹名到完整路径的映射
_folder_path_map = {}
_model_path_map = {}


def scan_model_folders():
    """扫描output目录下的训练文件夹，只显示文件夹名"""
    global _folder_path_map
    import gradio as gr
    _folder_path_map = {}
    
    base_path = get_output_dir()
    if not base_path.exists():
        return gr.update(choices=[], value=None)
    
    folders = []
    for item in base_path.iterdir():
        if item.is_dir():
            # 检查是否包含.pkg文件或patchcore相关文件
            has_pkg = any(item.rglob('*.pkg'))
            has_model = any(item.rglob('model_*.pth')) or any(item.rglob('memory_bank.pt'))
            if has_pkg or has_model:
                folder_name = item.name
                _folder_path_map[folder_name] = str(item)
                folders.append(folder_name)
    
    folders = sorted(folders, reverse=True)
    if folders:
        return gr.update(choices=folders, value=folders[0])
    return gr.update(choices=[], value=None)


def scan_models_in_selected_folder(folder_name: str):
    """扫描选定文件夹下的模型文件，只显示相对路径"""
    global _model_path_map
    import gradio as gr
    _model_path_map = {}
    
    if not folder_name:
        return gr.update(choices=[], value=None)
    
    # 从映射获取完整路径
    folder_path = _folder_path_map.get(folder_name)
    if not folder_path:
        return gr.update(choices=[], value=None)
    
    folder = Path(folder_path)
    if not folder.exists():
        return gr.update(choices=[], value=None)
    
    # 扫描.pkg文件和模型目录
    model_items = []
    
    # 扫描.pkg文件
    for pkg in folder.rglob('*.pkg'):
        rel_path = str(pkg.relative_to(folder))
        _model_path_map[rel_path] = str(pkg)
        model_items.append(rel_path)
    
    # 如果没有.pkg文件，扫描包含memory_bank.pt的目录
    if not model_items:
        for mb in folder.rglob('memory_bank.pt'):
            model_dir = mb.parent
            rel_path = str(model_dir.relative_to(folder))
            if rel_path == '.':
                rel_path = folder.name
            _model_path_map[rel_path] = str(model_dir)
            model_items.append(rel_path)
    
    model_items = sorted(model_items, reverse=True)
    if model_items:
        return gr.update(choices=model_items, value=model_items[0])
    return gr.update(choices=[], value=None)


def get_full_model_path(rel_path: str) -> str:
    """根据相对路径获取完整模型路径"""
    import os
    if not rel_path:
        return ""
    # 先检查映射
    if rel_path in _model_path_map:
        return _model_path_map[rel_path]
    # 如果是完整路径则直接返回
    if os.path.isabs(rel_path) or os.path.exists(rel_path):
        return rel_path
    return rel_path


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

def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            html = f'<div style="background:#e6f4ea;border-radius:8px;padding:14px;border:1px solid rgba(52,168,83,0.2);"><div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;"><span>✅</span><span style="font-weight:600;color:#1e8e3e;">GPU可用({n}个)</span></div>'
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                html += f'<div style="background:white;padding:8px 12px;border-radius:6px;margin-bottom:4px;font-size:12px;"><strong>GPU{i}:</strong> {p.name} ({p.total_memory/(1024**3):.1f}GB)</div>'
            return html + '</div>'
        return '<div style="background:#fef7e0;border-radius:8px;padding:14px;border:1px solid rgba(251,188,4,0.3);"><span>⚠️</span> GPU不可用，将使用CPU</div>'
    except: return '<div style="background:#f8f9fa;border-radius:8px;padding:14px;color:#5f6368;">无法检测GPU状态</div>'

def check_dependencies():
    deps = [('torch','PyTorch'),('torchvision','TorchVision'),('numpy','NumPy'),('sklearn','Scikit-learn'),('cv2','OpenCV'),('PIL','Pillow'),('faiss','Faiss'),('timm','TIMM')]
    html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">'
    for mod, name in deps:
        try:
            __import__('sklearn' if mod=='sklearn' else mod)
            html += f'<div style="background:#e6f4ea;padding:8px 10px;border-radius:5px;display:flex;align-items:center;gap:6px;"><span style="color:#1e8e3e;font-weight:600;">✓</span><span style="font-size:12px;">{name}</span></div>'
        except: html += f'<div style="background:#fce8e6;padding:8px 10px;border-radius:5px;display:flex;align-items:center;gap:6px;"><span style="color:#d93025;font-weight:600;">✗</span><span style="font-size:12px;">{name}</span></div>'
    return html + '</div>'

def create_settings_panel():
    def _sec(i, t): return f'<div style="display:flex;align-items:center;gap:8px;margin:16px 0 10px;padding-bottom:6px;border-bottom:1px solid #e8eaed;"><span style="font-size:18px;">{i}</span><span style="font-size:14px;font-weight:600;color:#202124;">{t}</span></div>'
    with gr.Row():
        with gr.Column():
            gr.HTML(_sec("🖥️", "硬件配置")); gr.HTML(get_gpu_info())
            gr.Dropdown(label="默认计算设备", choices=["auto","cuda:0","cuda:1","cpu"], value="auto")
            gr.HTML(_sec("📂", "路径配置")); gr.Textbox(label="默认输出目录", value="./output"); gr.Textbox(label="缓存目录", value="./cache")
            gr.HTML(_sec("🔧", "高级选项"))
            with gr.Row(): gr.Checkbox(label="启用TensorRT加速", value=True); gr.Checkbox(label="启用FP16推理", value=True)
            gr.Slider(label="数据加载进程数", minimum=0, maximum=16, value=4, step=1); gr.Button("💾 保存设置", variant="primary")
        with gr.Column():
            gr.HTML(_sec("📦", "依赖状态")); gr.HTML(check_dependencies())
            gr.HTML(_sec("🧹", "缓存管理")); gr.HTML('<div style="background:#f8f9fa;padding:14px;border-radius:6px;font-size:12px;color:#5f6368;">缓存大小: 计算中...</div>')
            with gr.Row(): gr.Button("🔄 刷新", size="sm"); gr.Button("🗑️ 清除缓存", size="sm", variant="stop")
            gr.HTML(_sec("📝", "日志设置")); gr.Dropdown(label="日志级别", choices=["DEBUG","INFO","WARNING","ERROR"], value="INFO")

def create_help_panel():
    gr.HTML('''<div style="max-width:100%;">
<div style="background:linear-gradient(135deg,#e8f0fe,#d2e3fc);border-radius:10px;padding:20px;margin-bottom:20px;border:1px solid rgba(26,115,232,0.2);">
<h2 style="color:#1967d2;margin:0 0 12px;font-size:18px;">🚀 快速开始</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
<div style="background:white;padding:14px;border-radius:8px;text-align:center;"><div style="font-size:28px;margin-bottom:6px;">1️⃣</div><div style="font-weight:600;color:#202124;margin-bottom:3px;font-size:13px;">准备数据</div><div style="font-size:11px;color:#5f6368;">良品放入good/</div></div>
<div style="background:white;padding:14px;border-radius:8px;text-align:center;"><div style="font-size:28px;margin-bottom:6px;">2️⃣</div><div style="font-weight:600;color:#202124;margin-bottom:3px;font-size:13px;">配置参数</div><div style="font-size:11px;color:#5f6368;">数据配置页面</div></div>
<div style="background:white;padding:14px;border-radius:8px;text-align:center;"><div style="font-size:28px;margin-bottom:6px;">3️⃣</div><div style="font-weight:600;color:#202124;margin-bottom:3px;font-size:13px;">开始训练</div><div style="font-size:11px;color:#5f6368;">查看训练状态</div></div>
<div style="background:white;padding:14px;border-radius:8px;text-align:center;"><div style="font-size:28px;margin-bottom:6px;">4️⃣</div><div style="font-weight:600;color:#202124;margin-bottom:3px;font-size:13px;">评估检测</div><div style="font-size:11px;color:#5f6368;">批量检测导出</div></div>
</div></div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
<div style="background:white;border-radius:10px;padding:18px;border:1px solid #e8eaed;">
<h3 style="color:#202124;margin:0 0 12px;font-size:15px;">📁 数据集格式</h3>
<div style="background:#1e1e1e;border-radius:6px;padding:12px;font-family:monospace;font-size:12px;color:#d4d4d4;"><pre style="margin:0;">my_dataset/
├── good/      # 良品图片(必需)
│   ├── img001.jpg
│   └── ...
└── defect/    # 异常图片(可选)</pre></div>
</div>

<div style="background:white;border-radius:10px;padding:18px;border:1px solid #e8eaed;">
<h3 style="color:#202124;margin:0 0 12px;font-size:15px;">🎚️ 阈值指南</h3>
<table style="width:100%;border-collapse:collapse;font-size:12px;">
<tr style="background:#f8f9fa;"><th style="padding:8px;text-align:left;">阈值</th><th style="padding:8px;text-align:left;">模式</th><th style="padding:8px;text-align:left;">场景</th></tr>
<tr><td style="padding:8px;border-bottom:1px solid #e8eaed;"><strong style="color:#34a853;">30-40</strong></td><td style="padding:8px;border-bottom:1px solid #e8eaed;">高召回</td><td style="padding:8px;border-bottom:1px solid #e8eaed;">安全关键</td></tr>
<tr><td style="padding:8px;border-bottom:1px solid #e8eaed;"><strong style="color:#1a73e8;">40-55</strong></td><td style="padding:8px;border-bottom:1px solid #e8eaed;">平衡</td><td style="padding:8px;border-bottom:1px solid #e8eaed;">日常使用</td></tr>
<tr><td style="padding:8px;border-bottom:1px solid #e8eaed;"><strong style="color:#f9ab00;">55-70</strong></td><td style="padding:8px;border-bottom:1px solid #e8eaed;">高精度</td><td style="padding:8px;border-bottom:1px solid #e8eaed;">减少误报</td></tr>
<tr><td style="padding:8px;"><strong style="color:#ea4335;">70+</strong></td><td style="padding:8px;">极严格</td><td style="padding:8px;">仅明显异常</td></tr>
</table>
</div>
</div>

<div style="background:#fef7e0;border-radius:10px;padding:18px;margin-top:16px;border:1px solid rgba(251,188,4,0.3);">
<h3 style="color:#b06000;margin:0 0 12px;font-size:15px;">🔧 常见问题</h3>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
<div style="background:white;padding:12px;border-radius:6px;"><div style="font-weight:600;color:#202124;margin-bottom:6px;font-size:12px;">Q: 内存不足？</div><div style="font-size:11px;color:#5f6368;">降低图像尺寸、减小采样率、启用FP16</div></div>
<div style="background:white;padding:12px;border-radius:6px;"><div style="font-weight:600;color:#202124;margin-bottom:6px;font-size:12px;">Q: 速度慢？</div><div style="font-size:11px;color:#5f6368;">用ResNet18、启用TensorRT</div></div>
<div style="background:white;padding:12px;border-radius:6px;"><div style="font-weight:600;color:#202124;margin-bottom:6px;font-size:12px;">Q: 误报多？</div><div style="font-size:11px;color:#5f6368;">提高阈值、增加训练数据</div></div>
</div>
</div>
</div>''')

def create_app():
    theme = create_custom_theme()
    with gr.Blocks(title="PatchCore 异常检测系统", theme=theme, css=CUSTOM_CSS) as app:        
        # 居中的标题
        gr.HTML('''<div style="background:linear-gradient(135deg,#1a73e8,#1557b0);border-radius:12px;padding:24px 32px;margin-bottom:20px;box-shadow:0 4px 16px rgba(26,115,232,0.3);">
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
    <h1 style="color:white;font-size:26px;font-weight:700;margin:0 0 8px;display:flex;align-items:center;gap:12px;">
        🔬 PatchCore 工业级异常检测系统
        <span style="background:rgba(255,255,255,0.2);padding:4px 14px;border-radius:20px;font-size:12px;font-weight:500;">✓ v2.1</span>
    </h1>
    <p style="color:rgba(255,255,255,0.9);font-size:14px;margin:0;">
        基于深度学习的无监督异常检测 | 仅需良品数据 | TensorRT加速
    </p>
</div>
</div>''')
        
        # 独立的顶层Tab - 不再嵌套在"模型训练"下
        with gr.Tabs():
            with gr.Tab("📁 数据配置", id="config"):
                from gui.components.config_panel import create_config_panel
                create_config_panel()
            
            with gr.Tab("📊 训练状态", id="status"):
                from gui.components.train_status_panel import create_status_panel
                create_status_panel()
            
            with gr.Tab("📈 评估验证", id="eval"):
                from gui.components.eval_panel import create_eval_panel
                create_eval_panel()
            
            with gr.Tab("🔍 批量检测", id="inference"):
                from gui.components.inference_panel import create_inference_panel
                create_inference_panel()
            
            with gr.Tab("⚙️ 系统设置", id="settings"):
                create_settings_panel()
            
            with gr.Tab("❓ 帮助文档", id="help"):
                create_help_panel()
        
        gr.HTML('<div style="text-align:center;padding:16px;color:#5f6368;font-size:12px;margin-top:16px;border-top:1px solid #e8eaed;">PatchCore v2.1 | Powered by Anomalib & Gradio</div>')
    return app

def main():
    # 【关键】绕过系统代理，防止Gradio 6.0自检被代理拦截
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
    os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
    
    # 解析命令行参数（DL-Hub 兼容）
    parser = argparse.ArgumentParser(description='PatchCore 异常检测系统')
    parser.add_argument('--task-dir', type=str, default=None, help='DL-Hub 任务目录')
    parser.add_argument('--port', type=int, default=7860, help='Gradio 服务端口')
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
    
    app.launch(server_name="127.0.0.1", server_port=launch_port, share=False, inbrowser=launch_inbrowser)

if __name__ == "__main__":
    main()
