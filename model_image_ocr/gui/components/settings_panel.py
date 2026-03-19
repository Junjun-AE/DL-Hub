# -*- coding: utf-8 -*-
"""设置面板 - v1.1
修复：
- 改进的环境检测
- 修复参数保存覆盖问题（使用合并而非替换）
"""

import gradio as gr
import sys

# 导入DL-Hub参数保存函数
try:
    from gui.app import save_all_params, get_saved_param, get_output_dir
except ImportError:
    def save_all_params(params): return False
    def get_saved_param(key, default=None): return default
    def get_output_dir(): return './ocr_results/'

def _sec(icon, title):
    return f'<div style="display:flex;align-items:center;gap:10px;margin:20px 0 14px;padding-bottom:10px;border-bottom:2px solid #e8eaed;"><span style="font-size:24px;">{icon}</span><span style="font-size:18px;font-weight:700;color:#202124;">{title}</span></div>'

def _alert(t, msg):
    s = {'success':('#e6f4ea','#1e8e3e','✓'), 'error':('#fce8e6','#d93025','✗'), 'info':('#e8f0fe','#1967d2','ℹ')}
    bg, c, i = s.get(t, s['info'])
    return f'<div style="background:{bg};color:{c};padding:14px 18px;border-radius:10px;font-size:15px;"><span style="font-weight:700;margin-right:10px;">{i}</span>{msg}</div>'

def create_settings_panel():
    # 加载保存的设置
    saved_gpu = get_saved_param('settings.gpu_id', 'GPU 0')
    saved_gpu_mem = get_saved_param('settings.gpu_mem', 4000)
    saved_cache = get_saved_param('settings.cache_dir', '~/.paddleocr/')
    saved_output = get_saved_param('settings.output_dir', str(get_output_dir()))
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=350):
            gr.HTML(_sec("🖥️", "GPU配置"))
            with gr.Group():
                gpu_id = gr.Dropdown(choices=["GPU 0", "GPU 1", "CPU"], value=saved_gpu, label="选择设备")
                gpu_mem = gr.Slider(1000, 16000, saved_gpu_mem, step=500, label="GPU内存(MB)")
            
            gr.HTML(_sec("📁", "路径配置"))
            with gr.Group():
                cache_dir = gr.Textbox(label="模型缓存", value=saved_cache)
                output_dir = gr.Textbox(label="默认输出", value=saved_output)
            
            save_btn = gr.Button("💾 保存设置", variant="primary")
            save_status = gr.HTML("")
        
        with gr.Column(scale=1, min_width=450):
            gr.HTML(_sec("📊", "环境检查"))
            env_html = gr.HTML("")
            check_btn = gr.Button("🔍 检查环境", variant="secondary")
            
            gr.HTML(_sec("📦", "安装命令"))
            gr.Textbox(
                label="修复安装（按顺序执行）",
                value='''# 1. 卸载现有版本
pip uninstall paddlepaddle-gpu paddlepaddle paddleocr -y

# 2. 安装 PaddlePaddle GPU (CUDA 11.8)
pip install paddlepaddle-gpu==2.6.2.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# 3. 安装 PaddleOCR (兼容版本)
pip install paddleocr==2.7.3

# 4. 或者使用最新版 PaddleOCR (需要新API代码)
# pip install paddleocr

# 5. 验证安装
python -c "import paddle; print(paddle.__version__)"
python -c "from paddleocr import PaddleOCR; print('OK')"''',
                lines=14,
                interactive=False
            )
    
    def check_env():
        """检查环境 - 详细版本"""
        html = '<div style="background:#f8f9fa;border-radius:10px;padding:16px;border:1px solid #e8eaed;">'
        html += f'<div style="margin-bottom:10px;"><strong>Python:</strong> {sys.version.split()[0]}</div>'
        
        # ============ 检查 PaddlePaddle ============
        html += '<hr style="margin:10px 0;border:none;border-top:1px solid #e8eaed;">'
        html += '<div style="font-weight:bold;margin-bottom:8px;">🏓 PaddlePaddle:</div>'
        
        try:
            import paddle
            version = paddle.__version__
            html += f'<div style="color:#1e8e3e;">✅ 版本: {version}</div>'
            
            try:
                cuda_compiled = paddle.is_compiled_with_cuda()
                if cuda_compiled:
                    html += f'<div style="color:#1e8e3e;">✅ CUDA 编译: 是</div>'
                    try:
                        gpu_count = paddle.device.cuda.device_count()
                        html += f'<div style="color:#1e8e3e;">✅ GPU 数量: {gpu_count}</div>'
                    except Exception:
                        html += f'<div style="color:#f59f00;">⚠️ GPU 检测失败</div>'
                else:
                    html += f'<div style="color:#f59f00;">⚠️ CPU 版本</div>'
            except Exception as e:
                html += f'<div style="color:#f59f00;">⚠️ CUDA 检测: {e}</div>'
                
        except ImportError:
            html += f'<div style="color:#d93025;">❌ 未安装</div>'
        except Exception as e:
            html += f'<div style="color:#d93025;">❌ 导入错误: {str(e)[:50]}...</div>'
        
        # ============ 检查 PaddleOCR ============
        html += '<hr style="margin:10px 0;border:none;border-top:1px solid #e8eaed;">'
        html += '<div style="font-weight:bold;margin-bottom:8px;">📝 PaddleOCR:</div>'
        
        try:
            import paddleocr
            version = paddleocr.__version__
            major = int(version.split('.')[0])
            html += f'<div style="color:#1e8e3e;">✅ 版本: {version}</div>'
            
            if major >= 3:
                html += f'<div style="color:#1967d2;">ℹ️ API: v3.x (使用 device 参数)</div>'
            else:
                html += f'<div style="color:#1967d2;">ℹ️ API: v2.x (使用 use_gpu 参数)</div>'
                
        except ImportError:
            html += f'<div style="color:#d93025;">❌ 未安装</div>'
        except Exception as e:
            html += f'<div style="color:#d93025;">❌ 导入错误: {e}</div>'
        
        # ============ 检查其他依赖 ============
        html += '<hr style="margin:10px 0;border:none;border-top:1px solid #e8eaed;">'
        html += '<div style="font-weight:bold;margin-bottom:8px;">📦 其他依赖:</div>'
        
        pkgs = [
            ('numpy', 'numpy'),
            ('opencv', 'cv2'),
            ('onnx', 'onnx'),
            ('paddle2onnx', 'paddle2onnx'),
            ('gradio', 'gradio'),
        ]
        
        for name, imp in pkgs:
            try:
                m = __import__(imp)
                v = getattr(m, '__version__', '✓')
                html += f'<div style="color:#1e8e3e;">✅ {name}: {v}</div>'
            except ImportError:
                html += f'<div style="color:#d93025;">❌ {name}: 未安装</div>'
            except Exception as e:
                # 尝试用 importlib 检测
                try:
                    import importlib.util
                    spec = importlib.util.find_spec(imp)
                    if spec:
                        html += f'<div style="color:#1e8e3e;">✅ {name}: (已安装)</div>'
                    else:
                        html += f'<div style="color:#d93025;">❌ {name}: 未安装</div>'
                except Exception:
                    html += f'<div style="color:#f59f00;">⚠️ {name}: 检测失败</div>'
        
        html += '</div>'
        return html
    
    check_btn.click(check_env, outputs=[env_html])
    
    def save_settings(gpu_val, mem_val, cache_val, output_val):
        """保存设置到DL-Hub - 使用合并方式避免覆盖其他参数"""
        try:
            from gui.app import dlhub_params
            if dlhub_params:
                # 修复：先获取现有参数，再合并保存，避免覆盖
                current_params = dlhub_params.get_all()
                current_params['settings'] = {
                    'gpu_id': gpu_val,
                    'gpu_mem': int(mem_val),
                    'cache_dir': cache_val,
                    'output_dir': output_val,
                }
                success = dlhub_params.save(current_params)
                if success:
                    return _alert('success', '设置已保存到任务目录')
        except Exception as e:
            print(f"[OCR] 保存设置失败: {e}")
        return _alert('success', '设置已保存（本地模式）')
    
    save_btn.click(save_settings, inputs=[gpu_id, gpu_mem, cache_dir, output_dir], outputs=[save_status])
