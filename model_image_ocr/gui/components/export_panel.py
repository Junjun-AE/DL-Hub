# -*- coding: utf-8 -*-
"""
OCR模型导出面板 - V7.0

重大更新:
1. 适配工业推理工具集 - 只支持动态batch，不支持动态尺寸
2. 完善config.json配置格式，与推理工具集兼容
3. 修复ONNX导出使用固定输入尺寸
4. TensorRT/OpenVINO导出使用固定输入尺寸

设计原则:
- 检测模型: 固定尺寸 (H, W)，batch动态
- 识别模型: 固定尺寸 (H, W)，batch动态
- 推理时通过预处理resize到固定尺寸
"""

import gradio as gr
import os
import sys
import json
import shutil
import tempfile
import zipfile
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime

# 添加父目录以便导入dlhub_params
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 使用单例模式获取DL-Hub参数管理器
dlhub_params = None
try:
    from dlhub_params import get_dlhub_params
    dlhub_params = get_dlhub_params()
except Exception:
    pass


# ============ 版本信息 ============
EXPORT_VERSION = "7.0.0"
CONFIG_VERSION = "2.0"  # config.json 格式版本


def get_default_export_dir():
    """获取默认导出目录，使用DL-Hub任务的output目录"""
    if dlhub_params and dlhub_params.is_dlhub_mode:
        return str(dlhub_params.get_output_dir())
    return './output'


def _sec(icon, title):
    return f'<div style="display:flex;align-items:center;gap:10px;margin:20px 0 14px;padding-bottom:10px;border-bottom:2px solid #e8eaed;"><span style="font-size:24px;">{icon}</span><span style="font-size:18px;font-weight:700;color:#202124;">{title}</span></div>'


def get_pretrained_models_dir():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent.parent.parent.parent
    models_dir = root_dir / 'pretrained_model' / 'images_ocr'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def check_frameworks():
    frameworks = {}
    try:
        import tensorrt as trt
        frameworks['tensorrt'] = {'ok': True, 'ver': trt.__version__}
    except Exception:
        frameworks['tensorrt'] = {'ok': False, 'ver': None}
    try:
        from openvino.runtime import Core
        frameworks['openvino'] = {'ok': True, 'ver': 'installed'}
    except Exception:
        frameworks['openvino'] = {'ok': False, 'ver': None}
    try:
        import paddle2onnx
        frameworks['paddle2onnx'] = {'ok': True, 'ver': paddle2onnx.__version__}
    except Exception:
        frameworks['paddle2onnx'] = {'ok': False, 'ver': None}
    return frameworks


def find_models():
    models = {'det': [], 'rec': []}
    local_dir = get_pretrained_models_dir()
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".paddlex", "official_models")
    
    search_paths = [(str(local_dir), "本地"), (cache_dir, "缓存")]
    found_names = set()
    
    for base_dir, source in search_paths:
        print(f"[find_models] 搜索: {base_dir}")
        if not os.path.exists(base_dir):
            continue
        try:
            for name in os.listdir(base_dir):
                model_path = os.path.join(base_dir, name)
                if not os.path.isdir(model_path):
                    continue
                name_lower = name.lower()
                if 'det' in name_lower:
                    model_type = 'det'
                elif 'rec' in name_lower:
                    model_type = 'rec'
                else:
                    continue
                files = os.listdir(model_path)
                if 'inference.pdiparams' not in files:
                    continue
                if name in found_names:
                    continue
                found_names.add(name)
                models[model_type].append({
                    'name': name, 'path': model_path, 'source': source,
                    'display': f"{name} [{source}]"
                })
                print(f"[find_models]   ✓ {name} [{source}]")
        except Exception as e:
            print(f"[find_models] 错误: {e}")
    print(f"[find_models] 结果: det={len(models['det'])}, rec={len(models['rec'])}")
    return models


def download_models():
    logs = []
    logs.append("╔" + "═" * 58 + "╗")
    logs.append("║" + "下载 OCR 模型".center(54) + "║")
    logs.append("╚" + "═" * 58 + "╝")
    local_dir = get_pretrained_models_dir()
    logs.append(f"\n📁 目标目录: {local_dir}")
    try:
        from paddleocr import PaddleOCR
        import numpy as np
        logs.append("\n[1] 初始化 PaddleOCR...")
        ocr = PaddleOCR(lang='ch', device='cpu', use_doc_orientation_classify=False, use_doc_unwarping=False)
        logs.append("    ✓ 初始化成功")
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr.predict(dummy)
        logs.append("    ✓ 验证通过")
        logs.append("\n[2] 复制模型...")
        home = os.path.expanduser("~")
        cache_dir = os.path.join(home, ".paddlex", "official_models")
        if os.path.exists(cache_dir):
            for name in os.listdir(cache_dir):
                src = os.path.join(cache_dir, name)
                dst = os.path.join(local_dir, name)
                if not os.path.isdir(src):
                    continue
                if 'det' in name.lower() or 'rec' in name.lower():
                    if os.path.exists(dst):
                        logs.append(f"    - {name} (已存在)")
                    else:
                        shutil.copytree(src, dst)
                        logs.append(f"    ✓ {name}")
        logs.append(f"\n[3] 搜索结果:")
        models = find_models()
        for m in models['det']:
            logs.append(f"    🔍 {m['display']}")
        for m in models['rec']:
            logs.append(f"    📝 {m['display']}")
        logs.append(f"\n✅ 完成！请点击「🔄 刷新」")
    except Exception as e:
        logs.append(f"\n❌ 错误: {e}")
        import traceback
        logs.append(traceback.format_exc())
    return "\n".join(logs)


def compute_file_md5(filepath):
    """计算文件MD5"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


# ============ ONNX 导出 (固定尺寸，动态batch) ============

def export_to_onnx(model_dir, output_path, input_shape, logs):
    """
    导出 ONNX - 固定尺寸，动态batch
    
    Args:
        model_dir: Paddle模型目录
        output_path: 输出ONNX路径
        input_shape: 输入形状 [C, H, W] (不含batch)
        logs: 日志列表
    
    Returns:
        bool: 是否成功
    """
    pdmodel_path = os.path.join(model_dir, 'inference.pdmodel')
    json_path = os.path.join(model_dir, 'inference.json')
    pdiparams_path = os.path.join(model_dir, 'inference.pdiparams')
    
    if not os.path.exists(pdiparams_path):
        logs.append(f"        ❌ 缺少参数文件: inference.pdiparams")
        return False
    
    # 确定模型文件
    model_file = pdmodel_path if os.path.exists(pdmodel_path) else json_path
    if not os.path.exists(model_file):
        logs.append(f"        ❌ 缺少模型文件")
        return False
    
    logs.append(f"        输入形状: [batch, {input_shape[0]}, {input_shape[1]}, {input_shape[2]}]")
    
    try:
        import paddle2onnx
        
        logs.append(f"        使用 paddle2onnx 转换...")
        
        # 使用固定输入尺寸，batch设为1（后续会设置动态batch）
        paddle2onnx.export(
            model_file,
            pdiparams_path,
            output_path,
            opset_version=14,
            enable_onnx_checker=True,
        )
        
        if os.path.exists(output_path):
            # 修改ONNX使batch维度动态
            _make_onnx_dynamic_batch(output_path, logs)
            
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logs.append(f"        ✓ model.onnx ({size_mb:.1f} MB)")
            return True
        
        logs.append(f"        ❌ 未生成文件")
        return False
        
    except Exception as e:
        logs.append(f"        ❌ paddle2onnx异常: {e}")
        import traceback
        logs.append(traceback.format_exc()[:200])
        return False


def _make_onnx_dynamic_batch(onnx_path, logs):
    """
    修改ONNX模型，使batch维度动态
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        
        model = onnx.load(onnx_path)
        
        # 修改输入的batch维度为动态
        for input_tensor in model.graph.input:
            if input_tensor.type.tensor_type.shape.dim:
                # 设置第一个维度为动态
                input_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch'
                input_tensor.type.tensor_type.shape.dim[0].ClearField('dim_value')
        
        # 修改输出的batch维度为动态
        for output_tensor in model.graph.output:
            if output_tensor.type.tensor_type.shape.dim:
                output_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch'
                output_tensor.type.tensor_type.shape.dim[0].ClearField('dim_value')
        
        onnx.save(model, onnx_path)
        logs.append(f"        ✓ 已设置动态batch")
        
    except ImportError:
        logs.append(f"        ⚠ onnx库未安装，跳过动态batch设置")
    except Exception as e:
        logs.append(f"        ⚠ 设置动态batch失败: {str(e)[:50]}")


# ============ TensorRT 导出 (固定尺寸，动态batch) ============

def build_tensorrt_engine(onnx_path, engine_path, fp16, workspace_gb, input_shape, max_batch_size, logs):
    """
    构建 TensorRT 引擎 - 固定尺寸，动态batch
    
    Args:
        onnx_path: ONNX模型路径
        engine_path: 输出引擎路径
        fp16: 是否使用FP16
        workspace_gb: 工作空间大小(GB)
        input_shape: 输入形状 [C, H, W] (不含batch)
        max_batch_size: 最大batch大小
        logs: 日志列表
    
    Returns:
        bool: 是否成功
    """
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logs.append(f"        ❌ 解析错误: {parser.get_error(i)}")
                return False
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logs.append(f"        ✓ 启用 FP16")
        
        # 设置动态batch profile - 只有batch维度动态，尺寸固定
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        c, h, w = input_shape
        
        # min, opt, max 只有batch不同，尺寸相同
        min_shape = (1, c, h, w)
        opt_shape = (max(1, max_batch_size // 2), c, h, w)
        max_shape = (max_batch_size, c, h, w)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        logs.append(f"        输入: {input_name}")
        logs.append(f"        形状: batch=[1-{max_batch_size}], CHW=[{c},{h},{w}]")
        logs.append(f"        ⏳ 构建 TensorRT 引擎...")
        
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            logs.append(f"        ❌ 构建失败")
            return False
        
        with open(engine_path, 'wb') as f:
            f.write(engine)
        
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        logs.append(f"        ✓ model.engine ({size_mb:.1f} MB)")
        return True
        
    except ImportError:
        logs.append(f"        ❌ TensorRT 未安装")
        return False
    except Exception as e:
        logs.append(f"        ❌ TensorRT异常: {e}")
        return False


# ============ OpenVINO 导出 ============

def export_to_openvino(onnx_path, output_dir, logs):
    """
    导出 OpenVINO 格式
    """
    try:
        import openvino as ov
        
        logs.append(f"        ⏳ 转换 OpenVINO...")
        
        core = ov.Core()
        model = core.read_model(onnx_path)
        
        xml_path = os.path.join(output_dir, "model.xml")
        ov.save_model(model, xml_path, compress_to_fp16=True)
        
        bin_path = os.path.join(output_dir, "model.bin")
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            xml_size = os.path.getsize(xml_path) / 1024
            bin_size = os.path.getsize(bin_path) / (1024 * 1024)
            logs.append(f"        ✓ model.xml ({xml_size:.1f} KB)")
            logs.append(f"        ✓ model.bin ({bin_size:.1f} MB)")
            return True
        
        logs.append(f"        ❌ 文件未生成")
        return False
        
    except ImportError:
        logs.append(f"        ❌ OpenVINO 未安装")
        return False
    except Exception as e:
        logs.append(f"        ❌ OpenVINO异常: {e}")
        return False


# ============ 配置文件生成 (V2格式，兼容推理工具集) ============

def create_det_config(name, params, files, input_shape, max_batch_size):
    """
    创建检测模型配置 - V2格式
    
    Args:
        name: 模型名称
        params: 参数字典
        files: 文件字典
        input_shape: [C, H, W]
        max_batch_size: 最大batch
    """
    c, h, w = input_shape
    
    return {
        "config_version": CONFIG_VERSION,
        "export_version": EXPORT_VERSION,
        "export_time": datetime.now().isoformat(),
        
        "model_info": {
            "name": name,
            "task": "ocr_detection",
            "algorithm": "DB",
            "framework": "PaddleOCR",
        },
        
        "files": files,
        
        "input": {
            "name": "x",
            "shape": [-1, c, h, w],  # batch动态，尺寸固定
            "dtype": "float32",
            "layout": "NCHW",
        },
        
        "output": {
            "name": "sigmoid_0.tmp_0",
            "shape": [-1, 1, h, w],
            "dtype": "float32",
        },
        
        "inference": {
            "max_batch_size": max_batch_size,
            "fp16": params.get('fp16', True),
        },
        
        "preprocess": {
            "resize": {
                "target_height": h,
                "target_width": w,
                "keep_ratio": True,
                "padding": True,
                "pad_value": 0,
                "interp": "LINEAR",
            },
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "to_rgb": True,
            },
        },
        
        "postprocess": {
            "algorithm": "DB",
            "thresh": params['thresh'],
            "box_thresh": params['box_thresh'],
            "unclip_ratio": params['unclip_ratio'],
            "max_candidates": params['max_candidates'],
            "use_dilation": params['use_dilation'],
            "score_mode": params['score_mode'],
        },
    }


def create_rec_config(name, params, files, dict_file, input_shape, max_batch_size):
    """
    创建识别模型配置 - V2格式
    
    Args:
        name: 模型名称
        params: 参数字典
        files: 文件字典
        dict_file: 字符字典文件名
        input_shape: [C, H, W]
        max_batch_size: 最大batch
    """
    c, h, w = input_shape
    
    return {
        "config_version": CONFIG_VERSION,
        "export_version": EXPORT_VERSION,
        "export_time": datetime.now().isoformat(),
        
        "model_info": {
            "name": name,
            "task": "ocr_recognition",
            "algorithm": "CTC",
            "framework": "PaddleOCR",
        },
        
        "files": files,
        "character_dict": dict_file,
        
        "input": {
            "name": "x",
            "shape": [-1, c, h, w],  # batch动态，尺寸固定
            "dtype": "float32",
            "layout": "NCHW",
        },
        
        "output": {
            "name": "softmax_0.tmp_0",
            "shape": [-1, -1, "num_classes"],  # [batch, seq_len, num_classes]
            "dtype": "float32",
        },
        
        "inference": {
            "max_batch_size": max_batch_size,
            "fp16": params.get('fp16', True),
        },
        
        "preprocess": {
            "resize": {
                "target_height": h,
                "target_width": w,
                "keep_ratio": True,
                "padding": "right",
                "pad_value": 0,
                "interp": "LINEAR",
            },
            "normalize": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "to_rgb": True,
            },
        },
        
        "postprocess": {
            "algorithm": "CTC",
            "use_space_char": params['use_space_char'],
            "score_thresh": params['score_thresh'],
            "remove_duplicates": True,
        },
    }


# ============ 主导出函数 ============

def do_export(det_sel, rec_sel, export_onnx, export_trt, export_openvino, use_fp16, workspace_gb,
              det_input_h, det_input_w, det_max_batch, det_thresh, det_box_thresh, det_max_cand, det_unclip, det_dilation, det_score_mode,
              rec_input_h, rec_input_w, rec_max_batch, rec_space, rec_score_th):
    """
    执行模型导出
    
    重要变更 (V7.0):
    - 检测/识别模型使用固定输入尺寸
    - 只支持动态batch
    - config.json格式升级为V2
    """
    logs = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_output_dir = get_default_export_dir()
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_models = find_models()
    det_path, det_name, rec_path, rec_name = None, None, None, None
    
    if det_sel and det_sel != "不导出":
        det_name = det_sel.split(" [")[0]
        for m in all_models['det']:
            if m['name'] == det_name:
                det_path = m['path']
                break
    
    if rec_sel and rec_sel != "不导出":
        rec_name = rec_sel.split(" [")[0]
        for m in all_models['rec']:
            if m['name'] == rec_name:
                rec_path = m['path']
                break
    
    if not det_path and not rec_path:
        logs.append("\n❌ 没有选择模型！")
        return "\n".join(logs), None
    
    # 创建输出目录
    model_prefix = []
    if det_name:
        parts = det_name.split('_')
        if len(parts) >= 2:
            model_prefix.append(f"{parts[0]}_{parts[1]}")
        else:
            model_prefix.append(det_name[:20])
    if rec_name and not model_prefix:
        parts = rec_name.split('_')
        if len(parts) >= 2:
            model_prefix.append(f"{parts[0]}_{parts[1]}")
        else:
            model_prefix.append(rec_name[:20])
    
    folder_name = f"{'_'.join(model_prefix)}_{ts}" if model_prefix else f"ocr_export_{ts}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logs.append("╔" + "═" * 58 + "╗")
    logs.append("║" + f"OCR 模型导出 V{EXPORT_VERSION}".center(54) + "║")
    logs.append("╚" + "═" * 58 + "╝")
    logs.append(f"\n📁 输出目录: {output_dir}")
    
    # 检查框架
    fw = check_frameworks()
    logs.append("\n📦 推理框架:")
    logs.append("─" * 40)
    if fw['paddle2onnx']['ok']:
        logs.append(f"  ✓ paddle2onnx {fw['paddle2onnx']['ver']}")
    else:
        logs.append(f"  ⚠ paddle2onnx 未安装")
    if export_trt:
        if fw['tensorrt']['ok']:
            logs.append(f"  ✓ TensorRT {fw['tensorrt']['ver']}")
        else:
            logs.append(f"  ⚠ TensorRT 未安装，跳过")
            export_trt = False
    if export_openvino:
        if fw['openvino']['ok']:
            logs.append(f"  ✓ OpenVINO")
        else:
            logs.append(f"  ⚠ OpenVINO 未安装，跳过")
            export_openvino = False
    
    logs.append(f"\n📋 导出配置:")
    logs.append("─" * 40)
    logs.append(f"  ⚡ 动态batch: ✓ (固定尺寸，仅batch动态)")
    if det_path:
        logs.append(f"  🔍 检测: {det_name}")
        logs.append(f"     输入: [{det_max_batch}, 3, {det_input_h}, {det_input_w}]")
    if rec_path:
        logs.append(f"  📝 识别: {rec_name}")
        logs.append(f"     输入: [{rec_max_batch}, 3, {rec_input_h}, {rec_input_w}]")
    
    with tempfile.TemporaryDirectory() as temp:
        # ============ 导出检测模型 ============
        if det_path:
            logs.append(f"\n🔍 导出检测模型")
            logs.append("─" * 40)
            det_out = os.path.join(temp, "det")
            os.makedirs(det_out, exist_ok=True)
            det_files = {}
            det_input_shape = [3, det_input_h, det_input_w]
            
            # ONNX
            logs.append("    [ONNX]")
            onnx_path = os.path.join(det_out, "model.onnx")
            if export_to_onnx(det_path, onnx_path, det_input_shape, logs):
                det_files['onnx'] = 'model.onnx'
                
                # TensorRT
                if export_trt:
                    logs.append("    [TensorRT]")
                    engine_path = os.path.join(det_out, "model.engine")
                    if build_tensorrt_engine(onnx_path, engine_path, use_fp16, workspace_gb, 
                                            det_input_shape, det_max_batch, logs):
                        det_files['tensorrt'] = 'model.engine'
                
                # OpenVINO
                if export_openvino:
                    logs.append("    [OpenVINO]")
                    if export_to_openvino(onnx_path, det_out, logs):
                        det_files['openvino'] = {'xml': 'model.xml', 'bin': 'model.bin'}
            
            # 生成配置
            det_params = {
                'thresh': det_thresh,
                'box_thresh': det_box_thresh,
                'max_candidates': det_max_cand,
                'unclip_ratio': det_unclip,
                'use_dilation': det_dilation,
                'score_mode': det_score_mode,
                'fp16': use_fp16,
            }
            config = create_det_config(det_name, det_params, det_files, det_input_shape, det_max_batch)
            with open(os.path.join(det_out, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logs.append(f"    ✓ config.json (V{CONFIG_VERSION})")
        
        # ============ 导出识别模型 ============
        if rec_path:
            logs.append(f"\n📝 导出识别模型")
            logs.append("─" * 40)
            rec_out = os.path.join(temp, "rec")
            os.makedirs(rec_out, exist_ok=True)
            rec_files = {}
            rec_input_shape = [3, rec_input_h, rec_input_w]
            
            # ONNX
            logs.append("    [ONNX]")
            onnx_path = os.path.join(rec_out, "model.onnx")
            if export_to_onnx(rec_path, onnx_path, rec_input_shape, logs):
                rec_files['onnx'] = 'model.onnx'
                
                # TensorRT
                if export_trt:
                    logs.append("    [TensorRT]")
                    engine_path = os.path.join(rec_out, "model.engine")
                    if build_tensorrt_engine(onnx_path, engine_path, use_fp16, workspace_gb,
                                            rec_input_shape, rec_max_batch, logs):
                        rec_files['tensorrt'] = 'model.engine'
                
                # OpenVINO
                if export_openvino:
                    logs.append("    [OpenVINO]")
                    if export_to_openvino(onnx_path, rec_out, logs):
                        rec_files['openvino'] = {'xml': 'model.xml', 'bin': 'model.bin'}
            
            # 复制字符字典
            dict_file = None
            for df in ['ppocr_keys_v1.txt', 'chinese_keys.txt', 'en_dict.txt']:
                dp = os.path.join(rec_path, df)
                if os.path.exists(dp):
                    shutil.copy(dp, rec_out)
                    dict_file = df
                    logs.append(f"    ✓ {df}")
                    break
            
            if dict_file is None:
                # 尝试从缓存查找
                home = os.path.expanduser("~")
                cache_dict = os.path.join(home, ".paddleocr", "ppocr_keys_v1.txt")
                if os.path.exists(cache_dict):
                    shutil.copy(cache_dict, rec_out)
                    dict_file = "ppocr_keys_v1.txt"
                    logs.append(f"    ✓ {dict_file} (来自缓存)")
            
            # 生成配置
            rec_params = {
                'use_space_char': rec_space,
                'score_thresh': rec_score_th,
                'fp16': use_fp16,
            }
            config = create_rec_config(rec_name, rec_params, rec_files, dict_file, rec_input_shape, rec_max_batch)
            with open(os.path.join(rec_out, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logs.append(f"    ✓ config.json (V{CONFIG_VERSION})")
        
        # ============ 打包模型 ============
        pkg_paths = []
        
        if det_path:
            det_pkg_name = f"det_{det_name}_{ts}.pkg"
            det_pkg_path = os.path.join(output_dir, det_pkg_name)
            logs.append(f"\n📦 打包检测模型:")
            logs.append("─" * 40)
            
            with zipfile.ZipFile(det_pkg_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                det_out_dir = os.path.join(temp, "det")
                for root, dirs, files in os.walk(det_out_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        arc = os.path.relpath(fp, det_out_dir)
                        zf.write(fp, arc)
                        logs.append(f"    + {arc}")
            
            det_pkg_size = os.path.getsize(det_pkg_path) / (1024 * 1024)
            logs.append(f"    ✓ {det_pkg_name} ({det_pkg_size:.1f} MB)")
            pkg_paths.append(det_pkg_path)
        
        if rec_path:
            rec_pkg_name = f"rec_{rec_name}_{ts}.pkg"
            rec_pkg_path = os.path.join(output_dir, rec_pkg_name)
            logs.append(f"\n📦 打包识别模型:")
            logs.append("─" * 40)
            
            with zipfile.ZipFile(rec_pkg_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                rec_out_dir = os.path.join(temp, "rec")
                for root, dirs, files in os.walk(rec_out_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        arc = os.path.relpath(fp, rec_out_dir)
                        zf.write(fp, arc)
                        logs.append(f"    + {arc}")
            
            rec_pkg_size = os.path.getsize(rec_pkg_path) / (1024 * 1024)
            logs.append(f"    ✓ {rec_pkg_name} ({rec_pkg_size:.1f} MB)")
            pkg_paths.append(rec_pkg_path)
    
    logs.append("\n" + "═" * 60)
    logs.append(f"✅ 导出完成!")
    logs.append(f"📁 输出目录: {output_dir}")
    if det_path:
        logs.append(f"   🔍 检测: det_{det_name}_{ts}.pkg")
        logs.append(f"      输入: [1-{det_max_batch}, 3, {det_input_h}, {det_input_w}]")
    if rec_path:
        logs.append(f"   📝 识别: rec_{rec_name}_{ts}.pkg")
        logs.append(f"      输入: [1-{rec_max_batch}, 3, {rec_input_h}, {rec_input_w}]")
    logs.append("═" * 60)
    
    # Gradio文件下载
    gradio_files = []
    if pkg_paths:
        import tempfile as tf
        gradio_temp_dir = tf.mkdtemp(prefix="ocr_export_")
        for pkg_path in pkg_paths:
            pkg_name = os.path.basename(pkg_path)
            gradio_pkg_path = os.path.join(gradio_temp_dir, pkg_name)
            shutil.copy2(pkg_path, gradio_pkg_path)
            gradio_files.append(gradio_pkg_path)
    
    return "\n".join(logs), gradio_files if gradio_files else None


# ============ UI 辅助函数 ============

def _build_file_tree(file_list):
    """构建文件树结构"""
    tree = {}
    for filepath, size in file_list:
        parts = filepath.replace('\\', '/').split('/')
        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {'__type': 'dir', '__children': {}}
            current = current[part]['__children']
        filename = parts[-1]
        current[filename] = {'__type': 'file', '__size': size}
    return tree


def _format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _get_file_icon(filename):
    """根据文件类型返回图标"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    icons = {
        'onnx': '🔷',
        'engine': '⚡',
        'xml': '📄',
        'bin': '💾',
        'json': '⚙️',
        'txt': '📝',
        'pkg': '📦',
    }
    return icons.get(ext, '📄')


# ============ 创建UI面板 ============

def create_export_panel():
    """创建导出面板UI"""
    models = find_models()
    det_ch = ["不导出"] + [m['display'] for m in models['det']]
    rec_ch = ["不导出"] + [m['display'] for m in models['rec']]
    det_def = det_ch[1] if len(det_ch) > 1 else "不导出"
    rec_def = rec_ch[1] if len(rec_ch) > 1 else "不导出"
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(_sec("📦", "模型选择"))
            gr.Markdown(f"**模型目录:** `{get_pretrained_models_dir()}`")
            det_model = gr.Dropdown(choices=det_ch, value=det_def, label="🔍 检测模型")
            rec_model = gr.Dropdown(choices=rec_ch, value=rec_def, label="📝 识别模型")
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新", size="sm")
                download_btn = gr.Button("⬇️ 下载模型", size="sm", variant="primary")
            
            gr.HTML(_sec("🚀", "推理框架"))
            export_onnx = gr.Checkbox(label="✓ ONNX Runtime (必选)", value=True, interactive=False)
            export_trt = gr.Checkbox(label="TensorRT", value=True)
            export_openvino = gr.Checkbox(label="OpenVINO", value=False)
            with gr.Row():
                use_fp16 = gr.Checkbox(label="FP16", value=True, info="半精度加速")
                workspace_gb = gr.Slider(1, 8, 4, step=1, label="TRT显存(GB)")
            
            gr.HTML(_sec("🔧", "检测模型参数"))
            gr.Markdown("**⚠️ 重要:** 输入尺寸固定，只支持动态batch")
            with gr.Accordion("输入配置", open=True):
                with gr.Row():
                    det_input_h = gr.Slider(320, 1920, 960, step=32, label="输入高度", info="推理时resize到此尺寸")
                    det_input_w = gr.Slider(320, 1920, 960, step=32, label="输入宽度", info="推理时resize到此尺寸")
                det_max_batch = gr.Slider(1, 16, 4, step=1, label="最大Batch", info="动态batch上限")
            with gr.Accordion("后处理参数", open=False):
                det_thresh = gr.Slider(0.1, 0.9, 0.3, step=0.05, label="thresh", info="二值化阈值")
                det_box_thresh = gr.Slider(0.3, 0.9, 0.6, step=0.05, label="box_thresh", info="框置信度")
                det_unclip = gr.Slider(1.0, 3.0, 1.5, step=0.1, label="unclip_ratio", info="框扩展比例")
                det_max_cand = gr.Slider(100, 2000, 1000, step=100, label="max_candidates")
                det_dilation = gr.Checkbox(label="use_dilation", value=False)
                det_score_mode = gr.Radio(["fast", "slow"], value="fast", label="score_mode")
            
            gr.HTML(_sec("🔧", "识别模型参数"))
            with gr.Accordion("输入配置", open=True):
                with gr.Row():
                    rec_input_h = gr.Slider(32, 64, 48, step=8, label="输入高度", info="V3:32, V4/V5:48")
                    rec_input_w = gr.Slider(160, 640, 320, step=32, label="输入宽度", info="文本行宽度")
                rec_max_batch = gr.Slider(1, 64, 32, step=1, label="最大Batch", info="识别可用更大batch")
            with gr.Accordion("后处理参数", open=False):
                rec_space = gr.Checkbox(label="use_space_char", value=True)
                rec_score_th = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="score_thresh")
            
            export_btn = gr.Button("🚀 开始导出", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.HTML(_sec("📋", "导出日志"))
            export_log = gr.Textbox(label="", lines=35, max_lines=50, show_label=False)
            gr.HTML(_sec("📁", "输出文件"))
            output_files = gr.Files(label="导出的模型包 (.pkg)")
    
    def refresh():
        models = find_models()
        det_ch = ["不导出"] + [m['display'] for m in models['det']]
        rec_ch = ["不导出"] + [m['display'] for m in models['rec']]
        return (
            gr.Dropdown(choices=det_ch, value=det_ch[1] if len(det_ch) > 1 else "不导出"),
            gr.Dropdown(choices=rec_ch, value=rec_ch[1] if len(rec_ch) > 1 else "不导出")
        )
    
    refresh_btn.click(refresh, outputs=[det_model, rec_model])
    download_btn.click(download_models, outputs=[export_log])
    export_btn.click(
        do_export,
        inputs=[
            det_model, rec_model,
            export_onnx, export_trt, export_openvino,
            use_fp16, workspace_gb,
            det_input_h, det_input_w, det_max_batch,
            det_thresh, det_box_thresh, det_max_cand, det_unclip, det_dilation, det_score_mode,
            rec_input_h, rec_input_w, rec_max_batch,
            rec_space, rec_score_th
        ],
        outputs=[export_log, output_files]
    )