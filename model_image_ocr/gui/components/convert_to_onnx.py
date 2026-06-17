# -*- coding: utf-8 -*-
"""
手动转换 PaddleX 模型到 ONNX

使用方法:
    python convert_to_onnx.py --model_dir ./pretrained_models/PP-OCRv5_server_det
"""

import os
import sys
import argparse
import tempfile
import shutil


def convert_paddlex_to_onnx(model_dir, output_dir=None):
    """转换 PaddleX 模型到 ONNX"""
    
    if output_dir is None:
        output_dir = model_dir
    
    model_name = os.path.basename(model_dir)
    print(f"\n{'='*60}")
    print(f"转换模型: {model_name}")
    print(f"输入目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    # 检查文件
    json_path = os.path.join(model_dir, 'inference.json')
    params_path = os.path.join(model_dir, 'inference.pdiparams')
    pdmodel_path = os.path.join(model_dir, 'inference.pdmodel')
    
    if os.path.exists(pdmodel_path):
        print("[INFO] 已有 inference.pdmodel，直接转换")
        return _convert_with_paddle2onnx(pdmodel_path, params_path, output_dir)
    
    if not os.path.exists(json_path):
        print(f"[ERROR] 找不到 {json_path}")
        return False
    
    print(f"[INFO] 检测到 PaddleX 格式 (inference.json)")
    
    # 方法1: 使用 PaddleX 导出
    print("\n[方法1] 尝试 PaddleX 导出...")
    try:
        from paddlex import create_model
        
        model = create_model(model_name=model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.export(temp_dir, format='onnx')
            
            for f in os.listdir(temp_dir):
                if f.endswith('.onnx'):
                    onnx_path = os.path.join(output_dir, 'inference.onnx')
                    shutil.copy(os.path.join(temp_dir, f), onnx_path)
                    print(f"[SUCCESS] 导出成功: {onnx_path}")
                    return True
        
        print("[WARN] PaddleX 未生成 ONNX 文件")
        
    except ImportError:
        print("[WARN] PaddleX 未安装")
    except Exception as e:
        print(f"[WARN] PaddleX 导出失败: {e}")
    
    # 方法2: 先转 paddle 格式，再转 onnx
    print("\n[方法2] 尝试先转换为 Paddle 格式...")
    try:
        import paddle
        paddle.enable_static()
        
        exe = paddle.static.Executor(paddle.CPUPlace())
        
        print(f"  加载模型...")
        [prog, feed_target_names, fetch_targets] = paddle.static.load_inference_model(
            model_dir, 
            exe,
            model_filename='inference.json',
            params_filename='inference.pdiparams'
        )
        
        print(f"  输入: {feed_target_names}")
        
        # 保存为标准格式
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model = os.path.join(temp_dir, 'inference')
            
            # 获取 feed_vars
            feed_vars = [prog.global_block().var(name) for name in feed_target_names]
            
            paddle.static.save_inference_model(
                temp_model,
                feed_vars,
                fetch_targets,
                exe,
                program=prog
            )
            
            temp_pdmodel = temp_model + '.pdmodel'
            temp_params = temp_model + '.pdiparams'
            
            if os.path.exists(temp_pdmodel):
                print(f"  转换成功，使用 paddle2onnx...")
                return _convert_with_paddle2onnx(temp_pdmodel, temp_params, output_dir)
            else:
                print(f"  [ERROR] 转换失败")
        
    except Exception as e:
        print(f"[WARN] Paddle 转换失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法3: 直接用 paddle2onnx (json)
    print("\n[方法3] 尝试 paddle2onnx 直接转换 json...")
    try:
        import paddle2onnx
        
        onnx_path = os.path.join(output_dir, 'inference.onnx')
        
        paddle2onnx.export(
            json_path,
            params_path,
            onnx_path,
            opset_version=14
        )
        
        if os.path.exists(onnx_path):
            print(f"[SUCCESS] 导出成功: {onnx_path}")
            return True
            
    except Exception as e:
        print(f"[WARN] paddle2onnx 失败: {e}")
    
    print("\n[ERROR] 所有方法都失败了")
    print("\n建议尝试:")
    print("  1. pip install paddlex")
    print("  2. paddlex --export_onnx --model_dir=<model_dir> --save_dir=<output_dir>")
    
    return False


def _convert_with_paddle2onnx(pdmodel, pdiparams, output_dir):
    """使用 paddle2onnx 转换"""
    try:
        import paddle2onnx
        
        onnx_path = os.path.join(output_dir, 'inference.onnx')
        
        paddle2onnx.export(
            pdmodel,
            pdiparams,
            onnx_path,
            opset_version=14,
            enable_onnx_checker=True
        )
        
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"[SUCCESS] 导出成功: {onnx_path} ({size_mb:.1f} MB)")
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERROR] paddle2onnx 失败: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='转换 PaddleX 模型到 ONNX')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录 (默认同模型目录)')
    
    args = parser.parse_args()
    
    success = convert_paddlex_to_onnx(args.model_dir, args.output_dir)
    sys.exit(0 if success else 1)
