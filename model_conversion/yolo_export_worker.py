#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO ONNX 导出工作脚本（子进程版本）

用于在独立进程中执行 Ultralytics YOLO 模型的 ONNX 导出，
避免在同一进程中连续调用 YOLO.export() 导致的崩溃问题。

使用方法：
    python yolo_export_worker.py <config_json_path>

配置文件格式 (JSON):
{
    "model_path": "path/to/model.pt",
    "output_path": "path/to/output.onnx",
    "opset": 17,
    "dynamic_batch": true,
    "simplify": false
}

输出：
    成功时写入结果文件: <config_json_path>.result
    {
        "success": true,
        "output_path": "path/to/output.onnx",
        "message": "导出成功"
    }

作者: Model Conversion Team
版本: 1.0.0
"""

import sys
import os
import json
import traceback


def export_yolo_to_onnx(config: dict) -> dict:
    """
    执行 YOLO 模型到 ONNX 的导出
    
    Args:
        config: 配置字典
            - model_path: YOLO 模型路径 (.pt)
            - output_path: ONNX 输出路径
            - opset: ONNX opset 版本 (默认 17)
            - dynamic_batch: 是否启用动态 batch (默认 True)
            - simplify: 是否简化 ONNX (默认 False)
    
    Returns:
        结果字典
    """
    try:
        model_path = config.get('model_path')
        output_path = config.get('output_path')
        opset = config.get('opset', 17)
        dynamic_batch = config.get('dynamic_batch', True)
        simplify = config.get('simplify', False)
        
        print(f"[YOLO Export Worker] 开始导出...", flush=True)
        print(f"  模型路径: {model_path}", flush=True)
        print(f"  输出路径: {output_path}", flush=True)
        print(f"  Opset: {opset}", flush=True)
        print(f"  动态 Batch: {dynamic_batch}", flush=True)
        
        # 检查模型文件
        if not os.path.exists(model_path):
            return {
                'success': False,
                'output_path': None,
                'message': f'模型文件不存在: {model_path}'
            }
        
        # 导入 Ultralytics
        from ultralytics import YOLO
        
        # 加载模型
        print(f"[YOLO Export Worker] 加载模型...", flush=True)
        model = YOLO(model_path)
        
        # 构建导出参数
        export_kwargs = {
            'format': 'onnx',
            'opset': opset,
            'simplify': simplify,
            'dynamic': dynamic_batch,
            'half': False,  # ONNX 导出不使用半精度
            'device': 'cpu',  # CPU 上导出更稳定
        }
        
        # 执行导出
        print(f"[YOLO Export Worker] 调用 YOLO.export()...", flush=True)
        exported_path = model.export(**export_kwargs)
        print(f"[YOLO Export Worker] 导出完成: {exported_path}", flush=True)
        
        # 检查导出结果
        if exported_path and os.path.exists(exported_path):
            # 如果目标路径不同，移动文件
            if str(exported_path) != str(output_path):
                import shutil
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                shutil.move(str(exported_path), output_path)
                print(f"[YOLO Export Worker] 文件移动到: {output_path}", flush=True)
            
            # 获取文件大小
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            return {
                'success': True,
                'output_path': output_path,
                'file_size_mb': file_size_mb,
                'message': f'导出成功: {output_path}'
            }
        else:
            return {
                'success': False,
                'output_path': None,
                'message': f'导出失败: 输出文件不存在'
            }
            
    except Exception as e:
        error_msg = f"导出异常: {str(e)}\n{traceback.format_exc()}"
        print(f"[YOLO Export Worker] {error_msg}", flush=True)
        return {
            'success': False,
            'output_path': None,
            'message': error_msg
        }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python yolo_export_worker.py <config_json_path>", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    result_path = config_path + '.result'
    
    try:
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"[YOLO Export Worker] 读取配置: {config_path}", flush=True)
        
        # 执行导出
        result = export_yolo_to_onnx(config)
        
        # 写入结果
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[YOLO Export Worker] 结果已写入: {result_path}", flush=True)
        
        # 返回状态码
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        error_result = {
            'success': False,
            'output_path': None,
            'message': f'工作进程异常: {str(e)}\n{traceback.format_exc()}'
        }
        
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        
        print(f"[YOLO Export Worker] 异常: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
