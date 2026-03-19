#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegFormer ONNX 导出工作脚本（子进程版本）

用于在独立进程中执行 MMSegmentation SegFormer 模型的 ONNX 导出，
避免内存泄漏和进程内冲突问题。

使用方法：
    python segformer_export_worker.py <config_json_path>

配置文件格式 (JSON):
{
    "model_path": "path/to/model.pth",
    "output_path": "path/to/output.onnx",
    "opset": 17,
    "dynamic_batch": true,
    "input_shape": [1, 3, 512, 512],
    "model_name": "segformer_b0"  # 可选，用于推断模型规模
}

输出：
    成功时写入结果文件: <config_json_path>.result
    {
        "success": true,
        "output_path": "path/to/output.onnx",
        "message": "导出成功",
        "num_classes": 5
    }

修复内容：
    - 从模型 checkpoint 的 decode_head.conv_seg.weight 推断类别数
    - 支持 linear_pred.weight 和 conv_seg.weight 两种命名方式

作者: Model Conversion Team
版本: 1.1.0
"""

import sys
import os
import json
import traceback


# SegFormer 变体配置
SEGFORMER_VARIANTS = {
    'segformer_b0': {
        'embed_dims': [32, 64, 160, 256],
        'depths': [2, 2, 2, 2],
        'num_heads': [1, 2, 5, 8],
    },
    'segformer_b1': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [2, 2, 2, 2],
        'num_heads': [1, 2, 5, 8],
    },
    'segformer_b2': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 4, 6, 3],
        'num_heads': [1, 2, 5, 8],
    },
    'segformer_b3': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 4, 18, 3],
        'num_heads': [1, 2, 5, 8],
    },
    'segformer_b4': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 8, 27, 3],
        'num_heads': [1, 2, 5, 8],
    },
    'segformer_b5': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 6, 40, 3],
        'num_heads': [1, 2, 5, 8],
    },
}


def get_num_classes_from_checkpoint(checkpoint: dict) -> int:
    """
    从 checkpoint 推断类别数
    
    优先级:
    1. checkpoint['num_classes']
    2. checkpoint['nc']
    3. checkpoint['model_metadata']['num_classes']
    4. len(checkpoint['class_names'])
    5. 从 meta.dataset_meta.classes 推断
    6. 从 state_dict 的 decode_head.conv_seg.weight 推断
    7. 从 state_dict 的 decode_head.linear_pred.weight 推断
    
    Args:
        checkpoint: 模型 checkpoint 字典
        
    Returns:
        类别数
        
    Raises:
        ValueError: 无法推断类别数
    """
    # 方法1: 直接获取 num_classes
    if 'num_classes' in checkpoint:
        nc = checkpoint['num_classes']
        if nc is not None and nc > 0:
            print(f"  [SegFormer Export Worker] 类别数来源: checkpoint['num_classes'] = {nc}", flush=True)
            return int(nc)
    
    # 方法2: 获取 nc
    if 'nc' in checkpoint:
        nc = checkpoint['nc']
        if nc is not None and nc > 0:
            print(f"  [SegFormer Export Worker] 类别数来源: checkpoint['nc'] = {nc}", flush=True)
            return int(nc)
    
    # 方法3: 从 model_metadata 获取
    metadata = checkpoint.get('model_metadata', {})
    if 'num_classes' in metadata:
        nc = metadata['num_classes']
        if nc is not None and nc > 0:
            print(f"  [SegFormer Export Worker] 类别数来源: model_metadata['num_classes'] = {nc}", flush=True)
            return int(nc)
    
    # 方法4: 从 class_names 推断
    if 'class_names' in checkpoint:
        names = checkpoint['class_names']
        if isinstance(names, (list, tuple)) and len(names) > 0:
            nc = len(names)
            print(f"  [SegFormer Export Worker] 类别数来源: len(class_names) = {nc}", flush=True)
            return nc
    
    # 方法5: 从 meta.dataset_meta.classes 推断
    meta = checkpoint.get('meta', {})
    dataset_meta = meta.get('dataset_meta', {})
    if 'classes' in dataset_meta:
        classes = dataset_meta['classes']
        if isinstance(classes, (list, tuple)) and len(classes) > 0:
            nc = len(classes)
            print(f"  [SegFormer Export Worker] 类别数来源: len(meta.dataset_meta.classes) = {nc}", flush=True)
            return nc
    
    # 方法6 & 7: 从 state_dict 推断
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', {}))
    if isinstance(state_dict, dict):
        # 方法6: 查找 decode_head.conv_seg.weight (MMSegmentation 标准命名)
        for key in state_dict.keys():
            if 'decode_head' in key and 'conv_seg.weight' in key:
                # conv_seg.weight 形状为 [num_classes, channels, 1, 1]
                nc = state_dict[key].shape[0]
                if nc > 0:
                    print(f"  [SegFormer Export Worker] 类别数来源: {key}.shape[0] = {nc}", flush=True)
                    return int(nc)
        
        # 方法7: 查找 decode_head.linear_pred.weight
        for key in state_dict.keys():
            if 'decode_head' in key and 'linear_pred.weight' in key:
                nc = state_dict[key].shape[0]
                if nc > 0:
                    print(f"  [SegFormer Export Worker] 类别数来源: {key}.shape[0] = {nc}", flush=True)
                    return int(nc)
    
    raise ValueError("无法从 checkpoint 推断类别数")


def get_model_scale(model_name: str) -> str:
    """
    从模型名称推断模型规模
    
    Args:
        model_name: 模型名称，如 'segformer_b0', 'SegFormer_mit_b2' 等
        
    Returns:
        规模字符串，如 'b0', 'b1', 等
    """
    model_name_lower = model_name.lower()
    
    for scale in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']:
        if scale in model_name_lower:
            return scale
    
    # 默认使用 b0
    return 'b0'


def export_segformer_to_onnx(config: dict) -> dict:
    """
    执行 SegFormer 模型到 ONNX 的导出
    
    Args:
        config: 配置字典
            - model_path: SegFormer 模型路径 (.pth)
            - output_path: ONNX 输出路径
            - opset: ONNX opset 版本 (默认 17)
            - dynamic_batch: 是否启用动态 batch (默认 True)
            - input_shape: 输入形状 [B, C, H, W] (默认 [1, 3, 512, 512])
            - model_name: 模型名称，用于推断规模 (可选)
    
    Returns:
        结果字典
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    try:
        model_path = config.get('model_path')
        output_path = config.get('output_path')
        opset = config.get('opset', 17)
        dynamic_batch = config.get('dynamic_batch', True)
        input_shape = config.get('input_shape', [1, 3, 512, 512])
        model_name = config.get('model_name', 'segformer_b0')
        
        print(f"[SegFormer Export Worker] 读取配置: {config.get('config_path', 'N/A')}", flush=True)
        print(f"[SegFormer Export Worker] 开始导出...", flush=True)
        print(f"  模型路径: {model_path}", flush=True)
        print(f"  输出路径: {output_path}", flush=True)
        print(f"  Opset: {opset}", flush=True)
        print(f"  动态 Batch: {dynamic_batch}", flush=True)
        print(f"  输入形状: {input_shape}", flush=True)
        print(f"  配置中的模型名称: {model_name}", flush=True)
        
        # 检查模型文件
        if not os.path.exists(model_path):
            return {
                'success': False,
                'output_path': None,
                'message': f'模型文件不存在: {model_path}'
            }
        
        # 加载 checkpoint
        print(f"[SegFormer Export Worker] 加载模型...", flush=True)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 从 checkpoint 推断类别数
        try:
            num_classes = get_num_classes_from_checkpoint(checkpoint)
        except ValueError as e:
            return {
                'success': False,
                'output_path': None,
                'message': f'无法推断类别数: {e}'
            }
        
        print(f"  类别数: {num_classes}", flush=True)
        
        # 推断模型规模
        scale = get_model_scale(model_name)
        variant_key = f'segformer_{scale}'
        variant_config = SEGFORMER_VARIANTS.get(variant_key, SEGFORMER_VARIANTS['segformer_b0'])
        print(f"  模型规模: {scale} (从模型名称'{model_name}'推断)", flush=True)
        
        # 获取训练尺寸
        train_size = input_shape[2] if len(input_shape) >= 3 else 512
        print(f"  训练尺寸: {train_size}", flush=True)
        
        # 导入 MMSegmentation 组件
        print(f"[SegFormer Export Worker] 使用标准配置重建模型...", flush=True)
        try:
            from mmseg.models.backbones import MixVisionTransformer
            from mmseg.models.decode_heads import SegformerHead
        except ImportError as e:
            return {
                'success': False,
                'output_path': None,
                'message': f'MMSegmentation 导入失败: {e}'
            }
        
        # 构建 Backbone
        embed_dims = variant_config['embed_dims']
        depths = variant_config['depths']
        num_heads = variant_config['num_heads']
        
        backbone = MixVisionTransformer(
            in_channels=3,
            embed_dims=embed_dims[0],
            num_stages=4,
            num_layers=depths,
            num_heads=num_heads,
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )
        
        # 构建 Decode Head
        decode_head = SegformerHead(
            in_channels=embed_dims,
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=num_classes,  # 使用推断的类别数
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        )
        
        # 加载权重
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', {}))
        if state_dict:
            backbone_state = {}
            decode_head_state = {}
            
            for key, value in state_dict.items():
                clean_key = key
                for prefix in ['module.', '_orig_mod.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                
                if clean_key.startswith('backbone.'):
                    new_key = clean_key[len('backbone.'):]
                    backbone_state[new_key] = value
                elif clean_key.startswith('decode_head.'):
                    new_key = clean_key[len('decode_head.'):]
                    decode_head_state[new_key] = value
            
            if backbone_state:
                backbone.load_state_dict(backbone_state, strict=False)
            if decode_head_state:
                decode_head.load_state_dict(decode_head_state, strict=False)
        
        backbone.eval()
        decode_head.eval()
        
        # 创建 ONNX 导出友好的包装器
        class SegFormerONNXWrapper(nn.Module):
            """SegFormer ONNX 导出包装器"""
            def __init__(self, backbone, decode_head, align_corners=False):
                super().__init__()
                self.backbone = backbone
                self.decode_head = decode_head
                self.align_corners = align_corners
            
            def forward(self, x):
                # Backbone 前向
                features = self.backbone(x)
                
                # Decode head 前向
                out = self.decode_head(features)
                
                # 上采样到输入尺寸
                out = F.interpolate(
                    out,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners
                )
                
                return out
        
        align_corners = getattr(decode_head, 'align_corners', False)
        model = SegFormerONNXWrapper(backbone, decode_head, align_corners)
        model.eval()
        
        # 创建 dummy input
        dummy_input = torch.randn(input_shape)
        
        # 准备动态轴
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 导出 ONNX
        print(f"[SegFormer Export Worker] 导出 ONNX...", flush=True)
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                verbose=False,
            )
        
        print(f"[SegFormer Export Worker] 导出完成: {output_path}", flush=True)
        
        # 验证 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"[SegFormer Export Worker] ONNX 模型验证通过", flush=True)
        except Exception as e:
            print(f"[SegFormer Export Worker] ONNX 验证警告: {e}", flush=True)
        
        return {
            'success': True,
            'output_path': output_path,
            'message': '导出成功',
            'num_classes': num_classes
        }
        
    except Exception as e:
        error_msg = f"导出异常: {str(e)}\n{traceback.format_exc()}"
        print(f"[SegFormer Export Worker] {error_msg}", flush=True)
        return {
            'success': False,
            'output_path': None,
            'message': error_msg
        }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python segformer_export_worker.py <config_json_path>", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    result_path = config_path + '.result'
    
    try:
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        config['config_path'] = config_path
        
        # 执行导出
        result = export_segformer_to_onnx(config)
        
        # 写入结果
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[SegFormer Export Worker] 结果已写入: {result_path}", flush=True)
        
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
        
        print(f"[SegFormer Export Worker] 异常: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
