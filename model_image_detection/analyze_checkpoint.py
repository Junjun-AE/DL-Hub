"""
分析 YOLO Checkpoint 内容
用于调试 checkpoint 缺少字段的问题

使用方法:
    python analyze_checkpoint.py best_model.pt
"""

import sys
import torch
from pathlib import Path


def analyze(ckpt_path: str):
    print(f"\n{'='*60}")
    print(f"分析 Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print(f"\n📁 文件大小: {Path(ckpt_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # 顶层字段
    print(f"\n📋 顶层字段 ({len(ckpt)} 个):")
    for key in sorted(ckpt.keys()):
        value = ckpt[key]
        if value is None:
            print(f"  {key}: None")
        elif isinstance(value, dict):
            print(f"  {key}: dict ({len(value)} items)")
            # 显示前几个
            for k, v in list(value.items())[:3]:
                print(f"    └─ {k}: {type(v).__name__} = {str(v)[:50]}")
            if len(value) > 3:
                print(f"    └─ ... ({len(value) - 3} more)")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} ({len(value)} items)")
        elif isinstance(value, torch.nn.Module):
            print(f"  {key}: {type(value).__module__}.{type(value).__name__}")
        else:
            val_str = str(value)[:80]
            print(f"  {key}: {type(value).__name__} = {val_str}")
    
    # 检查关键字段
    print(f"\n🔍 关键字段检查:")
    
    # nc
    if 'nc' in ckpt:
        print(f"  ✅ nc: {ckpt['nc']}")
    else:
        print(f"  ❌ nc: 缺失")
    
    # names
    if 'names' in ckpt:
        names = ckpt['names']
        if isinstance(names, dict):
            print(f"  ✅ names: dict ({len(names)} classes)")
        else:
            print(f"  ✅ names: {type(names)}")
    else:
        print(f"  ❌ names: 缺失")
    
    # train_args
    if 'train_args' in ckpt:
        train_args = ckpt['train_args']
        if isinstance(train_args, dict):
            print(f"  ✅ train_args: dict ({len(train_args)} items)")
            # 检查关键子字段
            for key in ['nc', 'imgsz', 'model', 'data']:
                if key in train_args:
                    print(f"    └─ {key}: {train_args[key]}")
        else:
            print(f"  ⚠️ train_args: {type(train_args)} (不是 dict)")
    else:
        print(f"  ❌ train_args: 缺失")
    
    # custom_metadata
    if 'custom_metadata' in ckpt:
        print(f"  ✅ custom_metadata: 存在")
    else:
        print(f"  ❌ custom_metadata: 缺失")
    
    # 检查 model 对象
    print(f"\n🧠 Model 对象分析:")
    model = ckpt.get('model')
    if model is not None:
        print(f"  类型: {type(model).__module__}.{type(model).__name__}")
        
        # 检查模型属性
        if hasattr(model, 'nc'):
            print(f"  model.nc: {model.nc}")
        else:
            print(f"  model.nc: ❌ 无此属性")
        
        if hasattr(model, 'names'):
            names = model.names
            if isinstance(names, dict):
                print(f"  model.names: dict ({len(names)} classes)")
                print(f"    前3个: {dict(list(names.items())[:3])}")
            else:
                print(f"  model.names: {names}")
        else:
            print(f"  model.names: ❌ 无此属性")
        
        if hasattr(model, 'yaml'):
            yaml_info = model.yaml
            if isinstance(yaml_info, dict):
                print(f"  model.yaml: dict")
                print(f"    nc: {yaml_info.get('nc', 'N/A')}")
            else:
                print(f"  model.yaml: {type(yaml_info)}")
        else:
            print(f"  model.yaml: ❌ 无此属性")
    else:
        print(f"  ❌ model 字段为空")
    
    # 结论
    print(f"\n{'='*60}")
    print("结论:")
    print(f"{'='*60}")
    
    has_nc = 'nc' in ckpt
    has_names = 'names' in ckpt
    has_train_args = 'train_args' in ckpt and isinstance(ckpt.get('train_args'), dict)
    
    if has_nc and has_names and has_train_args:
        print("✅ Checkpoint 格式正确，应该兼容 model_importer")
    else:
        print("❌ Checkpoint 缺少必要字段:")
        if not has_nc:
            print("   - 缺少 'nc' (类别数)")
        if not has_names:
            print("   - 缺少 'names' (类别名称)")
        if not has_train_args:
            print("   - 缺少 'train_args' (训练参数)")
        
        print("\n建议:")
        print("  1. 检查是否使用了最新版的 trainer.py")
        print("  2. 使用 fix_checkpoint.py 修复")
        print("  3. 或者检查 weights/best.pt 是否有这些字段")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_checkpoint.py <checkpoint.pt>")
        sys.exit(1)
    
    analyze(sys.argv[1])
