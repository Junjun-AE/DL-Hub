"""
修复 YOLO Checkpoint 脚本

用于给旧版训练工具生成的 checkpoint 添加 model_importer 需要的字段

使用方法:
    python fix_checkpoint.py input.pt output.pt --nc 16 --imgsz 640

或交互模式:
    python fix_checkpoint.py input.pt output.pt
"""

import argparse
import torch
from pathlib import Path


def analyze_checkpoint(ckpt_path: str) -> dict:
    """分析 checkpoint 内容"""
    print(f"\n📁 加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print(f"\n📋 Checkpoint 字段:")
    for key, value in ckpt.items():
        if key == 'model':
            print(f"  {key}: {type(value).__module__}.{type(value).__name__}")
            # 尝试获取模型信息
            if hasattr(value, 'nc'):
                print(f"    └─ model.nc = {value.nc}")
            if hasattr(value, 'names'):
                print(f"    └─ model.names = {value.names}")
            if hasattr(value, 'yaml'):
                yaml_info = value.yaml if isinstance(value.yaml, dict) else "..."
                if isinstance(yaml_info, dict):
                    print(f"    └─ model.yaml.nc = {yaml_info.get('nc', 'N/A')}")
        elif key == 'names':
            if isinstance(value, dict):
                print(f"  {key}: {dict(list(value.items())[:3])}... ({len(value)} classes)")
            else:
                print(f"  {key}: {value}")
        elif key == 'train_args':
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in list(value.items())[:5]:
                    print(f"    └─ {k}: {v}")
            else:
                print(f"  {key}: {type(value)}")
        elif key == 'custom_metadata':
            print(f"  {key}: ✅ 存在")
        else:
            val_str = str(value)[:50] if value is not None else "None"
            print(f"  {key}: {type(value).__name__} = {val_str}")
    
    return ckpt


def get_model_info(ckpt: dict) -> tuple:
    """从模型对象中提取信息"""
    nc = None
    names = None
    imgsz = 640
    model_name = None
    
    model = ckpt.get('model')
    if model is not None:
        # 尝试从模型获取 nc
        if hasattr(model, 'nc'):
            nc = model.nc
        elif hasattr(model, 'yaml') and isinstance(model.yaml, dict):
            nc = model.yaml.get('nc')
        
        # 尝试从模型获取 names
        if hasattr(model, 'names'):
            names = model.names
        
        # 尝试获取模型名称
        if hasattr(model, 'yaml') and isinstance(model.yaml, dict):
            model_name = model.yaml.get('yaml_file', '')
            if model_name:
                model_name = Path(model_name).stem
    
    # 从 train_args 获取
    train_args = ckpt.get('train_args', {})
    if isinstance(train_args, dict):
        if nc is None:
            nc = train_args.get('nc')
        imgsz = train_args.get('imgsz', 640)
        if model_name is None:
            model_name = train_args.get('model', '')
            if model_name:
                model_name = Path(model_name).stem.replace('.pt', '')
    
    # 从顶层字段获取
    if nc is None:
        nc = ckpt.get('nc')
    if names is None:
        names = ckpt.get('names')
    
    return nc, names, imgsz, model_name


def fix_checkpoint(
    input_path: str,
    output_path: str,
    nc: int = None,
    names: list = None,
    imgsz: int = 640,
    model_name: str = None,
):
    """
    修复 checkpoint，添加 model_importer.py 需要的所有字段
    
    model_importer.py 的 YOLODetectionHandler 需要:
    1. framework = 'ultralytics'  ← can_handle() 识别
    2. _original_model            ← rebuild() 使用
    3. nc, names                  ← get_num_classes() 使用
    4. yaml                       ← get_model_name() 推断版本
    """
    # 加载原始 checkpoint
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # 尝试自动获取信息
    auto_nc, auto_names, auto_imgsz, auto_model_name = get_model_info(ckpt)
    
    # 使用自动获取的值或用户提供的值
    nc = nc or auto_nc
    imgsz = imgsz or auto_imgsz
    model_name = model_name or auto_model_name or 'yolov8n'
    
    if nc is None:
        raise ValueError("无法自动获取类别数，请使用 --nc 参数指定")
    
    # 生成 names
    if names is None:
        if auto_names is not None:
            names = auto_names
        else:
            # 生成默认 names
            names = {i: str(i) for i in range(nc)}
    
    # 确保 names 是字典格式
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    
    print(f"\n🔧 修复信息:")
    print(f"  nc: {nc}")
    print(f"  names: {dict(list(names.items())[:3])}... ({len(names)} classes)")
    print(f"  imgsz: {imgsz}")
    print(f"  model_name: {model_name}")
    
    # ============================================
    # 1. framework 字段 - can_handle() 识别用
    # ============================================
    ckpt['framework'] = 'ultralytics'
    
    # ============================================
    # 2. _original_model 字段 - rebuild() 使用
    # ============================================
    if 'model' in ckpt and ckpt['model'] is not None:
        model_obj = ckpt['model']
        if hasattr(model_obj, 'state_dict'):
            ckpt['_original_model'] = model_obj
            print(f"  ✅ 已添加 _original_model")
    
    # ============================================
    # 3. nc 和 names 字段
    # ============================================
    ckpt['nc'] = nc
    ckpt['names'] = names
    
    # ============================================
    # 4. yaml 字段 - get_model_name() 推断版本
    # ============================================
    model_name_lower = model_name.lower().replace('.pt', '')
    yaml_config = {
        'nc': nc,
        'yaml_file': f"{model_name_lower}.yaml",
    }
    
    if 'yaml' in ckpt and isinstance(ckpt['yaml'], dict):
        yaml_config.update(ckpt['yaml'])
        yaml_config['nc'] = nc
    
    ckpt['yaml'] = yaml_config
    
    # ============================================
    # 5. train_args 字段
    # ============================================
    if 'train_args' not in ckpt or ckpt['train_args'] is None:
        ckpt['train_args'] = {}
    
    if not isinstance(ckpt['train_args'], dict):
        ckpt['train_args'] = {}
    
    ckpt['train_args'].update({
        'nc': nc,
        'imgsz': imgsz,
        'model': f"{model_name}.pt" if not model_name.endswith('.pt') else model_name,
    })
    
    # ============================================
    # 6. custom_metadata 字段 (预处理信息)
    # ============================================
    class_names_list = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    
    ckpt['custom_metadata'] = {
        'model_name': model_name.replace('.pt', ''),
        'num_classes': nc,
        'class_names': class_names_list,
        'input_size': imgsz,
        'input_spec': {
            'shape': (1, 3, imgsz, imgsz),
            'color_format': 'RGB',
            'pixel_range': (0, 255),
            'normalize_method': 'divide_255',
            'normalize_mean': (0.0, 0.0, 0.0),
            'normalize_std': (1.0, 1.0, 1.0),
            'value_range': (0.0, 1.0),
            'letterbox_color': (114, 114, 114),
        },
    }
    
    # 保存
    torch.save(ckpt, output_path)
    print(f"\n✅ 已保存修复后的 checkpoint: {output_path}")
    
    # 验证
    print(f"\n📋 验证修复后的 checkpoint:")
    verify_ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
    print(f"  framework: {verify_ckpt.get('framework')}")
    print(f"  nc: {verify_ckpt.get('nc')}")
    print(f"  names: {verify_ckpt.get('names') is not None} ({len(verify_ckpt.get('names', {}))} classes)")
    print(f"  yaml: {verify_ckpt.get('yaml') is not None}")
    print(f"  _original_model: {verify_ckpt.get('_original_model') is not None}")
    print(f"  train_args: {verify_ckpt.get('train_args') is not None}")
    print(f"  custom_metadata: {verify_ckpt.get('custom_metadata') is not None}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='修复 YOLO Checkpoint')
    parser.add_argument('input', help='输入 checkpoint 路径')
    parser.add_argument('output', nargs='?', help='输出 checkpoint 路径 (默认覆盖原文件)')
    parser.add_argument('--nc', type=int, help='类别数')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸 (默认 640)')
    parser.add_argument('--model', type=str, help='模型名称 (如 yolov8n)')
    parser.add_argument('--analyze', action='store_true', help='只分析，不修复')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output or input_path.replace('.pt', '_fixed.pt')
    
    if args.analyze:
        analyze_checkpoint(input_path)
        return
    
    # 先分析
    ckpt = analyze_checkpoint(input_path)
    
    # 尝试自动获取 nc
    nc, names, imgsz, model_name = get_model_info(ckpt)
    
    if args.nc:
        nc = args.nc
    if args.imgsz:
        imgsz = args.imgsz
    if args.model:
        model_name = args.model
    
    # 如果无法自动获取，提示用户
    if nc is None:
        print("\n⚠️  无法自动获取类别数")
        nc_input = input("请输入类别数 (nc): ")
        nc = int(nc_input)
    
    if model_name is None:
        print("\n⚠️  无法自动获取模型名称")
        model_input = input("请输入模型名称 (如 yolov8n, 直接回车使用默认): ")
        model_name = model_input.strip() or 'yolo'
    
    # 修复
    fix_checkpoint(
        input_path=input_path,
        output_path=output_path,
        nc=nc,
        names=names,
        imgsz=imgsz,
        model_name=model_name,
    )


if __name__ == '__main__':
    main()
