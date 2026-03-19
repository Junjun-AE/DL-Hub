# -*- coding: utf-8 -*-
"""
DL-Hub 应用手动测试脚本
======================
用于直接测试异常检测应用，查看具体错误信息

使用方法：
1. 打开命令提示符 (cmd)
2. cd 到 Deep_learning_tools2 目录
3. 运行: python test_patchcore.py

如果要测试OCR：
python test_patchcore.py --app ocr
"""

import sys
import os
from pathlib import Path

def test_app(app_type='anomaly'):
    """直接测试应用启动"""
    
    base_dir = Path(__file__).parent
    
    # 应用路径配置
    app_configs = {
        'anomaly': {
            'name': 'PatchCore 异常检测',
            'path': base_dir / 'model_image_patchcore' / 'app.py',
        },
        'ocr': {
            'name': 'OCR 识别',
            'path': base_dir / 'model_image_ocr' / 'app.py',
        },
        'classification': {
            'name': '图像分类',
            'path': base_dir / 'model_image_classification' / 'app.py',
        },
        'detection': {
            'name': '目标检测',
            'path': base_dir / 'model_image_detection' / 'app.py',
        },
        'segmentation': {
            'name': '语义分割',
            'path': base_dir / 'model_image_segmentation' / 'app.py',
        },
    }
    
    if app_type not in app_configs:
        print(f"未知的应用类型: {app_type}")
        print(f"支持的类型: {list(app_configs.keys())}")
        return
    
    config = app_configs[app_type]
    app_path = config['path']
    app_name = config['name']
    
    print("=" * 60)
    print(f" 测试 {app_name}")
    print("=" * 60)
    
    if not app_path.exists():
        print(f"❌ 应用文件不存在: {app_path}")
        return
    
    print(f"✓ 应用文件存在: {app_path}")
    print(f"\n正在启动应用...")
    print("-" * 60)
    
    # 创建临时任务目录
    task_dir = base_dir / 'test_temp_task'
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    os.environ['DLHUB_TASK_DIR'] = str(task_dir)
    os.environ['DLHUB_TASK_ID'] = 'test_task'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 切换到应用目录
    os.chdir(app_path.parent)
    
    # 添加应用目录到路径
    sys.path.insert(0, str(app_path.parent))
    
    # 尝试导入并运行
    try:
        print(f"\n>>> 切换到目录: {app_path.parent}")
        print(f">>> 执行: python {app_path.name} --port 7865")
        print("-" * 60)
        
        # 直接执行脚本
        import subprocess
        result = subprocess.run(
            [sys.executable, str(app_path), '--port', '7865'],
            cwd=str(app_path.parent),
            env=os.environ.copy()
        )
        
        print("-" * 60)
        print(f"应用退出，返回码: {result.returncode}")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='测试DL-Hub应用')
    parser.add_argument('--app', type=str, default='anomaly',
                       choices=['anomaly', 'ocr', 'classification', 'detection', 'segmentation'],
                       help='要测试的应用类型')
    args = parser.parse_args()
    
    test_app(args.app)
