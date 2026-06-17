#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PatchCore CLI 命令行工具

用法:
    python cli.py train --data ./dataset --output ./output
    python cli.py predict --model ./model.pkg --image ./test.jpg
    python cli.py batch --model ./model.pkg --input ./images --output ./results
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_command(args):
    """训练命令"""
    from config import TrainingConfig
    from engine.trainer import PatchCoreTrainer, TrainingCallback
    
    print("=" * 50)
    print("🚀 PatchCore 训练")
    print("=" * 50)
    
    # 创建配置
    config = TrainingConfig()
    config.dataset_dir = args.data
    config.output_dir = args.output
    
    if args.backbone:
        config.backbone.name = args.backbone
    if args.image_size:
        config.image_size = args.image_size
    if args.coreset_ratio:
        config.memory_bank.coreset_sampling_ratio = args.coreset_ratio
    if args.device:
        config.device = args.device
    
    # 验证配置
    errors = config.validate()
    if errors:
        print("❌ 配置错误:")
        for err in errors:
            print(f"   - {err}")
        return 1
    
    # 创建训练器
    trainer = PatchCoreTrainer(config)
    
    # 设置回调
    def on_log(msg):
        print(msg)
    
    callback = TrainingCallback(on_log=on_log)
    trainer.set_callback(callback)
    
    # 训练
    result = trainer.train()
    
    if result.success:
        print("\n✅ 训练成功!")
        print(f"   模型路径: {result.export_path}")
        print(f"   Memory Bank: {result.memory_bank_size} 特征")
        print(f"   默认阈值: {result.default_threshold:.1f}")
        return 0
    else:
        print(f"\n❌ 训练失败: {result.message}")
        return 1


def predict_command(args):
    """预测命令"""
    from inference.predictor import PatchCorePredictor, create_visualization
    from PIL import Image
    import numpy as np
    
    print("=" * 50)
    print("🔍 PatchCore 单图预测")
    print("=" * 50)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    predictor = PatchCorePredictor.from_package(args.model)
    
    if args.threshold:
        predictor.set_threshold(args.threshold)
    
    # 预测
    print(f"检测图像: {args.image}")
    result = predictor.predict(args.image, return_visualization=True)
    
    # 显示结果
    status = "🔴 异常" if result.is_anomaly else "🟢 正常"
    print(f"\n检测结果:")
    print(f"   状态: {status}")
    print(f"   分数: {result.score:.1f} / 100")
    print(f"   阈值: {predictor.get_threshold():.1f}")
    print(f"   推理时间: {result.inference_time_ms:.1f} ms")
    
    # 保存可视化
    if args.output:
        import cv2
        
        original = np.array(Image.open(args.image).convert('RGB'))
        vis = create_visualization(original, result)
        
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stem = Path(args.image).stem
        
        if 'heatmap' in vis:
            cv2.imwrite(str(output_path / f'{stem}_heatmap.jpg'),
                       cv2.cvtColor(vis['heatmap'], cv2.COLOR_RGB2BGR))
        
        if 'overlay' in vis:
            cv2.imwrite(str(output_path / f'{stem}_overlay.jpg'),
                       cv2.cvtColor(vis['overlay'], cv2.COLOR_RGB2BGR))
        
        print(f"\n可视化已保存到: {output_path}")
    
    return 0 if not result.is_anomaly else 1


def batch_command(args):
    """批量预测命令"""
    from inference.predictor import PatchCorePredictor, create_visualization
    from data.dataset import scan_image_directory
    from PIL import Image
    import numpy as np
    import pandas as pd
    import cv2
    from datetime import datetime
    from tqdm import tqdm
    
    print("=" * 50)
    print("🔄 PatchCore 批量预测")
    print("=" * 50)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    predictor = PatchCorePredictor.from_package(args.model)
    
    if args.threshold:
        predictor.set_threshold(args.threshold)
    
    # 扫描图像
    images = scan_image_directory(args.input)
    print(f"找到 {len(images)} 张图像")
    
    if not images:
        print("❌ 没有找到图像文件")
        return 1
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'anomaly').mkdir(exist_ok=True)
    (output_path / 'normal').mkdir(exist_ok=True)
    
    if args.save_vis:
        (output_path / 'visualizations').mkdir(exist_ok=True)
    
    # 批量预测
    results = []
    anomaly_count = 0
    
    for img_path in tqdm(images, desc="检测中"):
        result = predictor.predict(img_path, return_visualization=args.save_vis)
        
        filename = Path(img_path).name
        status = "异常" if result.is_anomaly else "正常"
        
        results.append({
            'filename': filename,
            'score': result.score,
            'is_anomaly': result.is_anomaly,
            'status': status,
        })
        
        if result.is_anomaly:
            anomaly_count += 1
        
        # 复制到分类目录
        import shutil
        dest = output_path / ('anomaly' if result.is_anomaly else 'normal') / filename
        shutil.copy(img_path, dest)
        
        # 保存可视化
        if args.save_vis and result.anomaly_map is not None:
            original = np.array(Image.open(img_path).convert('RGB'))
            vis = create_visualization(original, result)
            
            if 'overlay' in vis:
                vis_path = output_path / 'visualizations' / f'{Path(filename).stem}_overlay.jpg'
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis['overlay'], cv2.COLOR_RGB2BGR))
    
    # 保存CSV
    df = pd.DataFrame(results)
    csv_path = output_path / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 统计
    print(f"\n检测完成!")
    print(f"   总计: {len(results)} 张")
    print(f"   异常: {anomaly_count} 张 ({anomaly_count/len(results)*100:.1f}%)")
    print(f"   正常: {len(results)-anomaly_count} 张")
    print(f"   结果已保存到: {output_path}")
    print(f"   CSV报告: {csv_path}")
    
    return 0


def gui_command(args):
    """启动GUI"""
    from app import create_app
    
    print("=" * 50)
    print("🖥️ 启动 PatchCore GUI")
    print("=" * 50)
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def main():
    parser = argparse.ArgumentParser(
        description='PatchCore 工业异常检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', '-d', required=True, help='数据集目录')
    train_parser.add_argument('--output', '-o', default='./output', help='输出目录')
    train_parser.add_argument('--backbone', '-b', help='Backbone网络')
    train_parser.add_argument('--image-size', '-s', type=int, help='输入尺寸')
    train_parser.add_argument('--coreset-ratio', '-r', type=float, help='CoreSet采样率')
    train_parser.add_argument('--device', help='计算设备')
    
    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='单图预测')
    predict_parser.add_argument('--model', '-m', required=True, help='模型路径')
    predict_parser.add_argument('--image', '-i', required=True, help='输入图像')
    predict_parser.add_argument('--output', '-o', help='输出目录')
    predict_parser.add_argument('--threshold', '-t', type=float, help='检测阈值')
    
    # batch 命令
    batch_parser = subparsers.add_parser('batch', help='批量预测')
    batch_parser.add_argument('--model', '-m', required=True, help='模型路径')
    batch_parser.add_argument('--input', '-i', required=True, help='输入目录')
    batch_parser.add_argument('--output', '-o', default='./batch_results', help='输出目录')
    batch_parser.add_argument('--threshold', '-t', type=float, help='检测阈值')
    batch_parser.add_argument('--save-vis', action='store_true', help='保存可视化')
    
    # gui 命令
    gui_parser = subparsers.add_parser('gui', help='启动GUI界面')
    gui_parser.add_argument('--host', default='127.0.0.1', help='服务器地址')
    gui_parser.add_argument('--port', type=int, default=7860, help='服务器端口')
    gui_parser.add_argument('--share', action='store_true', help='创建公开链接')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'predict':
        return predict_command(args)
    elif args.command == 'batch':
        return batch_command(args)
    elif args.command == 'gui':
        return gui_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
