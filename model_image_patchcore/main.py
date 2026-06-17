#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PatchCore 工业级异常检测系统

主入口文件
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PatchCore 工业级异常检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动GUI界面
  python main.py gui
  
  # 命令行训练
  python main.py train --data ./dataset --output ./output
  
  # 批量推理
  python main.py predict --model ./model.pkg --input ./images --output ./results
  
  # 导出模型
  python main.py export --model ./output --format tensorrt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # GUI命令
    gui_parser = subparsers.add_parser('gui', help='启动图形界面')
    gui_parser.add_argument('--port', type=int, default=7860, help='服务端口')
    gui_parser.add_argument('--share', action='store_true', help='生成公开链接')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', '-d', required=True, help='数据集目录')
    train_parser.add_argument('--output', '-o', default='./output', help='输出目录')
    train_parser.add_argument('--backbone', default='wide_resnet50_2', 
                             choices=['wide_resnet50_2', 'resnet50', 'resnet18'],
                             help='Backbone网络')
    train_parser.add_argument('--image-size', type=int, default=256, help='输入图像尺寸')
    train_parser.add_argument('--coreset-ratio', type=float, default=0.01, help='CoreSet采样率')
    train_parser.add_argument('--pca-dim', type=int, default=256, help='PCA降维维度')
    train_parser.add_argument('--knn-k', type=int, default=9, help='KNN近邻数')
    train_parser.add_argument('--device', default='auto', help='计算设备')
    train_parser.add_argument('--no-tensorrt', action='store_true', help='禁用TensorRT导出')
    
    # 推理命令
    predict_parser = subparsers.add_parser('predict', help='批量推理')
    predict_parser.add_argument('--model', '-m', required=True, help='模型路径')
    predict_parser.add_argument('--input', '-i', required=True, help='输入图片目录')
    predict_parser.add_argument('--output', '-o', default='./results', help='输出目录')
    predict_parser.add_argument('--threshold', type=float, default=50, help='检测阈值')
    predict_parser.add_argument('--batch-size', type=int, default=4, help='批处理大小')
    predict_parser.add_argument('--save-heatmap', action='store_true', help='保存热力图')
    
    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--model', '-m', required=True, help='模型目录')
    export_parser.add_argument('--output', '-o', help='输出路径')
    export_parser.add_argument('--format', choices=['pkg', 'onnx', 'tensorrt'], 
                              default='pkg', help='导出格式')
    export_parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'],
                              default='fp16', help='TensorRT精度')
    
    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--model', '-m', required=True, help='模型路径')
    eval_parser.add_argument('--data', '-d', required=True, help='验证数据目录')
    eval_parser.add_argument('--threshold', type=float, help='指定阈值（不指定则自动优化）')
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        run_gui(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'predict':
        run_predict(args)
    elif args.command == 'export':
        run_export(args)
    elif args.command == 'eval':
        run_eval(args)
    else:
        parser.print_help()


def run_gui(args):
    """启动GUI界面"""
    from gui.app import create_app
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


def run_train(args):
    """命令行训练"""
    from config import TrainingConfig
    from engine.trainer import PatchCoreTrainer, TrainingCallback
    
    # 创建配置
    config = TrainingConfig()
    config.dataset_dir = args.data
    config.output_dir = args.output
    config.image_size = args.image_size
    config.device = args.device
    config.backbone.name = args.backbone
    config.memory_bank.coreset_sampling_ratio = args.coreset_ratio
    config.memory_bank.pca_components = args.pca_dim
    config.knn.k = args.knn_k
    config.export.tensorrt_enabled = not args.no_tensorrt
    
    # 创建训练器
    trainer = PatchCoreTrainer(config)
    
    # 设置回调
    callback = TrainingCallback(
        on_log=lambda msg: print(msg),
        on_phase_start=lambda name, cur, total: print(f"\n[{cur}/{total}] {name}"),
    )
    trainer.set_callback(callback)
    
    # 训练
    result = trainer.train()
    
    if result.success:
        print(f"\n✅ 训练完成!")
        print(f"   用时: {result.total_time_seconds:.1f}s")
        print(f"   模型: {result.export_path}")
    else:
        print(f"\n❌ 训练失败: {result.message}")
        sys.exit(1)


def run_predict(args):
    """批量推理"""
    import json
    import csv
    from pathlib import Path
    from inference.predictor import PatchCorePredictor, create_visualization
    from data.dataset import scan_image_directory
    import cv2
    
    # 加载模型
    print(f"加载模型: {args.model}")
    predictor = PatchCorePredictor.from_package(args.model)
    predictor.set_threshold(args.threshold)
    
    # 获取图片列表
    image_paths = scan_image_directory(args.input)
    print(f"找到 {len(image_paths)} 张图片")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_heatmap:
        (output_dir / 'heatmaps').mkdir(exist_ok=True)
    
    # 推理
    results = []
    anomaly_count = 0
    
    for i, img_path in enumerate(image_paths):
        result = predictor.predict(img_path, return_visualization=args.save_heatmap)
        
        record = {
            'filename': Path(img_path).name,
            'score': result.score,
            'is_anomaly': result.is_anomaly,
            'inference_time_ms': result.inference_time_ms,
        }
        results.append(record)
        
        if result.is_anomaly:
            anomaly_count += 1
            
            if args.save_heatmap and result.anomaly_map is not None:
                vis = create_visualization(img_path, result)
                heatmap_path = output_dir / 'heatmaps' / f'{Path(img_path).stem}_heatmap.jpg'
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(vis['heatmap'], cv2.COLOR_RGB2BGR))
        
        # 进度
        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
            print(f"进度: {i+1}/{len(image_paths)} | 异常: {anomaly_count}")
    
    # 保存结果
    csv_path = output_dir / 'detection_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'score', 'is_anomaly', 'inference_time_ms'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ 检测完成!")
    print(f"   总数: {len(results)}")
    print(f"   异常: {anomaly_count} ({anomaly_count/len(results):.1%})")
    print(f"   结果: {csv_path}")


def run_export(args):
    """导出模型"""
    print(f"导出功能开发中...")
    # TODO: 实现导出命令


def run_eval(args):
    """评估模型"""
    import numpy as np
    from inference.predictor import PatchCorePredictor
    from data.dataset import AnomalyDataset
    from evaluation.threshold import IndustrialThresholdCalibrator
    
    # 加载模型
    print(f"加载模型: {args.model}")
    predictor = PatchCorePredictor.from_package(args.model)
    
    # 加载数据
    print(f"加载数据: {args.data}")
    dataset = AnomalyDataset(
        root_dir=args.data,
        image_size=predictor.config['preprocessing']['input_size'][0],
        split='all',
    )
    
    # 推理
    good_scores = []
    defect_scores = []
    
    for i, sample in enumerate(dataset):
        result = predictor.predict(sample['image_path'], return_visualization=False)
        
        if sample['label'] == 0:
            good_scores.append(result.score)
        else:
            defect_scores.append(result.score)
        
        if (i + 1) % 50 == 0:
            print(f"进度: {i+1}/{len(dataset)}")
    
    good_scores = np.array(good_scores)
    defect_scores = np.array(defect_scores)
    
    # 设置阈值
    if args.threshold is not None:
        threshold = args.threshold
    else:
        # 自动优化
        calibrator = IndustrialThresholdCalibrator()
        calibrator.calibrate(good_scores, defect_scores)
        threshold = calibrator.threshold_presets.get('optimal_f1', 50.0)
    
    # 计算指标
    fp = np.sum(good_scores >= threshold)
    fpr = fp / len(good_scores)
    
    print(f"\n📊 评估结果 (阈值={threshold:.1f})")
    print(f"\n良品 ({len(good_scores)}张):")
    print(f"  误判: {fp} ({fpr:.2%})")
    
    if len(defect_scores) > 0:
        tp = np.sum(defect_scores >= threshold)
        recall = tp / len(defect_scores)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n异常 ({len(defect_scores)}张):")
        print(f"  检出: {tp} ({recall:.2%})")
        print(f"\n综合指标:")
        print(f"  精确率: {precision:.2%}")
        print(f"  F1分数: {f1:.4f}")


if __name__ == '__main__':
    main()
