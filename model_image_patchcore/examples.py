#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PatchCore API 使用示例

展示如何通过Python代码使用PatchCore进行训练和推理
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def example_training():
    """训练示例"""
    print("=" * 50)
    print("示例1: 训练 PatchCore 模型")
    print("=" * 50)
    
    from config import TrainingConfig
    from engine.trainer import PatchCoreTrainer, TrainingCallback
    
    # 1. 创建配置
    config = TrainingConfig()
    
    # 基本设置
    config.dataset_dir = "./data/mvtec/bottle"  # 数据集目录
    config.output_dir = "./output"               # 输出目录
    config.image_size = 256                      # 输入尺寸
    config.device = 'auto'                       # 自动选择GPU/CPU
    
    # Backbone 设置
    config.backbone.name = 'wide_resnet50_2'     # 骨干网络
    config.backbone.layers = ['layer2', 'layer3'] # 特征层
    
    # Memory Bank 设置
    config.memory_bank.coreset_sampling_ratio = 0.01  # 采样率 1%
    config.memory_bank.pca_components = 256           # PCA降维到256
    
    # KNN 设置
    config.knn.k = 9                             # K近邻数
    config.knn.index_type = 'auto'               # 自动选择索引类型
    
    # 导出设置
    config.export.tensorrt_enabled = True        # 导出TensorRT
    config.export.tensorrt_precision = 'fp16'    # FP16精度
    
    # 2. 创建训练器
    trainer = PatchCoreTrainer(config)
    
    # 3. 设置回调 (可选)
    def on_log(message):
        print(message)
    
    def on_phase_start(phase_name, current, total):
        print(f"[{current}/{total}] {phase_name}")
    
    callback = TrainingCallback(
        on_log=on_log,
        on_phase_start=on_phase_start,
    )
    trainer.set_callback(callback)
    
    # 4. 开始训练
    result = trainer.train()
    
    # 5. 查看结果
    if result.success:
        print(f"\n✅ 训练成功!")
        print(f"   导出路径: {result.export_path}")
        print(f"   Memory Bank: {result.memory_bank_size} 特征")
        print(f"   默认阈值: {result.default_threshold:.1f}")
        print(f"   总用时: {result.total_time_seconds:.1f}秒")
    else:
        print(f"\n❌ 训练失败: {result.message}")
    
    return result


def example_inference():
    """推理示例"""
    print("\n" + "=" * 50)
    print("示例2: 使用模型进行推理")
    print("=" * 50)
    
    from inference.predictor import PatchCorePredictor, create_visualization
    
    # 1. 加载模型
    model_path = "./output/exports/patchcore_model.pkg"
    predictor = PatchCorePredictor.from_package(model_path)
    
    # 2. 设置阈值 (可选)
    predictor.set_threshold(50.0)
    
    # 3. 单图预测
    result = predictor.predict("test_image.jpg", return_visualization=True)
    
    print(f"\n检测结果:")
    print(f"   分数: {result.score:.1f}")
    print(f"   状态: {'异常' if result.is_anomaly else '正常'}")
    print(f"   推理时间: {result.inference_time_ms:.1f} ms")
    
    # 4. 获取可视化
    if result.anomaly_map is not None:
        import numpy as np
        from PIL import Image
        
        original = np.array(Image.open("test_image.jpg").convert('RGB'))
        vis = create_visualization(original, result)
        
        # vis 包含: 'original', 'heatmap', 'overlay', 'contour', 'binary'
        print(f"   可视化keys: {list(vis.keys())}")
    
    return result


def example_batch_inference():
    """批量推理示例"""
    print("\n" + "=" * 50)
    print("示例3: 批量推理")
    print("=" * 50)
    
    from inference.predictor import PatchCorePredictor
    from data.dataset import scan_image_directory
    
    # 1. 加载模型
    predictor = PatchCorePredictor.from_package("./output/exports/model.pkg")
    
    # 2. 扫描图像目录
    images = scan_image_directory("./test_images")
    print(f"找到 {len(images)} 张图像")
    
    # 3. 批量预测
    results = predictor.predict_batch(images, batch_size=4)
    
    # 4. 统计
    anomaly_count = sum(1 for r in results if r.is_anomaly)
    print(f"\n统计:")
    print(f"   总计: {len(results)}")
    print(f"   异常: {anomaly_count}")
    print(f"   正常: {len(results) - anomaly_count}")
    
    return results


def example_threshold_calibration():
    """阈值校准示例"""
    print("\n" + "=" * 50)
    print("示例4: 阈值校准")
    print("=" * 50)
    
    import numpy as np
    from evaluation.threshold import IndustrialThresholdCalibrator
    
    # 模拟分数
    good_scores = np.random.normal(30, 10, 100)   # 良品分数
    defect_scores = np.random.normal(70, 15, 20)  # 异常分数
    
    # 1. 创建校准器
    calibrator = IndustrialThresholdCalibrator()
    
    # 2. 仅用良品校准 (工业常见场景)
    result = calibrator.calibrate(good_scores)
    
    print(f"\n校准结果 (仅良品):")
    print(f"   P1: {result['normalization']['p1']:.2f}")
    print(f"   P99: {result['normalization']['p99']:.2f}")
    print(f"   默认阈值: {result['default_threshold']:.1f}")
    
    # 3. 如果有异常样本，可以优化阈值
    result = calibrator.calibrate(good_scores, defect_scores)
    
    print(f"\n校准结果 (包含异常):")
    print(f"   最优F1阈值: {result['thresholds'].get('optimal_f1', 50):.1f}")
    if 'optimal_f1_metrics' in result['thresholds']:
        metrics = result['thresholds']['optimal_f1_metrics']
        print(f"   F1: {metrics.get('f1', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
    
    # 4. 归一化分数
    raw_score = 45.0
    normalized = calibrator.normalize_scores(np.array([raw_score]))[0]
    print(f"\n分数归一化: {raw_score:.1f} → {normalized:.1f}")
    
    return result


def example_template_usage():
    """场景模板示例"""
    print("\n" + "=" * 50)
    print("示例5: 使用场景模板")
    print("=" * 50)
    
    from config import TrainingConfig, SCENE_TEMPLATES
    
    # 查看可用模板
    print("\n可用场景模板:")
    for name, template in SCENE_TEMPLATES.items():
        print(f"   - {name}: {template['name']}")
    
    # 从模板创建配置
    config = TrainingConfig.from_template('high_precision')
    
    print(f"\n高精度模板配置:")
    print(f"   Backbone: {config.backbone.name}")
    print(f"   图像尺寸: {config.image_size}")
    print(f"   采样率: {config.memory_bank.coreset_sampling_ratio}")
    print(f"   PCA维度: {config.memory_bank.pca_components}")
    print(f"   KNN K值: {config.knn.k}")
    
    return config


def example_model_export():
    """模型导出示例"""
    print("\n" + "=" * 50)
    print("示例6: 模型导出与加载")
    print("=" * 50)
    
    from export.exporter import load_patchcore_model
    
    # 加载导出的模型
    model_data = load_patchcore_model("./output/exports/model.pkg")
    
    print(f"\n模型信息:")
    print(f"   Backbone: {model_data['config']['backbone']['name']}")
    print(f"   输入尺寸: {model_data['config']['preprocessing']['input_size']}")
    print(f"   Memory Bank: {len(model_data['features'])} 特征")
    print(f"   特征维度: {model_data['features'].shape[1]}")
    print(f"   默认阈值: {model_data['thresholds'].get('default', 50)}")
    
    return model_data


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("        PatchCore API 使用示例")
    print("=" * 60)
    
    # 注意: 以下示例需要实际的数据和模型文件才能运行
    # 这里仅展示API用法，请根据实际情况修改路径
    
    try:
        # 示例5: 场景模板 (无需数据)
        example_template_usage()
        
        # 示例4: 阈值校准 (使用模拟数据)
        example_threshold_calibration()
        
        print("\n" + "=" * 60)
        print("以下示例需要实际数据，请取消注释后运行:")
        print("=" * 60)
        
        # 示例1: 训练
        # example_training()
        
        # 示例2: 单图推理
        # example_inference()
        
        # 示例3: 批量推理
        # example_batch_inference()
        
        # 示例6: 模型导出
        # example_model_export()
        
    except Exception as e:
        print(f"\n❌ 示例运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
