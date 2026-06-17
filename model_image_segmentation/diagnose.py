"""
诊断工具 - 分析mIoU虚高和推理效果差的问题
运行方式: python diagnose.py <数据集目录>
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json


def diagnose_dataset(dataset_dir: str):
    """诊断数据集问题"""
    dataset_path = Path(dataset_dir)
    
    print("=" * 60)
    print("🔍 数据集诊断报告")
    print("=" * 60)
    
    # 1. 检查数据集结构
    print("\n📁 1. 数据集结构")
    train_images_dir = dataset_path / 'images' / 'train'
    val_images_dir = dataset_path / 'images' / 'val'
    train_masks_dir = dataset_path / 'masks' / 'train'
    val_masks_dir = dataset_path / 'masks' / 'val'
    
    train_images = list(train_images_dir.glob('*.[jpJP][npNP][gG]*')) if train_images_dir.exists() else []
    val_images = list(val_images_dir.glob('*.[jpJP][npNP][gG]*')) if val_images_dir.exists() else []
    train_masks = list(train_masks_dir.glob('*.png')) if train_masks_dir.exists() else []
    val_masks = list(val_masks_dir.glob('*.png')) if val_masks_dir.exists() else []
    
    print(f"   训练集图像: {len(train_images)} 张")
    print(f"   验证集图像: {len(val_images)} 张")
    print(f"   训练集Mask: {len(train_masks)} 个")
    print(f"   验证集Mask: {len(val_masks)} 个")
    
    # 【重要检查】验证集太小
    if len(val_images) < 5:
        print(f"\n   ⚠️ 警告: 验证集只有 {len(val_images)} 张图像，太少了！")
        print(f"      mIoU 100% 可能是因为验证集太简单")
        print(f"      建议: 增加验证集数量到至少 10-20 张")
    
    # 2. 检查训练集和验证集是否重叠
    print("\n📊 2. 数据集重叠检查")
    train_stems = set(img.stem for img in train_images)
    val_stems = set(img.stem for img in val_images)
    overlap = train_stems & val_stems
    
    if overlap:
        print(f"   ❌ 发现 {len(overlap)} 个重叠文件!")
        print(f"      重叠文件: {list(overlap)[:5]}...")
        print(f"      这会导致 mIoU 虚高！")
    else:
        print(f"   ✅ 训练集和验证集无重叠")
    
    # 3. 分析Mask中的类别分布
    print("\n📈 3. 类别分布分析")
    
    all_class_pixels = {}  # 每个类别的总像素数
    all_background_ratio = []  # 每张图的背景比例
    
    # 分析训练集
    print("   分析训练集...")
    for mask_path in train_masks[:20]:  # 只分析前20张
        try:
            mask = np.array(Image.open(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            
            for cls_id, count in zip(unique, counts):
                if cls_id not in all_class_pixels:
                    all_class_pixels[cls_id] = 0
                all_class_pixels[cls_id] += count
            
            total_pixels = mask.shape[0] * mask.shape[1]
            bg_pixels = np.sum(mask == 255)
            all_background_ratio.append(bg_pixels / total_pixels * 100)
        except Exception as e:
            print(f"   ⚠️ 无法读取 {mask_path.name}: {e}")
    
    # 统计结果
    total_all = sum(all_class_pixels.values())
    print(f"\n   类别像素占比:")
    for cls_id in sorted(all_class_pixels.keys()):
        ratio = all_class_pixels[cls_id] / total_all * 100
        cls_name = "背景" if cls_id == 255 else f"类别{cls_id}"
        print(f"      {cls_name}: {ratio:.2f}%")
    
    # 背景占比分析
    if all_background_ratio:
        avg_bg = np.mean(all_background_ratio)
        print(f"\n   平均背景占比: {avg_bg:.1f}%")
        
        if avg_bg > 95:
            print(f"   ⚠️ 警告: 背景占比过高 ({avg_bg:.1f}%)")
            print(f"      这意味着缺陷区域很小，即使预测不准，mIoU也可能很高")
            print(f"      建议: 使用 mDice 或 per-class IoU 更能反映真实效果")
    
    # 4. 检查模型元数据
    print("\n🔧 4. 模型配置检查")
    metadata_path = dataset_path.parent / 'model_metadata.json'
    
    # 查找输出目录中的元数据
    output_dirs = list(Path('.').glob('output/SegFormer_*'))
    if output_dirs:
        latest_output = sorted(output_dirs)[-1]
        metadata_path = latest_output / 'model_metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"   模型: {metadata.get('model_name', 'Unknown')}")
        print(f"   类别数: {metadata.get('num_classes', 'Unknown')}")
        print(f"   输入尺寸: {metadata.get('img_size', 'Unknown')}")
        print(f"   预处理: {metadata.get('preprocess', 'Unknown')}")
        
        metrics = metadata.get('metrics', {})
        print(f"\n   训练指标:")
        print(f"      最佳 mIoU: {metrics.get('best_mIoU', 'N/A')}")
        print(f"      最佳 Epoch: {metrics.get('best_epoch', 'N/A')}")
    else:
        print(f"   未找到模型元数据文件")
    
    # 5. 建议
    print("\n💡 5. 诊断建议")
    print("=" * 60)
    
    issues_found = False
    
    if len(val_images) < 5:
        issues_found = True
        print("""
   【问题】验证集太小
   
   原因: 只有少量验证图像时，模型很容易在这些图上表现完美
   
   解决方案:
   1. 增加验证集数量到 10-20 张
   2. 使用不同场景/条件的图像作为验证集
   3. 确保验证集具有代表性
""")
    
    if overlap:
        issues_found = True
        print("""
   【问题】训练集和验证集重叠
   
   原因: 相同的图像同时出现在训练和验证集中
   
   解决方案:
   1. 重新划分数据集，确保无重叠
   2. 删除验证集中与训练集相同的图像
""")
    
    if all_background_ratio and np.mean(all_background_ratio) > 95:
        issues_found = True
        print("""
   【问题】背景占比过高（缺陷区域太小）
   
   原因: 当缺陷只占图像的很小部分时，即使缺陷预测完全错误，
         整体mIoU仍可能很高（因为背景预测正确）
   
   解决方案:
   1. 关注 per-class IoU（逐类别IoU），特别是缺陷类别
   2. 使用 mDice 作为辅助指标
   3. 裁剪缺陷区域进行局部评估
""")
    
    if not issues_found:
        print("""
   未发现明显问题。如果推理效果仍然差，请检查:
   
   1. 测试图像是否与训练数据分布一致（光照、角度、分辨率）
   2. 使用的模型文件是否正确（best_model.pth）
   3. 预处理参数是否一致（img_size=512）
""")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python diagnose.py <数据集目录>")
        print("例如: python diagnose.py ./mmseg_dataset")
        sys.exit(1)
    
    diagnose_dataset(sys.argv[1])
