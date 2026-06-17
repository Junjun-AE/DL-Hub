"""
数据集验证器 - 验证LabelMe格式和YOLO格式数据集
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Set
from dataclasses import dataclass


# 支持的图像格式
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}


@dataclass
class LabelMeDatasetInfo:
    """LabelMe数据集信息"""
    is_valid: bool
    message: str
    images_dir: str = ""
    jsons_dir: str = ""
    total_images: int = 0
    total_jsons: int = 0
    matched_pairs: int = 0
    num_classes: int = 0
    class_ids: List[int] = None
    total_boxes: int = 0
    boxes_per_class: Dict[int, int] = None
    
    def __post_init__(self):
        if self.class_ids is None:
            self.class_ids = []
        if self.boxes_per_class is None:
            self.boxes_per_class = {}


def is_image_file(filename: str) -> bool:
    """检查文件是否是支持的图像格式"""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def validate_labelme_dataset(
    images_dir: str,
    jsons_dir: str,
) -> LabelMeDatasetInfo:
    """
    验证LabelMe格式数据集
    
    Args:
        images_dir: 图像文件夹路径
        jsons_dir: JSON标注文件夹路径
    
    Returns:
        LabelMeDatasetInfo对象
    """
    images_path = Path(images_dir)
    jsons_path = Path(jsons_dir)
    
    # 检查路径是否存在
    if not images_path.exists():
        return LabelMeDatasetInfo(
            is_valid=False,
            message=f"❌ 图像文件夹不存在: {images_dir}",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
        )
    
    if not jsons_path.exists():
        return LabelMeDatasetInfo(
            is_valid=False,
            message=f"❌ JSON文件夹不存在: {jsons_dir}",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
        )
    
    if not images_path.is_dir():
        return LabelMeDatasetInfo(
            is_valid=False,
            message=f"❌ 图像路径不是文件夹: {images_dir}",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
        )
    
    if not jsons_path.is_dir():
        return LabelMeDatasetInfo(
            is_valid=False,
            message=f"❌ JSON路径不是文件夹: {jsons_dir}",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
        )
    
    # 收集图像文件
    image_stems = {}  # stem -> full path
    for f in images_path.iterdir():
        if f.is_file() and is_image_file(f.name):
            image_stems[f.stem] = f
    
    total_images = len(image_stems)
    
    if total_images == 0:
        return LabelMeDatasetInfo(
            is_valid=False,
            message="❌ 图像文件夹中没有找到支持的图像文件",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
        )
    
    # 收集JSON文件并匹配
    json_files = list(jsons_path.glob('*.json'))
    total_jsons = len(json_files)
    
    if total_jsons == 0:
        return LabelMeDatasetInfo(
            is_valid=False,
            message="❌ JSON文件夹中没有找到JSON文件",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
            total_images=total_images,
        )
    
    # 统计匹配和标注信息
    matched_pairs = 0
    class_ids: Set[int] = set()
    boxes_per_class: Dict[int, int] = {}
    total_boxes = 0
    invalid_labels = []
    
    for json_file in json_files:
        stem = json_file.stem
        
        if stem not in image_stems:
            continue
        
        matched_pairs += 1
        
        # 解析JSON获取标注信息
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for shape in data.get('shapes', []):
                if shape.get('shape_type') != 'rectangle':
                    continue
                
                label = shape.get('label', '')
                try:
                    class_id = int(label)
                    class_ids.add(class_id)
                    boxes_per_class[class_id] = boxes_per_class.get(class_id, 0) + 1
                    total_boxes += 1
                except ValueError:
                    if label not in invalid_labels:
                        invalid_labels.append(label)
        except Exception as e:
            pass  # 跳过无法解析的JSON
    
    if matched_pairs == 0:
        return LabelMeDatasetInfo(
            is_valid=False,
            message="❌ 没有找到匹配的图像-JSON配对\n   请确保图像和JSON文件名对应（如: 1.jpg 对应 1.json）",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
            total_images=total_images,
            total_jsons=total_jsons,
        )
    
    if total_boxes == 0:
        return LabelMeDatasetInfo(
            is_valid=False,
            message="❌ 没有找到有效的标注框",
            images_dir=images_dir,
            jsons_dir=jsons_dir,
            total_images=total_images,
            total_jsons=total_jsons,
            matched_pairs=matched_pairs,
        )
    
    # 统计无标注图像（负样本/良品）
    json_stems = {json_file.stem for json_file in json_files}
    negative_samples = sum(1 for stem in image_stems if stem not in json_stems)
    
    # 构建成功消息
    sorted_class_ids = sorted(class_ids)
    class_str = ', '.join(str(c) for c in sorted_class_ids[:10])
    if len(sorted_class_ids) > 10:
        class_str += f'... (共{len(sorted_class_ids)}个)'
    
    message = (
        f"✅ 数据集验证通过\n"
        f"   📁 图像文件夹: {images_dir}\n"
        f"   📁 JSON文件夹: {jsons_dir}\n"
        f"   🖼️ 图像数量: {total_images}\n"
        f"   📄 JSON数量: {total_jsons}\n"
        f"   🔗 有标注: {matched_pairs} 张\n"
        f"   ✨ 无标注(良品): {negative_samples} 张\n"
        f"   📊 类别数: {len(class_ids)}\n"
        f"   📦 标注框: {total_boxes}\n"
        f"   📋 类别ID: {class_str}"
    )
    
    if invalid_labels:
        message += f"\n   ⚠️ 跳过非数字标签: {', '.join(invalid_labels[:5])}"
    
    return LabelMeDatasetInfo(
        is_valid=True,
        message=message,
        images_dir=images_dir,
        jsons_dir=jsons_dir,
        total_images=total_images,
        total_jsons=total_jsons,
        matched_pairs=matched_pairs,
        num_classes=len(class_ids),
        class_ids=sorted_class_ids,
        total_boxes=total_boxes,
        boxes_per_class=boxes_per_class,
    )


def quick_validate_paths(images_dir: str, jsons_dir: str) -> Tuple[bool, str]:
    """
    快速验证路径是否有效
    
    Args:
        images_dir: 图像文件夹路径
        jsons_dir: JSON文件夹路径
    
    Returns:
        (是否有效, 消息)
    """
    if not images_dir or not images_dir.strip():
        return False, "⏳ 请输入图像文件夹路径"
    
    if not jsons_dir or not jsons_dir.strip():
        return False, "⏳ 请输入JSON文件夹路径"
    
    images_path = Path(images_dir.strip())
    jsons_path = Path(jsons_dir.strip())
    
    if not images_path.exists():
        return False, f"❌ 图像文件夹不存在: {images_dir}"
    
    if not jsons_path.exists():
        return False, f"❌ JSON文件夹不存在: {jsons_dir}"
    
    return True, "✅ 路径有效，点击验证按钮进行详细检查"
