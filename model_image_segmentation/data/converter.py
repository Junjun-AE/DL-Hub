"""
LabelMe JSON → MMSegmentation 格式转换器
支持自动划分训练/验证集
将polygon标注转换为分割mask (PNG格式)
背景像素值为255 (ignore_index)
"""

import json
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ConversionResult:
    """转换结果"""
    success: bool
    message: str
    output_dir: str = ""
    total_images: int = 0
    train_images: int = 0
    val_images: int = 0
    num_classes: int = 0
    actual_class_count: int = 0
    class_ids: List[int] = None
    class_names: List[str] = None
    negative_samples: int = 0
    
    def __post_init__(self):
        if self.class_ids is None:
            self.class_ids = []
        if self.class_names is None:
            self.class_names = []


class LabelMeToMMSegConverter:
    """
    LabelMe JSON格式转换为MMSegmentation格式
    
    输入结构:
        images/           # 图像文件夹
            1.jpg
            2.png
            3.jpg         # 可以没有对应的JSON（作为负样本/良品）
        jsons/            # JSON标注文件夹
            1.json
            2.json
    
    输出结构 (MMSeg格式):
        dataset/
            images/
                train/
                val/
            masks/
                train/    # PNG格式，像素值=类别ID，背景=255
                val/
            splits/
                train.txt
                val.txt
            class_info.txt
    
    类别ID处理:
    - 背景像素值 = 255 (ignore_index)
    - 缺陷类别从0开始: 0, 1, 2, ...
    - 多边形重叠时，后绘制覆盖先绘制
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    def __init__(
        self,
        images_dir: str,
        jsons_dir: str,
        output_dir: str,
        val_split: float = 0.2,
        seed: int = 42,
        num_classes: int = 0,
        include_negative_samples: bool = True,
    ):
        """
        初始化转换器
        
        Args:
            images_dir: 图像文件夹路径
            jsons_dir: JSON标注文件夹路径
            output_dir: 输出目录路径
            val_split: 验证集比例 (0.0-1.0)
            seed: 随机种子
            num_classes: 类别总数（必填，不含背景）
            include_negative_samples: 是否包含无标注的图像作为负样本
        """
        self.images_dir = Path(images_dir)
        self.jsons_dir = Path(jsons_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.seed = seed
        self.num_classes = num_classes
        self.include_negative_samples = include_negative_samples
        
        # 原始类别ID集合（从数据中扫描到的）
        self.original_class_ids: Set[int] = set()
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'skipped_images': 0,
            'skipped_shapes': 0,
            'negative_samples': 0,
            'total_polygons': 0,
        }
    
    def validate_input(self) -> Tuple[bool, str]:
        """验证输入目录"""
        if not self.images_dir.exists():
            return False, f"图像文件夹不存在: {self.images_dir}"
        
        if not self.images_dir.is_dir():
            return False, f"图像路径不是文件夹: {self.images_dir}"
        
        if not self.jsons_dir.exists():
            return False, f"JSON文件夹不存在: {self.jsons_dir}"
        
        if not self.jsons_dir.is_dir():
            return False, f"JSON路径不是文件夹: {self.jsons_dir}"
        
        image_files = self._get_image_files()
        if len(image_files) == 0:
            return False, "图像文件夹中没有找到支持的图像文件"
        
        json_files = list(self.jsons_dir.glob('*.json'))
        if len(json_files) == 0:
            return False, "JSON文件夹中没有找到JSON文件"
        
        return True, f"验证通过: 找到 {len(image_files)} 张图像, {len(json_files)} 个JSON文件"
    
    def _get_image_files(self) -> List[Path]:
        """获取所有图像文件"""
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        return image_files
    
    def _create_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'masks' / 'train',
            self.output_dir / 'masks' / 'val',
            self.output_dir / 'splits',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _scan_all_classes(self, samples: List[Tuple[Path, Optional[Path]]]):
        """扫描所有类别ID"""
        for image_file, json_file in samples:
            if json_file is None:
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for shape in data.get('shapes', []):
                    if shape.get('shape_type') != 'polygon':
                        continue
                    
                    label = shape.get('label', '')
                    try:
                        class_id = int(label)
                        self.original_class_ids.add(class_id)
                    except ValueError:
                        pass
            except Exception:
                pass
    
    def _collect_samples(self) -> List[Tuple[Path, Optional[Path]]]:
        """收集所有样本"""
        samples = []
        json_stems = {f.stem for f in self.jsons_dir.glob('*.json')}
        
        # 有标注的样本
        for json_file in self.jsons_dir.glob('*.json'):
            stem = json_file.stem
            image_file = None
            
            for ext in self.SUPPORTED_EXTENSIONS:
                candidate = self.images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
                candidate = self.images_dir / f"{stem}{ext.upper()}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if image_file is not None:
                samples.append((image_file, json_file))
        
        # 负样本（无标注的图像）
        if self.include_negative_samples:
            for ext in self.SUPPORTED_EXTENSIONS:
                for image_file in self.images_dir.glob(f'*{ext}'):
                    if image_file.stem not in json_stems:
                        samples.append((image_file, None))
                        self.stats['negative_samples'] += 1
                for image_file in self.images_dir.glob(f'*{ext.upper()}'):
                    if image_file.stem not in json_stems:
                        samples.append((image_file, None))
                        self.stats['negative_samples'] += 1
        
        self.stats['total_images'] = len(samples)
        return samples
    
    def _split_samples(self, samples: List[Tuple[Path, Optional[Path]]]) -> Tuple[List, List]:
        """划分训练/验证集"""
        random.seed(self.seed)
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        if self.val_split <= 0:
            train_samples = samples_copy
            val_samples = samples_copy
            self.stats['train_images'] = len(train_samples)
            self.stats['val_images'] = len(val_samples)
        else:
            val_size = int(len(samples_copy) * self.val_split)
            val_samples = samples_copy[:val_size]
            train_samples = samples_copy[val_size:]
            self.stats['train_images'] = len(train_samples)
            self.stats['val_images'] = len(val_samples)
        
        return train_samples, val_samples
    
    def _create_mask_from_json(self, json_file: Path, img_width: int, img_height: int) -> np.ndarray:
        """
        从JSON创建分割mask
        
        Args:
            json_file: JSON文件路径
            img_width: 图像宽度
            img_height: 图像高度
        
        Returns:
            mask数组，shape=(H, W)，dtype=uint8
        """
        # 初始化为背景（255）
        mask = np.full((img_height, img_width), 255, dtype=np.uint8)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 按顺序绘制多边形（后绘制覆盖先绘制）
        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'polygon':
                self.stats['skipped_shapes'] += 1
                continue
            
            label = shape.get('label', '')
            try:
                class_id = int(label)
            except ValueError:
                self.stats['skipped_shapes'] += 1
                continue
            
            if class_id < 0 or class_id >= self.num_classes:
                self.stats['skipped_shapes'] += 1
                continue
            
            points = shape.get('points', [])
            if len(points) < 3:
                self.stats['skipped_shapes'] += 1
                continue
            
            # 转换为整数坐标
            polygon = [(int(p[0]), int(p[1])) for p in points]
            
            # 使用PIL绘制多边形
            pil_mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(pil_mask)
            draw.polygon(polygon, fill=class_id)
            mask = np.array(pil_mask)
            
            self.stats['total_polygons'] += 1
        
        return mask
    
    def _convert_samples(self, samples: List[Tuple[Path, Optional[Path]]], split: str) -> List[str]:
        """
        转换样本
        
        Args:
            samples: 样本列表
            split: 'train' 或 'val'
        
        Returns:
            文件名列表（用于生成splits文件）
        """
        file_list = []
        
        for image_file, json_file in samples:
            # 复制图像
            dst_image = self.output_dir / 'images' / split / image_file.name
            shutil.copy(image_file, dst_image)
            
            # 获取图像尺寸
            with Image.open(image_file) as img:
                img_width, img_height = img.size
            
            # 生成mask
            if json_file is None:
                # 负样本：全背景
                mask = np.full((img_height, img_width), 255, dtype=np.uint8)
            else:
                mask = self._create_mask_from_json(json_file, img_width, img_height)
            
            # 保存mask为PNG
            mask_filename = f"{image_file.stem}.png"
            mask_path = self.output_dir / 'masks' / split / mask_filename
            Image.fromarray(mask).save(mask_path)
            
            file_list.append(image_file.stem)
        
        return file_list
    
    def _generate_splits(self, train_files: List[str], val_files: List[str]):
        """生成splits文件"""
        train_txt = self.output_dir / 'splits' / 'train.txt'
        val_txt = self.output_dir / 'splits' / 'val.txt'
        
        with open(train_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_files))
        
        with open(val_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_files))
    
    def _generate_class_info(self):
        """生成类别信息文件"""
        info_file = self.output_dir / 'class_info.txt'
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# 语义分割类别信息\n")
            f.write(f"# 总类别数(不含背景): {self.num_classes}\n")
            f.write(f"# 背景像素值: 255 (ignore_index)\n")
            f.write(f"# 实际有标注数据的类别数: {len(self.original_class_ids)}\n")
            f.write(f"# 实际有标注数据的类别ID: {sorted(self.original_class_ids)}\n")
            f.write(f"# 负样本(良品)数量: {self.stats['negative_samples']}\n")
            f.write(f"# 总多边形数量: {self.stats['total_polygons']}\n")
            f.write("# ========================\n")
            f.write("# 类别索引 -> 类别名称:\n")
            for i in range(self.num_classes):
                has_data = "✓" if i in self.original_class_ids else "✗"
                f.write(f"  {i} -> '{i}' [{has_data}]\n")
    
    def convert(self) -> ConversionResult:
        """执行转换"""
        # 验证输入
        valid, msg = self.validate_input()
        if not valid:
            return ConversionResult(success=False, message=msg)
        
        try:
            # 创建输出目录
            self._create_dirs()
            
            # 收集所有样本
            samples = self._collect_samples()
            
            if len(samples) == 0:
                return ConversionResult(
                    success=False,
                    message="没有找到有效的图像-JSON配对"
                )
            
            # 扫描所有类别ID
            self._scan_all_classes(samples)
            
            # 验证num_classes
            if len(self.original_class_ids) > 0:
                max_class_id = max(self.original_class_ids)
                min_required_classes = max_class_id + 1
                
                if self.num_classes <= 0:
                    return ConversionResult(
                        success=False,
                        message=f"❌ 必须指定类别数(num_classes)\n"
                                f"   检测到的类别ID: {sorted(self.original_class_ids)}\n"
                                f"   最大类别ID: {max_class_id}\n"
                                f"   num_classes 至少需要: {min_required_classes}"
                    )
                
                if self.num_classes < min_required_classes:
                    return ConversionResult(
                        success=False,
                        message=f"❌ num_classes ({self.num_classes}) 太小\n"
                                f"   检测到的类别ID: {sorted(self.original_class_ids)}\n"
                                f"   最大类别ID: {max_class_id}\n"
                                f"   num_classes 至少需要: {min_required_classes}"
                    )
            elif self.num_classes <= 0:
                return ConversionResult(
                    success=False,
                    message="❌ 必须指定类别数(num_classes > 0)"
                )
            
            # 划分训练/验证集
            train_samples, val_samples = self._split_samples(samples)
            
            # 转换样本
            train_files = self._convert_samples(train_samples, 'train')
            val_files = self._convert_samples(val_samples, 'val')
            
            # 生成splits文件
            self._generate_splits(train_files, val_files)
            
            # 生成类别信息文件
            self._generate_class_info()
            
            # 生成类别名称列表
            class_names = [str(i) for i in range(self.num_classes)]
            
            result = ConversionResult(
                success=True,
                message="转换成功",
                output_dir=str(self.output_dir),
                total_images=self.stats['total_images'],
                train_images=self.stats['train_images'],
                val_images=self.stats['val_images'],
                num_classes=self.num_classes,
                actual_class_count=len(self.original_class_ids),
                class_ids=sorted(self.original_class_ids),
                class_names=class_names,
                negative_samples=self.stats['negative_samples'],
            )
            
            return result
            
        except Exception as e:
            import traceback
            return ConversionResult(
                success=False,
                message=f"转换失败: {str(e)}\n{traceback.format_exc()}"
            )
    
    def get_preview_samples(self, num_samples: int = 4, seed: Optional[int] = None) -> List[Tuple[Path, Path]]:
        """获取用于预览的样本"""
        samples = self._collect_samples()
        samples = [(img, json) for img, json in samples if json is not None]
        
        if len(samples) <= num_samples:
            return samples
        
        random.seed(seed if seed is not None else self.seed)
        return random.sample(samples, num_samples)


def convert_labelme_to_mmseg(
    images_dir: str,
    jsons_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
    num_classes: int = 0,
    include_negative_samples: bool = True,
) -> ConversionResult:
    """
    便捷函数：将LabelMe格式转换为MMSegmentation格式
    """
    converter = LabelMeToMMSegConverter(
        images_dir=images_dir,
        jsons_dir=jsons_dir,
        output_dir=output_dir,
        val_split=val_split,
        seed=seed,
        num_classes=num_classes,
        include_negative_samples=include_negative_samples,
    )
    return converter.convert()
