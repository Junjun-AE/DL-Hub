"""
LabelMe JSON → YOLO TXT 格式转换器
支持自动划分训练/验证集
自动将不连续的类别ID映射为连续的0,1,2,3...
"""

import json
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field


@dataclass
class ConversionResult:
    """转换结果"""
    success: bool
    message: str
    output_dir: str = ""
    total_images: int = 0
    train_images: int = 0
    val_images: int = 0
    num_classes: int = 0          # 用户指定的类别总数
    actual_class_count: int = 0   # 实际有标注数据的类别数量
    class_ids: List[int] = None   # 实际存在的类别ID列表
    class_names: List[str] = None # 类别名称列表（长度=num_classes）
    train_boxes: int = 0
    val_boxes: int = 0
    data_yaml: str = ""
    negative_samples: int = 0     # 负样本（无标注）数量
    
    def __post_init__(self):
        if self.class_ids is None:
            self.class_ids = []
        if self.class_names is None:
            self.class_names = []


class LabelMeToYOLOConverter:
    """
    LabelMe JSON格式转换为YOLO格式
    
    输入结构:
        images/           # 图像文件夹
            1.jpg
            2.png
            3.jpg         # 可以没有对应的JSON（作为负样本/良品）
        jsons/            # JSON标注文件夹
            1.json
            2.json
    
    输出结构 (YOLO格式):
        dataset/
            images/
                train/
                val/
            labels/
                train/    # 负样本图像会有空的txt文件
                val/
            data.yaml
            class_info.txt
    
    类别ID处理:
    - 始终保留原始类别ID（不压缩）
    - num_classes 必须 >= max(原始类别ID) + 1
    - 例如：标注ID为[0,1,3,5]，num_classes必须>=6
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    def __init__(
        self,
        images_dir: str,
        jsons_dir: str,
        output_dir: str,
        val_split: float = 0.2,
        seed: int = 42,
        num_classes: int = 0,  # 必填参数，必须>=max(class_ids)+1
        include_negative_samples: bool = True,  # 是否包含无标注图像作为负样本
    ):
        """
        初始化转换器
        
        Args:
            images_dir: 图像文件夹路径
            jsons_dir: JSON标注文件夹路径
            output_dir: 输出目录路径
            val_split: 验证集比例 (0.0-1.0)
            seed: 随机种子
            num_classes: 类别总数（必填）
                - 必须 >= max(标注中的类别ID) + 1
                - 例如：标注有ID=5，则num_classes至少为6
            include_negative_samples: 是否包含无标注的图像作为负样本（良品）
                - True: 没有JSON的图像也会被包含，生成空的txt标注文件
                - False: 只处理有JSON标注的图像
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
            'train_boxes': 0,
            'val_boxes': 0,
            'skipped_images': 0,
            'skipped_shapes': 0,
            'negative_samples': 0,
        }
    
    def validate_input(self) -> Tuple[bool, str]:
        """
        验证输入目录
        
        Returns:
            (是否有效, 消息)
        """
        # 检查图像目录
        if not self.images_dir.exists():
            return False, f"图像文件夹不存在: {self.images_dir}"
        
        if not self.images_dir.is_dir():
            return False, f"图像路径不是文件夹: {self.images_dir}"
        
        # 检查JSON目录
        if not self.jsons_dir.exists():
            return False, f"JSON文件夹不存在: {self.jsons_dir}"
        
        if not self.jsons_dir.is_dir():
            return False, f"JSON路径不是文件夹: {self.jsons_dir}"
        
        # 检查是否有图像文件
        image_files = self._get_image_files()
        if len(image_files) == 0:
            return False, "图像文件夹中没有找到支持的图像文件"
        
        # 检查是否有JSON文件
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
    
    def convert(self) -> ConversionResult:
        """
        执行转换
        
        Returns:
            ConversionResult对象
        """
        # 验证输入
        valid, msg = self.validate_input()
        if not valid:
            return ConversionResult(success=False, message=msg)
        
        try:
            # 创建输出目录
            self._create_dirs()
            
            # 收集所有样本（图像-JSON配对）
            samples = self._collect_samples()
            
            if len(samples) == 0:
                return ConversionResult(
                    success=False,
                    message="没有找到有效的图像-JSON配对"
                )
            
            # 第一遍扫描：收集所有原始类别ID
            self._scan_all_classes(samples)
            
            if len(self.original_class_ids) == 0:
                return ConversionResult(
                    success=False,
                    message="没有找到有效的类别标注"
                )
            
            # 验证 num_classes
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
            
            # 划分训练/验证集
            train_samples, val_samples = self._split_samples(samples)
            
            # 转换并保存（直接使用原始类别ID，不做映射）
            self._convert_samples(train_samples, 'train')
            self._convert_samples(val_samples, 'val')
            
            # 生成 data.yaml
            yaml_path = self._generate_yaml()
            
            # 生成 class_names 列表
            class_names = [str(i) for i in range(self.num_classes)]
            
            # 构建结果
            result = ConversionResult(
                success=True,
                message="转换成功",
                output_dir=str(self.output_dir),
                total_images=self.stats['total_images'],
                train_images=self.stats['train_images'],
                val_images=self.stats['val_images'],
                num_classes=self.num_classes,                 # 用户指定的类别总数
                actual_class_count=len(self.original_class_ids),  # 实际有数据的类别数
                class_ids=sorted(self.original_class_ids),    # 实际存在的类别ID
                class_names=class_names,                      # ['0', '1', ..., 'num_classes-1']
                train_boxes=self.stats['train_boxes'],
                val_boxes=self.stats['val_boxes'],
                data_yaml=str(yaml_path),
                negative_samples=self.stats['negative_samples'],
            )
            
            return result
            
        except Exception as e:
            import traceback
            return ConversionResult(
                success=False,
                message=f"转换出错: {str(e)}\n{traceback.format_exc()}"
            )
    
    def _scan_all_classes(self, samples: List[Tuple[Path, Optional[Path]]]):
        """
        扫描所有样本收集原始类别ID
        
        Args:
            samples: 样本列表（json_path可能为None）
        """
        for _, json_file in samples:
            if json_file is None:
                continue  # 跳过负样本
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for shape in data.get('shapes', []):
                    if shape.get('shape_type') != 'rectangle':
                        continue
                    
                    label = shape.get('label', '')
                    try:
                        class_id = int(label)
                        self.original_class_ids.add(class_id)
                    except ValueError:
                        pass
            except Exception:
                pass
    
    def _create_dirs(self):
        """创建输出目录结构"""
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    def _collect_samples(self) -> List[Tuple[Path, Optional[Path]]]:
        """
        收集所有样本（图像-JSON配对，以及无标注的图像）
        
        Returns:
            [(image_path, json_path), ...]  # json_path可能为None（负样本）
        """
        samples = []
        json_stems = set()
        
        # 首先收集所有有JSON的图像
        for json_file in self.jsons_dir.glob('*.json'):
            stem = json_file.stem
            json_stems.add(stem)
            
            # 查找对应的图像文件
            image_file = None
            for ext in self.SUPPORTED_EXTENSIONS:
                candidate = self.images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
                # 尝试大写扩展名
                candidate = self.images_dir / f"{stem}{ext.upper()}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if image_file:
                samples.append((image_file, json_file))
            else:
                self.stats['skipped_images'] += 1
        
        # 如果需要包含负样本，收集没有JSON的图像
        if self.include_negative_samples:
            for ext in self.SUPPORTED_EXTENSIONS:
                for image_file in self.images_dir.glob(f'*{ext}'):
                    if image_file.stem not in json_stems:
                        samples.append((image_file, None))  # None表示没有标注
                        self.stats['negative_samples'] += 1
                # 大写扩展名
                for image_file in self.images_dir.glob(f'*{ext.upper()}'):
                    if image_file.stem not in json_stems:
                        samples.append((image_file, None))
                        self.stats['negative_samples'] += 1
        
        self.stats['total_images'] = len(samples)
        return samples
    
    def _split_samples(self, samples: List[Tuple[Path, Optional[Path]]]) -> Tuple[List, List]:
        """
        划分训练/验证集
        
        Args:
            samples: 样本列表
        
        Returns:
            (train_samples, val_samples)
        
        Note:
            如果 val_split = 0，训练集和验证集都使用全部数据
        """
        random.seed(self.seed)
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        if self.val_split <= 0:
            # val_split=0: 训练集和验证集都使用全部数据
            train_samples = samples_copy
            val_samples = samples_copy
            self.stats['train_images'] = len(train_samples)
            self.stats['val_images'] = len(val_samples)
        else:
            # 正常分割
            val_size = int(len(samples_copy) * self.val_split)
            val_samples = samples_copy[:val_size]
            train_samples = samples_copy[val_size:]
            self.stats['train_images'] = len(train_samples)
            self.stats['val_images'] = len(val_samples)
        
        return train_samples, val_samples
    
    def _convert_samples(self, samples: List[Tuple[Path, Optional[Path]]], split: str):
        """
        转换样本到YOLO格式
        
        Args:
            samples: 样本列表（json_path可能为None表示负样本）
            split: 'train' 或 'val'
        """
        for image_file, json_file in samples:
            # 复制图像
            dst_image = self.output_dir / 'images' / split / image_file.name
            shutil.copy(image_file, dst_image)
            
            if json_file is None:
                # 负样本：创建空的txt文件
                txt_file = self.output_dir / 'labels' / split / f"{image_file.stem}.txt"
                txt_file.touch()  # 创建空文件
            else:
                # 有标注：转换标注
                boxes = self._convert_annotation(json_file, split, image_file.stem)
                
                if split == 'train':
                    self.stats['train_boxes'] += boxes
                else:
                    self.stats['val_boxes'] += boxes
    
    def _convert_annotation(self, json_file: Path, split: str, stem: str) -> int:
        """
        转换单个JSON标注到YOLO TXT格式
        
        Args:
            json_file: JSON文件路径
            split: 'train' 或 'val'
            stem: 文件名（不含扩展名）
        
        Returns:
            标注框数量
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_width = data.get('imageWidth', 0)
        img_height = data.get('imageHeight', 0)
        
        if img_width <= 0 or img_height <= 0:
            return 0
        
        lines = []
        
        for shape in data.get('shapes', []):
            # 只处理矩形标注
            if shape.get('shape_type') != 'rectangle':
                self.stats['skipped_shapes'] += 1
                continue
            
            # 获取原始类别ID（label是数字字符串）
            label = shape.get('label', '')
            try:
                class_id = int(label)
            except ValueError:
                self.stats['skipped_shapes'] += 1
                continue
            
            # 验证类别ID在有效范围内
            if class_id < 0 or class_id >= self.num_classes:
                self.stats['skipped_shapes'] += 1
                continue
            
            # 获取边界框坐标
            points = shape.get('points', [])
            if len(points) < 2:
                self.stats['skipped_shapes'] += 1
                continue
            
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 确保坐标顺序正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 转换为YOLO格式 (归一化的 x_center, y_center, width, height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # 确保坐标在[0, 1]范围内
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            # 跳过无效的框
            if width <= 0 or height <= 0:
                self.stats['skipped_shapes'] += 1
                continue
            
            # 直接使用原始类别ID写入YOLO格式（不做任何映射）
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 保存TXT标注文件
        txt_file = self.output_dir / 'labels' / split / f"{stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return len(lines)
    
    def _generate_yaml(self) -> Path:
        """
        生成data.yaml配置文件
        
        Returns:
            yaml文件路径
        """
        import yaml
        
        # 类别数量使用用户指定的值
        nc = self.num_classes
        
        # 创建names列表: ['0', '1', '2', ..., 'num_classes-1']
        names = [str(i) for i in range(nc)]
        
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': nc,
            'names': names,
        }
        
        yaml_file = self.output_dir / 'data.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        # 保存类别信息文件（方便查看和调试）
        info_file = self.output_dir / 'class_info.txt'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# 类别信息说明\n")
            f.write(f"# 总类别数(nc): {nc}\n")
            f.write(f"# 实际有标注数据的类别数: {len(self.original_class_ids)}\n")
            f.write(f"# 实际有标注数据的类别ID: {sorted(self.original_class_ids)}\n")
            f.write(f"# 无标注数据的类别ID: {sorted(set(range(nc)) - self.original_class_ids)}\n")
            f.write(f"# 负样本(良品)数量: {self.stats['negative_samples']}\n")
            f.write("# ========================\n")
            f.write("# 类别索引 -> 类别名称:\n")
            for i in range(nc):
                has_data = "✓" if i in self.original_class_ids else "✗"
                f.write(f"  {i} -> '{i}' [{has_data}]\n")
        
        return yaml_file
    
    def get_preview_samples(self, num_samples: int = 4, seed: Optional[int] = None) -> List[Tuple[Path, Path]]:
        """
        获取用于预览的样本
        
        Args:
            num_samples: 样本数量
            seed: 随机种子（用于换一批）
        
        Returns:
            [(image_path, json_path), ...]
        """
        samples = self._collect_samples()
        # 过滤掉负样本
        samples = [(img, json) for img, json in samples if json is not None]
        
        if len(samples) <= num_samples:
            return samples
        
        # 使用指定的种子或默认种子
        random.seed(seed if seed is not None else self.seed)
        return random.sample(samples, num_samples)


def convert_labelme_to_yolo(
    images_dir: str,
    jsons_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
    num_classes: int = 0,
    include_negative_samples: bool = True,
) -> ConversionResult:
    """
    便捷函数：将LabelMe格式转换为YOLO格式
    
    Args:
        images_dir: 图像文件夹路径
        jsons_dir: JSON标注文件夹路径
        output_dir: 输出目录路径
        val_split: 验证集比例
        seed: 随机种子
        num_classes: 类别总数（必填，必须 >= max(标注ID) + 1）
        include_negative_samples: 是否包含无标注的图像作为负样本
    
    Returns:
        ConversionResult对象
    """
    converter = LabelMeToYOLOConverter(
        images_dir=images_dir,
        jsons_dir=jsons_dir,
        output_dir=output_dir,
        val_split=val_split,
        seed=seed,
        num_classes=num_classes,
        include_negative_samples=include_negative_samples,
    )
    return converter.convert()
