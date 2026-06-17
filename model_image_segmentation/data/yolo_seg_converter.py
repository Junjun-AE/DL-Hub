"""
LabelMe JSON → YOLO-seg (实例分割) 格式转换器
=============================================
与图像分割模块的 LabelMeToMMSegConverter 共用同一份 polygon 标注源，
区别仅在输出目标：

- MMSeg 转换器: polygon -> PNG mask (逐像素类别图, 背景=255)
- 本转换器:     polygon -> 归一化 polygon TXT (逐实例, Ultralytics YOLO-seg 格式)

输入结构 (与 MMSeg 转换器完全一致):
    images/           # 图像文件夹
        1.jpg
        2.png
        3.jpg         # 可以没有对应的 JSON (作为负样本/良品)
    jsons/            # JSON 标注文件夹 (LabelMe polygon)
        1.json
        2.json

输出结构 (YOLO-seg 格式):
    dataset/
        images/
            train/
            val/
        labels/
            train/    # 负样本图像 -> 空 txt
            val/
        data.yaml
        class_info.txt

标签行格式 (每行一个实例, 坐标归一化到 [0,1]):
    <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>   (至少 3 个点 / 6 个值)

类别约定 (与 MMSeg 转换器一致):
- LabelMe label 为数字字符串 "0","1","2"... 直接作为 class_id, 不做名称映射
- num_classes 必须 >= max(class_id)+1
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
    """转换结果 (字段与 MMSeg 转换器保持一致, 便于 app 层统一处理)。"""
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
    data_yaml: str = ""          # YOLO-seg 专属: data.yaml 路径
    total_polygons: int = 0

    def __post_init__(self):
        if self.class_ids is None:
            self.class_ids = []
        if self.class_names is None:
            self.class_names = []


class LabelMeToYOLOSegConverter:
    """LabelMe polygon JSON -> YOLO-seg polygon TXT 转换器。"""

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

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
        self.images_dir = Path(images_dir)
        self.jsons_dir = Path(jsons_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.seed = seed
        self.num_classes = num_classes
        self.include_negative_samples = include_negative_samples

        self.original_class_ids: Set[int] = set()
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'train_instances': 0,
            'val_instances': 0,
            'negative_samples': 0,
            'total_polygons': 0,
            'skipped_shapes': 0,
            'skipped_images': 0,
        }

    # ------------------------------------------------------------------ #
    # 输入验证
    # ------------------------------------------------------------------ #
    def validate_input(self) -> Tuple[bool, str]:
        if not self.images_dir.exists():
            return False, f"图像文件夹不存在: {self.images_dir}"
        if not self.jsons_dir.exists():
            return False, f"JSON 文件夹不存在: {self.jsons_dir}"
        image_files = self._get_image_files()
        if len(image_files) == 0:
            return False, f"图像文件夹中没有支持的图像: {self.images_dir}"
        json_files = list(self.jsons_dir.glob('*.json'))
        if len(json_files) == 0:
            return False, f"JSON 文件夹中没有 JSON 文件: {self.jsons_dir}"
        return True, "验证通过"

    def _get_image_files(self) -> List[Path]:
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.images_dir.glob(f'*{ext}'))
            files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        return files

    # ------------------------------------------------------------------ #
    # 目录 / 样本收集 (逻辑与 MMSeg 转换器一致)
    # ------------------------------------------------------------------ #
    def _create_dirs(self):
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    def _scan_all_classes(self, samples: List[Tuple[Path, Optional[Path]]]):
        for _, json_file in samples:
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
                        self.original_class_ids.add(int(label))
                    except ValueError:
                        pass
            except Exception:
                pass

    def _collect_samples(self) -> List[Tuple[Path, Optional[Path]]]:
        samples = []
        json_stems = {f.stem for f in self.jsons_dir.glob('*.json')}

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
            else:
                self.stats['skipped_images'] += 1

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
        random.seed(self.seed)
        samples_copy = samples.copy()
        random.shuffle(samples_copy)

        if self.val_split <= 0:
            train_samples = samples_copy
            val_samples = samples_copy
        else:
            val_size = int(len(samples_copy) * self.val_split)
            val_samples = samples_copy[:val_size]
            train_samples = samples_copy[val_size:]

        self.stats['train_images'] = len(train_samples)
        self.stats['val_images'] = len(val_samples)
        return train_samples, val_samples

    # ------------------------------------------------------------------ #
    # 核心: polygon -> 归一化 YOLO-seg TXT
    # ------------------------------------------------------------------ #
    def _convert_annotation(self, json_file: Path, split: str, stem: str) -> int:
        """
        将单个 LabelMe JSON 转为 YOLO-seg 标签文件。

        Returns:
            实例 (有效多边形) 数量
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width = data.get('imageWidth', 0)
        img_height = data.get('imageHeight', 0)

        # 若 JSON 未带尺寸, 尝试从图像读取
        if img_width <= 0 or img_height <= 0:
            try:
                from PIL import Image
                img_path = None
                for ext in self.SUPPORTED_EXTENSIONS:
                    cand = self.images_dir / f"{stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
                if img_path is not None:
                    with Image.open(img_path) as im:
                        img_width, img_height = im.size
            except Exception:
                pass

        txt_file = self.output_dir / 'labels' / split / f"{stem}.txt"

        if img_width <= 0 or img_height <= 0:
            txt_file.write_text("", encoding='utf-8')
            return 0

        lines = []
        for shape in data.get('shapes', []):
            # 只处理多边形 (实例分割必需)
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
            # 至少 3 个点才构成多边形
            if len(points) < 3:
                self.stats['skipped_shapes'] += 1
                continue

            coords = []
            valid = True
            for p in points:
                if len(p) < 2:
                    valid = False
                    break
                xn = max(0.0, min(1.0, float(p[0]) / img_width))
                yn = max(0.0, min(1.0, float(p[1]) / img_height))
                coords.append(f"{xn:.6f}")
                coords.append(f"{yn:.6f}")

            if not valid or len(coords) < 6:
                self.stats['skipped_shapes'] += 1
                continue

            lines.append(f"{class_id} " + " ".join(coords))
            self.stats['total_polygons'] += 1

        txt_file.write_text("\n".join(lines), encoding='utf-8')
        return len(lines)

    def _convert_samples(self, samples: List[Tuple[Path, Optional[Path]]], split: str):
        for image_file, json_file in samples:
            dst_image = self.output_dir / 'images' / split / image_file.name
            shutil.copy(image_file, dst_image)

            if json_file is None:
                # 负样本: 空 txt
                txt_file = self.output_dir / 'labels' / split / f"{image_file.stem}.txt"
                txt_file.write_text("", encoding='utf-8')
            else:
                n = self._convert_annotation(json_file, split, image_file.stem)
                if split == 'train':
                    self.stats['train_instances'] += n
                else:
                    self.stats['val_instances'] += n

    # ------------------------------------------------------------------ #
    # data.yaml / class_info
    # ------------------------------------------------------------------ #
    def _generate_yaml(self) -> Path:
        import yaml
        nc = self.num_classes
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
        return yaml_file

    def _generate_class_info(self):
        info_file = self.output_dir / 'class_info.txt'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# YOLO-seg 实例分割类别信息\n")
            f.write(f"# 总类别数(nc): {self.num_classes}\n")
            f.write(f"# 实际有标注数据的类别数: {len(self.original_class_ids)}\n")
            f.write(f"# 实际有标注数据的类别ID: {sorted(self.original_class_ids)}\n")
            f.write(f"# 无标注数据的类别ID: {sorted(set(range(self.num_classes)) - self.original_class_ids)}\n")
            f.write(f"# 负样本(良品)数量: {self.stats['negative_samples']}\n")
            f.write(f"# 总多边形(实例)数量: {self.stats['total_polygons']}\n")
            f.write("# ========================\n")
            f.write("# 类别索引 -> 类别名称:\n")
            for i in range(self.num_classes):
                has_data = "✓" if i in self.original_class_ids else "✗"
                f.write(f"  {i} -> '{i}' [{has_data}]\n")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def convert(self) -> ConversionResult:
        valid, msg = self.validate_input()
        if not valid:
            return ConversionResult(success=False, message=msg)

        try:
            self._create_dirs()
            samples = self._collect_samples()
            if len(samples) == 0:
                return ConversionResult(success=False, message="没有找到有效的图像-JSON配对")

            self._scan_all_classes(samples)

            # 校验 num_classes
            if len(self.original_class_ids) > 0:
                max_class_id = max(self.original_class_ids)
                min_required = max_class_id + 1
                if self.num_classes <= 0:
                    return ConversionResult(
                        success=False,
                        message=(f"❌ 必须指定类别数(num_classes)\n"
                                 f"   检测到的类别ID: {sorted(self.original_class_ids)}\n"
                                 f"   最大类别ID: {max_class_id}\n"
                                 f"   num_classes 至少需要: {min_required}")
                    )
                if self.num_classes < min_required:
                    return ConversionResult(
                        success=False,
                        message=(f"❌ num_classes ({self.num_classes}) 太小\n"
                                 f"   检测到的类别ID: {sorted(self.original_class_ids)}\n"
                                 f"   最大类别ID: {max_class_id}\n"
                                 f"   num_classes 至少需要: {min_required}")
                    )
            elif self.num_classes <= 0:
                return ConversionResult(success=False, message="❌ 必须指定类别数(num_classes > 0)")

            train_samples, val_samples = self._split_samples(samples)
            self._convert_samples(train_samples, 'train')
            self._convert_samples(val_samples, 'val')

            yaml_file = self._generate_yaml()
            self._generate_class_info()

            class_names = [str(i) for i in range(self.num_classes)]

            return ConversionResult(
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
                data_yaml=str(yaml_file),
                total_polygons=self.stats['total_polygons'],
            )
        except Exception as e:
            import traceback
            return ConversionResult(
                success=False,
                message=f"转换失败: {str(e)}\n{traceback.format_exc()}"
            )

    def get_preview_samples(self, num_samples: int = 4, seed: Optional[int] = None) -> List[Tuple[Path, Path]]:
        samples = self._collect_samples()
        samples = [(img, js) for img, js in samples if js is not None]
        if len(samples) <= num_samples:
            return samples
        random.seed(seed if seed is not None else self.seed)
        return random.sample(samples, num_samples)


def convert_labelme_to_yolo_seg(
    images_dir: str,
    jsons_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
    num_classes: int = 0,
    include_negative_samples: bool = True,
) -> ConversionResult:
    """便捷函数: LabelMe -> YOLO-seg 格式。"""
    converter = LabelMeToYOLOSegConverter(
        images_dir=images_dir,
        jsons_dir=jsons_dir,
        output_dir=output_dir,
        val_split=val_split,
        seed=seed,
        num_classes=num_classes,
        include_negative_samples=include_negative_samples,
    )
    return converter.convert()
