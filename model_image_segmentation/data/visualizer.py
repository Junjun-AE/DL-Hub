"""
分割可视化模块 - 用于预览LabelMe标注和分割结果
【工业级修复版】- 严格错误处理，无静默跳过，无假数据
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


# 预定义颜色列表（用于不同类别）
COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青
    (255, 128, 0),    # 橙
    (128, 0, 255),    # 紫
    (0, 255, 128),    # 春绿
    (255, 0, 128),    # 玫红
    (128, 255, 0),    # 黄绿
    (0, 128, 255),    # 天蓝
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """获取类别对应的颜色"""
    return COLORS[class_id % len(COLORS)]


def visualize_labelme_polygon(
    image_path: str,
    json_path: str,
    output_path: Optional[str] = None,
    alpha: float = 0.5,
    line_width: int = 2,
    strict_mode: bool = True,
) -> Image.Image:
    """
    可视化LabelMe多边形标注（分割用）
    【工业级修复版】- 严格错误处理，有异常必须报错
    
    Args:
        image_path: 图像文件路径
        json_path: JSON标注文件路径
        output_path: 输出图像路径（可选）
        alpha: 填充透明度
        line_width: 边框线宽
        strict_mode: 严格模式，遇到无效标注会报错
    
    Returns:
        PIL Image对象
        
    Raises:
        ImportError: Pillow未安装
        FileNotFoundError: 文件不存在
        ValueError: 标注格式无效
    """
    if Image is None:
        raise ImportError("❌ 需要安装Pillow: pip install Pillow")
    
    # 严格检查文件存在性
    if not Path(image_path).exists():
        raise FileNotFoundError(f"❌ 图像文件不存在: {image_path}")
    
    if not Path(json_path).exists():
        raise FileNotFoundError(f"❌ JSON文件不存在: {json_path}")
    
    # 加载图像（带严格错误处理）
    try:
        img = Image.open(image_path).convert('RGBA')
    except Exception as e:
        raise ValueError(f"❌ 无法打开图像文件 {image_path}: {e}")
    
    # 创建叠加层
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 加载标注（带严格错误处理）
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON格式错误 {json_path}: {e}")
    except Exception as e:
        raise ValueError(f"❌ 无法读取JSON文件 {json_path}: {e}")
    
    # 验证JSON结构
    if not isinstance(data, dict):
        raise ValueError(f"❌ JSON格式无效，应为字典: {json_path}")
    
    shapes = data.get('shapes', [])
    if not isinstance(shapes, list):
        raise ValueError(f"❌ JSON中shapes字段应为列表: {json_path}")
    
    # 绘制每个多边形
    polygon_count = 0
    for idx, shape in enumerate(shapes):
        shape_type = shape.get('shape_type', '')
        
        # 跳过非polygon类型（这是允许的，因为可能有其他标注类型）
        if shape_type != 'polygon':
            continue
        
        label = shape.get('label', '')
        points = shape.get('points', [])
        
        # 严格验证点数
        if len(points) < 3:
            if strict_mode:
                raise ValueError(
                    f"❌ 多边形点数不足3个\n"
                    f"   文件: {json_path}\n"
                    f"   形状索引: {idx}\n"
                    f"   点数: {len(points)}"
                )
            continue
        
        # 转换为整数坐标
        try:
            polygon = [(int(float(p[0])), int(float(p[1]))) for p in points]
        except (ValueError, TypeError, IndexError) as e:
            if strict_mode:
                raise ValueError(
                    f"❌ 多边形坐标无效\n"
                    f"   文件: {json_path}\n"
                    f"   形状索引: {idx}\n"
                    f"   错误: {e}"
                )
            continue
        
        # 获取颜色 - 严格要求label为数字
        try:
            class_id = int(label)
            if class_id < 0:
                raise ValueError("类别ID不能为负数")
            color = get_color_for_class(class_id)
        except ValueError as e:
            if strict_mode:
                raise ValueError(
                    f"❌ 标注label必须为非负整数\n"
                    f"   文件: {json_path}\n"
                    f"   形状索引: {idx}\n"
                    f"   label值: '{label}'\n"
                    f"   错误: {e}"
                )
            continue
        
        # 绘制填充
        fill_color = (*color, int(255 * alpha))
        draw.polygon(polygon, fill=fill_color, outline=color)
        
        # 绘制边框（加粗）
        for i in range(line_width):
            draw.polygon(polygon, outline=color)
        
        polygon_count += 1
    
    # 合并图层
    img = Image.alpha_composite(img, overlay)
    result = img.convert('RGB')
    
    # 保存或返回
    if output_path:
        try:
            result.save(output_path)
        except Exception as e:
            raise ValueError(f"❌ 无法保存图像到 {output_path}: {e}")
    
    return result


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    ignore_index: int = 255,  # 【修复】默认值改为255，背景不显示
    class_names: Optional[Dict[int, str]] = None,
) -> Image.Image:
    """
    将分割mask叠加到原图上显示
    【工业级修复版】- 修正ignore_index默认值为255（背景不显示）
    
    Args:
        image: 原图数组 (H, W, C) RGB格式
        mask: 分割mask (H, W)，像素值为类别ID
        alpha: 叠加透明度
        ignore_index: 忽略的像素值（默认255=背景/忽略区域）
        class_names: 类别名称字典
    
    Returns:
        叠加后的PIL Image
        
    Raises:
        ImportError: Pillow未安装
        ValueError: 输入格式无效
    """
    if Image is None:
        raise ImportError("❌ 需要安装Pillow: pip install Pillow")
    
    # 验证输入
    if image is None:
        raise ValueError("❌ 输入图像不能为None")
    if mask is None:
        raise ValueError("❌ 输入mask不能为None")
    
    # 确保图像是RGB格式
    if isinstance(image, Image.Image):
        img = image.convert('RGBA')
    else:
        if not isinstance(image, np.ndarray):
            raise ValueError(f"❌ 图像类型无效: {type(image)}")
        img = Image.fromarray(image).convert('RGBA')
    
    # 验证mask
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"❌ mask类型无效: {type(mask)}")
    if mask.ndim != 2:
        raise ValueError(f"❌ mask应为2维数组，当前维度: {mask.ndim}")
    
    # 构建忽略类别集合（跳过背景255和用户指定的ignore_index）
    skip_classes = {ignore_index, 255}
    
    # 创建彩色mask叠加层
    H, W = mask.shape
    color_mask = np.zeros((H, W, 4), dtype=np.uint8)
    
    # 为每个类别着色（跳过背景和忽略类）
    unique_classes = np.unique(mask)
    for cls_id in unique_classes:
        if cls_id in skip_classes:
            continue
        
        color = get_color_for_class(int(cls_id))
        class_pixels = mask == cls_id
        color_mask[class_pixels] = (*color, int(255 * alpha))
    
    # 确保尺寸匹配
    if (H, W) != (img.size[1], img.size[0]):
        mask_img = Image.fromarray(color_mask, mode='RGBA')
        mask_img = mask_img.resize((img.size[0], img.size[1]), Image.NEAREST)
        overlay = mask_img
    else:
        overlay = Image.fromarray(color_mask, mode='RGBA')
    
    # 合并图层
    result = Image.alpha_composite(img, overlay)
    
    return result.convert('RGB')


def visualize_segmentation_result(
    image_path: str,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
    ignore_index: int = 255,  # 【修复】默认值改为255
    class_names: Optional[Dict[int, str]] = None,
    output_path: Optional[str] = None,
) -> Image.Image:
    """可视化分割预测结果【工业级修复版】"""
    if Image is None:
        raise ImportError("❌ 需要安装Pillow: pip install Pillow")
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"❌ 图像文件不存在: {image_path}")
    
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        raise ValueError(f"❌ 无法打开图像 {image_path}: {e}")
    
    if pred_mask is None:
        raise ValueError("❌ 预测mask不能为None")
    
    if pred_mask.shape[:2] != img.shape[:2]:
        mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
        mask_pil = mask_pil.resize((img.shape[1], img.shape[0]), Image.NEAREST)
        pred_mask = np.array(mask_pil)
    
    result = visualize_mask_overlay(img, pred_mask, alpha, ignore_index, class_names)
    
    if output_path:
        try:
            result.save(output_path)
        except Exception as e:
            raise ValueError(f"❌ 无法保存到 {output_path}: {e}")
    
    return result


def create_preview_grid(
    images: List[Image.Image],
    grid_size: Tuple[int, int] = (2, 2),
    cell_size: Tuple[int, int] = (800, 800),
    preserve_resolution: bool = False,
    max_cell_size: Tuple[int, int] = (2000, 2000),
) -> Image.Image:
    """创建图像网格预览【工业级修复版】- 严格验证输入"""
    if Image is None:
        raise ImportError("❌ 需要安装Pillow: pip install Pillow")
    
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        raise ValueError("❌ 没有有效的预览图像")
    
    rows, cols = grid_size
    
    if preserve_resolution:
        processed_images = []
        for img in valid_images:
            w, h = img.size
            max_w, max_h = max_cell_size
            
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            processed_images.append(img)
        
        max_width = max(img.size[0] for img in processed_images)
        max_height = max(img.size[1] for img in processed_images)
        cell_w, cell_h = max_width, max_height
    else:
        cell_w, cell_h = cell_size
        processed_images = []
        for img in valid_images:
            resized = img.resize((cell_w, cell_h), Image.LANCZOS)
            processed_images.append(resized)
    
    grid_width = cols * cell_w
    grid_height = rows * cell_h
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(processed_images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        x = col * cell_w + (cell_w - img.size[0]) // 2
        y = row * cell_h + (cell_h - img.size[1]) // 2
        grid.paste(img, (x, y))
    
    return grid


def preview_dataset(
    images_dir: str,
    jsons_dir: str,
    num_samples: int = 4,
    seed: int = None,
    high_resolution: bool = True,
    strict_mode: bool = False,
) -> Image.Image:
    """预览数据集【工业级修复版】"""
    if Image is None:
        raise ImportError("❌ 需要安装Pillow: pip install Pillow")
    
    images_path = Path(images_dir)
    jsons_path = Path(jsons_dir)
    
    if not images_path.exists():
        raise FileNotFoundError(f"❌ 图像文件夹不存在: {images_dir}")
    if not jsons_path.exists():
        raise FileNotFoundError(f"❌ JSON文件夹不存在: {jsons_dir}")
    
    samples = []
    for json_file in jsons_path.glob('*.json'):
        stem = json_file.stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
            img_file = images_path / f"{stem}{ext}"
            if img_file.exists():
                samples.append((img_file, json_file))
                break
    
    if len(samples) == 0:
        raise ValueError(f"❌ 没有找到匹配的图像-JSON配对\n   图像目录: {images_dir}\n   JSON目录: {jsons_dir}")
    
    if seed is not None:
        random.seed(seed)
    selected = random.sample(samples, min(num_samples, len(samples)))
    
    preview_images = []
    errors = []
    
    for img_path, json_path in selected:
        try:
            vis_img = visualize_labelme_polygon(str(img_path), str(json_path), strict_mode=strict_mode)
            preview_images.append(vis_img)
        except Exception as e:
            errors.append(f"{img_path.name}: {e}")
            if strict_mode:
                raise
    
    if len(preview_images) == 0:
        error_details = "\n".join(errors) if errors else "未知错误"
        raise ValueError(f"❌ 所有图像预览失败:\n{error_details}")
    
    n = len(preview_images)
    if n == 1:
        return preview_images[0]
    elif n == 2:
        grid_size = (1, 2)
    elif n <= 4:
        grid_size = (2, 2)
    else:
        cols = 3
        rows = (n + cols - 1) // cols
        grid_size = (rows, cols)
    
    if high_resolution:
        return create_preview_grid(preview_images, grid_size, preserve_resolution=True, max_cell_size=(2000, 2000))
    else:
        return create_preview_grid(preview_images, grid_size, cell_size=(800, 800))


def preview_segmentation_dataset(
    images_dir: str,
    masks_dir: str,
    num_samples: int = 4,
    seed: int = None,
    alpha: float = 0.5,
) -> Image.Image:
    """预览分割数据集【工业级修复版】"""
    if Image is None:
        raise ImportError("❌ 需要安装Pillow: pip install Pillow")
    
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    
    if not images_path.exists():
        raise FileNotFoundError(f"❌ 图像文件夹不存在: {images_dir}")
    if not masks_path.exists():
        raise FileNotFoundError(f"❌ Mask文件夹不存在: {masks_dir}")
    
    samples = []
    for mask_file in masks_path.glob('*.png'):
        stem = mask_file.stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
            img_file = images_path / f"{stem}{ext}"
            if img_file.exists():
                samples.append((img_file, mask_file))
                break
    
    if len(samples) == 0:
        raise ValueError(f"❌ 没有找到匹配的图像-Mask配对")
    
    if seed is not None:
        random.seed(seed)
    selected = random.sample(samples, min(num_samples, len(samples)))
    
    preview_images = []
    errors = []
    
    for img_path, mask_path in selected:
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            mask = np.array(Image.open(mask_path))
            vis_img = visualize_mask_overlay(img, mask, alpha=alpha, ignore_index=255)
            preview_images.append(vis_img)
        except Exception as e:
            errors.append(f"{img_path.name}: {e}")
    
    if len(preview_images) == 0:
        error_details = "\n".join(errors) if errors else "未知错误"
        raise ValueError(f"❌ 所有图像预览失败:\n{error_details}")
    
    n = len(preview_images)
    if n == 1:
        return preview_images[0]
    elif n == 2:
        grid_size = (1, 2)
    elif n <= 4:
        grid_size = (2, 2)
    else:
        cols = 3
        rows = (n + cols - 1) // cols
        grid_size = (rows, cols)
    
    return create_preview_grid(preview_images, grid_size, preserve_resolution=True, max_cell_size=(2000, 2000))
