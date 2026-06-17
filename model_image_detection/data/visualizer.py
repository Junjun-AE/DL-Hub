"""
标注可视化模块 - 用于预览LabelMe标注和YOLO检测结果
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
    """
    获取类别对应的颜色
    
    Args:
        class_id: 类别ID
    
    Returns:
        RGB颜色元组
    """
    return COLORS[class_id % len(COLORS)]


def visualize_labelme_annotation(
    image_path: str,
    json_path: str,
    output_path: Optional[str] = None,
    line_width: int = 3,
    font_size: int = 16,
) -> Optional[Image.Image]:
    """
    可视化LabelMe标注
    
    Args:
        image_path: 图像文件路径
        json_path: JSON标注文件路径
        output_path: 输出图像路径（可选）
        line_width: 边框线宽
        font_size: 字体大小
    
    Returns:
        PIL Image对象
    """
    if Image is None:
        raise ImportError("需要安装Pillow: pip install Pillow")
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 根据图像大小动态调整线宽和字体
    img_width, img_height = img.size
    scale_factor = max(img_width, img_height) / 1000
    line_width = max(2, int(line_width * scale_factor))
    font_size = max(12, int(font_size * scale_factor))
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    
    # 加载标注
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 绘制每个标注框
    for shape in data.get('shapes', []):
        if shape.get('shape_type') != 'rectangle':
            continue
        
        label = shape.get('label', '')
        points = shape.get('points', [])
        
        if len(points) < 2:
            continue
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # 确保坐标顺序正确
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 获取颜色
        try:
            class_id = int(label)
            color = get_color_for_class(class_id)
        except ValueError:
            color = (255, 0, 0)  # 默认红色
        
        # 绘制边框
        for i in range(line_width):
            draw.rectangle(
                [x1 - i, y1 - i, x2 + i, y2 + i],
                outline=color
            )
        
        # 绘制标签背景
        text = str(label)
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # 绘制标签文字
        draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)
    
    # 保存或返回
    if output_path:
        img.save(output_path)
    
    return img


def visualize_yolo_prediction(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    conf_threshold: float = 0.25,
    line_width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """
    可视化YOLO检测结果
    
    Args:
        image: 图像数组 (H, W, C)
        boxes: 边界框数组 (N, 4) [x1, y1, x2, y2]
        class_ids: 类别ID数组 (N,)
        scores: 置信度数组 (N,)
        class_names: 类别名称字典
        conf_threshold: 置信度阈值
        line_width: 边框线宽
        font_size: 字体大小
    
    Returns:
        PIL Image对象
    """
    if Image is None:
        raise ImportError("需要安装Pillow: pip install Pillow")
    
    # 转换为PIL图像
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
    draw = ImageDraw.Draw(img)
    
    # 根据图像大小动态调整线宽和字体
    img_width, img_height = img.size
    scale_factor = max(img_width, img_height) / 1000
    line_width = max(2, int(line_width * scale_factor))
    font_size = max(12, int(font_size * scale_factor))
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    
    # 绘制每个检测框
    for box, class_id, score in zip(boxes, class_ids, scores):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box
        class_id = int(class_id)
        
        # 获取颜色
        color = get_color_for_class(class_id)
        
        # 绘制边框
        for i in range(line_width):
            draw.rectangle(
                [x1 - i, y1 - i, x2 + i, y2 + i],
                outline=color
            )
        
        # 准备标签文字
        if class_names and class_id in class_names:
            label = f"{class_names[class_id]} {score:.2f}"
        else:
            label = f"{class_id} {score:.2f}"
        
        # 绘制标签背景
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # 绘制标签文字
        draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    return img


def visualize_ultralytics_result(
    result,
    conf_threshold: float = 0.25,
    line_width: int = 3,
) -> Optional[Image.Image]:
    """
    可视化ultralytics检测结果
    
    Args:
        result: ultralytics Results对象
        conf_threshold: 置信度阈值
        line_width: 边框线宽
    
    Returns:
        PIL Image对象
    """
    if Image is None:
        raise ImportError("需要安装Pillow: pip install Pillow")
    
    # 使用ultralytics内置的绘制方法
    try:
        # 获取绘制后的图像
        plotted = result.plot(
            conf=conf_threshold,
            line_width=line_width,
            pil=True,
        )
        
        if isinstance(plotted, np.ndarray):
            return Image.fromarray(plotted)
        return plotted
    except Exception as e:
        print(f"可视化出错: {e}")
        return None


def create_preview_grid(
    images: List[Image.Image],
    grid_size: Tuple[int, int] = (2, 2),
    cell_size: Tuple[int, int] = (800, 800),
    preserve_resolution: bool = False,
    max_cell_size: Tuple[int, int] = (2000, 2000),
) -> Image.Image:
    """
    创建图像网格预览
    
    Args:
        images: 图像列表
        grid_size: 网格尺寸 (rows, cols)
        cell_size: 单元格尺寸 (width, height)，preserve_resolution=False时使用
        preserve_resolution: 是否保持原始分辨率
        max_cell_size: preserve_resolution=True时单个图像的最大尺寸
    
    Returns:
        网格图像
    """
    if Image is None:
        raise ImportError("需要安装Pillow: pip install Pillow")
    
    rows, cols = grid_size
    
    if preserve_resolution:
        # 高分辨率模式：根据图像实际大小动态计算网格尺寸
        max_cell_w, max_cell_h = max_cell_size
        
        # 计算每个单元格的实际尺寸（按最大图像计算）
        actual_cell_sizes = []
        for img in images[:rows * cols]:
            # 限制单个图像的最大尺寸
            w, h = img.size
            if w > max_cell_w or h > max_cell_h:
                scale = min(max_cell_w / w, max_cell_h / h)
                w, h = int(w * scale), int(h * scale)
            actual_cell_sizes.append((w, h))
        
        if not actual_cell_sizes:
            cell_w, cell_h = cell_size
        else:
            cell_w = max(s[0] for s in actual_cell_sizes) + 20  # 添加边距
            cell_h = max(s[1] for s in actual_cell_sizes) + 20
    else:
        cell_w, cell_h = cell_size
    
    # 创建画布
    grid_img = Image.new('RGB', (cols * cell_w, rows * cell_h), (255, 255, 255))
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        # 缩放图像以适应单元格
        img_resized = img.copy()
        target_w = cell_w - 10
        target_h = cell_h - 10
        
        if preserve_resolution:
            # 高分辨率模式：只在超出最大尺寸时缩放
            w, h = img_resized.size
            if w > target_w or h > target_h:
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = img_resized.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            # 普通模式：总是缩放到目标尺寸
            img_resized.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 计算居中位置
        x = col * cell_w + (cell_w - img_resized.width) // 2
        y = row * cell_h + (cell_h - img_resized.height) // 2
        
        # 粘贴到网格
        grid_img.paste(img_resized, (x, y))
    
    return grid_img


def preview_dataset(
    images_dir: str,
    jsons_dir: str,
    num_samples: int = 4,
    seed: int = None,
    high_resolution: bool = True,
) -> Optional[Image.Image]:
    """
    预览数据集（显示带标注的样本）
    
    Args:
        images_dir: 图像文件夹
        jsons_dir: JSON标注文件夹
        num_samples: 预览样本数
        seed: 随机种子（None表示每次随机）
        high_resolution: 是否使用高分辨率模式
    
    Returns:
        预览网格图像
    """
    if Image is None:
        raise ImportError("需要安装Pillow: pip install Pillow")
    
    images_path = Path(images_dir)
    jsons_path = Path(jsons_dir)
    
    # 收集配对的样本
    samples = []
    for json_file in jsons_path.glob('*.json'):
        stem = json_file.stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
            img_file = images_path / f"{stem}{ext}"
            if img_file.exists():
                samples.append((img_file, json_file))
                break
    
    if len(samples) == 0:
        return None
    
    # 随机选择样本
    if seed is not None:
        random.seed(seed)
    selected = random.sample(samples, min(num_samples, len(samples)))
    
    # 可视化每个样本
    preview_images = []
    for img_path, json_path in selected:
        try:
            vis_img = visualize_labelme_annotation(str(img_path), str(json_path))
            preview_images.append(vis_img)
        except Exception as e:
            print(f"预览出错: {e}")
    
    if len(preview_images) == 0:
        return None
    
    # 确定网格大小
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
        # 高分辨率模式：保持原始分辨率，每个单元格最大2000x2000
        return create_preview_grid(
            preview_images, 
            grid_size, 
            preserve_resolution=True,
            max_cell_size=(2000, 2000)  # 提高到 2000x2000
        )
    else:
        # 普通模式：缩小到固定尺寸
        return create_preview_grid(preview_images, grid_size, cell_size=(800, 800))  # 提高到 800x800