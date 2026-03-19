"""
SevSeg-YOLO 数据集验证器
验证6列标签格式、data.yaml配置、图像文件完整性
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    message: str
    num_classes: int = 0
    class_names: List[str] = None
    train_count: int = 0
    val_count: int = 0
    total_annotations: int = 0
    has_score: int = 0
    no_score: int = 0
    score_coverage: float = 0.0

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = []


def validate_label_file(label_path: str, expected_nc: int = None) -> Tuple[bool, str, Dict]:
    """
    验证单个标签文件是否符合SevSeg-YOLO格式

    Args:
        label_path: 标签文件路径
        expected_nc: 期望的类别数

    Returns:
        (is_valid, message, stats)
    """
    stats = {"lines": 0, "has_score": 0, "no_score": 0, "errors": []}

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return False, f"无法读取文件: {e}", stats

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        stats["lines"] += 1

        # 检查列数: 5列(纯检测) 或 6列(含severity)
        if len(parts) not in (5, 6):
            stats["errors"].append(f"第{i}行: 列数={len(parts)}，期望5或6列")
            continue

        try:
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            stats["errors"].append(f"第{i}行: 数值解析失败")
            continue

        # 验证class_id
        if class_id < 0:
            stats["errors"].append(f"第{i}行: class_id={class_id} 不能为负")

        if expected_nc is not None and class_id >= expected_nc:
            stats["errors"].append(f"第{i}行: class_id={class_id} >= nc={expected_nc}")

        # 验证归一化坐标
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0 or val > 1:
                stats["errors"].append(f"第{i}行: {name}={val} 超出[0,1]范围")

        # 检查severity列
        if len(parts) == 6:
            try:
                score = float(parts[5])
                if score == -1:
                    stats["no_score"] += 1  # -1 表示缺失
                elif 0 <= score <= 10:
                    stats["has_score"] += 1
                else:
                    stats["errors"].append(f"第{i}行: severity={score} 超出[-1, 10]范围")
            except ValueError:
                stats["errors"].append(f"第{i}行: severity列解析失败")
        else:
            stats["no_score"] += 1

    if stats["errors"]:
        err_preview = "\n".join(stats["errors"][:5])
        if len(stats["errors"]) > 5:
            err_preview += f"\n... 还有{len(stats['errors']) - 5}个错误"
        return False, f"标签格式错误:\n{err_preview}", stats

    return True, "OK", stats


def validate_data_yaml(yaml_path: str) -> ValidationResult:
    """
    验证data.yaml是否是有效的SevSeg-YOLO数据集配置

    Args:
        yaml_path: data.yaml文件路径

    Returns:
        ValidationResult
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return ValidationResult(False, f"❌ data.yaml不存在: {yaml_path}")

    try:
        import yaml
        data = None
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(yaml_path, 'r', encoding=enc) as f:
                    data = yaml.safe_load(f)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if data is None:
            return ValidationResult(False, f"❌ data.yaml编码无法识别（已尝试utf-8/gbk/gb2312/latin-1）")
    except Exception as e:
        return ValidationResult(False, f"❌ data.yaml解析失败: {e}")

    if not isinstance(data, dict):
        return ValidationResult(False, "❌ data.yaml格式无效")

    # 检查必需字段
    nc = data.get('nc', 0)
    names = data.get('names', {})
    train_path = data.get('train', '')
    val_path = data.get('val', '')
    base_path = data.get('path', '')

    if nc <= 0:
        return ValidationResult(False, "❌ nc (类别数) 必须 > 0")

    if not train_path:
        return ValidationResult(False, "❌ 缺少 train 路径")

    if not val_path:
        return ValidationResult(False, "❌ 缺少 val 路径")

    # 解析类别名
    if isinstance(names, dict):
        class_names = [names.get(i, f"class_{i}") for i in range(nc)]
    elif isinstance(names, list):
        class_names = names[:nc]
    else:
        class_names = [f"class_{i}" for i in range(nc)]

    # 验证路径
    if base_path:
        base = Path(base_path)
        train_dir = base / train_path
        val_dir = base / val_path
    else:
        base = yaml_path.parent
        train_dir = base / train_path
        val_dir = base / val_path

    # 统计图像数量
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    train_count = 0
    val_count = 0

    if train_dir.exists():
        train_count = sum(1 for f in train_dir.iterdir() if f.suffix.lower() in img_exts)
    else:
        return ValidationResult(False, f"❌ 训练集目录不存在: {train_dir}")

    if val_dir.exists():
        val_count = sum(1 for f in val_dir.iterdir() if f.suffix.lower() in img_exts)
    else:
        return ValidationResult(False, f"❌ 验证集目录不存在: {val_dir}")

    if train_count == 0:
        return ValidationResult(False, f"❌ 训练集为空: {train_dir}")

    # 检查labels目录
    labels_train = train_dir.parent.parent / 'labels' / train_dir.name
    labels_val = val_dir.parent.parent / 'labels' / val_dir.name

    total_annotations = 0
    has_score = 0
    no_score = 0

    for labels_dir in [labels_train, labels_val]:
        if labels_dir.exists():
            for txt_file in labels_dir.glob("*.txt"):
                ok, msg, stats = validate_label_file(str(txt_file), expected_nc=nc)
                total_annotations += stats["lines"]
                has_score += stats["has_score"]
                no_score += stats["no_score"]

    score_coverage = has_score / max(total_annotations, 1)

    # 构建消息
    lines = [
        f"✅ 数据集验证通过",
        f"   📁 路径: {yaml_path}",
        f"   📊 类别数: {nc} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})",
        f"   🖼️ 训练集: {train_count} 张",
        f"   🖼️ 验证集: {val_count} 张",
        f"   🏷️ 总标注: {total_annotations}",
        f"   📏 Severity标注: {has_score}/{total_annotations} ({score_coverage:.0%}覆盖率)",
    ]

    if score_coverage == 0:
        lines.append("   ⚠️ 无Severity标注，ScoreHead将不会学习")
    elif score_coverage < 0.3:
        lines.append(f"   ⚠️ Severity标注覆盖率较低({score_coverage:.0%})，建议补充标注")

    # 检查score配置
    score_cfg = data.get('score', {})
    if score_cfg and score_cfg.get('enabled', False):
        lines.append("   ✅ Score配置已启用")
    else:
        lines.append("   ℹ️ data.yaml中无score配置块（训练时将使用默认参数）")

    return ValidationResult(
        is_valid=True,
        message="\n".join(lines),
        num_classes=nc,
        class_names=class_names,
        train_count=train_count,
        val_count=val_count,
        total_annotations=total_annotations,
        has_score=has_score,
        no_score=no_score,
        score_coverage=score_coverage,
    )


def validate_images_jsons(images_dir: str, jsons_dir: str) -> str:
    """
    验证图像和JSON标注目录（LabelMe格式，用于数据转换前检查）

    Args:
        images_dir: 图像文件夹路径
        jsons_dir: JSON标注文件夹路径

    Returns:
        验证消息字符串
    """
    images_path = Path(images_dir) if images_dir else None
    jsons_path = Path(jsons_dir) if jsons_dir else None

    if not images_dir or not images_dir.strip():
        return "⏳ 请输入图像文件夹路径"

    if not images_path or not images_path.exists():
        return f"❌ 图像目录不存在: {images_dir}"

    if not jsons_dir or not jsons_dir.strip():
        return "⏳ 请输入JSON标注文件夹路径"

    if not jsons_path or not jsons_path.exists():
        return f"❌ JSON目录不存在: {jsons_dir}"

    # 统计
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [f for f in images_path.iterdir() if f.suffix.lower() in img_exts]
    jsons = list(jsons_path.glob("*.json"))

    if not images:
        return f"❌ 图像目录中无图片文件: {images_dir}"

    if not jsons:
        return f"❌ JSON目录中无标注文件: {jsons_dir}"

    # 检查匹配
    img_stems = {f.stem for f in images}
    json_stems = {f.stem for f in jsons}
    matched = img_stems & json_stems
    unmatched_imgs = img_stems - json_stems
    unmatched_jsons = json_stems - img_stems

    lines = [
        f"✅ 数据目录验证",
        f"   🖼️ 图像: {len(images)} 张",
        f"   📝 JSON: {len(jsons)} 个",
        f"   ✅ 匹配: {len(matched)} 对",
    ]

    if unmatched_imgs:
        lines.append(f"   ⚠️ {len(unmatched_imgs)} 张图片无对应JSON")
    if unmatched_jsons:
        lines.append(f"   ⚠️ {len(unmatched_jsons)} 个JSON无对应图片")

    # 抽样检查severity字段
    import json as json_mod
    has_severe = 0
    no_severe = 0
    total_shapes = 0
    sample_count = min(10, len(jsons))

    for jf in jsons[:sample_count]:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json_mod.load(f)
            for shape in data.get("shapes", []):
                total_shapes += 1
                if shape.get("severe") is not None:
                    has_severe += 1
                else:
                    no_severe += 1
        except Exception:
            pass

    if total_shapes > 0:
        lines.append(f"   📏 Severity字段 (抽样{sample_count}个): {has_severe}/{total_shapes} 有值")

    return "\n".join(lines)
