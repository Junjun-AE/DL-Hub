#!/usr/bin/env python3
"""
LabelMe JSON → SevSeg-YOLO 6列标注转换工具

将 LabelMe 格式的 JSON 标注转换为 SevSeg-YOLO 所需的 6列 YOLO 格式:
  class_id  cx_norm  cy_norm  w_norm  h_norm  score

输入结构:
  dataset/
  ├── images/         ← jpg/png 图像
  └── jsons/          ← LabelMe JSON 标注 (含 severe 字段)

输出结构:
  output/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/      ← 6列 YOLO 格式 txt
  │   └── val/
  └── data.yaml       ← SevSeg-YOLO 数据集配置

JSON 字段映射:
  shapes[i].label      → class_id (整数, 从 0 开始)
  shapes[i].points     → [[x1,y1],[x2,y2]] 矩形框 → 归一化 cx,cy,w,h
  shapes[i].severe     → score (0-10 严重程度)
  imageWidth/Height    → 归一化分母

用法:
  python experiments/convert_labelme.py \
      --images /path/to/images \
      --jsons /path/to/jsons \
      --output /path/to/output \
      --val-ratio 0.2 \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_labelme_json(json_path: str) -> dict | None:
    """解析单个 LabelMe JSON 文件。

    Returns:
        dict with keys: image_name, width, height, annotations
        annotations: list of dict with keys: class_id, bbox_xyxy, score
        Returns None if parse fails.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠️ JSON 解析失败 {json_path}: {e}")
        return None

    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)
    img_name = data.get("imagePath", "")

    if img_w <= 0 or img_h <= 0:
        print(f"  ⚠️ 无效尺寸 {json_path}: {img_w}×{img_h}")
        return None

    annotations = []
    for shape in data.get("shapes", []):
        # 只处理矩形框
        if shape.get("shape_type") != "rectangle":
            print(f"  ⚠️ 跳过非矩形标注 ({shape.get('shape_type')}) in {json_path}")
            continue

        # 类别
        label = shape.get("label", "")
        try:
            class_id = int(label)
        except ValueError:
            print(f"  ⚠️ 非整数类别 '{label}' in {json_path}, 跳过")
            continue

        # 边界框: [[x1,y1], [x2,y2]]
        points = shape.get("points", [])
        if len(points) != 2:
            print(f"  ⚠️ 点数不是 2 ({len(points)}) in {json_path}, 跳过")
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        # 确保 x1<x2, y1<y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 裁剪到图像范围
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 - x1 < 1 or y2 - y1 < 1:
            continue  # 无效框

        # 严重程度分数
        score = shape.get("severe", None)
        if score is not None:
            score = float(score)
            if score < 0 or score > 10:
                print(f"  ⚠️ score={score} 超范围 in {json_path}, clamp 到 [0,10]")
                score = max(0.0, min(10.0, score))

        annotations.append({
            "class_id": class_id,
            "bbox_xyxy": (x1, y1, x2, y2),
            "score": score,
        })

    return {
        "image_name": img_name,
        "width": img_w,
        "height": img_h,
        "annotations": annotations,
    }


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """绝对像素 xyxy → 归一化 YOLO cxcywh"""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def format_yolo_line(class_id: int, cx: float, cy: float, w: float, h: float,
                      score: float | None, force_6col: bool = False) -> str:
    """生成 YOLO 格式行。

    当 force_6col=True 时，无 score 的标注用 -1 占位（SevSeg-YOLO 会在
    verify_image_label 中将 -1 转为 NaN）。这确保同一文件中所有行列数一致，
    避免 np.array() 的 inhomogeneous shape 错误。
    """
    base = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    if score is not None:
        return f"{base} {score:.1f}"
    elif force_6col:
        return f"{base} -1"  # 占位符，verify_image_label 会转为 NaN
    return base


def convert_dataset(
    images_dir: str,
    jsons_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    class_remap: dict[int, int] | None = None,
):
    """执行完整的数据集转换。

    Args:
        images_dir: 图像文件夹路径
        jsons_dir: JSON 标注文件夹路径
        output_dir: 输出路径
        val_ratio: 验证集比例
        seed: 随机种子 (划分用)
        class_remap: 可选的类别 ID 重映射 {原始ID: 新ID}
    """
    images_dir = Path(images_dir)
    jsons_dir = Path(jsons_dir)
    output_dir = Path(output_dir)

    # 扫描所有 JSON
    json_files = sorted(jsons_dir.glob("*.json"))
    print(f"\n{'═'*60}")
    print(f"  LabelMe → SevSeg-YOLO 转换")
    print(f"{'═'*60}")
    print(f"  图像目录: {images_dir}")
    print(f"  标注目录: {jsons_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  JSON 文件: {len(json_files)}")
    print(f"  验证集比例: {val_ratio}")
    print(f"  随机种子: {seed}")

    if not json_files:
        print("  ❌ 未找到 JSON 文件!")
        return

    # 解析所有标注
    parsed = []
    stats = Counter()  # 统计

    for jf in json_files:
        result = parse_labelme_json(str(jf))
        if result is None:
            stats["parse_failed"] += 1
            continue

        # 查找对应图像
        img_name = result["image_name"]
        img_path = None
        for ext in ["", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = images_dir / (Path(img_name).stem + ext) if ext else images_dir / img_name
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # 尝试用 JSON 文件名找图像
            stem = jf.stem
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = images_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        if img_path is None:
            stats["image_not_found"] += 1
            print(f"  ⚠️ 图像未找到: {img_name} (JSON: {jf.name})")
            continue

        parsed.append({
            "json_path": jf,
            "image_path": img_path,
            **result,
        })
        stats["total_images"] += 1
        stats["total_annotations"] += len(result["annotations"])

        for ann in result["annotations"]:
            stats[f"class_{ann['class_id']}"] += 1
            if ann["score"] is not None:
                stats["has_score"] += 1
            else:
                stats["no_score"] += 1

    print(f"\n  解析结果:")
    print(f"    有效图像: {stats['total_images']}")
    print(f"    总标注数: {stats['total_annotations']}")
    print(f"    有 score: {stats['has_score']}")
    print(f"    无 score: {stats['no_score']}")
    print(f"    解析失败: {stats['parse_failed']}")
    print(f"    图像缺失: {stats['image_not_found']}")

    # 类别统计
    class_ids = sorted(set(
        ann["class_id"]
        for p in parsed
        for ann in p["annotations"]
    ))
    print(f"    类别: {class_ids}")
    for cid in class_ids:
        print(f"      class {cid}: {stats[f'class_{cid}']} 个标注")

    if not parsed:
        print("  ❌ 无有效数据!")
        return

    # Score 分布统计
    all_scores = [
        ann["score"]
        for p in parsed
        for ann in p["annotations"]
        if ann["score"] is not None
    ]
    if all_scores:
        scores_arr = np.array(all_scores)
        print(f"\n  Score 分布:")
        print(f"    min={scores_arr.min():.1f}, max={scores_arr.max():.1f}, "
              f"mean={scores_arr.mean():.1f}, std={scores_arr.std():.1f}")
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
            n = ((scores_arr >= lo) & (scores_arr <= hi)).sum()
            print(f"    [{lo}-{hi}]: {n} ({n/len(scores_arr):.1%})")

    # 划分 train/val
    random.seed(seed)
    indices = list(range(len(parsed)))
    random.shuffle(indices)
    n_val = max(1, int(len(parsed) * val_ratio))
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    print(f"\n  数据划分: train={len(train_indices)}, val={len(val_indices)}")

    # 创建输出目录
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 类别重映射
    if class_remap is None:
        # 自动: 将原始类别 ID 映射为 0-based 连续 ID
        class_remap = {cid: i for i, cid in enumerate(class_ids)}
    print(f"  类别映射: {class_remap}")

    # 写入
    written = Counter()
    for idx, p in enumerate(parsed):
        split = "val" if idx in val_indices else "train"

        # 复制图像
        dst_img = output_dir / "images" / split / p["image_path"].name
        if not dst_img.exists():
            shutil.copy2(p["image_path"], dst_img)

        # 生成标签文件
        label_name = p["image_path"].stem + ".txt"
        dst_label = output_dir / "labels" / split / label_name

        # 检测同一图像中是否混合有/无 score → 强制统一为 6 列
        anns = p["annotations"]
        has_any_score = any(a["score"] is not None for a in anns)
        has_any_missing = any(a["score"] is None for a in anns)
        force_6col = has_any_score and has_any_missing  # 混合情况

        lines = []
        for ann in anns:
            original_cid = ann["class_id"]
            mapped_cid = class_remap.get(original_cid, original_cid)
            cx, cy, w, h = xyxy_to_yolo(*ann["bbox_xyxy"], p["width"], p["height"])
            line = format_yolo_line(mapped_cid, cx, cy, w, h, ann["score"], force_6col)
            lines.append(line)

        with open(dst_label, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        written[split] += 1

    print(f"  写入: train={written['train']}, val={written['val']}")

    # 生成 data.yaml
    nc = len(class_remap)
    names = {v: f"class_{k}" for k, v in class_remap.items()}
    # 尝试从原始标注推断类别名 (LabelMe 的 label 就是类别名/数字)
    # 你可以在这里手动修改类别名
    yaml_content = f"""# SevSeg-YOLO 数据集配置 (自动生成)
# 来源: {images_dir}
# 生成时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}

path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {nc}
names:
"""
    for i in range(nc):
        yaml_content += f"  {i}: {names.get(i, f'class_{i}')}\n"

    yaml_content += f"""
# SevSeg-YOLO score configuration
score:
  enabled: true
  range: [0.0, 10.0]
  lambda_score: 0.05
  gaussian_sigma: 0.1
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n  ✅ 转换完成!")
    print(f"  data.yaml: {yaml_path}")
    print(f"  训练集: {output_dir / 'images' / 'train'} ({written['train']} 张)")
    print(f"  验证集: {output_dir / 'images' / 'val'} ({written['val']} 张)")

    # 验证抽样
    print(f"\n  标签文件样例:")
    for split in ["train", "val"]:
        txts = sorted((output_dir / "labels" / split).glob("*.txt"))
        if txts:
            sample = txts[0]
            with open(sample) as f:
                content = f.read().strip()
            print(f"    {split}/{sample.name}:")
            for line in content.split("\n")[:3]:
                print(f"      {line}")

    return {
        "output_dir": str(output_dir),
        "yaml_path": str(yaml_path),
        "train_count": written["train"],
        "val_count": written["val"],
        "nc": nc,
        "class_remap": class_remap,
        "stats": dict(stats),
    }


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LabelMe JSON → SevSeg-YOLO 6列标注转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换 (自动 80/20 划分)
  python experiments/convert_labelme.py \\
      --images /data/project/images \\
      --jsons /data/project/jsons \\
      --output /root/sevseg-yolo-project/data/defect_v1

  # 指定验证集比例和类别名
  python experiments/convert_labelme.py \\
      --images /data/images \\
      --jsons /data/jsons \\
      --output data/defect_v1 \\
      --val-ratio 0.15 \\
      --class-names "scratch,bubble,dent"

  # 转换完成后直接验证
  python experiments/convert_labelme.py \\
      --images /data/images \\
      --jsons /data/jsons \\
      --output data/defect_v1 \\
      --verify
        """,
    )
    parser.add_argument("--images", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--jsons", type=str, required=True, help="JSON 标注文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例 (默认 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--class-names", type=str, default=None,
                        help="类别名 (逗号分隔, 按 ID 顺序)")
    parser.add_argument("--verify", action="store_true",
                        help="转换后运行数据质量预检")
    args = parser.parse_args()

    result = convert_dataset(
        images_dir=args.images,
        jsons_dir=args.jsons,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if result is None:
        sys.exit(1)

    # 如果指定了类别名，更新 YAML
    if args.class_names and result:
        names = args.class_names.split(",")
        yaml_path = result["yaml_path"]
        with open(yaml_path) as f:
            content = f.read()
        for i, name in enumerate(names):
            content = content.replace(f"  {i}: class_{list(result['class_remap'].keys())[i] if i < len(result['class_remap']) else i}", f"  {i}: {name.strip()}")
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"\n  类别名已更新: {names}")

    # 验证
    if args.verify and result:
        print(f"\n{'═'*60}")
        print(f"  数据质量预检")
        print(f"{'═'*60}")
        sys.path.insert(0, REPO)
        from ultralytics.data.utils import validate_label_line

        nc = result["nc"]
        errors = 0
        total_lines = 0
        for split in ["train", "val"]:
            label_dir = Path(result["output_dir"]) / "labels" / split
            for f in sorted(label_dir.glob("*.txt")):
                with open(f) as fh:
                    for i, line in enumerate(fh, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        total_lines += 1
                        ok, err, hs = validate_label_line(parts, num_cls=nc)
                        if not ok:
                            print(f"    ❌ {f.name}:{i}: {err}")
                            errors += 1

        print(f"  总标注行: {total_lines}")
        print(f"  格式错误: {errors}")
        print(f"  {'✅ 数据集就绪!' if errors == 0 else '❌ 请修复格式错误'}")


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main()
