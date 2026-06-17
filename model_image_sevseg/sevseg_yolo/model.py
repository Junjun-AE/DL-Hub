"""
SevSegYOLO — 统一入口类
========================

一行代码完成工业缺陷的 检测 + 严重程度评分 + 零标注近似分割:

    from sevseg_yolo import SevSegYOLO

    model = SevSegYOLO("best.pt")
    result = model.predict("image.jpg")

    for det in result.detections:
        print(det.bbox, det.severity, det.class_name, det.mask.shape)

    result.visualize().save("output.jpg")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════

@dataclass
class Detection:
    """单个缺陷检测结果。

    Attributes:
        bbox: [x1, y1, x2, y2] 原图坐标
        class_id: 类别ID
        class_name: 类别名称
        confidence: 检测置信度 [0, 1]
        severity: 严重程度分数 [0, 10]
        mask: 二值掩膜 {0, 1}, 尺寸 = bbox区域大小 (rh, rw)
        fill_ratio: 掩膜填充率
    """
    bbox: list[int]
    class_id: int
    class_name: str
    confidence: float
    severity: float
    mask: np.ndarray
    fill_ratio: float

    @property
    def area(self) -> int:
        """bbox面积(像素)"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def severity_level(self) -> str:
        """严重程度等级: low(<4) / medium(4-7) / high(>7)"""
        if self.severity < 4:
            return "low"
        elif self.severity < 7:
            return "medium"
        return "high"

    @property
    def color(self) -> tuple[int, int, int]:
        """BGR颜色 (按严重程度: 绿/黄/红)"""
        if self.severity < 4:
            return (0, 200, 0)
        elif self.severity < 7:
            return (0, 200, 255)
        return (0, 0, 255)

    def __repr__(self):
        return (f"Detection(cls={self.class_name}, sev={self.severity:.1f}, "
                f"conf={self.confidence:.2f}, bbox={self.bbox}, fill={self.fill_ratio:.3f})")


@dataclass
class SevSegResult:
    """单张图像的完整预测结果。

    Attributes:
        image: 原始图像 (BGR, np.ndarray)
        image_path: 图像路径
        detections: Detection列表
    """
    image: np.ndarray
    image_path: str = ""
    detections: list[Detection] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def filter(self, min_severity: float = 0, min_confidence: float = 0,
               class_ids: list[int] = None) -> "SevSegResult":
        """按条件过滤检测结果。"""
        filtered = []
        for d in self.detections:
            if d.severity < min_severity:
                continue
            if d.confidence < min_confidence:
                continue
            if class_ids is not None and d.class_id not in class_ids:
                continue
            filtered.append(d)
        return SevSegResult(image=self.image, image_path=self.image_path,
                            detections=filtered)

    def visualize(self, show_mask: bool = True, show_label: bool = True,
                  mask_alpha: float = 0.5) -> "SevSegResult":
        """在图像上绘制检测结果。返回self以支持链式调用。

        Args:
            show_mask: 是否绘制mask覆盖
            show_label: 是否绘制标签
            mask_alpha: mask透明度 [0, 1]
        """
        self._vis = self.image.copy()

        for det in self.detections:
            bx1, by1, bx2, by2 = det.bbox
            color = det.color

            # Bbox
            cv2.rectangle(self._vis, (bx1, by1), (bx2, by2), color, 2)

            # Mask overlay
            if show_mask and det.mask is not None and det.mask.sum() > 0:
                roi = self._vis[by1:by2, bx1:bx2]
                m = det.mask
                if m.shape != roi.shape[:2]:
                    m = cv2.resize(m, (roi.shape[1], roi.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                color_arr = np.array(color, dtype=np.float32)
                roi[m > 0] = (roi[m > 0].astype(np.float32) * (1 - mask_alpha) +
                               color_arr * mask_alpha).astype(np.uint8)

            # Label
            if show_label:
                label = f"{det.class_name} Severity={det.severity:.1f} Conf={det.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(self._vis, (bx1, by1 - th - 6), (bx1 + tw, by1), color, -1)
                cv2.putText(self._vis, label, (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return self

    def save(self, path: str, quality: int = 95) -> str:
        """保存可视化结果。"""
        img = getattr(self, '_vis', self.image)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return str(path)

    def show(self):
        """显示可视化结果 (需要GUI环境)。"""
        img = getattr(self, '_vis', self.image)
        cv2.imshow("SevSeg-YOLO", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def to_dict(self) -> list[dict]:
        """转换为字典列表 (方便JSON序列化)。"""
        return [{
            "bbox": d.bbox,
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "severity": d.severity,
            "severity_level": d.severity_level,
            "fill_ratio": d.fill_ratio,
            "mask_shape": list(d.mask.shape) if d.mask is not None else None,
        } for d in self.detections]

    def __repr__(self):
        return f"SevSegResult({self.num_detections} detections, image={self.image_path})"

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)


# ═══════════════════════════════════════════════════════
# 主入口类
# ═══════════════════════════════════════════════════════

class SevSegYOLO:
    """SevSeg-YOLO: 工业缺陷检测 + 严重程度评分 + 零标注近似分割。

    一行代码完成完整推理:
        model = SevSegYOLO("best.pt")
        result = model.predict("image.jpg")

    Args:
        weights: 模型权重路径 (.pt)
        device: 设备 ("0" = GPU 0, "cpu" = CPU)
        conf: 默认置信度阈值
        imgsz: 默认输入尺寸
        mask_enabled: 是否启用MaskGenerator
        mask_version: MaskGenerator版本 ("v3" 或 "v2")
            v3: 双峰通道选择+Canny引导上采样 (推荐，默认)
            v2: 方差通道选择+Canny引导上采样 (兼容旧版)
    """

    def __init__(
        self,
        weights: str,
        device: str = "0",
        conf: float = 0.15,
        imgsz: int = 640,
        mask_enabled: bool = True,
        mask_version: str = "v3",
    ):
        import torch
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import ScoreDetect

        self.conf = conf
        self.imgsz = imgsz
        self.mask_enabled = mask_enabled
        self.mask_version = mask_version.lower().strip()
        self.device = device

        # 加载模型
        self._model = YOLO(weights)
        self._raw = self._model.model
        self._raw.eval()
        self._dev = torch.device(
            f"cuda:{device}" if device != "cpu" and torch.cuda.is_available() else "cpu"
        )
        self._raw.to(self._dev)

        # 获取类别名
        self.class_names = self._model.names if hasattr(self._model, 'names') else {}

        # 注册特征Hook
        self._hooked = {}
        self._hooks = []

        if hasattr(self._raw, 'model') and len(self._raw.model) > 2:
            self._hooks.append(self._raw.model[2].register_forward_hook(
                lambda m, i, o: self._hooked.update(
                    {"layer2": o.detach().cpu().numpy()[0]})))

        for md in self._raw.modules():
            if isinstance(md, ScoreDetect):
                self._hooks.append(md.register_forward_hook(self._feature_hook))
                break

        # 初始化MaskGenerator
        if mask_enabled:
            if self.mask_version == "v3":
                from sevseg_yolo.mask_generator_v3 import MaskGeneratorV3
                self._mask_gen = MaskGeneratorV3(
                    topk_channels=48, adaptive_block=15, adaptive_C=-5,
                    morph_close_k=7, morph_open_k=3,
                    channel_select="bimodal",  # V3唯一改进: 双峰通道选择
                )
            else:
                from sevseg_yolo.mask_generator_v2 import MaskGeneratorV2
                self._mask_gen = MaskGeneratorV2(
                    topk_channels=48, guided_radius=6, guided_eps=0.005,
                    adaptive_block=15, adaptive_C=-5, morph_close_k=7, morph_open_k=3,
                )
        else:
            self._mask_gen = None

        # 模型规模信息
        total_params = sum(p.numel() for p in self._raw.parameters())
        self.model_info = {
            "weights": weights,
            "params": total_params,
            "device": str(self._dev),
            "mask_enabled": mask_enabled,
            "mask_version": self.mask_version,
        }

    def _feature_hook(self, mod, inp, out):
        if isinstance(inp, tuple) and len(inp) > 0:
            x = inp[0]
            if isinstance(x, list) and len(x) >= 2:
                self._hooked["p3"] = x[0].detach().cpu().numpy()[0]
                self._hooked["p4"] = x[1].detach().cpu().numpy()[0]
                if len(x) > 2:
                    self._hooked["p5"] = x[2].detach().cpu().numpy()[0]

    def predict(
        self,
        source: Union[str, np.ndarray, list],
        conf: float = None,
        imgsz: int = None,
    ) -> Union[SevSegResult, list[SevSegResult]]:
        """预测: 检测 + 严重程度评分 + Mask生成。

        Args:
            source: 图像路径、numpy数组(BGR)、或路径列表
            conf: 置信度阈值 (None则用默认)
            imgsz: 输入尺寸 (None则用默认)

        Returns:
            SevSegResult (单张) 或 list[SevSegResult] (多张)
        """
        conf = conf or self.conf
        imgsz = imgsz or self.imgsz

        # 处理批量输入
        if isinstance(source, list):
            return [self._predict_single(s, conf, imgsz) for s in source]

        return self._predict_single(source, conf, imgsz)

    def _predict_single(self, source, conf, imgsz) -> SevSegResult:
        """单张图像预测。"""
        # 读取图像
        if isinstance(source, str):
            orig = cv2.imread(source)
            img_path = source
        elif isinstance(source, np.ndarray):
            orig = source.copy()
            img_path = "<numpy>"
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if orig is None:
            return SevSegResult(image=np.zeros((1, 1, 3), np.uint8),
                                image_path=str(source))

        oh, ow = orig.shape[:2]

        # 运行检测
        self._hooked.clear()
        preds = self._model(source, conf=conf, imgsz=imgsz,
                            device=self._dev, verbose=False)

        if (not preds or preds[0].boxes is None or
                len(preds[0].boxes) == 0 or "p3" not in self._hooked):
            return SevSegResult(image=orig, image_path=img_path)

        # 解析模型输入尺寸
        p3 = self._hooked["p3"]
        input_h, input_w = p3.shape[1] * 8, p3.shape[2] * 8

        detections = []
        for i in range(len(preds[0].boxes)):
            bb = preds[0].boxes.xyxy[i].cpu().numpy()
            cls_id = int(preds[0].boxes.cls[i].item())
            confidence = preds[0].boxes.conf[i].item()
            # SevSeg-YOLO: severity stored in _raw_pred[:, 6] or boxes.data[:, 6]
            severity = 0.0
            if hasattr(preds[0], '_raw_pred') and preds[0]._raw_pred is not None:
                raw = preds[0]._raw_pred
                if raw.shape[-1] >= 7:
                    severity = raw[i, 6].item() * 10
            elif preds[0].boxes.data.shape[-1] >= 7:
                severity = preds[0].boxes.data[i, 6].item() * 10

            # 原图坐标
            bx1, by1 = max(0, int(bb[0])), max(0, int(bb[1]))
            bx2, by2 = min(ow, int(bb[2])), min(oh, int(bb[3]))
            rh, rw = by2 - by1, bx2 - bx1
            if rh < 3 or rw < 3:
                continue

            # 生成Mask
            mask = np.zeros((rh, rw), dtype=np.uint8)
            if self._mask_gen is not None and "p3" in self._hooked:
                try:
                    # V2 和 V3 使用完全相同的调用方式和坐标系统
                    g = min(input_h / oh, input_w / ow)
                    px = round((input_w - ow * g) / 2 - 0.1)
                    py = round((input_h - oh * g) / 2 - 0.1)
                    bbox_model = [
                        max(0, bb[0] * g + px), max(0, bb[1] * g + py),
                        min(input_w, bb[2] * g + px), min(input_h, bb[3] * g + py),
                    ]
                    guide_crop = orig[by1:by2, bx1:bx2]
                    mask_raw = self._mask_gen.generate(
                        bbox=bbox_model,
                        feat_layer2=self._hooked.get("layer2"),
                        feat_p3_fpn=self._hooked["p3"],
                        feat_p4_fpn=self._hooked["p4"],
                        feat_p5_fpn=self._hooked.get("p5"),
                        guide_crop=guide_crop,
                        input_hw=(input_h, input_w),
                    )
                    mask = cv2.resize(mask_raw, (rw, rh),
                                      interpolation=cv2.INTER_NEAREST)
                except Exception:
                    pass

            class_name = self.class_names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                bbox=[bx1, by1, bx2, by2],
                class_id=cls_id,
                class_name=class_name,
                confidence=round(confidence, 4),
                severity=round(severity, 2),
                mask=mask,
                fill_ratio=round(float(mask.mean()), 4),
            ))

        return SevSegResult(image=orig, image_path=img_path,
                            detections=detections)

    def train(self, data: str, epochs: int = 105, batch: int = 32, **kwargs):
        """训练模型。

        Args:
            data: data.yaml路径
            epochs: 训练轮数
            batch: batch size
            **kwargs: 其他Ultralytics训练参数
        """
        return self._model.train(
            task="score_detect", data=data, epochs=epochs, batch=batch,
            imgsz=self.imgsz, mixup=0.0, **kwargs
        )

    def val(self, data: str, **kwargs):
        """验证模型。"""
        return self._model.val(
            task="score_detect", data=data, imgsz=self.imgsz, **kwargs
        )

    def export(self, format: str = "engine", half: bool = True, **kwargs):
        """导出模型 (ONNX/TensorRT)。"""
        return self._model.export(
            format=format, half=half, imgsz=self.imgsz, **kwargs
        )

    def __del__(self):
        for h in self._hooks:
            h.remove()

    def __repr__(self):
        mv = self.mask_version if self.mask_enabled else 'off'
        return (f"SevSegYOLO(params={self.model_info['params']/1e6:.2f}M, "
                f"device={self.model_info['device']}, "
                f"mask={mv})")
