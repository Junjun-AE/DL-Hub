# -*- coding: utf-8 -*-
"""
PatchCore 推理预测器

支持多种后端:
1. PyTorch (开发调试)
2. ONNX Runtime (跨平台)
3. TensorRT (高性能部署)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image
import cv2

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PredictionResult:
    """预测结果"""
    # 图像级结果
    score: float = 0.0               # 归一化分数 (0-100)
    is_anomaly: bool = False         # 是否异常
    raw_score: float = 0.0           # 原始分数
    
    # 像素级结果
    anomaly_map: np.ndarray = None   # 异常热力图 (H, W), 0-100
    binary_mask: np.ndarray = None   # 二值掩码 (H, W)
    
    # 元信息
    inference_time_ms: float = 0.0
    threshold: float = 50.0
    
    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'is_anomaly': self.is_anomaly,
            'raw_score': self.raw_score,
            'inference_time_ms': self.inference_time_ms,
            'threshold': self.threshold,
        }


class PatchCorePredictor:
    """
    PatchCore 预测器
    
    使用方式:
    ```python
    predictor = PatchCorePredictor.from_package("model.pkg")
    result = predictor.predict(image)
    print(f"Score: {result.score}, Anomaly: {result.is_anomaly}")
    ```
    """
    
    def __init__(
        self,
        config: Dict,
        backbone,
        pca_model: Optional[Dict],
        memory_bank: np.ndarray,
        faiss_index,
        normalization: Dict,
        thresholds: Dict,
        device: str = 'auto',
    ):
        """
        初始化预测器
        
        Args:
            config: 模型配置
            backbone: Backbone网络
            pca_model: PCA模型
            memory_bank: Memory Bank
            faiss_index: Faiss索引
            normalization: 归一化参数
            thresholds: 阈值配置
            device: 计算设备
        """
        self.config = config
        self.backbone = backbone
        self.pca_model = pca_model
        self.memory_bank = memory_bank
        self.faiss_index = faiss_index
        self.normalization = normalization
        self.thresholds = thresholds
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 只对PyTorch模型调用to()和eval()，ONNXBackbone不需要
        if self.backbone is not None and not isinstance(self.backbone, ONNXBackbone):
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()
        
        # 当前阈值
        self.threshold = thresholds.get('default', 50.0)
        
        # 预处理参数
        self.image_size = config['preprocessing']['input_size'][0]
        self.mean = np.array(config['preprocessing']['normalize']['mean'])
        self.std = np.array(config['preprocessing']['normalize']['std'])
        
        # KNN参数
        self.knn_k = config['knn']['k']
        
        # 后处理参数
        self.gaussian_sigma = config['postprocessing']['gaussian_blur'].get('sigma', 4.0)
    
    @classmethod
    def from_package(cls, model_path: str, device: str = 'auto') -> 'PatchCorePredictor':
        """
        从模型包加载预测器
        
        Args:
            model_path: 模型路径 (.pkg文件或目录)
            device: 计算设备
        
        Returns:
            PatchCorePredictor实例
        """
        from export.exporter import load_patchcore_model
        
        model_data = load_patchcore_model(model_path)
        
        # 加载Backbone
        backbone = cls._load_backbone(model_data, device)
        
        return cls(
            config=model_data['config'],
            backbone=backbone,
            pca_model=model_data['pca_model'],
            memory_bank=model_data['features'],
            faiss_index=model_data['faiss_index'],
            normalization=model_data['normalization'],
            thresholds=model_data['thresholds'],
            device=device,
        )
    
    @staticmethod
    def _load_backbone(model_data: Dict, device: str):
        """加载Backbone网络"""
        model_dir = model_data['model_dir']
        config = model_data['config']
        
        backbone_name = config['backbone']['name']
        layers = config['backbone']['layers']
        
        # 尝试从ONNX加载
        onnx_path = model_dir / 'backbone' / 'backbone.onnx'
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                # 使用ONNX Runtime
                return ONNXBackbone(str(onnx_path), device)
            except ImportError:
                pass
        
        # 使用timm重建
        try:
            import timm
            
            backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[2, 3],
            )
            return backbone
        except ImportError:
            raise ImportError("需要安装 onnxruntime 或 timm")
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_visualization: bool = True,
    ) -> PredictionResult:
        """
        预测单张图像
        
        Args:
            image: 输入图像 (路径/numpy数组/PIL图像)
            return_visualization: 是否返回可视化结果
        
        Returns:
            PredictionResult
        """
        start_time = time.time()
        
        # 预处理
        input_tensor, original_size = self._preprocess(image)
        
        # 特征提取
        features = self._extract_features(input_tensor)
        
        # PCA变换
        if self.pca_model is not None:
            features = self._apply_pca(features)
        
        # KNN距离计算
        distances = self._compute_distances(features)
        
        # 生成异常图
        anomaly_map = self._generate_anomaly_map(distances, original_size)
        
        # 计算图像级分数
        raw_score = float(np.max(anomaly_map))
        
        # 归一化分数
        score = self._normalize_score(raw_score)
        
        # 判断是否异常
        is_anomaly = score >= self.threshold
        
        # 创建结果
        result = PredictionResult(
            score=score,
            is_anomaly=is_anomaly,
            raw_score=raw_score,
            threshold=self.threshold,
            inference_time_ms=(time.time() - start_time) * 1000,
        )
        
        if return_visualization:
            # 归一化异常图
            result.anomaly_map = self._normalize_anomaly_map(anomaly_map)
            
            # 生成二值掩码
            result.binary_mask = (result.anomaly_map >= self.threshold).astype(np.uint8) * 255
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 4,
    ) -> List[PredictionResult]:
        """批量预测"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # 暂时使用单张预测
            for img in batch:
                results.append(self.predict(img, return_visualization=False))
        
        return results
    
    def _preprocess(self, image) :
        """预处理图像"""
        # 加载图像
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size  # (W, H)
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # 转numpy
        image = np.array(image).astype(np.float32) / 255.0
        
        # 归一化
        image = (image - self.mean) / self.std
        
        # 转tensor (H, W, C) -> (1, C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        
        tensor = torch.from_numpy(image).float().to(self.device)
        
        return tensor, original_size
    
    def _extract_features(self, input_tensor) -> np.ndarray:
        """提取特征"""
        with torch.no_grad():
            if isinstance(self.backbone, ONNXBackbone):
                features = self.backbone(input_tensor.cpu().numpy())
            else:
                feature_maps = self.backbone(input_tensor)
                features = self._process_feature_maps(feature_maps)
        
        return features
    
    def _process_feature_maps(self, feature_maps) -> np.ndarray:
        """处理特征图"""
        if isinstance(feature_maps, dict):
            feature_list = list(feature_maps.values())
        else:
            feature_list = feature_maps
        
        # 统一尺寸
        target_size = feature_list[0].shape[-2:]
        processed = []
        
        for feat in feature_list:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            processed.append(feat)
        
        # 拼接
        features = torch.cat(processed, dim=1)
        
        # 重排为 (N, C)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        return features.cpu().numpy().astype(np.float32)
    
    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """应用PCA变换"""
        components = self.pca_model['components']
        mean = self.pca_model['mean']
        
        centered = features - mean
        transformed = np.dot(centered, components.T)
        
        return transformed.astype(np.float32)
    
    def _compute_distances(self, features: np.ndarray) -> np.ndarray:
        """计算KNN距离"""
        if self.faiss_index is not None:
            distances, _ = self.faiss_index.search(features, self.knn_k)
        else:
            # Numpy fallback
            dists = np.linalg.norm(
                features[:, np.newaxis] - self.memory_bank[np.newaxis],
                axis=2
            )
            indices = np.argsort(dists, axis=1)[:, :self.knn_k]
            distances = np.take_along_axis(dists, indices, axis=1)
        
        # 取最近邻距离
        return distances[:, 0]
    
    def _generate_anomaly_map(self, distances: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """生成异常图"""
        # 计算特征图尺寸
        H = W = self.image_size // 8
        
        # Reshape
        anomaly_map = distances.reshape(H, W)
        
        # 上采样到原始尺寸
        anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
        
        # 高斯平滑
        if self.gaussian_sigma > 0:
            kernel_size = int(4 * self.gaussian_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            anomaly_map = cv2.GaussianBlur(anomaly_map, (kernel_size, kernel_size), self.gaussian_sigma)
        
        return anomaly_map
    
    def _normalize_score(self, raw_score: float) -> float:
        """归一化分数到0-100"""
        p1 = self.normalization['p1']
        p99 = self.normalization['p99']
        
        if p99 - p1 < 1e-8:
            return 50.0
        
        normalized = (raw_score - p1) / (p99 - p1) * 100
        return max(0, normalized)
    
    def _normalize_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """归一化异常图"""
        p1 = self.normalization['p1']
        p99 = self.normalization['p99']
        
        if p99 - p1 < 1e-8:
            return np.full_like(anomaly_map, 50.0)
        
        normalized = (anomaly_map - p1) / (p99 - p1) * 100
        return np.clip(normalized, 0, 100).astype(np.float32)
    
    def set_threshold(self, threshold: float):
        """设置阈值"""
        self.threshold = max(0, min(100, threshold))
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold
    
    def get_threshold_presets(self) -> Dict[str, float]:
        """获取预设阈值"""
        return {k: v for k, v in self.thresholds.items() 
                if isinstance(v, (int, float))}


class ONNXBackbone:
    """ONNX Backbone封装"""
    
    def __init__(self, onnx_path: str, device: str = 'auto'):
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
    
    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: input_array})
        
        # 合并多层特征
        if len(outputs) > 1:
            # 上采样到相同尺寸
            target_size = outputs[0].shape[-2:]
            processed = []
            for out in outputs:
                if out.shape[-2:] != target_size:
                    # 简单上采样
                    out = np.repeat(np.repeat(out, 2, axis=2), 2, axis=3)
                    out = out[:, :, :target_size[0], :target_size[1]]
                processed.append(out)
            features = np.concatenate(processed, axis=1)
        else:
            features = outputs[0]
        
        B, C, H, W = features.shape
        features = features.transpose(0, 2, 3, 1).reshape(-1, C)
        
        return features.astype(np.float32)


def create_visualization(
    image: np.ndarray,
    result: PredictionResult,
    show_score: bool = True,
) -> Dict[str, np.ndarray]:
    """
    创建可视化结果
    
    Returns:
        {
            'original': 原图,
            'heatmap': 热力图,
            'overlay': 叠加图,
            'binary': 二值图,
            'contour': 轮廓图,
        }
    """
    if isinstance(image, (str, Path)):
        image = np.array(Image.open(image).convert('RGB'))
    
    H, W = image.shape[:2]
    
    visualizations = {
        'original': image.copy(),
    }
    
    if result.anomaly_map is not None:
        anomaly_map = result.anomaly_map
        
        # 调整尺寸
        if anomaly_map.shape != (H, W):
            anomaly_map = cv2.resize(anomaly_map, (W, H))
        
        # 热力图
        heatmap_normalized = np.clip(anomaly_map / 100, 0, 1)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        visualizations['heatmap'] = heatmap_colored
        
        # 叠加图
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        visualizations['overlay'] = overlay
        
        # 二值图
        if result.binary_mask is not None:
            binary = result.binary_mask
            if binary.shape != (H, W):
                binary = cv2.resize(binary, (W, H))
            visualizations['binary'] = binary
            
            # 轮廓图
            contour_img = image.copy()
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            visualizations['contour'] = contour_img
    
    # 添加分数标注
    if show_score:
        for key in ['overlay', 'original']:
            if key in visualizations:
                img = visualizations[key].copy()
                status = "ANOMALY" if result.is_anomaly else "NORMAL"
                color = (255, 0, 0) if result.is_anomaly else (0, 255, 0)
                
                cv2.putText(
                    img, f"Score: {result.score:.1f} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                visualizations[key] = img
    
    return visualizations
