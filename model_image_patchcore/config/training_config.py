# -*- coding: utf-8 -*-
"""
PatchCore 训练配置

包含所有训练和导出相关的配置参数
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .default_config import (
    DEFAULT_BACKBONE, DEFAULT_LAYERS, DEFAULT_IMAGE_SIZE,
    DEFAULT_CORESET_RATIO, DEFAULT_PCA_COMPONENTS, DEFAULT_KNN_K,
    DEFAULT_TENSORRT_PRECISION, DEFAULT_THRESHOLD,
    DEFAULT_USE_FP16_FEATURES, DEFAULT_FEATURE_CHUNK_SIZE,
    DEFAULT_INCREMENTAL_PCA, DEFAULT_PCA_BATCH_SIZE,
    DEFAULT_RANDOM_PROJECTION_DIM, DEFAULT_CORESET_MAX_ITER,
    DEFAULT_FEATURE_BATCH_SIZE, DEFAULT_GAUSSIAN_SIGMA,
)


@dataclass
class BackboneConfig:
    """Backbone 配置"""
    name: str = DEFAULT_BACKBONE
    layers: List[str] = field(default_factory=lambda: DEFAULT_LAYERS.copy())
    pretrained: bool = True
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        from .default_config import BACKBONE_OPTIONS
        if self.name in BACKBONE_OPTIONS:
            return BACKBONE_OPTIONS[self.name]['total_dim']
        return 1536  # 默认


@dataclass
class MemoryBankConfig:
    """Memory Bank 配置"""
    # CoreSet 采样
    coreset_sampling_ratio: float = DEFAULT_CORESET_RATIO
    coreset_max_iter: int = DEFAULT_CORESET_MAX_ITER
    
    # PCA 降维
    pca_enabled: bool = True  # 自动判断
    pca_components: int = DEFAULT_PCA_COMPONENTS
    pca_variance_threshold: float = 0.995
    
    # 特征白化 (可选)
    whitening_enabled: bool = False
    whitening_epsilon: float = 1e-5
    
    # 存储格式
    feature_dtype: str = 'float16'


@dataclass
class KNNConfig:
    """KNN 配置"""
    k: int = DEFAULT_KNN_K
    index_type: str = 'auto'  # auto, Flat, IVFFlat, IVFPQ
    
    # IVF 参数
    nlist: int = 100
    nprobe: int = 10
    
    # PQ 参数 (仅 IVFPQ)
    pq_m: int = 32
    pq_bits: int = 8
    
    # 距离度量
    metric: str = 'L2'


@dataclass
class PostprocessConfig:
    """后处理配置"""
    # 上采样
    upsample_mode: str = 'bilinear'
    
    # 高斯平滑
    gaussian_blur_enabled: bool = True
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA
    gaussian_kernel_size: int = 33
    
    # 分数聚合
    score_aggregation: str = 'max'  # max, mean, percentile_99


@dataclass
class NormalizationConfig:
    """归一化配置"""
    method: str = 'percentile'  # percentile, min_max, z_score
    percentile_min: float = 1.0
    percentile_max: float = 99.0
    output_range: tuple = (0.0, 100.0)


@dataclass
class ThresholdConfig:
    """阈值配置"""
    method: str = 'percentile'  # percentile, f1, manual
    default_threshold: float = DEFAULT_THRESHOLD
    
    # 百分位数方法参数
    percentile: float = 99.5
    
    # 预设阈值
    presets: Dict[str, float] = field(default_factory=lambda: {
        'ultra_sensitive': 30.0,
        'sensitive': 40.0,
        'balanced': 50.0,
        'strict': 65.0,
        'very_strict': 80.0,
    })


@dataclass
class ExportConfig:
    """导出配置"""
    # 导出格式
    export_enabled: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['pkg', 'onnx'])
    
    # TensorRT
    tensorrt_enabled: bool = True
    tensorrt_precision: str = DEFAULT_TENSORRT_PRECISION
    tensorrt_int8_enabled: bool = False
    tensorrt_int8_calibration_samples: int = 500
    tensorrt_max_batch_size: int = 16
    tensorrt_workspace_gb: float = 4.0
    
    # Faiss
    faiss_index_type: str = 'auto'
    faiss_gpu: bool = True
    
    # 打包
    package_enabled: bool = True
    package_compression: bool = True


@dataclass
class OptimizationConfig:
    """性能优化配置"""
    # 内存优化
    use_fp16_features: bool = DEFAULT_USE_FP16_FEATURES
    feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE
    
    # PCA 优化
    incremental_pca: bool = DEFAULT_INCREMENTAL_PCA
    pca_batch_size: int = DEFAULT_PCA_BATCH_SIZE
    
    # CoreSet 优化
    random_projection_enabled: bool = True
    random_projection_dim: int = DEFAULT_RANDOM_PROJECTION_DIM
    
    # 批处理
    feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    """数据增强配置
    
    ⚠️ 重要：增强仅用于训练，推理时不使用，以保证一致性。
    """
    # 总开关
    enabled: bool = False
    
    # 翻转
    horizontal_flip: bool = False
    vertical_flip: bool = False
    
    # 旋转
    rotation_degrees: int = 0  # 0=不旋转, 90=90度, 180=任意角度
    
    # 颜色
    brightness_range: float = 0.0  # 0=不变, 0.1=±10%
    contrast_range: float = 0.0    # 0=不变, 0.1=±10%
    
    # 兼容旧字段
    @property
    def brightness_enabled(self) -> bool:
        return self.brightness_range > 0
    
    @property
    def contrast_enabled(self) -> bool:
        return self.contrast_range > 0
    
    @property
    def rotation_enabled(self) -> bool:
        return self.rotation_degrees > 0
    
    @property
    def hflip_enabled(self) -> bool:
        return self.horizontal_flip
    
    @property
    def vflip_enabled(self) -> bool:
        return self.vertical_flip


@dataclass
class TrainingConfig:
    """完整训练配置"""
    # 基本配置
    dataset_dir: str = ""
    output_dir: str = "./output"
    image_size: int = DEFAULT_IMAGE_SIZE
    device: str = 'auto'
    
    # 各模块配置
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    memory_bank: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建"""
        # 处理嵌套配置
        if 'backbone' in data and isinstance(data['backbone'], dict):
            data['backbone'] = BackboneConfig(**data['backbone'])
        if 'memory_bank' in data and isinstance(data['memory_bank'], dict):
            data['memory_bank'] = MemoryBankConfig(**data['memory_bank'])
        if 'knn' in data and isinstance(data['knn'], dict):
            data['knn'] = KNNConfig(**data['knn'])
        if 'postprocess' in data and isinstance(data['postprocess'], dict):
            data['postprocess'] = PostprocessConfig(**data['postprocess'])
        if 'normalization' in data and isinstance(data['normalization'], dict):
            data['normalization'] = NormalizationConfig(**data['normalization'])
        if 'threshold' in data and isinstance(data['threshold'], dict):
            data['threshold'] = ThresholdConfig(**data['threshold'])
        if 'export' in data and isinstance(data['export'], dict):
            data['export'] = ExportConfig(**data['export'])
        if 'optimization' in data and isinstance(data['optimization'], dict):
            data['optimization'] = OptimizationConfig(**data['optimization'])
        if 'augmentation' in data and isinstance(data['augmentation'], dict):
            data['augmentation'] = AugmentationConfig(**data['augmentation'])
        
        return cls(**data)
    
    @classmethod
    def from_template(cls, template_name: str) -> 'TrainingConfig':
        """从场景模板创建配置"""
        from .default_config import SCENE_TEMPLATES
        
        if template_name not in SCENE_TEMPLATES:
            template_name = 'default'
        
        template = SCENE_TEMPLATES[template_name]
        
        config = cls()
        config.image_size = template.get('image_size', DEFAULT_IMAGE_SIZE)
        config.backbone.name = template.get('backbone', DEFAULT_BACKBONE)
        config.memory_bank.coreset_sampling_ratio = template.get('coreset_ratio', DEFAULT_CORESET_RATIO)
        config.memory_bank.pca_components = template.get('pca_components', DEFAULT_PCA_COMPONENTS)
        config.knn.k = template.get('knn_k', DEFAULT_KNN_K)
        
        return config
    
    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        if not self.dataset_dir:
            errors.append("数据集目录不能为空")
        elif not Path(self.dataset_dir).exists():
            errors.append(f"数据集目录不存在: {self.dataset_dir}")
        
        if self.image_size < 64 or self.image_size > 1024:
            errors.append(f"图像尺寸应在64-1024之间: {self.image_size}")
        
        if self.memory_bank.coreset_sampling_ratio <= 0 or self.memory_bank.coreset_sampling_ratio > 1:
            errors.append(f"CoreSet采样率应在(0, 1]之间: {self.memory_bank.coreset_sampling_ratio}")
        
        if self.knn.k < 1:
            errors.append(f"KNN的K值应>=1: {self.knn.k}")
        
        return errors


def get_backbone_info(backbone_name: str) -> str:
    """获取Backbone信息文本"""
    from .default_config import BACKBONE_OPTIONS
    
    if backbone_name not in BACKBONE_OPTIONS:
        return "未知的Backbone"
    
    info = BACKBONE_OPTIONS[backbone_name]
    return f"""
**{info['name']}**
- 特征维度: {info['total_dim']}-D
- 速度: {info['speed']}
- 精度: {info['accuracy']}
- {info['description']}
""".strip()
