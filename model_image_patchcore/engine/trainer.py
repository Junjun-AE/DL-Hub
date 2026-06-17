# -*- coding: utf-8 -*-
"""
PatchCore 训练引擎

工业级异常检测模型训练器，包含完整的训练、优化和导出流程
"""

import gc
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
import warnings

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


@dataclass
class TrainingCallback:
    """训练回调函数集合"""
    on_phase_start: Optional[Callable[[str, int, int], None]] = None  # phase_name, current, total
    on_phase_end: Optional[Callable[[str, Dict], None]] = None
    on_progress: Optional[Callable[[int, int, str], None]] = None  # current, total, message
    on_log: Optional[Callable[[str], None]] = None
    should_stop: Optional[Callable[[], bool]] = None


@dataclass
class TrainingResult:
    """训练结果"""
    success: bool = True
    message: str = ""
    
    # 时间统计
    total_time_seconds: float = 0
    phase_times: Dict[str, float] = field(default_factory=dict)
    
    # 数据统计
    num_images: int = 0
    num_patches: int = 0
    sampled_patches: int = 0
    
    # Memory Bank 信息
    memory_bank_size: int = 0
    feature_dim: int = 0
    pca_variance_explained: float = 0.0
    
    # 阈值信息
    normalization_params: Dict[str, float] = field(default_factory=dict)
    default_threshold: float = 50.0
    
    # 导出路径
    export_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'message': self.message,
            'total_time_seconds': self.total_time_seconds,
            'phase_times': self.phase_times,
            'num_images': self.num_images,
            'num_patches': self.num_patches,
            'sampled_patches': self.sampled_patches,
            'memory_bank_size': self.memory_bank_size,
            'feature_dim': self.feature_dim,
            'pca_variance_explained': self.pca_variance_explained,
            'normalization_params': self.normalization_params,
            'default_threshold': self.default_threshold,
            'export_path': self.export_path,
        }


class PatchCoreTrainer:
    """
    PatchCore 训练器
    
    完整的训练流程:
    1. 特征提取 (分块)
    2. PCA降维 (增量)
    3. CoreSet采样 (优化)
    4. 构建Faiss索引
    5. 阈值校准
    6. 模型导出
    """
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: TrainingConfig 配置对象
        """
        from config import TrainingConfig
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，请运行: pip install torch")
        
        self.config = config
        self.callback = TrainingCallback()
        
        # 设置设备
        self.device = self._setup_device()
        
        # 训练状态
        self._stop_requested = False
        self.start_time = None
        self.current_phase = ""
        
        # 训练产出
        self.backbone = None
        self.features = None
        self.pca_model = None
        self.memory_bank = None
        self.faiss_index = None
        self.normalization_params = None
        self.threshold_config = None
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self._setup_output_dirs()
    
    def _setup_device(self):
        """设置计算设备"""
        device_str = self.config.device
        
        if device_str == 'auto':
            if torch.cuda.is_available():
                device_str = 'cuda:0'
            else:
                device_str = 'cpu'
        
        return torch.device(device_str)
    
    def _setup_output_dirs(self):
        """创建输出目录结构 - 只创建必要的目录"""
        # 只创建主输出目录和exports目录（实际存放模型的位置）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'exports').mkdir(exist_ok=True)
    
    def set_callback(self, callback: TrainingCallback):
        """设置回调函数"""
        self.callback = callback
    
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        if self.callback.on_log:
            self.callback.on_log(full_message)
        else:
            print(full_message)
    
    def _check_stop(self) -> bool:
        """检查是否请求停止"""
        if self._stop_requested:
            return True
        if self.callback.should_stop and self.callback.should_stop():
            self._stop_requested = True
            return True
        return False
    
    def request_stop(self):
        """请求停止训练"""
        self._stop_requested = True
        self._log("⏹️ 收到停止请求...")
    
    def train(self) -> TrainingResult:
        """
        执行完整训练流程
        
        Returns:
            TrainingResult: 训练结果
        """
        result = TrainingResult()
        self.start_time = time.time()
        
        try:
            # 验证配置
            errors = self.config.validate()
            if errors:
                result.success = False
                result.message = "配置验证失败: " + "; ".join(errors)
                return result
            
            self._log("=" * 50)
            self._log("🚀 开始 PatchCore 训练")
            self._log("=" * 50)
            
            # ========== 阶段1: 初始化 ==========
            phase_start = time.time()
            self.current_phase = "initialization"
            if self.callback.on_phase_start:
                self.callback.on_phase_start("初始化", 1, 6)
            
            self._log("📋 阶段 1/6: 初始化...")
            self._init_backbone()
            dataloader = self._create_dataloader()
            result.num_images = len(dataloader.dataset)
            
            result.phase_times['initialization'] = time.time() - phase_start
            
            if self._check_stop():
                return self._create_stopped_result(result)
            
            # ========== 阶段2: 特征提取 ==========
            phase_start = time.time()
            self.current_phase = "feature_extraction"
            if self.callback.on_phase_start:
                self.callback.on_phase_start("特征提取", 2, 6)
            
            self._log("🔍 阶段 2/6: 特征提取...")
            self.features = self._extract_features(dataloader)
            result.num_patches = len(self.features)
            result.feature_dim = self.features.shape[1]
            self._log(f"   提取完成: {result.num_patches:,} patches, {result.feature_dim}-D")
            
            result.phase_times['feature_extraction'] = time.time() - phase_start
            
            # 清理GPU内存
            self._clear_gpu_memory()
            
            if self._check_stop():
                return self._create_stopped_result(result)
            
            # ========== 阶段3: PCA降维 ==========
            phase_start = time.time()
            self.current_phase = "pca"
            if self.callback.on_phase_start:
                self.callback.on_phase_start("PCA降维", 3, 6)
            
            self._log("📉 阶段 3/6: PCA降维...")
            self.features, self.pca_model, variance_explained = self._apply_pca(self.features)
            result.pca_variance_explained = variance_explained
            result.feature_dim = self.features.shape[1]
            self._log(f"   降维完成: {result.feature_dim}-D, 解释方差: {variance_explained:.2%}")
            
            result.phase_times['pca'] = time.time() - phase_start
            
            if self._check_stop():
                return self._create_stopped_result(result)
            
            # ========== 阶段4: CoreSet采样 ==========
            phase_start = time.time()
            self.current_phase = "coreset"
            if self.callback.on_phase_start:
                self.callback.on_phase_start("CoreSet采样", 4, 6)
            
            self._log("🎯 阶段 4/6: CoreSet采样...")
            self.memory_bank = self._coreset_sampling(self.features)
            result.sampled_patches = len(self.memory_bank)
            result.memory_bank_size = result.sampled_patches
            self._log(f"   采样完成: {result.num_patches:,} → {result.sampled_patches:,} "
                     f"({result.sampled_patches/result.num_patches:.2%})")
            
            # 释放原始特征
            del self.features
            self.features = None
            gc.collect()
            
            result.phase_times['coreset'] = time.time() - phase_start
            
            if self._check_stop():
                return self._create_stopped_result(result)
            
            # ========== 阶段5: 构建索引和阈值 ==========
            phase_start = time.time()
            self.current_phase = "indexing"
            if self.callback.on_phase_start:
                self.callback.on_phase_start("构建索引", 5, 6)
            
            self._log("🔧 阶段 5/6: 构建Faiss索引...")
            self.faiss_index = self._build_faiss_index(self.memory_bank)
            
            self._log("📊 计算归一化参数和阈值...")
            self.normalization_params, self.threshold_config = self._calibrate_threshold(dataloader)
            result.normalization_params = self.normalization_params
            result.default_threshold = self.threshold_config['default']
            
            result.phase_times['indexing'] = time.time() - phase_start
            
            if self._check_stop():
                return self._create_stopped_result(result)
            
            # ========== 阶段6: 导出模型 ==========
            phase_start = time.time()
            self.current_phase = "export"
            if self.callback.on_phase_start:
                self.callback.on_phase_start("模型导出", 6, 6)
            
            if self.config.export.export_enabled:
                self._log("📦 阶段 6/6: 导出模型...")
                export_path = self._export_model()
                result.export_path = str(export_path)
                self._log(f"   导出完成: {export_path}")
            else:
                self._log("⏭️ 阶段 6/6: 跳过导出 (已禁用)")
            
            result.phase_times['export'] = time.time() - phase_start
            
            # ========== 完成 ==========
            result.total_time_seconds = time.time() - self.start_time
            result.success = True
            result.message = "训练成功完成"
            
            self._log("=" * 50)
            self._log(f"✅ 训练完成! 总用时: {result.total_time_seconds:.1f}s")
            self._log(f"   Memory Bank: {result.memory_bank_size:,} 特征")
            self._log(f"   默认阈值: {result.default_threshold:.1f}")
            if result.export_path:
                self._log(f"   模型路径: {result.export_path}")
            self._log("=" * 50)
            
            if self.callback.on_phase_end:
                self.callback.on_phase_end("完成", result.to_dict())
            
        except Exception as e:
            import traceback
            result.success = False
            result.message = f"训练失败: {str(e)}"
            result.total_time_seconds = time.time() - self.start_time
            self._log(f"❌ 训练失败: {e}")
            traceback.print_exc()
        
        return result
    
    def _create_stopped_result(self, result: TrainingResult) -> TrainingResult:
        """创建停止结果"""
        result.success = False
        result.message = "训练被用户停止"
        result.total_time_seconds = time.time() - self.start_time
        self._log("⏹️ 训练已停止")
        return result
    
    def _clear_gpu_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _init_backbone(self):
        """初始化Backbone网络"""
        try:
            # 尝试使用 anomalib
            from anomalib.models.image.patchcore.torch_model import PatchcoreModel
            from anomalib.models.components import TimmFeatureExtractor
            
            self.backbone = TimmFeatureExtractor(
                backbone=self.config.backbone.name,
                layers=self.config.backbone.layers,
                pre_trained=self.config.backbone.pretrained,
            ).to(self.device)
            self.backbone.eval()
            self._log(f"   Backbone: {self.config.backbone.name} (anomalib)")
            self._backbone_type = 'anomalib'
            
        except ImportError:
            # 使用 timm 直接加载
            self._log("   Anomalib不可用，使用timm加载backbone...")
            self._init_backbone_timm()
    
    def _init_backbone_timm(self):
        """使用timm初始化Backbone"""
        try:
            import timm
            
            self.backbone = timm.create_model(
                self.config.backbone.name,
                pretrained=self.config.backbone.pretrained,
                features_only=True,
                out_indices=[2, 3],  # layer2, layer3
            ).to(self.device)
            self.backbone.eval()
            self._log(f"   Backbone: {self.config.backbone.name} (timm)")
            self._backbone_type = 'timm'
            
        except ImportError:
            raise ImportError("需要安装 anomalib 或 timm: pip install anomalib timm")
    
    def _create_dataloader(self) :
        """创建数据加载器"""
        from data.dataset import AnomalyDataset
        
        dataset = AnomalyDataset(
            root_dir=self.config.dataset_dir,
            image_size=self.config.image_size,
            split='train',
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.optimization.feature_batch_size,
            shuffle=False,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
        )
        
        self._log(f"   数据集: {len(dataset)} 张图像")
        
        return dataloader
    
    def _extract_features(self, dataloader) -> np.ndarray:
        """
        分块特征提取
        
        [Bug-8修复] 添加更彻底的GPU内存管理，防止内存泄漏
        """
        all_features = []
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="特征提取")):
                if self._check_stop():
                    break
                
                images = batch['image'].to(self.device)
                
                # 提取特征
                if self._backbone_type == 'anomalib':
                    feature_maps = self.backbone(images)
                    features = self._process_anomalib_features(feature_maps)
                else:
                    feature_maps = self.backbone(images)
                    features = self._process_timm_features(feature_maps)
                
                # 转换为FP16节省内存
                if self.config.optimization.use_fp16_features:
                    features = features.half()
                
                # [Bug-8修复] 立即转移到CPU并释放GPU tensor
                features_np = features.cpu().numpy()
                all_features.append(features_np)
                
                # [Bug-8修复] 显式删除中间变量
                del images, features
                if 'feature_maps' in locals():
                    del feature_maps
                
                # 更新进度
                if self.callback.on_progress:
                    self.callback.on_progress(
                        batch_idx + 1, total_batches,
                        f"批次 {batch_idx + 1}/{total_batches}"
                    )
                
                # [Bug-8修复] 更频繁地清理GPU内存
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # [Bug-8修复] 特征提取完成后的最终清理
        self._clear_gpu_memory()
        
        return np.concatenate(all_features, axis=0).astype(np.float32)
    
    def _clear_gpu_memory(self):
        """
        [Bug-8修复] 彻底清理GPU内存
        """
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 记录当前GPU内存使用情况
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                self._log(f"   GPU内存: 已分配 {allocated:.1f}MB, 已保留 {reserved:.1f}MB")
    
    def _process_anomalib_features(self, feature_maps: Dict):
        """处理anomalib特征"""
        features_list = []
        target_size = None
        
        for layer_name in self.config.backbone.layers:
            feat = feature_maps[layer_name]
            
            if target_size is None:
                target_size = feat.shape[-2:]
            
            # 上采样到相同尺寸
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            features_list.append(feat)
        
        # 拼接通道
        features = torch.cat(features_list, dim=1)  # (B, C, H, W)
        
        # 重排为 (B*H*W, C)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        return features
    
    def _process_timm_features(self, feature_maps: List):
        """处理timm特征"""
        target_size = feature_maps[0].shape[-2:]
        
        features_list = []
        for feat in feature_maps:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            features_list.append(feat)
        
        features = torch.cat(features_list, dim=1)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        return features
    
    def _apply_pca(self, features: np.ndarray) -> Tuple[np.ndarray, Any, float]:
        """应用PCA降维"""
        from config import AUTO_PCA_THRESHOLD_DIM
        
        original_dim = features.shape[1]
        target_dim = self.config.memory_bank.pca_components
        
        # 判断是否需要PCA
        if not self.config.memory_bank.pca_enabled:
            self._log("   PCA已禁用，跳过降维")
            return features, None, 1.0
        
        if original_dim <= target_dim:
            self._log(f"   原始维度({original_dim})<=目标维度({target_dim})，跳过PCA")
            return features, None, 1.0
        
        # 增量PCA
        if self.config.optimization.incremental_pca:
            return self._incremental_pca(features, target_dim)
        else:
            return self._standard_pca(features, target_dim)
    
    def _incremental_pca(self, features: np.ndarray, n_components: int) -> Tuple[np.ndarray, Any, float]:
        """增量PCA，避免大矩阵运算"""
        from sklearn.decomposition import IncrementalPCA
        
        ipca = IncrementalPCA(n_components=n_components)
        batch_size = self.config.optimization.pca_batch_size
        n_batches = (len(features) + batch_size - 1) // batch_size
        
        # 分批拟合
        self._log(f"   增量PCA拟合 ({n_batches} 批次)...")
        for i in range(n_batches):
            if self._check_stop():
                break
            
            start = i * batch_size
            end = min((i + 1) * batch_size, len(features))
            ipca.partial_fit(features[start:end])
            
            # 进度回调 (拟合占50%)
            if self.callback.on_progress:
                self.callback.on_progress(i + 1, n_batches * 2, f"PCA拟合 {i+1}/{n_batches}")
        
        # 分批转换
        transformed = []
        for i in range(n_batches):
            if self._check_stop():
                break
            
            start = i * batch_size
            end = min((i + 1) * batch_size, len(features))
            transformed.append(ipca.transform(features[start:end]))
            
            # 进度回调 (转换占50%)
            if self.callback.on_progress:
                self.callback.on_progress(n_batches + i + 1, n_batches * 2, f"PCA转换 {i+1}/{n_batches}")
        
        variance_explained = float(np.sum(ipca.explained_variance_ratio_))
        
        return np.concatenate(transformed, axis=0), ipca, variance_explained
    
    def _standard_pca(self, features: np.ndarray, n_components: int) -> Tuple[np.ndarray, Any, float]:
        """标准PCA"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(features)
        variance_explained = float(np.sum(pca.explained_variance_ratio_))
        
        return transformed, pca, variance_explained
    
    def _coreset_sampling(self, features: np.ndarray) -> np.ndarray:
        """CoreSet采样"""
        n_samples = int(len(features) * self.config.memory_bank.coreset_sampling_ratio)
        n_samples = max(n_samples, 100)  # 至少100个样本
        
        if len(features) <= n_samples:
            self._log(f"   特征数量({len(features)})<=目标采样数({n_samples})，跳过采样")
            return features
        
        # 使用优化的k-Center贪心算法
        if self.config.optimization.random_projection_enabled:
            selected_indices = self._kcenter_greedy_optimized(features, n_samples)
        else:
            selected_indices = self._kcenter_greedy(features, n_samples)
        
        return features[selected_indices]
    
    def _kcenter_greedy_optimized(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """优化的k-Center贪心算法，使用随机投影加速"""
        from sklearn.random_projection import GaussianRandomProjection
        
        # 随机投影降维加速距离计算
        proj_dim = min(self.config.optimization.random_projection_dim, features.shape[1])
        rp = GaussianRandomProjection(n_components=proj_dim, random_state=42)
        projected = rp.fit_transform(features)
        
        return self._kcenter_greedy(projected, n_samples, original_n=len(features))
    
    def _kcenter_greedy(self, features: np.ndarray, n_samples: int, original_n: int = None) -> np.ndarray:
        """k-Center贪心算法"""
        try:
            import faiss
            use_faiss = True
        except ImportError:
            use_faiss = False
            self._log("   警告: Faiss不可用，使用numpy计算距离（较慢）")
        
        n = len(features)
        selected = [np.random.randint(n)]
        min_distances = np.full(n, np.inf)
        
        max_iter = min(n_samples, self.config.memory_bank.coreset_max_iter)
        
        for i in range(1, n_samples):
            if self._check_stop():
                break
            
            # 更新进度
            if self.callback.on_progress and i % 10 == 0:
                self.callback.on_progress(i, n_samples, f"采样 {i}/{n_samples}")
            
            if i < max_iter:
                # 更新最小距离
                last_selected = features[selected[-1:]]
                
                if use_faiss:
                    # 使用Faiss加速
                    index = faiss.IndexFlatL2(features.shape[1])
                    index.add(last_selected.astype(np.float32))
                    distances, _ = index.search(features.astype(np.float32), 1)
                    distances = distances.flatten()
                else:
                    # Numpy计算
                    distances = np.linalg.norm(features - last_selected, axis=1)
                
                min_distances = np.minimum(min_distances, distances)
                
                # 选择最远点
                farthest = np.argmax(min_distances)
                selected.append(farthest)
            else:
                # 随机补充剩余点
                remaining = n_samples - len(selected)
                candidates = list(set(range(n)) - set(selected))
                selected.extend(np.random.choice(candidates, remaining, replace=False).tolist())
                break
        
        # 完成进度
        if self.callback.on_progress:
            self.callback.on_progress(n_samples, n_samples, "采样完成")
        
        return np.array(selected)
    
    def _build_faiss_index(self, features: np.ndarray):
        """构建Faiss索引"""
        try:
            import faiss
        except ImportError:
            raise ImportError("需要安装 faiss: pip install faiss-cpu 或 faiss-gpu")
        
        features = features.astype(np.float32)
        dim = features.shape[1]
        n = len(features)
        
        # 自动选择索引类型
        index_type = self.config.knn.index_type
        from config import FAISS_AUTO_THRESHOLDS
        
        if index_type == 'auto':
            if n < FAISS_AUTO_THRESHOLDS['flat_max']:
                index_type = 'Flat'
            elif n < FAISS_AUTO_THRESHOLDS['ivf_max']:
                index_type = 'IVFFlat'
            else:
                index_type = 'IVFPQ'
        
        self._log(f"   索引类型: {index_type} (n={n:,})")
        
        if index_type == 'Flat':
            index = faiss.IndexFlatL2(dim)
        elif index_type == 'IVFFlat':
            nlist = min(int(np.sqrt(n)), self.config.knn.nlist)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(features)
            index.nprobe = self.config.knn.nprobe
        elif index_type == 'IVFPQ':
            nlist = int(np.sqrt(n))
            m = min(dim // 4, self.config.knn.pq_m)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, self.config.knn.pq_bits)
            index.train(features)
            index.nprobe = self.config.knn.nprobe
        else:
            raise ValueError(f"未知的索引类型: {index_type}")
        
        # 添加特征
        index.add(features)
        
        # 尝试转到GPU
        if self.config.export.faiss_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                self._log("   索引已转移到GPU")
            except Exception as e:
                self._log(f"   警告: 无法使用GPU索引: {e}")
        
        return index
    
    def _calibrate_threshold(self, dataloader) -> Tuple[Dict, Dict]:
        """校准阈值"""
        from evaluation.threshold import IndustrialThresholdCalibrator
        
        # 计算所有训练样本的分数
        scores = self._compute_scores(dataloader)
        
        # 校准
        calibrator = IndustrialThresholdCalibrator()
        calibration_result = calibrator.calibrate(scores)
        
        return calibration_result['normalization'], calibration_result['thresholds']
    
    def _compute_scores(self, dataloader) -> np.ndarray:
        """计算训练集分数用于校准"""
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="计算分数"):
                if self._check_stop():
                    break
                
                images = batch['image'].to(self.device)
                
                # 提取特征
                if self._backbone_type == 'anomalib':
                    feature_maps = self.backbone(images)
                    features = self._process_anomalib_features(feature_maps)
                else:
                    feature_maps = self.backbone(images)
                    features = self._process_timm_features(feature_maps)
                
                features = features.cpu().numpy().astype(np.float32)
                
                # PCA变换
                if self.pca_model is not None:
                    features = self.pca_model.transform(features)
                
                # KNN距离
                distances, _ = self.faiss_index.search(features, self.config.knn.k)
                
                # 取每个patch的最近邻距离的最大值
                patch_scores = distances[:, 0]  # 最近邻距离
                
                # 重排为图像级分数
                B = images.shape[0]
                H = W = self.config.image_size // 8
                patch_scores = patch_scores.reshape(B, H * W)
                image_scores = np.max(patch_scores, axis=1)  # 图像级分数
                
                all_scores.extend(image_scores.tolist())
        
        return np.array(all_scores)
    
    def _export_model(self) -> Path:
        """导出模型"""
        from export.exporter import PatchCoreExporter
        
        exporter = PatchCoreExporter(
            config=self.config,
            backbone=self.backbone,
            pca_model=self.pca_model,
            memory_bank=self.memory_bank,
            faiss_index=self.faiss_index,
            normalization_params=self.normalization_params,
            threshold_config=self.threshold_config,
        )
        
        export_path = exporter.export(
            output_dir=self.output_dir,
            progress_callback=self.callback.on_progress,
        )
        
        return export_path
