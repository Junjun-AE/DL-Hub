# -*- coding: utf-8 -*-
"""
PatchCore 模型导出器

支持导出为:
1. .pkg 单一打包文件 (推荐)
2. ONNX 格式
3. TensorRT 引擎
"""

import os
import json
import shutil
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import tempfile

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PatchCoreExporter:
    """PatchCore 模型导出器"""
    
    def __init__(
        self,
        config,
        backbone,
        pca_model,
        memory_bank: np.ndarray,
        faiss_index,
        normalization_params: Dict,
        threshold_config: Dict,
    ):
        """
        初始化导出器
        
        Args:
            config: 训练配置
            backbone: Backbone网络
            pca_model: PCA模型 (可为None)
            memory_bank: Memory Bank特征
            faiss_index: Faiss索引
            normalization_params: 归一化参数
            threshold_config: 阈值配置
        """
        self.config = config
        self.backbone = backbone
        self.pca_model = pca_model
        self.memory_bank = memory_bank
        self.faiss_index = faiss_index
        self.normalization_params = normalization_params
        self.threshold_config = threshold_config
    
    def export(
        self,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """
        导出模型
        
        Args:
            output_dir: 输出目录
            progress_callback: 进度回调
        
        Returns:
            导出的模型路径
        """
        output_dir = Path(output_dir)
        exports_dir = output_dir / 'exports'
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            model_dir = temp_dir / 'patchcore_model'
            model_dir.mkdir()
            
            # 创建子目录
            (model_dir / 'backbone').mkdir()
            (model_dir / 'memory_bank').mkdir()
            (model_dir / 'normalization').mkdir()
            (model_dir / 'threshold').mkdir()
            (model_dir / 'metadata').mkdir()
            
            if progress_callback:
                progress_callback(1, 6, "保存配置...")
            
            # 1. 保存配置
            self._save_config(model_dir)
            
            if progress_callback:
                progress_callback(2, 6, "导出Backbone...")
            
            # 2. 导出Backbone
            self._export_backbone(model_dir / 'backbone')
            
            if progress_callback:
                progress_callback(3, 6, "保存Memory Bank...")
            
            # 3. 保存Memory Bank
            self._save_memory_bank(model_dir / 'memory_bank')
            
            if progress_callback:
                progress_callback(4, 6, "保存归一化参数...")
            
            # 4. 保存归一化参数
            self._save_normalization(model_dir / 'normalization')
            
            if progress_callback:
                progress_callback(5, 6, "保存阈值配置...")
            
            # 5. 保存阈值配置
            self._save_threshold(model_dir / 'threshold')
            
            if progress_callback:
                progress_callback(6, 6, "打包模型...")
            
            # 6. 保存元数据
            self._save_metadata(model_dir / 'metadata')
            
            # 7. 创建manifest
            self._create_manifest(model_dir)
            
            # 8. 打包
            if self.config.export.package_enabled:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pkg_name = f"patchcore_{timestamp}.pkg"
                pkg_path = exports_dir / pkg_name
                
                self._create_package(model_dir, pkg_path)
                
                return pkg_path
            else:
                # 复制到输出目录
                final_dir = exports_dir / 'patchcore_model'
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(model_dir, final_dir)
                
                return final_dir
    
    def _save_config(self, model_dir: Path):
        """保存完整配置"""
        from config import BACKBONE_OPTIONS
        
        backbone_info = BACKBONE_OPTIONS.get(self.config.backbone.name, {})
        
        config_dict = {
            'model_info': {
                'name': 'PatchCore',
                'version': '1.0.0',
                'export_date': datetime.now().isoformat(),
                'export_tool_version': '1.0.0',
            },
            'backbone': {
                'name': self.config.backbone.name,
                'layers': self.config.backbone.layers,
                'feature_dim': backbone_info.get('total_dim', 1536),
                'pretrained': 'imagenet',
            },
            'preprocessing': {
                'input_size': [self.config.image_size, self.config.image_size],
                'resize_mode': 'bilinear',
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                },
                'input_format': 'RGB',
                'input_dtype': 'float32',
                'input_range': [0.0, 1.0],
            },
            'feature_extraction': {
                'pca': {
                    'enabled': self.pca_model is not None,
                    'n_components': self.config.memory_bank.pca_components if self.pca_model else 0,
                },
            },
            'memory_bank': {
                'sampling_method': 'coreset',
                'sampling_ratio': self.config.memory_bank.coreset_sampling_ratio,
                'size': len(self.memory_bank),
                'feature_dtype': 'float16',
            },
            'knn': {
                'k': self.config.knn.k,
                'index_type': self.config.knn.index_type,
                'metric': 'L2',
            },
            'postprocessing': {
                'upsample_mode': self.config.postprocess.upsample_mode,
                'gaussian_blur': {
                    'enabled': self.config.postprocess.gaussian_blur_enabled,
                    'sigma': self.config.postprocess.gaussian_sigma,
                },
                'score_aggregation': self.config.postprocess.score_aggregation,
            },
            'normalization': self.normalization_params,
            'thresholds': self.threshold_config,
        }
        
        with open(model_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def _export_backbone(self, backbone_dir: Path):
        """导出Backbone网络"""
        if not TORCH_AVAILABLE:
            return
        
        self.backbone.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(
            1, 3, self.config.image_size, self.config.image_size,
            device=next(self.backbone.parameters()).device
        )
        
        # 导出ONNX
        if 'onnx' in self.config.export.export_formats or self.config.export.tensorrt_enabled:
            onnx_path = backbone_dir / 'backbone.onnx'
            
            try:
                # 获取输出名称
                with torch.no_grad():
                    output = self.backbone(dummy_input)
                
                if isinstance(output, dict):
                    output_names = list(output.keys())
                else:
                    output_names = [f'feature_{i}' for i in range(len(output))]
                
                torch.onnx.export(
                    self.backbone,
                    dummy_input,
                    str(onnx_path),
                    input_names=['input'],
                    output_names=output_names,
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        **{name: {0: 'batch_size'} for name in output_names}
                    },
                    opset_version=14,
                )
                
                print(f"   ONNX导出成功: {onnx_path}")
                
            except Exception as e:
                print(f"   ONNX导出失败: {e}")
        
        # 导出TensorRT (如果可用)
        if self.config.export.tensorrt_enabled:
            self._export_tensorrt(backbone_dir, backbone_dir / 'backbone.onnx')
    
    def _export_tensorrt(self, backbone_dir: Path, onnx_path: Path):
        """导出TensorRT引擎"""
        try:
            import tensorrt as trt
            TRT_AVAILABLE = True
        except ImportError:
            TRT_AVAILABLE = False
            print("   TensorRT不可用，跳过引擎导出")
            return
        
        if not onnx_path.exists():
            print("   ONNX文件不存在，跳过TensorRT导出")
            return
        
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            # 解析ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"   ONNX解析错误: {parser.get_error(i)}")
                    return
            
            # 配置
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 
                                        int(self.config.export.tensorrt_workspace_gb * (1 << 30)))
            
            # 设置精度
            precision = self.config.export.tensorrt_precision
            if precision == 'fp16' and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8' and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # TODO: 添加INT8校准器
            
            # 动态batch
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            
            min_shape = (1, 3, self.config.image_size, self.config.image_size)
            opt_shape = (4, 3, self.config.image_size, self.config.image_size)
            max_shape = (self.config.export.tensorrt_max_batch_size, 3, 
                        self.config.image_size, self.config.image_size)
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # 构建引擎
            engine = builder.build_serialized_network(network, config)
            
            if engine:
                engine_path = backbone_dir / f'backbone_{precision}.engine'
                with open(engine_path, 'wb') as f:
                    f.write(engine)
                print(f"   TensorRT引擎导出成功: {engine_path}")
            else:
                print("   TensorRT引擎构建失败")
                
        except Exception as e:
            print(f"   TensorRT导出失败: {e}")
    
    def _save_memory_bank(self, memory_bank_dir: Path):
        """保存Memory Bank"""
        # 保存特征
        features = self.memory_bank.astype(np.float16)
        np.save(memory_bank_dir / 'features.npy', features)
        
        # 保存Faiss索引
        try:
            import faiss
            
            # 如果是GPU索引，先转到CPU
            if hasattr(self.faiss_index, 'index'):
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            else:
                cpu_index = self.faiss_index
            
            faiss.write_index(cpu_index, str(memory_bank_dir / 'faiss_index.bin'))
            
        except Exception as e:
            print(f"   Faiss索引保存失败: {e}")
        
        # 保存PCA模型
        if self.pca_model is not None:
            pca_data = {
                'components': self.pca_model.components_.astype(np.float16),
                'mean': self.pca_model.mean_.astype(np.float32),
                'explained_variance_ratio': self.pca_model.explained_variance_ratio_.astype(np.float32),
            }
            np.savez(memory_bank_dir / 'pca_model.npz', **pca_data)
    
    def _save_normalization(self, normalization_dir: Path):
        """保存归一化参数"""
        with open(normalization_dir / 'params.json', 'w') as f:
            json.dump(self.normalization_params, f, indent=2)
    
    def _save_threshold(self, threshold_dir: Path):
        """保存阈值配置"""
        with open(threshold_dir / 'config.json', 'w') as f:
            json.dump(self.threshold_config, f, indent=2)
    
    def _save_metadata(self, metadata_dir: Path):
        """保存元数据"""
        metadata = {
            'export_date': datetime.now().isoformat(),
            'training_config': {
                'backbone': self.config.backbone.name,
                'image_size': self.config.image_size,
                'coreset_ratio': self.config.memory_bank.coreset_sampling_ratio,
                'pca_components': self.config.memory_bank.pca_components,
                'knn_k': self.config.knn.k,
            },
            'memory_bank_info': {
                'size': len(self.memory_bank),
                'feature_dim': self.memory_bank.shape[1],
            },
        }
        
        with open(metadata_dir / 'info.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_manifest(self, model_dir: Path):
        """创建文件清单"""
        manifest = {
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'files': {},
        }
        
        # 计算所有文件的校验和
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(model_dir)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                manifest['files'][str(rel_path)] = {
                    'size': file_path.stat().st_size,
                    'md5': file_hash,
                }
        
        with open(model_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _create_package(self, model_dir: Path, pkg_path: Path):
        """创建打包文件"""
        with zipfile.ZipFile(pkg_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_dir)
                    zf.write(file_path, arcname)
        
        print(f"   模型包大小: {pkg_path.stat().st_size / (1024*1024):.2f} MB")


def load_patchcore_model(model_path: str) -> Dict[str, Any]:
    """
    加载PatchCore模型
    
    Args:
        model_path: 模型路径 (.pkg文件或目录)
    
    Returns:
        模型组件字典
    """
    model_path = Path(model_path)
    
    # 如果是pkg文件，解压到临时目录
    if model_path.suffix == '.pkg':
        import tempfile
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(temp_dir)
        model_dir = Path(temp_dir)
    else:
        model_dir = model_path
    
    # 加载配置
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # 加载Memory Bank
    features = np.load(model_dir / 'memory_bank' / 'features.npy')
    
    # 加载Faiss索引
    try:
        import faiss
        faiss_index = faiss.read_index(str(model_dir / 'memory_bank' / 'faiss_index.bin'))
    except Exception:
        faiss_index = None
    
    # 加载PCA模型
    pca_path = model_dir / 'memory_bank' / 'pca_model.npz'
    if pca_path.exists():
        pca_data = np.load(pca_path)
        pca_model = {
            'components': pca_data['components'],
            'mean': pca_data['mean'],
        }
    else:
        pca_model = None
    
    # 加载归一化参数
    with open(model_dir / 'normalization' / 'params.json', 'r') as f:
        normalization = json.load(f)
    
    # 加载阈值配置
    with open(model_dir / 'threshold' / 'config.json', 'r') as f:
        thresholds = json.load(f)
    
    return {
        'config': config,
        'features': features,
        'faiss_index': faiss_index,
        'pca_model': pca_model,
        'normalization': normalization,
        'thresholds': thresholds,
        'model_dir': model_dir,
    }
