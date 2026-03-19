# -*- coding: utf-8 -*-
"""
OCR 模型导出工具
支持导出为 ONNX 和 TensorRT 格式
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

class OCRModelExporter:
    """OCR模型导出器"""
    
    def __init__(self, model_dir: str):
        """
        初始化导出器
        
        Args:
            model_dir: PaddleOCR模型目录
        """
        self.model_dir = Path(model_dir)
        self.det_model_dir = None
        self.rec_model_dir = None
        self.cls_model_dir = None
        
        self._find_models()
    
    def _find_models(self):
        """查找模型文件"""
        for subdir in self.model_dir.iterdir():
            if subdir.is_dir():
                if 'det' in subdir.name:
                    self.det_model_dir = subdir
                elif 'rec' in subdir.name:
                    self.rec_model_dir = subdir
                elif 'cls' in subdir.name:
                    self.cls_model_dir = subdir
    
    def export_to_onnx(
        self,
        output_dir: str,
        det_input_shape: Tuple[int, int] = (640, 640),
        rec_input_height: int = 48,
        opset_version: int = 14,
    ) -> Dict[str, str]:
        """
        导出为ONNX格式
        
        Args:
            output_dir: 输出目录
            det_input_shape: 检测模型输入尺寸
            rec_input_height: 识别模型输入高度
            opset_version: ONNX opset版本
        
        Returns:
            导出的模型路径字典
        """
        try:
            import paddle
            from paddle.static import InputSpec
            import paddle2onnx
        except ImportError:
            raise ImportError("请安装: pip install paddlepaddle paddle2onnx")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # 导出检测模型
        if self.det_model_dir:
            det_onnx = output_dir / 'det_model.onnx'
            self._export_paddle_to_onnx(
                self.det_model_dir,
                det_onnx,
                input_shape=[1, 3, det_input_shape[0], det_input_shape[1]],
                opset_version=opset_version,
            )
            exported['det'] = str(det_onnx)
            print(f"✅ 检测模型导出成功: {det_onnx}")
        
        # 导出识别模型
        if self.rec_model_dir:
            rec_onnx = output_dir / 'rec_model.onnx'
            self._export_paddle_to_onnx(
                self.rec_model_dir,
                rec_onnx,
                input_shape=[1, 3, rec_input_height, 320],  # 动态宽度
                opset_version=opset_version,
            )
            exported['rec'] = str(rec_onnx)
            print(f"✅ 识别模型导出成功: {rec_onnx}")
        
        # 复制字符字典
        dict_path = self._find_char_dict()
        if dict_path:
            shutil.copy(dict_path, output_dir / 'ppocr_keys_v1.txt')
            exported['char_dict'] = str(output_dir / 'ppocr_keys_v1.txt')
        
        # 保存配置
        config = {
            'det_input_shape': list(det_input_shape),
            'rec_input_height': rec_input_height,
            'opset_version': opset_version,
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        return exported
    
    def _export_paddle_to_onnx(
        self,
        paddle_model_dir: Path,
        onnx_path: Path,
        input_shape: list,
        opset_version: int,
    ):
        """导出Paddle模型为ONNX"""
        import paddle2onnx
        
        model_file = paddle_model_dir / 'inference.pdmodel'
        params_file = paddle_model_dir / 'inference.pdiparams'
        
        if not model_file.exists():
            # 尝试其他命名
            for f in paddle_model_dir.glob('*.pdmodel'):
                model_file = f
                break
            for f in paddle_model_dir.glob('*.pdiparams'):
                params_file = f
                break
        
        paddle2onnx.export(
            str(model_file),
            str(params_file),
            str(onnx_path),
            opset_version=opset_version,
            auto_upgrade_opset=True,
            verbose=False,
            enable_onnx_checker=True,
            enable_experimental_op=True,
            enable_dev_version=True,
        )
    
    def _find_char_dict(self) -> Optional[Path]:
        """查找字符字典"""
        search_paths = [
            self.model_dir / 'ppocr_keys_v1.txt',
            self.rec_model_dir / 'ppocr_keys_v1.txt' if self.rec_model_dir else None,
            Path.home() / '.paddleocr' / 'ppocr_keys_v1.txt',
        ]
        
        for path in search_paths:
            if path and path.exists():
                return path
        
        return None
    
    def export_to_tensorrt(
        self,
        onnx_dir: str,
        output_dir: str,
        fp16: bool = True,
        int8: bool = False,
        workspace_gb: int = 4,
        max_batch_size: int = 8,
    ) -> Dict[str, str]:
        """
        导出为TensorRT格式
        
        Args:
            onnx_dir: ONNX模型目录
            output_dir: 输出目录
            fp16: 使用FP16精度
            int8: 使用INT8精度
            workspace_gb: 工作空间大小(GB)
            max_batch_size: 最大batch大小
        
        Returns:
            导出的引擎路径字典
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("请安装TensorRT")
        
        onnx_dir = Path(onnx_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # 导出检测引擎
        det_onnx = onnx_dir / 'det_model.onnx'
        if det_onnx.exists():
            det_engine = output_dir / 'det_model.engine'
            self._build_trt_engine(
                det_onnx, det_engine,
                fp16=fp16, int8=int8,
                workspace_gb=workspace_gb,
                max_batch_size=max_batch_size,
                dynamic_shapes={
                    'x': [(1, 3, 640, 640), (4, 3, 640, 640), (max_batch_size, 3, 640, 640)]
                }
            )
            exported['det'] = str(det_engine)
            print(f"✅ 检测引擎导出成功: {det_engine}")
        
        # 导出识别引擎
        rec_onnx = onnx_dir / 'rec_model.onnx'
        if rec_onnx.exists():
            rec_engine = output_dir / 'rec_model.engine'
            self._build_trt_engine(
                rec_onnx, rec_engine,
                fp16=fp16, int8=int8,
                workspace_gb=workspace_gb,
                max_batch_size=max_batch_size,
                dynamic_shapes={
                    'x': [(1, 3, 48, 100), (1, 3, 48, 320), (1, 3, 48, 800)]
                }
            )
            exported['rec'] = str(rec_engine)
            print(f"✅ 识别引擎导出成功: {rec_engine}")
        
        # 复制字符字典
        char_dict = onnx_dir / 'ppocr_keys_v1.txt'
        if char_dict.exists():
            shutil.copy(char_dict, output_dir / 'ppocr_keys_v1.txt')
            exported['char_dict'] = str(output_dir / 'ppocr_keys_v1.txt')
        
        return exported
    
    def _build_trt_engine(
        self,
        onnx_path: Path,
        engine_path: Path,
        fp16: bool,
        int8: bool,
        workspace_gb: int,
        max_batch_size: int,
        dynamic_shapes: Dict = None,
    ):
        """构建TensorRT引擎"""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # 解析ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX解析错误: {parser.get_error(i)}")
                raise RuntimeError("ONNX解析失败")
        
        # 配置
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        
        # 动态shape
        if dynamic_shapes:
            profile = builder.create_optimization_profile()
            for name, shapes in dynamic_shapes.items():
                min_shape, opt_shape, max_shape = shapes
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        
        # 构建引擎
        engine = builder.build_serialized_network(network, config)
        
        if engine:
            with open(engine_path, 'wb') as f:
                f.write(engine)
        else:
            raise RuntimeError("TensorRT引擎构建失败")


def download_paddleocr_models(
    det_model: str = 'ch_PP-OCRv4_det',
    rec_model: str = 'ch_PP-OCRv4_rec',
    output_dir: str = './models',
) -> Dict[str, str]:
    """
    下载PaddleOCR预训练模型
    
    Args:
        det_model: 检测模型名称
        rec_model: 识别模型名称
        output_dir: 输出目录
    
    Returns:
        模型路径字典
    """
    from . import MODEL_CONFIGS
    import urllib.request
    import tarfile
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = {}
    
    # 下载检测模型
    if det_model in MODEL_CONFIGS['det']:
        url = MODEL_CONFIGS['det'][det_model]['url']
        det_dir = output_dir / det_model
        if not det_dir.exists():
            print(f"下载检测模型: {det_model}...")
            _download_and_extract(url, output_dir)
        downloaded['det'] = str(det_dir)
    
    # 下载识别模型
    if rec_model in MODEL_CONFIGS['rec']:
        url = MODEL_CONFIGS['rec'][rec_model]['url']
        rec_dir = output_dir / rec_model
        if not rec_dir.exists():
            print(f"下载识别模型: {rec_model}...")
            _download_and_extract(url, output_dir)
        downloaded['rec'] = str(rec_dir)
    
    return downloaded


def _download_and_extract(url: str, output_dir: Path):
    """下载并解压"""
    import urllib.request
    import tarfile
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        with tarfile.open(tmp.name, 'r') as tar:
            tar.extractall(output_dir)
        os.unlink(tmp.name)
