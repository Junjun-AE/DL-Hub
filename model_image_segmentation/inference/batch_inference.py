"""
批量推理模块 - 支持滑动窗口推理和结果可视化
"""

import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def cv2_imread_chinese(image_path: str) -> np.ndarray:
    """
    支持中文路径的图像读取
    
    Args:
        image_path: 图像路径（支持中文）
    
    Returns:
        BGR格式的图像数组
    """
    import cv2
    # 尝试直接读取
    img = cv2.imread(image_path)
    if img is not None:
        return img
    
    # 如果失败，使用numpy从文件读取
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def cv2_imwrite_chinese(image_path: str, img: np.ndarray) -> bool:
    """
    支持中文路径的图像保存
    
    Args:
        image_path: 保存路径（支持中文）
        img: BGR格式的图像数组
    
    Returns:
        是否保存成功
    """
    import cv2
    try:
        # 尝试直接保存
        success = cv2.imwrite(image_path, img)
        if success:
            return True
    except Exception:
        pass
    
    # 如果失败，使用numpy保存
    try:
        ext = os.path.splitext(image_path)[1]
        _, img_encoded = cv2.imencode(ext, img)
        img_encoded.tofile(image_path)
        return True
    except Exception:
        return False


def pil_open_chinese(image_path: str) -> 'Image.Image':
    """
    支持中文路径的PIL图像读取
    
    Args:
        image_path: 图像路径（支持中文）
    
    Returns:
        PIL Image对象
    """
    # 方法1: 直接打开
    try:
        return Image.open(image_path)
    except Exception:
        pass
    
    # 方法2: 使用二进制方式读取
    try:
        with open(image_path, 'rb') as f:
            return Image.open(f).copy()
    except Exception:
        pass
    
    # 方法3: 使用numpy和cv2
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        import cv2
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = img_bgr[:, :, ::-1]
            return Image.fromarray(img_rgb)
    except Exception:
        pass
    
    raise IOError(f"无法读取图像: {image_path}")


def find_latest_model(output_root: str = 'output', model_type: str = 'best') -> Optional[str]:
    """
    自动查找最近训练的模型
    
    Args:
        output_root: 输出目录根路径（默认'output'）
        model_type: 模型类型
            - 'best': 最佳模型 (best_model.pth)
            - 'last': 最后模型 (last_model.pth)
            - 'epoch_N': 指定epoch的模型 (epoch_N.pth)
    
    Returns:
        模型路径，未找到返回None
    
    Example:
        >>> model_path = find_latest_model()  # 自动找最新的best_model.pth
        >>> model_path = find_latest_model(model_type='last')  # 最新的last_model.pth
        >>> model_path = find_latest_model(model_type='epoch_50')  # 最新的epoch_50.pth
    """
    output_root = Path(output_root)
    if not output_root.exists():
        print(f"⚠️ 输出目录不存在: {output_root}")
        return None
    
    # 确定模型文件名
    if model_type == 'best':
        model_filename = 'best_model.pth'
    elif model_type == 'last':
        model_filename = 'last_model.pth'
    elif model_type.startswith('epoch_'):
        model_filename = f'{model_type}.pth'
    else:
        model_filename = f'{model_type}.pth'
    
    # 查找所有训练目录（按时间排序）
    train_dirs = []
    for d in output_root.iterdir():
        if d.is_dir() and d.name.startswith('SegFormer_'):
            weights_dir = d / 'weights'
            model_path = weights_dir / model_filename
            if model_path.exists():
                # 获取目录的修改时间
                mtime = d.stat().st_mtime
                train_dirs.append((mtime, model_path))
    
    if not train_dirs:
        print(f"⚠️ 未找到任何训练模型")
        print(f"   查找路径: {output_root}/SegFormer_*/weights/{model_filename}")
        return None
    
    # 按时间排序，返回最新的
    train_dirs.sort(reverse=True)
    latest_model = train_dirs[0][1]
    
    print(f"🔍 自动找到最新模型: {latest_model}")
    return str(latest_model)


def list_available_models(output_root: str = 'output') -> List[Dict[str, Any]]:
    """
    列出所有可用的训练模型
    
    Args:
        output_root: 输出目录根路径
    
    Returns:
        模型信息列表，每个元素包含:
        - path: 模型路径
        - train_dir: 训练目录名
        - model_type: 模型类型 (best/last/epoch_N)
        - created_time: 创建时间
        - metadata: 元数据（如果有）
    """
    output_root = Path(output_root)
    if not output_root.exists():
        return []
    
    models = []
    for d in output_root.iterdir():
        if d.is_dir() and d.name.startswith('SegFormer_'):
            weights_dir = d / 'weights'
            if not weights_dir.exists():
                continue
            
            # 查找该目录下所有模型
            for model_file in weights_dir.glob('*.pth'):
                model_info = {
                    'path': str(model_file),
                    'train_dir': d.name,
                    'model_type': model_file.stem,
                    'created_time': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                }
                
                # 尝试读取元数据
                try:
                    ckpt = torch.load(model_file, map_location='cpu')
                    if 'num_classes' in ckpt:
                        model_info['metadata'] = {
                            'num_classes': ckpt.get('num_classes'),
                            'class_names': ckpt.get('class_names'),
                            'img_size': ckpt.get('img_size'),
                            'best_mIoU': ckpt.get('metrics', {}).get('best_mIoU'),
                        }
                except Exception:
                    pass
                
                models.append(model_info)
    
    # 按创建时间排序
    models.sort(key=lambda x: x['created_time'], reverse=True)
    return models


def print_available_models(output_root: str = 'output'):
    """打印所有可用模型的信息"""
    from datetime import datetime
    
    models = list_available_models(output_root)
    if not models:
        print(f"⚠️ 未找到任何训练模型")
        print(f"   请先运行训练，模型将保存在 {output_root}/SegFormer_*/weights/ 目录下")
        return
    
    print(f"\n{'='*70}")
    print(f"📦 可用模型列表 (共 {len(models)} 个)")
    print(f"{'='*70}")
    
    current_train_dir = None
    for model in models:
        # 打印训练目录分隔
        if model['train_dir'] != current_train_dir:
            current_train_dir = model['train_dir']
            print(f"\n📁 {current_train_dir}")
        
        # 打印模型信息
        model_type = model['model_type']
        path = model['path']
        
        # 元数据信息
        meta_str = ""
        if 'metadata' in model:
            meta = model['metadata']
            if meta.get('best_mIoU'):
                meta_str = f" | mIoU={meta['best_mIoU']:.4f}"
            if meta.get('num_classes'):
                meta_str += f" | {meta['num_classes']}类"
        
        print(f"   ├── {model_type}.pth{meta_str}")
    
    print(f"\n{'='*70}")
    print(f"💡 使用方法:")
    print(f"   model_path = find_latest_model()  # 自动加载最新best模型")
    print(f"   model_path = find_latest_model(model_type='last')  # 最新last模型")
    print(f"{'='*70}\n")


@dataclass
class InferenceResult:
    """推理结果"""
    image_path: str
    pred_mask: np.ndarray  # (H, W) 预测mask
    class_counts: Dict[int, int]  # 每个类别的像素数
    inference_time: float  # 推理耗时（秒）


@dataclass
class BatchInferenceResult:
    """批量推理结果"""
    success: bool
    message: str
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    total_time: float = 0.0
    output_dir: str = ""


class SegFormerInference:
    """
    SegFormer推理器 - 基于MMSegmentation
    
    ================================================================
    预处理策略：方案A - 直接Resize（工业标准）
    ================================================================
    
    【预处理流程】：
      1. cv2.imread() → BGR (H, W, 3)
      2. cv2.resize(img_size, img_size) → BGR (img_size, img_size, 3)
      3. BGR → RGB
      4. (pixel - mean) / std → float32
         mean = [123.675, 116.28, 103.53]
         std  = [58.395, 57.12, 57.375]
      5. HWC → CHW → (1, 3, img_size, img_size)
    
    【后处理流程】：
      1. 模型输出 → (img_size, img_size) 预测
      2. cv2.resize(pred, (orig_w, orig_h)) → 回原图尺寸
    
    【TensorRT部署代码】：
    ```python
    import cv2
    import numpy as np
    
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    IMG_SIZE = 512  # 从配置读取
    
    def preprocess(image_path):
        img = cv2.imread(image_path)
        orig_h, orig_w = img.shape[:2]
        
        # 直接resize到固定尺寸
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # 归一化
        img = img[:, :, ::-1].astype(np.float32)  # BGR→RGB
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # HWC→NCHW
        
        return img, (orig_h, orig_w)
    
    def postprocess(pred, orig_size):
        # pred shape: (img_size, img_size)
        pred = cv2.resize(pred.astype(np.uint8), (orig_size[1], orig_size[0]),
                         interpolation=cv2.INTER_NEAREST)
        return pred
    ```
    
    ================================================================
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0',
        use_sliding_window: bool = False,
        window_size: int = 512,
        stride: int = 256,
        conf_threshold: Optional[float] = None,
        auto_load: bool = True,
        model_type: str = 'best',
        output_root: str = 'output',
    ):
        """
        初始化推理器
        
        Args:
            checkpoint_path: 模型checkpoint路径
                - 指定路径: 使用指定的模型文件
                - None + auto_load=True: 自动查找最新模型
            device: 推理设备 ('cuda:0', 'cpu', etc.)
            use_sliding_window: 是否对大图像使用滑动窗口（默认False）
            window_size: 滑动窗口大小
            stride: 滑动步长
            conf_threshold: 置信度阈值（None表示使用argmax）
            auto_load: 是否自动加载最新模型（当checkpoint_path=None时）
            model_type: 自动加载时的模型类型 ('best', 'last', 'epoch_N')
            output_root: 自动查找时的输出目录根路径
        
        Example:
            # 方式1: 手动指定模型路径
            >>> inferencer = SegFormerInference(checkpoint_path='output/.../best_model.pth')
            
            # 方式2: 自动加载最新的best模型
            >>> inferencer = SegFormerInference()
            
            # 方式3: 自动加载最新的last模型
            >>> inferencer = SegFormerInference(model_type='last')
            
            # 方式4: 自动加载最新的epoch_50模型
            >>> inferencer = SegFormerInference(model_type='epoch_50')
        """
        if torch is None:
            raise ImportError("需要安装PyTorch: pip install torch")
        
        # 自动查找模型
        if checkpoint_path is None:
            if auto_load:
                checkpoint_path = find_latest_model(output_root, model_type)
                if checkpoint_path is None:
                    raise FileNotFoundError(
                        f"❌ 未找到任何训练模型\n"
                        f"   请先运行训练，或手动指定 checkpoint_path\n"
                        f"   查找路径: {output_root}/SegFormer_*/weights/{model_type}_model.pth"
                    )
            else:
                raise ValueError("❌ 请指定 checkpoint_path 或设置 auto_load=True")
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        
        # 模型信息
        self.model = None
        self.cfg = None
        self.num_classes = 0
        self.class_names = []
        self.img_size = 512
        self.metadata = {}  # 存储模型元数据
        
        # 加载模型
        self._load_model()
        
        # 验证模型已加载
        if self.model is None:
            raise RuntimeError("❌ 模型加载失败")
    
    def _load_model(self):
        """加载MMSeg模型"""
        try:
            from mmseg.apis import init_model
            from mmengine.config import Config
        except ImportError:
            raise ImportError("❌ MMSegmentation未正确安装")
        
        # ============================================
        # 1. 先尝试从模型文件读取元数据
        # ============================================
        try:
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')
            if 'num_classes' in ckpt:
                self.metadata = {
                    'num_classes': ckpt.get('num_classes'),
                    'class_names': ckpt.get('class_names', []),
                    'img_size': ckpt.get('img_size', 512),
                    'input_spec': ckpt.get('input_spec', {}),
                    'preprocess': ckpt.get('preprocess', 'resize'),
                    'metrics': ckpt.get('metrics', {}),
                    'save_type': ckpt.get('save_type', 'unknown'),
                }
                print(f"📦 从模型文件读取元数据:")
                print(f"   保存类型: {self.metadata['save_type']}")
                if self.metadata['metrics'].get('best_mIoU'):
                    print(f"   最佳mIoU: {self.metadata['metrics']['best_mIoU']:.4f}")
        except Exception as e:
            print(f"⚠️ 无法读取模型元数据: {e}")
            self.metadata = {}
        
        # ============================================
        # 2. 查找配置文件
        # ============================================
        config_path = None
        for possible_path in [
            self.checkpoint_path.parent.parent / 'config.py',
            self.checkpoint_path.parent / 'config.py',
        ]:
            if possible_path.exists():
                config_path = possible_path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                f"❌ 未找到配置文件\n"
                f"   请确保config.py位于以下位置之一:\n"
                f"   - {self.checkpoint_path.parent.parent / 'config.py'}\n"
                f"   - {self.checkpoint_path.parent / 'config.py'}"
            )
        
        print(f"📝 加载配置: {config_path}")
        
        # 加载配置
        self.cfg = Config.fromfile(str(config_path))
        
        # 验证test_pipeline存在
        if not hasattr(self.cfg, 'test_pipeline') or self.cfg.test_pipeline is None:
            raise ValueError(
                "❌ 配置文件缺少test_pipeline\n"
                "   请使用新版本的训练器重新训练模型"
            )
        
        # 打印预处理流程（用于调试验证）
        print("📋 推理预处理流程（方案A：直接Resize）:")
        for i, step in enumerate(self.cfg.test_pipeline):
            step_type = step.get('type', 'Unknown')
            if step_type == 'Resize':
                scale = step.get('scale', 'N/A')
                keep_ratio = step.get('keep_ratio', False)
                print(f"   {i+1}. {step_type}(scale={scale}, keep_ratio={keep_ratio})")
            elif step_type == 'Pad':
                size = step.get('size', 'N/A')
                print(f"   {i+1}. {step_type}(size={size})")
            else:
                print(f"   {i+1}. {step_type}")
        
        # 加载模型
        self.model = init_model(self.cfg, str(self.checkpoint_path), device=self.device)
        self.model.eval()
        
        # 获取类别信息
        if hasattr(self.cfg, 'model') and 'decode_head' in self.cfg.model:
            self.num_classes = self.cfg.model.decode_head.get('num_classes', 2)
        
        # 获取类别名称
        if hasattr(self.cfg, 'train_dataloader'):
            dataset_cfg = self.cfg.train_dataloader.get('dataset', {})
            metainfo = dataset_cfg.get('metainfo', {})
            self.class_names = metainfo.get('classes', [])
        
        if not self.class_names:
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        
        # 获取图像尺寸（从test_pipeline的Resize或Pad步骤）
        self.img_size = 512  # 默认值
        for step in self.cfg.test_pipeline:
            if step.get('type') == 'Pad':
                size = step.get('size', (512, 512))
                if isinstance(size, (list, tuple)):
                    self.img_size = size[0]
                else:
                    self.img_size = size
                break
            elif step.get('type') == 'Resize':
                scale = step.get('scale', (512, 512))
                if isinstance(scale, (list, tuple)):
                    self.img_size = scale[0]
                else:
                    self.img_size = scale
        
        # 更新滑动窗口大小
        self.window_size = self.img_size
        
        # ============================================
        # 5. 优先使用模型内嵌元数据覆盖配置信息
        # ============================================
        if self.metadata:
            if self.metadata.get('num_classes'):
                self.num_classes = self.metadata['num_classes']
            if self.metadata.get('class_names'):
                self.class_names = self.metadata['class_names']
            if self.metadata.get('img_size'):
                self.img_size = self.metadata['img_size']
                self.window_size = self.img_size
        
        # 打印模型信息
        print(f"✅ 模型加载成功")
        print(f"   模型路径: {self.checkpoint_path}")
        print(f"   类别数: {self.num_classes}")
        print(f"   类别名: {self.class_names}")
        print(f"   输入尺寸: {self.img_size}×{self.img_size}")
        print(f"   归一化: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]")
        print(f"   预处理: Resize(keep_ratio=False) - 直接缩放到固定尺寸")
    
    def predict_single(self, image_path: str) -> np.ndarray:
        """
        使用MMSeg API推理单张图像
        
        预处理流程（方案A：直接Resize）：
        1. LoadImageFromFile → BGR图像 (H, W, 3)
        2. Resize(keep_ratio=False) → 直接缩放到 (img_size, img_size)
        3. PackSegInputs → 打包为Tensor
        4. SegDataPreProcessor → BGR转RGB + ImageNet归一化
        5. 模型前向传播
        6. 输出自动resize回原图尺寸 (H, W)
        
        Args:
            image_path: 图像路径
        
        Returns:
            预测mask (H, W)，像素值为类别ID，与原图尺寸一致
        """
        from mmseg.apis import inference_model
        
        # 使用MMSeg官方推理API
        # inference_model内部会：
        # 1. 使用test_pipeline预处理（Resize到固定尺寸）
        # 2. 使用data_preprocessor归一化（BGR→RGB + normalize）
        # 3. 模型前向传播
        # 4. 自动resize结果回原图尺寸
        result = inference_model(self.model, image_path)
        
        # 提取预测mask
        pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
        
        return pred_mask.astype(np.int32)
    
    def predict_with_confidence(
        self, 
        image_path: str, 
        conf_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        带置信度过滤的推理【新增功能】
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值，低于此值的像素设为背景(255)
        
        Returns:
            (pred_mask, confidence_map): 预测mask和置信度图
        """
        import torch
        from mmseg.apis import inference_model
        
        # 使用MMSeg官方推理API
        result = inference_model(self.model, image_path)
        
        # 获取logits（如果可用）
        if hasattr(result, 'seg_logits') and result.seg_logits is not None:
            logits = result.seg_logits.data  # (C, H, W)
            
            # 计算softmax概率
            probs = torch.softmax(logits, dim=0)  # (C, H, W)
            
            # 获取最大概率和对应类别
            max_probs, pred_mask = probs.max(dim=0)  # (H, W)
            
            # 转为numpy
            max_probs = max_probs.cpu().numpy()
            pred_mask = pred_mask.cpu().numpy()
            
            # 置信度过滤：低于阈值的设为背景(255)
            if conf_threshold > 0:
                low_conf_mask = max_probs < conf_threshold
                pred_mask[low_conf_mask] = 255
            
            return pred_mask.astype(np.int32), max_probs.astype(np.float32)
        else:
            # 如果没有logits，使用普通推理
            pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
            confidence_map = np.ones_like(pred_mask, dtype=np.float32)
            return pred_mask.astype(np.int32), confidence_map
    
    def predict_file_with_confidence(
        self, 
        image_path: str, 
        conf_threshold: float = 0.5
    ) -> 'InferenceResult':
        """
        预测单个图像文件（带置信度过滤）【新增功能】
        
        Args:
            image_path: 图像文件路径
            conf_threshold: 置信度阈值
        
        Returns:
            InferenceResult对象
        """
        start_time = time.time()
        
        # 检查图像尺寸决定推理方式
        img = pil_open_chinese(image_path)
        W, H = img.size
        
        if self.use_sliding_window and (H > self.img_size * 1.5 or W > self.img_size * 1.5):
            # 滑动窗口暂不支持置信度过滤
            pred_mask = self.predict_with_sliding_window(image_path)
        else:
            pred_mask, _ = self.predict_with_confidence(image_path, conf_threshold)
        
        # 统计类别像素数（跳过背景类255）
        class_counts = {}
        unique, counts = np.unique(pred_mask, return_counts=True)
        for cls_id, count in zip(unique, counts):
            if cls_id != 255:  # 只跳过忽略类
                class_counts[int(cls_id)] = int(count)
        
        inference_time = time.time() - start_time
        
        return InferenceResult(
            image_path=image_path,
            pred_mask=pred_mask,
            class_counts=class_counts,
            inference_time=inference_time,
        )
    
    def predict_with_sliding_window(self, image_path: str) -> np.ndarray:
        """
        使用滑动窗口推理大图像（适用于大图像，保持原始分辨率）
        
        ================================================================
        为什么滑动窗口与训练一致？
        ================================================================
        训练时：RandomCrop 从大图中裁剪出 img_size × img_size 的完整图像块
        推理时：滑动窗口切出 img_size × img_size 的完整图像块
        
        每个窗口都是完整的图像内容，与训练时的输入一致！
        ================================================================
        
        处理流程：
        1. 计算padding确保能完整覆盖
        2. 滑动窗口切分（窗口大小=img_size，重叠50%）
        3. 每个窗口独立推理：
           - 保存为临时PNG（保持BGR格式）
           - 使用inference_model推理
           - 累加到投票图
        4. 投票合并：vote_map / count_map → argmax
        5. 裁剪回原图尺寸
        
        Args:
            image_path: 图像路径
        
        Returns:
            预测mask (H, W)，与原图尺寸一致
        """
        import cv2
        from mmseg.apis import inference_model
        
        # 加载原图（支持中文路径）
        img = cv2_imread_chinese(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        H, W = img.shape[:2]
        
        # 如果图像不大，直接推理
        if H <= self.img_size * 1.5 and W <= self.img_size * 1.5:
            return self.predict_single(image_path)
        
        print(f"   使用滑动窗口: 原图 {W}x{H}, 窗口 {self.window_size}, 步长 {self.stride}")
        
        # 计算padding（确保能完整覆盖）
        pad_h = (self.stride - (H - self.window_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.window_size) % self.stride) % self.stride
        
        if H < self.window_size:
            pad_h = self.window_size - H
        if W < self.window_size:
            pad_w = self.window_size - W
        
        if pad_h > 0 or pad_w > 0:
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            img_padded = img
        
        padded_H, padded_W = img_padded.shape[:2]
        
        # 初始化投票计数
        vote_map = np.zeros((self.num_classes, padded_H, padded_W), dtype=np.float32)
        count_map = np.zeros((padded_H, padded_W), dtype=np.float32)
        
        # 临时文件用于保存窗口
        import tempfile
        import os
        
        # 统计窗口数量
        num_windows = 0
        for y in range(0, padded_H - self.window_size + 1, self.stride):
            for x in range(0, padded_W - self.window_size + 1, self.stride):
                num_windows += 1
        
        print(f"   窗口数量: {num_windows}")
        
        # 滑动窗口推理
        window_idx = 0
        for y in range(0, padded_H - self.window_size + 1, self.stride):
            for x in range(0, padded_W - self.window_size + 1, self.stride):
                window_idx += 1
                
                # 提取窗口（完整的img_size×img_size块，与训练时的RandomCrop输出一致）
                window = img_padded[y:y+self.window_size, x:x+self.window_size]
                
                # 保存临时图像（保持BGR格式，LoadImageFromFile会正确处理）
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, window)
                
                try:
                    # 推理（使用相同的test_pipeline和data_preprocessor）
                    result = inference_model(self.model, tmp_path)
                    pred = result.pred_sem_seg.data.cpu().numpy()[0]
                    
                    # 累加到投票图（one-hot编码）
                    for c in range(self.num_classes):
                        vote_map[c, y:y+self.window_size, x:x+self.window_size] += (pred == c).astype(np.float32)
                    count_map[y:y+self.window_size, x:x+self.window_size] += 1
                finally:
                    os.unlink(tmp_path)
        
        # 计算最终预测（投票）
        count_map[count_map == 0] = 1
        vote_map = vote_map / count_map
        pred_mask = vote_map.argmax(axis=0).astype(np.int32)
        
        # 裁剪回原始尺寸
        pred_mask = pred_mask[:H, :W]
        
        return pred_mask
    
    def predict_file(self, image_path: str) -> InferenceResult:
        """
        预测单个图像文件
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            InferenceResult对象
        """
        start_time = time.time()
        
        # 检查图像尺寸决定推理方式
        img = pil_open_chinese(image_path)
        W, H = img.size
        
        if self.use_sliding_window and (H > self.img_size * 1.5 or W > self.img_size * 1.5):
            pred_mask = self.predict_with_sliding_window(image_path)
        else:
            pred_mask = self.predict_single(image_path)
        
        # 统计类别像素数（跳过背景类0和忽略类255）
        class_counts = {}
        unique, counts = np.unique(pred_mask, return_counts=True)
        for cls_id, count in zip(unique, counts):
            if cls_id != 0 and cls_id != 255:  # 跳过背景和忽略类
                class_counts[int(cls_id)] = int(count)
        
        inference_time = time.time() - start_time
        
        return InferenceResult(
            image_path=image_path,
            pred_mask=pred_mask,
            class_counts=class_counts,
            inference_time=inference_time,
        )


def batch_inference(
    checkpoint_path: str,
    images_dir: str,
    output_dir: str,
    device: str = 'cuda:0',
    use_sliding_window: bool = True,
    window_size: int = 512,
    stride: int = 256,
    conf_threshold: Optional[float] = None,
    save_overlay: bool = True,
    overlay_alpha: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchInferenceResult:
    """
    批量推理
    
    Args:
        checkpoint_path: 模型checkpoint路径
        images_dir: 图像文件夹路径
        output_dir: 输出文件夹路径
        device: 推理设备
        use_sliding_window: 是否使用滑动窗口
        window_size: 滑动窗口大小
        stride: 滑动步长
        conf_threshold: 置信度阈值
        save_overlay: 是否保存叠加可视化图
        overlay_alpha: 叠加透明度
        progress_callback: 进度回调函数 (current, total, image_name)
    
    Returns:
        BatchInferenceResult对象
    """
    from data.visualizer import visualize_mask_overlay
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    if not images_path.exists():
        return BatchInferenceResult(
            success=False,
            message=f"❌ 图像文件夹不存在: {images_dir}"
        )
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f'*{ext}'))
        image_files.extend(images_path.glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        return BatchInferenceResult(
            success=False,
            message="❌ 没有找到图像文件"
        )
    
    # 创建推理器
    try:
        inferencer = SegFormerInference(
            checkpoint_path=checkpoint_path,
            device=device,
            use_sliding_window=use_sliding_window,
            window_size=window_size,
            stride=stride,
            conf_threshold=conf_threshold,
        )
    except Exception as e:
        return BatchInferenceResult(
            success=False,
            message=f"❌ 加载模型失败: {str(e)}"
        )
    
    # 批量推理
    start_time = time.time()
    processed = 0
    failed = 0
    
    for i, image_file in enumerate(image_files):
        try:
            if progress_callback:
                progress_callback(i + 1, len(image_files), image_file.name)
            
            # 推理
            result = inferencer.predict_file(str(image_file))
            
            # 保存结果
            if save_overlay:
                # 加载原图
                img = np.array(pil_open_chinese(image_file).convert('RGB'))
                
                # 生成叠加图（跳过背景255）
                overlay_img = visualize_mask_overlay(
                    img, 
                    result.pred_mask, 
                    alpha=overlay_alpha,
                    ignore_index=255,  # 【工业级修复】跳过背景255
                )
                
                # 保存（支持中文路径）- 【工业级修复】失败必须报错
                output_file = output_path / f"{image_file.stem}_overlay.png"
                try:
                    overlay_img.save(str(output_file))
                except Exception as save_error:
                    # 尝试用临时文件
                    import tempfile
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            overlay_img.save(tmp.name)
                            shutil.copy(tmp.name, str(output_file))
                            os.unlink(tmp.name)
                    except Exception as e:
                        raise RuntimeError(f"❌ 无法保存图像 {output_file}: {save_error}, 备用方案也失败: {e}")
            
            processed += 1
            
        except Exception as e:
            # 【工业级修复】记录详细错误信息
            import traceback
            error_msg = f"❌ 处理失败 {image_file.name}: {e}\n{traceback.format_exc()}"
            print(error_msg)
            failed += 1
            # 如果失败率过高，提前终止
            if failed > len(image_files) * 0.5:
                raise RuntimeError(f"❌ 批量推理失败率过高 ({failed}/{len(image_files)}), 请检查模型和数据")
    
    total_time = time.time() - start_time
    
    # 【工业级修复】如果全部失败，抛出异常
    if processed == 0:
        raise RuntimeError(f"❌ 批量推理全部失败，共 {failed} 张图像")
    
    return BatchInferenceResult(
        success=True,
        message=f"✅ 批量推理完成\n   处理: {processed} 张\n   失败: {failed} 张\n   用时: {total_time:.1f} 秒",
        total_images=len(image_files),
        processed_images=processed,
        failed_images=failed,
        total_time=total_time,
        output_dir=str(output_path),
    )


def mask_to_labelme_json(
    mask: np.ndarray,
    image_path: str,
    class_names: List[str],
    image_height: int,
    image_width: int,
    min_area: int = 100,
    simplify_tolerance: float = 3.0,
) -> dict:
    """
    将分割mask转换为LabelMe格式的JSON
    
    Args:
        mask: 分割mask (H, W)，像素值为类别ID
        image_path: 图像文件名
        class_names: 类别名称列表
        image_height: 图像高度
        image_width: 图像宽度
        min_area: 最小区域面积（过滤噪声）
        simplify_tolerance: 轮廓简化容差（越大点越少）
    
    Returns:
        LabelMe格式的JSON字典
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("需要安装OpenCV: pip install opencv-python")
    
    shapes = []
    
    # 遍历每个类别（跳过背景类0）
    unique_classes = np.unique(mask)
    for cls_id in unique_classes:
        if cls_id == 0 or cls_id == 255:  # 跳过背景和忽略类
            continue
        
        # 创建该类别的二值mask
        binary_mask = (mask == cls_id).astype(np.uint8) * 255
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 获取类别名称 - 使用数字作为label，和训练数据保持一致
        label = str(cls_id)
        
        # 处理每个轮廓
        for contour in contours:
            # 计算面积，过滤小区域
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 简化轮廓 - 使用更大的容差来减少点数
            # epsilon = 轮廓周长的百分比，或固定值
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(simplify_tolerance, perimeter * 0.02)  # 至少2%的周长
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 至少需要3个点
            if len(approx) < 3:
                continue
            
            # 如果点数还是太多，继续简化
            if len(approx) > 20:
                epsilon = perimeter * 0.05  # 增加到5%
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 转换为点列表 [[x, y], [x, y], ...]
            points = approx.reshape(-1, 2).tolist()
            
            # 添加shape
            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
    
    # 构建LabelMe JSON
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,  # 不包含base64数据，节省空间
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
    
    return labelme_json


def batch_inference_to_json(
    checkpoint_path: str,
    images_dir: str,
    output_dir: str,
    device: str = 'cuda:0',
    use_sliding_window: bool = True,
    window_size: int = 512,
    stride: int = 256,
    min_area: int = 100,
    simplify_tolerance: float = 2.0,
    conf_threshold: float = 0.0,  # 【新增】置信度阈值
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchInferenceResult:
    """
    批量推理并输出LabelMe格式JSON标注文件
    【工业级修复版】- 添加置信度阈值支持
    
    Args:
        checkpoint_path: 模型checkpoint路径
        images_dir: 图像文件夹路径
        output_dir: 输出文件夹路径（保存JSON文件）
        device: 推理设备
        use_sliding_window: 是否使用滑动窗口
        window_size: 滑动窗口大小
        stride: 滑动步长
        min_area: 最小区域面积（过滤噪声）
        simplify_tolerance: 轮廓简化容差
        conf_threshold: 置信度阈值（0-1，0表示不过滤）【新增】
        progress_callback: 进度回调函数
    
    Returns:
        BatchInferenceResult对象
    """
    import json
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    if not images_path.exists():
        return BatchInferenceResult(
            success=False,
            message=f"❌ 图像文件夹不存在: {images_dir}"
        )
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f'*{ext}'))
        image_files.extend(images_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))  # 去重排序
    
    if len(image_files) == 0:
        return BatchInferenceResult(
            success=False,
            message="❌ 没有找到图像文件"
        )
    
    # 创建推理器
    try:
        inferencer = SegFormerInference(
            checkpoint_path=checkpoint_path,
            device=device,
            use_sliding_window=use_sliding_window,
            window_size=window_size,
            stride=stride,
            conf_threshold=None,
        )
    except Exception as e:
        return BatchInferenceResult(
            success=False,
            message=f"❌ 加载模型失败: {str(e)}"
        )
    
    # 获取类别名称
    class_names = inferencer.class_names if inferencer.class_names else [f"class_{i}" for i in range(inferencer.num_classes)]
    
    # 批量推理
    start_time = time.time()
    processed = 0
    failed = 0
    total_shapes = 0
    
    for i, image_file in enumerate(image_files):
        try:
            if progress_callback:
                progress_callback(i + 1, len(image_files), image_file.name)
            
            # 加载图像获取尺寸
            img = pil_open_chinese(image_file)
            img_width, img_height = img.size
            
            # 推理【修改】支持置信度阈值
            if conf_threshold > 0:
                result = inferencer.predict_file_with_confidence(str(image_file), conf_threshold)
            else:
                result = inferencer.predict_file(str(image_file))
            
            # 转换为LabelMe JSON
            labelme_json = mask_to_labelme_json(
                mask=result.pred_mask,
                image_path=image_file.name,
                class_names=class_names,
                image_height=img_height,
                image_width=img_width,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance,
            )
            
            # 保存JSON文件
            json_file = output_path / f"{image_file.stem}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(labelme_json, f, ensure_ascii=False, indent=2)
            
            total_shapes += len(labelme_json['shapes'])
            processed += 1
            
        except Exception as e:
            # 【工业级修复】记录详细错误
            import traceback
            print(f"❌ 处理失败 {image_file.name}: {e}")
            print(traceback.format_exc())
            failed += 1
    
    total_time = time.time() - start_time
    
    # 【工业级修复】如果全部失败，返回失败状态
    if processed == 0:
        return BatchInferenceResult(
            success=False,
            message=f"❌ 批量推理全部失败，共 {failed} 张图像\n请检查模型和数据是否匹配",
            total_images=len(image_files),
            failed_images=failed,
        )
    
    # 构建结果消息
    conf_info = f"\n   置信度阈值: {conf_threshold}" if conf_threshold > 0 else ""
    
    return BatchInferenceResult(
        success=True,
        message=(
            f"✅ 批量推理完成\n"
            f"   处理图像: {processed} 张\n"
            f"   失败: {failed} 张\n"
            f"   检测区域: {total_shapes} 个{conf_info}\n"
            f"   用时: {total_time:.1f} 秒\n"
            f"   输出目录: {output_path}"
        ),
        total_images=len(image_files),
        processed_images=processed,
        failed_images=failed,
        total_time=total_time,
        output_dir=str(output_path),
    )