# -*- coding: utf-8 -*-
"""
OCR 推理引擎
支持: PaddleOCR / ONNX Runtime / TensorRT
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import time

@dataclass
class OCRResult:
    """OCR识别结果"""
    boxes: List[np.ndarray]      # 检测框列表 [(4, 2), ...]
    texts: List[str]             # 识别文本列表
    scores: List[float]          # 置信度列表
    det_time: float = 0.0        # 检测耗时(ms)
    rec_time: float = 0.0        # 识别耗时(ms)
    total_time: float = 0.0      # 总耗时(ms)
    
    def to_dict(self) -> Dict:
        return {
            'results': [
                {'box': box.tolist(), 'text': text, 'score': score}
                for box, text, score in zip(self.boxes, self.texts, self.scores)
            ],
            'det_time_ms': self.det_time,
            'rec_time_ms': self.rec_time,
            'total_time_ms': self.total_time,
        }


class PaddleOCREngine:
    """PaddleOCR 引擎 - 直接使用预训练模型"""
    
    def __init__(
        self,
        det_model: str = 'ch_PP-OCRv4_det',
        rec_model: str = 'ch_PP-OCRv4_rec',
        use_angle_cls: bool = False,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ):
        """
        初始化PaddleOCR引擎
        
        Args:
            det_model: 检测模型名称
            rec_model: 识别模型名称
            use_angle_cls: 是否使用方向分类
            use_gpu: 是否使用GPU
            gpu_id: GPU ID
        """
        try:
            from paddleocr import PaddleOCR
            import paddleocr
        except ImportError:
            raise ImportError("请安装PaddleOCR: pip install paddleocr paddlepaddle-gpu")
        
        # 检查版本并使用正确的API
        version = getattr(paddleocr, '__version__', '2.0.0')
        major_version = int(version.split('.')[0])
        self._ocr_v3 = major_version >= 3  # 保存版本信息用于 ocr() 调用
        
        if major_version >= 3:
            # PaddleOCR 3.x: use device, no show_log or gpu_mem
            device = 'gpu' if use_gpu else 'cpu'
            self.ocr = PaddleOCR(
                det_model_dir=None,
                rec_model_dir=None,
                use_angle_cls=use_angle_cls,
                lang='ch',
                device=device,
            )
        else:
            # PaddleOCR 2.x
            self.ocr = PaddleOCR(
                det_model_dir=None,
                rec_model_dir=None,
                use_angle_cls=use_angle_cls,
                lang='ch',
                use_gpu=use_gpu,
                gpu_mem=4000,
                show_log=False,
            )
        
        self.det_model = det_model
        self.rec_model = rec_model
        self.use_gpu = use_gpu
    
    def __call__(self, image: Union[str, np.ndarray]) -> OCRResult:
        """执行OCR识别"""
        return self.predict(image)
    
    def predict(self, image: Union[str, np.ndarray]) -> OCRResult:
        """
        执行OCR识别
        
        Args:
            image: 图像路径或numpy数组
        
        Returns:
            OCRResult
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行OCR
        det_start = time.time()
        # PaddleOCR 3.x: ocr() 不再接受 cls/rec/det 参数，可能使用 predict()
        if self._ocr_v3:
            if hasattr(self.ocr, 'predict'):
                results = self.ocr.predict(image)
            else:
                results = self.ocr.ocr(image)
        else:
            results = self.ocr.ocr(image, cls=False)
        total_time = (time.time() - start_time) * 1000
        
        # 解析结果 - 兼容 PaddleOCR 2.x 和 3.x 格式
        boxes, texts, scores = [], [], []
        if results:
            result_list = results[0] if isinstance(results, list) and len(results) > 0 else results
            if result_list:
                for item in result_list:
                    try:
                        # PaddleOCR 3.x dict 格式
                        if isinstance(item, dict):
                            box = item.get('dt_polys', item.get('box', item.get('points', [])))
                            text = item.get('rec_text', item.get('text', ''))
                            score = item.get('rec_score', item.get('score', 0.0))
                            if box:
                                boxes.append(np.array(box))
                                texts.append(str(text))
                                scores.append(float(score) if score else 0.0)
                        # PaddleOCR 2.x list 格式
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            box = np.array(item[0])
                            text_info = item[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text, score = text_info[0], text_info[1]
                            elif isinstance(text_info, dict):
                                text = text_info.get('text', text_info.get('rec_text', ''))
                                score = text_info.get('score', text_info.get('rec_score', 0.0))
                            else:
                                continue
                            boxes.append(box)
                            texts.append(str(text))
                            scores.append(float(score))
                    except (IndexError, TypeError, ValueError):
                        continue
        
        return OCRResult(
            boxes=boxes,
            texts=texts,
            scores=scores,
            total_time=total_time,
        )
    
    def detect_only(self, image: Union[str, np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """仅执行文本检测"""
        start_time = time.time()
        
        if isinstance(image, str):
            image = cv2.imread(image)
        
        boxes = []
        # PaddleOCR 3.x: ocr() 不再接受 rec 参数，执行完整OCR后只取检测结果
        if self._ocr_v3:
            if hasattr(self.ocr, 'predict'):
                results = self.ocr.predict(image)
            else:
                results = self.ocr.ocr(image)
            det_time = (time.time() - start_time) * 1000
            if results:
                result_list = results[0] if isinstance(results, list) and len(results) > 0 else results
                if result_list:
                    for item in result_list:
                        try:
                            if isinstance(item, dict):
                                box = item.get('dt_polys', item.get('box', item.get('points', [])))
                                if box:
                                    boxes.append(np.array(box))
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                boxes.append(np.array(item[0]))
                        except (IndexError, TypeError, ValueError):
                            continue
        else:
            # 使用内部检测器
            results = self.ocr.ocr(image, rec=False)
            det_time = (time.time() - start_time) * 1000
            if results and results[0]:
                for line in results[0]:
                    boxes.append(np.array(line))
        
        return boxes, det_time
    
    def recognize_only(self, image: Union[str, np.ndarray], boxes: List[np.ndarray]) -> Tuple[List[str], List[float], float]:
        """仅执行文本识别"""
        start_time = time.time()
        
        if isinstance(image, str):
            image = cv2.imread(image)
        
        texts, scores = [], []
        for box in boxes:
            # 裁剪文本区域
            cropped = self._crop_text_region(image, box)
            # PaddleOCR 3.x: ocr() 不再接受 det 参数，执行完整OCR
            if self._ocr_v3:
                if hasattr(self.ocr, 'predict'):
                    result = self.ocr.predict(cropped)
                else:
                    result = self.ocr.ocr(cropped)
                
                text, score = '', 0.0
                if result:
                    result_list = result[0] if isinstance(result, list) and len(result) > 0 else result
                    if result_list:
                        item = result_list[0] if isinstance(result_list, list) else result_list
                        try:
                            if isinstance(item, dict):
                                text = item.get('rec_text', item.get('text', ''))
                                score = item.get('rec_score', item.get('score', 0.0))
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                text_info = item[1]
                                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                    text, score = text_info[0], text_info[1]
                                elif isinstance(text_info, dict):
                                    text = text_info.get('text', text_info.get('rec_text', ''))
                                    score = text_info.get('score', text_info.get('rec_score', 0.0))
                        except (IndexError, TypeError, ValueError):
                            pass
                texts.append(str(text))
                scores.append(float(score) if score else 0.0)
            else:
                result = self.ocr.ocr(cropped, det=False)
                if result and result[0]:
                    texts.append(result[0][0][0])
                    scores.append(result[0][0][1])
                else:
                    texts.append('')
                    scores.append(0.0)
        
        rec_time = (time.time() - start_time) * 1000
        return texts, scores, rec_time
    
    def _crop_text_region(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """裁剪文本区域"""
        box = box.astype(np.int32)
        x_min, y_min = box.min(axis=0)
        x_max, y_max = box.max(axis=0)
        return image[y_min:y_max, x_min:x_max]


class ONNXOCREngine:
    """ONNX Runtime OCR 引擎"""
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        rec_char_dict_path: str,
        use_gpu: bool = True,
    ):
        """
        初始化ONNX OCR引擎
        
        Args:
            det_model_path: 检测模型ONNX路径
            rec_model_path: 识别模型ONNX路径
            rec_char_dict_path: 字符字典路径
            use_gpu: 是否使用GPU
        """
        import onnxruntime as ort
        
        # 设置Provider
        providers = ['CPUExecutionProvider']
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        # 加载模型
        self.det_session = ort.InferenceSession(det_model_path, providers=providers)
        self.rec_session = ort.InferenceSession(rec_model_path, providers=providers)
        
        # 加载字符字典
        self.char_dict = self._load_char_dict(rec_char_dict_path)
        
        # 预处理参数
        self.det_input_size = (640, 640)
        self.rec_input_height = 48
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def _load_char_dict(self, dict_path: str) -> List[str]:
        """加载字符字典"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f]
        return ['blank'] + chars + ['<eos>']
    
    def predict(self, image: Union[str, np.ndarray]) -> OCRResult:
        """执行OCR识别"""
        start_time = time.time()
        
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. 文本检测
        det_start = time.time()
        boxes = self._detect(image)
        det_time = (time.time() - det_start) * 1000
        
        # 2. 文本识别
        rec_start = time.time()
        texts, scores = self._recognize(image, boxes)
        rec_time = (time.time() - rec_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return OCRResult(
            boxes=boxes,
            texts=texts,
            scores=scores,
            det_time=det_time,
            rec_time=rec_time,
            total_time=total_time,
        )
    
    def _detect(self, image: np.ndarray) -> List[np.ndarray]:
        """文本检测"""
        # 预处理
        h, w = image.shape[:2]
        resized = cv2.resize(image, self.det_input_size)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        
        # 推理
        input_name = self.det_session.get_inputs()[0].name
        output = self.det_session.run(None, {input_name: input_tensor})[0]
        
        # 后处理 (简化版DBNet后处理)
        boxes = self._db_postprocess(output[0], (h, w))
        
        return boxes
    
    def _db_postprocess(self, pred: np.ndarray, original_size: Tuple[int, int], thresh: float = 0.3) -> List[np.ndarray]:
        """DBNet后处理"""
        pred = pred[0] if pred.ndim == 3 else pred
        h, w = original_size
        
        # 二值化
        binary = (pred > thresh).astype(np.uint8) * 255
        
        # 调整到原始尺寸
        binary = cv2.resize(binary, (w, h))
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            boxes.append(box.astype(np.int32))
        
        return boxes
    
    def _recognize(self, image: np.ndarray, boxes: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """文本识别"""
        texts, scores = [], []
        
        for box in boxes:
            # 裁剪并矫正
            cropped = self._crop_and_rectify(image, box)
            if cropped is None or cropped.size == 0:
                texts.append('')
                scores.append(0.0)
                continue
            
            # 预处理
            h, w = cropped.shape[:2]
            new_w = int(w * self.rec_input_height / h)
            resized = cv2.resize(cropped, (new_w, self.rec_input_height))
            normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            input_tensor = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            
            # 推理
            input_name = self.rec_session.get_inputs()[0].name
            output = self.rec_session.run(None, {input_name: input_tensor})[0]
            
            # 解码
            text, score = self._ctc_decode(output[0])
            texts.append(text)
            scores.append(score)
        
        return texts, scores
    
    def _crop_and_rectify(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """裁剪并矫正文本区域"""
        box = box.astype(np.float32)
        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        
        if width <= 0 or height <= 0:
            return None
        
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    
    def _ctc_decode(self, output: np.ndarray) -> Tuple[str, float]:
        """CTC解码"""
        # 贪心解码
        indices = np.argmax(output, axis=1)
        scores = np.max(output, axis=1)
        
        # 去重和去blank
        text = ''
        prev_idx = -1
        char_scores = []
        
        for idx, score in zip(indices, scores):
            if idx != 0 and idx != prev_idx:  # 0是blank
                if idx < len(self.char_dict):
                    text += self.char_dict[idx]
                    char_scores.append(score)
            prev_idx = idx
        
        avg_score = np.mean(char_scores) if char_scores else 0.0
        return text, float(avg_score)


class TensorRTOCREngine:
    """TensorRT OCR 引擎 - 高性能部署"""
    
    def __init__(
        self,
        det_engine_path: str,
        rec_engine_path: str,
        rec_char_dict_path: str,
    ):
        """
        初始化TensorRT OCR引擎
        
        Args:
            det_engine_path: 检测引擎路径 (.engine)
            rec_engine_path: 识别引擎路径 (.engine)
            rec_char_dict_path: 字符字典路径
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("请安装TensorRT和PyCUDA")
        
        self.trt = trt
        self.cuda = cuda
        
        # 加载引擎
        self.det_engine = self._load_engine(det_engine_path)
        self.rec_engine = self._load_engine(rec_engine_path)
        
        # 创建执行上下文
        self.det_context = self.det_engine.create_execution_context()
        self.rec_context = self.rec_engine.create_execution_context()
        
        # 加载字符字典
        with open(rec_char_dict_path, 'r', encoding='utf-8') as f:
            self.char_dict = ['blank'] + [line.strip() for line in f] + ['<eos>']
        
        # 分配GPU内存
        self._allocate_buffers()
    
    def _load_engine(self, engine_path: str):
        """加载TensorRT引擎"""
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            return self.trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    def _allocate_buffers(self):
        """分配GPU缓冲区"""
        # 检测模型缓冲区
        self.det_inputs = []
        self.det_outputs = []
        self.det_bindings = []
        
        for i in range(self.det_engine.num_io_tensors):
            name = self.det_engine.get_tensor_name(i)
            shape = self.det_engine.get_tensor_shape(name)
            dtype = self.trt.nptype(self.det_engine.get_tensor_dtype(name))
            size = self.trt.volume(shape)
            
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.det_bindings.append(int(device_mem))
            
            if self.det_engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.det_inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.det_outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def predict(self, image: Union[str, np.ndarray]) -> OCRResult:
        """执行OCR识别"""
        start_time = time.time()
        
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测
        det_start = time.time()
        boxes = self._detect_trt(image)
        det_time = (time.time() - det_start) * 1000
        
        # 识别
        rec_start = time.time()
        texts, scores = self._recognize_trt(image, boxes)
        rec_time = (time.time() - rec_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return OCRResult(
            boxes=boxes,
            texts=texts,
            scores=scores,
            det_time=det_time,
            rec_time=rec_time,
            total_time=total_time,
        )
    
    def _detect_trt(self, image: np.ndarray) -> List[np.ndarray]:
        """TensorRT文本检测"""
        # 预处理
        h, w = image.shape[:2]
        input_tensor = self._preprocess_det(image)
        
        # 复制到GPU
        np.copyto(self.det_inputs[0]['host'], input_tensor.ravel())
        self.cuda.memcpy_htod(self.det_inputs[0]['device'], self.det_inputs[0]['host'])
        
        # 执行推理
        self.det_context.execute_v2(self.det_bindings)
        
        # 复制回CPU
        self.cuda.memcpy_dtoh(self.det_outputs[0]['host'], self.det_outputs[0]['device'])
        output = self.det_outputs[0]['host'].reshape(self.det_outputs[0]['shape'])
        
        # 后处理
        boxes = self._db_postprocess(output, (h, w))
        
        return boxes
    
    def _recognize_trt(self, image: np.ndarray, boxes: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """TensorRT文本识别"""
        texts, scores = [], []
        
        for box in boxes:
            cropped = self._crop_and_rectify(image, box)
            if cropped is None:
                texts.append('')
                scores.append(0.0)
                continue
            
            # 预处理并推理
            text, score = self._recognize_single_trt(cropped)
            texts.append(text)
            scores.append(score)
        
        return texts, scores
    
    def _preprocess_det(self, image: np.ndarray) -> np.ndarray:
        """检测预处理"""
        resized = cv2.resize(image, (640, 640))
        normalized = (resized.astype(np.float32) / 255.0 - 0.485) / 0.229
        return normalized.transpose(2, 0, 1)[np.newaxis]
    
    def _db_postprocess(self, pred: np.ndarray, original_size: Tuple[int, int]) -> List[np.ndarray]:
        """DBNet后处理"""
        # 简化实现
        pred = pred[0, 0] if pred.ndim == 4 else pred[0]
        binary = (pred > 0.3).astype(np.uint8) * 255
        binary = cv2.resize(binary, (original_size[1], original_size[0]))
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.int32)
            boxes.append(box)
        
        return boxes
    
    def _crop_and_rectify(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """裁剪并矫正"""
        box = box.astype(np.float32)
        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        
        if width <= 0 or height <= 0:
            return None
        
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box, dst)
        return cv2.warpPerspective(image, M, (width, height))
    
    def _recognize_single_trt(self, cropped: np.ndarray) -> Tuple[str, float]:
        """单个文本区域识别"""
        # 简化实现，实际需要完整的TensorRT推理
        return '', 0.0


def create_engine(engine_type: str = 'paddle', **kwargs):
    """
    创建OCR引擎
    
    Args:
        engine_type: 引擎类型 ('paddle', 'onnx', 'tensorrt')
        **kwargs: 引擎参数
    
    Returns:
        OCR引擎实例
    """
    if engine_type == 'paddle':
        return PaddleOCREngine(**kwargs)
    elif engine_type == 'onnx':
        return ONNXOCREngine(**kwargs)
    elif engine_type == 'tensorrt':
        return TensorRTOCREngine(**kwargs)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
