# -*- coding: utf-8 -*-
"""
工业级阈值校准器

基于百分位数法，仅使用良品数据进行阈值校准
适用于工业场景（训练数据通常只有良品）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ThresholdResult:
    """阈值结果"""
    threshold: float = 50.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    method: str = 'percentile'
    
    def to_dict(self) -> Dict:
        return {
            'threshold': self.threshold,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'method': self.method,
        }


class IndustrialThresholdCalibrator:
    """
    工业级阈值校准器
    
    设计原则:
    1. 仅使用良品数据即可校准（工业常见场景）
    2. 输出归一化到0-100范围，类似置信度
    3. 提供多个预设阈值供用户选择
    4. 阈值调整直观易懂（调高=更严格=更少报警）
    """
    
    def __init__(self):
        self.normalization_params = {}
        self.threshold_presets = {}
        self.score_distribution = None
    
    def calibrate(
        self,
        good_scores: np.ndarray,
        defect_scores: np.ndarray = None,
        method: str = 'percentile',
    ) -> Dict:
        """
        校准阈值
        
        Args:
            good_scores: 良品分数 (必需)
            defect_scores: 异常分数 (可选，用于优化)
            method: 校准方法 ('percentile', 'sigma', 'f1')
        
        Returns:
            校准结果字典
        """
        good_scores = np.asarray(good_scores).flatten()
        
        if len(good_scores) == 0:
            raise ValueError("良品分数不能为空")
        
        # 保存分数分布
        self.score_distribution = good_scores.copy()
        
        # Step 1: 计算归一化参数
        self.normalization_params = self._compute_normalization_params(good_scores)
        
        # Step 2: 计算预设阈值
        self.threshold_presets = self._compute_threshold_presets(good_scores)
        
        # Step 3: 如果有异常样本，计算最优阈值
        if defect_scores is not None and len(defect_scores) > 0:
            defect_scores = np.asarray(defect_scores).flatten()
            optimal_result = self._optimize_with_defects(good_scores, defect_scores)
            self.threshold_presets.update(optimal_result)
        
        return self.get_calibration_result()
    
    def _compute_normalization_params(self, scores: np.ndarray) -> Dict:
        """计算归一化参数"""
        return {
            'method': 'percentile',
            'p1': float(np.percentile(scores, 1)),
            'p5': float(np.percentile(scores, 5)),
            'p50': float(np.percentile(scores, 50)),
            'p95': float(np.percentile(scores, 95)),
            'p99': float(np.percentile(scores, 99)),
            'p99_5': float(np.percentile(scores, 99.5)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
        }
    
    def _compute_threshold_presets(self, scores: np.ndarray) -> Dict:
        """计算预设阈值点"""
        p1 = self.normalization_params['p1']
        p99 = self.normalization_params['p99']
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        
        # 归一化函数
        def normalize(raw_score):
            if p99 - p1 < 1e-8:
                return 50.0
            return (raw_score - p1) / (p99 - p1) * 100
        
        presets = {
            # 基于分布的预设
            'ultra_sensitive': 30.0,   # 高召回，可能较多误报
            'sensitive': 40.0,         # 较高召回
            'balanced': 50.0,          # 默认平衡点 (P99)
            'strict': 65.0,            # 较高精度
            'very_strict': 80.0,       # 极少误报
            
            # 基于sigma的预设
            '2_sigma': max(0, normalize(mean + 2 * std)),
            '3_sigma': max(0, normalize(mean + 3 * std)),
            
            # 默认值
            'default': 50.0,
        }
        
        # 计算每个预设对应的假阳率估计
        presets['fpr_estimates'] = {}
        for name in ['ultra_sensitive', 'sensitive', 'balanced', 'strict', 'very_strict']:
            thresh = presets[name]
            raw_thresh = self._denormalize(thresh)
            fpr = np.mean(scores >= raw_thresh)
            presets['fpr_estimates'][name] = float(fpr)
        
        return presets
    
    def _optimize_with_defects(self, good_scores: np.ndarray, defect_scores: np.ndarray) -> Dict:
        """使用异常样本优化阈值"""
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
        except ImportError:
            return {}
        
        # 归一化
        good_norm = self.normalize_scores(good_scores)
        defect_norm = self.normalize_scores(defect_scores)
        
        # 合并
        all_scores = np.concatenate([good_norm, defect_norm])
        labels = np.concatenate([np.zeros(len(good_norm)), np.ones(len(defect_norm))])
        
        # 搜索最优F1阈值
        best_f1 = 0
        best_thresh = 50.0
        best_metrics = {}
        
        for thresh in np.linspace(10, 90, 81):
            preds = (all_scores >= thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {
                    'f1': float(f1),
                    'precision': float(precision_score(labels, preds, zero_division=0)),
                    'recall': float(recall_score(labels, preds, zero_division=0)),
                }
        
        return {
            'optimal_f1': float(best_thresh),
            'optimal_f1_metrics': best_metrics,
        }
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到0-100范围"""
        scores = np.asarray(scores).flatten()
        
        p1 = self.normalization_params['p1']
        p99 = self.normalization_params['p99']
        
        if p99 - p1 < 1e-8:
            return np.full_like(scores, 50.0)
        
        normalized = (scores - p1) / (p99 - p1) * 100
        return np.clip(normalized, 0, 200)  # 允许超过100
    
    def _denormalize(self, normalized_score: float) -> float:
        """反归一化"""
        p1 = self.normalization_params['p1']
        p99 = self.normalization_params['p99']
        return normalized_score / 100 * (p99 - p1) + p1
    
    def get_calibration_result(self) -> Dict:
        """获取完整校准结果"""
        return {
            'normalization': self.normalization_params,
            'thresholds': self.threshold_presets,
            'default_threshold': self.threshold_presets.get('default', 50.0),
            'recommended_range': [30.0, 70.0],
            'usage_guide': {
                '30-40': '高召回模式 - 适合安全关键应用，会有较多误报',
                '40-55': '平衡模式 - 推荐日常使用',
                '55-70': '高精度模式 - 减少误报，可能漏检轻微异常',
                '70+': '极严格模式 - 仅检测明显异常',
            },
        }
    
    def evaluate_threshold(
        self,
        good_scores: np.ndarray,
        defect_scores: np.ndarray,
        threshold: float,
    ) -> ThresholdResult:
        """评估指定阈值的性能"""
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
        except ImportError:
            return ThresholdResult(threshold=threshold)
        
        good_norm = self.normalize_scores(good_scores)
        defect_norm = self.normalize_scores(defect_scores)
        
        all_scores = np.concatenate([good_norm, defect_norm])
        labels = np.concatenate([np.zeros(len(good_norm)), np.ones(len(defect_norm))])
        
        preds = (all_scores >= threshold).astype(int)
        
        return ThresholdResult(
            threshold=threshold,
            f1=float(f1_score(labels, preds, zero_division=0)),
            precision=float(precision_score(labels, preds, zero_division=0)),
            recall=float(recall_score(labels, preds, zero_division=0)),
            method='evaluation',
        )
    
    def analyze_threshold_curve(
        self,
        good_scores: np.ndarray,
        defect_scores: np.ndarray,
        num_points: int = 50,
    ) -> Dict:
        """分析阈值曲线"""
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
        except ImportError:
            return {}
        
        good_norm = self.normalize_scores(good_scores)
        defect_norm = self.normalize_scores(defect_scores)
        
        all_scores = np.concatenate([good_norm, defect_norm])
        labels = np.concatenate([np.zeros(len(good_norm)), np.ones(len(defect_norm))])
        
        thresholds = np.linspace(10, 90, num_points)
        f1_scores = []
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            preds = (all_scores >= thresh).astype(int)
            f1_scores.append(float(f1_score(labels, preds, zero_division=0)))
            precisions.append(float(precision_score(labels, preds, zero_division=0)))
            recalls.append(float(recall_score(labels, preds, zero_division=0)))
        
        return {
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
        }


class ThresholdManager:
    """阈值管理器 - 用于推理时的阈值应用"""
    
    def __init__(self, normalization_params: Dict, default_threshold: float = 50.0):
        self.normalization_params = normalization_params
        self.threshold = default_threshold
    
    def normalize(self, raw_score: float) -> float:
        """归一化单个分数"""
        p1 = self.normalization_params['p1']
        p99 = self.normalization_params['p99']
        
        if p99 - p1 < 1e-8:
            return 50.0
        
        normalized = (raw_score - p1) / (p99 - p1) * 100
        return max(0, normalized)
    
    def normalize_batch(self, raw_scores: np.ndarray) -> np.ndarray:
        """归一化批量分数"""
        p1 = self.normalization_params['p1']
        p99 = self.normalization_params['p99']
        
        if p99 - p1 < 1e-8:
            return np.full_like(raw_scores, 50.0)
        
        normalized = (raw_scores - p1) / (p99 - p1) * 100
        return np.clip(normalized, 0, 200)
    
    def is_anomaly(self, normalized_score: float) -> bool:
        """判断是否异常"""
        return normalized_score >= self.threshold
    
    def set_threshold(self, threshold: float):
        """设置阈值"""
        self.threshold = max(0, min(100, threshold))
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold
