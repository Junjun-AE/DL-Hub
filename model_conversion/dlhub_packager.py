# -*- coding: utf-8 -*-
"""
DL-Hub 模型打包器
=================
将转换后的模型文件打包为统一的 .dlhub 文件格式。

.dlhub 文件本质是一个 zip 压缩包，内部结构：

    model_package.dlhub
    ├── model/                    # 模型文件
    │   ├── model.engine          # TensorRT
    │   ├── model.onnx            # ONNX (ORT 或备用)
    │   ├── model.xml + model.bin # OpenVINO
    │   └── ...
    ├── deploy_config.json        # 统一部署配置
    ├── manifest.json             # 文件清单 + 校验
    └── README.txt                # 人可读说明

deploy_config.json 统一格式（所有任务通用）：
{
    "dlhub_version": "1.0",
    "task_type": "det",                   # cls/det/seg/anomaly/ocr/sevseg
    "backend": "tensorrt",                # tensorrt/openvino/ort
    "model_name": "yolo26m",
    "model_files": ["model/model.engine"],
    "input": {
        "shape": [1, 3, 640, 640],
        "dtype": "float32",
        "color_format": "RGB",
        "pixel_range": [0, 255],
        "normalize_method": "divide_255",
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
        "letterbox_color": [114, 114, 114]
    },
    "output": {
        "num_classes": 5,
        "class_names": ["cat", "dog", ...],
        "class_to_idx": {"cat": 0, "dog": 1, ...}
    },
    "precision": "fp16",
    "dynamic_batch": false,
    "export_info": {
        "exported_at": "2026-03-18T12:00:00",
        "source_framework": "ultralytics",
        "ultralytics_version": "8.4.0",
        "pytorch_version": "2.1.0"
    }
}

使用方法：
    from model_conversion.dlhub_packager import DLHubPackager

    # 打包
    packager = DLHubPackager()
    pkg_path = packager.pack(
        output_dir="/path/to/converted/tensorrt/20260318_120000",
        task_type="det",
        deploy_config={...},   # 可选，自动从目录推断
    )
    # -> /path/to/converted/tensorrt/20260318_120000/yolo26m_tensorrt_fp16.dlhub

    # 解包
    files = packager.unpack("/path/to/model.dlhub", "/path/to/extract")
"""

import os
import json
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


# 后端 -> 模型文件扩展名映射
BACKEND_FILE_PATTERNS = {
    'tensorrt': ['*.engine', '*.trt'],
    'openvino': ['*.xml', '*.bin'],
    'ort': ['*.onnx'],
    'onnxruntime': ['*.onnx'],
}

# 额外的通用文件（始终打包）
COMMON_PATTERNS = ['*.yaml', '*.json', '*.onnx']

# 任务类型规范化
TASK_TYPE_MAP = {
    'classification': 'cls',
    'cls': 'cls',
    'detection': 'det',
    'det': 'det',
    'segmentation': 'seg',
    'seg': 'seg',
    'anomaly': 'anomaly',
    'patchcore': 'anomaly',
    'ocr': 'ocr',
    'sevseg': 'sevseg',
    'score_detect': 'sevseg',
    # [P5-3 修复] YOLO-seg 实例分割：保留独立类型，预处理走 YOLO 系(divide_255+letterbox)
    'instance_seg': 'instance_seg',
    'instance_segmentation': 'instance_seg',
}


def _file_sha256(filepath: str, chunk_size: int = 8192) -> str:
    """计算文件SHA256"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _detect_backend(directory: Path) -> str:
    """从目录中的文件自动检测后端类型"""
    files = [f.name for f in directory.rglob('*') if f.is_file()]
    files_str = ' '.join(files).lower()

    if any(f.endswith('.engine') or f.endswith('.trt') for f in files):
        return 'tensorrt'
    if any(f.endswith('.xml') for f in files) and any(f.endswith('.bin') for f in files):
        return 'openvino'
    if any(f.endswith('.onnx') for f in files):
        return 'ort'
    return 'unknown'


def _collect_model_files(directory: Path, backend: str) -> List[Path]:
    """收集需要打包的模型文件"""
    files = []
    patterns = BACKEND_FILE_PATTERNS.get(backend, [])

    # 收集后端特定文件
    for pattern in patterns:
        files.extend(directory.rglob(pattern))

    # 收集通用配置文件
    for pattern in COMMON_PATTERNS:
        for f in directory.rglob(pattern):
            if f not in files:
                files.append(f)

    # 去重 + 排除 .dlhub 文件本身
    seen = set()
    unique = []
    for f in files:
        if f.suffix == '.dlhub':
            continue
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return sorted(unique, key=lambda x: x.name)


def _read_deploy_config_from_dir(directory: Path) -> Dict[str, Any]:
    """尝试从目录中读取已有的配置信息"""
    config = {}

    # 尝试读取 model_config.yaml
    for name in ['model_config.yaml', 'model_config.yml', 'config.yaml', 'deploy_config.json']:
        cfg_file = directory / name
        if cfg_file.exists():
            try:
                if name.endswith('.json'):
                    with open(cfg_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                else:
                    import yaml
                    with open(cfg_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                break
            except Exception:
                continue

    return config


class DLHubPackager:
    """DL-Hub 模型打包器"""

    DLHUB_VERSION = "1.0"

    def pack(
        self,
        output_dir: str,
        task_type: str = None,
        backend: str = None,
        deploy_config: Dict[str, Any] = None,
        model_name: str = None,
        precision: str = None,
        include_onnx: bool = True,
    ) -> Optional[str]:
        """
        将转换后的模型目录打包为 .dlhub 文件

        Args:
            output_dir: 转换输出目录（包含模型文件和配置）
            task_type: 任务类型 (cls/det/seg/anomaly/ocr/sevseg)
            backend: 后端类型 (tensorrt/openvino/ort)，自动检测
            deploy_config: 部署配置字典，自动从目录读取
            model_name: 模型名称，自动推断
            precision: 精度 (fp16/fp32/int8)，自动推断
            include_onnx: 是否包含ONNX文件

        Returns:
            .dlhub 文件路径，失败返回 None
        """
        directory = Path(output_dir)
        if not directory.exists():
            print(f"❌ 目录不存在: {output_dir}")
            return None

        # 自动检测后端
        if not backend:
            backend = _detect_backend(directory)
            if backend == 'unknown':
                print(f"⚠️ 无法自动检测后端类型，请手动指定")
                return None

        # 规范化任务类型
        if task_type:
            task_type = TASK_TYPE_MAP.get(task_type.lower(), task_type.lower())
        else:
            task_type = 'unknown'

        # 收集文件
        model_files = _collect_model_files(directory, backend)
        if not include_onnx and backend != 'ort':
            model_files = [f for f in model_files if f.suffix != '.onnx']

        if not model_files:
            print(f"❌ 目录中未找到模型文件: {output_dir}")
            return None

        # 读取或合并配置
        existing_config = _read_deploy_config_from_dir(directory)
        if deploy_config:
            existing_config.update(deploy_config)
        final_config = existing_config

        # 推断模型名称
        if not model_name:
            model_name = final_config.get('model_name') or final_config.get('model', {}).get('name', '')
            if not model_name:
                # 从文件名推断
                for f in model_files:
                    if f.suffix in ('.engine', '.xml', '.onnx'):
                        model_name = f.stem.replace('_fp16', '').replace('_fp32', '').replace('_int8', '')
                        break
                if not model_name:
                    model_name = 'model'

        # 推断精度
        if not precision:
            precision = final_config.get('precision', 'fp16')

        # 构建统一的 deploy_config.json
        unified_config = self._build_deploy_config(
            task_type=task_type,
            backend=backend,
            model_name=model_name,
            precision=precision,
            model_files=[f.name for f in model_files],
            source_config=final_config,
        )

        # 生成包名
        pkg_name = f"{model_name}_{backend}_{precision}.dlhub"
        pkg_path = directory / pkg_name

        # 打包
        try:
            manifest = {'version': self.DLHUB_VERSION, 'created': datetime.now().isoformat(), 'files': {}}

            with zipfile.ZipFile(pkg_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 写入模型文件
                for f in model_files:
                    arcname = f"model/{f.name}"
                    zf.write(f, arcname)
                    manifest['files'][arcname] = {
                        'size': f.stat().st_size,
                        'sha256': _file_sha256(str(f)),
                    }

                # 写入 deploy_config.json
                config_str = json.dumps(unified_config, indent=2, ensure_ascii=False)
                zf.writestr('deploy_config.json', config_str)

                # 写入 manifest.json
                zf.writestr('manifest.json', json.dumps(manifest, indent=2))

                # 写入 README.txt
                readme = self._generate_readme(model_name, backend, precision, task_type, model_files)
                zf.writestr('README.txt', readme)

            print(f"✅ 已打包: {pkg_path} ({pkg_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return str(pkg_path)

        except Exception as e:
            print(f"❌ 打包失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unpack(self, dlhub_path: str, extract_dir: str = None) -> Optional[Dict[str, Any]]:
        """
        解包 .dlhub 文件

        Args:
            dlhub_path: .dlhub 文件路径
            extract_dir: 解包目标目录，默认为同名目录

        Returns:
            deploy_config 字典，失败返回 None
        """
        dlhub_path = Path(dlhub_path)
        if not dlhub_path.exists():
            print(f"❌ 文件不存在: {dlhub_path}")
            return None

        if not extract_dir:
            extract_dir = dlhub_path.with_suffix('')
        extract_dir = Path(extract_dir)

        try:
            with zipfile.ZipFile(dlhub_path, 'r') as zf:
                zf.extractall(extract_dir)

            # 读取 deploy_config
            config_path = extract_dir / 'deploy_config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ 已解包: {extract_dir}")
                print(f"   模型: {config.get('model_name', 'unknown')}")
                print(f"   后端: {config.get('backend', 'unknown')}")
                print(f"   任务: {config.get('task_type', 'unknown')}")
                return config
            else:
                print(f"⚠️ 解包成功但缺少 deploy_config.json")
                return {}

        except Exception as e:
            print(f"❌ 解包失败: {e}")
            return None

    def _build_deploy_config(
        self, task_type, backend, model_name, precision, model_files, source_config
    ) -> Dict[str, Any]:
        """构建统一的部署配置（Batch 4: P7-1~P7-4, P7-7）

        关键改进：
        - [P7-1] 同时从 source_config['preprocessing'] 和模块写的 'input_spec' 读取真实预处理值，
          不再因键名不匹配而丢弃保存值、盲目回退默认。
        - [P7-2/P7-3] 预处理由"族(family)"决定：SegFormer 语义分割=imagenet+stretch(无letterbox)，
          YOLO 系(det/sevseg/instance_seg)=divide_255+letterbox。消除 seg 被笼统套 imagenet/letterbox 的矛盾。
        - [P7-4] 新增 output.postprocess：按任务保存 conf/iou/max_det/end2end/mask_threshold/argmax/softmax/ignore_index。
        - [P7-7] normalize_method 取值统一为 imagenet / divide_255 / none。
        """
        # ---- 来源信息（兼容多种历史结构）----
        preprocessing = source_config.get('preprocessing', {}) or {}
        # [P7-1] 模块实际写的是顶层/嵌套 input_spec，这里也读它
        input_spec = source_config.get('input_spec', {}) or {}
        output_info = source_config.get('output', {}) or {}
        model_info = source_config.get('model', {}) or {}
        normalize = preprocessing.get('normalize', {}) or {}

        # ---- 预处理"族" 决定默认值（P7-2/P7-3）----
        # 语义分割(SegFormer)与实例分割(YOLO-seg)预处理完全不同，必须分开。
        if task_type in ('det', 'sevseg', 'instance_seg'):
            family = 'yolo'
            default_preprocess = 'divide_255'
            default_norm_mean = [0.0, 0.0, 0.0]
            default_norm_std = [1.0, 1.0, 1.0]
            default_resize_mode = 'letterbox'
            default_letterbox = [114, 114, 114]
        elif task_type == 'seg':
            family = 'segformer'
            default_preprocess = 'imagenet'
            default_norm_mean = [0.485, 0.456, 0.406]
            default_norm_std = [0.229, 0.224, 0.225]
            default_resize_mode = 'stretch'   # SegFormer 直接 resize，无 letterbox
            default_letterbox = None
        elif task_type == 'cls':
            family = 'classification'
            default_preprocess = 'imagenet'
            default_norm_mean = [0.485, 0.456, 0.406]
            default_norm_std = [0.229, 0.224, 0.225]
            default_resize_mode = 'stretch'
            default_letterbox = None
        else:  # anomaly / ocr / 其它
            family = 'generic'
            default_preprocess = 'divide_255'
            default_norm_mean = [0.0, 0.0, 0.0]
            default_norm_std = [1.0, 1.0, 1.0]
            default_resize_mode = 'stretch'
            default_letterbox = None

        # ---- 实际取值：优先用保存的真实值(input_spec / preprocessing)，否则用族默认 ----
        def _first(*vals, fallback=None):
            for v in vals:
                if v is not None and v != [] and v != {}:
                    return v
            return fallback

        input_shape = _first(
            model_info.get('input_shape'),
            input_spec.get('shape'),
            preprocessing.get('input_shape'),
            source_config.get('input_shape'),
            fallback=[1, 3, 640, 640],
        )

        # [P7-7] 归一化方式取值统一化
        raw_method = _first(
            input_spec.get('normalize_method'),
            preprocessing.get('normalize_method'),
            fallback=default_preprocess,
        )
        normalize_method = self._normalize_method_vocab(raw_method, default_preprocess)

        normalize_mean = _first(
            input_spec.get('normalize_mean'),
            normalize.get('mean'),
            fallback=default_norm_mean,
        )
        normalize_std = _first(
            input_spec.get('normalize_std'),
            normalize.get('std'),
            fallback=default_norm_std,
        )
        color_format = _first(
            input_spec.get('color_format'),
            preprocessing.get('channel_order'),
            fallback='RGB',
        )
        pixel_range = _first(
            input_spec.get('pixel_range'),
            input_spec.get('value_range'),
            preprocessing.get('value_range'),
            fallback=[0, 255],
        )
        # letterbox 取值：保存值优先，否则族默认
        letterbox_color = _first(
            input_spec.get('letterbox_color'),
            preprocessing.get('letterbox_color'),
            fallback=default_letterbox,
        )
        resize_mode = _first(
            input_spec.get('resize_mode'),
            preprocessing.get('resize_mode'),
            fallback=default_resize_mode,
        )

        num_classes = _first(
            output_info.get('num_classes'),
            source_config.get('num_classes'),
        )
        class_names = _first(
            output_info.get('class_names'),
            source_config.get('class_names'),
            fallback=[],
        )

        # ---- 后处理参数（P7-4）----
        postprocess = self._build_postprocess(task_type, source_config, num_classes)

        input_block = {
            'shape': input_shape,
            'dtype': 'float32',
            'color_format': color_format,
            'pixel_range': pixel_range,
            # 预处理族字段（新）
            'preprocess': normalize_method,        # imagenet / divide_255 / none
            'resize_mode': resize_mode,            # letterbox / stretch / shortest_side
            'normalize_method': normalize_method,  # 兼容旧字段名（与 preprocess 同值）
            'normalize_mean': normalize_mean,
            'normalize_std': normalize_std,
        }
        if letterbox_color:
            input_block['letterbox_color'] = letterbox_color
            input_block['letterbox_stride'] = _first(
                input_spec.get('letterbox_stride'),
                fallback=32,
            )

        config = {
            'dlhub_version': self.DLHUB_VERSION,
            'task_type': task_type,
            'task_family': family,           # 新增：预处理/部署族
            'backend': backend,
            'model_name': model_name,
            'precision': precision,
            'model_files': [f"model/{f}" for f in model_files],
            'input': input_block,
            'output': {
                'num_classes': num_classes,
                'class_names': class_names,
                'postprocess': postprocess,   # [P7-4] 后处理参数
            },
            'dynamic_batch': source_config.get('dynamic_batch', False),
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'source_framework': source_config.get('framework', 'ultralytics'),
            },
        }

        # [P7-5] 可选附带训练参数供追溯（不影响部署）
        train_args = source_config.get('train_args') or source_config.get('training')
        if train_args:
            config['train_args'] = train_args

        return config

    @staticmethod
    def _normalize_method_vocab(method, default):
        """[P7-7] 把各种历史写法归一到统一取值: imagenet / divide_255 / none。"""
        if not method:
            return default
        m = str(method).strip().lower()
        if m in ('imagenet', 'imagenet_pixel', 'imagenet_norm', 'standard'):
            return 'imagenet'
        if m in ('divide_255', 'div255', 'divide255', 'scale_255', '0_1', '0-1', 'rescale'):
            return 'divide_255'
        if m in ('none', 'raw', 'identity', ''):
            return 'none'
        return default

    @staticmethod
    def _build_postprocess(task_type, source_config, num_classes) -> Dict[str, Any]:
        """[P7-4] 按任务构建后处理参数。优先读 source_config 中已保存的值，否则给合理默认。"""
        saved = source_config.get('postprocess', {}) or {}

        def g(key, default):
            return saved.get(key, source_config.get(key, default))

        if task_type in ('det', 'sevseg', 'instance_seg'):
            pp = {
                'conf_threshold': g('conf_threshold', g('conf', 0.25)),
                'iou_threshold': g('iou_threshold', g('iou', 0.45)),
                'max_det': g('max_det', 300),
                'end2end': bool(g('end2end', False)),
                'output_layout': g('output_layout', 'xywh_conf_cls'),
            }
            if task_type == 'instance_seg':
                # 实例分割额外的 mask 后处理
                pp['mask_threshold'] = g('mask_threshold', 0.5)
                pp['proto_shape'] = g('proto_shape', [1, 32, 160, 160])
                pp['output_layout'] = g('output_layout', 'detect+proto')
                # [v15] 显式写出 proto 与模型输入的下采样比例, 部署端无需靠约定(proto×4)推断。
                # 标准 YOLO-seg proto 边长 = 输入边长 / 4, 即 mask_ratio=4 / proto_stride=4。
                pp['mask_ratio'] = g('mask_ratio', 4)
                pp['proto_stride'] = g('proto_stride', g('mask_ratio', 4))
            if task_type == 'sevseg':
                # [v15.1] SevSeg = 检测 + severity 评分 + 零标注近似分割 (MaskGeneratorV3)。
                # 近似分割不是模型直接输出, 而是基于 backbone 多尺度特征图 (P3/P4/P5) 的
                # 特征级算法。部署端要复现它, 必须:
                #   1) ONNX 把 P3/P4/P5 特征图作为额外输出导出 (det_output + feat_p3/p4/p5);
                #   2) 部署端 engine.infer(return_all_outputs=True) 取出特征图;
                #   3) 部署端用 MaskGeneratorV3 逐框生成精细掩码。
                # 这里把近似分割的算法标识与全部超参显式写进 deploy_config, 供部署端 1:1 复现。
                #
                # 注: PyTorch 版 MaskGeneratorV3 还用了 layer2 特征, 但导出链路 (export.py)
                # 仅导出 P3/P4/P5 (layer2 通道多、压缩收益低), 故部署端为"仅 P3/P4/P5"降级版,
                # 与训练侧的细微差异已知且可接受。feature_layout 显式声明导出了哪些特征。
                ms = saved.get('mask_seg', {}) or {}

                def mg(key, default):
                    return ms.get(key, source_config.get(key, default))

                pp['severity_scale'] = g('severity_scale', 10.0)
                pp['approx_seg'] = {
                    'enabled': bool(mg('enabled', True)),
                    'method': 'MaskGeneratorV3',
                    # 特征来源: 导出的 ONNX 额外输出节点名 (det_output 之外)
                    'feature_layout': mg('feature_layout', ['feat_p3', 'feat_p4', 'feat_p5']),
                    # 特征图相对模型输入的下采样步长 (P3=8, P4=16, P5=32)
                    'feature_strides': mg('feature_strides', [8, 16, 32]),
                    # 是否对导出特征做了 PCA 压缩 (export.py 用 1x1 Conv 嵌入 PCA), 及压缩维度
                    'pca_compressed': bool(mg('pca_compressed', False)),
                    'pca_dim': mg('pca_dim', None),
                    # ── MaskGeneratorV3 超参 (与 model.py 中 v3 实例化参数一致) ──
                    'topk_channels': mg('topk_channels', 48),
                    'channel_select': mg('channel_select', 'bimodal'),   # V3 默认双峰
                    'channel_weight': mg('channel_weight', 'equal'),
                    'normalize_mode': mg('normalize_mode', 'minmax'),
                    'upsample_guide': mg('upsample_guide', 'canny'),
                    'gradient_snap': bool(mg('gradient_snap', False)),
                    'adaptive_block': mg('adaptive_block', 15),
                    'adaptive_C': mg('adaptive_C', -5),
                    'morph_close_k': mg('morph_close_k', 7),
                    'morph_open_k': mg('morph_open_k', 3),
                    'min_adaptive_size': mg('min_adaptive_size', 15),
                    'channel_threshold': mg('channel_threshold', 128),
                    'canny_low': mg('canny_low', 50),
                    'canny_high': mg('canny_high', 150),
                    'edge_dilate_k': mg('edge_dilate_k', 3),
                    'smooth_ksize': mg('smooth_ksize', 5),
                    'smooth_sigma': mg('smooth_sigma', 2.0),
                }
            return pp
        elif task_type == 'seg':
            return {
                'argmax_axis': g('argmax_axis', 1),
                'ignore_index': g('ignore_index', 255),
            }
        elif task_type == 'cls':
            return {
                'apply_softmax': bool(g('apply_softmax', True)),
                'topk': g('topk', 5),
            }
        elif task_type == 'anomaly':
            return {
                'threshold': g('threshold', None),  # 异常分数阈值，训练若产出则带上
                'score_type': g('score_type', 'image_level'),
            }
        else:
            return {}

    def _generate_readme(self, model_name, backend, precision, task_type, model_files) -> str:
        """生成人可读的README"""
        files_list = '\n'.join(f"  - {f.name} ({f.stat().st_size / 1024:.0f} KB)" for f in model_files)
        return f"""DL-Hub Model Package
====================
Model:     {model_name}
Task:      {task_type}
Backend:   {backend}
Precision: {precision}
Packed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files:
{files_list}

Usage:
  1. 解包: python -c "from model_conversion.dlhub_packager import DLHubPackager; DLHubPackager().unpack('this_file.dlhub')"
  2. 或直接用 unzip 解压（.dlhub 就是 zip 格式）
  3. 读取 deploy_config.json 获取预处理参数和类别信息
  4. 加载 model/ 目录下的模型文件进行推理
"""


# ==================== 便捷函数 ====================

def pack_model(output_dir: str, task_type: str = None, **kwargs) -> Optional[str]:
    """便捷打包函数"""
    return DLHubPackager().pack(output_dir, task_type=task_type, **kwargs)


def unpack_model(dlhub_path: str, extract_dir: str = None) -> Optional[Dict[str, Any]]:
    """便捷解包函数"""
    return DLHubPackager().unpack(dlhub_path, extract_dir)


# ==================== CLI ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DL-Hub Model Packager')
    sub = parser.add_subparsers(dest='command')

    # pack
    pack_parser = sub.add_parser('pack', help='打包模型为 .dlhub')
    pack_parser.add_argument('dir', help='模型目录')
    pack_parser.add_argument('--task', default=None, help='任务类型')
    pack_parser.add_argument('--backend', default=None, help='后端')
    pack_parser.add_argument('--name', default=None, help='模型名称')
    pack_parser.add_argument('--precision', default=None, help='精度')

    # unpack
    unpack_parser = sub.add_parser('unpack', help='解包 .dlhub')
    unpack_parser.add_argument('file', help='.dlhub 文件路径')
    unpack_parser.add_argument('--output', default=None, help='解包目录')

    args = parser.parse_args()

    if args.command == 'pack':
        result = pack_model(args.dir, task_type=args.task, backend=args.backend,
                           model_name=args.name, precision=args.precision)
        if result:
            print(f"\n✅ 打包完成: {result}")
    elif args.command == 'unpack':
        config = unpack_model(args.file, args.output)
        if config:
            print(f"\n✅ 解包完成")
            print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
