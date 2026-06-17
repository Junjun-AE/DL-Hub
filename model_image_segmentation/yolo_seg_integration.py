"""
YOLO-seg 集成模块 (方案 X)
==========================
把 YOLO 实例分割的「训练 / 转换 / 推理」逻辑集中在此处，作为与现有 SegFormer
路径并存的第二套实现。app.py 仅需:
  1. import 本模块的 handler
  2. 在训练 Tab 加一个「分割类型」radio + 一组 YOLO-seg 专属 UI 组件 (默认隐藏)
  3. radio 切换两套 UI 组件的显隐
  4. start 按钮按 radio 分发到本模块或原 SegFormer handler

本模块复用 app.py 传入的共享对象 (state / conversion_state / dlhub_params)，
因此监控 Timer、日志、进度条等基础设施全部沿用，无需重写。

SegFormer 现有逻辑完全不受影响。
"""

import os
import sys
import json
import time
import threading
import subprocess
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------- #
# 这些引用由 app.py 在 import 后通过 bind() 注入，避免循环 import
# ---------------------------------------------------------------------------- #
_ctx = {
    'state': None,             # TrainingState 实例
    'conversion_state': None,  # ConversionState 实例
    'dlhub_params': None,      # DLHubParams 实例或 None
    'get_output_dir': None,    # callable -> Path
    'parse_gpu_choice': None,  # callable(str)->(device_str, gpu_ids)
}


def bind(state, conversion_state, dlhub_params, get_output_dir, parse_gpu_choice):
    """由 app.py 调用，注入共享对象。"""
    _ctx['state'] = state
    _ctx['conversion_state'] = conversion_state
    _ctx['dlhub_params'] = dlhub_params
    _ctx['get_output_dir'] = get_output_dir
    _ctx['parse_gpu_choice'] = parse_gpu_choice


def _state():
    return _ctx['state']


def _conv_state():
    return _ctx['conversion_state']


def _params():
    return _ctx['dlhub_params']


def _output_dir() -> Path:
    fn = _ctx['get_output_dir']
    return fn() if fn else Path('./output')


# ---------------------------------------------------------------------------- #
# 模型类型探测: 判断一个 .pt 是 YOLO-seg 还是 SegFormer
# ---------------------------------------------------------------------------- #
def is_yolo_seg_model(model_path: str) -> bool:
    """
    判断给定模型路径是否为 YOLO-seg 实例分割模型。

    判定优先级:
    1. 文件名含 'yolo' 且含 '-seg' / 'seg'
    2. checkpoint['custom_metadata']['task'] == 'segment'
    3. checkpoint 顶层含 ultralytics 特征 (train_args / task=='segment')
    """
    if not model_path:
        return False
    name = os.path.basename(str(model_path)).lower()
    if 'yolo' in name and 'seg' in name:
        return True
    if not os.path.exists(model_path):
        return False
    if not str(model_path).endswith('.pt'):
        return False
    try:
        import torch
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            meta = ckpt.get('custom_metadata', {})
            if isinstance(meta, dict) and meta.get('task') == 'segment':
                return True
            # ultralytics checkpoint 常含 'train_args' / 'model' 且 task 字段
            ta = ckpt.get('train_args', {})
            if isinstance(ta, dict) and ta.get('task') == 'segment':
                return True
            if str(ckpt.get('task', '')).lower() == 'segment':
                return True
    except Exception:
        pass
    return False


# ============================================================================ #
# 训练
# ============================================================================ #
def start_training(
    images_dir: str,
    jsons_dir: str,
    num_classes,
    family: str,
    model_scale: str,
    epochs,
    batch_size,
    img_size,
    learning_rate,
    val_split,
    include_negative: bool,
    patience,
    save_period,
    optimizer: str,
    weight_decay,
    momentum,
    warmup_epochs,
    cos_lr: bool,
    mosaic,
    mixup,
    fliplr,
    freeze_backbone: bool,
    device_choice: str,
):
    """YOLO-seg 训练入口 (对应 SegFormer 的 on_start_training)。

    Returns: (status_text, start_btn_update, stop_btn_update) — 与原版一致。
    """
    import gradio as gr
    st = _state()

    if st.is_training:
        return "⚠️ 已有训练正在进行中", gr.update(interactive=False), gr.update(interactive=True)

    if not images_dir or not jsons_dir:
        return "❌ 请填写图像和JSON文件夹路径", gr.update(interactive=True), gr.update(interactive=False)

    st.reset()
    st.is_training = True
    st.total_epochs = int(epochs)
    st.start_time = time.time()

    thread = threading.Thread(
        target=_run_training_thread,
        args=(images_dir, jsons_dir, num_classes, family, model_scale, epochs,
              batch_size, img_size, learning_rate, val_split, include_negative,
              patience, save_period, optimizer, weight_decay, momentum,
              warmup_epochs, cos_lr, mosaic, mixup, fliplr, freeze_backbone, device_choice),
        daemon=True,
    )
    thread.start()
    return ("🚀 YOLO-seg 训练已启动，请查看下方监控...",
            gr.update(interactive=False), gr.update(interactive=True))


def _run_training_thread(
    images_dir, jsons_dir, num_classes, family, model_scale, epochs,
    batch_size, img_size, learning_rate, val_split, include_negative,
    patience, save_period, optimizer, weight_decay, momentum,
    warmup_epochs, cos_lr, mosaic, mixup, fliplr, freeze_backbone, device_choice,
):
    st = _state()
    params = _params()
    try:
        from config.yolo_seg_registry import get_model_config
        from data.yolo_seg_converter import LabelMeToYOLOSegConverter
        from engine.yolo_seg_trainer import YOLOSegTrainer, TrainingCallback

        st.add_log("=" * 50)
        st.add_log("🚀 开始 YOLO-seg 实例分割训练")
        st.add_log("=" * 50)

        # 保存参数 (合并写入, 不覆盖其他 section)
        if params:
            cur = params.get_all()
            cur['data'] = {
                'images_dir': images_dir, 'jsons_dir': jsons_dir,
                'image_size': img_size, 'val_split': val_split,
                'include_negative': include_negative, 'num_classes': num_classes,
            }
            cur['model'] = {
                'seg_type': 'instance', 'family': family, 'scale': model_scale,
                'freeze_backbone': freeze_backbone,
            }
            cur['training_yoloseg'] = {
                'batch_size': batch_size, 'epochs': epochs,
                'learning_rate': learning_rate, 'optimizer': optimizer,
                'patience': patience, 'save_period': save_period,
                'weight_decay': weight_decay, 'momentum': momentum,
                'warmup_epochs': warmup_epochs, 'cos_lr': cos_lr,
            }
            cur['augmentation_yoloseg'] = {
                'mosaic': mosaic, 'mixup': mixup, 'fliplr': fliplr,
            }
            cur['device'] = device_choice
            params.save(cur)

        # 设备
        parse_gpu = _ctx['parse_gpu_choice']
        device_str, gpu_ids = parse_gpu(device_choice)
        device = device_str if device_str != 'cpu' else 'cpu'
        st.add_log(f"🖥️ 设备: {device_choice} -> {'cuda:'+device if device != 'cpu' else 'cpu'}")

        # 模型配置
        config = get_model_config(family, model_scale)
        if config is None:
            st.add_log(f"❌ 无效的模型: {family}/{model_scale}")
            st.is_training = False
            return
        model_name = config['ultralytics_name']
        st.add_log(f"🧠 模型: {model_name}")

        # 输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = _output_dir()
        output_dir = output_base / f"{config['name']}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        st.output_dir = str(output_dir)
        st.add_log(f"📁 输出目录: {output_dir}")

        # 清历史
        if params:
            params.clear_history(auto_save=False)
            params.clear_logs('training', auto_save=False)
            params.save_history({
                'total_epochs': epochs, 'current_epoch': 0,
                'output_dir': str(output_dir), 'completed': False,
                'seg_type': 'instance',
            }, auto_save=True)

        # 数据转换 (polygon -> YOLO-seg TXT)
        st.add_log("📦 开始数据转换 (polygon -> YOLO-seg)...")
        dataset_dir = output_dir / 'dataset'
        converter = LabelMeToYOLOSegConverter(
            images_dir=images_dir.strip(), jsons_dir=jsons_dir.strip(),
            output_dir=str(dataset_dir), val_split=float(val_split),
            num_classes=int(num_classes), include_negative_samples=include_negative,
        )
        result = converter.convert()
        if not result.success:
            st.add_log(f"❌ 数据转换失败: {result.message}")
            st.is_training = False
            return
        st.add_log(f"✅ 数据转换成功:")
        st.add_log(f"   训练集: {result.train_images} 张 | 验证集: {result.val_images} 张")
        st.add_log(f"   类别数: {result.num_classes} | 实例(多边形)数: {result.total_polygons}")
        st.add_log(f"   负样本: {result.negative_samples} 张")

        st.dataset_dir = str(dataset_dir)
        st.class_names = result.class_names

        # 回调 (复用 SegFormer 的 state 字段: train_losses / val_mIoU 复用为 mask mAP)
        def on_epoch_end(epoch: int, m: Dict):
            st.current_epoch = epoch
            # seg loss 累加为 train_losses (box+seg+cls)
            total_loss = m.get('box_loss', 0) + m.get('seg_loss', 0) + m.get('cls_loss', 0)
            st.train_losses.append(float(total_loss))
            # 复用 val_mIoU 曲线显示 mask mAP50-95, val_mDice 显示 mask mAP50
            st.val_mIoU.append(float(m.get('mask_map50_95', 0)))
            st.val_mDice.append(float(m.get('mask_map50', 0)))
            st.best_mIoU = float(m.get('best_mask_map50_95', 0))
            st.best_epoch = int(m.get('best_epoch', epoch))
            if params:
                params.update_history_epoch({
                    'current_epoch': epoch,
                    'train_loss': float(total_loss),
                    'mIoU': float(m.get('mask_map50_95', 0)),   # 复用字段名
                    'mDice': float(m.get('mask_map50', 0)),
                    'best_mIoU': float(m.get('best_mask_map50_95', 0)),
                    'best_epoch': int(m.get('best_epoch', epoch)),
                }, auto_save=(epoch % 5 == 0))

        def on_log(msg: str):
            st.add_log(msg)
            if params:
                params.append_log(msg, 'training', auto_save=False)

        def should_stop():
            return st.should_stop

        callback = TrainingCallback(
            on_epoch_end=on_epoch_end, on_log=on_log, should_stop=should_stop,
        )

        st.add_log("🔧 创建 YOLO-seg 训练器...")
        trainer = YOLOSegTrainer(
            model_name=model_name,
            data_yaml=result.data_yaml,
            output_dir=str(output_dir),
            epochs=int(epochs), batch_size=int(batch_size), img_size=int(img_size),
            learning_rate=float(learning_rate), optimizer=optimizer,
            patience=int(patience), freeze_backbone=freeze_backbone,
            device=device, weight_decay=float(weight_decay), momentum=float(momentum),
            warmup_epochs=float(warmup_epochs), cos_lr=cos_lr,
            mosaic=float(mosaic), mixup=float(mixup), fliplr=float(fliplr),
            save_period=int(save_period), callback=callback,
            class_names=result.class_names,
        )
        st.trainer = trainer

        st.add_log("━" * 50)
        st.add_log("🚀 开始训练")
        st.add_log("━" * 50)
        trainer.train()

        elapsed = time.time() - st.start_time
        st.add_log("━" * 50)
        st.add_log(f"✅ 训练完成！用时 {elapsed/60:.1f} 分钟")
        st.add_log(f"🏆 最佳 mask mAP@50-95: {st.best_mIoU:.4f} (Epoch {st.best_epoch})")
        st.add_log("━" * 50)

        if params:
            params.mark_training_complete(best_metric=st.best_mIoU, best_epoch=st.best_epoch)

    except Exception as e:
        st.add_log(f"❌ 错误: {str(e)}")
        st.add_log(traceback.format_exc())
    finally:
        st.is_training = False
        if params:
            try:
                params.save_logs(st.logs, 'training', auto_save=True)
            except Exception:
                pass


# ============================================================================ #
# 转换 (YOLO-seg -> ONNX/TensorRT/OpenVINO via yolo_export_worker)
# ============================================================================ #
def start_conversion(
    model_path: str,
    target_backend: str,
    precision: str,
    device: str,
    workspace_gb,
    dynamic_batch: bool,
    min_batch, opt_batch, max_batch,
    output_dir_in: str,
    end2end: bool,
    calib_images_dir: str = "",
):
    """YOLO-seg 转换入口。Returns (status, logs)。"""
    cs = _conv_state()
    if cs.is_running:
        return "⚠️ 转换正在进行中...", "\n".join(cs.logs[-50:])
    if not model_path:
        return "❌ 请选择要转换的模型", ""
    if not os.path.exists(model_path):
        return f"❌ 模型文件不存在: {model_path}", ""

    cs.reset()
    cs.is_running = True
    cs.start_time = time.time()

    if not output_dir_in:
        output_dir_in = str(Path(model_path).parent.parent / "converted")

    thread = threading.Thread(
        target=_run_conversion_thread,
        args=(model_path, target_backend, precision, device, workspace_gb,
              dynamic_batch, min_batch, opt_batch, max_batch, output_dir_in,
              end2end, calib_images_dir),
        daemon=True,
    )
    thread.start()
    return "🚀 YOLO-seg 转换已启动...", ""


def _run_conversion_thread(
    model_path, target_backend, precision, device, workspace_gb,
    dynamic_batch, min_batch, opt_batch, max_batch, output_dir_in,
    end2end, calib_images_dir,
):
    cs = _conv_state()
    conversion_tool_path = str(Path(__file__).parent.parent / "model_conversion")

    backend_dir_map = {
        'tensorrt': 'tensorrt', 'trt': 'tensorrt',
        'openvino': 'openvino', 'ov': 'openvino',
        'ort': 'onnxruntime', 'onnxruntime': 'onnxruntime',
    }
    backend_subdir = backend_dir_map.get(target_backend.lower(), target_backend.lower())
    date_subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(output_dir_in) / backend_subdir / date_subdir)

    ctx = None
    try:
        cs.add_log("━" * 50)
        cs.add_log("🚀 开始 YOLO-seg 模型转换")
        cs.add_log("━" * 50)
        cs.add_log(f"📂 模型: {model_path}")
        cs.add_log(f"🎯 目标: {target_backend} ({precision})")
        cs.add_log(f"📁 输出: {output_dir}")
        cs.add_log(f"⚙️ end2end: {end2end} (YOLO26 NMS-free 开关)")

        os.makedirs(output_dir, exist_ok=True)

        model_stem = Path(model_path).stem
        onnx_path = str(Path(output_dir) / f"{model_stem}.onnx")

        # ---- Stage 4: ONNX 导出 (子进程 worker) ----
        cs.add_log("\n📦 Stage 4: 导出 ONNX (YOLO-seg)...")
        # seg + fp16 时, 为避免 yolo26 end2end+half 的 FP32 混精坑, 强制 half=False;
        # 量化交给 Stage 5 后端处理 (TRT/OV 的 fp16/int8)
        export_half = False
        export_config = {
            'model_path': model_path,
            'output_path': onnx_path,
            'opset': 17,
            'dynamic_batch': bool(dynamic_batch),
            'simplify': False,
            'task': 'segment',
            'end2end': bool(end2end),
            'half': export_half,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(export_config, f)
            config_path = f.name

        worker_script = os.path.join(conversion_tool_path, 'yolo_export_worker.py')
        export_success = False
        if os.path.exists(worker_script):
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            try:
                result = subprocess.run(
                    [sys.executable, worker_script, config_path],
                    capture_output=True, text=True, encoding='utf-8',
                    errors='replace', timeout=600, env=env,
                )
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            print(f"  {line}", flush=True)
                result_path = config_path + '.result'
                if os.path.exists(result_path):
                    with open(result_path, 'r', encoding='utf-8') as f:
                        export_result = json.load(f)
                    if export_result.get('success'):
                        onnx_path = export_result.get('output_path', onnx_path)
                        export_success = True
                        cs.add_log("   ✅ 子进程导出成功")
                    else:
                        cs.add_log(f"   ❌ 导出失败: {export_result.get('message', '')[:200]}")
                    try:
                        os.unlink(config_path)
                        os.unlink(result_path)
                    except Exception:
                        pass
            except subprocess.TimeoutExpired:
                cs.add_log("   ❌ 子进程导出超时(10分钟)")
            except Exception as e:
                cs.add_log(f"   ❌ 子进程导出异常: {e}")
        else:
            cs.add_log("   ❌ 未找到 yolo_export_worker.py")

        if not export_success:
            cs.add_log("❌ ONNX 导出失败")
            return
        cs.add_log(f"   ✅ ONNX: {onnx_path}")

        # 若目标就是 ONNX Runtime, 到此可直接打包
        # 否则进入 Stage 5 调用 model_conversion 的后端转换
        # ---- Stage 5: 后端转换 ----
        if target_backend.lower() in ('onnxruntime', 'ort'):
            cs.add_log("\n✅ 目标为 ONNX Runtime, 直接使用导出的 ONNX")
            final_model = onnx_path
        else:
            cs.add_log(f"\n🔧 Stage 5: ONNX -> {target_backend}...")
            if conversion_tool_path not in sys.path:
                sys.path.insert(0, conversion_tool_path)
            # 强制重载转换模块, 避免状态污染 (与 SegFormer 转换一致的策略)
            for mod_name in list(sys.modules.keys()):
                if any(t in mod_name for t in [
                    'model_converter', 'converter_tensorrt', 'converter_openvino',
                    'converter_ort', 'conversion_validator', 'cuda_utils',
                ]):
                    try:
                        del sys.modules[mod_name]
                    except Exception:
                        pass
            try:
                from model_converter import ConversionConfig, ModelConverter, TargetBackend, PrecisionMode
            except Exception as e:
                cs.add_log(f"   ❌ 导入转换模块失败: {e}")
                return

            backend_enum = {
                'tensorrt': TargetBackend.TENSORRT,
                'openvino': TargetBackend.OPENVINO,
            }.get(target_backend.lower())
            prec_enum = {
                'fp32': PrecisionMode.FP32, 'fp16': PrecisionMode.FP16,
                'int8': PrecisionMode.INT8, 'mixed': PrecisionMode.MIXED,
            }.get(precision.lower(), PrecisionMode.FP16)

            conv_config = ConversionConfig(
                target_backend=backend_enum,
                precision_mode=prec_enum,
                enable_validation=False,  # 禁用验证, 避免 OpenVINO 导入崩溃 (与 SegFormer/检测一致)
            )
            # 动态 batch (TRT 的 profile 仅设输入, seg 双输出自动推导, 已验证)
            if target_backend.lower() == 'tensorrt':
                conv_config.trt_workspace_gb = int(workspace_gb)
                conv_config.trt_dynamic_batch_enabled = bool(dynamic_batch)
                conv_config.trt_min_batch = int(min_batch)
                conv_config.trt_opt_batch = int(opt_batch)
                conv_config.trt_max_batch = int(max_batch)
            elif target_backend.lower() == 'openvino':
                conv_config.ov_dynamic_batch_enabled = bool(dynamic_batch)
                conv_config.ov_min_batch = int(min_batch)
                conv_config.ov_max_batch = int(max_batch)

            if precision.lower() in ('int8', 'mixed') and calib_images_dir:
                conv_config.calib_data_path = calib_images_dir

            try:
                converter = ModelConverter(conv_config)
                ext = '.engine' if target_backend.lower() == 'tensorrt' else '_openvino'
                out_model = str(Path(output_dir) / f"{model_stem}{ext}")
                conv_result = converter.convert(
                    onnx_path=onnx_path,
                    output_path=out_model,
                )
                if not (conv_result and getattr(conv_result, 'success', False)):
                    msg = getattr(conv_result, 'message', '未知错误') if conv_result else '无返回'
                    cs.add_log(f"   ❌ 后端转换失败: {msg}")
                    return
                out_files = getattr(conv_result, 'output_files', None) or [out_model]
                final_model = out_files[0]
                cs.add_log(f"   ✅ 转换完成: {os.path.basename(str(final_model))}")
            except Exception as e:
                cs.add_log(f"   ❌ 后端转换异常: {e}")
                cs.add_log(traceback.format_exc())
                return

        # ---- 打包 .dlhub ----
        cs.add_log("\n📦 打包 .dlhub...")
        try:
            if conversion_tool_path not in sys.path:
                sys.path.insert(0, conversion_tool_path)
            from dlhub_packager import DLHubPackager
            packager = DLHubPackager()
            # [P7-4] 传入真实后处理参数（尤其 end2end），让 deploy_config 的 postprocess 段准确。
            dlhub_path = packager.pack(
                output_dir=output_dir,
                task_type='instance_seg',
                backend=target_backend,
                precision=precision,
                deploy_config={
                    'postprocess': {
                        'end2end': bool(end2end),
                        'conf_threshold': 0.25,
                        'iou_threshold': 0.45,
                        'max_det': 300,
                        'mask_threshold': 0.5,
                        'output_layout': 'detect+proto',
                    }
                },
            )
            if dlhub_path:
                cs.add_log(f"   ✅ 已打包: {os.path.basename(dlhub_path)}")
            else:
                cs.add_log("   ⚠️ 打包跳过")
        except Exception as e:
            cs.add_log(f"   ⚠️ 打包失败(非致命): {e}")

        cs.output_path = output_dir
        elapsed = time.time() - cs.start_time
        cs.add_log("\n" + "━" * 50)
        cs.add_log(f"✅ 转换完成！用时 {elapsed:.1f} 秒")
        cs.add_log(f"📁 输出目录: {output_dir}")
        cs.add_log("━" * 50)

    except Exception as e:
        cs.add_log(f"\n❌ 错误: {str(e)}")
        cs.add_log(traceback.format_exc())
    finally:
        cs.is_running = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        params = _params()
        if params:
            try:
                params.save_logs(cs.logs, 'conversion', auto_save=True)
            except Exception:
                pass


# ============================================================================ #
# 推理 (YOLO-seg 实例分割: 框 + mask + 置信度, 保存 JSON)
# ============================================================================ #
def run_batch_inference(
    model_path: str,
    images_dir: str,
    output_dir: str,
    device: str,
    conf_threshold: float,
    save_images: bool = True,
    save_json: bool = True,
) -> str:
    """YOLO-seg 批量推理。返回结果文本摘要。"""
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        logs = []
        def log(m):
            logs.append(m)
            print(f"[YOLO-seg infer] {m}", flush=True)

        if not model_path or not os.path.exists(model_path):
            return f"❌ 模型不存在: {model_path}"
        if not images_dir or not os.path.isdir(images_dir):
            return f"❌ 图像文件夹不存在: {images_dir}"

        use_cuda = ('GPU' in device or 'cuda' in device.lower()) and torch.cuda.is_available()
        dev = '0' if use_cuda else 'cpu'
        log(f"🖥️ 设备: {'GPU' if use_cuda else 'CPU'}")

        # 类别名
        class_names = []
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            meta = ckpt.get('custom_metadata', {}) if isinstance(ckpt, dict) else {}
            if isinstance(meta, dict) and meta.get('class_names'):
                class_names = meta['class_names']
        except Exception:
            pass

        model = YOLO(model_path, task='segment')
        log("✅ 模型加载成功")

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        images = sorted([p for p in Path(images_dir).rglob('*') if p.suffix.lower() in exts])
        if not images:
            return "❌ 未找到图片"
        log(f"📂 找到 {len(images)} 张图片")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        img_out = out / 'images'
        json_out = out / 'jsons'
        if save_images:
            img_out.mkdir(exist_ok=True)
        if save_json:
            json_out.mkdir(exist_ok=True)

        total_instances = 0
        results_summary = []
        for i, img_path in enumerate(images):
            try:
                results = model.predict(source=str(img_path), conf=float(conf_threshold),
                                        device=dev, save=False, verbose=False)
                if not results:
                    continue
                r = results[0]
                instances = []
                if r.masks is not None and r.boxes is not None:
                    polys = r.masks.xy   # 像素坐标多边形 list[np.ndarray(K,2)]
                    boxes = r.boxes
                    for j in range(len(boxes)):
                        cls_id = int(boxes.cls[j].cpu().numpy())
                        conf = float(boxes.conf[j].cpu().numpy())
                        box = boxes.xyxy[j].cpu().numpy().tolist()
                        poly = polys[j].tolist() if j < len(polys) else []
                        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                        instances.append({
                            'class_id': cls_id, 'class_name': cls_name,
                            'confidence': round(conf, 4),
                            'bbox': [round(float(b), 2) for b in box],
                            'polygon': [[round(float(x), 2), round(float(y), 2)] for x, y in poly],
                        })
                total_instances += len(instances)
                results_summary.append({
                    'image': img_path.name, 'num_instances': len(instances),
                    'instances': instances,
                })

                if save_images:
                    annotated = r.plot()   # 自动叠加 mask + 框 + 标签
                    cv2.imwrite(str(img_out / img_path.name), annotated)
                if save_json:
                    with open(json_out / f"{img_path.stem}.json", 'w', encoding='utf-8') as f:
                        json.dump({'image': img_path.name, 'instances': instances},
                                  f, ensure_ascii=False, indent=2)

                if (i + 1) % 10 == 0 or (i + 1) == len(images):
                    log(f"   进度: {i+1}/{len(images)}")
            except Exception as e:
                log(f"   ⚠️ 跳过 {img_path.name}: {e}")

        # 汇总
        summary_path = out / 'instance_seg_results.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_images': len(images),
                'total_instances': total_instances,
                'results': results_summary,
            }, f, ensure_ascii=False, indent=2)

        log(f"\n📊 结果已保存: {summary_path}")
        log(f"📈 总图片: {len(images)} | 总实例: {total_instances} | "
            f"平均每张: {total_instances/len(images):.1f}")

        # 把 state 的批量预览目录设上, 供 UI 预览复用
        st = _state()
        st.batch_images_dir = str(images_dir)
        st.batch_output_dir = str(img_out)

        return "\n".join(logs)
    except Exception as e:
        return f"❌ 推理失败: {e}\n{traceback.format_exc()}"


def find_trained_weights(output_dir: str) -> Optional[str]:
    """在训练输出目录内定位 YOLO-seg best.pt。"""
    if not output_dir:
        return None
    base = Path(output_dir)
    candidates = [base / 'weights' / 'best.pt', base / 'weights' / 'last.pt']
    for c in candidates:
        if c.exists():
            return str(c)
    for best in base.rglob('best.pt'):
        return str(best)
    for last in base.rglob('last.pt'):
        return str(last)
    return None


def is_yoloseg_output_dir(output_dir: str) -> bool:
    """判断某训练输出目录是否为 YOLO-seg (依据 model_metadata.json 或 best.pt)。"""
    if not output_dir:
        return False
    meta = Path(output_dir) / 'model_metadata.json'
    if meta.exists():
        try:
            with open(meta, 'r', encoding='utf-8') as f:
                m = json.load(f)
            if m.get('task') == 'segment' or ('best_mask_mAP50_95' in m):
                return True
        except Exception:
            pass
    # 退而求其次: 有 weights/best.pt 且能识别为 YOLO-seg
    w = find_trained_weights(output_dir)
    if w and is_yolo_seg_model(w):
        return True
    return False


def test_inference_preview(conf_threshold: float = 0.25, num_images: int = 4):
    """
    YOLO-seg 验证集测试推理预览 (对应 SegFormer 的 on_test_inference)。
    返回 (PIL.Image 或 None, status_text)。
    """
    st = _state()
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO

        if not st.output_dir:
            return None, "⚠️ 请先完成训练"

        weights = find_trained_weights(st.output_dir)
        if not weights:
            return None, "⚠️ 未找到训练好的模型 (best.pt)"

        dataset_dir = st.dataset_dir
        if not dataset_dir:
            return None, "⚠️ 未找到数据集目录"

        val_img_dir = Path(dataset_dir) / 'images' / 'val'
        if not val_img_dir.exists():
            val_img_dir = Path(dataset_dir) / 'images' / 'train'
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        val_images = sorted([p for p in val_img_dir.glob('*') if p.suffix.lower() in exts])[:num_images]
        if not val_images:
            return None, "⚠️ 验证集中没有图像"

        st.add_log(f"🔍 YOLO-seg 测试推理: {Path(weights).name} (conf={conf_threshold})")
        model = YOLO(weights, task='segment')

        tiles = []
        for img_path in val_images:
            try:
                results = model.predict(source=str(img_path), conf=float(conf_threshold),
                                        save=False, verbose=False)
                if results:
                    annotated = results[0].plot()  # BGR
                    tiles.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            except Exception as e:
                st.add_log(f"   ⚠️ 预览失败 {img_path.name}: {e}")

        if not tiles:
            return None, "⚠️ 推理失败"

        # 拼 2x2 网格
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        for idx, t in enumerate(tiles[:4]):
            r, c = idx // 2, idx % 2
            th, tw = t.shape[:2]
            canvas[r*h:r*h+th, c*w:c*w+tw] = t

        grid = Image.fromarray(canvas)
        st.segmentation_preview = grid
        st.add_log(f"✅ YOLO-seg 测试推理完成: {len(tiles)} 张")
        return grid, f"✅ 实例分割预览完成: {len(tiles)} 张 (conf={conf_threshold})"
    except Exception as e:
        return None, f"❌ 推理失败: {e}\n{traceback.format_exc()}"
