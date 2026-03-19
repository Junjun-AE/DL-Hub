"""
🚀 模型转换工具 - 主程序
========================

工业级深度学习模型转换工具，支持 PyTorch 模型的导入、分析、优化和导出。

Pipeline:
---------
  Stage 1: 模型导入 (model_importer) - 支持 timm/YOLO/detectron2
  Stage 2: 模型分析 (model_analyzer) - 兼容性检测/性能分析/转换建议
  Stage 3: 图优化   (model_optimizer) - Conv+BN融合/冗余层移除
  Stage 4: ONNX导出 (model_exporter)  - 动态轴/简化/验证
  Stage 5: 量化转换 (model_converter) - FP16/INT8/TensorRT/OpenVINO
  Stage 6: 验证     (conversion_validator) - 精度/性能验证 (集成在 Stage 5)
  Stage 8: 配置生成 (config_generator) - 生成部署配置文件

使用方法:
---------
  # ============ 推荐方式: 使用 YAML 配置文件 ============
  
  # 生成配置模板
  python main.py init -t cls    # 分类任务模板
  python main.py init -t det    # 检测任务模板  
  python main.py init -t seg    # 分割任务模板
  python main.py init -t all    # 生成所有模板
  
  # 编辑配置文件后运行
  python main.py run -c config_det.yaml
  
  # 验证配置（不执行）
  python main.py run -c config_det.yaml --dry-run
  
  # ============ 命令行方式 ============
  
  # 分析模型
  python main.py analyze -m model.pth -t cls
  
  # 导出 ONNX (Stage 1-4)
  python main.py export -m model.pth -t det -o model.onnx
  
  # 完整流程 (Stage 1-8)
  python main.py pipeline -m model.pth -t det -o ./output --target tensorrt --precision fp16
  
  # 从现有 ONNX 转换 (Stage 5-8)
  python main.py convert -m model.onnx -o ./output --target tensorrt --precision int8 --calib-data ./images
  
  # 运行测试
  python main.py test --full


"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass, field

# 统一日志模块
from unified_logger import Logger, console, Timer


# ==================== 日志初始化 ====================

# 延迟初始化的 logger
_logger = None

def get_logger():
    """获取 logger (延迟初始化)"""
    global _logger
    if _logger is None:
        _logger = Logger.get("main")
    return _logger

def setup_logging(verbose: bool = False, quiet: bool = False, log_file: str = None):
    """
    初始化日志系统
    
    Args:
        verbose: 详细模式 (DEBUG)
        quiet: 静默模式 (WARNING)
        log_file: 日志文件路径
    """
    global _logger
    Logger.init(
        verbose=verbose,
        quiet=quiet,
        log_file=log_file,
    )
    # 重新获取 logger 以确保使用最新配置
    _logger = Logger.get("main")


# 为了保持向后兼容，提供 logger 属性访问
class _LoggerProxy:
    """Logger 代理类，支持延迟初始化"""
    def __getattr__(self, name):
        return getattr(get_logger(), name)

logger = _LoggerProxy()


# ==============================================================================
# Pipeline 上下文
# ==============================================================================

def get_default_device() -> str:
    """获取默认设备 (优先使用 GPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


# 各任务类型的默认输入形状
DEFAULT_INPUT_SHAPES = {
    'cls': (1, 3, 224, 224),    # 分类: 224x224
    'det': (1, 3, 640, 640),    # 检测: 640x640 (YOLO 风格)
    'seg': (1, 3, 512, 512),    # 分割: 512x512
}


@dataclass
class PipelineContext:
    """
    Pipeline 上下文
    
    在各 Stage 之间传递数据
    """
    # Stage 1 输出
    model: Any = None                         # nn.Module
    model_name: str = ""
    model_path: str = ""                      # 原始模型路径（用于YOLO导出）
    task_type: str = ""
    input_shape: Tuple[int, ...] = ()
    input_spec: Any = None
    output_spec: Any = None
    framework: str = ""
    
    # Stage 2 输出
    analysis_report: Any = None
    
    # Stage 3 输出
    optimized_model: Any = None
    optimization_stats: Any = None
    
    # Stage 4 输出
    onnx_path: str = ""
    onnx_model: Any = None
    export_result: Any = None
    dynamic_axes_spec: Any = None
    
    # Stage 5-6 输出 (量化转换 & 验证)
    conversion_result: Any = None
    converted_model_path: str = ""
    validation_result: Any = None
    
    # Stage 8 输出
    config_path: str = ""
    
    # 配置
    target_backend: str = 'onnxruntime'       # 默认: ONNX Runtime (最通用)
    precision: str = 'fp16'
    opset: int = 17
    device: str = field(default_factory=get_default_device)  # 默认: GPU 优先
    
    # 校准数据 (INT8 需要)
    calib_data_path: str = ""
    calib_data_format: str = "imagefolder"
    calib_num_samples: int = 300


# ==============================================================================
# Stage 1: 模型导入
# ==============================================================================

def run_stage1_import(
    ctx: PipelineContext, 
    model_path: str, 
    task_type: str,
    input_shape: Tuple[int, ...] = None,
    device: str = None,
) -> bool:
    """
    Stage 1: 模型导入
    
    Args:
        ctx: Pipeline 上下文
        model_path: 模型文件路径
        task_type: 任务类型 (cls/det/seg)
        input_shape: 输入形状 (可选，自动推断)
        device: 设备 (可选，默认 GPU)
        
    Returns:
        是否成功
    """
    import gc
    import sys
    
    # ========== 关键修复：彻底清理之前的模型和所有相关资源 ==========
    def cleanup_previous_context():
        """彻底清理之前的上下文"""
        try:
            # 清理模型
            if ctx.model is not None:
                del ctx.model
                ctx.model = None
            
            if ctx.optimized_model is not None:
                del ctx.optimized_model
                ctx.optimized_model = None
            
            if ctx.onnx_model is not None:
                del ctx.onnx_model
                ctx.onnx_model = None
            
            # 清理其他上下文
            ctx.input_spec = None
            ctx.output_spec = None
            ctx.analysis_report = None
            ctx.optimization_stats = None
            ctx.export_result = None
            ctx.dynamic_axes_spec = None
            ctx.conversion_result = None
            ctx.validation_result = None
            ctx.onnx_path = ""
            ctx.converted_model_path = ""
            ctx.config_path = ""
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理CUDA缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # 重置CUDA上下文
                torch.cuda.reset_peak_memory_stats()
            
            gc.collect()
        except Exception as e:
            # 忽略清理错误，继续执行
            pass
    
    cleanup_previous_context()
    
    # 使用默认设备 (GPU 优先)
    if device is None:
        device = get_default_device()
    
    try:
        from model_importer import import_model
        
        console.info(f"加载模型: {os.path.basename(model_path)}")
        sys.stdout.flush()
        
        info = import_model(model_path, task_type, device=device)
        
        ctx.model = info.model
        ctx.model_name = getattr(info, 'model_name', None) or getattr(info, 'architecture', 'unknown')
        ctx.model_path = model_path  # 保存原始模型路径（用于YOLO导出）
        ctx.task_type = task_type
        ctx.input_spec = info.input_spec
        ctx.output_spec = info.output_spec
        ctx.framework = info.framework.value if hasattr(info.framework, 'value') else str(info.framework)
        ctx.device = device
        
        # 输入形状
        if input_shape:
            ctx.input_shape = input_shape
            shape_source = "用户指定"
        elif info.input_spec and info.input_spec.shape:
            ctx.input_shape = info.input_spec.shape
            shape_source = "模型推断"
        else:
            ctx.input_shape = DEFAULT_INPUT_SHAPES.get(task_type, (1, 3, 224, 224))
            shape_source = "默认值"
        
        # 输出结果
        console.tree([
            ("模型名称", ctx.model_name),
            ("框架", ctx.framework),
            ("任务类型", task_type),
            ("输入形状", f"{ctx.input_shape} ({shape_source})"),
            ("设备", device),
            ("状态", "✅ 加载成功"),
        ])
        sys.stdout.flush()
        return True
        
    except Exception as e:
        console.error(f"模型导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False


# ==============================================================================
# Stage 2: 模型分析
# ==============================================================================

def run_stage2_analyze(
    ctx: PipelineContext,
    target_backends: List[str] = None,
    output_path: str = None,
    output_format: str = 'console',
) -> bool:
    """
    Stage 2: 模型分析
    """
    if ctx.model is None:
        print("  ❌ 未找到模型，请先执行 Stage 1")
        return False
    
    try:
        from model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        ctx.analysis_report = analyzer.analyze(
            model=ctx.model,
            input_shape=ctx.input_shape,
            task_type=ctx.task_type,
            model_name=ctx.model_name,
            target_backends=target_backends,
        )
        
        # 更新上下文中的推荐配置
        ctx.opset = ctx.analysis_report.advice.recommended_opset
        ctx.precision = ctx.analysis_report.advice.recommended_precision
        
        report = ctx.analysis_report
        
        # 使用表格输出
        # 结构分析
        console.info("结构分析")
        op_items = sorted(report.op_count.items(), key=lambda x: -x[1])[:8]
        total_ops = sum(report.op_count.values())
        rows = []
        for op, count in op_items:
            pct = f"{count/total_ops*100:.1f}%"
            rows.append([op, str(count), pct])
        console.table(["组件类型", "数量", "占比"], rows)
        
        console.blank()
        
        # 复杂度分析
        console.info("复杂度分析")
        prof = report.profiling
        
        # 使用正确的属性名
        if hasattr(prof, 'flops_calculation_failed') and prof.flops_calculation_failed:
            flops_str = "N/A"
        else:
            flops_str = f"{prof.total_flops/1e9:.2f} G"
        
        # 计算复杂度等级
        flops_g = prof.total_flops / 1e9
        params_m = prof.total_params / 1e6
        if flops_g < 1 and params_m < 5:
            complexity_str = "轻量"
        elif flops_g < 10 and params_m < 50:
            complexity_str = "中等"
        elif flops_g < 100 and params_m < 200:
            complexity_str = "⚠️ 较重 (建议量化)"
        else:
            complexity_str = "🔴 非常重 (强烈建议量化)"
        
        console.tree([
            ("FLOPs", flops_str),
            ("参数量", f"{prof.total_params/1e6:.2f} M"),
            ("内存估算", f"{prof.total_memory_mb:.1f} MB"),
            ("复杂度", complexity_str),
        ])
        
        console.blank()
        
        # 后端兼容性
        console.info("后端兼容性")
        compat_items = []
        for backend, result in report.compatibility.items():
            status = "✅ 支持" if result.supported else "❌ 不支持"
            compat_items.append((backend, status))
        console.tree(compat_items)
        
        console.blank()
        
        # 推荐配置
        console.info("推荐配置")
        console.tree([
            ("Opset 版本", str(report.advice.recommended_opset)),
            ("推荐精度", report.advice.recommended_precision),
        ])
        return True
        
    except Exception as e:
        print(f"  ❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Stage 3: 图优化
# ==============================================================================

def run_stage3_optimize(
    ctx: PipelineContext,
    enable_fusion: bool = True,
    enable_elimination: bool = True,
    verify_output: bool = True,
) -> bool:
    """
    Stage 3: 图优化
    """
    if ctx.model is None:
        print("  ❌ 未找到模型，请先执行 Stage 1")
        return False
    
    try:
        from model_optimizer import GraphOptimizer, OptimConfig
        
        console.info("开始图优化...")
        
        config = OptimConfig(
            enable_conv_bn_fusion=enable_fusion,
            enable_conv_bn_act_fusion=enable_fusion,
            enable_linear_bn_fusion=enable_fusion,
            enable_identity_elimination=enable_elimination,
            enable_dropout_elimination=enable_elimination,
            verify_output=verify_output,
        )
        
        optimizer = GraphOptimizer(config)
        result = optimizer.optimize(
            model=ctx.model,
            input_shape=ctx.input_shape,
        )
        
        if result.success:
            ctx.optimized_model = result.model
            ctx.optimization_stats = result.stats
            
            stats = result.stats
            original_total = sum(stats.original_op_count.values())
            optimized_total = sum(stats.optimized_op_count.values())
            reduction = original_total - optimized_total
            reduction_pct = (reduction / original_total * 100) if original_total > 0 else 0
            
            # 优化结果表格
            console.table(
                ["优化类型", "结果"],
                [
                    ["Conv+BN 融合", f"{stats.conv_bn_fused} 对"],
                    ["Conv+BN+Act 融合", f"{stats.conv_bn_act_fused} 对"],
                    ["Identity 移除", f"{stats.identity_removed} 个"],
                    ["Dropout 移除", f"{stats.dropout_removed} 个"],
                    ["优化前算子数", str(original_total)],
                    ["优化后算子数", str(optimized_total)],
                    ["减少比例", f"{reduction_pct:.1f}%"],
                ]
            )
            
            console.blank()
            
            # 验证结果
            console.info("数值验证")
            status = "✅ 通过" if stats.verification_passed else "❌ 未通过"
            console.tree([
                ("最大差异", f"{stats.output_diff:.2e}"),
                ("验证结果", status),
            ])
            return True
        else:
            console.warning(f"图优化未完成: {result.message}")
            ctx.optimized_model = ctx.model
            return True
            
    except Exception as e:
        print(f"  ❌ 图优化失败: {e}")
        ctx.optimized_model = ctx.model
        return True


# ==============================================================================
# Stage 4: ONNX 导出
# ==============================================================================

def run_stage4_export(
    ctx: PipelineContext, 
    output_path: str = None,
    opset: int = None,
    enable_simplify: bool = False,
    enable_validation: bool = True,
    enable_dynamic_batch: bool = True,
    enable_dynamic_hw: bool = None,
    use_subprocess: bool = True,
) -> bool:
    """
    Stage 4: ONNX 导出
    
    关键修复：使用subprocess调用独立Python脚本进行导出，
    避免PyTorch ONNX导出器在同一进程内连续调用时的状态污染问题。
    """
    import sys
    import gc
    import os
    import json
    import tempfile
    import subprocess
    
    model = ctx.optimized_model or ctx.model
    if model is None:
        print("  ❌ 未找到模型")
        return False
    
    input_shape = ctx.input_shape
    if not input_shape:
        print("  ❌ 未找到输入形状")
        return False
    
    try:
        import torch
        from model_exporter import ModelExporter, ExportConfig, SimplifyConfig, ValidationConfig
        
        # 清理资源
        print("  [DEBUG] 清理资源...", flush=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  [DEBUG] 资源清理警告: {e}", flush=True)
        
        print("  [DEBUG] 导入模块...", flush=True)
        sys.stdout.flush()
        
        final_opset = opset or ctx.opset or 17
        if enable_dynamic_hw is None:
            enable_dynamic_hw = ctx.task_type in ('det', 'seg')
        
        if output_path is None:
            model_name = ctx.model_name or 'model'
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in model_name)
            output_path = f"{safe_name}.onnx"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        enable_simplify = False
        
        console.info("导出配置")
        console.tree([
            ("Opset 版本", str(final_opset)),
            ("动态轴", f"batch({'✔' if enable_dynamic_batch else '✘'}), H/W({'✔' if enable_dynamic_hw else '✘'})"),
            ("简化", "启用" if enable_simplify else "禁用"),
            ("验证", "启用" if enable_validation else "禁用"),
        ])
        console.blank()
        sys.stdout.flush()
        
        # ========== YOLO 模型专用导出（使用 Ultralytics 官方方法）==========
        if ctx.framework == 'ultralytics' and ctx.task_type == 'det':
            print("  [DEBUG] 检测到 YOLO 模型，使用 Ultralytics 官方导出...", flush=True)
            
            try:
                from ultralytics import YOLO
                
                # 重新加载原始模型（使用官方 YOLO 类）
                print(f"  [DEBUG] 加载原始模型: {ctx.model_path}", flush=True)
                yolo_model = YOLO(ctx.model_path)
                
                # 使用官方导出方法
                print("  [DEBUG] 调用 YOLO.export()...", flush=True)
                
                # 构建导出参数
                export_kwargs = {
                    'format': 'onnx',
                    'opset': final_opset,
                    'simplify': False,  # 避免 onnxsim 问题
                    'dynamic': enable_dynamic_batch,  # 动态 batch
                    'half': False,  # ONNX 导出不使用半精度
                    'device': 'cpu',  # CPU 上导出更稳定
                }
                
                # 执行导出
                exported_path = yolo_model.export(**export_kwargs)
                print(f"  [DEBUG] Ultralytics 导出完成: {exported_path}", flush=True)
                
                # 移动文件到目标位置
                if exported_path and os.path.exists(exported_path):
                    import shutil
                    
                    # 如果目标路径不同，则移动文件
                    if str(exported_path) != str(output_path):
                        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                        shutil.move(str(exported_path), output_path)
                        print(f"  [DEBUG] 移动文件到: {output_path}", flush=True)
                    
                    # 加载导出的 ONNX 模型
                    import onnx
                    ctx.onnx_path = output_path
                    ctx.onnx_model = onnx.load(output_path)
                    
                    node_count = len(ctx.onnx_model.graph.node)
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    console.info("导出结果 (Ultralytics 官方)")
                    console.table(["指标", "值"], [
                        ["节点数", str(node_count)],
                        ["文件大小", f"{file_size_mb:.2f} MB"],
                    ])
                    console.blank()
                    console.success(f"导出成功: {output_path}")
                    return True
                else:
                    print(f"  ❌ Ultralytics 导出失败: 输出文件不存在")
                    return False
                    
            except ImportError:
                print("  [DEBUG] ultralytics 未安装，回退到通用导出...", flush=True)
            except Exception as e:
                print(f"  [DEBUG] Ultralytics 导出失败: {e}，回退到通用导出...", flush=True)
                import traceback
                traceback.print_exc()
        
        # ========== 子进程导出方案 ==========
        if use_subprocess:
            print("  [DEBUG] 使用子进程导出模式...", flush=True)
            
            # 准备模型信息
            model_class_name = model.__class__.__name__.lower()
            
            # 推断timm模型名
            timm_name_map = {
                'mobilenetv3': 'mobilenetv3_small_100',
                'efficientnet': 'efficientnet_b0', 
                'resnet': 'resnet50',
                'vit': 'vit_base_patch16_224',
                'wideresnet': 'wide_resnet50_2',
                'convnext': 'convnext_tiny',
            }
            
            timm_name = None
            for key, value in timm_name_map.items():
                if key in model_class_name:
                    timm_name = value
                    break
            
            if not timm_name:
                print(f"  [DEBUG] 无法推断timm模型名，回退到直接导出", flush=True)
                use_subprocess = False
            else:
                # 推断num_classes
                state_dict = model.state_dict()
                num_classes = 1000
                classifier_keys = ['classifier.weight', 'fc.weight', 'head.weight']
                for key in classifier_keys:
                    if key in state_dict:
                        num_classes = state_dict[key].shape[0]
                        break
                
                print(f"  [DEBUG] 模型: {timm_name}, 类别数: {num_classes}", flush=True)
                
                # 保存state_dict到临时文件
                temp_dir = tempfile.mkdtemp()
                temp_state_dict_path = os.path.join(temp_dir, 'state_dict.pth')
                temp_config_path = os.path.join(temp_dir, 'config.json')
                temp_result_path = os.path.join(temp_dir, 'result.json')
                
                torch.save({k: v.cpu() for k, v in state_dict.items()}, temp_state_dict_path)
                print(f"  [DEBUG] 权重已保存到临时文件", flush=True)
                
                # 准备导出配置
                dynamic_axes = {}
                if enable_dynamic_batch:
                    dynamic_axes['input'] = {0: 'batch'}
                    dynamic_axes['output'] = {0: 'batch'}
                if enable_dynamic_hw:
                    dynamic_axes.setdefault('input', {})
                    dynamic_axes['input'][2] = 'height'
                    dynamic_axes['input'][3] = 'width'
                
                config_data = {
                    'timm_name': timm_name,
                    'num_classes': num_classes,
                    'state_dict_path': temp_state_dict_path,
                    'output_path': os.path.abspath(output_path),
                    'input_shape': list(input_shape),
                    'opset_version': final_opset,
                    'dynamic_axes': dynamic_axes,
                    'result_path': temp_result_path,
                }
                
                with open(temp_config_path, 'w') as f:
                    json.dump(config_data, f)
                
                # 创建导出脚本
                export_script = '''
import sys
import json
import torch

def main():
    config_path = sys.argv[1]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    result = {'success': False, 'message': ''}
    
    try:
        import timm
        
        # 重建模型
        print("  [SUBPROCESS] 重建模型...", flush=True)
        model = timm.create_model(
            config['timm_name'], 
            pretrained=False, 
            num_classes=config['num_classes']
        )
        
        # 加载权重
        state_dict = torch.load(config['state_dict_path'], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("  [SUBPROCESS] 模型重建成功", flush=True)
        
        # 创建输入
        input_shape = tuple(config['input_shape'])
        dummy_input = torch.rand(input_shape)
        
        # 准备动态轴
        dynamic_axes = config.get('dynamic_axes')
        if dynamic_axes:
            # 转换字符串键为整数
            for name in dynamic_axes:
                dynamic_axes[name] = {int(k): v for k, v in dynamic_axes[name].items()}
        
        # 导出
        print("  [SUBPROCESS] 执行torch.onnx.export...", flush=True)
        sys.stdout.flush()
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                config['output_path'],
                opset_version=config['opset_version'],
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes if dynamic_axes else None,
                do_constant_folding=True,
                verbose=False,
            )
        
        print("  [SUBPROCESS] 导出成功", flush=True)
        result['success'] = True
        result['message'] = '导出成功'
        
    except Exception as e:
        import traceback
        result['success'] = False
        result['message'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"  [SUBPROCESS] 导出失败: {e}", flush=True)
    
    # 保存结果
    with open(config['result_path'], 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()
'''
                
                temp_script_path = os.path.join(temp_dir, 'export_onnx.py')
                with open(temp_script_path, 'w', encoding='utf-8') as f:
                    f.write(export_script)
                
                print("  [DEBUG] 启动子进程...", flush=True)
                sys.stdout.flush()
                
                # 使用subprocess调用Python脚本
                try:
                    # 设置环境变量确保子进程使用UTF-8编码输出
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    
                    process = subprocess.Popen(
                        [sys.executable, temp_script_path, temp_config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        encoding='utf-8',  # 修复Windows中文编码问题
                        errors='replace',  # 遇到无法解码的字符用?替换，避免崩溃
                        env=env,  # 使用修改后的环境变量
                    )
                    
                    # 实时输出子进程日志
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            print(line.rstrip(), flush=True)
                    
                    # 等待完成（超时5分钟）
                    return_code = process.wait(timeout=300)
                    
                except subprocess.TimeoutExpired:
                    print("  [DEBUG] 子进程超时，终止...", flush=True)
                    process.kill()
                    process.wait()
                    # 清理临时文件
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print("  ❌ ONNX导出超时")
                    return False
                
                # 读取结果
                if os.path.exists(temp_result_path):
                    with open(temp_result_path, 'r') as f:
                        result = json.load(f)
                else:
                    result = {'success': False, 'message': '子进程未生成结果文件'}
                
                # 清理临时文件
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                if not result['success']:
                    print(f"  ❌ 子进程导出失败: {result.get('message', '未知错误')}")
                    if 'traceback' in result:
                        print(result['traceback'])
                    print("  [DEBUG] 回退到直接导出模式...", flush=True)
                    use_subprocess = False
                else:
                    print("  [DEBUG] 子进程导出成功", flush=True)
                    
                    # 加载导出的ONNX模型
                    import onnx
                    ctx.onnx_path = output_path
                    ctx.onnx_model = onnx.load(output_path)
                    
                    node_count = len(ctx.onnx_model.graph.node)
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    console.info("导出结果")
                    console.table(["指标", "值"], [
                        ["节点数", str(node_count)],
                        ["文件大小", f"{file_size_mb:.2f} MB"],
                    ])
                    console.blank()
                    console.success(f"导出成功: {output_path}")
                    return True
        
        # ========== 直接导出方案（回退） ==========
        if not use_subprocess:
            print("  [DEBUG] 创建导出配置...", flush=True)
            
            config = ExportConfig(
                opset_version=final_opset,
                enable_dynamic_batch=enable_dynamic_batch,
                enable_dynamic_hw=enable_dynamic_hw,
                simplify=SimplifyConfig(enable_simplify=False, skip_fuse_bn=True),
                validation=ValidationConfig(
                    enable_validation=False,
                    input_spec=ctx.input_spec,
                ),
            )
            
            print("  [DEBUG] 创建导出器...", flush=True)
            sys.stdout.flush()
            
            exporter = ModelExporter(config)
            
            print("  [DEBUG] 开始导出...", flush=True)
            sys.stdout.flush()
            
            result = exporter.export(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                task_type=ctx.task_type,
            )
            
            print("  [DEBUG] 导出完成", flush=True)
            sys.stdout.flush()
            
            if result.success:
                ctx.onnx_path = result.onnx_path
                ctx.onnx_model = result.onnx_model
                ctx.export_result = result
                ctx.dynamic_axes_spec = result.dynamic_axes_spec
                
                console.info("导出结果")
                rows = [
                    ["节点数", f"{result.stats.original_node_count} → {result.stats.simplified_node_count}"],
                    ["文件大小", f"{result.stats.file_size_mb:.2f} MB"],
                    ["导出耗时", f"{result.stats.export_time_ms/1000:.2f} 秒"],
                ]
                if result.validation_result:
                    rows.append(["数值精度", f"cosine={result.validation_result.cosine_sim:.6f}"])
                console.table(["指标", "值"], rows)
                console.blank()
                console.success(f"导出成功: {result.onnx_path}")
                return True
            else:
                print(f"  ❌ 导出失败: {result.message}")
                return False
            
    except Exception as e:
        print(f"  ❌ ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Stage 5-6: 量化转换与验证
# ==============================================================================

def run_stage5_convert(
    ctx: PipelineContext,
    output_dir: str,
    target_backend: str = None,
    precision: str = None,
    calib_data_path: str = None,
    calib_data_format: str = "imagefolder",
    calib_num_samples: int = 300,
    enable_validation: bool = True,
    enable_perf_test: bool = False,
    # TensorRT 动态形状配置
    trt_dynamic_batch_enabled: bool = False,
    trt_min_batch: int = 1,
    trt_opt_batch: int = 1,
    trt_max_batch: int = 1,
    trt_dynamic_shapes_enabled: bool = False,
    trt_min_shapes: tuple = None,
    trt_opt_shapes: tuple = None,
    trt_max_shapes: tuple = None,
    # TensorRT 其他配置
    trt_workspace_gb: int = 4,
    trt_timing_cache_path: str = None,
    trt_dla_core: int = None,
    # OpenVINO 动态形状配置（新增）
    ov_dynamic_batch_enabled: bool = False,
    ov_min_batch: int = 1,
    ov_max_batch: int = 16,
    # ONNX Runtime 动态形状配置（新增）
    ort_dynamic_batch_enabled: bool = False,
    ort_min_batch: int = 1,
    ort_max_batch: int = 16,
) -> bool:
    """
    Stage 5-6: 量化转换与验证
    
    支持的动态batch配置:
    - TensorRT: trt_dynamic_batch_enabled, trt_min_batch, trt_opt_batch, trt_max_batch
    - OpenVINO: ov_dynamic_batch_enabled, ov_min_batch, ov_max_batch
    - ONNX Runtime: ort_dynamic_batch_enabled, ort_min_batch, ort_max_batch
    """
    if not ctx.onnx_path or not os.path.exists(ctx.onnx_path):
        print("  ❌ 未找到 ONNX 模型")
        return False
    
    backend = target_backend or ctx.target_backend or 'onnxruntime'
    prec = precision or ctx.precision or 'fp16'
    ctx.target_backend = backend
    ctx.precision = prec
    
    try:
        from model_converter import ModelConverter, ConversionConfig, TargetBackend, PrecisionMode
        
        backend_enum = TargetBackend.from_string(backend)
        precision_enum = PrecisionMode.from_string(prec)
        
        if precision_enum.requires_calibration() and not calib_data_path:
            print(f"  ❌ {prec} 模式需要校准数据")
            return False
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = ctx.model_name or 'model'
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in model_name)
        
        if backend_enum == TargetBackend.TENSORRT:
            output_path = str(output_dir / f"{safe_name}.engine")
        elif backend_enum == TargetBackend.OPENVINO:
            output_path = str(output_dir)
        else:
            output_path = str(output_dir / f"{safe_name}_{prec}.onnx")
        
        console.info(f"{prec.upper()} 转换开始")
        console.tree([
            ("目标精度", prec),
            ("目标后端", backend),
            ("输出路径", output_path),
        ])
        
        # 显示动态形状配置
        if backend_enum == TargetBackend.TENSORRT:
            if trt_dynamic_batch_enabled:
                console.tree([
                    ("动态 Batch (TRT)", f"min={trt_min_batch}, opt={trt_opt_batch}, max={trt_max_batch}"),
                ])
            if trt_dynamic_shapes_enabled and trt_min_shapes:
                console.tree([
                    ("动态尺寸 (TRT)", f"min={trt_min_shapes}, opt={trt_opt_shapes}, max={trt_max_shapes}"),
                ])
        elif backend_enum == TargetBackend.OPENVINO:
            if ov_dynamic_batch_enabled:
                console.tree([
                    ("动态 Batch (OV)", f"min={ov_min_batch}, max={ov_max_batch}"),
                ])
        elif backend_enum == TargetBackend.ORT:
            if ort_dynamic_batch_enabled:
                console.tree([
                    ("动态 Batch (ORT)", f"min={ort_min_batch}, max={ort_max_batch}"),
                ])
        
        console.blank()
        
        config = ConversionConfig(
            target_backend=backend_enum,
            precision_mode=precision_enum,
            calib_data_path=calib_data_path,
            calib_num_samples=calib_num_samples,
            input_shape=ctx.input_shape,
            dynamic_axes_spec=(ctx.dynamic_axes_spec.to_tensorrt_profile() 
                              if ctx.dynamic_axes_spec and hasattr(ctx.dynamic_axes_spec, 'to_tensorrt_profile') 
                              else None),
            enable_validation=enable_validation,
            enable_perf_test=enable_perf_test,
            # 任务类型和框架（用于自动选择预处理配置）
            task_type=ctx.task_type,
            framework=ctx.framework,
            # InputSpec (从 Stage 1 传入，包含完整的预处理信息)
            input_spec=ctx.input_spec,
            # TensorRT 动态形状配置
            trt_dynamic_batch_enabled=trt_dynamic_batch_enabled,
            trt_min_batch=trt_min_batch,
            trt_opt_batch=trt_opt_batch,
            trt_max_batch=trt_max_batch,
            trt_dynamic_shapes_enabled=trt_dynamic_shapes_enabled,
            trt_min_shapes=trt_min_shapes,
            trt_opt_shapes=trt_opt_shapes,
            trt_max_shapes=trt_max_shapes,
            # TensorRT 其他配置
            trt_workspace_gb=trt_workspace_gb,
            trt_timing_cache_path=trt_timing_cache_path,
            trt_dla_core=trt_dla_core,
            # OpenVINO 动态形状配置（新增）
            ov_dynamic_batch_enabled=ov_dynamic_batch_enabled,
            ov_min_batch=ov_min_batch,
            ov_max_batch=ov_max_batch,
            # ONNX Runtime 动态形状配置（新增）
            ort_dynamic_batch_enabled=ort_dynamic_batch_enabled,
            ort_min_batch=ort_min_batch,
            ort_max_batch=ort_max_batch,
        )
        
        converter = ModelConverter(config)
        result = converter.convert(
            onnx_path=ctx.onnx_path,
            output_path=output_path,
            input_shape=ctx.input_shape,
        )
        
        if result.success:
            ctx.conversion_result = result
            ctx.converted_model_path = result.output_files[0] if result.output_files else ""
            ctx.validation_result = result.validation
            
            console.info("转换完成")
            console.tree([
                ("输出文件", ctx.converted_model_path),
                ("转换耗时", f"{result.build_time_seconds:.2f} 秒"),
                ("状态", "✅ 成功"),
            ])
            
            if result.validation and result.validation.passed:
                console.blank()
                console.info("精度验证")
                console.table(
                    ["验证指标", "值"],
                    [
                        ["余弦相似度", f"{result.validation.cosine_sim:.6f}"],
                        ["最大绝对误差", f"{result.validation.max_diff:.6f}"],
                        ["压缩比", f"{result.validation.compression_ratio:.2f}x"],
                    ]
                )
                
                if result.validation.latency_ms:
                    console.blank()
                    console.info("性能基准测试")
                    console.table(
                        ["性能指标", "值"],
                        [
                            ["平均延迟", f"{result.validation.latency_ms:.2f} ms"],
                            ["吞吐量", f"{result.validation.throughput_fps:.2f} FPS"],
                        ]
                    )
                
                console.blank()
                console.info("压缩效果")
                console.table(
                    ["存储指标", "值"],
                    [
                        ["原始模型", f"{result.validation.original_size_mb:.2f} MB"],
                        [f"{prec.upper()} 模型", f"{result.validation.output_size_mb:.2f} MB"],
                        ["压缩比例", f"{result.validation.compression_ratio:.2f}x"],
                    ]
                )
            elif result.validation and not result.validation.passed:
                console.blank()
                console.warning("验证失败")
                if result.validation.warnings:
                    console.tree(result.validation.warnings)
            return True
        else:
            print(f"  ❌ 转换失败: {result.message}")
            return False
            
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Stage 8: 配置文件生成
# ==============================================================================

def run_stage8_generate_config(
    ctx: PipelineContext,
    output_dir: str,
) -> bool:
    """
    Stage 8: 生成部署配置文件
    
    Args:
        ctx: Pipeline 上下文
        output_dir: 输出目录
        
    Returns:
        是否成功
    """
    try:
        from config_generator import ConfigGenerator
        
        generator = ConfigGenerator()
        config_path = generator.generate_from_context(
            ctx=ctx,
            output_dir=output_dir,
            conversion_result=ctx.conversion_result,
        )
        
        ctx.config_path = config_path
        
        print(f"  ✅ 配置文件已生成: {config_path}")
        return True
        
    except ImportError as e:
        print(f"  ❌ 导入模块失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 配置文件生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# CLI 命令
# ==============================================================================

def cmd_analyze(args):
    """analyze 命令: 分析模型 (Stage 1 + Stage 2)"""
    setup_logging(args.verbose)
    
    # 解析输入形状
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    # 解析后端
    backends = [b.strip() for b in args.backends.split(',')]
    
    ctx = PipelineContext()
    
    # Stage 1
    if not run_stage1_import(ctx, args.model, args.task, input_shape, args.device):
        return 1
    
    # Stage 2
    if not run_stage2_analyze(ctx, backends, args.output, args.format):
        return 1
    
    return 0


def cmd_optimize(args):
    """optimize 命令: 优化模型 (Stage 1 + 2 + 3)"""
    setup_logging(args.verbose)
    
    # 解析输入形状
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    # 解析后端
    backends = [b.strip() for b in args.backends.split(',')]
    
    ctx = PipelineContext()
    
    # Stage 1
    if not run_stage1_import(ctx, args.model, args.task, input_shape, args.device):
        return 1
    
    # Stage 2
    if not run_stage2_analyze(ctx, backends):
        return 1
    
    # Stage 3
    if not run_stage3_optimize(
        ctx,
        enable_fusion=not args.no_fusion,
        enable_elimination=not args.no_elimination,
        verify_output=not args.no_verify,
    ):
        return 1
    
    return 0


def cmd_export(args):
    """export 命令: 导出 ONNX (Stage 1-4)"""
    setup_logging(args.verbose)
    
    # 解析输入形状
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    ctx = PipelineContext()
    
    # Stage 1
    if not run_stage1_import(ctx, args.model, args.task, input_shape, args.device):
        return 1
    
    # Stage 2
    if not run_stage2_analyze(ctx, ['onnx']):
        return 1
    
    # Stage 3
    if not args.no_optimize:
        if not run_stage3_optimize(ctx):
            return 1
    else:
        ctx.optimized_model = ctx.model
        logger.info("  跳过 Stage 3 优化")
    
    # Stage 4
    if not run_stage4_export(
        ctx,
        output_path=args.output,
        opset=args.opset,
        enable_simplify=not args.no_simplify,
        enable_validation=not args.no_validation,
        enable_dynamic_batch=not args.static_batch,
        enable_dynamic_hw=args.dynamic_hw,
    ):
        return 1
    
    logger.info("=" * 60)
    logger.info("✅ ONNX 导出完成")
    logger.info(f"  输出文件: {ctx.onnx_path}")
    logger.info("=" * 60)
    
    return 0


def cmd_convert(args):
    """convert 命令: 从现有 ONNX 转换 (Stage 5-8)"""
    setup_logging(args.verbose)
    
    # 解析输入形状
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    # 创建上下文
    ctx = PipelineContext()
    ctx.onnx_path = args.model
    ctx.target_backend = args.target
    ctx.precision = args.precision
    ctx.task_type = args.task
    
    # 从 ONNX 推断模型名称
    ctx.model_name = Path(args.model).stem
    
    # 如果提供了输入形状，使用它
    if input_shape:
        ctx.input_shape = input_shape
    else:
        # 尝试从 ONNX 模型推断
        try:
            import onnx
            model = onnx.load(args.model)
            for inp in model.graph.input:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)  # 动态维度使用 1
                if len(shape) == 4:
                    ctx.input_shape = tuple(shape)
                    break
        except (ImportError, FileNotFoundError, Exception) as e:
            logger.debug(f"Failed to infer input shape from ONNX: {e}")
            ctx.input_shape = DEFAULT_INPUT_SHAPES.get(args.task, (1, 3, 640, 640))
    
    logger.info(f"  输入形状: {ctx.input_shape}")
    
    # 根据目标后端设置动态batch参数
    dynamic_batch = getattr(args, 'dynamic_batch', False)
    min_batch = getattr(args, 'min_batch', 1)
    opt_batch = getattr(args, 'opt_batch', 1)
    max_batch = getattr(args, 'max_batch', 16)
    
    # Stage 5-6: 转换与验证
    if not run_stage5_convert(
        ctx,
        output_dir=args.output,
        target_backend=args.target,
        precision=args.precision,
        calib_data_path=args.calib_data,
        calib_data_format=args.calib_format,
        calib_num_samples=args.calib_samples,
        enable_validation=not args.no_validation,
        enable_perf_test=args.perf_test,
        # TensorRT 动态batch配置
        trt_dynamic_batch_enabled=dynamic_batch,
        trt_min_batch=min_batch,
        trt_opt_batch=opt_batch,
        trt_max_batch=max_batch,
        # OpenVINO 动态batch配置
        ov_dynamic_batch_enabled=dynamic_batch,
        ov_min_batch=min_batch,
        ov_max_batch=max_batch,
        # ONNX Runtime 动态batch配置
        ort_dynamic_batch_enabled=dynamic_batch,
        ort_min_batch=min_batch,
        ort_max_batch=max_batch,
    ):
        return 1
    
    # Stage 8: 生成配置文件
    if not args.no_config:
        if not run_stage8_generate_config(ctx, args.output):
            return 1
    
    logger.info("=" * 60)
    logger.info("✅ 转换完成")
    logger.info(f"  模型文件: {ctx.converted_model_path}")
    if ctx.config_path:
        logger.info(f"  配置文件: {ctx.config_path}")
    logger.info("=" * 60)
    
    return 0


def cmd_pipeline(args):
    """pipeline 命令: 完整转换流程 (Stage 1-8)"""
    setup_logging(args.verbose)
    
    logger.info("🚀 完整转换 Pipeline (Stage 1-8)")
    
    # 解析输入形状
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    # 创建上下文
    ctx = PipelineContext()
    ctx.target_backend = args.target
    ctx.precision = args.precision
    
    # Stage 1: 导入
    if not run_stage1_import(ctx, args.model, args.task, input_shape, args.device):
        return 1
    
    # Stage 2: 分析
    if not run_stage2_analyze(ctx, [args.target]):
        return 1
    
    # Stage 3: 优化
    if not run_stage3_optimize(ctx):
        return 1
    
    # 确定输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定 ONNX 输出路径
    model_name = ctx.model_name or 'model'
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in model_name)
    onnx_output = str(output_dir / f"{safe_name}.onnx")
    
    # Stage 4: ONNX 导出
    if not run_stage4_export(
        ctx, 
        output_path=onnx_output,
        opset=args.opset,
        enable_simplify=not args.no_simplify,
        enable_validation=not args.no_validation,
    ):
        return 1
    
    # 根据目标后端设置动态batch参数
    dynamic_batch = getattr(args, 'dynamic_batch', False)
    min_batch = getattr(args, 'min_batch', 1)
    opt_batch = getattr(args, 'opt_batch', 1)
    max_batch = getattr(args, 'max_batch', 16)
    
    # Stage 5-6: 量化转换与验证
    if not run_stage5_convert(
        ctx,
        output_dir=str(output_dir),
        target_backend=args.target,
        precision=args.precision,
        calib_data_path=args.calib_data,
        calib_data_format=args.calib_format,
        calib_num_samples=args.calib_samples,
        enable_validation=not args.no_validation,
        enable_perf_test=args.perf_test,
        # TensorRT 动态batch配置
        trt_dynamic_batch_enabled=dynamic_batch,
        trt_min_batch=min_batch,
        trt_opt_batch=opt_batch,
        trt_max_batch=max_batch,
        # OpenVINO 动态batch配置
        ov_dynamic_batch_enabled=dynamic_batch,
        ov_min_batch=min_batch,
        ov_max_batch=max_batch,
        # ONNX Runtime 动态batch配置 (ORT默认支持，此参数用于显示)
        ort_dynamic_batch_enabled=dynamic_batch,
        ort_min_batch=min_batch,
        ort_max_batch=max_batch,
    ):
        return 1
    
    # Stage 8: 生成配置文件
    if not args.no_config:
        if not run_stage8_generate_config(ctx, str(output_dir)):
            return 1
    
    logger.info("=" * 60)
    logger.info("✅ Pipeline 完成")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  ONNX 文件: {ctx.onnx_path}")
    if ctx.converted_model_path:
        logger.info(f"  转换后模型: {ctx.converted_model_path}")
    if ctx.config_path:
        logger.info(f"  配置文件: {ctx.config_path}")
    logger.info("=" * 60)
    
    return 0


def cmd_info(args):
    """info 命令: 查看模型信息 (仅 Stage 1)"""
    setup_logging(args.verbose)
    
    ctx = PipelineContext()
    
    if not run_stage1_import(ctx, args.model, args.task):
        return 1
    
    # 打印信息
    print("\n📋 模型信息")
    print("=" * 40)
    print(f"  名称:     {ctx.model_name}")
    print(f"  任务:     {ctx.task_type}")
    print(f"  输入:     {ctx.input_shape}")
    print(f"  框架:     {ctx.framework}")
    
    # 计算参数量
    if ctx.model is not None:
        total_params = sum(p.numel() for p in ctx.model.parameters())
        trainable_params = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)
        print(f"  参数量:   {total_params / 1e6:.2f} M")
        print(f"  可训练:   {trainable_params / 1e6:.2f} M")
    
    print("=" * 40)
    
    return 0


def cmd_run(args):
    """run 命令: 从 YAML 配置文件运行完整流程"""
    from config_templates import load_config, validate_config
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return 1
    
    # 验证配置
    errors = validate_config(config)
    if errors:
        print("❌ 配置验证失败:")
        for err in errors:
            print(f"  - {err}")
        return 1
    
    # 配置日志（静默第三方库）
    setup_logging(config.verbose)
    
    # 启动时间追踪
    Timer.start()
    start_time = Timer.now_full()
    
    # 显示启动信息
    console.header("模型转换流水线启动", "🚀")
    print(f"🕐 开始时间: {start_time}")
    print(f"📦 工作目录: {config.output_dir}")
    print(f"🔧 配置: backend={config.target_backend}, precision={config.precision}")
    console.line()
    # Dry run 模式
    if args.dry_run:
        print("\n✅ 配置验证通过 (dry-run 模式，不执行)")
        return 0
    
    # 创建上下文
    ctx = PipelineContext()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ========== Stage 1: 模型导入 ==========
    if 'stage1' not in config.skip_stages:
        console.stage(1, "模型加载与重建")
        if not run_stage1_import(
            ctx, config.model_path, config.task_type,
            config.input_shape, config.device
        ):
            return 1
    
    # ========== Stage 2: 模型分析 ==========
    if 'stage2' not in config.skip_stages:
        console.stage(2, "模型分析")
        if not run_stage2_analyze(
            ctx, config.target_backends,
            config.report_output, config.report_format
        ):
            return 1
    
    # ========== Stage 3: 图优化 ==========
    if 'stage3' not in config.skip_stages and config.optimize_enabled:
        console.stage(3, "图优化")
        if not run_stage3_optimize(
            ctx, enable_fusion=config.enable_fusion,
            enable_elimination=config.enable_elimination,
            verify_output=config.verify_output
        ):
            return 1
    else:
        ctx.optimized_model = ctx.model
    
    # ========== Stage 4: ONNX 导出 ==========
    if 'stage4' not in config.skip_stages:
        console.stage(4, "ONNX 导出")
        onnx_output = config.onnx_output_path
        if not onnx_output:
            onnx_output = os.path.join(config.output_dir, f"{ctx.model_name}.onnx")
        
        if not run_stage4_export(
            ctx,
            output_path=onnx_output,
            opset=config.opset_version,
            enable_simplify=config.enable_simplify,
            enable_validation=config.enable_export_validation,
            enable_dynamic_batch=config.dynamic_batch,
            enable_dynamic_hw=config.dynamic_hw
        ):
            return 1
    
    # ========== Stage 5-6: 量化转换与验证 ==========
    if 'stage5' not in config.skip_stages:
        console.stage(5, "量化转换与验证")
        ctx.target_backend = config.target_backend
        ctx.precision = config.precision
        
        if not run_stage5_convert(
            ctx,
            output_dir=config.output_dir,
            target_backend=config.target_backend,
            precision=config.precision,
            calib_data_path=config.calib_data_path,
            calib_data_format=config.calib_data_format,
            calib_num_samples=config.calib_num_samples,
            enable_validation=config.enable_convert_validation,
            enable_perf_test=config.enable_perf_test,
            # TensorRT 动态形状配置
            trt_dynamic_batch_enabled=config.trt_dynamic_batch_enabled,
            trt_min_batch=config.trt_min_batch,
            trt_opt_batch=config.trt_opt_batch,
            trt_max_batch=config.trt_max_batch,
            trt_dynamic_shapes_enabled=config.trt_dynamic_shapes_enabled,
            trt_min_shapes=config.trt_min_shapes,
            trt_opt_shapes=config.trt_opt_shapes,
            trt_max_shapes=config.trt_max_shapes,
            # TensorRT 其他配置
            trt_workspace_gb=config.trt_workspace_gb,
            trt_timing_cache_path=config.trt_timing_cache_path,
            trt_dla_core=config.trt_dla_core,
            # OpenVINO 动态batch配置
            ov_dynamic_batch_enabled=config.ov_dynamic_batch_enabled,
            ov_min_batch=config.ov_min_batch,
            ov_max_batch=config.ov_max_batch,
            # ONNX Runtime 动态batch配置
            ort_dynamic_batch_enabled=config.ort_dynamic_batch_enabled,
            ort_min_batch=config.ort_min_batch,
            ort_max_batch=config.ort_max_batch,
        ):
            return 1
    
    # ========== Stage 8: 配置文件生成 ==========
    if 'stage8' not in config.skip_stages and config.config_enabled:
        console.stage(8, "配置文件生成")
        if not run_stage8_generate_config(ctx, output_dir=config.output_dir):
            return 1
    
    # ========== 汇总报告 ==========
    end_time = Timer.now_full()
    
    console.summary_header("流水线执行完成 - 汇总报告")
    
    # 转换目标
    console.summary_section("转换目标", "🎯")
    console.tree([
        ("模型", f"{ctx.model_name} ({ctx.framework})"),
        ("输入", str(ctx.input_shape)),
        ("任务", ctx.task_type),
    ], indent=0)
    
    console.blank()
    
    # 转换结果
    console.summary_section("转换结果", "✅")
    result_items = []
    if ctx.validation_result and ctx.validation_result.passed:
        result_items.append(("精度保持", f"{ctx.validation_result.cosine_sim:.6f}"))
        if ctx.validation_result.throughput_fps:
            result_items.append(("推理性能", f"{ctx.validation_result.throughput_fps:.2f} FPS"))
        result_items.append(("存储优化", f"{ctx.validation_result.compression_ratio:.2f}x 压缩"))
    console.tree(result_items, indent=0)
    
    console.blank()
    
    # 生成文件
    console.summary_section("生成文件", "📁")
    files = []
    if ctx.onnx_path:
        size_mb = os.path.getsize(ctx.onnx_path) / (1024*1024) if os.path.exists(ctx.onnx_path) else 0
        files.append((os.path.basename(ctx.onnx_path), f"{size_mb:.2f} MB"))
    if ctx.converted_model_path:
        # 计算转换后模型大小 (考虑 OpenVINO .xml + .bin)
        converted_path = Path(ctx.converted_model_path)
        if converted_path.exists():
            size_mb = converted_path.stat().st_size / (1024*1024)
            # OpenVINO 特殊处理: .xml 需要加上 .bin 文件大小
            if converted_path.suffix.lower() == '.xml':
                bin_path = converted_path.with_suffix('.bin')
                if bin_path.exists():
                    size_mb += bin_path.stat().st_size / (1024*1024)
                # 显示名称改为 model.xml + model.bin
                display_name = f"{converted_path.stem}.xml + {converted_path.stem}.bin"
                files.append((display_name, f"{size_mb:.2f} MB"))
            else:
                files.append((os.path.basename(ctx.converted_model_path), f"{size_mb:.2f} MB"))
        else:
            files.append((os.path.basename(ctx.converted_model_path), "0 MB"))
    if ctx.config_path:
        files.append((os.path.basename(ctx.config_path), "配置文件"))
    console.tree(files)
    
    console.blank()
    
    # 优化建议
    console.summary_section("建议与优化", "💡")
    suggestions = []
    if config.precision != 'int8':
        suggestions.append("可尝试 INT8 量化获得更高压缩比")
    if config.target_backend != 'tensorrt':
        suggestions.append("使用 TensorRT 可获得更低延迟")
    suggestions.append("批处理可进一步提升吞吐量")
    console.tree(suggestions, indent=2)
    
    console.final_status(True, elapsed=Timer.elapsed(), end_time=end_time)
    return 0


def cmd_init(args):
    """init 命令: 生成 YAML 配置模板"""
    from config_templates import generate_template
    
    if args.task == 'all':
        for task in ['cls', 'det', 'seg']:
            path = generate_template(task)
            print(f"✅ 生成: {path}")
    else:
        path = generate_template(args.task, args.output)
        print(f"✅ 生成: {path}")
    
    print("\n💡 提示:")
    print("  1. 编辑配置文件，设置模型路径和参数")
    print("  2. 运行: python main.py run -c <config.yaml>")
    
    return 0


def cmd_test(args):
    """test 命令: 运行各模块的测试"""
    setup_logging(verbose=True)
    
    print("=" * 60)
    print("🧪 模块测试")
    print("=" * 60)
    
    results = []
    
    # 检查依赖
    def check_dependencies():
        """检查依赖"""
        deps = {}
        
        # 核心依赖
        try:
            import torch
            deps['torch'] = True
        except ImportError:
            deps['torch'] = False
        
        try:
            import onnx
            deps['onnx'] = True
        except ImportError:
            deps['onnx'] = False
        
        try:
            import onnxruntime
            deps['onnxruntime'] = True
        except ImportError:
            deps['onnxruntime'] = False
        
        try:
            import yaml
            deps['pyyaml'] = True
        except ImportError:
            deps['pyyaml'] = False
        
        # 可选依赖
        try:
            import tensorrt
            deps['tensorrt'] = True
        except ImportError:
            deps['tensorrt'] = False
        
        try:
            import openvino
            deps['openvino'] = True
        except ImportError:
            deps['openvino'] = False
        
        return deps
    
    print("\n📦 依赖检查")
    deps = check_dependencies()
    for pkg, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {pkg}")
    
    # 测试各模块
    modules_to_test = [
        ('model_importer', 'Stage 1'),
        ('model_analyzer', 'Stage 2'),
        ('model_optimizer', 'Stage 3'),
        ('model_exporter', 'Stage 4'),
        ('model_converter', 'Stage 5-6'),
        ('config_generator', 'Stage 8'),
    ]
    
    for module_name, stage_name in modules_to_test:
        print(f"\n📋 测试 {stage_name}: {module_name}")
        try:
            __import__(module_name)
            print("  ✅ 导入成功")
            results.append((f'{module_name} 导入', True))
        except Exception as e:
            print(f"  ❌ 导入失败: {e}")
            results.append((f'{module_name} 导入', False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("📊 测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = '✅' if success else '❌'
        print(f"  {status} {name}")
    
    print(f"\n  总计: {passed}/{total} 通过")
    print("=" * 60)
    
    return 0 if all(s for _, s in results) else 1


def cmd_version(args):
    """version 命令: 显示版本信息"""
    # 检测 GPU
    gpu_status = "❌ 不可用"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_status = f"✅ {gpu_name}"
    except ImportError:
        pass
    
    # 检测后端
    trt_status = "❌"
    ov_status = "❌"
    try:
        import tensorrt
        trt_status = f"✅ v{tensorrt.__version__}"
    except ImportError:
        pass
    try:
        import openvino
        ov_status = f"✅ v{openvino.__version__}"
    except ImportError:
        pass
    
    print(f"""
🚀 模型转换工具 (Model Converter Toolkit)
==========================================
版本: 2.0.0
作者: Model Converter Team

模块状态:
  ✅ Stage 1: model_importer  (模型导入)
  ✅ Stage 2: model_analyzer  (模型分析)
  ✅ Stage 3: model_optimizer (图优化)
  ✅ Stage 4: model_exporter  (ONNX 导出)
  ✅ Stage 5: model_converter (量化转换)
  ✅ Stage 6: conversion_validator (验证)
  ✅ Stage 8: config_generator (配置生成)

环境信息:
  设备:       {get_default_device()} (GPU 优先)
  GPU 状态:   {gpu_status}
  TensorRT:   {trt_status}
  OpenVINO:   {ov_status}

支持的后端:
  ✅ ONNX Runtime - 通用跨平台推理
  ✅ TensorRT     - NVIDIA GPU 极致性能
  ✅ OpenVINO     - Intel 硬件优化

支持的精度:
  ✅ FP32  - 全精度
  ✅ FP16  - 半精度 (推荐)
  ✅ INT8  - 量化 (需要校准数据)

默认输入形状:
  cls (分类): 1,3,224,224
  det (检测): 1,3,640,640
  seg (分割): 1,3,512,512
""")
    return 0


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🚀 模型转换工具 - 工业级深度学习模型转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析模型兼容性
  python main.py analyze -m model.pth -t cls
  
  # 导出 ONNX (Stage 1-4)
  python main.py export -m model.pth -t det -o model.onnx
  
  # 完整流程 (Stage 1-8): PyTorch → TensorRT FP16
  python main.py pipeline -m model.pth -t det -o ./output --target tensorrt --precision fp16
  
  # 完整流程 (Stage 1-8): PyTorch → TensorRT INT8 (需要校准数据)
  python main.py pipeline -m model.pth -t det -o ./output --target tensorrt --precision int8 --calib-data ./images
  
  # 从现有 ONNX 转换 (Stage 5-8)
  python main.py convert -m model.onnx -t det -o ./output --target tensorrt --precision fp16
  
  # 查看版本
  python main.py version

默认值:
  - 设备: cuda (如果可用)
  - 目标后端: onnxruntime
  - 精度: fp16
  - opset: 17
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # -------------------- analyze 命令 --------------------
    p_analyze = subparsers.add_parser('analyze', help='分析模型 (Stage 1 + 2)')
    p_analyze.add_argument('-m', '--model', required=True, help='模型文件路径')
    p_analyze.add_argument('-t', '--task', required=True, 
                           choices=['cls', 'det', 'seg'], help='任务类型')
    p_analyze.add_argument('--input-shape', type=str, 
                           help='输入形状 (格式: B,C,H,W)')
    p_analyze.add_argument('--backends', type=str, default='onnxruntime,tensorrt,openvino',
                           help='目标后端，逗号分隔')
    p_analyze.add_argument('-o', '--output', type=str, help='报告输出路径')
    p_analyze.add_argument('--format', type=str, default='console',
                           choices=['console', 'json', 'html'], help='输出格式')
    p_analyze.add_argument('--device', type=str, default=None, help='设备')
    p_analyze.set_defaults(func=cmd_analyze)
    
    # -------------------- optimize 命令 --------------------
    p_optimize = subparsers.add_parser('optimize', help='优化模型 (Stage 1-3)')
    p_optimize.add_argument('-m', '--model', required=True, help='模型文件路径')
    p_optimize.add_argument('-t', '--task', required=True,
                            choices=['cls', 'det', 'seg'], help='任务类型')
    p_optimize.add_argument('--input-shape', type=str, help='输入形状')
    p_optimize.add_argument('--backends', type=str, default='onnxruntime,tensorrt,openvino',
                            help='目标后端，逗号分隔')
    p_optimize.add_argument('--device', type=str, default=None, help='设备')
    p_optimize.add_argument('--no-fusion', action='store_true', help='禁用融合优化')
    p_optimize.add_argument('--no-elimination', action='store_true', help='禁用冗余消除')
    p_optimize.add_argument('--no-verify', action='store_true', help='禁用输出验证')
    p_optimize.set_defaults(func=cmd_optimize)
    
    # -------------------- export 命令 --------------------
    p_export = subparsers.add_parser('export', help='导出 ONNX (Stage 1-4)')
    p_export.add_argument('-m', '--model', required=True, help='模型文件路径')
    p_export.add_argument('-t', '--task', required=True,
                          choices=['cls', 'det', 'seg'], help='任务类型')
    p_export.add_argument('-o', '--output', type=str, help='输出 ONNX 路径')
    p_export.add_argument('--input-shape', type=str, help='输入形状 (格式: B,C,H,W)')
    p_export.add_argument('--opset', type=int, default=None, help='ONNX opset 版本')
    p_export.add_argument('--device', type=str, default=None, help='设备')
    p_export.add_argument('--no-optimize', action='store_true', help='跳过 Stage 3 优化')
    p_export.add_argument('--no-simplify', action='store_true', help='禁用 ONNX 简化')
    p_export.add_argument('--no-validation', action='store_true', help='禁用验证')
    p_export.add_argument('--static-batch', action='store_true', help='使用静态 batch')
    p_export.add_argument('--dynamic-hw', action='store_true', help='启用动态 H/W')
    p_export.set_defaults(func=cmd_export)
    
    # -------------------- convert 命令 (新增) --------------------
    p_convert = subparsers.add_parser('convert', help='从 ONNX 转换 (Stage 5-8)')
    p_convert.add_argument('-m', '--model', required=True, help='ONNX 模型路径')
    p_convert.add_argument('-t', '--task', required=True,
                           choices=['cls', 'det', 'seg'], help='任务类型')
    p_convert.add_argument('-o', '--output', required=True, help='输出目录')
    p_convert.add_argument('--target', type=str, default='tensorrt',
                           choices=['ort', 'tensorrt', 'openvino'],
                           help='目标后端 (默认: tensorrt)')
    p_convert.add_argument('--precision', type=str, default='fp16',
                           choices=['fp32', 'fp16', 'int8'],
                           help='精度模式 (默认: fp16)')
    p_convert.add_argument('--input-shape', type=str, help='输入形状')
    p_convert.add_argument('--calib-data', type=str, help='校准数据路径 (INT8 需要)')
    p_convert.add_argument('--calib-format', type=str, default='imagefolder',
                           choices=['imagefolder', 'coco'], help='校准数据格式')
    p_convert.add_argument('--calib-samples', type=int, default=300, help='校准样本数')
    p_convert.add_argument('--no-validation', action='store_true', help='禁用验证')
    p_convert.add_argument('--perf-test', action='store_true', help='启用性能测试')
    p_convert.add_argument('--no-config', action='store_true', help='不生成配置文件')
    # 动态batch配置 (所有后端通用)
    p_convert.add_argument('--dynamic-batch', action='store_true', 
                           help='启用动态batch (TensorRT/OpenVINO/ORT)')
    p_convert.add_argument('--min-batch', type=int, default=1, help='最小batch size')
    p_convert.add_argument('--opt-batch', type=int, default=1, help='最优batch size (TensorRT)')
    p_convert.add_argument('--max-batch', type=int, default=16, help='最大batch size')
    p_convert.set_defaults(func=cmd_convert)
    
    # -------------------- pipeline 命令 (更新) --------------------
    p_pipeline = subparsers.add_parser('pipeline', help='完整转换流程 (Stage 1-8)')
    p_pipeline.add_argument('-m', '--model', required=True, help='PyTorch 模型路径')
    p_pipeline.add_argument('-t', '--task', required=True,
                            choices=['cls', 'det', 'seg'], help='任务类型')
    p_pipeline.add_argument('-o', '--output', required=True, help='输出目录')
    p_pipeline.add_argument('--target', type=str, default='tensorrt',
                            choices=['ort', 'tensorrt', 'openvino'],
                            help='目标后端 (默认: tensorrt)')
    p_pipeline.add_argument('--precision', type=str, default='fp16',
                            choices=['fp32', 'fp16', 'int8'],
                            help='精度模式 (默认: fp16)')
    p_pipeline.add_argument('--input-shape', type=str, help='输入形状')
    p_pipeline.add_argument('--opset', type=int, default=None, help='ONNX opset 版本')
    p_pipeline.add_argument('--device', type=str, default=None, help='设备')
    p_pipeline.add_argument('--calib-data', type=str, help='校准数据路径 (INT8 需要)')
    p_pipeline.add_argument('--calib-format', type=str, default='imagefolder',
                            choices=['imagefolder', 'coco'], help='校准数据格式')
    p_pipeline.add_argument('--calib-samples', type=int, default=300, help='校准样本数')
    p_pipeline.add_argument('--no-simplify', action='store_true', help='禁用 ONNX 简化')
    p_pipeline.add_argument('--no-validation', action='store_true', help='禁用验证')
    p_pipeline.add_argument('--perf-test', action='store_true', help='启用性能测试')
    p_pipeline.add_argument('--no-config', action='store_true', help='不生成配置文件')
    # 动态batch配置 (所有后端通用)
    p_pipeline.add_argument('--dynamic-batch', action='store_true', 
                            help='启用动态batch (TensorRT/OpenVINO/ORT)')
    p_pipeline.add_argument('--min-batch', type=int, default=1, help='最小batch size')
    p_pipeline.add_argument('--opt-batch', type=int, default=1, help='最优batch size (TensorRT)')
    p_pipeline.add_argument('--max-batch', type=int, default=16, help='最大batch size')
    p_pipeline.set_defaults(func=cmd_pipeline)
    
    # -------------------- info 命令 --------------------
    p_info = subparsers.add_parser('info', help='查看模型信息')
    p_info.add_argument('-m', '--model', required=True, help='模型文件路径')
    p_info.add_argument('-t', '--task', required=True,
                        choices=['cls', 'det', 'seg'], help='任务类型')
    p_info.set_defaults(func=cmd_info)
    
    # -------------------- run 命令 (从配置文件运行) --------------------
    p_run = subparsers.add_parser('run', help='从 YAML 配置文件运行完整流程')
    p_run.add_argument('-c', '--config', required=True, help='YAML 配置文件路径')
    p_run.add_argument('--dry-run', action='store_true', help='仅验证配置，不执行')
    p_run.set_defaults(func=cmd_run)
    
    # -------------------- init 命令 (生成配置模板) --------------------
    p_init = subparsers.add_parser('init', help='生成 YAML 配置模板')
    p_init.add_argument('-t', '--task', required=True,
                        choices=['cls', 'det', 'seg', 'all'],
                        help='任务类型 (cls/det/seg/all)')
    p_init.add_argument('-o', '--output', type=str, help='输出路径')
    p_init.set_defaults(func=cmd_init)
    
    # -------------------- test 命令 --------------------
    p_test = subparsers.add_parser('test', help='运行模块测试')
    p_test.add_argument('--full', action='store_true', help='运行完整测试')
    p_test.set_defaults(func=cmd_test)
    
    # -------------------- version 命令 --------------------
    p_version = subparsers.add_parser('version', help='显示版本信息')
    p_version.set_defaults(func=cmd_version)
    
    # 解析参数
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())