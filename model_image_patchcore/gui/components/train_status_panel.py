# -*- coding: utf-8 -*-
"""PatchCore 训练状态面板 - v7.3
修复：
- 借鉴目标检测样式，使用缓存机制减少闪动
- 修复参数保存覆盖问题（使用合并而非替换）
- 从app.py统一导入dlhub_params（避免多实例问题）
"""

import gradio as gr
import threading, time
import sys
from pathlib import Path

# 从app.py导入共享的dlhub_params实例（延迟导入，在函数内进行）
dlhub_params = None

def _get_dlhub_params():
    """延迟获取dlhub_params，避免循环导入"""
    global dlhub_params
    if dlhub_params is None:
        try:
            from gui.app import dlhub_params as _params
            dlhub_params = _params
        except ImportError:
            pass
    return dlhub_params

training_state = {
    'trainer': None, 
    'is_training': False, 
    'should_stop': False, 
    'logs': [], 
    'current_phase': '', 
    'phase_progress': 0, 
    'total_progress': 0, 
    'current_phase_idx': 0, 
    'total_phases': 6, 
    'start_time': None, 
    'result': None,
}

# 缓存上一次的HTML，避免无变化时的刷新
_cached_html = {
    'progress': None,
    'last_total': -1,
    'last_phase': -1,
    'last_phase_idx': -1,
    'last_is_training': None,
    'final_sent': False,
}


def _load_saved_state():
    """加载保存的状态（延迟获取dlhub_params）"""
    global training_state
    params = _get_dlhub_params()
    if params:
        saved_history = params.get_history()
        saved_logs = params.get_logs('training')
        if saved_history:
            training_state['total_progress'] = saved_history.get('total_progress', 0)
            training_state['current_phase_idx'] = saved_history.get('current_phase_idx', 0)
            training_state['current_phase'] = saved_history.get('current_phase', '')
            if saved_history.get('completed'):
                training_state['total_progress'] = 100
                training_state['current_phase_idx'] = 7
            print(f"[DL-Hub] ✓ 已恢复训练历史: 进度 {training_state['total_progress']}%")
        if saved_logs:
            training_state['logs'] = saved_logs
            print(f"[DL-Hub] ✓ 已恢复训练日志: {len(saved_logs)} 行")

# 注意：不在模块加载时调用，改为在create_status_panel中调用
# _load_saved_state()


def make_progress_html(total_progress, phase_name, phase_progress, phase_idx, elapsed_str, eta_str):
    """创建进度条HTML - 借鉴目标检测样式"""
    
    # 阶段配置
    phases = ['初始化', '特征提取', 'PCA', 'CoreSet', '索引', '导出']
    
    # 生成阶段步骤HTML
    steps_html = ''
    for i, name in enumerate(phases):
        idx = i + 1
        if idx < phase_idx:
            # 已完成
            circle_style = "background: #34a853; border-color: #34a853;"
            icon = "✓"
            text_style = "color: #34a853;"
            line_style = "background: #34a853;"
        elif idx == phase_idx:
            # 当前
            circle_style = "background: linear-gradient(135deg, #667eea, #764ba2); border-color: #667eea;"
            icon = str(idx)
            text_style = "color: #8be9fd; font-weight: 600;"
            line_style = "background: #3d3d5c;"
        else:
            # 未开始
            circle_style = "background: #2d2d44; border-color: #3d3d5c;"
            icon = str(idx)
            text_style = "color: #666;"
            line_style = "background: #3d3d5c;"
        
        steps_html += f'''
            <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
                <div style="width: 36px; height: 36px; border-radius: 50%; {circle_style}
                            border: 2px solid; display: flex; align-items: center; justify-content: center;
                            font-size: 14px; font-weight: 600; color: white;">{icon}</div>
                <div style="margin-top: 6px; font-size: 11px; {text_style} white-space: nowrap;">{name}</div>
            </div>
        '''
        if i < len(phases) - 1:
            steps_html += f'<div style="flex: 1; height: 2px; {line_style} margin-top: 18px;"></div>'
    
    return f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
            <span style="color: #fff; font-weight: bold;">训练进度</span>
            <span style="color: #8be9fd;">{phase_name or '等待开始'} ({phase_idx}/6)</span>
        </div>
        <div style="background: #2d2d44; border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 16px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {total_progress}%;"></div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 20px; padding: 0 10px;">
            {steps_html}
        </div>
        <div style="display: flex; justify-content: space-around; border-top: 1px solid #3d3d5c; padding-top: 15px;">
            <div style="text-align: center;">
                <div style="color: #8be9fd; font-size: 1.5em; font-weight: bold;">{total_progress}%</div>
                <div style="color: #888; font-size: 0.9em;">总进度</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #50fa7b; font-size: 1.5em; font-weight: bold;">{phase_progress}%</div>
                <div style="color: #888; font-size: 0.9em;">阶段进度</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #ff79c6; font-size: 1.5em; font-weight: bold;">{elapsed_str or '--'}</div>
                <div style="color: #888; font-size: 0.9em;">已用时间</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #f1fa8c; font-size: 1.5em; font-weight: bold;">{eta_str or '--'}</div>
                <div style="color: #888; font-size: 0.9em;">预计剩余</div>
            </div>
        </div>
    </div>
    '''


def start_training_task(data_dir_val, output_dir_val, backbone_val, image_size_val, coreset_ratio_val, pca_components_val, knn_k_val, export_tensorrt_val, tensorrt_precision_val, aug_hflip_val, aug_vflip_val, aug_rotate_val, aug_brightness_val, aug_contrast_val, use_fp16_features_val, incremental_pca_val, feature_chunk_size_val, random_projection_val, device_val, num_workers_val):
    global training_state, _cached_html
    from datetime import datetime
    from pathlib import Path
    
    # 获取共享的dlhub_params实例
    params = _get_dlhub_params()
    
    # 重置缓存（包括final_sent标志）
    _cached_html = {'progress': None, 'last_total': -1, 'last_phase': -1, 'last_phase_idx': -1, 'last_is_training': None, 'final_sent': False}
    
    if training_state['is_training']: return False, "训练正在进行中..."
    if not data_dir_val: return False, "请先选择数据集目录"
    
    # 创建带时间戳的输出目录（与分类/检测/分割任务保持一致）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_short = backbone_val.replace('wide_resnet50_2', 'WR50').replace('resnet', 'R')
    output_base = Path(output_dir_val)
    actual_output_dir = output_base / f"PatchCore_{backbone_short}_{timestamp}"
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    
    training_state.update({
        'is_training': True, 'should_stop': False, 'logs': [], 
        'current_phase': '初始化', 'phase_progress': 0, 'total_progress': 0, 
        'current_phase_idx': 1, 'start_time': time.time(), 'result': None,
    })
    
    if params:
        params.clear_history(auto_save=False)
        params.clear_logs('training', auto_save=False)
        params.save_history({
            'total_phases': 6, 'current_phase_idx': 1,
            'output_dir': str(actual_output_dir), 'completed': False
        }, auto_save=True)
        # 修复：使用合并方式保存参数，避免覆盖已有的config等参数
        current_params = params.get_all()
        current_params['data'] = {'data_dir': data_dir_val, 'image_size': image_size_val}
        current_params['model'] = {'backbone': backbone_val, 'coreset_ratio': coreset_ratio_val, 'pca_components': pca_components_val, 'knn_k': knn_k_val}
        current_params['export'] = {'tensorrt': export_tensorrt_val, 'precision': tensorrt_precision_val}
        current_params['device'] = device_val
        params.save(current_params)
    
    try:
        from config import TrainingConfig
        from engine.trainer import PatchCoreTrainer, TrainingCallback
        
        config = TrainingConfig()
        # 使用带时间戳的输出目录
        config.dataset_dir, config.output_dir, config.image_size, config.device = data_dir_val, str(actual_output_dir), int(image_size_val), device_val
        config.backbone.name = backbone_val
        config.memory_bank.coreset_sampling_ratio, config.memory_bank.pca_components = float(coreset_ratio_val) / 100.0, int(pca_components_val)
        config.knn.k = int(knn_k_val)
        config.export.tensorrt_enabled, config.export.tensorrt_precision = export_tensorrt_val, tensorrt_precision_val
        config.augmentation.horizontal_flip, config.augmentation.vertical_flip, config.augmentation.random_rotation = aug_hflip_val, aug_vflip_val, aug_rotate_val
        config.augmentation.brightness, config.augmentation.contrast = aug_brightness_val, aug_contrast_val
        config.optimization.use_fp16_features, config.optimization.incremental_pca = use_fp16_features_val, incremental_pca_val
        config.optimization.feature_chunk_size, config.optimization.random_projection_enabled, config.optimization.num_workers = int(feature_chunk_size_val), random_projection_val, int(num_workers_val)
        
        trainer = PatchCoreTrainer(config)
        training_state['trainer'] = trainer
        
        # 记录输出目录
        training_state['logs'].append(f"📁 输出目录: {actual_output_dir}")
        
        def on_log(msg):
            training_state['logs'].append(msg)
            if len(training_state['logs']) > 100:
                training_state['logs'] = training_state['logs'][-100:]
            if params:
                params.append_log(msg, 'training', auto_save=False)
        
        def on_phase_start(phase_name, current, total):
            training_state.update({
                'current_phase': phase_name, 'current_phase_idx': current, 
                'total_phases': total, 'phase_progress': 0, 
                'total_progress': int((current - 1) / total * 100)
            })
            if params:
                # 保存历史状态
                params.save_history({
                    'current_phase': phase_name, 'current_phase_idx': current,
                    'total_phases': total, 'total_progress': training_state['total_progress']
                }, auto_save=False)
                # 同时保存日志，避免应用关闭时丢失
                params.save_logs(training_state['logs'], 'training', auto_save=True)
        
        def on_progress(current, total, message):
            if total > 0:
                phase_pct = int(current / total * 100)
                training_state['phase_progress'] = phase_pct
                pi, tp = training_state['current_phase_idx'], training_state['total_phases']
                new_total = int((pi - 1) / tp * 100) + int(phase_pct / tp)
                
                # 每10%总进度保存一次日志，避免应用关闭时丢失
                old_total = training_state['total_progress']
                if new_total // 10 > old_total // 10 and params:
                    params.save_logs(training_state['logs'], 'training', auto_save=True)
                    params.save_history({
                        'current_phase': training_state['current_phase'],
                        'current_phase_idx': pi,
                        'total_phases': tp,
                        'total_progress': new_total
                    }, auto_save=True)
                
                training_state['total_progress'] = new_total
        
        def should_stop():
            return training_state['should_stop']
        
        trainer.set_callback(TrainingCallback(on_log=on_log, on_phase_start=on_phase_start, on_progress=on_progress, should_stop=should_stop))
        
        def train_thread():
            try:
                result = trainer.train()
                training_state.update({'result': result, 'total_progress': 100, 'phase_progress': 100, 'current_phase_idx': 7})
                if params:
                    params.mark_training_complete(best_metric=result.default_threshold if result and result.success else None)
            except Exception as e:
                import traceback
                training_state['logs'].append(f"❌ 训练错误: {e}")
                traceback.print_exc()
            finally:
                training_state['is_training'] = False
                if params:
                    params.save_logs(training_state['logs'], 'training', auto_save=True)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return True, "训练已启动"
    except Exception as e:
        import traceback
        traceback.print_exc()
        training_state['is_training'] = False
        return False, f"启动失败: {e}"


def create_status_panel():
    """创建训练状态面板"""
    
    # 延迟加载保存的状态（确保app.py已初始化）
    _load_saved_state()
    
    # ===== 训练控制区 =====
    gr.Markdown("### 🚀 训练控制")
    with gr.Row():
        start_from_status_btn = gr.Button("🚀 开始训练", variant="primary", size="lg", scale=2)
        stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg", scale=1)
    
    train_status = gr.Textbox(label="状态", value="📊 就绪，等待开始", interactive=False, lines=1)
    
    # ===== 进度显示区 =====
    gr.Markdown("### 📈 训练进度")
    progress_html = gr.HTML(value=make_progress_html(0, "", 0, 0, None, None))
    
    # ===== 训练结果 =====
    gr.Markdown("### 📋 训练结果")
    result_summary = gr.Textbox(label="结果摘要", value="训练完成后显示", interactive=False, lines=6)
    
    # ===== 训练日志 =====
    gr.Markdown("### 📝 训练日志")
    log_output = gr.Textbox(label="", lines=12, max_lines=15, interactive=False, show_label=False)
    
    # ===== 辅助函数 =====
    def get_time_str(seconds):
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}时{minutes}分"
    
    def stop_training():
        global training_state
        if not training_state['is_training']:
            return "📊 当前无训练任务"
        training_state['should_stop'] = True
        if training_state['trainer']:
            training_state['trainer'].request_stop()
        return "⏹️ 正在停止..."
    
    def refresh_status():
        """刷新状态 - 使用缓存机制减少闪动"""
        global training_state, _cached_html
        
        # 检查是否需要更新进度HTML
        # 只在进度数值变化时才更新，时间变化不触发重绘
        total = training_state['total_progress']
        phase = training_state['phase_progress']
        phase_idx = training_state['current_phase_idx']
        is_training = training_state['is_training']
        
        # 只有进度变化时才更新HTML（排除时间更新）
        need_update_html = (
            total != _cached_html['last_total'] or
            phase != _cached_html['last_phase'] or
            phase_idx != _cached_html['last_phase_idx'] or
            is_training != _cached_html.get('last_is_training', None)
        )
        
        # 状态文本
        if training_state['is_training']:
            status = f"🏃 训练中 - {training_state['current_phase']}"
        elif training_state['result'] and training_state['result'].success:
            status = "✅ 训练完成"
        elif training_state['result']:
            status = "❌ 训练失败"
        else:
            status = "📊 就绪，等待开始"
        
        # 进度HTML - 只在需要时更新
        if need_update_html:
            # 已用时间
            elapsed_str = None
            eta_str = None
            if training_state['start_time'] and training_state['is_training']:
                elapsed = time.time() - training_state['start_time']
                elapsed_str = get_time_str(elapsed)
                # 预估剩余时间
                if total > 0 and total < 100:
                    estimated_total = elapsed / (total / 100)
                    remaining = estimated_total - elapsed
                    eta_str = get_time_str(max(0, remaining)) if remaining > 0 else "即将完成"
            
            progress = make_progress_html(
                total,
                training_state['current_phase'],
                phase,
                phase_idx,
                elapsed_str,
                eta_str
            )
            _cached_html['progress'] = progress
            _cached_html['last_total'] = total
            _cached_html['last_phase'] = phase
            _cached_html['last_phase_idx'] = phase_idx
            _cached_html['last_is_training'] = is_training
        else:
            progress = gr.update()  # 不更新HTML，避免闪动
        
        # 结果摘要
        if training_state['result'] and training_state['result'].success:
            r = training_state['result']
            result_str = f"""✅ 训练成功完成

图像数量: {r.num_images}
Memory Bank: {r.memory_bank_size:,}
特征维度: {r.feature_dim}-D
默认阈值: {r.default_threshold:.1f}

📦 模型路径: {r.export_path}"""
        elif training_state['result']:
            result_str = "❌ 训练失败，请查看日志"
        else:
            result_str = "训练完成后显示"
        
        # 日志
        logs = "\n".join(training_state['logs'][-30:])
        
        return status, progress, result_str, logs
    
    # ===== 事件绑定 =====
    stop_btn.click(fn=stop_training, outputs=[train_status])
    start_from_status_btn.click(fn=lambda: "📊 请在「数据配置」页面点击开始训练", outputs=[train_status])
    
    # 定时刷新 - 改为3秒，减少闪动
    def auto_refresh():
        # 训练未开始且无结果时，不刷新
        if not training_state['is_training'] and training_state['result'] is None:
            return gr.update(), gr.update(), gr.update(), gr.update()
        # 训练完成后，仅刷新一次然后停止
        if not training_state['is_training'] and training_state['result'] is not None:
            # 返回最终状态后，后续刷新都返回update()
            if _cached_html.get('final_sent'):
                return gr.update(), gr.update(), gr.update(), gr.update()
            _cached_html['final_sent'] = True
        return refresh_status()
    
    refresh_timer = gr.Timer(value=3.0)  # 改为3秒刷新
    refresh_timer.tick(fn=auto_refresh, outputs=[train_status, progress_html, result_summary, log_output])
