# -*- coding: utf-8 -*-
"""
应用启动路由 - 增强版
===========
管理Gradio训练应用的启动和停止

修复：
- 详细的日志输出
- 智能等待应用启动
- 更好的错误信息
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import time
import socket
import traceback

from ..services.task_service import TaskService
from ..services.process_service import ProcessService
from ..config import get_app_path

router = APIRouter()

process_service = ProcessService()
task_service = TaskService()


def _build_app_url(request: Optional[Request], port: int) -> str:
    """[P8-3 修复] 用访问者实际使用的主机名生成 URL，而非写死 localhost。

    这样从另一台机器通过服务器 IP 访问 DL-Hub 时，返回的任务界面 URL
    指向服务器而不是访问者自己的机器。无法取得 host 时回退 localhost。
    """
    host = "localhost"
    try:
        if request is not None and request.url and request.url.hostname:
            host = request.url.hostname
    except Exception:
        host = "localhost"
    return f"http://{host}:{port}"

TRAINING_APP_PORT = 7861


def log_debug(msg: str):
    """调试日志"""
    print(f"[AppLauncher] {msg}")


class LaunchResponse(BaseModel):
    success: bool
    url: str
    task_id: str
    message: Optional[str] = None


class AppStatusResponse(BaseModel):
    running: bool
    current_task_id: Optional[str] = None
    url: Optional[str] = None


def _is_port_listening(port: int, timeout: float = 1.0) -> bool:
    """检查端口是否在监听"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex(('127.0.0.1', port))
            return result == 0
    except Exception:
        return False


def _is_http_ready(port: int, timeout: float = 2.0) -> bool:
    """[P8-2] 检查 Gradio 是否真的在响应 HTTP（端口 listening ≠ 页面可用）。

    用极轻量的 HTTP GET 探测；任何 HTTP 响应(含 4xx/5xx)都说明服务已在处理请求。
    探测失败不代表一定没就绪，调用方据此做"宽限"判断。
    """
    try:
        import http.client
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
        conn.request('GET', '/')
        resp = conn.getresponse()
        conn.close()
        return resp.status > 0
    except Exception:
        return False


def _wait_for_app_startup(port: int, max_wait: int = 15) -> bool:
    """等待应用启动完成

    [P8-2 改进] 端口 listening 后再做 HTTP 就绪探测，尽量等到 Gradio 真正能响应，
    降低前端新标签页瞬间命中"连接被拒/502"的概率。在 max_wait 内尽力等 HTTP 就绪，
    若只到达 listening 也在超时附近放行（避免无限等待）。
    """
    log_debug(f"等待应用在端口 {port} 上启动...")
    
    start_time = time.time()
    check_count = 0
    port_listened_at = None
    
    while time.time() - start_time < max_wait:
        check_count += 1
        
        # 检查进程是否还在运行
        if not process_service.is_app_running():
            log_debug(f"检查 #{check_count}: 进程已退出")
            return False
        
        # 检查端口是否可连接
        if _is_port_listening(port):
            if port_listened_at is None:
                port_listened_at = time.time()
            # [P8-2] 端口通了之后，进一步确认 HTTP 可响应
            if _is_http_ready(port):
                elapsed = time.time() - start_time
                log_debug(f"检查 #{check_count}: 端口 {port} HTTP 已就绪 (耗时 {elapsed:.1f}秒)")
                return True
            # HTTP 还没就绪，但端口已通：若已等待超过 6 秒仍只 listening，先放行
            if time.time() - port_listened_at > 6:
                log_debug(f"检查 #{check_count}: 端口已通但HTTP未确认, 宽限放行")
                return True
        
        if check_count % 5 == 0:
            log_debug(f"检查 #{check_count}: 等待中...")
        
        time.sleep(0.5)
    
    # 超时
    elapsed = time.time() - start_time
    is_running = process_service.is_app_running()
    is_listening = _is_port_listening(port)
    log_debug(f"等待超时 ({elapsed:.1f}秒): 进程运行={is_running}, 端口监听={is_listening}")
    
    return is_running and is_listening


@router.post("/launch/{task_id}", response_model=LaunchResponse)
async def launch_app(task_id: str, force: bool = False, request: Request = None):
    """启动训练应用"""
    log_debug(f"=== 启动任务: {task_id}, force={force} ===")
    
    try:
        # 获取任务信息
        task = task_service.get_task(task_id)
        
        if not task:
            log_debug(f"任务不存在: {task_id}")
            raise HTTPException(404, f"任务不存在: {task_id}")
        
        task_type = task['task_type']
        task_dir = task['path']
        conda_env = task.get('conda_env')
        
        log_debug(f"任务信息: type={task_type}, dir={task_dir}, env={conda_env}")
        
        # 检查是否有正在运行的应用
        if process_service.is_app_running():
            current_task_id = process_service.current_task_id
            log_debug(f"已有应用在运行: {current_task_id}")
            
            if current_task_id == task_id:
                current_port = process_service.current_port
                log_debug(f"同一任务已在运行，端口: {current_port}")
                return LaunchResponse(
                    success=True,
                    url=_build_app_url(request, current_port),
                    task_id=task_id,
                    message="任务已在运行中"
                )
            
            if not force:
                current_task = task_service.get_task(current_task_id) if current_task_id else None
                current_task_name = current_task.get('task_name', '未知任务') if current_task else '未知任务'
                log_debug(f"返回冲突错误，当前任务: {current_task_name}")
                raise HTTPException(409, f"APP_RUNNING:{current_task_id}:{current_task_name}")
            
            # 强制启动，先停止当前应用
            log_debug("强制停止当前应用...")
            process_service.stop_app()
            time.sleep(3)
        
        # 获取应用路径
        app_path = get_app_path(task_type)
        log_debug(f"应用路径: {app_path}")
        
        if not app_path:
            from ..config import DEFAULT_APP_PATHS
            expected_path = DEFAULT_APP_PATHS.get(task_type, "未定义")
            raise HTTPException(
                500, 
                f"未找到 {task_type} 类型的训练应用。期望路径: {expected_path}"
            )
        
        if not app_path.exists():
            raise HTTPException(500, f"应用文件不存在: {app_path}")
        
        # 启动应用
        log_debug("开始启动应用...")
        try:
            success, actual_port = process_service.launch_app(
                app_path=str(app_path),
                task_dir=task_dir,
                task_id=task_id,
                conda_env=conda_env,
                port=TRAINING_APP_PORT
            )
            log_debug(f"launch_app 返回: success={success}, port={actual_port}")
        except RuntimeError as e:
            error_msg = str(e)
            log_debug(f"启动RuntimeError: {error_msg}")
            if "已有应用在运行" in error_msg:
                raise HTTPException(409, "有其他应用正在运行，请先停止或使用强制启动")
            # 直接返回具体的错误信息，包括日志
            raise HTTPException(500, f"启动失败: {error_msg}")
        except Exception as e:
            log_debug(f"启动异常: {e}")
            log_debug(traceback.format_exc())
            raise HTTPException(500, f"启动失败: {e}")
        
        # 等待应用启动完成
        startup_ok = _wait_for_app_startup(actual_port, max_wait=15)
        
        if not startup_ok:
            # 检查进程状态
            if not process_service.is_app_running():
                logs = process_service.get_recent_logs(30)
                log_text = "\n".join(logs[-15:]) if logs else "无日志"
                log_debug(f"应用启动后退出")
                raise HTTPException(500, f"应用启动后退出，请检查环境配置:\n{log_text}")
        
        url = _build_app_url(request, actual_port)
        log_debug(f"应用URL: {url}")
        
        # 最终检查
        if _is_port_listening(actual_port):
            message = f"应用已启动 (端口: {actual_port})"
        else:
            message = f"应用正在启动中，请稍候刷新页面 (端口: {actual_port})"
        
        return LaunchResponse(
            success=True,
            url=url,
            task_id=task_id,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_debug(f"未预期的异常: {e}")
        log_debug(traceback.format_exc())
        raise HTTPException(500, f"服务器错误: {e}")


@router.post("/stop")
async def stop_app():
    """停止当前运行的应用"""
    log_debug("=== 停止应用请求 ===")
    
    if not process_service.is_app_running():
        log_debug("没有运行中的应用")
        return {"success": True, "message": "没有正在运行的应用"}
    
    success = process_service.stop_app()
    log_debug(f"停止结果: {success}")
    
    if success:
        return {"success": True, "message": "应用已停止"}
    else:
        raise HTTPException(500, "停止应用失败")


@router.get("/status", response_model=AppStatusResponse)
async def get_app_status(request: Request = None):
    """获取应用运行状态"""
    running = process_service.is_app_running()
    current_task_id = process_service.current_task_id if running else None
    current_port = process_service.current_port if running else None
    url = _build_app_url(request, current_port) if running and current_port else None
    
    return AppStatusResponse(
        running=running,
        current_task_id=current_task_id,
        url=url
    )


@router.get("/logs")
async def get_app_logs(lines: int = 100):
    """获取应用日志"""
    logs = process_service.get_recent_logs(lines)
    return {
        "logs": logs,
        "running": process_service.is_app_running()
    }


@router.post("/restart/{task_id}")
async def restart_app(task_id: str):
    """重启应用"""
    log_debug(f"=== 重启任务: {task_id} ===")
    
    if process_service.is_app_running():
        process_service.stop_app()
        time.sleep(3)
    
    return await launch_app(task_id)
