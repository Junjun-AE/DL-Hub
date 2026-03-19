# -*- coding: utf-8 -*-
"""
DL-Hub Backend - FastAPI 主入口
==============================

[优化-P4] 支持配置热重载
[优化-P6] 支持WebSocket实时日志推送
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import logging
import asyncio
import json
from typing import Set, Dict
from datetime import datetime

from .routers import workspace, tasks, app_launcher, system, auth
from .config import get_config, CONFIG_FILE, reload_config, start_config_watcher, stop_config_watcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dlhub")

# 创建FastAPI应用
app = FastAPI(
    title="DL-Hub API",
    description="深度学习任务管理平台后端API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS配置 - 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 服务器启动密钥 ====================
# 每次服务器启动生成新的随机密钥
# 登录时token包含此密钥前缀，验证时检查前缀是否匹配
# 服务器重启 → 密钥变了 → 旧cookie失效 → 必须重新登录
import secrets as _secrets
SERVER_SECRET = _secrets.token_hex(8)  # 每次启动都不同
logger.info(f"🔑 服务器会话密钥已生成 (重启后旧登录失效)")


# ==================== 认证中间件 ====================
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    
    skip_paths = [
        '/api/auth/', '/api/health', '/api/docs', '/api/redoc', '/openapi.json',
        '/assets/', '/manuals/', '/logout', '/ws/',
        '/vite.svg', '/favicon.ico',
    ]
    
    need_auth = path.startswith('/api/') and not any(path.startswith(s) for s in skip_paths)
    
    if need_auth:
        token = request.cookies.get("dlhub_token")
        # 验证 token 包含当前服务器密钥前缀
        if not token or not token.startswith(SERVER_SECRET):
            return JSONResponse(
                status_code=401,
                content={"detail": "未登录或会话已过期", "redirect": "/"},
                headers={"X-Auth-Required": "true"},
            )
    
    response = await call_next(request)
    return response


# ==================== [优化-P6] WebSocket连接管理器 ====================

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # task_id -> set of websockets
        self.task_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """建立连接"""
        await websocket.accept()
        async with self._lock:
            if task_id not in self.task_connections:
                self.task_connections[task_id] = set()
            self.task_connections[task_id].add(websocket)
        logger.info(f"WebSocket连接建立 [task={task_id}]，当前连接数: {len(self.task_connections.get(task_id, set()))}")
    
    async def disconnect(self, websocket: WebSocket, task_id: str):
        """断开连接"""
        async with self._lock:
            if task_id in self.task_connections:
                self.task_connections[task_id].discard(websocket)
                if not self.task_connections[task_id]:
                    del self.task_connections[task_id]
        logger.info(f"WebSocket连接断开 [task={task_id}]")
    
    async def send_to_task(self, task_id: str, message: str):
        """发送消息给特定任务的所有连接"""
        async with self._lock:
            if task_id not in self.task_connections:
                return
            
            dead_connections = set()
            for connection in self.task_connections[task_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    dead_connections.add(connection)
            
            # 清理断开的连接
            self.task_connections[task_id] -= dead_connections
    
    async def broadcast_all(self, message: str):
        """广播消息给所有连接"""
        async with self._lock:
            for task_id, connections in self.task_connections.items():
                dead_connections = set()
                for connection in connections:
                    try:
                        await connection.send_text(message)
                    except Exception:
                        dead_connections.add(connection)
                connections -= dead_connections

manager = ConnectionManager()


# 注册API路由
app.include_router(auth.router, prefix="/api/auth", tags=["用户认证"])
app.include_router(workspace.router, prefix="/api/workspace", tags=["工作空间"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["任务管理"])
app.include_router(app_launcher.router, prefix="/api/app", tags=["应用启动"])
app.include_router(system.router, prefix="/api/system", tags=["系统信息"])


# ==================== [优化-P6] WebSocket日志端点 ====================

@app.websocket("/ws/logs/{task_id}")
async def websocket_logs(websocket: WebSocket, task_id: str):
    """
    [优化-P6] WebSocket实时日志推送
    
    客户端连接后会收到:
    1. 历史日志（最近100条）
    2. 实时新日志
    3. 每5秒的心跳消息
    """
    await manager.connect(websocket, task_id)
    
    try:
        # 获取ProcessService
        from .services.process_service import ProcessService
        process_service = ProcessService()
        
        # 检查是否是当前运行的任务
        if process_service.current_task_id != task_id:
            await websocket.send_json({
                "type": "error",
                "message": f"任务 {task_id} 未在运行"
            })
            await manager.disconnect(websocket, task_id)
            return
        
        # 发送历史日志
        history_logs = process_service.get_recent_logs(100)
        await websocket.send_json({
            "type": "history",
            "logs": history_logs,
            "count": len(history_logs)
        })
        
        # 注册日志回调
        log_queue = asyncio.Queue()
        
        def on_new_log(log_line: str):
            try:
                asyncio.get_event_loop().call_soon_threadsafe(
                    log_queue.put_nowait, log_line
                )
            except Exception:
                pass
        
        unsubscribe = process_service.subscribe_logs(on_new_log)
        
        try:
            # 心跳和日志推送循环
            last_heartbeat = asyncio.get_event_loop().time()
            
            while True:
                try:
                    # 等待新日志（带超时）
                    log_line = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                    await websocket.send_json({
                        "type": "log",
                        "content": log_line,
                        "timestamp": datetime.now().isoformat()
                    })
                except asyncio.TimeoutError:
                    # 检查是否需要发送心跳
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat > 5:
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.now().isoformat()
                        })
                        last_heartbeat = current_time
                    
                    # 检查进程是否还在运行
                    if not process_service.is_app_running():
                        await websocket.send_json({
                            "type": "status",
                            "status": "stopped",
                            "message": "训练进程已停止"
                        })
                        break
                
                # 检查客户端消息（非阻塞）
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                    # 处理客户端命令（如停止训练）
                    try:
                        cmd = json.loads(data)
                        if cmd.get("action") == "stop":
                            process_service.stop_app()
                            await websocket.send_json({
                                "type": "status",
                                "status": "stopping",
                                "message": "正在停止训练..."
                            })
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass
                    
        finally:
            unsubscribe()
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket客户端断开 [task={task_id}]")
    except Exception as e:
        logger.error(f"WebSocket错误 [task={task_id}]: {e}")
    finally:
        await manager.disconnect(websocket, task_id)


# ==================== API端点 ====================

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "service": "DL-Hub",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/config")
async def get_app_config():
    """获取应用配置"""
    config = get_config()
    return {
        "configured": True,
        "tasks_count": len(config.get("tasks", [])),
        "version": config.get("version", "2.0")
    }


@app.post("/api/config/reload")
async def reload_app_config():
    """
    [优化-P4] 手动重新加载配置
    """
    new_config = reload_config()
    return {
        "success": True,
        "message": "配置已重新加载",
        "tasks_count": len(new_config.get("tasks", []))
    }


# ==================== 用户指南路由 ====================

# 用户指南目录 - 支持多种可能的目录名
def get_manual_dir():
    """获取用户指南目录，支持多种命名"""
    base_dir = Path(__file__).parent.parent.parent
    # 可能的目录名称（优先级从高到低）
    possible_names = ["user_manual", "用户指南", "manuals", "#U7528#U6237#U6307#U5357"]
    for name in possible_names:
        manual_dir = base_dir / name
        if manual_dir.exists():
            logger.info(f"📖 用户指南目录: {manual_dir}")
            return manual_dir
    # 如果都不存在，返回默认路径
    logger.warning(f"⚠️ 用户指南目录不存在，默认路径: {base_dir / 'user_manual'}")
    return base_dir / "user_manual"

USER_MANUAL_DIR = get_manual_dir()

# 任务ID到手册文件名的映射
MANUAL_FILE_MAP = {
    "classification": "classification-manual.html",
    "detection": "detection-manual.html",
    "segmentation": "segmentation-manual.html",
    "anomaly": "anomaly-detection-manual.html",
    "ocr": "ocr-manual.html",
    "sevseg": "sevseg-manual.html",
}

# 挂载用户手册为静态文件（直接访问，更快）
if USER_MANUAL_DIR.exists():
    app.mount("/manuals", StaticFiles(directory=str(USER_MANUAL_DIR), html=True), name="manuals")


@app.get("/api/manual/{task_type}")
async def get_user_manual(task_type: str):
    """
    获取用户指南HTML文件
    
    Args:
        task_type: 任务类型 (classification, detection, segmentation, anomaly, ocr)
    """
    if task_type not in MANUAL_FILE_MAP:
        return JSONResponse(
            status_code=404,
            content={"error": f"未找到 {task_type} 的使用指南"}
        )
    
    manual_file = USER_MANUAL_DIR / MANUAL_FILE_MAP[task_type]
    
    if not manual_file.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"使用指南文件不存在: {manual_file.name}"}
        )
    
    return FileResponse(
        manual_file,
        media_type="text/html"
    )


@app.get("/api/manuals")
async def list_available_manuals():
    """列出所有可用的使用指南"""
    available = {}
    for task_type, filename in MANUAL_FILE_MAP.items():
        manual_file = USER_MANUAL_DIR / filename
        available[task_type] = {
            "filename": filename,
            "exists": manual_file.exists(),
            "url": f"/api/manual/{task_type}"
        }
    return {"manuals": available, "directory": str(USER_MANUAL_DIR)}


# ==================== 静态文件服务（服务端 Cookie 认证门控） ====================

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist.exists():
    # 挂载静态资源（JS/CSS 可缓存）
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # CDN 离线资源目录（login 页面的 React/Babel/Tailwind）
    vendor_dir = frontend_dist / "vendor"
    if vendor_dir.exists():
        app.mount("/vendor", StaticFiles(directory=str(vendor_dir)), name="vendor")

    _NO_CACHE = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    _AUTH_COOKIE = "dlhub_token"

    def _is_valid_token(request: Request) -> bool:
        """检查cookie中的token是否包含当前服务器密钥"""
        token = request.cookies.get(_AUTH_COOKIE)
        return bool(token and token.startswith(SERVER_SECRET))

    @app.get("/")
    async def serve_login(request: Request):
        """根路径：有效cookie→302到/app，无→登录页"""
        if _is_valid_token(request):
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/app", status_code=302)
        login_page = frontend_dist / "login.html"
        if login_page.exists():
            return FileResponse(login_page, headers=_NO_CACHE)
        return FileResponse(frontend_dist / "index.html", headers=_NO_CACHE)

    @app.get("/app")
    async def serve_app(request: Request):
        """主应用：有效cookie→返回页面，无→302到登录页"""
        if not _is_valid_token(request):
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/", status_code=302)
        return FileResponse(frontend_dist / "index.html", headers=_NO_CACHE)

    @app.get("/logout")
    async def serve_logout():
        """注销：清cookie + 直接返回登录页（不用302重定向，避免中间态请求泄漏）"""
        login_page = frontend_dist / "login.html"
        if login_page.exists():
            resp = FileResponse(login_page, headers=_NO_CACHE)
        else:
            resp = FileResponse(frontend_dist / "index.html", headers=_NO_CACHE)
        # 清除认证cookie（必须与set_cookie使用相同的path/httponly/samesite参数）
        resp.delete_cookie(key=_AUTH_COOKIE, path="/", httponly=True, samesite="lax")
        return resp

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        """SPA兜底"""
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return JSONResponse({"error": "Not Found"}, status_code=404)
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        if full_path.startswith("app"):
            if not _is_valid_token(request):
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url="/", status_code=302)
            return FileResponse(frontend_dist / "index.html", headers=_NO_CACHE)
        login_page = frontend_dist / "login.html"
        if login_page.exists():
            return FileResponse(login_page, headers=_NO_CACHE)
        return FileResponse(frontend_dist / "index.html", headers=_NO_CACHE)
else:
    @app.get("/")
    async def no_frontend():
        """前端未构建时的提示"""
        return JSONResponse({
            "message": "DL-Hub API 正在运行",
            "note": "前端尚未构建，请运行 'cd dlhub/frontend && npm install && npm run build'",
            "api_docs": "/api/docs",
            "websocket": "ws://host/ws/logs/{task_id}"
        })


# ==================== 异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )


# ==================== 生命周期事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("🚀 DL-Hub Backend 启动中...")
    logger.info(f"📁 配置文件: {CONFIG_FILE}")
    
    config = get_config()
    tasks_count = len(config.get("tasks", []))
    logger.info(f"📋 已注册任务数: {tasks_count}")
    
    # [优化-P4] 启动配置文件监视器
    start_config_watcher(interval=2.0)
    
    if frontend_dist.exists():
        logger.info(f"🎨 前端目录: {frontend_dist}")
    else:
        logger.warning("⚠️  前端尚未构建")
    
    logger.info("🔌 WebSocket日志端点: /ws/logs/{task_id}")
    logger.info("✅ DL-Hub Backend v2.0 启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("👋 DL-Hub Backend 正在关闭...")
    
    # [优化-P4] 停止配置文件监视器
    stop_config_watcher()
    
    # 停止所有运行中的训练进程
    from .services.process_service import ProcessService
    process_service = ProcessService()
    if process_service.is_app_running():
        logger.info("🛑 停止运行中的训练应用...")
        process_service.stop_app()
    
    logger.info("✅ DL-Hub Backend 已关闭")
