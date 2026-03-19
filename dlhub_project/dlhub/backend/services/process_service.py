# -*- coding: utf-8 -*-
"""
进程服务 - 增强版
=======
管理Gradio训练应用的进程
支持使用指定的Conda环境

修复：
- 端口冲突自动切换
- 详细的日志输出帮助诊断问题
- 更彻底的进程清理
- [Bug-1修复] 线程安全的单例模式
- [优化-P6] WebSocket实时日志推送支持
"""

from pathlib import Path
from typing import Optional, List, Tuple, Callable, Set
import subprocess
import sys
import threading
import time
import os
import signal
import queue
import platform
import socket
import asyncio
from datetime import datetime


def log_debug(msg: str):
    """调试日志 - 会输出到控制台帮助诊断"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}][ProcessService] {msg}")


class ProcessService:
    """
    进程管理服务（线程安全单例模式）
    
    [Bug-1修复]: 使用完全线程安全的单例实现，
    将所有检查都放在锁内部，避免竞态条件。
    """
    
    _instance = None
    _lock = threading.RLock()  # 使用可重入锁，防止死锁
    _init_lock = threading.Lock()  # 初始化专用锁
    
    def __new__(cls):
        # [Bug-1修复] 将所有检查都放在锁内部，确保线程安全
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        # 使用独立的初始化锁，防止重复初始化
        with self._init_lock:
            if self._initialized:
                return
            
            self._process: Optional[subprocess.Popen] = None
            self._current_task_id: Optional[str] = None
            self._current_conda_env: Optional[str] = None
            self._current_port: Optional[int] = None
            self._log_queue: queue.Queue = queue.Queue(maxsize=2000)  # 增大队列
            self._log_thread: Optional[threading.Thread] = None
            self._stopping = False
            
            # [优化-P6] WebSocket日志订阅者
            self._log_subscribers: Set[Callable[[str], None]] = set()
            self._subscriber_lock = threading.Lock()
            
            self._initialized = True
            log_debug("ProcessService 初始化完成")
    
    # ==================== [优化-P6] WebSocket日志推送 ====================
    
    def subscribe_logs(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """
        订阅实时日志
        
        Args:
            callback: 接收日志的回调函数
            
        Returns:
            取消订阅的函数
        """
        with self._subscriber_lock:
            self._log_subscribers.add(callback)
            log_debug(f"新的日志订阅者，当前订阅数: {len(self._log_subscribers)}")
        
        def unsubscribe():
            with self._subscriber_lock:
                self._log_subscribers.discard(callback)
                log_debug(f"取消日志订阅，剩余订阅数: {len(self._log_subscribers)}")
        
        return unsubscribe
    
    def _notify_subscribers(self, log_line: str):
        """通知所有日志订阅者"""
        with self._subscriber_lock:
            for callback in list(self._log_subscribers):
                try:
                    callback(log_line)
                except Exception as e:
                    log_debug(f"通知订阅者失败: {e}")
    
    @property
    def current_task_id(self) -> Optional[str]:
        return self._current_task_id
    
    @property
    def current_conda_env(self) -> Optional[str]:
        return self._current_conda_env
    
    @property
    def current_port(self) -> Optional[int]:
        return self._current_port
    
    def is_app_running(self) -> bool:
        """检查应用是否在运行"""
        if self._stopping:
            return False
            
        if self._process is None:
            return False
        
        poll_result = self._process.poll()
        if poll_result is not None:
            log_debug(f"进程已退出，退出码: {poll_result}")
            self._cleanup_state()
            return False
        
        return True
    
    def _cleanup_state(self):
        """清理进程状态"""
        self._process = None
        self._current_task_id = None
        self._current_conda_env = None
        self._current_port = None
        self._stopping = False
    
    def find_available_port(self, start_port: int = 7861, max_attempts: int = 50) -> int:
        """查找可用端口"""
        tried_ports = []
        for port in range(start_port, start_port + max_attempts):
            if self._is_port_available(port):
                if tried_ports:
                    log_debug(f"端口 {tried_ports[:5]}... 被占用，使用端口 {port}")
                return port
            tried_ports.append(port)
        
        raise RuntimeError(f"无法找到可用端口 (尝试了 {start_port}-{start_port + max_attempts - 1})")
    
    def _is_port_available(self, port: int) -> bool:
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(('127.0.0.1', port))
                return True
        except (OSError, socket.error):
            return False
    
    def _wait_for_port_release(self, port: int, timeout: int = 10) -> bool:
        """等待端口释放"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_port_available(port):
                return True
            time.sleep(0.5)
        return False
    
    def launch_app(
        self,
        app_path: str,
        task_dir: str,
        task_id: str,
        conda_env: Optional[str] = None,
        port: int = 7861
    ) -> Tuple[bool, int]:
        """启动Gradio应用"""
        log_debug(f"=== 启动应用 ===")
        log_debug(f"  app_path: {app_path}")
        log_debug(f"  task_dir: {task_dir}")
        log_debug(f"  task_id: {task_id}")
        log_debug(f"  conda_env: {conda_env}")
        
        # 检查是否有应用在运行
        if self._process is not None:
            poll_result = self._process.poll()
            if poll_result is None:
                raise RuntimeError("已有应用在运行，请先停止")
            else:
                self._cleanup_state()
        
        app_path = Path(app_path)
        task_dir = Path(task_dir)
        
        if not app_path.exists():
            raise FileNotFoundError(f"应用文件不存在: {app_path}")
        
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找可用端口
        try:
            actual_port = self.find_available_port(port)
            log_debug(f"使用端口: {actual_port}")
        except RuntimeError as e:
            raise RuntimeError(f"无法启动应用: {e}")
        
        cwd = app_path.parent
        
        # 设置环境变量
        env = os.environ.copy()
        env['DLHUB_TASK_DIR'] = str(task_dir)
        env['DLHUB_TASK_ID'] = task_id
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        # 【关键修复】绕过系统代理，防止Gradio 6.0自检请求被代理拦截返回502
        env['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
        env['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
        
        # 传递用户信息给子进程（供用户组件显示）
        # 从 users.json 读取最近登录的用户信息
        try:
            from .auth_service import _load_users
            users_data = _load_users()
            users_list = users_data.get('users', [])
            # 找最近登录的用户
            latest = None
            for u in users_list:
                if u.get('last_login'):
                    if latest is None or u['last_login'] > latest.get('last_login', ''):
                        latest = u
            if latest:
                env['DLHUB_USER_NAME'] = latest.get('username', '')
                env['DLHUB_USER_NICKNAME'] = latest.get('nickname', latest.get('username', ''))
        except Exception:
            pass
        
        # 构建命令
        if conda_env:
            cmd = self._build_conda_command(conda_env, app_path, task_dir, actual_port)
            log_debug(f"Conda命令: {cmd}")
        else:
            cmd = [
                sys.executable,
                str(app_path),
                '--task-dir', str(task_dir),
                '--port', str(actual_port)
            ]
            log_debug(f"命令: {' '.join(cmd)}")
        
        # 启动进程
        try:
            if platform.system() == 'Windows':
                if conda_env:
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=str(cwd),
                        env=env,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1,
                        shell=True,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=str(cwd),
                        env=env,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
            else:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd),
                    env=env,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
            
            log_debug(f"进程已启动，PID: {self._process.pid}")
            
            self._current_task_id = task_id
            self._current_conda_env = conda_env
            self._current_port = actual_port
            self._stopping = False
            
            # 启动日志收集线程
            self._start_log_collection()
            
            # 等待一小段时间，检查进程是否立即崩溃
            time.sleep(0.5)
            if self._process.poll() is not None:
                exit_code = self._process.poll()
                logs = self.get_recent_logs(30)
                log_text = "\n".join(logs) if logs else "无日志"
                self._cleanup_state()
                raise RuntimeError(f"应用启动后立即退出 (退出码: {exit_code})\n日志:\n{log_text}")
            
            return True, actual_port
            
        except RuntimeError:
            raise
        except Exception as e:
            log_debug(f"启动异常: {e}")
            self._cleanup_state()
            raise RuntimeError(f"启动应用失败: {e}")
    
    def _build_conda_command(
        self, 
        conda_env: str, 
        app_path: Path, 
        task_dir: Path, 
        port: int
    ) -> str:
        app_path_str = str(app_path)
        task_dir_str = str(task_dir)
        
        if platform.system() == 'Windows':
            if ' ' in app_path_str:
                app_path_str = f'"{app_path_str}"'
            if ' ' in task_dir_str:
                task_dir_str = f'"{task_dir_str}"'
            
            cmd = (
                f'cmd /c "conda activate {conda_env} && '
                f'python {app_path_str} --task-dir {task_dir_str} --port {port}"'
            )
            return cmd
        else:
            return [
                'conda', 'run', '-n', conda_env, '--no-capture-output',
                'python', str(app_path),
                '--task-dir', str(task_dir),
                '--port', str(port)
            ]
    
    def stop_app(self) -> bool:
        """停止当前运行的应用"""
        log_debug("=== 停止应用 ===")
        
        self._stopping = True
        
        if self._process is None:
            self._cleanup_state()
            return True
        
        if self._process.poll() is not None:
            self._cleanup_state()
            return True
        
        saved_port = self._current_port
        saved_pid = self._process.pid
        log_debug(f"停止进程 PID={saved_pid}, 端口={saved_port}")
        
        try:
            if platform.system() == 'Windows':
                # 使用taskkill终止进程树
                try:
                    result = subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(saved_pid)],
                        capture_output=True,
                        timeout=10
                    )
                    log_debug(f"taskkill 返回码: {result.returncode}")
                except Exception as e:
                    log_debug(f"taskkill 失败: {e}")
                    try:
                        self._process.kill()
                    except Exception:
                        pass
            else:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                except Exception as e:
                    log_debug(f"killpg 失败: {e}")
                    try:
                        self._process.terminate()
                    except Exception:
                        pass
            
            # 等待进程退出
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass
            
            self._cleanup_state()
            
            # 等待端口释放
            if saved_port:
                log_debug(f"等待端口 {saved_port} 释放...")
                if self._wait_for_port_release(saved_port, timeout=8):
                    log_debug(f"端口 {saved_port} 已释放")
                else:
                    log_debug(f"警告: 端口 {saved_port} 可能仍被占用")
                    self._kill_process_by_port(saved_port)
            
            return True
            
        except Exception as e:
            log_debug(f"停止应用异常: {e}")
            try:
                if self._process:
                    self._process.kill()
            except Exception:
                pass
            
            self._cleanup_state()
            
            if saved_port:
                self._kill_process_by_port(saved_port)
                time.sleep(2)
            
            return False
    
    def _kill_process_by_port(self, port: int):
        """通过端口查找并杀死进程"""
        if platform.system() != 'Windows':
            return
        
        try:
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                pid_int = int(pid)
                                if pid_int > 0:
                                    log_debug(f"杀死占用端口 {port} 的进程 {pid}")
                                    subprocess.run(
                                        ['taskkill', '/F', '/PID', pid],
                                        capture_output=True,
                                        timeout=5
                                    )
                            except Exception:
                                pass
        except Exception as e:
            log_debug(f"通过端口杀进程异常: {e}")
    
    def get_recent_logs(self, lines: int = 100) -> List[str]:
        """获取最近的日志"""
        logs = []
        
        while not self._log_queue.empty():
            try:
                logs.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        
        for log in logs[-1000:]:
            try:
                self._log_queue.put_nowait(log)
            except queue.Full:
                break
        
        return logs[-lines:]
    
    def _start_log_collection(self):
        """启动日志收集线程"""
        if self._log_thread and self._log_thread.is_alive():
            return
        
        while not self._log_queue.empty():
            try:
                self._log_queue.get_nowait()
            except queue.Empty:
                break
        
        self._log_thread = threading.Thread(
            target=self._collect_logs,
            daemon=True
        )
        self._log_thread.start()
    
    def _collect_logs(self):
        """收集进程输出日志"""
        if not self._process or not self._process.stdout:
            return
        
        try:
            for line in self._process.stdout:
                if line:
                    while self._log_queue.full():
                        try:
                            self._log_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    try:
                        if isinstance(line, bytes):
                            line = line.decode('utf-8', errors='replace')
                        clean_line = line.rstrip()
                        
                        # 添加时间戳
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        formatted_line = f"[{timestamp}] {clean_line}"
                        
                        self._log_queue.put_nowait(formatted_line)
                        
                        # 输出应用日志到控制台
                        print(f"[APP] {clean_line}")
                        
                        # [优化-P6] 通知WebSocket订阅者
                        self._notify_subscribers(formatted_line)
                        
                    except queue.Full:
                        pass
                    except Exception as e:
                        print(f"[日志错误] {e}")
        except Exception as e:
            print(f"[日志收集错误] {e}")
    
    def get_log_stream(self):
        """
        [优化-P6] 获取日志流生成器，用于WebSocket
        
        Yields:
            str: 日志行
        """
        # 先返回历史日志
        for log in self.get_recent_logs(100):
            yield log
        
        # 然后返回新日志（通过轮询队列）
        last_check = time.time()
        while self.is_app_running():
            try:
                log = self._log_queue.get(timeout=0.5)
                yield log
            except queue.Empty:
                # 每5秒发送心跳
                if time.time() - last_check > 5:
                    yield "[HEARTBEAT]"
                    last_check = time.time()
