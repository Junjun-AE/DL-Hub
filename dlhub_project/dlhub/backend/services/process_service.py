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
import collections
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
            # [P1-3 修复] 用 deque(maxlen) 作为日志缓冲，替代会被破坏性读取的 queue.Queue。
            # deque 自动丢弃最旧的，append 线程安全；读取时拷贝快照、不破坏数据、不腰斩容量。
            self._log_buffer = collections.deque(maxlen=2000)
            self._log_buffer_lock = threading.Lock()
            self._log_thread: Optional[threading.Thread] = None
            self._stopping = False

            # [P1-1 修复] 进程操作锁（可重入）：序列化 launch/stop/状态读写，
            # 避免并发启动竞态、端口竞态、以及日志线程与 _cleanup_state 的状态竞争。
            # 注意与单例创建用的 _lock 区分：这是实例级操作锁。
            self._op_lock = threading.RLock()
            
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
        with self._op_lock:
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
        """清理进程状态（调用方应持有 _op_lock；RLock 可重入）"""
        with self._op_lock:
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
        """启动Gradio应用

        [P1-1 修复] 全程持有 _op_lock：把"检查是否在运行 → 选端口 → 启动进程 → 写状态"
        作为一个原子临界区，避免并发启动时两个请求都通过检查、或抢到同一端口。
        """
        # 整个启动过程串行化（RLock 可重入，内部调 is_app_running/_cleanup_state 安全）
        with self._op_lock:
            return self._launch_app_locked(app_path, task_dir, task_id, conda_env, port)

    def _launch_app_locked(
        self,
        app_path: str,
        task_dir: str,
        task_id: str,
        conda_env: Optional[str] = None,
        port: int = 7861
    ) -> Tuple[bool, int]:
        """launch_app 的实际实现（调用方已持有 _op_lock）。"""
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
            # [P4-5] 启动前检查 conda 是否可用，给出清晰错误而非晦涩失败
            import shutil as _shutil
            if _shutil.which('conda') is None:
                raise RuntimeError(
                    "未找到 conda 命令。请确认：① 已安装 Anaconda/Miniconda；"
                    "② conda 在系统 PATH 中（或从 Anaconda Prompt 启动本平台）。"
                    "若不使用 conda 环境，可在任务设置中清空 Conda 环境。"
                )
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
                # [P1-4 修复] conda 命令现在是 argv 列表(conda run)，不再需要 shell=True。
                # 这样 self._process.pid 就是 conda/python 而非 cmd.exe，杀进程时不会留孤儿 python。
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
            
            # [P1-5 改进] 之前只 sleep(0.5) 检查一次，重型 app(conda激活/mmseg/ultralytics import)
            # 往往 0.5s 内还没崩。改为在 2 秒内每 0.25s 轮询一次，更可能抓到"启动后不久即崩溃"，
            # 同时不显著拖慢正常启动的返回（一旦确认存活立即返回）。完整就绪仍由外层 _wait_for_app_startup 处理。
            early_crash_deadline = time.time() + 2.0
            while time.time() < early_crash_deadline:
                time.sleep(0.25)
                poll = self._process.poll()
                if poll is not None:
                    exit_code = poll
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
    ):
        """构建 conda 启动命令。

        [P4-5 修复] Windows 之前用 `cmd /c "conda activate <env> && python ..."`，
        在非交互 shell 下 `conda activate` 经常失败（需先 conda init）。
        改为两个平台都用 `conda run -n <env> python ...`：
        - `conda run` 不依赖 shell 的 activate 钩子，非交互环境可用；
        - [P1-4] 返回 argv 列表而非字符串，使 Windows 也能不用 shell=True 启动，
          这样进程 PID 就是 conda/python，而不是 cmd.exe（避免杀进程时 python 变孤儿）。
        返回 list[str]（argv），两个平台统一。
        """
        return [
            'conda', 'run', '-n', conda_env, '--no-capture-output',
            'python', str(app_path),
            '--task-dir', str(task_dir),
            '--port', str(port),
        ]
    
    def stop_app(self) -> bool:
        """停止当前运行的应用

        [P1-1 修复] 全程持有 _op_lock，避免与 launch/状态读取竞态。
        """
        with self._op_lock:
            return self._stop_app_locked()

    def _stop_app_locked(self) -> bool:
        """stop_app 实际实现（调用方已持有 _op_lock）。"""
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
        """通过端口查找并杀死进程（Windows / Linux / macOS）"""
        system = platform.system()

        if system != 'Windows':
            # [P1-2 修复] Linux/macOS：用 lsof 找占用端口的 PID 并强杀；
            # lsof 不可用时回退 fuser。之前此处直接 return，导致非 Windows 残留进程杀不掉。
            try:
                # 优先 lsof（macOS/多数 Linux 发行版都有）
                result = subprocess.run(
                    ['lsof', '-ti', f'TCP:{port}', '-sTCP:LISTEN'],
                    capture_output=True, text=True, timeout=10, errors='replace'
                )
                pids = [p.strip() for p in result.stdout.split('\n') if p.strip()]
                if not pids:
                    # 回退：不带状态过滤再试一次
                    result = subprocess.run(
                        ['lsof', '-ti', f':{port}'],
                        capture_output=True, text=True, timeout=10, errors='replace'
                    )
                    pids = [p.strip() for p in result.stdout.split('\n') if p.strip()]
                for pid in pids:
                    try:
                        pid_int = int(pid)
                        if pid_int > 0:
                            log_debug(f"杀死占用端口 {port} 的进程 {pid_int} (lsof)")
                            os.kill(pid_int, signal.SIGKILL)
                    except (ValueError, ProcessLookupError):
                        pass
                    except Exception as e:
                        log_debug(f"kill {pid} 失败: {e}")
            except FileNotFoundError:
                # lsof 不存在，回退 fuser
                try:
                    subprocess.run(
                        ['fuser', '-k', f'{port}/tcp'],
                        capture_output=True, timeout=10
                    )
                    log_debug(f"已用 fuser 清理端口 {port}")
                except Exception as e:
                    log_debug(f"fuser 清理端口 {port} 失败(lsof/fuser 均不可用): {e}")
            except Exception as e:
                log_debug(f"Linux/Mac 通过端口杀进程异常: {e}")
            return

        # Windows 分支
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
        """获取最近的日志（非破坏性快照读取）

        [P1-3 修复] 之前从 queue 取空再回插，导致容量腰斩、与写线程竞争乱序。
        现在直接拷贝 deque 快照返回最后 N 条，不动原数据。
        """
        with self._log_buffer_lock:
            snapshot = list(self._log_buffer)
        if lines is None or lines <= 0:
            return snapshot
        return snapshot[-lines:]
    
    def _start_log_collection(self):
        """启动日志收集线程"""
        if self._log_thread and self._log_thread.is_alive():
            return

        # [P1-3] 清空上一次的日志缓冲
        with self._log_buffer_lock:
            self._log_buffer.clear()

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
                    try:
                        if isinstance(line, bytes):
                            line = line.decode('utf-8', errors='replace')
                        clean_line = line.rstrip()
                        
                        # 添加时间戳
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        formatted_line = f"[{timestamp}] {clean_line}"
                        
                        # [P1-3] deque(maxlen) 自动丢弃最旧的，无需手动处理满队列
                        with self._log_buffer_lock:
                            self._log_buffer.append(formatted_line)
                        
                        # 输出应用日志到控制台
                        print(f"[APP] {clean_line}")
                        
                        # [优化-P6] 通知WebSocket订阅者
                        self._notify_subscribers(formatted_line)
                        
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
        # [P1-3] 先返回历史日志快照，并记录已读位置
        with self._log_buffer_lock:
            snapshot = list(self._log_buffer)
        for log in snapshot[-100:]:
            yield log
        seen = len(snapshot)

        # 然后增量返回新日志（基于 deque 快照长度做游标，不破坏缓冲）
        last_check = time.time()
        while self.is_app_running():
            with self._log_buffer_lock:
                current = list(self._log_buffer)
            cur_len = len(current)
            if cur_len > seen:
                # 有新日志
                for log in current[seen:]:
                    yield log
                seen = cur_len
                last_check = time.time()
            else:
                # deque 可能因 maxlen 滚动导致长度不增，但内容变化；
                # 简单稳健起见：若长度回退（已达上限滚动），重置游标到末尾附近
                if cur_len < seen:
                    seen = cur_len
                time.sleep(0.3)
                # 每5秒发送心跳
                if time.time() - last_check > 5:
                    yield "[HEARTBEAT]"
                    last_check = time.time()
