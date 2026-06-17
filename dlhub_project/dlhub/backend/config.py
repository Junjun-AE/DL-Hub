# -*- coding: utf-8 -*-
"""
DL-Hub 配置管理
===============
管理应用配置和任务路径

修复：
- 路径比对使用标准化格式
- 添加调试日志
- 确保保存成功

[优化-P4] 支持配置热重载
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import os
import threading
import time

# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent.parent / "dlhub_config.json"

# 配置缓存
_config_cache: Optional[Dict[str, Any]] = None
_config_mtime: float = 0
_config_size: int = -1  # [P4-4] 配合 mtime 做缓存校验
_config_lock = threading.Lock()

# [优化-P4] 配置变更回调列表
_config_change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
_config_watcher_thread: Optional[threading.Thread] = None
_config_watcher_running = False

# DL-Hub签名，用于验证任务目录
DLHUB_SIGNATURE = "dlhub_v2_task"

# 默认应用路径配置
# 修复：路径与实际文件位置对应
DEFAULT_APP_PATHS = {
    'classification': 'model_image_classification/app.py',      # 修复：移除 gui/
    'detection': 'model_image_detection/app.py',                # 修复：移除 gui/
    'segmentation': 'model_image_segmentation/app.py',          # 修复：移除 gui/
    'anomaly': 'model_image_patchcore/app.py',                  # 修复：目录名改为 patchcore
    'ocr': 'model_image_ocr/app.py',                            # 保持不变
    'sevseg': 'model_image_sevseg/app.py',                      # SevSeg-YOLO 工业缺陷检测
}


# ==================== [优化-P4] 配置热重载 ====================

def reload_config() -> Dict[str, Any]:
    """
    [优化-P4] 强制重新加载配置文件
    
    Returns:
        重新加载的配置
    """
    global _config_cache, _config_mtime, _config_size
    
    with _config_lock:
        _config_cache = None
        _config_mtime = 0
        _config_size = -1
    
    return get_config()


def register_config_change_callback(callback: Callable[[Dict[str, Any]], None]):
    """
    [优化-P4] 注册配置变更回调
    
    Args:
        callback: 当配置变更时调用的函数
    """
    global _config_change_callbacks
    if callback not in _config_change_callbacks:
        _config_change_callbacks.append(callback)
        print(f"[Config] 注册配置变更回调，当前回调数: {len(_config_change_callbacks)}")


def unregister_config_change_callback(callback: Callable[[Dict[str, Any]], None]):
    """
    [优化-P4] 取消注册配置变更回调
    """
    global _config_change_callbacks
    if callback in _config_change_callbacks:
        _config_change_callbacks.remove(callback)


def start_config_watcher(interval: float = 2.0):
    """
    [优化-P4] 启动配置文件监视器
    
    Args:
        interval: 检查间隔（秒）
    """
    global _config_watcher_thread, _config_watcher_running
    
    if _config_watcher_thread is not None and _config_watcher_thread.is_alive():
        return  # 已经在运行
    
    _config_watcher_running = True
    
    def watcher():
        global _config_mtime
        last_mtime = _config_mtime
        
        while _config_watcher_running:
            try:
                if CONFIG_FILE.exists():
                    current_mtime = CONFIG_FILE.stat().st_mtime
                    if current_mtime != last_mtime and last_mtime != 0:
                        print(f"[Config] 检测到配置文件变更，正在重新加载...")
                        new_config = reload_config()
                        
                        # 通知所有回调
                        for callback in _config_change_callbacks:
                            try:
                                callback(new_config)
                            except Exception as e:
                                print(f"[Config] 回调执行失败: {e}")
                    
                    last_mtime = current_mtime
            except Exception as e:
                print(f"[Config] 监视器错误: {e}")
            
            time.sleep(interval)
    
    _config_watcher_thread = threading.Thread(target=watcher, daemon=True)
    _config_watcher_thread.start()
    print(f"[Config] 配置文件监视器已启动，检查间隔: {interval}秒")


def stop_config_watcher():
    """
    [优化-P4] 停止配置文件监视器
    """
    global _config_watcher_running
    _config_watcher_running = False
    print("[Config] 配置文件监视器已停止")


def normalize_path(path: str) -> str:
    """
    标准化路径格式
    - 统一使用正斜杠
    - 去除末尾斜杠
    - 解析为绝对路径
    """
    try:
        p = Path(path).resolve()
        return str(p).replace('\\', '/')
    except Exception:
        return path.replace('\\', '/').rstrip('/')


def get_app_path(task_type: str) -> Optional[Path]:
    """
    获取任务类型对应的应用路径
    
    支持两种配置方式：
    1. 绝对路径：直接指定完整路径
    2. 相对路径：相对于 app_base_dir
    """
    config = get_config()
    
    # 获取基础目录
    app_base_dir = config.get("app_base_dir", "")
    if app_base_dir:
        app_base_dir = Path(app_base_dir)
    
    # 获取应用路径配置
    app_paths = config.get("app_paths", {})
    
    # 尝试从配置获取
    app_relative = app_paths.get(task_type) or DEFAULT_APP_PATHS.get(task_type)
    
    if not app_relative:
        print(f"[get_app_path] 未找到任务类型 {task_type} 的应用路径配置")
        return None
    
    app_path = Path(app_relative)
    
    # 如果是绝对路径，直接使用
    if app_path.is_absolute():
        if app_path.exists():
            return app_path
        else:
            print(f"[get_app_path] 绝对路径不存在: {app_path}")
            return None
    
    # 相对路径：尝试多种搜索策略
    search_bases = []
    
    # 1. 配置的基础目录
    if app_base_dir and app_base_dir.exists():
        search_bases.append(app_base_dir)
    
    # 2. 配置文件所在目录的父目录
    config_parent = CONFIG_FILE.parent.parent
    if config_parent.exists():
        search_bases.append(config_parent)

    # [P4-6 修复] 2b. 以本文件(config.py)位置推导的项目根作为稳定锚点，
    # 不依赖启动时的工作目录(cwd 可能因从不同目录启动而变化)。
    # config.py 位于 .../dlhub_project/dlhub/backend/config.py
    try:
        backend_dir = Path(__file__).resolve().parent          # backend/
        dlhub_pkg_dir = backend_dir.parent                      # dlhub/
        dlhub_project_dir = dlhub_pkg_dir.parent                # dlhub_project/
        project_root = dlhub_project_dir.parent                 # 项目根
        for anchor in (dlhub_project_dir, project_root):
            if anchor.exists() and anchor not in search_bases:
                search_bases.append(anchor)
    except Exception:
        pass

    # 3. 当前工作目录的父目录（最后兜底）
    cwd_parent = Path.cwd().parent
    if cwd_parent.exists():
        search_bases.append(cwd_parent)
    
    # 搜索应用文件
    for base in search_bases:
        full_path = base / app_relative
        if full_path.exists():
            print(f"[get_app_path] 找到应用: {full_path}")
            return full_path
    
    print(f"[get_app_path] 未找到应用文件 '{app_relative}'")
    print(f"[get_app_path] 已搜索的路径:")
    for base in search_bases:
        print(f"  - {base / app_relative}")
    print(f"[get_app_path] 提示: 请确保 dlhub_project 与 model_image_* 目录在同一父目录下")
    
    return None


def get_config() -> Dict[str, Any]:
    """
    获取全局配置
    
    使用文件缓存，只在文件变更时重新读取
    
    Returns:
        配置字典
    """
    global _config_cache, _config_mtime, _config_size
    
    with _config_lock:
        # 检查文件是否存在
        if not CONFIG_FILE.exists():
            return {
                "version": "2.0",
                "tasks": [],
                "settings": {}
            }
        
        # 检查缓存是否有效
        # [P4-4 修复] 同时比较 mtime 和文件大小：某些文件系统 mtime 精度为 1-2s，
        # 同一秒内的两次外部修改 mtime 可能相同；加上 size 比较可显著降低读到旧缓存的概率。
        try:
            st = CONFIG_FILE.stat()
            current_mtime = st.st_mtime
            current_size = st.st_size
            if (_config_cache is not None
                    and current_mtime == _config_mtime
                    and current_size == _config_size):
                return _config_cache.copy()
        except Exception:
            pass
        
        # 读取配置文件
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                _config_cache = json.load(f)
            
            # 兼容旧版本配置
            if "tasks" not in _config_cache:
                _config_cache["tasks"] = []
            if "version" not in _config_cache:
                _config_cache["version"] = "2.0"
            
            _st = CONFIG_FILE.stat()
            _config_mtime = _st.st_mtime
            _config_size = _st.st_size
            return _config_cache.copy()
        except (json.JSONDecodeError, IOError) as e:
            print(f"读取配置文件失败: {e}")
            return {
                "version": "2.0",
                "tasks": [],
                "settings": {}
            }


def save_config(config: Dict[str, Any]) -> bool:
    """
    保存全局配置

    [P4-1 修复] 原子写入：先写临时文件 → fsync → 替换正式文件，并保留 .bak 备份。
    避免写到一半崩溃/断电导致 dlhub_config.json 损坏、所有任务丢失。

    Args:
        config: 配置字典

    Returns:
        是否保存成功
    """
    global _config_cache, _config_mtime, _config_size

    with _config_lock:
        try:
            # 确保目录存在
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            # [P4-1] 写入前先备份当前有效配置（用于损坏时回滚）
            try:
                if CONFIG_FILE.exists() and CONFIG_FILE.stat().st_size > 0:
                    bak_file = CONFIG_FILE.with_suffix('.json.bak')
                    import shutil as _shutil
                    _shutil.copy2(CONFIG_FILE, bak_file)
            except Exception as _be:
                # 备份失败不阻断主流程，仅告警
                print(f"[save_config] 备份配置失败(非致命): {_be}")

            # [P4-1] 原子写入：临时文件 → flush+fsync → replace
            temp_file = CONFIG_FILE.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass  # 某些文件系统不支持 fsync，忽略
            temp_file.replace(CONFIG_FILE)  # 原子替换

            # 更新缓存
            _config_cache = config.copy()
            _st = CONFIG_FILE.stat()
            _config_mtime = _st.st_mtime
            _config_size = _st.st_size

            print(f"[save_config] 配置已保存，任务数: {len(config.get('tasks', []))}")
            return True
        except (IOError, OSError) as e:
            print(f"[save_config] 保存配置文件失败: {e}")
            return False


def update_config(mutator) -> bool:
    """
    [P4-3 修复] 在单个锁内完成"读-改-写"，避免并发竞态丢任务。

    用法:
        def _add(cfg):
            cfg.setdefault('tasks', []).append(new_task)
            return cfg
        update_config(_add)

    Args:
        mutator: 接收当前 config(dict)、返回修改后 config(dict) 的函数。
                 若返回 None 视为放弃本次修改（不写入）。

    Returns:
        是否保存成功
    """
    global _config_cache, _config_mtime, _config_size

    with _config_lock:
        # 在锁内读取最新配置（不经过可能过期的缓存读取路径）
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    current = json.load(f)
            else:
                current = {"version": "2.0", "tasks": [], "settings": {}}
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"[update_config] 读取配置失败，尝试用缓存: {e}")
            current = (_config_cache.copy() if _config_cache else
                       {"version": "2.0", "tasks": [], "settings": {}})

        if "tasks" not in current:
            current["tasks"] = []
        if "version" not in current:
            current["version"] = "2.0"

        # 应用修改
        try:
            new_config = mutator(current)
        except Exception as e:
            print(f"[update_config] mutator 执行失败: {e}")
            return False

        if new_config is None:
            return True  # 放弃修改，视为成功（无变化）

        # 在锁内直接做原子写（复用 save_config 的写盘逻辑，但锁是可重入的 _config_lock）
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                if CONFIG_FILE.exists() and CONFIG_FILE.stat().st_size > 0:
                    import shutil as _shutil
                    _shutil.copy2(CONFIG_FILE, CONFIG_FILE.with_suffix('.json.bak'))
            except Exception:
                pass
            temp_file = CONFIG_FILE.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            temp_file.replace(CONFIG_FILE)
            _config_cache = new_config.copy()
            _st = CONFIG_FILE.stat()
            _config_mtime = _st.st_mtime
            _config_size = _st.st_size
            return True
        except (IOError, OSError) as e:
            print(f"[update_config] 保存失败: {e}")
            return False


def get_task_paths(auto_cleanup: bool = True) -> List[Dict[str, Any]]:
    """
    获取所有任务路径
    
    Args:
        auto_cleanup: 是否自动清理不存在的任务路径
    
    Returns:
        任务路径列表，每项包含 path, task_type, task_id
    """
    config = get_config()
    tasks = config.get("tasks", [])
    
    if auto_cleanup and tasks:
        # [P3-6 修复] 区分"已删除"与"暂不可访问"(如外接盘未挂载)。
        # 仅当父目录存在、但任务目录确实不存在时，才认为是真删除并清理；
        # 若连父目录/盘符根都不可达(盘未挂载)，保留记录，避免盘恢复后任务丢失。
        valid_tasks = []
        invalid_count = 0

        for task in tasks:
            raw = task.get("path", "")
            task_path = Path(raw)
            keep = True
            try:
                if task_path.exists():
                    keep = True
                else:
                    parent = task_path.parent
                    anchor = Path(task_path.anchor) if task_path.anchor else parent
                    parent_reachable = False
                    try:
                        parent_reachable = parent.exists() or (anchor and anchor.exists())
                    except Exception:
                        parent_reachable = False
                    if parent_reachable:
                        # 父目录可达但任务目录不在 → 确实被删除
                        keep = False
                    else:
                        # 父目录/盘符不可达 → 暂不可访问，保留记录
                        keep = True
                        print(f"[get_task_paths] 任务目录暂不可访问(疑似盘未挂载)，保留记录: {task_path}")
            except Exception:
                keep = True  # 任何异常都保守保留

            if keep:
                valid_tasks.append(task)
            else:
                invalid_count += 1
                print(f"[get_task_paths] 任务目录已删除，清理记录: {task_path}")

        if invalid_count > 0:
            config["tasks"] = valid_tasks
            save_config(config)
            print(f"[get_task_paths] 已自动清理 {invalid_count} 个已删除的任务记录")

        return valid_tasks
    
    return tasks


def add_task_path(task_path: str, task_type: str, task_id: str) -> bool:
    """
    添加任务路径到配置

    [P4-3 修复] 用 update_config 在单锁内完成读-改-写，避免并发建任务互相覆盖。

    Args:
        task_path: 任务目录路径
        task_type: 任务类型
        task_id: 任务ID

    Returns:
        是否添加成功
    """
    normalized_path = normalize_path(task_path)

    def _mutator(config):
        tasks = config.get("tasks", [])
        # 检查是否已存在（标准化路径比对）
        for t in tasks:
            if normalize_path(t.get("path", "")) == normalized_path:
                print(f"[add_task_path] 任务已存在: {normalized_path}")
                return None  # 放弃修改
        tasks.append({
            "path": task_path,
            "task_type": task_type,
            "task_id": task_id,
            "added_at": datetime.now().isoformat()
        })
        config["tasks"] = tasks
        return config

    result = update_config(_mutator)
    if result:
        print(f"[add_task_path] 任务已添加: {task_path}")
    else:
        print(f"[add_task_path] 添加任务失败: {task_path}")
    return result


def remove_task_path(task_path: str) -> bool:
    """
    从配置中移除任务路径

    [P4-3 修复] 用 update_config 在单锁内完成读-改-写。

    Args:
        task_path: 任务目录路径

    Returns:
        是否移除成功
    """
    normalized_path = normalize_path(task_path)

    def _mutator(config):
        tasks = config.get("tasks", [])
        new_tasks = [t for t in tasks if normalize_path(t.get("path", "")) != normalized_path]
        if len(new_tasks) == len(tasks):
            print(f"[remove_task_path] 任务不存在: {task_path}")
            return None  # 无变化
        config["tasks"] = new_tasks
        return config

    result = update_config(_mutator)
    if result:
        print(f"[remove_task_path] 任务已移除: {task_path}")
    return result


def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    """
    通过任务ID获取任务信息
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务信息，不存在返回None
    """
    tasks = get_task_paths()
    for task in tasks:
        if task.get("task_id") == task_id:
            return task
    return None


def update_task_in_config(task_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新配置中的任务信息

    [P4-3 修复] 用 update_config 单锁内读-改-写。

    Args:
        task_id: 任务ID
        updates: 要更新的字段

    Returns:
        是否更新成功
    """
    state = {"updated": False}

    def _mutator(config):
        tasks = config.get("tasks", [])
        for task in tasks:
            if task.get("task_id") == task_id:
                task.update(updates)
                state["updated"] = True
                break
        if not state["updated"]:
            return None
        config["tasks"] = tasks
        return config

    ok = update_config(_mutator)
    return ok and state["updated"]
