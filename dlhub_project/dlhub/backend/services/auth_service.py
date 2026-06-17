# -*- coding: utf-8 -*-
"""
认证服务
=======
管理用户注册、登录、本地JSON文件存储

用户数据保存在 dlhub_config 同级目录下的 users.json
密码使用 hashlib sha256 加盐哈希存储
"""

import json
import hashlib
import secrets
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import threading


# 用户数据文件路径 - 与 dlhub_config.json 同级
def _get_users_file() -> Path:
    """获取用户数据文件路径"""
    base_dir = Path(__file__).parent.parent.parent  # dlhub_project/
    return base_dir / "users.json"


_lock = threading.Lock()


# ==================== [P2-1 修复] 服务端会话存储 ====================
# 之前的实现：token = SERVER_SECRET(公共8字节) + 随机串，但随机串从不存储，
# 中间件只校验 startswith(SERVER_SECRET) → 任何人构造该前缀的 cookie 即可通过，
# 且 /check 会回显前 8 位（即 SERVER_SECRET）。属于可绕过的伪认证。
#
# 现改为：登录成功生成随机 token，服务端存 token -> {username, nickname, created, expires}。
# 中间件校验 token 是否在表中且未过期。退出/过期即失效。
import time as _time

_sessions: Dict[str, Dict[str, Any]] = {}
_session_lock = threading.Lock()

# 会话有效期（秒）。默认 7 天；可按需调整。
SESSION_TTL_SECONDS = 7 * 24 * 3600


def create_session(username: str, nickname: str = "") -> str:
    """创建一个服务端会话，返回随机 token。"""
    token = secrets.token_urlsafe(32)
    now = _time.time()
    with _session_lock:
        # 顺手清理过期会话，避免无限增长
        _cleanup_expired_sessions_locked(now)
        _sessions[token] = {
            "username": username,
            "nickname": nickname or username,
            "created_at": now,
            "expires_at": now + SESSION_TTL_SECONDS,
        }
    return token


def validate_session(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """校验 token：有效则返回会话信息(并滑动续期)，否则返回 None。"""
    if not token:
        return None
    now = _time.time()
    with _session_lock:
        sess = _sessions.get(token)
        if not sess:
            return None
        if sess["expires_at"] < now:
            # 已过期，删除
            _sessions.pop(token, None)
            return None
        # 滑动续期：每次有效访问刷新有效期
        sess["expires_at"] = now + SESSION_TTL_SECONDS
        return dict(sess)


def destroy_session(token: Optional[str]) -> None:
    """注销：删除会话。"""
    if not token:
        return
    with _session_lock:
        _sessions.pop(token, None)


def _cleanup_expired_sessions_locked(now: float) -> None:
    """清理过期会话（调用方需持有 _session_lock）。"""
    expired = [t for t, s in _sessions.items() if s["expires_at"] < now]
    for t in expired:
        _sessions.pop(t, None)


def _load_users() -> Dict[str, Any]:
    """加载用户数据"""
    users_file = _get_users_file()
    if not users_file.exists():
        return {"users": []}
    try:
        with open(users_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return {"users": []}


def _save_users(data: Dict[str, Any]) -> bool:
    """保存用户数据"""
    users_file = _get_users_file()
    try:
        users_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = users_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_file.replace(users_file)
        return True
    except Exception as e:
        print(f"[Auth] 保存用户数据失败: {e}")
        return False


def _hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """
    对密码进行加盐哈希
    
    Returns:
        (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return hashed, salt


def _verify_password(password: str, hashed: str, salt: str) -> bool:
    """验证密码"""
    check_hash, _ = _hash_password(password, salt)
    return check_hash == hashed


def register_user(username: str, password: str, nickname: str = "") -> Tuple[bool, str, Optional[Dict]]:
    """
    注册新用户
    
    Args:
        username: 用户名（唯一标识，不强制邮箱格式）
        password: 密码（至少4个字符）
        nickname: 昵称（可选）
    
    Returns:
        (success, message, user_info)
    """
    # 基本校验
    if not username or not username.strip():
        return False, "用户名不能为空", None
    
    username = username.strip()
    
    if len(username) < 2:
        return False, "用户名至少需要2个字符", None
    
    if not password or len(password) < 4:
        return False, "密码至少需要4个字符", None
    
    with _lock:
        data = _load_users()
        
        # 检查用户名是否已存在
        for user in data.get("users", []):
            if user.get("username", "").lower() == username.lower():
                return False, "用户名已存在", None
        
        # 创建用户
        hashed_pw, salt = _hash_password(password)
        
        user_info = {
            "id": secrets.token_hex(8),
            "username": username,
            "nickname": nickname.strip() or username,
            "password_hash": hashed_pw,
            "salt": salt,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
        }
        
        data.setdefault("users", []).append(user_info)
        
        if not _save_users(data):
            return False, "保存失败，请重试", None
        
        # 返回安全的用户信息（不含密码）
        safe_info = {
            "id": user_info["id"],
            "username": user_info["username"],
            "nickname": user_info["nickname"],
            "created_at": user_info["created_at"],
        }
        
        return True, "注册成功", safe_info


def login_user(username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    用户登录
    
    Args:
        username: 用户名
        password: 密码
    
    Returns:
        (success, message, user_info)
    """
    if not username or not password:
        return False, "请输入用户名和密码", None
    
    username = username.strip()
    
    with _lock:
        data = _load_users()
        
        for user in data.get("users", []):
            if user.get("username", "").lower() == username.lower():
                # 找到用户，验证密码
                if _verify_password(password, user["password_hash"], user["salt"]):
                    # 更新最后登录时间
                    user["last_login"] = datetime.now().isoformat()
                    _save_users(data)

                    # [P2-1 修复] 生成服务端会话 token（存表 + 过期），不再用一次性丢弃的随机串
                    token = create_session(user["username"], user.get("nickname", user["username"]))

                    safe_info = {
                        "id": user["id"],
                        "username": user["username"],
                        "nickname": user["nickname"],
                        "created_at": user["created_at"],
                        "last_login": user["last_login"],
                        "token": token,
                    }

                    return True, "登录成功", safe_info
                else:
                    return False, "密码错误", None
        
        return False, "用户不存在", None


def get_user_count() -> int:
    """获取用户数量"""
    data = _load_users()
    return len(data.get("users", []))
