# -*- coding: utf-8 -*-
"""
认证路由（Cookie 版）
==================
登录成功 → 设置 HttpOnly cookie (dlhub_token)
/app 路由检查此 cookie，无则 302 到登录页
这样无论浏览器怎么缓存 JS，都无法绕过服务端认证
"""

from fastapi import APIRouter, HTTPException, Response, Cookie
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from ..services.auth_service import register_user, login_user, get_user_count

router = APIRouter()

# Cookie 名称（session cookie - 关闭浏览器即过期，重新打开需要登录）
COOKIE_NAME = "dlhub_token"


class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: Optional[str] = ""


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def api_register(req: RegisterRequest):
    """用户注册"""
    success, message, user_info = register_user(
        username=req.username,
        password=req.password,
        nickname=req.nickname or "",
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message, "user": user_info}


@router.post("/login")
async def api_login(req: LoginRequest, response: Response):
    """
    用户登录
    成功后设置 HttpOnly cookie，浏览器自动携带
    """
    success, message, user_info = login_user(
        username=req.username,
        password=req.password,
    )
    if not success:
        raise HTTPException(status_code=401, detail=message)

    # [P2-1 修复] cookie 直接存服务端会话 token（不再拼接公共 SERVER_SECRET 前缀）
    raw_token = user_info.get("token", "")
    response.set_cookie(
        key=COOKIE_NAME,
        value=raw_token,
        httponly=True,
        samesite="lax",
        path="/",
    )

    return {"success": True, "message": message, "user": user_info}


@router.post("/logout")
async def api_logout(response: Response, dlhub_token: Optional[str] = Cookie(None)):
    """注销 - 销毁服务端会话 + 清除 cookie（参数必须与set_cookie一致才能正确删除）"""
    from ..services.auth_service import destroy_session
    destroy_session(dlhub_token)  # [P2-1] 服务端失效
    response.delete_cookie(key=COOKIE_NAME, path="/", httponly=True, samesite="lax")
    return {"success": True, "message": "已退出登录"}


@router.get("/check")
async def api_check(dlhub_token: Optional[str] = Cookie(None)):
    """
    检查当前认证状态（供前端调用）

    [P2-1 修复] 校验服务端会话是否有效，不再回显 token 片段（避免泄露）。
    """
    from ..services.auth_service import validate_session
    sess = validate_session(dlhub_token)
    if not sess:
        return JSONResponse(status_code=401, content={"authenticated": False})
    return {
        "authenticated": True,
        "username": sess.get("username"),
        "nickname": sess.get("nickname"),
    }


@router.get("/status")
async def api_auth_status():
    """获取认证系统状态"""
    count = get_user_count()
    return {"active": True, "user_count": count, "has_users": count > 0}
