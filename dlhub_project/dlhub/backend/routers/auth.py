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

    # 设置认证 cookie
    # token 前缀加上服务器密钥，重启后旧cookie自动失效
    from ..main import SERVER_SECRET
    raw_token = user_info.get("token", "")
    cookie_token = SERVER_SECRET + raw_token
    response.set_cookie(
        key=COOKIE_NAME,
        value=cookie_token,
        httponly=True,
        samesite="lax",
        path="/",
    )

    return {"success": True, "message": message, "user": user_info}


@router.post("/logout")
async def api_logout(response: Response):
    """注销 - 清除 cookie（参数必须与set_cookie一致才能正确删除）"""
    response.delete_cookie(key=COOKIE_NAME, path="/", httponly=True, samesite="lax")
    return {"success": True, "message": "已退出登录"}


@router.get("/check")
async def api_check(dlhub_token: Optional[str] = Cookie(None)):
    """
    检查当前认证状态（供前端调用）
    """
    if not dlhub_token:
        return JSONResponse(status_code=401, content={"authenticated": False})
    # token 存在即视为已认证（简化版，生产环境应验证 token 有效性）
    return {"authenticated": True, "token": dlhub_token[:8] + "..."}


@router.get("/status")
async def api_auth_status():
    """获取认证系统状态"""
    count = get_user_count()
    return {"active": True, "user_count": count, "has_users": count > 0}
