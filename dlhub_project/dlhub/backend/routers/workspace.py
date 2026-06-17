# -*- coding: utf-8 -*-
"""
工作空间路由
===========
提供目录浏览功能（用于选择工作目录和导入任务）
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import os
import platform

router = APIRouter()


# ==================== 数据模型 ====================

class DirectoryItem(BaseModel):
    """目录项"""
    name: str
    path: str
    type: str  # 'directory', 'file', 'drive'


class BrowseResponse(BaseModel):
    """浏览响应"""
    current: str
    parent: Optional[str]
    items: List[DirectoryItem]


class ValidateResponse(BaseModel):
    """验证响应"""
    valid: bool
    message: str
    is_dlhub_task: bool = False
    task_type: Optional[str] = None


# ==================== API路由 ====================

@router.get("/browse", response_model=BrowseResponse)
async def browse_directory(path: str = Query("", description="要浏览的路径")):
    """
    浏览目录
    
    - path: 要浏览的路径，空字符串返回根目录/驱动器列表
    """
    # 如果路径为空，返回根目录
    if not path:
        if platform.system() == 'Windows':
            # Windows: 返回驱动器列表
            drives = []
            for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
                drive_path = f"{letter}:/"
                if os.path.exists(drive_path):
                    drives.append(DirectoryItem(
                        name=f"{letter}:",
                        path=drive_path,
                        type='drive'
                    ))
            return BrowseResponse(
                current='',
                parent=None,
                items=drives
            )
        else:
            # Linux/Mac: 从根目录开始
            path = '/'
    
    # 规范化路径
    try:
        target_path = Path(path).resolve()
    except Exception as e:
        raise HTTPException(400, f"无效的路径: {e}")
    
    if not target_path.exists():
        raise HTTPException(404, f"路径不存在: {path}")
    
    if not target_path.is_dir():
        raise HTTPException(400, f"路径不是目录: {path}")
    
    # 获取父目录
    parent = None
    if target_path.parent != target_path:
        parent = str(target_path.parent)
    elif platform.system() == 'Windows':
        parent = ''  # 返回驱动器列表
    
    # 列出目录内容
    items = []
    try:
        for item in sorted(target_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # 跳过隐藏文件（以.开头）
            if item.name.startswith('.'):
                continue
            
            try:
                item_type = 'directory' if item.is_dir() else 'file'
                items.append(DirectoryItem(
                    name=item.name,
                    path=str(item),
                    type=item_type
                ))
            except PermissionError:
                continue
    except PermissionError:
        raise HTTPException(403, f"没有权限访问目录: {path}")
    
    return BrowseResponse(
        current=str(target_path),
        parent=parent,
        items=items
    )


@router.post("/validate", response_model=ValidateResponse)
async def validate_directory(path: str = Query(..., description="要验证的路径")):
    """
    验证目录是否可以作为工作目录或任务目录
    
    - 检查目录是否存在
    - 检查是否有写入权限
    - 检查是否是DL-Hub任务目录
    """
    try:
        target_path = Path(path).resolve()
    except Exception as e:
        return ValidateResponse(
            valid=False,
            message=f"无效的路径: {e}"
        )
    
    if not target_path.exists():
        return ValidateResponse(
            valid=False,
            message="目录不存在"
        )
    
    if not target_path.is_dir():
        return ValidateResponse(
            valid=False,
            message="路径不是目录"
        )
    
    # 检查写入权限
    try:
        test_file = target_path / ".dlhub_write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        return ValidateResponse(
            valid=False,
            message="没有写入权限"
        )
    except Exception as e:
        return ValidateResponse(
            valid=False,
            message=f"无法写入目录: {e}"
        )
    
    # 检查是否是DL-Hub任务目录
    task_json = target_path / ".dlhub" / "task.json"
    is_dlhub_task = task_json.exists()
    task_type = None
    
    if is_dlhub_task:
        try:
            import json
            with open(task_json, 'r', encoding='utf-8') as f:
                task_meta = json.load(f)
            task_type = task_meta.get('task_type')
        except Exception:
            pass
    
    return ValidateResponse(
        valid=True,
        message="目录可用" if not is_dlhub_task else f"这是一个{task_type or '未知类型'}任务目录",
        is_dlhub_task=is_dlhub_task,
        task_type=task_type
    )


@router.get("/status")
async def get_workspace_status():
    """
    获取工作空间状态（兼容旧API）
    
    新版本不再需要全局工作空间，总是返回已配置
    """
    return {
        "configured": True,
        "workspace": None,
        "message": "新版本不再需要全局工作空间"
    }


@router.post("/mkdir")
async def create_directory(parent_path: str = Query(..., description="父目录路径"), name: str = Query(..., description="新文件夹名称")):
    """
    新建文件夹（问题-4）
    """
    if not name or not name.strip():
        raise HTTPException(400, "文件夹名称不能为空")
    
    # 过滤非法字符
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '', name.strip())
    if not safe_name:
        raise HTTPException(400, "文件夹名称包含非法字符")
    
    try:
        parent = Path(parent_path).resolve()
    except Exception as e:
        raise HTTPException(400, f"无效的路径: {e}")
    
    if not parent.exists():
        raise HTTPException(404, f"父目录不存在: {parent_path}")
    
    if not parent.is_dir():
        raise HTTPException(400, f"路径不是目录: {parent_path}")
    
    new_dir = parent / safe_name
    
    if new_dir.exists():
        raise HTTPException(400, f"文件夹已存在: {safe_name}")
    
    try:
        new_dir.mkdir(parents=False, exist_ok=False)
        return {
            "success": True,
            "path": str(new_dir),
            "name": safe_name
        }
    except PermissionError:
        raise HTTPException(403, "没有权限创建文件夹")
    except Exception as e:
        raise HTTPException(500, f"创建文件夹失败: {e}")


@router.get("/disk-space")
async def get_disk_space(path: str = Query(..., description="要检查的路径")):
    """
    获取指定路径的磁盘空间信息（问题-8）
    """
    try:
        import shutil
        target_path = Path(path).resolve()
        
        # 如果路径不存在，使用父目录
        while not target_path.exists() and target_path.parent != target_path:
            target_path = target_path.parent
        
        if not target_path.exists():
            raise HTTPException(404, f"路径不存在: {path}")
        
        total, used, free = shutil.disk_usage(target_path)
        
        def format_size(size):
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            elif size < 1024 * 1024 * 1024:
                return f"{size / (1024 * 1024):.1f} MB"
            else:
                return f"{size / (1024 * 1024 * 1024):.2f} GB"
        
        return {
            "path": str(target_path),
            "total": total,
            "used": used,
            "free": free,
            "total_formatted": format_size(total),
            "used_formatted": format_size(used),
            "free_formatted": format_size(free),
            "percent_used": round(used / total * 100, 1) if total > 0 else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"获取磁盘空间失败: {e}")
