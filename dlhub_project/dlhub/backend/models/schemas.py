# -*- coding: utf-8 -*-
"""
Pydantic 数据模型
=================
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==================== 通用模型 ====================

class SuccessResponse(BaseModel):
    """成功响应"""
    success: bool = True
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error: str
    detail: Optional[str] = None

# ==================== 工作空间模型 ====================

class WorkspaceConfig(BaseModel):
    """工作空间配置"""
    path: str
    configured: bool = False

# ==================== 任务模型 ====================

class TaskCreate(BaseModel):
    """创建任务请求"""
    task_type: str = Field(..., description="任务类型")
    task_name: str = Field(..., min_length=1, max_length=100, description="任务名称")
    description: Optional[str] = Field("", max_length=500, description="任务描述")

class TaskUpdate(BaseModel):
    """更新任务请求"""
    task_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class TaskMetric(BaseModel):
    """任务指标"""
    name: str
    value: float

class TaskTrainingInfo(BaseModel):
    """训练信息"""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    epochs_completed: Optional[int] = None
    best_metric: Optional[TaskMetric] = None

class Task(BaseModel):
    """任务完整信息"""
    task_id: str
    task_type: str
    task_name: str
    description: str = ""
    path: str
    status: str = "idle"
    created_at: str
    updated_at: str
    best_metric: Optional[Dict[str, Any]] = None
    training_info: Optional[TaskTrainingInfo] = None

class TaskList(BaseModel):
    """任务列表"""
    tasks: List[Task]
    total: int

# ==================== 应用启动模型 ====================

class AppLaunchRequest(BaseModel):
    """启动应用请求"""
    task_id: str

class AppLaunchResponse(BaseModel):
    """启动应用响应"""
    success: bool
    url: Optional[str] = None
    task_id: Optional[str] = None
    message: Optional[str] = None

class AppStatus(BaseModel):
    """应用状态"""
    running: bool
    current_task_id: Optional[str] = None
    current_task_type: Optional[str] = None
    url: Optional[str] = None

# ==================== 系统信息模型 ====================

class GpuInfo(BaseModel):
    """GPU信息"""
    index: int
    name: str
    memory_total: str
    memory_used: str
    utilization: str

class SystemGpu(BaseModel):
    """系统GPU信息"""
    available: bool
    driver_version: Optional[str] = None
    gpus: List[GpuInfo] = []

class CondaEnv(BaseModel):
    """Conda环境"""
    name: str
    path: str

class CondaEnvList(BaseModel):
    """Conda环境列表"""
    envs: List[CondaEnv]
    error: Optional[str] = None
