# -*- coding: utf-8 -*-
"""
任务管理路由
===========
支持自定义工作目录和Conda环境
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Dict, Any
from pathlib import Path
import json

from ..services.task_service import TaskService
from ..services.status_monitor import get_task_status, get_task_metrics

router = APIRouter()

# 任务服务实例
task_service = TaskService()


# ==================== 数据模型 ====================

class TaskCreateRequest(BaseModel):
    """创建任务请求"""
    task_type: str
    task_name: str
    work_dir: str  # 工作目录
    conda_env: str  # Conda环境名称
    description: Optional[str] = ""


class TaskImportRequest(BaseModel):
    """导入任务请求"""
    task_dir: str  # 任务目录路径
    task_type: str  # 期望的任务类型
    conda_env: Optional[str] = None  # 可选的Conda环境（用于旧任务）
    force: bool = False  # 是否强制导入（替换已存在的任务记录）


class TaskUpdateRequest(BaseModel):
    """更新任务请求"""
    task_name: Optional[str] = None
    description: Optional[str] = None
    conda_env: Optional[str] = None


class TaskMetric(BaseModel):
    """任务指标"""
    name: str
    value: float


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    task_type: str
    task_name: str
    description: str
    path: str
    status: str
    conda_env: Optional[str] = None
    created_at: str
    updated_at: str
    best_metric: Optional[TaskMetric] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""
    tasks: List[TaskResponse]
    total: int
    running_task_id: Optional[str] = None  # 当前正在运行的任务ID


# ==================== API路由 ====================

@router.get("", response_model=TaskListResponse)
async def list_tasks(task_type: Optional[str] = Query(None, description="任务类型过滤")):
    """
    获取任务列表
    
    - task_type: 可选，按任务类型过滤 (classification/detection/segmentation/anomaly/ocr)
    - 返回中包含 running_task_id 字段，标识当前正在运行的任务
    """
    # 获取当前运行的任务ID
    from ..services.process_service import ProcessService
    process_service = ProcessService()
    running_task_id = process_service.current_task_id if process_service.is_app_running() else None
    
    tasks = task_service.list_tasks(task_type)
    
    # 转换为响应格式
    task_responses = []
    for task in tasks:
        task_path = Path(task['path'])
        # 改进：传递running_task_id以更准确地判断状态
        status = get_task_status(task_path, running_task_id, task['task_id'])
        metrics = get_task_metrics(task_path, task['task_type'])
        
        task_responses.append(TaskResponse(
            task_id=task['task_id'],
            task_type=task['task_type'],
            task_name=task['task_name'],
            description=task.get('description', ''),
            path=task['path'],
            status=status,
            conda_env=task.get('conda_env'),
            created_at=task['created_at'],
            updated_at=task.get('updated_at', task['created_at']),
            best_metric=TaskMetric(**metrics) if metrics else None
        ))
    
    return TaskListResponse(tasks=task_responses, total=len(task_responses), running_task_id=running_task_id)


@router.post("")
async def create_task(req: TaskCreateRequest):
    """
    创建新任务
    
    需要指定：
    - task_type: 任务类型
    - task_name: 任务名称
    - work_dir: 工作目录
    - conda_env: Conda环境名称
    """
    # 验证任务类型
    valid_types = ['classification', 'detection', 'segmentation', 'anomaly', 'ocr', 'sevseg']
    if req.task_type not in valid_types:
        raise HTTPException(400, f"无效的任务类型: {req.task_type}，支持: {', '.join(valid_types)}")
    
    # 验证任务名称
    if not req.task_name or not req.task_name.strip():
        raise HTTPException(400, "任务名称不能为空")
    
    # 验证工作目录
    if not req.work_dir or not req.work_dir.strip():
        raise HTTPException(400, "请选择工作目录")
    
    # 验证Conda环境
    if not req.conda_env or not req.conda_env.strip():
        raise HTTPException(400, "请选择Conda环境")
    
    try:
        task = task_service.create_task(
            task_type=req.task_type,
            task_name=req.task_name.strip(),
            work_dir=req.work_dir.strip(),
            conda_env=req.conda_env.strip(),
            description=req.description or ""
        )
        
        return {
            "success": True,
            "task": TaskResponse(
                task_id=task['task_id'],
                task_type=task['task_type'],
                task_name=task['task_name'],
                description=task.get('description', ''),
                path=task['path'],
                status='idle',
                conda_env=task.get('conda_env'),
                created_at=task['created_at'],
                updated_at=task['updated_at'],
                best_metric=None
            )
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"创建任务失败: {e}")


@router.post("/import")
async def import_task(req: TaskImportRequest):
    """
    导入已有任务目录
    
    - task_dir: 任务目录路径
    - task_type: 期望的任务类型（用于验证）
    - conda_env: 可选，为缺少环境配置的旧任务指定Conda环境
    """
    if not req.task_dir or not req.task_dir.strip():
        raise HTTPException(400, "请选择要导入的任务目录")
    
    try:
        task = task_service.import_task(
            task_dir=req.task_dir.strip(),
            expected_type=req.task_type,
            conda_env=req.conda_env,
            force=req.force  # 支持强制导入（替换已存在的任务）
        )
        
        # BUG修复：获取running_task_id以准确判断状态
        from ..services.process_service import ProcessService
        process_service = ProcessService()
        running_task_id = process_service.current_task_id if process_service.is_app_running() else None
        
        task_path = Path(task['path'])
        status = get_task_status(task_path, running_task_id, task['task_id'])
        metrics = get_task_metrics(task_path, task['task_type'])
        
        return {
            "success": True,
            "task": TaskResponse(
                task_id=task['task_id'],
                task_type=task['task_type'],
                task_name=task['task_name'],
                description=task.get('description', ''),
                path=task['path'],
                status=status,
                conda_env=task.get('conda_env'),
                created_at=task['created_at'],
                updated_at=task.get('updated_at', task['created_at']),
                best_metric=TaskMetric(**metrics) if metrics else None
            )
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"导入任务失败: {e}")


@router.get("/{task_id}")
async def get_task(task_id: str):
    """
    获取任务详情
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    # BUG修复：获取running_task_id以准确判断状态
    from ..services.process_service import ProcessService
    process_service = ProcessService()
    running_task_id = process_service.current_task_id if process_service.is_app_running() else None
    
    task_path = Path(task['path'])
    status = get_task_status(task_path, running_task_id, task_id)
    metrics = get_task_metrics(task_path, task['task_type'])
    
    return TaskResponse(
        task_id=task['task_id'],
        task_type=task['task_type'],
        task_name=task['task_name'],
        description=task.get('description', ''),
        path=task['path'],
        status=status,
        conda_env=task.get('conda_env'),
        created_at=task['created_at'],
        updated_at=task.get('updated_at', task['created_at']),
        best_metric=TaskMetric(**metrics) if metrics else None
    )


@router.put("/{task_id}")
async def update_task(task_id: str, req: TaskUpdateRequest):
    """
    更新任务信息
    """
    updates = {}
    if req.task_name is not None:
        updates['task_name'] = req.task_name
    if req.description is not None:
        updates['description'] = req.description
    if req.conda_env is not None:
        updates['conda_env'] = req.conda_env
    
    if not updates:
        raise HTTPException(400, "没有要更新的字段")
    
    success = task_service.update_task(task_id, updates)
    if not success:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    return {"success": True, "message": "任务更新成功"}


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务
    
    注意：这会删除任务目录下的所有文件，包括训练产生的模型
    
    BUG修复：增加对正在使用任务的检查，防止删除正在运行的任务
    """
    # 检查任务是否存在
    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    # BUG修复：检查任务是否正在被使用（通过ProcessService）
    from ..services.process_service import ProcessService
    process_service = ProcessService()
    
    if process_service.is_app_running() and process_service.current_task_id == task_id:
        raise HTTPException(400, "任务正在使用中，请先结束任务")
    
    # 检查任务是否正在训练（通过状态检测）
    task_path = Path(task['path'])
    running_task_id = process_service.current_task_id if process_service.is_app_running() else None
    status = get_task_status(task_path, running_task_id, task_id)
    
    if status == 'training':
        raise HTTPException(400, "任务正在训练中，请先停止训练")
    
    # 删除任务
    success = task_service.delete_task(task_id)
    if not success:
        raise HTTPException(500, "删除任务失败")
    
    return {"success": True, "message": "任务已删除"}


@router.get("/{task_id}/status")
async def get_task_status_api(task_id: str):
    """
    获取任务状态
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    # BUG修复：获取running_task_id以准确判断状态
    from ..services.process_service import ProcessService
    process_service = ProcessService()
    running_task_id = process_service.current_task_id if process_service.is_app_running() else None
    
    task_path = Path(task['path'])
    status = get_task_status(task_path, running_task_id, task_id)
    metrics = get_task_metrics(task_path, task['task_type'])
    
    return {
        "task_id": task_id,
        "status": status,
        "metrics": metrics
    }


@router.get("/{task_id}/size")
async def get_task_size(task_id: str):
    """
    获取任务的磁盘占用（问题-6）
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    task_path = Path(task['path'])
    
    if not task_path.exists():
        return {"size_bytes": 0, "size_formatted": "0 B", "file_count": 0}
    
    try:
        total_size = 0
        file_count = 0
        for item in task_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
        
        # 格式化大小
        if total_size < 1024:
            size_formatted = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_formatted = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_formatted = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_formatted = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        
        return {
            "size_bytes": total_size,
            "size_formatted": size_formatted,
            "file_count": file_count
        }
    except Exception as e:
        raise HTTPException(500, f"获取任务大小失败: {e}")


@router.post("/{task_id}/copy")
async def copy_task(task_id: str, new_name: str = None):
    """
    复制任务（优化-7）
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    try:
        new_task = task_service.copy_task(task_id, new_name)
        
        return {
            "success": True,
            "task": TaskResponse(
                task_id=new_task['task_id'],
                task_type=new_task['task_type'],
                task_name=new_task['task_name'],
                description=new_task.get('description', ''),
                path=new_task['path'],
                status='idle',
                conda_env=new_task.get('conda_env'),
                created_at=new_task['created_at'],
                updated_at=new_task['updated_at'],
                best_metric=None
            )
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"复制任务失败: {e}")


@router.get("/{task_id}/params")
async def get_task_params(task_id: str):
    """
    获取任务的UI参数
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    task_path = Path(task['path'])
    params_file = task_path / ".dlhub" / "ui_params.json"
    
    if not params_file.exists():
        return {"params": None, "message": "任务尚未配置参数"}
    
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            "params": data.get("params", {}),
            "saved_at": data.get("saved_at"),
            "version": data.get("version", "1.0")
        }
    except Exception as e:
        raise HTTPException(500, f"读取参数失败: {e}")


@router.post("/{task_id}/params")
async def save_task_params(task_id: str, params: Dict[str, Any]):
    """
    保存任务的UI参数
    
    这是核心功能：将App界面的参数保存到任务目录，
    下次打开任务时可以加载这些参数
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    
    task_path = Path(task['path'])
    dlhub_dir = task_path / ".dlhub"
    dlhub_dir.mkdir(exist_ok=True)
    params_file = dlhub_dir / "ui_params.json"
    
    try:
        from datetime import datetime
        data = {
            "version": "1.0",
            "task_type": task.get('task_type'),
            "task_id": task_id,
            "saved_at": datetime.now().isoformat(),
            "params": params
        }
        
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 更新任务的updated_at时间
        task_service.update_task(task_id, {})
        
        return {
            "success": True,
            "message": "参数保存成功",
            "saved_at": data["saved_at"]
        }
    except Exception as e:
        raise HTTPException(500, f"保存参数失败: {e}")


# ==================== 参数模板 ====================

# 五大任务的默认参数模板
PARAM_TEMPLATES = {
    "classification": {
        "data": {
            "train_dir": "",
            "val_dir": "",
            "test_dir": "",
            "image_size": 224,
            "num_classes": 0,
            "class_names": []
        },
        "model": {
            "backbone": "resnet50",
            "pretrained": True,
            "freeze_layers": 0,
            "dropout": 0.0
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "weight_decay": 0.0001,
            "early_stopping": True,
            "patience": 10,
            "workers": 4
        },
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation": 15,
            "color_jitter": True,
            "random_crop": True,
            "normalize": True,
            "mixup": False,
            "cutout": False
        },
        "export": {
            "format": "onnx",
            "quantize": False,
            "input_shape": [1, 3, 224, 224]
        }
    },
    "detection": {
        "data": {
            "data_yaml": "",
            "train_images": "",
            "val_images": "",
            "image_size": 640,
            "num_classes": 0,
            "class_names": []
        },
        "model": {
            "architecture": "yolov8",
            "variant": "n",
            "pretrained": "yolov8n.pt"
        },
        "training": {
            "batch_size": 16,
            "epochs": 300,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "patience": 50,
            "workers": 8,
            "device": "0"
        },
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0
        },
        "export": {
            "format": "onnx",
            "half": False,
            "dynamic": False,
            "simplify": True,
            "opset": 12
        }
    },
    "segmentation": {
        "data": {
            "train_images": "",
            "train_masks": "",
            "val_images": "",
            "val_masks": "",
            "image_size": [512, 512],
            "num_classes": 0,
            "class_names": [],
            "ignore_index": 255
        },
        "model": {
            "architecture": "segformer",
            "variant": "b2",
            "pretrained": True,
            "encoder_weights": "imagenet"
        },
        "training": {
            "batch_size": 8,
            "epochs": 200,
            "learning_rate": 0.0001,
            "optimizer": "adamw",
            "scheduler": "poly",
            "weight_decay": 0.01,
            "loss": "cross_entropy",
            "workers": 4
        },
        "augmentation": {
            "random_crop": True,
            "random_flip": True,
            "color_jitter": True,
            "random_scale": [0.5, 2.0],
            "normalize": True
        },
        "export": {
            "format": "onnx",
            "input_shape": [1, 3, 512, 512]
        }
    },
    "anomaly": {
        "data": {
            "good_dir": "",
            "test_dir": "",
            "mask_dir": "",
            "image_size": 224,
            "center_crop": 224
        },
        "model": {
            "algorithm": "patchcore",
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.01,
            "num_neighbors": 9
        },
        "training": {
            "batch_size": 32,
            "seed": 42
        },
        "threshold": {
            "method": "auto",
            "percentile": 99.5,
            "manual_value": None
        },
        "visualization": {
            "show_heatmap": True,
            "show_segmentation": True,
            "colormap": "jet"
        },
        "export": {
            "format": "pkg",
            "include_visualizer": True
        }
    },
    "ocr": {
        "data": {
            "train_images": "",
            "train_labels": "",
            "val_images": "",
            "val_labels": "",
            "max_text_length": 25,
            "character_set": "chinese"
        },
        "model": {
            "det_model": "ch_PP-OCRv4_det",
            "rec_model": "ch_PP-OCRv4_rec",
            "cls_model": "ch_ppocr_mobile_v2.0_cls",
            "use_angle_cls": True,
            "lang": "ch"
        },
        "detection": {
            "det_algorithm": "DB",
            "det_limit_side_len": 960,
            "det_db_thresh": 0.3,
            "det_db_box_thresh": 0.6,
            "det_db_unclip_ratio": 1.5,
            "use_dilation": False
        },
        "recognition": {
            "rec_algorithm": "SVTR_LCNet",
            "rec_image_shape": "3,48,320",
            "rec_batch_num": 6
        },
        "training": {
            "batch_size": 64,
            "epochs": 500,
            "learning_rate": 0.001,
            "optimizer": "adam"
        },
        "export": {
            "format": "onnx",
            "include_det": True,
            "include_rec": True,
            "include_cls": True
        }
    },
    "sevseg": {
        "data": {
            "data_yaml": "",
            "images_dir": "",
            "jsons_dir": "",
            "image_size": 640,
            "num_classes": 0,
            "class_names": []
        },
        "model": {
            "scale": "中",
            "architecture": "yolo26-score",
            "pretrained": "yolo26m.pt"
        },
        "training": {
            "batch_size": 32,
            "epochs": 105,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "patience": 50,
            "workers": 4,
            "device": "0",
            "mixup": 0.0,
            "mosaic": 1.0,
            "fliplr": 0.5,
            "cos_lr": True
        },
        "score": {
            "lambda_score": 0.05,
            "gaussian_sigma": 0.10,
            "score_loss": "gaussian_nll"
        },
        "export": {
            "format": "onnx",
            "half": True,
            "imgsz": 640
        }
    }
}


@router.get("/params/template/{task_type}")
async def get_params_template(task_type: str):
    """
    获取指定任务类型的默认参数模板
    
    用于创建新任务时初始化参数，或重置参数为默认值
    """
    if task_type not in PARAM_TEMPLATES:
        raise HTTPException(400, f"未知的任务类型: {task_type}")
    
    return {
        "task_type": task_type,
        "params": PARAM_TEMPLATES[task_type],
        "description": f"{task_type}任务的默认参数模板"
    }
