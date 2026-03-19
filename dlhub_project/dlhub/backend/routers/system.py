# -*- coding: utf-8 -*-
"""
系统信息路由
===========
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import platform
import subprocess
import json
import os

router = APIRouter()


# ==================== 数据模型 ====================

class GpuInfo(BaseModel):
    """GPU信息"""
    index: int
    name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: int   # %


class GpuResponse(BaseModel):
    """GPU响应"""
    available: bool
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpus: List[GpuInfo] = []


class SystemInfo(BaseModel):
    """系统信息"""
    os: str
    os_version: str
    python_version: str
    cpu_count: int
    memory_total: int  # GB


class CondaEnv(BaseModel):
    """Conda环境"""
    name: str
    path: str
    python_version: Optional[str] = None
    is_active: bool = False


# ==================== API路由 ====================

@router.get("/info", response_model=SystemInfo)
async def get_system_info():
    """
    获取系统信息
    """
    import psutil
    
    return SystemInfo(
        os=platform.system(),
        os_version=platform.version(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        memory_total=round(psutil.virtual_memory().total / (1024**3), 1)
    )


@router.get("/gpu", response_model=GpuResponse)
async def get_gpu_info():
    """
    获取GPU信息
    """
    try:
        # 使用nvidia-smi获取GPU信息
        result = subprocess.run(
            [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return GpuResponse(available=False)
        
        # 解析GPU信息
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append(GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total=int(parts[2]),
                    memory_used=int(parts[3]),
                    memory_free=int(parts[4]),
                    utilization=int(parts[5]) if parts[5] != '[N/A]' else 0
                ))
        
        # 获取驱动版本
        driver_version = None
        cuda_version = None
        
        result2 = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result2.returncode == 0 and result2.stdout.strip():
            driver_version = result2.stdout.strip().split('\n')[0]
        
        # 获取CUDA版本
        result3 = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result3.returncode == 0:
            for line in result3.stdout.split('\n'):
                if 'CUDA Version' in line:
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        cuda_version = match.group(1)
                    break
        
        return GpuResponse(
            available=True,
            driver_version=driver_version,
            cuda_version=cuda_version,
            gpus=gpus
        )
        
    except FileNotFoundError:
        return GpuResponse(available=False)
    except subprocess.TimeoutExpired:
        return GpuResponse(available=False)
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return GpuResponse(available=False)


@router.get("/conda/envs")
async def get_conda_envs():
    """
    获取Conda环境列表
    支持多种获取方式：
    1. conda env list 命令
    2. 直接扫描常见的Anaconda/Miniconda安装目录
    """
    envs = []
    error = None
    
    # 方法1: 尝试使用conda命令获取
    try:
        result = subprocess.run(
            ['conda', 'env', 'list', '--json'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
            
            for env_path in data.get('envs', []):
                env_path = env_path.replace('\\', '/')
                
                if '/envs/' in env_path:
                    name = env_path.split('/envs/')[-1]
                else:
                    name = 'base'
                
                # 获取Python版本
                python_version = _get_env_python_version(env_path)
                
                envs.append(CondaEnv(
                    name=name,
                    path=env_path,
                    python_version=python_version,
                    is_active=(name == current_env)
                ))
            
            if envs:
                return {"envs": envs}
                
    except FileNotFoundError:
        error = "Conda命令未找到"
    except subprocess.TimeoutExpired:
        error = "获取Conda环境超时"
    except Exception as e:
        error = str(e)
    
    # 方法2: 直接扫描常见目录
    scanned_envs = _scan_conda_directories()
    if scanned_envs:
        return {"envs": scanned_envs, "source": "scan"}
    
    return {"envs": [], "error": error or "未找到Conda环境"}


def _get_env_python_version(env_path: str) -> Optional[str]:
    """获取环境的Python版本"""
    try:
        # 直接查找python可执行文件
        if platform.system() == 'Windows':
            python_exe = os.path.join(env_path, 'python.exe')
        else:
            python_exe = os.path.join(env_path, 'bin', 'python')
        
        if os.path.exists(python_exe):
            result = subprocess.run(
                [python_exe, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().replace('Python ', '')
    except Exception:
        pass
    return None


def _scan_conda_directories() -> List[CondaEnv]:
    """扫描常见的Conda安装目录
    
    BUG-4修复：支持中文用户名路径
    """
    envs = []
    scanned_paths = set()
    
    # 常见的Conda安装位置
    possible_conda_roots = []
    
    # BUG-4修复：使用多种方式获取用户目录
    user_homes = set()
    
    # 方式1: os.path.expanduser
    try:
        user_homes.add(os.path.expanduser('~'))
    except Exception:
        pass
    
    # 方式2: USERPROFILE环境变量（Windows）
    if platform.system() == 'Windows':
        userprofile = os.environ.get('USERPROFILE', '')
        if userprofile:
            user_homes.add(userprofile)
        
        # 方式3: HOMEDRIVE + HOMEPATH
        homedrive = os.environ.get('HOMEDRIVE', '')
        homepath = os.environ.get('HOMEPATH', '')
        if homedrive and homepath:
            user_homes.add(os.path.join(homedrive, homepath))
    else:
        # 方式4: HOME环境变量（Linux/Mac）
        home = os.environ.get('HOME', '')
        if home:
            user_homes.add(home)
    
    if platform.system() == 'Windows':
        for user_home in user_homes:
            possible_conda_roots.extend([
                os.path.join(user_home, 'anaconda3'),
                os.path.join(user_home, 'miniconda3'),
                os.path.join(user_home, 'Anaconda3'),
                os.path.join(user_home, 'Miniconda3'),
                os.path.join(user_home, 'AppData', 'Local', 'anaconda3'),
                os.path.join(user_home, 'AppData', 'Local', 'miniconda3'),
            ])
        
        # 系统级安装
        possible_conda_roots.extend([
            'C:/ProgramData/anaconda3',
            'C:/ProgramData/miniconda3',
            'C:/Anaconda3',
            'C:/Miniconda3',
            'D:/Anaconda3',
            'D:/Miniconda3',
        ])
        
        # 从CONDA_PREFIX环境变量获取
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            if 'envs' in conda_prefix:
                root = conda_prefix.split('envs')[0].rstrip('/\\')
                possible_conda_roots.insert(0, root)
            else:
                possible_conda_roots.insert(0, conda_prefix)
        
        # 从PATH中查找conda
        path_env = os.environ.get('PATH', '')
        for path_item in path_env.split(os.pathsep):
            if 'conda' in path_item.lower() or 'anaconda' in path_item.lower():
                # 尝试找到根目录
                path_obj = Path(path_item)
                for _ in range(5):  # 最多向上5层
                    if (path_obj / 'envs').exists():
                        possible_conda_roots.insert(0, str(path_obj))
                        break
                    path_obj = path_obj.parent
    else:
        # Linux/Mac
        for user_home in user_homes:
            possible_conda_roots.extend([
                os.path.join(user_home, 'anaconda3'),
                os.path.join(user_home, 'miniconda3'),
            ])
        possible_conda_roots.extend([
            '/opt/anaconda3',
            '/opt/miniconda3',
            '/usr/local/anaconda3',
            '/usr/local/miniconda3',
        ])
    
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    for conda_root in possible_conda_roots:
        if not conda_root or not os.path.isdir(conda_root):
            continue
        
        # 添加base环境
        base_path = conda_root
        if base_path not in scanned_paths and os.path.exists(base_path):
            scanned_paths.add(base_path)
            python_version = _get_env_python_version(base_path)
            if python_version:
                envs.append(CondaEnv(
                    name='base',
                    path=base_path.replace('\\', '/'),
                    python_version=python_version,
                    is_active=('base' == current_env)
                ))
        
        # 扫描envs目录
        envs_dir = os.path.join(conda_root, 'envs')
        if os.path.isdir(envs_dir):
            try:
                for env_name in os.listdir(envs_dir):
                    env_path = os.path.join(envs_dir, env_name)
                    if not os.path.isdir(env_path):
                        continue
                    if env_path in scanned_paths:
                        continue
                    
                    scanned_paths.add(env_path)
                    
                    python_version = _get_env_python_version(env_path)
                    if python_version:
                        envs.append(CondaEnv(
                            name=env_name,
                            path=env_path.replace('\\', '/'),
                            python_version=python_version,
                            is_active=(env_name == current_env)
                        ))
            except PermissionError:
                continue
    
    return envs


@router.post("/conda/scan")
async def scan_conda_directory(path: str):
    """
    扫描指定目录下的Conda环境
    用于用户手动指定Anaconda安装目录
    """
    from pydantic import BaseModel
    
    if not os.path.isdir(path):
        return {"envs": [], "error": f"目录不存在: {path}"}
    
    envs = []
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    # 检查是否是conda根目录（包含envs子目录）
    envs_dir = os.path.join(path, 'envs')
    
    if os.path.isdir(envs_dir):
        # 这是conda根目录，扫描envs下的所有环境
        # 先添加base环境
        python_version = _get_env_python_version(path)
        if python_version:
            envs.append(CondaEnv(
                name='base',
                path=path.replace('\\', '/'),
                python_version=python_version,
                is_active=('base' == current_env)
            ))
        
        # 扫描envs目录
        for env_name in os.listdir(envs_dir):
            env_path = os.path.join(envs_dir, env_name)
            if os.path.isdir(env_path):
                python_version = _get_env_python_version(env_path)
                if python_version:
                    envs.append(CondaEnv(
                        name=env_name,
                        path=env_path.replace('\\', '/'),
                        python_version=python_version,
                        is_active=(env_name == current_env)
                    ))
    else:
        # 检查path本身是否是一个环境目录
        python_version = _get_env_python_version(path)
        if python_version:
            # 从路径提取环境名
            env_name = os.path.basename(path)
            envs.append(CondaEnv(
                name=env_name,
                path=path.replace('\\', '/'),
                python_version=python_version,
                is_active=(env_name == current_env)
            ))
        else:
            return {"envs": [], "error": "指定目录不是有效的Conda环境"}
    
    return {"envs": envs}


@router.get("/disk")
async def get_disk_info():
    """
    获取磁盘信息
    """
    import psutil
    
    disks = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disks.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "fstype": partition.fstype,
                "total": round(usage.total / (1024**3), 1),  # GB
                "used": round(usage.used / (1024**3), 1),
                "free": round(usage.free / (1024**3), 1),
                "percent": usage.percent
            })
        except PermissionError:
            continue
    
    return {"disks": disks}


# ==================== [新增] 系统资源实时监控 ====================

@router.get("/resources")
async def get_system_resources():
    """
    [UI优化] 获取系统资源实时状态
    
    返回CPU、内存、GPU的实时使用情况
    """
    import psutil
    from datetime import datetime
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "freq": None
        },
        "memory": {
            "total": round(psutil.virtual_memory().total / (1024**3), 2),
            "used": round(psutil.virtual_memory().used / (1024**3), 2),
            "percent": psutil.virtual_memory().percent
        },
        "gpu": None
    }
    
    # CPU频率
    try:
        freq = psutil.cpu_freq()
        if freq:
            result["cpu"]["freq"] = round(freq.current, 0)
    except Exception:
        pass
    
    # GPU信息
    try:
        gpu_result = subprocess.run(
            [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if gpu_result.returncode == 0:
            gpus = []
            for line in gpu_result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total": int(parts[2]),
                        "memory_used": int(parts[3]),
                        "utilization": int(parts[4]) if parts[4] != '[N/A]' else 0,
                        "temperature": int(parts[5]) if parts[5] != '[N/A]' else None
                    })
            result["gpu"] = gpus
    except Exception:
        pass
    
    return result
