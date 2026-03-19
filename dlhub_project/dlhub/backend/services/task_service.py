# -*- coding: utf-8 -*-
"""
任务服务
=======
管理任务的创建、查询、更新、删除
支持任意位置的工作目录

修复的BUG:
- BUG 1: 任务目录重复创建风险 (添加short_id到目录名)
- BUG 2: Conda环境验证 (添加验证函数)
- BUG 3: 重复导入检测 (导入前检查)
- BUG 6: 特殊字符处理完善 (更严格的过滤)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import uuid
import shutil
import subprocess
import os
from datetime import datetime
import re
import platform

from ..config import (
    get_task_paths, add_task_path, remove_task_path, 
    get_task_by_id, DLHUB_SIGNATURE
)


class TaskService:
    """任务管理服务"""
    
    # 有效的任务类型
    VALID_TASK_TYPES = ['classification', 'detection', 'segmentation', 'anomaly', 'ocr', 'sevseg']
    
    # 任务类型对应的依赖（用于环境检查提示）
    TASK_DEPENDENCIES = {
        'classification': ['torch', 'timm'],
        'detection': ['torch', 'ultralytics'],
        'segmentation': ['torch', 'mmsegmentation'],
        'anomaly': ['torch', 'scikit-learn'],
        'ocr': ['paddlepaddle', 'paddleocr'],
        'sevseg': ['torch', 'ultralytics', 'opencv-contrib-python'],
    }
    
    def __init__(self):
        """初始化任务服务"""
        pass
    
    def validate_conda_env(self, env_name: str) -> tuple:
        """
        验证Conda环境是否存在且可用
        
        [Bug-5修复]: 改进验证逻辑，更加宽松
        - 即使conda命令不可用，也尝试目录检查
        - 最终允许用户自行决定是否继续
        
        Args:
            env_name: 环境名称
            
        Returns:
            (is_valid, warning_message): 
            - is_valid: True表示可以继续，False表示必须停止（几乎不会返回False）
            - warning_message: 警告信息，None表示验证完全通过
        """
        import os
        
        verified_by = None  # 记录验证方式
        conda_available = True  # 记录conda命令是否可用
        
        # 方法1: 检查环境是否在conda env list中
        try:
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True,
                text=True,
                timeout=15,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                # 检查环境名是否在输出中
                for line in result.stdout.split('\n'):
                    # 格式: envname    path 或 envname *  path (带*表示当前激活)
                    parts = line.split()
                    if parts and (parts[0] == env_name or (len(parts) > 1 and parts[0] == env_name)):
                        verified_by = "conda_env_list"
                        return True, None  # 完全验证通过
                    # 也检查base环境
                    if env_name == 'base' and 'base' in line.lower():
                        verified_by = "conda_env_list_base"
                        return True, None
        except FileNotFoundError:
            # conda命令不存在，但不直接返回错误，继续尝试其他方法
            conda_available = False
            print(f"[TaskService] conda命令不可用，尝试目录检查...")
        except subprocess.TimeoutExpired:
            print(f"[TaskService] conda env list超时，尝试目录检查...")
        except Exception as e:
            print(f"[TaskService] conda env list出错: {e}，尝试目录检查...")
        
        # 方法2: 直接检查常见的conda环境目录
        possible_paths = self._get_possible_env_paths(env_name)
        for env_path in possible_paths:
            if os.path.isdir(env_path):
                # 检查是否有python可执行文件
                if platform.system() == 'Windows':
                    python_exe = os.path.join(env_path, 'python.exe')
                else:
                    python_exe = os.path.join(env_path, 'bin', 'python')
                
                if os.path.exists(python_exe):
                    verified_by = "directory_check"
                    print(f"[TaskService] 通过目录检查找到环境: {env_path}")
                    return True, None  # 通过目录检查验证
        
        # 方法3: 如果前面都失败了，返回警告但允许继续
        # 用户可能使用了我们未检测到的路径，允许他们自行决定
        if not conda_available:
            warning_msg = (
                f"⚠️ Conda命令不在系统PATH中，且未在常见位置找到环境 '{env_name}'。\n"
                f"   这不会阻止您创建任务，但启动时可能会失败。\n"
                f"   建议：\n"
                f"   1. 确认Anaconda/Miniconda已正确安装\n"
                f"   2. 确认环境名称正确\n"
                f"   3. 您仍可继续，系统会在启动时使用完整路径调用"
            )
        else:
            warning_msg = (
                f"⚠️ 未在Conda环境列表中找到 '{env_name}'。\n"
                f"   可能的原因：\n"
                f"   1. 环境名称拼写错误\n"
                f"   2. 环境尚未创建\n"
                f"   3. 环境在其他Conda安装中\n"
                f"   您仍可继续创建任务，但启动时可能会失败。"
            )
        
        # 始终返回True，让用户自己决定是否继续
        return True, warning_msg
    
    def _get_possible_env_paths(self, env_name: str) -> list:
        """
        获取可能的环境路径
        
        扩展搜索范围，包括更多常见安装位置
        """
        import os
        paths = []
        seen = set()  # 避免重复
        
        def add_path(p):
            p = os.path.normpath(p)
            if p not in seen:
                seen.add(p)
                paths.append(p)
        
        # 从CONDA_PREFIX推断
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            if 'envs' in conda_prefix:
                base_path = conda_prefix.split('envs')[0].rstrip('/\\')
            else:
                base_path = conda_prefix
            
            if env_name == 'base':
                add_path(base_path)
            else:
                add_path(os.path.join(base_path, 'envs', env_name))
        
        # 从CONDA_EXE推断（更可靠）
        conda_exe = os.environ.get('CONDA_EXE', '')
        if conda_exe:
            # conda.exe通常在 anaconda3/Scripts/conda.exe 或 anaconda3/condabin/conda.exe
            conda_dir = os.path.dirname(os.path.dirname(conda_exe))
            if os.path.isdir(conda_dir):
                if env_name == 'base':
                    add_path(conda_dir)
                else:
                    add_path(os.path.join(conda_dir, 'envs', env_name))
        
        # 用户目录 - 多种获取方式
        user_homes = set()
        try:
            user_homes.add(os.path.expanduser('~'))
        except Exception:
            pass
        
        # Windows特殊处理
        if platform.system() == 'Windows':
            userprofile = os.environ.get('USERPROFILE', '')
            if userprofile:
                user_homes.add(userprofile)
            
            homedrive = os.environ.get('HOMEDRIVE', '')
            homepath = os.environ.get('HOMEPATH', '')
            if homedrive and homepath:
                user_homes.add(os.path.join(homedrive, homepath))
        else:
            home = os.environ.get('HOME', '')
            if home:
                user_homes.add(home)
        
        # 常见位置
        for user_home in user_homes:
            common_subdirs = [
                'anaconda3', 'miniconda3', 'Anaconda3', 'Miniconda3',
                'anaconda', 'miniconda', 'Anaconda', 'Miniconda',
                '.conda',  # 有些安装会在这里
            ]
            for subdir in common_subdirs:
                root = os.path.join(user_home, subdir)
                if env_name == 'base':
                    add_path(root)
                else:
                    add_path(os.path.join(root, 'envs', env_name))
            
            # AppData/Local (Windows)
            if platform.system() == 'Windows':
                appdata_local = os.path.join(user_home, 'AppData', 'Local')
                for subdir in common_subdirs:
                    root = os.path.join(appdata_local, subdir)
                    if env_name == 'base':
                        add_path(root)
                    else:
                        add_path(os.path.join(root, 'envs', env_name))
        
        # 系统级安装位置
        if platform.system() == 'Windows':
            system_roots = [
                'C:/ProgramData/anaconda3',
                'C:/ProgramData/miniconda3',
                'C:/Anaconda3',
                'C:/Miniconda3',
                'D:/Anaconda3',
                'D:/Miniconda3',
                'E:/Anaconda3',
                'E:/Miniconda3',
            ]
        else:
            system_roots = [
                '/opt/anaconda3',
                '/opt/miniconda3',
                '/usr/local/anaconda3',
                '/usr/local/miniconda3',
                '/home/anaconda3',
                '/home/miniconda3',
            ]
        
        for root in system_roots:
            if env_name == 'base':
                add_path(root)
            else:
                add_path(os.path.join(root, 'envs', env_name))
        
        # 从PATH中查找可能的conda安装
        path_env = os.environ.get('PATH', '')
        for path_item in path_env.split(os.pathsep):
            path_lower = path_item.lower()
            if 'conda' in path_lower or 'anaconda' in path_lower:
                # 尝试找到根目录
                p = Path(path_item)
                for _ in range(5):  # 最多向上5层
                    potential_root = str(p)
                    envs_dir = p / 'envs'
                    if envs_dir.exists():
                        if env_name == 'base':
                            add_path(potential_root)
                        else:
                            add_path(os.path.join(potential_root, 'envs', env_name))
                        break
                    p = p.parent
        
        return paths
    
    def create_task(
        self,
        task_type: str,
        task_name: str,
        work_dir: str,
        conda_env: str,
        description: str = "",
        skip_env_validation: bool = False
    ) -> Dict[str, Any]:
        """
        创建新任务
        
        Args:
            task_type: 任务类型 (classification/detection/segmentation/anomaly/ocr)
            task_name: 任务名称
            work_dir: 工作目录路径
            conda_env: Conda环境名称
            description: 任务描述
            skip_env_validation: 是否跳过环境验证
            
        Returns:
            任务信息字典
        """
        # 验证任务类型
        if task_type not in self.VALID_TASK_TYPES:
            raise ValueError(f"无效的任务类型: {task_type}")
        
        # 验证工作目录
        work_path = Path(work_dir)
        if not work_path.exists():
            raise ValueError(f"工作目录不存在: {work_dir}")
        if not work_path.is_dir():
            raise ValueError(f"路径不是目录: {work_dir}")
        
        # 验证Conda环境 (BUG 2 修复)
        if not skip_env_validation:
            env_valid, env_error = self.validate_conda_env(conda_env)
            if not env_valid:
                raise ValueError(f"Conda环境验证失败: {env_error}")
        
        # 生成任务ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%Y%m%d")
        short_id = uuid.uuid4().hex[:6]
        task_id = f"{task_type[:3]}_{timestamp}_{short_id}"
        
        # 清理任务名称 (BUG 6 修复)
        safe_name = self._sanitize_name(task_name)
        
        # 获取任务类型中文名
        type_cn = self._get_type_name(task_type)
        
        # 改进目录名格式：任务名称_任务类型_日期_短ID
        # 例如：猫狗分类_图像分类_20260129_a1b2c3
        task_dir = work_path / f"{safe_name}_{type_cn}_{date_str}_{short_id}"
        
        # 再次检查目录是否存在（极端情况下的额外保护）
        if task_dir.exists():
            extra_id = uuid.uuid4().hex[:4]
            task_dir = work_path / f"{safe_name}_{type_cn}_{date_str}_{short_id}_{extra_id}"
        
        task_dir.mkdir(parents=True, exist_ok=False)  # exist_ok=False 确保不覆盖
        
        # 创建必要的子目录
        (task_dir / ".dlhub").mkdir(exist_ok=True)
        (task_dir / "output").mkdir(exist_ok=True)
        
        # 创建任务元数据
        now = datetime.now().isoformat()
        task_meta = {
            "dlhub_signature": DLHUB_SIGNATURE,  # 签名用于验证
            "dlhub_version": "2.0",
            "task_id": task_id,
            "task_type": task_type,
            "task_name": task_name,
            "description": description,
            "conda_env": conda_env,
            "created_at": now,
            "updated_at": now,
            "status": "idle"
        }
        
        # 保存元数据
        meta_file = task_dir / ".dlhub" / "task.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(task_meta, f, ensure_ascii=False, indent=2)
        
        # 添加到配置（检查是否成功）
        if not add_task_path(str(task_dir), task_type, task_id):
            raise ValueError("保存任务配置失败，请检查配置文件权限")
        
        print(f"[TaskService] 任务创建成功: {task_id} -> {task_dir}")
        
        # 返回包含路径的任务信息
        task_meta['path'] = str(task_dir)
        return task_meta
    
    def import_task(self, task_dir: str, expected_type: str, conda_env: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        导入已有任务目录
        
        Args:
            task_dir: 任务目录路径
            expected_type: 期望的任务类型
            conda_env: 可选，为缺少环境配置的任务指定Conda环境
            force: 是否强制导入（替换已存在的任务记录）
            
        Returns:
            任务信息字典
            
        Raises:
            ValueError: 如果目录无效或任务类型不匹配
        """
        task_path = Path(task_dir)
        
        # 检查目录是否存在
        if not task_path.exists():
            raise ValueError(f"目录不存在: {task_dir}")
        if not task_path.is_dir():
            raise ValueError(f"路径不是目录: {task_dir}")
        
        # 检查是否已导入
        existing_tasks = get_task_paths(auto_cleanup=True)
        existing_task = None
        for t in existing_tasks:
            if Path(t.get("path", "")).resolve() == task_path.resolve():
                existing_task = t
                break
        
        if existing_task:
            if force:
                # 强制导入：删除旧记录
                remove_task_path(existing_task.get("path", ""))
                print(f"[TaskService] 强制导入：已删除旧记录 {existing_task.get('path')}")
            else:
                # 返回特殊错误码，让前端显示替换确认
                raise ValueError("TASK_EXISTS:该任务目录已导入，是否要替换？")
        
        # 检查是否有task.json
        meta_file = task_path / ".dlhub" / "task.json"
        if not meta_file.exists():
            raise ValueError("无效的任务目录：缺少 .dlhub/task.json 文件")
        
        # 读取元数据
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                task_meta = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("任务配置文件损坏：无法解析 task.json")
        
        # 验证签名 - 更灵活的验证方式
        # 签名只需要包含 "dlhub" 即可，支持不同版本
        signature = task_meta.get("dlhub_signature", "")
        if not signature or "dlhub" not in signature.lower():
            raise ValueError("任务目录验证失败：签名不匹配或任务已损坏（需要包含dlhub标识）")
        
        # 验证任务类型
        actual_type = task_meta.get("task_type")
        if actual_type != expected_type:
            raise ValueError(
                f"任务类型不匹配：该目录是 {self._get_type_name(actual_type)} 任务，"
                f"不能导入到 {self._get_type_name(expected_type)}"
            )
        
        # 检查必要字段
        required_fields = ["task_id", "task_type", "task_name"]
        for field in required_fields:
            if field not in task_meta:
                raise ValueError(f"任务配置不完整：缺少 {field} 字段")
        
        # BUG-5修复：处理缺少conda_env的旧任务
        if 'conda_env' not in task_meta or not task_meta.get('conda_env'):
            if conda_env:
                # 使用传入的环境名
                task_meta['conda_env'] = conda_env
                task_meta['updated_at'] = datetime.now().isoformat()
                # 保存更新后的配置
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(task_meta, f, ensure_ascii=False, indent=2)
            else:
                # 返回特殊标记，提示前端需要选择环境
                raise ValueError(
                    "NEED_CONDA_ENV:该任务缺少Conda环境配置，请选择一个环境后再导入"
                )
        
        # 添加到配置
        add_task_path(str(task_path), actual_type, task_meta["task_id"])
        
        # 返回任务信息
        task_meta['path'] = str(task_path)
        return task_meta
    
    def list_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出任务
        
        Args:
            task_type: 可选，按任务类型过滤
            
        Returns:
            任务列表
        """
        tasks = []
        # auto_cleanup=True 会自动清理已删除的目录 (BUG 7 修复)
        task_paths = get_task_paths(auto_cleanup=True)
        
        for task_info in task_paths:
            # 过滤任务类型
            if task_type and task_info.get("task_type") != task_type:
                continue
            
            task_path = Path(task_info.get("path", ""))
            
            # 检查目录是否存在
            if not task_path.exists():
                continue
            
            # 检查是否有task.json
            task_json = task_path / ".dlhub" / "task.json"
            if not task_json.exists():
                continue
            
            try:
                with open(task_json, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                
                task['path'] = str(task_path)
                tasks.append(task)
            except Exception as e:
                print(f"读取任务失败 {task_path}: {e}")
        
        # 按更新时间降序排序
        tasks.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return tasks
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单个任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息，不存在返回None
        """
        task_info = get_task_by_id(task_id)
        if not task_info:
            return None
        
        task_path = Path(task_info.get("path", ""))
        task_json = task_path / ".dlhub" / "task.json"
        
        if not task_json.exists():
            return None
        
        try:
            with open(task_json, 'r', encoding='utf-8') as f:
                task = json.load(f)
            task['path'] = str(task_path)
            return task
        except Exception as e:
            print(f"读取任务失败 {task_path}: {e}")
            return None
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新任务信息
        
        Args:
            task_id: 任务ID
            updates: 要更新的字段
            
        Returns:
            是否更新成功
        """
        task = self.get_task(task_id)
        if not task:
            return False
        
        task_dir = Path(task['path'])
        meta_file = task_dir / ".dlhub" / "task.json"
        
        # 读取当前元数据
        with open(meta_file, 'r', encoding='utf-8') as f:
            task_meta = json.load(f)
        
        # 更新字段（保护某些字段不被修改）
        protected_fields = ['task_id', 'created_at', 'path', 'dlhub_signature', 'dlhub_version']
        for key, value in updates.items():
            if key not in protected_fields:
                task_meta[key] = value
        
        # 更新时间戳
        task_meta['updated_at'] = datetime.now().isoformat()
        
        # 保存
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(task_meta, f, ensure_ascii=False, indent=2)
        
        return True
    
    def delete_task(self, task_id: str, delete_files: bool = True, force: bool = False) -> bool:
        """
        删除任务
        
        BUG-2修复：添加文件锁检查，支持强制删除
        
        Args:
            task_id: 任务ID
            delete_files: 是否删除文件（默认True）
            force: 是否强制删除（忽略文件锁错误）
            
        Returns:
            是否删除成功
            
        Raises:
            ValueError: 如果文件被锁定且非强制删除
        """
        task = self.get_task(task_id)
        if not task:
            return False
        
        task_dir = Path(task['path'])
        
        # 从配置中移除
        remove_task_path(str(task_dir))
        
        # 删除文件
        if delete_files and task_dir.exists():
            try:
                # BUG-2修复：先尝试检测文件是否可删除
                if not force:
                    locked_files = self._check_locked_files(task_dir)
                    if locked_files:
                        # 恢复配置
                        add_task_path(str(task_dir), task.get('task_type', ''), task_id)
                        raise ValueError(
                            f"以下文件正被其他程序占用，无法删除：\n" +
                            "\n".join(locked_files[:5]) +
                            (f"\n...等共{len(locked_files)}个文件" if len(locked_files) > 5 else "")
                        )
                
                shutil.rmtree(task_dir, onerror=self._handle_remove_error if force else None)
            except ValueError:
                raise  # 重新抛出文件锁错误
            except Exception as e:
                if not force:
                    # 恢复配置
                    add_task_path(str(task_dir), task.get('task_type', ''), task_id)
                    raise ValueError(f"删除任务文件失败: {e}")
                print(f"强制删除任务文件时出错 {task_dir}: {e}")
        
        return True
    
    def _check_locked_files(self, directory: Path) -> List[str]:
        """
        检查目录中是否有被锁定的文件
        
        [Bug-2修复]:
        - 使用只读模式检测，避免修改文件时间戳
        - 正确处理只读文件（不应被视为锁定）
        - 跳过符号链接
        - Windows下使用msvcrt检测排他锁
        
        Args:
            directory: 目录路径
            
        Returns:
            被锁定的文件列表
        """
        locked_files = []
        
        try:
            for item in directory.rglob('*'):
                # 跳过目录和符号链接
                if not item.is_file() or item.is_symlink():
                    continue
                
                try:
                    # 首先尝试以只读二进制模式打开
                    # 这不会修改文件的访问时间
                    with open(item, 'rb') as f:
                        # Windows下检查是否有排他锁
                        if platform.system() == 'Windows':
                            try:
                                import msvcrt
                                # 尝试获取非阻塞锁
                                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                                # 如果成功，立即释放
                                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                            except (IOError, OSError):
                                # 无法获取锁，说明文件被占用
                                locked_files.append(str(item.relative_to(directory)))
                            except ImportError:
                                # msvcrt不可用，跳过Windows锁检测
                                pass
                        else:
                            # Unix系统：尝试使用fcntl检测锁
                            try:
                                import fcntl
                                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            except (IOError, OSError):
                                locked_files.append(str(item.relative_to(directory)))
                            except ImportError:
                                # fcntl不可用
                                pass
                                
                except PermissionError:
                    # 权限被拒绝 - 可能是系统文件或被其他进程独占
                    # 但这不一定是"锁定"，可能只是权限问题
                    # 尝试检查是否只是只读
                    try:
                        if os.access(item, os.R_OK):
                            # 可读但不可写 - 可能只是只读文件，不算锁定
                            pass
                        else:
                            locked_files.append(str(item.relative_to(directory)))
                    except Exception:
                        locked_files.append(str(item.relative_to(directory)))
                except (IOError, OSError) as e:
                    # 其他IO错误，可能是文件被锁定
                    locked_files.append(str(item.relative_to(directory)))
                    
        except Exception as e:
            print(f"[TaskService] 检查锁定文件时出错: {e}")
        
        return locked_files
    
    def _handle_remove_error(self, func, path, exc_info):
        """
        处理删除文件时的错误（强制删除模式）
        """
        import stat
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            print(f"无法删除文件 {path}: {e}")
    
    def update_task_status(self, task_id: str, status: str, extra: Dict[str, Any] = None) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            extra: 额外信息
            
        Returns:
            是否更新成功
        """
        updates = {'status': status}
        if extra:
            updates.update(extra)
        return self.update_task(task_id, updates)
    
    def _sanitize_name(self, name: str) -> str:
        """
        清理名称，移除特殊字符 (BUG 6 完善修复)
        
        Args:
            name: 原始名称
            
        Returns:
            清理后的名称
        """
        if not name:
            return "unnamed"
        
        # 1. 移除控制字符 (NULL, 换行等)
        safe_name = re.sub(r'[\x00-\x1f\x7f]', '', name)
        
        # 2. 移除路径遍历字符
        safe_name = safe_name.replace('..', '')
        safe_name = safe_name.replace('/', '')
        safe_name = safe_name.replace('\\', '')
        
        # 3. 只保留中文、字母、数字、空格、下划线、短横线
        safe_name = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', safe_name)
        
        # 4. 将连续空格替换为单个下划线
        safe_name = re.sub(r'\s+', '_', safe_name.strip())
        
        # 5. 移除开头和结尾的下划线和短横线
        safe_name = safe_name.strip('_-')
        
        # 6. 限制长度
        if len(safe_name) > 50:
            safe_name = safe_name[:50].rstrip('_-')
        
        return safe_name if safe_name else "unnamed"
    
    def _get_type_name(self, task_type: str) -> str:
        """获取任务类型的中文名称"""
        type_names = {
            'classification': '图像分类',
            'detection': '目标检测',
            'segmentation': '语义分割',
            'anomaly': '异常检测',
            'ocr': 'OCR识别',
            'sevseg': '工业缺陷检测',
        }
        return type_names.get(task_type, task_type)
    
    def copy_task(self, task_id: str, new_name: Optional[str] = None) -> Dict[str, Any]:
        """
        复制任务（优化-7）
        
        Args:
            task_id: 源任务ID
            new_name: 新任务名称（可选，默认为"原名称_副本"）
            
        Returns:
            新任务信息字典
        """
        # 获取源任务
        source_task = self.get_task(task_id)
        if not source_task:
            raise ValueError(f"任务不存在: {task_id}")
        
        source_path = Path(source_task['path'])
        if not source_path.exists():
            raise ValueError(f"任务目录不存在: {source_path}")
        
        # 生成新任务名称
        if not new_name:
            new_name = f"{source_task['task_name']}_副本"
        
        # 创建新任务（在同一父目录下）
        parent_dir = source_path.parent
        
        new_task = self.create_task(
            task_type=source_task['task_type'],
            task_name=new_name,
            work_dir=str(parent_dir),
            conda_env=source_task.get('conda_env', ''),
            description=source_task.get('description', '') + " (复制自: " + source_task['task_name'] + ")"
        )
        
        new_path = Path(new_task['path'])
        
        # 复制数据文件（不复制.dlhub目录，因为已经创建了新的）
        try:
            for item in source_path.iterdir():
                if item.name == '.dlhub':
                    continue  # 跳过配置目录
                if item.name == 'output':
                    continue  # 跳过输出目录（通常很大）
                
                dest = new_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        except Exception as e:
            print(f"复制文件时出错: {e}")
            # 不因为文件复制失败而中断
        
        return new_task
