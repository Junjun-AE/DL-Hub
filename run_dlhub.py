#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL-Hub 启动入口（项目根目录）
==============================
此文件放在项目根目录，方便用户直接运行:
    python run_dlhub.py

内部转发到 dlhub_project/run_dlhub.py 执行。
"""

import sys
import os
from pathlib import Path

# 确保工作目录指向 dlhub_project
project_root = Path(__file__).parent.resolve()
dlhub_project_dir = project_root / 'dlhub_project'

if not dlhub_project_dir.exists():
    print(f"❌ 错误: 未找到 dlhub_project 目录: {dlhub_project_dir}")
    sys.exit(1)

# 将 dlhub_project 加入 sys.path 并切换工作目录
sys.path.insert(0, str(dlhub_project_dir))
os.chdir(str(dlhub_project_dir))

# 导入并运行
from run_dlhub import main
main()
