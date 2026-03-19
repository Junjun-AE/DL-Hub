# -*- coding: utf-8 -*-
"""PatchCore 工业级异常检测系统 - 入口文件"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.app import create_app, main

if __name__ == "__main__":
    main()
