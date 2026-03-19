#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL-Hub 启动脚本
===============
统一的深度学习任务管理平台启动入口
自动处理npm路径问题

修复：
- 端口占用检测和自动切换
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading
import platform
import shutil
import socket
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              🧠  DL-Hub v2.0                              ║
    ║         Deep Learning Workstation                         ║
    ║                                                           ║
    ║   分类 · 检测 · 分割 · 异常检测 · OCR · 缺陷评分  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind(('127.0.0.1', port))
            return True
    except (OSError, socket.error):
        return False


def find_available_port(start_port: int = 7860, max_attempts: int = 20) -> int:
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return -1


def get_process_using_port(port: int) -> str:
    """获取占用端口的进程信息（仅Windows）"""
    if platform.system() != 'Windows':
        return ""
    
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        # 尝试获取进程名
                        try:
                            tasklist = subprocess.run(
                                ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
                                capture_output=True,
                                text=True,
                                timeout=5,
                                encoding='utf-8',
                                errors='replace'
                            )
                            if tasklist.returncode == 0:
                                lines = tasklist.stdout.strip().split('\n')
                                if len(lines) > 1:
                                    # CSV格式："进程名","PID",...
                                    name = lines[1].split(',')[0].strip('"')
                                    return f"{name} (PID: {pid})"
                        except Exception:
                            pass
                        return f"PID: {pid}"
    except Exception:
        pass
    return ""


def find_npm():
    """
    查找npm可执行文件路径
    Windows上npm可能不在PATH中，需要特殊处理
    """
    # 首先尝试直接查找
    npm_path = shutil.which('npm')
    if npm_path:
        return npm_path
    
    # Windows特殊处理
    if platform.system() == 'Windows':
        # 常见的npm安装位置
        possible_paths = [
            # Node.js默认安装位置
            Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'nodejs' / 'npm.cmd',
            Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / 'nodejs' / 'npm.cmd',
            # AppData位置 (nvm, volta等)
            Path(os.environ.get('APPDATA', '')) / 'npm' / 'npm.cmd',
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'node' / 'npm.cmd',
            # 用户目录
            Path.home() / 'AppData' / 'Roaming' / 'npm' / 'npm.cmd',
            # fnm
            Path.home() / '.fnm' / 'node-versions' / '*' / 'installation' / 'npm.cmd',
        ]
        
        for npm_path in possible_paths:
            # 处理通配符
            if '*' in str(npm_path):
                parent = npm_path.parent.parent
                if parent.exists():
                    for version_dir in parent.iterdir():
                        potential = version_dir / 'installation' / 'npm.cmd'
                        if potential.exists():
                            return str(potential)
            elif npm_path.exists():
                return str(npm_path)
        
        # 尝试从node路径推断npm路径
        node_path = shutil.which('node')
        if node_path:
            node_dir = Path(node_path).parent
            npm_cmd = node_dir / 'npm.cmd'
            if npm_cmd.exists():
                return str(npm_cmd)
            npm = node_dir / 'npm'
            if npm.exists():
                return str(npm)
    
    return None


def find_node():
    """查找node可执行文件路径"""
    node_path = shutil.which('node')
    if node_path:
        return node_path
    
    # Windows特殊处理
    if platform.system() == 'Windows':
        possible_paths = [
            Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'nodejs' / 'node.exe',
            Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / 'nodejs' / 'node.exe',
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
    
    return None


def check_node():
    """检查Node.js是否安装"""
    node_path = find_node()
    if not node_path:
        return False, None
    
    try:
        result = subprocess.run(
            [node_path, '--version'], 
            capture_output=True, 
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"✅ Node.js 已安装: {version}")
        return True, node_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None


def check_npm():
    """检查npm是否安装"""
    npm_path = find_npm()
    if not npm_path:
        return False, None
    
    try:
        result = subprocess.run(
            [npm_path, '--version'], 
            capture_output=True, 
            text=True,
            check=True,
            shell=(platform.system() == 'Windows')  # Windows需要shell
        )
        version = result.stdout.strip()
        print(f"✅ npm 已安装: {version}")
        print(f"   路径: {npm_path}")
        return True, npm_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   npm检查失败: {e}")
        return False, None


def build_frontend(force_rebuild=False):
    """构建前端"""
    frontend_dir = Path(__file__).parent / 'dlhub' / 'frontend'
    dist_dir = frontend_dir / 'dist'
    
    if dist_dir.exists() and not force_rebuild:
        print("✅ 前端已构建，跳过构建步骤")
        return True
    
    node_ok, node_path = check_node()
    if not node_ok:
        print("❌ 错误: 需要安装 Node.js 来构建前端")
        print("   请访问 https://nodejs.org/ 下载安装")
        print("   安装完成后请重新打开终端")
        return False
    
    npm_ok, npm_path = check_npm()
    if not npm_ok:
        print("❌ 错误: npm 未找到")
        print("   npm通常随Node.js一起安装")
        print("   请尝试以下方法:")
        print("   1. 重新安装 Node.js: https://nodejs.org/")
        print("   2. 确保安装时勾选了 'Add to PATH'")
        print("   3. 重新打开终端/命令行窗口")
        if platform.system() == 'Windows':
            print("   4. 或尝试在新的PowerShell中运行")
        return False
    
    print("📦 正在构建前端...")
    print(f"   工作目录: {frontend_dir}")
    
    # 检查package.json是否存在
    if not (frontend_dir / 'package.json').exists():
        print(f"❌ 错误: {frontend_dir / 'package.json'} 不存在")
        return False
    
    # 设置环境变量，确保npm能找到node
    env = os.environ.copy()
    if node_path:
        node_dir = str(Path(node_path).parent)
        env['PATH'] = node_dir + os.pathsep + env.get('PATH', '')
    
    try:
        # 安装依赖
        print("   安装依赖中...")
        result = subprocess.run(
            [npm_path, 'install'], 
            cwd=frontend_dir, 
            check=True,
            capture_output=True,
            env=env,
            shell=(platform.system() == 'Windows'),
            encoding='utf-8',
            errors='replace'
        )
        print("   ✅ 依赖安装完成")
        
        # 构建
        print("   构建生产版本中...")
        result = subprocess.run(
            [npm_path, 'run', 'build'], 
            cwd=frontend_dir, 
            check=True,
            capture_output=True,
            env=env,
            shell=(platform.system() == 'Windows'),
            encoding='utf-8',
            errors='replace'
        )
        print("   ✅ 前端构建完成")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败")
        if e.stdout:
            print(f"   输出: {e.stdout}")
        if e.stderr:
            print(f"   错误: {e.stderr}")
        return False


def check_python_dependencies():
    """检查Python依赖"""
    required = ['fastapi', 'uvicorn', 'pydantic', 'psutil']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  缺少Python依赖: {', '.join(missing)}")
        print("   正在安装...")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'fastapi', 'uvicorn[standard]', 'pydantic', 'python-multipart', 'psutil'
            ], check=True, capture_output=True)
            print("   ✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 安装失败: {e}")
            return False
    else:
        print("✅ Python依赖已满足")
    
    return True


def setup_offline_resources():
    """
    下载 login 页面的 CDN 资源到本地，实现完全离线运行。
    首次运行时下载（需要网络），后续启动自动跳过。
    """
    dist_dir = Path(__file__).parent / 'dlhub' / 'frontend' / 'dist'
    vendor_dir = dist_dir / 'vendor'
    login_html = dist_dir / 'login.html'
    
    if not login_html.exists():
        return True  # 没有 login.html 就不需要处理
    
    # CDN 资源列表
    cdn_resources = [
        ("https://unpkg.com/react@18.3.1/umd/react.production.min.js", "react.production.min.js"),
        ("https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js", "react-dom.production.min.js"),
        ("https://unpkg.com/@babel/standalone@7.26.9/babel.min.js", "babel.min.js"),
        ("https://cdn.tailwindcss.com", "tailwindcss.js"),
    ]
    
    # 检查是否已经完成离线化
    login_content = login_html.read_text(encoding='utf-8')
    if '/vendor/' in login_content and all((vendor_dir / name).exists() for _, name in cdn_resources):
        print("✅ 离线资源已就绪")
        return True
    
    # 检查 login.html 中是否还有在线CDN引用
    has_online_cdn = any(url in login_content for url, _ in cdn_resources)
    if not has_online_cdn:
        print("✅ 登录页面已是离线版本")
        return True
    
    print("📦 首次运行：下载离线资源（仅需一次）...")
    vendor_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载资源
    download_ok = True
    for url, filename in cdn_resources:
        local_path = vendor_dir / filename
        if local_path.exists() and local_path.stat().st_size > 1024:
            print(f"   ✅ {filename} (已存在)")
            continue
        
        print(f"   📥 下载 {filename}...", end=" ", flush=True)
        try:
            import urllib.request
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={"User-Agent": "DL-Hub/2.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                data = resp.read()
            local_path.write_bytes(data)
            print(f"✅ ({len(data)//1024} KB)")
        except Exception as e:
            print(f"❌ ({e})")
            download_ok = False
    
    if not download_ok:
        print("   ⚠️  部分资源下载失败，登录页面可能需要在线访问")
        print("   💡 提示：连接网络后重新启动即可自动补全")
        return True  # 不阻断启动
    
    # 替换 login.html 中的 CDN 引用
    replacements = [
        ('src="https://cdn.tailwindcss.com"', 'src="/vendor/tailwindcss.js"'),
        ('src="https://unpkg.com/react@18.3.1/umd/react.production.min.js"', 'src="/vendor/react.production.min.js"'),
        ('src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js"', 'src="/vendor/react-dom.production.min.js"'),
        ('src="https://unpkg.com/@babel/standalone@7.26.9/babel.min.js"', 'src="/vendor/babel.min.js"'),
    ]
    
    # 备份原文件
    backup = dist_dir / 'login.html.online_backup'
    if not backup.exists():
        shutil.copy(login_html, backup)
    
    modified = False
    for old, new in replacements:
        if old in login_content:
            login_content = login_content.replace(old, new)
            modified = True
    
    if modified:
        login_html.write_text(login_content, encoding='utf-8')
        print("   ✅ 登录页面已切换为离线模式")
    
    print("✅ 离线资源配置完成")
    return True


def open_browser_delayed(url, delay=2):
    """延迟打开浏览器 - 强制新窗口加载"""
    def _open():
        time.sleep(delay)
        # 加时间戳参数强制浏览器不使用缓存的页面
        import time as _t
        bust_url = f"{url}?_t={int(_t.time())}"
        # 尝试打开新窗口（而非复用旧标签页）
        webbrowser.open_new(bust_url)
    
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DL-Hub 深度学习任务管理平台')
    parser.add_argument('--port', type=int, default=7860, help='服务端口 (默认: 7860)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务地址 (默认: 0.0.0.0)')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--rebuild', action='store_true', help='强制重新构建前端')
    parser.add_argument('--dev', action='store_true', help='开发模式 (启用热重载)')
    parser.add_argument('--skip-frontend', action='store_true', help='跳过前端构建检查')
    
    args = parser.parse_args()
    
    print_banner()
    
    # ==================== 自动修正 app_base_dir ====================
    # 确保 dlhub_config.json 中的 app_base_dir 指向当前代码目录
    # 这样5大子任务运行的是当前目录的代码（包含用户组件等新功能）
    try:
        config_file = Path(__file__).parent / 'dlhub_config.json'
        project_root = Path(__file__).parent.parent  # Deep_learning_tools3/
        if config_file.exists():
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            old_base = config_data.get('app_base_dir', '')
            new_base = str(project_root).replace('\\', '/')
            if old_base != new_base:
                config_data['app_base_dir'] = new_base
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                print(f"📂 已更新应用基础目录: {new_base}")
    except Exception as e:
        print(f"⚠️  更新配置失败(非致命): {e}")
    
    # 检查Python依赖
    print("\n🔍 检查依赖...")
    if not check_python_dependencies():
        sys.exit(1)
    
    # 离线资源检查（首次运行时下载CDN资源，后续自动跳过）
    print("\n📦 检查离线资源...")
    setup_offline_resources()
    
    # 构建前端
    if not args.skip_frontend:
        print("\n🔨 检查前端...")
        if not build_frontend(force_rebuild=args.rebuild):
            print("\n⚠️  前端构建失败")
            print("   可以使用 --skip-frontend 跳过前端构建")
            print("   API接口可用，但Web界面无法正常显示")
            response = input("\n   是否继续启动? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # 检查端口是否可用
    print(f"\n🔍 检查端口 {args.port}...")
    actual_port = args.port
    
    if not is_port_available(args.port):
        process_info = get_process_using_port(args.port)
        print(f"⚠️  端口 {args.port} 已被占用")
        if process_info:
            print(f"   占用进程: {process_info}")
        
        # 查找可用端口
        available_port = find_available_port(args.port + 1)
        
        if available_port > 0:
            print(f"\n   可选操作:")
            print(f"   1. 使用其他端口 {available_port}")
            print(f"   2. 手动关闭占用端口的程序后重试")
            print(f"   3. 退出")
            
            choice = input(f"\n   请选择 (1/2/3，默认1): ").strip()
            
            if choice == '2':
                print("\n   请手动关闭占用端口的程序，然后重新运行 DL-Hub")
                sys.exit(0)
            elif choice == '3':
                print("\n   已取消启动")
                sys.exit(0)
            else:
                actual_port = available_port
                print(f"\n✅ 将使用端口 {actual_port}")
        else:
            print(f"\n❌ 无法找到可用端口 (尝试了 {args.port}-{args.port + 19})")
            print("   请关闭一些程序后重试")
            sys.exit(1)
    else:
        print(f"✅ 端口 {args.port} 可用")
    
    # 启动服务器
    print(f"\n🚀 启动 DL-Hub 服务器...")
    print(f"   地址: http://localhost:{actual_port}")
    print(f"   API文档: http://localhost:{actual_port}/api/docs")
    print(f"   按 Ctrl+C 停止服务")
    print()
    
    # 自动打开浏览器
    if not args.no_browser:
        open_browser_delayed(f'http://localhost:{actual_port}')
    
    # 启动uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "dlhub.backend.main:app",
            host=args.host,
            port=actual_port,
            reload=args.dev,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 DL-Hub 已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
