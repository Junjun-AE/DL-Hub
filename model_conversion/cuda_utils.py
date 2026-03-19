"""
CUDA utilities for model conversion.

This module provides CUDA context management to avoid conflicts
between different CUDA-based inference engines.

FIXED VERSION:
1. Added missing exports (init_pycuda, PYCUDA_AVAILABLE, cuda)
2. Use singleton pattern for CUDA context to avoid conflicts
3. Proper integration with TensorRT's CUDA context management
4. Support for CUDA streams instead of explicit context push/pop
5. Added atexit handler to properly clean up CUDA context

Author: Model Converter Team
Version: 2.1.0 (Fixed CUDA context cleanup)
"""

import atexit
import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ============================================================================
# 全局变量 (修复: 添加 converter_tensorrt.py 需要的导出)
# ============================================================================

# PyCUDA 可用性标志
PYCUDA_AVAILABLE = False

# PyCUDA cuda 模块引用 (延迟初始化)
cuda = None

# 全局初始化标志
_pycuda_initialized = False
_init_lock = threading.Lock()

# 全局单例上下文管理器
_global_cuda_manager: Optional['CudaContextManager'] = None


def _cleanup_cuda_at_exit():
    """程序退出时清理 CUDA 上下文"""
    global _global_cuda_manager
    if _global_cuda_manager is not None:
        try:
            _global_cuda_manager.cleanup()
            _global_cuda_manager = None
            logger.debug("CUDA context cleaned up at exit")
        except Exception as e:
            # 在退出时忽略清理错误
            pass


# 注册退出处理器
atexit.register(_cleanup_cuda_at_exit)


def init_pycuda() -> bool:
    """
    初始化 PyCUDA (线程安全, 单例模式)
    
    修复说明:
    - 不使用 pycuda.autoinit，而是手动控制上下文创建
    - 使用全局单例确保整个程序使用同一个上下文
    - 这避免了 TensorRT 构建和验证阶段使用不同上下文的问题
    
    Returns:
        True 如果初始化成功，False 否则
    """
    global PYCUDA_AVAILABLE, cuda, _pycuda_initialized
    
    with _init_lock:
        if _pycuda_initialized:
            return PYCUDA_AVAILABLE
        
        try:
            import pycuda.driver as pycuda_driver
            
            # 手动初始化 CUDA driver (不创建上下文)
            pycuda_driver.init()
            
            # 检查是否有可用的 GPU
            device_count = pycuda_driver.Device.count()
            if device_count == 0:
                logger.warning("No CUDA devices found")
                _pycuda_initialized = True
                return False
            
            # 设置全局引用
            cuda = pycuda_driver
            PYCUDA_AVAILABLE = True
            _pycuda_initialized = True
            
            logger.info(f"PyCUDA initialized successfully, {device_count} device(s) found")
            return True
            
        except ImportError as e:
            logger.warning(f"PyCUDA not available: {e}")
            _pycuda_initialized = True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PyCUDA: {e}")
            _pycuda_initialized = True
            return False


def get_global_cuda_manager(device_id: int = 0) -> Optional['CudaContextManager']:
    """
    获取全局 CUDA 上下文管理器 (单例)
    
    这确保整个程序使用同一个 CUDA 上下文，
    避免 TensorRT 构建和验证阶段的上下文冲突。
    
    Args:
        device_id: CUDA 设备 ID
        
    Returns:
        CudaContextManager 单例，如果 CUDA 不可用则返回 None
    """
    global _global_cuda_manager
    
    if _global_cuda_manager is None:
        if not init_pycuda():
            return None
        try:
            _global_cuda_manager = CudaContextManager(device_id=device_id)
        except Exception as e:
            logger.error(f"Failed to create global CUDA manager: {e}")
            return None
    
    return _global_cuda_manager


def cleanup_cuda_context():
    """
    显式清理 CUDA 上下文（安全版本）
    
    这个函数可以在模型转换完成后调用，
    以确保 CUDA 上下文被正确释放，避免 PyCUDA 退出错误。
    
    注意：此函数设计为安全调用，即使没有初始化 CUDA 或已经清理过也不会报错。
    """
    global _global_cuda_manager, _pycuda_initialized
    
    if _global_cuda_manager is not None:
        try:
            _global_cuda_manager.cleanup()
            logger.debug("CUDA context cleaned up explicitly")
        except Exception as e:
            logger.debug(f"CUDA cleanup warning (can be ignored): {e}")
        finally:
            _global_cuda_manager = None
    
    # 重置初始化标志，允许下次重新初始化
    # 注意：不重置 _pycuda_initialized，因为 driver.init() 只需调用一次


class CudaContextManager:
    """
    Manages CUDA context for inference.
    
    FIXED VERSION:
    - Uses primary context to share with TensorRT
    - Proper context activation/deactivation
    - Support for CUDA streams
    - Thread-safe operations
    
    Important: TensorRT 8.x+ 推荐使用 CUDA 流而不是显式上下文管理
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize CUDA context manager.
        
        Args:
            device_id: CUDA device ID to use
        """
        if not init_pycuda():
            raise RuntimeError("PyCUDA is not available or failed to initialize")
        
        self.device_id = device_id
        self._context: Optional[Any] = None
        self._device: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._context_pushed = False
        self._lock = threading.Lock()
        
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize CUDA context using primary context."""
        try:
            # Get device
            self._device = cuda.Device(self.device_id)
            
            # 获取主上下文 (与 TensorRT 共享)
            self._context = self._device.retain_primary_context()
            
            # 激活上下文
            self._context.push()
            self._context_pushed = True
            
            # 创建 CUDA 流 (用于异步操作)
            self._stream = cuda.Stream()
            
            # Get device properties
            device_name = self._device.name()
            compute_capability = self._device.compute_capability()
            total_memory = self._device.total_memory() / (1024**3)  # GB
            
            logger.info(
                f"CUDA context initialized on device {self.device_id}: "
                f"{device_name} (compute {compute_capability[0]}.{compute_capability[1]}, "
                f"{total_memory:.2f} GB)"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDA context: {e}")
            raise
    
    def get_context(self) -> Any:
        """
        Get current CUDA context.
        
        Returns:
            CUDA context object
        """
        if self._context is None:
            raise RuntimeError("CUDA context not initialized")
        return self._context
    
    def get_device(self) -> Any:
        """
        Get CUDA device.
        
        Returns:
            CUDA device object
        """
        if self._device is None:
            raise RuntimeError("CUDA device not initialized")
        return self._device
    
    def get_stream(self) -> Any:
        """
        Get CUDA stream.
        
        Returns:
            CUDA stream object
        """
        return self._stream
    
    def ensure_context_active(self):
        """
        确保 CUDA 上下文处于活动状态。
        
        如果当前线程没有活动的上下文，则推入主上下文。
        这是线程安全的操作。
        
        修复：避免重复 push 导致的上下文栈不平衡问题
        """
        with self._lock:
            # 如果已经 push 过了，不要重复 push
            if self._context_pushed:
                return
            
            try:
                current = cuda.Context.get_current()
                if current is None:
                    self._context.push()
                    self._context_pushed = True
            except Exception:
                # 没有当前上下文，推入我们的上下文
                try:
                    self._context.push()
                    self._context_pushed = True
                except Exception as e:
                    # 可能已经 push 过了
                    logger.debug(f"Context push skipped: {e}")
    
    def synchronize(self):
        """Synchronize CUDA device."""
        if self._stream is not None:
            self._stream.synchronize()
        elif self._context is not None:
            self._context.synchronize()
    
    def get_memory_info(self) -> dict:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with 'free' and 'total' memory in bytes
        """
        if self._context is None:
            return {'free': 0, 'total': 0}
        
        try:
            self.ensure_context_active()
            free, total = cuda.mem_get_info()
            return {
                'free': free,
                'total': total,
                'used': total - free,
                'free_gb': free / (1024**3),
                'total_gb': total / (1024**3),
                'used_gb': (total - free) / (1024**3),
                'usage_percent': ((total - free) / total) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {'free': 0, 'total': 0}
    
    def cleanup(self):
        """Clean up CUDA context."""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.synchronize()
                except Exception:
                    pass
                self._stream = None
            
            if self._context is not None and self._context_pushed:
                try:
                    # 尝试多次 pop 以确保栈清空
                    for _ in range(3):  # 最多尝试 3 次
                        try:
                            current = cuda.Context.get_current()
                            if current is not None:
                                current.pop()
                            else:
                                break
                        except Exception:
                            break
                    self._context_pushed = False
                except Exception as e:
                    logger.debug(f"CUDA context pop warning (can be ignored): {e}")
                finally:
                    self._context_pushed = False
                
                # Note: 不销毁主上下文，因为它是共享的
                self._context = None
                
                logger.debug("CUDA context cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        self.ensure_context_active()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # 不在这里清理，让上下文保持活动状态以便 TensorRT 使用
        self.synchronize()
        return False
    
    def __del__(self):
        """Destructor."""
        # 全局管理器不在析构函数中清理，避免程序退出时的问题
        pass


# ============================================================================
# TensorRT 专用上下文管理器
# ============================================================================

class TensorRTContextManager:
    """
    专门为 TensorRT 设计的 CUDA 上下文管理器。
    
    与普通的 CudaContextManager 不同，这个类:
    1. 不使用 pycuda.autoinit
    2. 使用 CUDA 流进行内存传输
    3. 与 TensorRT 的内部上下文管理兼容
    
    使用方法:
    >>> with TensorRTContextManager() as ctx:
    ...     # 在这里执行 TensorRT 推理
    ...     ctx.memcpy_htod(device_ptr, host_array)
    ...     engine.execute(...)
    ...     ctx.memcpy_dtoh(host_array, device_ptr)
    """
    
    def __init__(self, device_id: int = 0):
        """
        初始化 TensorRT 上下文管理器。
        
        Args:
            device_id: CUDA 设备 ID
        """
        self.device_id = device_id
        self._cuda_manager = get_global_cuda_manager(device_id)
        
        if self._cuda_manager is None:
            raise RuntimeError("Failed to get CUDA manager")
    
    @property
    def stream(self) -> Any:
        """获取 CUDA 流"""
        return self._cuda_manager.get_stream()
    
    def mem_alloc(self, size_bytes: int) -> Any:
        """
        分配 GPU 内存。
        
        Args:
            size_bytes: 要分配的字节数
            
        Returns:
            GPU 内存分配对象
        """
        self._cuda_manager.ensure_context_active()
        return cuda.mem_alloc(size_bytes)
    
    def memcpy_htod(self, dest: Any, src) -> None:
        """
        从主机复制到设备 (异步)。
        
        Args:
            dest: 目标设备内存
            src: 源主机数组 (numpy)
        """
        self._cuda_manager.ensure_context_active()
        cuda.memcpy_htod_async(dest, src, self.stream)
    
    def memcpy_dtoh(self, dest, src: Any) -> None:
        """
        从设备复制到主机 (异步)。
        
        Args:
            dest: 目标主机数组 (numpy)
            src: 源设备内存
        """
        self._cuda_manager.ensure_context_active()
        cuda.memcpy_dtoh_async(dest, src, self.stream)
    
    def synchronize(self):
        """同步 CUDA 流"""
        if self.stream is not None:
            self.stream.synchronize()
    
    def __enter__(self):
        """Context manager entry."""
        self._cuda_manager.ensure_context_active()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.synchronize()
        return False


# ============================================================================
# 便捷函数
# ============================================================================

def check_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    return init_pycuda()


def get_cuda_device_info() -> dict:
    """
    Get information about all CUDA devices.
    
    Returns:
        Dictionary with device information
    """
    if not init_pycuda():
        return {'available': False, 'devices': []}
    
    try:
        device_count = cuda.Device.count()
        devices = []
        
        for i in range(device_count):
            device = cuda.Device(i)
            devices.append({
                'id': i,
                'name': device.name(),
                'compute_capability': device.compute_capability(),
                'total_memory_gb': device.total_memory() / (1024**3),
            })
        
        return {
            'available': True,
            'device_count': device_count,
            'devices': devices
        }
        
    except Exception as e:
        logger.error(f"Failed to get CUDA device info: {e}")
        return {'available': False, 'devices': [], 'error': str(e)}


def allocate_cuda_memory(size_bytes: int, device_id: int = 0) -> Any:
    """
    Allocate CUDA device memory.
    
    Args:
        size_bytes: Number of bytes to allocate
        device_id: CUDA device ID
        
    Returns:
        CUDA device memory allocation
    """
    manager = get_global_cuda_manager(device_id)
    if manager is None:
        raise RuntimeError("CUDA not available")
    
    manager.ensure_context_active()
    mem = cuda.mem_alloc(size_bytes)
    logger.debug(f"Allocated {size_bytes / (1024**2):.2f} MB on device {device_id}")
    return mem


def free_cuda_memory(mem: Any):
    """
    Free CUDA device memory.
    
    Args:
        mem: CUDA device memory allocation to free
    """
    try:
        if mem is not None:
            mem.free()
    except Exception as e:
        logger.warning(f"Error freeing CUDA memory: {e}")


# ============================================================================
# 模块初始化
# ============================================================================

# 在模块加载时尝试初始化 (但不创建上下文)
# 这只是检查 PyCUDA 是否可用
try:
    import pycuda.driver as _test_cuda
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False


# ============================================================================
# 测试
# ============================================================================

def _test_cuda_utils():
    """测试 CUDA 工具模块"""
    print("=" * 60)
    print("Testing cuda_utils module (FIXED VERSION)")
    print("=" * 60)
    
    print(f"\n1. PYCUDA_AVAILABLE: {PYCUDA_AVAILABLE}")
    
    print("\n2. Testing init_pycuda()...")
    result = init_pycuda()
    print(f"   Result: {result}")
    
    if result:
        print("\n3. Testing get_cuda_device_info()...")
        info = get_cuda_device_info()
        print(f"   Device count: {info.get('device_count', 0)}")
        for dev in info.get('devices', []):
            print(f"   - {dev['name']}: {dev['total_memory_gb']:.2f} GB")
        
        print("\n4. Testing CudaContextManager...")
        try:
            manager = get_global_cuda_manager()
            if manager:
                print(f"   Context created successfully")
                mem_info = manager.get_memory_info()
                print(f"   GPU Memory: {mem_info.get('used_gb', 0):.2f} / {mem_info.get('total_gb', 0):.2f} GB")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n5. Testing TensorRTContextManager...")
        try:
            with TensorRTContextManager() as ctx:
                print(f"   TensorRT context manager works!")
                # 测试内存分配
                mem = ctx.mem_alloc(1024 * 1024)  # 1 MB
                print(f"   Allocated 1 MB test memory")
                mem.free()
                print(f"   Memory freed successfully")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("cuda_utils tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_cuda_utils()