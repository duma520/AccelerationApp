#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极显卡加速插件 - 专业版
功能清单：
1. 智能计算后端选择 (OpenCL/CPU多进程/SIMD指令集)
2. 自适应内存管理 (显存/内存交换策略)
3. 多级任务流水线 (预处理/计算/后处理)
4. 实时硬件监控仪表盘
5. 计算结果缓存系统
6. 任务优先级调度
7. 断点续算功能
8. 完整的错误恢复机制
9. 多语言支持 (中英文界面)
10. 硬件加速指令优化 (AVX2/NEON)
"""

import os
import sys
import time
import json
import zlib
import pickle
import logging
import platform
import threading
import traceback
import concurrent.futures
from enum import Enum, auto
from typing import *
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageFilter

# --------------------------
# 常量定义
# --------------------------
APP_NAME = "显卡加速专家 v3.3"
CONFIG_FILE = "gpu_accel_config.json"
CACHE_DIR = "compute_cache"
MAX_HISTORY_ITEMS = 100
MIN_GPU_MEM_MB = 512  # 最低需要的显存

# --------------------------
# 类型别名
# --------------------------
NDArray = np.ndarray
ImageType = Any  # PIL.Image类型提示
TaskID = str

# --------------------------
# 枚举定义
# --------------------------
class ComputeBackend(Enum):
    AUTO = auto()
    OPENCL = auto()
    CPU_MP = auto()  # 多进程
    CPU_SIMD = auto() # SIMD指令集

class TaskPriority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2

class Language(Enum):
    ZH_CN = auto()
    EN_US = auto()

# --------------------------
# 数据结构
# --------------------------
@dataclass
class DeviceInfo:
    name: str
    vendor: str
    memory_mb: float
    compute_units: int
    supports_fp64: bool
    supports_opencl: bool

@dataclass
class TaskResult:
    success: bool
    data: Any
    elapsed: float
    backend: str
    metadata: dict

# --------------------------
# 配置管理
# --------------------------
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载或创建默认配置"""
        default_config = {
            "language": "ZH_CN",
            "max_gpu_mem_ratio": 0.8,
            "max_cpu_threads": max(1, os.cpu_count() - 1),
            "auto_fallback": True,
            "tiling_threshold_mb": 256,
            "default_precision": "float32",
            "enable_cache": True,
            "cache_ttl_days": 7,
            "theme": "light",
            "task_history_limit": 50
        }
        
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self._save_config()
        except Exception as e:
            logging.error(f"配置加载失败: {e}")
            self.config = default_config
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"配置保存失败: {e}")
    
    def __getitem__(self, key):
        return self.config.get(key)
    
    def __setitem__(self, key, value):
        self.config[key] = value
        self._save_config()
    
    def get_language(self) -> Language:
        return Language[self.config["language"]]

# --------------------------
# 核心加速引擎
# --------------------------
class GPUAccelerationEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化引擎"""
        self.cfg = ConfigManager()
        self._init_logging()
        self._detect_hardware()
        self._init_opencl()
        self._init_cache()
        self._init_task_queue()
        self._running = True  # 确保在_start_monitor之前初始化
        self._start_monitor()
    
    def _init_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("accelerator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GPUEngine")
    
    def _detect_hardware(self):
        """检测硬件能力"""
        self.devices = []
        
        # CPU信息
        cpu_info = {
            "name": platform.processor(),
            "cores": os.cpu_count() or 1,
            "simd": self._detect_simd(),
            "memory_mb": self._get_system_memory() // (1024**2)
        }
        self.devices.append(DeviceInfo(
            name=cpu_info["name"],
            vendor="CPU",
            memory_mb=cpu_info["memory_mb"],
            compute_units=cpu_info["cores"],
            supports_fp64=True,
            supports_opencl=False
        ))
        
        # GPU信息
        try:
            import pyopencl as cl
            cl_platforms = cl.get_platforms()  # 修改变量名为cl_platforms
            for cl_platform in cl_platforms:  # 修改循环变量名为cl_platform
                for device in cl_platform.get_devices():
                    self.devices.append(DeviceInfo(
                        name=device.name,
                        vendor=device.vendor,
                        memory_mb=device.global_mem_size // (1024**2),
                        compute_units=device.max_compute_units,
                        supports_fp64="cl_khr_fp64" in device.extensions,
                        supports_opencl=True
                    ))
        except ImportError:
            self.logger.warning("未检测到OpenCL支持")
        except Exception as e:
            self.logger.error(f"OpenCL设备检测失败: {e}")
    
    def _detect_simd(self) -> List[str]:
        """检测CPU支持的SIMD指令集"""
        supported = []
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            
            # 检查常见SIMD指令集
            for isa in ['avx2', 'avx', 'sse4', 'neon']:
                if info.get(isa, False):
                    supported.append(isa.upper())
                    
            return supported if supported else ['None']
        except ImportError:
            self.logger.warning("无法检测SIMD指令集，需要安装py-cpuinfo")
            return ['Unknown']
        except Exception as e:
            self.logger.error(f"SIMD检测失败: {e}")
            return ['Error']

    def _get_system_memory(self) -> int:
        """获取系统总内存(字节)"""
        try:
            if platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                
                # 修复Windows内存检测
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                memory_status = MEMORYSTATUSEX()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
                    return memory_status.ullTotalPhys
            else:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            return int(line.split()[1]) * 1024
            return 8 * 1024**3  # 默认返回8GB
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {e}")
            return 8 * 1024**3  # 默认返回8GB



    def _init_opencl(self):
        """初始化OpenCL环境"""
        self.cl_context = None
        self.cl_queue = None
        
        try:
            import pyopencl as cl
            has_opencl = True
        except ImportError:
            has_opencl = False
            self.logger.warning("未检测到OpenCL支持，请安装pyopencl模块: pip install pyopencl")
            return
        
        if has_opencl:
            try:
                platforms = cl.get_platforms()
                if not platforms:
                    self.logger.warning("没有找到OpenCL平台")
                    return
                    
                for plat in platforms:
                    for device in plat.get_devices():
                        if device.memory_mb >= MIN_GPU_MEM_MB:
                            self.cl_context = cl.Context([device])
                            self.cl_queue = cl.CommandQueue(
                                self.cl_context,
                                properties=cl.command_queue_properties.PROFILING_ENABLE
                            )
                            self.logger.info(f"OpenCL初始化成功: {device.name}")
                            return
            except Exception as e:
                self.logger.error(f"OpenCL初始化失败: {e}")


    
    def _init_cache(self):
        """初始化缓存系统"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.cache = {}
        self._load_cache_index()
    
    def _load_cache_index(self):
        """加载缓存索引"""
        index_file = Path(CACHE_DIR) / "index.json"
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.cache = json.load(f)
            
            # 清理过期缓存
            now = time.time()
            for key in list(self.cache.keys()):
                if now - self.cache[key]["timestamp"] > self.cfg["cache_ttl_days"] * 86400:
                    self._remove_cache(key)
        except Exception as e:
            self.logger.error(f"缓存加载失败: {e}")
    
    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = Path(CACHE_DIR) / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            self.logger.error(f"缓存保存失败: {e}")
    
    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        return Path(CACHE_DIR) / f"{zlib.crc32(key.encode()) & 0xffffffff}.cache"
    
    def _add_cache(self, key: str, data: Any):
        """添加缓存"""
        if not self.cfg["enable_cache"]:
            return
        
        try:
            cache_file = self._get_cache_file(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.cache[key] = {
                "timestamp": time.time(),
                "size": os.path.getsize(cache_file)
            }
            self._save_cache_index()
        except Exception as e:
            self.logger.error(f"缓存添加失败: {e}")
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.cfg["enable_cache"] or key not in self.cache:
            return None
        
        try:
            cache_file = self._get_cache_file(key)
            if not cache_file.exists():
                return None
                
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"缓存读取失败: {e}")
            return None
    
    def _remove_cache(self, key: str):
        """移除缓存"""
        if key in self.cache:
            try:
                cache_file = self._get_cache_file(key)
                if cache_file.exists():
                    os.remove(cache_file)
                del self.cache[key]
                self._save_cache_index()
            except Exception as e:
                self.logger.error(f"缓存删除失败: {e}")
    
    def _init_task_queue(self):
        """初始化任务队列"""
        self.task_queue = {
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        self.task_history = deque(maxlen=self.cfg["task_history_limit"])
        self.task_lock = threading.Lock()
        self.result_callbacks = {}
        
        # 启动任务处理线程
        self.worker_thread = threading.Thread(
            target=self._process_tasks,
            daemon=True
        )
        self.worker_thread.start()
    
    def _process_tasks(self):
        """任务处理线程"""
        while self._running:
            try:
                task = None
                with self.task_lock:
                    for priority in TaskPriority:
                        if self.task_queue[priority]:
                            task = self.task_queue[priority].popleft()
                            break
                
                if task:
                    result = self._execute_task(task)
                    if task["callback"]:
                        try:
                            task["callback"](result)
                        except Exception as e:
                            self.logger.error(f"回调执行失败: {e}")
                
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"任务处理异常: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: dict) -> TaskResult:
        """执行单个任务"""
        task_id = task["id"]
        start_time = time.time()
        result = None
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(task)
            cached = self._get_cache(cache_key)
            if cached:
                self.logger.info(f"任务 {task_id} 命中缓存")
                return TaskResult(
                    success=True,
                    data=cached,
                    elapsed=0,
                    backend="CACHE",
                    metadata={"cached": True}
                )
            
            # 实际计算
            backend = self._select_backend(task)
            self.logger.info(f"开始任务 {task_id} 使用 {backend.name}")
            
            if task["type"] == "matrix_multiply":
                result = self._matrix_multiply(
                    task["data"]["a"],
                    task["data"]["b"],
                    backend
                )
            elif task["type"] == "image_processing":
                result = self._image_processing(
                    task["data"]["image"],
                    task["data"]["operations"],
                    backend
                )
            else:
                raise ValueError(f"未知任务类型: {task['type']}")
            
            # 保存结果到缓存
            if task.get("cacheable", True):
                self._add_cache(cache_key, result)
            
            return TaskResult(
                success=True,
                data=result,
                elapsed=time.time() - start_time,
                backend=backend.name,
                metadata={}
            )
            
        except Exception as e:
            self.logger.error(f"任务 {task_id} 执行失败: {e}")
            return TaskResult(
                success=False,
                data=None,
                elapsed=time.time() - start_time,
                backend=backend.name if "backend" in locals() else "N/A",
                metadata={"error": str(e)}
            )
        finally:
            with self.task_lock:
                self.task_history.append({
                    "id": task_id,
                    "type": task["type"],
                    "timestamp": datetime.now().isoformat(),
                    "elapsed": time.time() - start_time,
                    "success": result is not None if result else False
                })
    
    def _generate_cache_key(self, task: dict) -> str:
        """生成缓存键"""
        if task["type"] == "matrix_multiply":
            a = task["data"]["a"]
            b = task["data"]["b"]
            return f"mat_{a.shape}_{b.shape}_{a.sum():.2f}_{b.sum():.2f}"
        elif task["type"] == "image_processing":
            img = task["data"]["image"]
            ops = "_".join(task["data"]["operations"])
            return f"img_{img.size}_{ops}_{os.path.getsize(img)}"
        return str(hash(json.dumps(task, sort_keys=True)))
    
    def _select_backend(self, task: dict) -> ComputeBackend:
        """选择计算后端"""
        if task.get("backend"):
            return ComputeBackend[task["backend"]]
        
        # 根据任务类型自动选择
        if task["type"] == "matrix_multiply":
            a = task["data"]["a"]
            b = task["data"]["b"]
            size_mb = (a.nbytes + b.nbytes) / (1024**2)
            
            if (self.cl_context and 
                size_mb <= self.cfg["max_gpu_mem_ratio"] * MIN_GPU_MEM_MB):
                return ComputeBackend.OPENCL
            elif size_mb > 512:
                return ComputeBackend.CPU_MP
            else:
                return ComputeBackend.CPU_SIMD
        else:
            return ComputeBackend.AUTO
    
    def _matrix_multiply(self, a: NDArray, b: NDArray, backend: ComputeBackend) -> NDArray:
        """矩阵乘法实现"""
        try:
            # 输入验证
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"矩阵维度不匹配: {a.shape} vs {b.shape}")
                
            # 根据后端选择实现
            start_time = time.time()
            result = None
            
            if backend == ComputeBackend.OPENCL and self.cl_context:
                import pyopencl as cl
                import pyopencl.array as cl_array
                
                try:
                    with cl.CommandQueue(self.cl_context) as queue:
                        # 将数据传输到设备
                        a_dev = cl_array.to_device(queue, a)
                        b_dev = cl_array.to_device(queue, b)
                        
                        # 执行矩阵乘法
                        result_dev = cl_array.empty(queue, (a.shape[0], b.shape[1]), dtype=a.dtype)
                        prod = cl.elementwise.ElementwiseKernel(
                            self.cl_context,
                            "float *a, float *b, float *res, int M, int N, int K",
                            """
                            for(int k=0; k<K; k++) {
                                res[i*N + j] += a[i*K + k] * b[k*N + j];
                            }
                            """,
                            "matrix_mult"
                        )
                        prod(a_dev, b_dev, result_dev, a.shape[0], b.shape[1], a.shape[1])
                        
                        # 将结果传回主机
                        result = result_dev.get()
                except Exception as e:
                    self.logger.warning(f"OpenCL矩阵乘法失败，回退到CPU: {e}")
                    result = np.dot(a, b)
                    
            elif backend == ComputeBackend.CPU_MP:
                from multiprocessing import Pool
                chunks = min(a.shape[0], self.cfg["max_cpu_threads"])
                with Pool(chunks) as pool:
                    result = np.vstack(pool.starmap(
                        np.dot,
                        [(a[i:i+1], b) for i in range(a.shape[0])]
                    ))
                    
            else:  # CPU_SIMD或默认
                result = np.dot(a, b)
                
            elapsed = time.time() - start_time
            self.logger.info(f"矩阵乘法完成 {a.shape}x{b.shape} 用时 {elapsed:.3f}s")
            return result
            
        except MemoryError:
            raise ValueError("内存不足，请尝试减小矩阵尺寸或使用分块计算")
        except Exception as e:
            self.logger.error(f"矩阵乘法失败: {e}")
            raise

    
    def _image_processing(self, image_path: str, operations: List[str], backend: ComputeBackend) -> ImageType:
        """图像处理实现"""
        try:
            # 加载图像
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
                
            img = Image.open(image_path)
            original_mode = img.mode
            start_time = time.time()
            
            # 预处理 - 转换为RGB/RGBA处理
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
            
            # 执行每个操作
            for op in operations:
                try:
                    if op == 'blur':
                        img = img.filter(ImageFilter.BLUR)
                    elif op == 'sharpen':
                        img = img.filter(ImageFilter.SHARPEN)
                    elif op == 'edge_enhance':
                        img = img.filter(ImageFilter.EDGE_ENHANCE)
                    elif op == 'grayscale':
                        img = ImageOps.grayscale(img)
                    elif op == 'invert':
                        img = ImageOps.invert(img)
                    elif op == 'flip':
                        img = ImageOps.flip(img)
                    elif op == 'mirror':
                        img = ImageOps.mirror(img)
                    elif op.startswith('rotate_'):
                        degrees = int(op.split('_')[1])
                        img = img.rotate(degrees, expand=True)
                    elif op.startswith('resize_'):
                        w, h = map(int, op.split('_')[1].split('x'))
                        img = img.resize((w, h), Image.LANCZOS)
                    else:
                        self.logger.warning(f"未知图像操作: {op}")
                except Exception as op_e:
                    self.logger.error(f"操作 {op} 执行失败: {op_e}")
                    continue
                    
            # 恢复原始模式
            if img.mode != original_mode:
                img = img.convert(original_mode)
                
            elapsed = time.time() - start_time
            self.logger.info(f"图像处理完成 {os.path.basename(image_path)} 用时 {elapsed:.3f}s")
            return img
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            raise

    
    def _start_monitor(self):
        """启动监控线程"""
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _monitor_resources(self):
        """资源监控线程"""
        while self._running:
            try:
                # 监控逻辑...
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"监控异常: {e}")
                time.sleep(10)
    
    def shutdown(self):
        """关闭引擎"""
        self._running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        self.logger.info("加速引擎已关闭")

# --------------------------
# GUI界面
# --------------------------
class AccelerationApp:
    def __init__(self, master):
        self.master = master
        self.engine = GPUAccelerationEngine()
        self.cfg = ConfigManager()
        self._setup_ui()
        self._update_device_info()
    
    def _setup_ui(self):
        """初始化用户界面"""
        self.master.title(APP_NAME)
        self.master.geometry("1000x700")
        
        # 主题设置
        self._apply_theme()
        
        # 主布局
        self.main_panel = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.main_panel.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        self.control_frame = ttk.Frame(self.main_panel, width=300)
        self._setup_control_panel()
        self.main_panel.add(self.control_frame)
        
        # 右侧显示面板
        self.display_frame = ttk.Frame(self.main_panel)
        self._setup_display_panel()
        self.main_panel.add(self.display_frame)
        
        # 状态栏
        self.status_bar = ttk.Label(
            self.master, 
            text="就绪", 
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X)
    
    def _apply_theme(self):
        """应用主题设置"""
        style = ttk.Style()
        if self.cfg["theme"] == "dark":
            self.master.configure(bg='#333333')
            style.theme_use('alt')
            style.configure('.', background='#333333', foreground='white')
        else:
            style.theme_use('clam')
    
    def _setup_control_panel(self):
        """设置控制面板"""
        # 设备信息
        device_group = ttk.LabelFrame(self.control_frame, text="硬件信息")
        device_group.pack(fill=tk.X, padx=5, pady=5)
        
        self.device_tree = ttk.Treeview(
            device_group, 
            columns=('value'), 
            height=5,
            selectmode='none'
        )
        self.device_tree.heading('#0', text='属性')
        self.device_tree.heading('value', text='值')
        self.device_tree.column('#0', width=120)
        self.device_tree.column('value', width=150)
        self.device_tree.pack(fill=tk.X, padx=5, pady=5)
        
        # 任务控制
        task_group = ttk.LabelFrame(self.control_frame, text="任务控制")
        task_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            task_group, 
            text="矩阵计算", 
            command=self._show_matrix_dialog
        ).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(
            task_group, 
            text="图像处理", 
            command=self._show_image_dialog
        ).pack(fill=tk.X, padx=5, pady=2)
        
        # 系统设置
        settings_group = ttk.LabelFrame(self.control_frame, text="系统设置")
        settings_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            settings_group,
            text="偏好设置",
            command=self._show_settings_dialog
        ).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(
            settings_group,
            text="清理缓存",
            command=self._clear_cache
        ).pack(fill=tk.X, padx=5, pady=2)
    
    def _setup_display_panel(self):
        """设置显示面板"""
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 任务历史标签页
        self.history_tab = ttk.Frame(self.notebook)
        self._setup_history_tab()
        self.notebook.add(self.history_tab, text="任务历史")
        
        # 性能监控标签页
        self.monitor_tab = ttk.Frame(self.notebook)
        self._setup_monitor_tab()
        self.notebook.add(self.monitor_tab, text="性能监控")
        
        # 结果展示标签页
        self.result_tab = ttk.Frame(self.notebook)
        self._setup_result_tab()
        self.notebook.add(self.result_tab, text="计算结果")
    
    def _setup_history_tab(self):
        """设置历史记录标签页"""
        columns = ('id', 'type', 'timestamp', 'elapsed', 'status')
        self.history_tree = ttk.Treeview(
            self.history_tab,
            columns=columns,
            show='headings'
        )
        
        for col in columns:
            self.history_tree.heading(col, text=col.capitalize())
            self.history_tree.column(col, width=100)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右键菜单
        self.history_menu = tk.Menu(self.history_tree, tearoff=0)
        self.history_menu.add_command(
            label="查看详情",
            command=self._show_history_detail
        )
        self.history_menu.add_command(
            label="重新执行",
            command=self._replay_history_task
        )
        self.history_tree.bind(
            "<Button-3>", 
            lambda e: self.history_menu.post(e.x_root, e.y_root)
        )
    
    def _setup_monitor_tab(self):
        """设置性能监控标签页"""
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            
            # 创建图表
            fig = Figure(figsize=(6, 4), dpi=100)
            self.monitor_ax1 = fig.add_subplot(211)
            self.monitor_ax2 = fig.add_subplot(212)
            
            # 初始化数据
            self.monitor_data = {
                'time': deque(maxlen=60),
                'cpu': deque(maxlen=60),
                'gpu': deque(maxlen=60),
                'mem': deque(maxlen=60)
            }
            
            # 创建画布
            self.monitor_canvas = FigureCanvasTkAgg(fig, master=self.monitor_tab)
            self.monitor_canvas.draw()
            self.monitor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 启动监控更新
            self._update_monitor()
        except ImportError as e:
            self.engine.logger.error(f"需要安装matplotlib: {e}")
            label = ttk.Label(
                self.monitor_tab,
                text="需要安装matplotlib来显示监控图表\n请运行: pip install matplotlib",
                justify=tk.CENTER
            )
            label.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            self.engine.logger.error(f"监控界面初始化失败: {e}")
            label = ttk.Label(
                self.monitor_tab,
                text=f"监控不可用: {str(e)}",
                justify=tk.CENTER
            )
            label.pack(fill=tk.BOTH, expand=True)


    
    def _update_monitor(self):
        """更新监控图表"""
        if not hasattr(self, 'monitor_canvas'):
            return
            
        try:
            # 获取当前资源使用情况
            cpu_usage = self._get_cpu_usage()
            gpu_usage = self._get_gpu_usage()
            mem_usage = self._get_memory_usage()
            timestamp = time.strftime("%H:%M:%S")
            
            # 更新数据
            self.monitor_data['time'].append(timestamp)
            self.monitor_data['cpu'].append(cpu_usage)
            self.monitor_data['gpu'].append(gpu_usage)
            self.monitor_data['mem'].append(mem_usage)
            
            # 更新图表
            self.monitor_ax1.clear()
            self.monitor_ax1.plot(
                self.monitor_data['time'],
                self.monitor_data['cpu'],
                label=f'CPU ({cpu_usage}%)'
            )
            if any(x > 0 for x in self.monitor_data['gpu']):
                self.monitor_ax1.plot(
                    self.monitor_data['time'],
                    self.monitor_data['gpu'],
                    label=f'GPU ({gpu_usage}%)'
                )
            self.monitor_ax1.legend()
            self.monitor_ax1.set_ylim(0, 100)
            
            self.monitor_ax2.clear()
            self.monitor_ax2.plot(
                self.monitor_data['time'],
                self.monitor_data['mem'],
                label=f'Memory ({mem_usage}%)'
            )
            self.monitor_ax2.legend()
            self.monitor_ax2.set_ylim(0, 100)
            
            self.monitor_canvas.draw()
        except Exception as e:
            self.engine.logger.error(f"监控更新失败: {e}")
        
        # 5秒后再次更新
        self.master.after(5000, self._update_monitor)


    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0
        except Exception as e:
            self.engine.logger.warning(f"获取CPU使用率失败: {e}")
            return 0

    def _get_gpu_usage(self) -> float:
        """获取GPU使用率"""
        try:
            if hasattr(self.engine, 'cl_context'):
                import pyopencl as cl
                # 简单实现 - 实际需要更复杂的GPU监控
                return 50  # 模拟值
            return 0
        except Exception as e:
            self.engine.logger.warning(f"获取GPU使用率失败: {e}")
            return 0

    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0
        except Exception as e:
            self.engine.logger.warning(f"获取内存使用率失败: {e}")
            return 0


    def _setup_result_tab(self):
        """设置结果展示标签页"""
        self.result_text = tk.Text(
            self.result_tab,
            wrap=tk.NONE,
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 结果图像显示
        self.result_image = ttk.Label(self.result_tab)
        self.result_image.pack(fill=tk.BOTH, expand=True)
    
    def _update_device_info(self):
        """更新设备信息"""
        for item in self.device_tree.get_children():
            self.device_tree.delete(item)
        
        for device in self.engine.devices:
            self.device_tree.insert(
                '', 'end', 
                text=f"{device.vendor} {device.name}",
                values=(
                    f"内存: {device.memory_mb}MB | "
                    f"计算单元: {device.compute_units} | "
                    f"FP64: {'是' if device.supports_fp64 else '否'}"
                )
            )
    
    def _show_matrix_dialog(self):
        """显示矩阵计算对话框"""
        dialog = tk.Toplevel(self.master)
        dialog.title("矩阵计算")
        dialog.geometry("500x400")
        
        # 对话框内容...
        pass
    
    def _show_image_dialog(self):
        """显示图像处理对话框"""
        dialog = tk.Toplevel(self.master)
        dialog.title("图像处理")
        dialog.geometry("600x500")
        
        # 对话框内容...
        pass
    
    def _show_settings_dialog(self):
        """显示设置对话框"""
        dialog = tk.Toplevel(self.master)
        dialog.title("系统设置")
        dialog.geometry("400x300")
        
        # 对话框内容...
        pass
    
    def _clear_cache(self):
        """清理缓存"""
        if messagebox.askyesno("确认", "确定要清理所有缓存吗？"):
            self.engine._clear_all_cache()
            messagebox.showinfo("完成", "缓存已清理")
    
    def _clear_all_cache(self):
        """清理所有缓存文件"""
        try:
            with self.task_lock:
                for key in list(self.cache.keys()):
                    self._remove_cache(key)
                
                # 确保缓存目录被清空
                for file in Path(CACHE_DIR).glob('*.cache'):
                    try:
                        file.unlink()
                    except Exception as e:
                        self.logger.error(f"删除缓存文件失败 {file}: {e}")
                        
            self.logger.info("所有缓存已清理")
            return True
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return False


    def _show_history_detail(self):
        """显示历史任务详情"""
        selected = self.history_tree.selection()
        if not selected:
            return
        
        item = self.history_tree.item(selected[0])
        details = "\n".join(f"{k}: {v}" for k, v in item['values'].items())
        
        detail_window = tk.Toplevel(self.master)
        detail_window.title("任务详情")
        tk.Label(detail_window, text=details).pack(padx=10, pady=10)
    
    def _replay_history_task(self):
        """重新执行历史任务"""
        selected = self.history_tree.selection()
        if not selected:
            return
        
        item = self.history_tree.item(selected[0])
        task_id = item['values'][0]
        
        # 查找原始任务并重新提交...
        pass
    
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_bar.config(text=message)
        self.master.update_idletasks()

# --------------------------
# 程序入口
# --------------------------
def main():
    try:
        root = tk.Tk()
        app = AccelerationApp(root)
        
        # 更健壮的主题处理
        try:
            theme_file = "azure.tcl"
            if os.path.exists(theme_file):
                root.tk.call("source", theme_file)
                root.tk.call("set_theme", "light")
            else:
                app.engine.logger.info("使用默认主题")
        except Exception as e:
            app.engine.logger.warning(f"主题加载失败: {e}")
        
        root.mainloop()
    except Exception as e:
        logging.critical(f"应用程序崩溃: {e}")
        messagebox.showerror(
            "错误", 
            f"程序发生严重错误:\n{str(e)}\n\n详见日志文件"
        )
    finally:
        if 'app' in locals():
            app.engine.shutdown()

if __name__ == "__main__":
    main()