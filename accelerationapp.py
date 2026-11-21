#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整版GPU加速插件 - PySide6升级版
功能清单：
1. 多后端计算 (OpenCL/CPU多进程/SIMD)
2. 智能内存管理
3. 任务队列系统
4. 计算结果缓存
5. 实时性能监控
6. 现代化PySide6 GUI界面
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
import numpy as np
from enum import Enum, auto
from typing import *
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageFilter

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QTabWidget,
    QTextEdit, QDialog, QDialogButtonBox, QMessageBox, QFileDialog,
    QCheckBox, QLineEdit, QGroupBox, QSplitter, QStatusBar,
    QProgressBar, QScrollArea, QGridLayout
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QPoint, QRect, QMutex, QWaitCondition
)
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QPalette, QColor, QLinearGradient,
    QPainter, QBrush, QPen, QIcon
)

# === 常量定义 ===
APP_NAME = "GPU加速专家"
CONFIG_FILE = "gpu_config.json"
CACHE_DIR = "gpu_cache"
LOG_FILE = "gpu_accelerator.log"
MIN_GPU_MEM = 512  # MB
VERSION = "2.0.0"

# === 类型定义 ===
class ComputeBackend(Enum):
    AUTO = "自动选择"
    OPENCL = "OpenCL GPU"
    CPU_MP = "CPU多进程"
    CPU_SIMD = "CPU SIMD"
    CACHE = "缓存读取"

class TaskPriority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2

@dataclass
class DeviceInfo:
    name: str
    type: str
    memory: float  # MB
    compute_units: int
    supports_fp64: bool
    vendor: str = "未知"

@dataclass
class TaskResult:
    success: bool
    data: Any
    elapsed: float
    backend: str
    message: str
    task_id: str = ""
    timestamp: float = 0.0

# === 工作线程 ===
class TaskWorker(QThread):
    task_completed = Signal(str, object)  # task_id, result
    task_progress = Signal(str, int)      # task_id, progress
    task_error = Signal(str, str)         # task_id, error_msg

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def run(self):
        while self.running:
            try:
                task = None
                with QMutexLocker(self.mutex):
                    for priority in TaskPriority:
                        if self.engine.task_queue[priority]:
                            task = self.engine.task_queue[priority].popleft()
                            break
                
                if task:
                    self.task_progress.emit(task["id"], 30)
                    result = self.engine._execute_task(task)
                    self.task_progress.emit(task["id"], 100)
                    
                    if result.success:
                        self.task_completed.emit(task["id"], result)
                    else:
                        self.task_error.emit(task["id"], result.message)
                    
                    if task.get("callback"):
                        try:
                            task["callback"](result)
                        except Exception as e:
                            self.engine.logger.error(f"回调执行失败: {e}")
                else:
                    self.msleep(100)
            except Exception as e:
                self.engine.logger.error(f"任务处理异常: {e}")
                self.msleep(1000)

    def stop(self):
        self.running = False
        self.condition.wakeAll()
        self.wait()

class QMutexLocker:
    def __init__(self, mutex):
        self.mutex = mutex
        self.mutex.lock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.mutex.unlock()

# === 核心引擎 ===
class GPUAccelerator:
    def __init__(self):
        self._init_config()
        self._init_logging()
        self._detect_hardware()
        self._init_opencl()
        self._init_cache()
        self._init_task_system()
        self.running = True

    def _init_config(self):
        """加载配置文件"""
        self.config = {
            "max_gpu_mem": 0.8,
            "max_cpu_threads": max(1, os.cpu_count() - 1),
            "enable_cache": True,
            "cache_ttl": 7,  # days
            "task_history": 100,
            "theme": "light",  # 默认改为亮色主题
            "language": "zh_CN"
        }
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.config.update(json.load(f))
        except Exception as e:
            logging.error(f"配置加载失败: {e}")

    def _init_logging(self):
        """初始化日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GPUAccelerator")

    def _get_system_memory(self):
        """获取系统内存"""
        try:
            import psutil
            return psutil.virtual_memory().total
        except:
            return 8 * 1024**3  # 默认8GB

    def _detect_hardware(self):
        """检测硬件信息"""
        self.devices = []
        
        # CPU信息
        cpu_cores = os.cpu_count() or 1
        try:
            cpu_name = platform.processor()
        except:
            cpu_name = f"CPU ({cpu_cores}核)"
        
        self.devices.append(DeviceInfo(
            name=cpu_name,
            type="CPU",
            memory=self._get_system_memory() // (1024**2),
            compute_units=cpu_cores,
            supports_fp64=True,
            vendor="Intel/AMD"
        ))
        
        # GPU信息
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                for device in platform.get_devices():
                    if device.type == cl.device_type.GPU:
                        self.devices.append(DeviceInfo(
                            name=device.name.strip(),
                            type="GPU",
                            memory=device.global_mem_size // (1024**2),
                            compute_units=device.max_compute_units,
                            supports_fp64="cl_khr_fp64" in device.extensions,
                            vendor=device.vendor.strip() if hasattr(device, 'vendor') else "未知"
                        ))
        except Exception as e:
            self.logger.warning(f"OpenCL检测失败: {e}")

    def _init_opencl(self):
        """初始化OpenCL环境"""
        self.cl_ctx = None
        self.cl_queue = None
        self.cl_available = False
        
        for device in self.devices:
            if device.type == "GPU" and device.memory >= MIN_GPU_MEM:
                try:
                    import pyopencl as cl
                    platforms = cl.get_platforms()
                    for platform in platforms:
                        for dev in platform.get_devices():
                            if dev.type == cl.device_type.GPU:
                                self.cl_ctx = cl.Context([dev])
                                self.cl_queue = cl.CommandQueue(
                                    self.cl_ctx, 
                                    properties=cl.command_queue_properties.PROFILING_ENABLE
                                )
                                self.cl_available = True
                                self.logger.info(f"OpenCL初始化成功: {device.name}")
                                break
                        if self.cl_available:
                            break
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
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            
            # 清理过期缓存
            now = time.time()
            for key in list(self.cache.keys()):
                if now - self.cache[key]["time"] > self.config["cache_ttl"] * 86400:
                    self._remove_cache(key)
        except Exception as e:
            self.logger.error(f"缓存加载失败: {e}")

    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = Path(CACHE_DIR) / "index.json"
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"缓存保存失败: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return Path(CACHE_DIR) / f"{zlib.crc32(key.encode()) & 0xffffffff}.dat"

    def add_cache(self, key: str, data: Any):
        """添加缓存"""
        if not self.config["enable_cache"]:
            return
        
        try:
            cache_file = self._get_cache_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.cache[key] = {
                "time": time.time(),
                "size": os.path.getsize(cache_file),
                "hits": 0
            }
            self._save_cache_index()
        except Exception as e:
            self.logger.error(f"缓存添加失败: {e}")

    def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.config["enable_cache"] or key not in self.cache:
            return None
        
        try:
            cache_file = self._get_cache_path(key)
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.cache[key]["hits"] = self.cache[key].get("hits", 0) + 1
                self.cache[key]["time"] = time.time()  # 更新访问时间
                self._save_cache_index()
                return data
        except Exception as e:
            self.logger.error(f"缓存读取失败: {e}")
            return None

    def _remove_cache(self, key: str):
        """移除缓存"""
        if key in self.cache:
            try:
                cache_file = self._get_cache_path(key)
                if cache_file.exists():
                    os.remove(cache_file)
                del self.cache[key]
                self._save_cache_index()
            except Exception as e:
                self.logger.error(f"缓存删除失败: {e}")

    def _init_task_system(self):
        """初始化任务系统"""
        self.task_queue = {
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        self.task_history = deque(maxlen=self.config["task_history"])
        self.task_lock = threading.Lock()
        self.results = {}
        
        # 创建工作线程
        self.worker = TaskWorker(self)
        self.worker.task_completed.connect(self._on_task_completed)
        self.worker.task_error.connect(self._on_task_error)
        self.worker.start()

    def _on_task_completed(self, task_id: str, result: TaskResult):
        """任务完成回调"""
        self.results[task_id] = result
        self.logger.info(f"任务 {task_id} 完成，耗时: {result.elapsed:.2f}s")

    def _on_task_error(self, task_id: str, error_msg: str):
        """任务错误回调"""
        self.logger.error(f"任务 {task_id} 失败: {error_msg}")

    def _execute_task(self, task: dict) -> TaskResult:
        """执行单个任务"""
        task_id = task["id"]
        start_time = time.time()
        backend_used = ComputeBackend.AUTO
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(task)
            cached = self.get_cache(cache_key)
            if cached is not None:
                return TaskResult(
                    success=True,
                    data=cached,
                    elapsed=0,
                    backend=ComputeBackend.CACHE.value,
                    message="从缓存加载",
                    task_id=task_id,
                    timestamp=time.time()
                )
            
            # 执行计算
            backend = self._select_backend(task)
            backend_used = backend
            self.logger.info(f"执行任务 {task_id} 使用 {backend.value}")
            
            if task["type"] == "matrix_mult":
                result = self._matrix_multiply(
                    task["a"], task["b"], backend
                )
            elif task["type"] == "image_process":
                result = self._process_image(
                    task["image"], task["ops"], backend
                )
            else:
                raise ValueError(f"未知任务类型: {task['type']}")
            
            # 保存结果
            if task.get("cacheable", True):
                self.add_cache(cache_key, result)
            
            return TaskResult(
                success=True,
                data=result,
                elapsed=time.time() - start_time,
                backend=backend.value,
                message="计算完成",
                task_id=task_id,
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"任务失败: {task_id} - {str(e)}")
            return TaskResult(
                success=False,
                data=None,
                elapsed=time.time() - start_time,
                backend=backend_used.value,
                message=str(e),
                task_id=task_id,
                timestamp=time.time()
            )
        finally:
            with self.task_lock:
                self.task_history.append({
                    "id": task_id,
                    "type": task["type"],
                    "time": datetime.now().isoformat(),
                    "elapsed": time.time() - start_time,
                    "status": "success" if "result" in locals() else "failed"
                })

    def _generate_cache_key(self, task: dict) -> str:
        """生成缓存键"""
        try:
            if task["type"] == "matrix_mult":
                a, b = task["a"], task["b"]
                return f"mat_{a.shape}_{b.shape}_{hash(a.tobytes())}_{hash(b.tobytes())}"
            elif task["type"] == "image_process":
                img_path = task["image"]
                img_stat = os.path.getsize(img_path) if os.path.exists(img_path) else 0
                return f"img_{img_path}_{img_stat}_{'_'.join(task['ops'])}"
        except:
            pass
        return f"task_{hash(json.dumps(task, default=str, sort_keys=True))}"

    def _select_backend(self, task: dict) -> ComputeBackend:
        """选择计算后端"""
        if task.get("backend"):
            return ComputeBackend[task["backend"]]
        
        # 自动选择逻辑
        if task["type"] == "matrix_mult":
            a, b = task["a"], task["b"]
            size_mb = (a.nbytes + b.nbytes) / (1024**2)
            
            if self.cl_available and size_mb <= MIN_GPU_MEM * self.config["max_gpu_mem"]:
                return ComputeBackend.OPENCL
            elif size_mb > 512:
                return ComputeBackend.CPU_MP
            else:
                return ComputeBackend.CPU_SIMD
        else:
            return ComputeBackend.AUTO

    def _matrix_multiply(self, a: np.ndarray, b: np.ndarray, backend: ComputeBackend) -> np.ndarray:
        """矩阵乘法实现"""
        if backend == ComputeBackend.OPENCL:
            return self._ocl_matrix_mult(a, b)
        elif backend == ComputeBackend.CPU_MP:
            return self._mp_matrix_mult(a, b)
        else:
            return np.dot(a, b)

    def _ocl_matrix_mult(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """OpenCL矩阵乘法"""
        import pyopencl as cl
        import pyopencl.array as cl_array
        
        # 准备数据
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
        
        # 创建缓冲区
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, result.nbytes)
        
        # 编译内核（优化版）
        kernel = """
        __kernel void matmul(
            __global const float* a,
            __global const float* b,
            __global float* c,
            int a_rows, int a_cols,
            int b_cols)
        {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            float sum = 0.0;
            int a_idx = row * a_cols;
            int b_idx = col;
            
            for(int k = 0; k < a_cols; k++) {
                sum += a[a_idx + k] * b[k * b_cols + col];
            }
            c[row * b_cols + col] = sum;
        }
        """
        
        try:
            prg = cl.Program(self.cl_ctx, kernel).build()
            
            # 执行内核
            event = prg.matmul(
                self.cl_queue,
                (a.shape[0], b.shape[1]),
                None,
                a_buf, b_buf, c_buf,
                np.int32(a.shape[0]), np.int32(a.shape[1]),
                np.int32(b.shape[1])
            )
            event.wait()
            
            # 获取结果
            cl.enqueue_copy(self.cl_queue, result, c_buf).wait()
            
        except Exception as e:
            self.logger.error(f"OpenCL执行失败，回退到CPU: {e}")
            result = np.dot(a, b)
        
        return result

    def _mp_matrix_mult(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """多进程矩阵乘法"""
        def worker(args):
            i, a_slice = args
            return (i, np.dot(a_slice, b))
        
        # 分块处理
        chunk_size = max(1, a.shape[0] // (self.config["max_cpu_threads"] * 2))
        chunks = [(i, a[i:i+chunk_size]) for i in range(0, a.shape[0], chunk_size)]
        
        result = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
        
        with ThreadPoolExecutor(max_workers=self.config["max_cpu_threads"]) as executor:
            futures = [executor.submit(worker, chunk) for chunk in chunks]
            for future in futures:
                i, res = future.result()
                result[i:i+res.shape[0]] = res
        
        return result

    def _process_image(self, image_path: str, operations: List[str], backend: ComputeBackend) -> Image.Image:
        """图像处理"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 执行操作链
                for op in operations:
                    if op == "grayscale":
                        img = ImageOps.grayscale(img)
                    elif op == "edge":
                        img = img.filter(ImageFilter.FIND_EDGES)
                    elif op == "blur":
                        img = img.filter(ImageFilter.GaussianBlur(radius=2))
                    elif op == "rotate":
                        img = img.rotate(90, expand=True)
                    elif op == "flip":
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif op == "invert":
                        if img.mode == 'RGB':
                            img = ImageOps.invert(img)
                        else:
                            img = ImageOps.invert(img.convert('RGB'))
                    elif op == "enhance":
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.5)
                
                return img.copy()
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            raise

    def submit_task(self, task: dict, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """提交任务"""
        task_id = f"task_{int(time.time()*1000)}_{hash(json.dumps(task, default=str))}"
        
        with self.task_lock:
            self.task_queue[priority].append({
                "id": task_id,
                **task
            })
        
        return task_id

    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """获取任务结果"""
        start = time.time()
        while timeout is None or (time.time() - start) < timeout:
            if task_id in self.results:
                return self.results[task_id]
            time.sleep(0.1)
        return None

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "tasks_pending": sum(len(q) for q in self.task_queue.values()),
            "tasks_completed": len(self.results),
            "cache_size": len(self.cache),
            "cache_hits": sum(v.get("hits", 0) for v in self.cache.values()),
            "devices": len(self.devices)
        }

    def shutdown(self):
        """关闭引擎"""
        self.running = False
        if hasattr(self, 'worker'):
            self.worker.stop()
        self.logger.info("加速器已关闭")

# === PySide6 GUI界面 ===
class MatrixDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("矩阵计算")
        self.setModal(True)
        self.resize(400, 300)
        self.result = None
        
        layout = QVBoxLayout(self)
        
        # 输入组
        input_group = QGroupBox("矩阵维度")
        input_layout = QGridLayout(input_group)
        
        input_layout.addWidget(QLabel("矩阵A 行数:"), 0, 0)
        self.a_rows = QLineEdit()
        self.a_rows.setText("512")
        input_layout.addWidget(self.a_rows, 0, 1)
        
        input_layout.addWidget(QLabel("矩阵A 列数:"), 1, 0)
        self.a_cols = QLineEdit()
        self.a_cols.setText("512")
        input_layout.addWidget(self.a_cols, 1, 1)
        
        input_layout.addWidget(QLabel("矩阵B 列数:"), 2, 0)
        self.b_cols = QLineEdit()
        self.b_cols.setText("512")
        input_layout.addWidget(self.b_cols, 2, 1)
        
        layout.addWidget(input_group)
        
        # 后端选择
        backend_group = QGroupBox("计算后端")
        backend_layout = QVBoxLayout(backend_group)
        
        self.backend_auto = QCheckBox("自动选择")
        self.backend_auto.setChecked(True)
        self.backend_auto.toggled.connect(self._on_backend_toggled)
        backend_layout.addWidget(self.backend_auto)
        
        self.backend_gpu = QCheckBox("GPU (OpenCL)")
        self.backend_gpu.setEnabled(False)
        backend_layout.addWidget(self.backend_gpu)
        
        self.backend_cpu = QCheckBox("CPU多进程")
        backend_layout.addWidget(self.backend_cpu)
        
        layout.addWidget(backend_group)
        
        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_backend_toggled(self, checked):
        if checked:
            self.backend_gpu.setChecked(False)
            self.backend_cpu.setChecked(False)
    
    def get_params(self):
        return {
            "rows": int(self.a_rows.text()),
            "a_cols": int(self.a_cols.text()),
            "b_cols": int(self.b_cols.text()),
            "backend": "AUTO" if self.backend_auto.isChecked() else
                      "OPENCL" if self.backend_gpu.isChecked() else
                      "CPU_MP" if self.backend_cpu.isChecked() else "AUTO"
        }

class ImageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像处理")
        self.setModal(True)
        self.resize(500, 400)
        self.image_path = ""
        
        layout = QVBoxLayout(self)
        
        # 文件选择
        file_group = QGroupBox("选择图片")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_label)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_image)
        file_layout.addWidget(browse_btn)
        
        layout.addWidget(file_group)
        
        # 操作选择
        ops_group = QGroupBox("图像操作")
        ops_layout = QVBoxLayout(ops_group)
        
        self.op_checkboxes = {}
        operations = [
            ("grayscale", "灰度化"),
            ("edge", "边缘检测"),
            ("blur", "高斯模糊"),
            ("rotate", "旋转90度"),
            ("flip", "水平翻转"),
            ("invert", "颜色反转"),
            ("enhance", "对比度增强")
        ]
        
        for op, label in operations:
            cb = QCheckBox(label)
            ops_layout.addWidget(cb)
            self.op_checkboxes[op] = cb
        
        layout.addWidget(ops_group)
        
        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.image_path = file_path
            self.file_label.setText(os.path.basename(file_path))
    
    def get_params(self):
        selected_ops = [
            op for op, cb in self.op_checkboxes.items() 
            if cb.isChecked()
        ]
        return {
            "image_path": self.image_path,
            "operations": selected_ops
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = GPUAccelerator()
        self.current_image = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_stats)
        self.update_timer.start(1000)  # 每秒更新
        self._setup_ui()
        self._apply_theme()
    
    def _setup_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.resize(1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # 硬件信息
        hw_group = QGroupBox("硬件信息")
        hw_layout = QVBoxLayout(hw_group)
        
        self.device_tree = QTreeWidget()
        self.device_tree.setHeaderLabels(["设备", "规格"])
        self.device_tree.setIndentation(10)
        hw_layout.addWidget(self.device_tree)
        
        left_layout.addWidget(hw_group)
        
        # 任务控制
        task_group = QGroupBox("任务控制")
        task_layout = QVBoxLayout(task_group)
        
        matrix_btn = QPushButton("🧮 矩阵计算")
        matrix_btn.clicked.connect(self._show_matrix_dialog)
        matrix_btn.setMinimumHeight(40)
        task_layout.addWidget(matrix_btn)
        
        image_btn = QPushButton("🖼️ 图像处理")
        image_btn.clicked.connect(self._show_image_dialog)
        image_btn.setMinimumHeight(40)
        task_layout.addWidget(image_btn)
        
        left_layout.addWidget(task_group)
        
        # 系统设置
        settings_group = QGroupBox("系统设置")
        settings_layout = QVBoxLayout(settings_group)
        
        clear_cache_btn = QPushButton("🗑️ 清理缓存")
        clear_cache_btn.clicked.connect(self._clear_cache)
        settings_layout.addWidget(clear_cache_btn)
        
        theme_btn = QPushButton("🎨 切换主题")
        theme_btn.clicked.connect(self._toggle_theme)
        settings_layout.addWidget(theme_btn)
        
        left_layout.addWidget(settings_group)
        
        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        for key in ["待处理任务", "已完成任务", "缓存数量", "缓存命中"]:
            label = QLabel(f"{key}: 0")
            stats_layout.addWidget(label)
            self.stats_labels[key] = label
        
        left_layout.addWidget(stats_group)
        
        left_layout.addStretch()
        
        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # 任务历史标签页
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["任务ID", "类型", "时间", "耗时", "状态"])
        self.history_tree.setAlternatingRowColors(True)
        history_layout.addWidget(self.history_tree)
        
        self.tab_widget.addTab(history_widget, "📋 任务历史")
        
        # 结果标签页
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setScaledContents(True)
        self.result_image_label.setMinimumSize(400, 300)
        result_layout.addWidget(self.result_image_label)
        
        self.tab_widget.addTab(result_widget, "📊 计算结果")
        
        # 性能监控标签页
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)
        
        self.monitor_text = QTextEdit()
        self.monitor_text.setReadOnly(True)
        monitor_layout.addWidget(self.monitor_text)
        
        self.tab_widget.addTab(monitor_widget, "📈 性能监控")
        
        right_layout.addWidget(self.tab_widget)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 更新设备信息
        self._update_device_info()
    
    def _apply_theme(self):
        """应用主题"""
        theme = self.engine.config.get("theme", "light")
        
        if theme == "light":
            # 亮色主题（默认）
            light_palette = QPalette()
            light_palette.setColor(QPalette.Window, QColor(240, 240, 240))
            light_palette.setColor(QPalette.WindowText, Qt.black)
            light_palette.setColor(QPalette.Base, Qt.white)
            light_palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
            light_palette.setColor(QPalette.ToolTipBase, Qt.white)
            light_palette.setColor(QPalette.ToolTipText, Qt.black)
            light_palette.setColor(QPalette.Text, Qt.black)
            light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
            light_palette.setColor(QPalette.ButtonText, Qt.black)
            light_palette.setColor(QPalette.BrightText, Qt.red)
            light_palette.setColor(QPalette.Link, QColor(0, 100, 200))
            light_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            light_palette.setColor(QPalette.HighlightedText, Qt.white)
            light_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
            light_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
            self.setPalette(light_palette)
            
            # 设置样式表优化显示
            self.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QTreeWidget {
                    alternate-background-color: #f5f5f5;
                }
                QPushButton {
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
                QLineEdit {
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    padding: 3px;
                }
                QTabWidget::pane {
                    border: 1px solid #cccccc;
                }
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
            """)
            
        elif theme == "dark":
            # 暗色主题
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
            self.setPalette(dark_palette)
            
            # 暗色主题样式表
            self.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #555555;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                    color: white;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QTreeWidget {
                    alternate-background-color: #353535;
                    color: white;
                }
                QPushButton {
                    background-color: #353535;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 5px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #454545;
                }
                QPushButton:pressed {
                    background-color: #252525;
                }
                QLineEdit {
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 3px;
                    background-color: #252525;
                    color: white;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                }
                QTabBar::tab {
                    background-color: #353535;
                    color: white;
                    padding: 5px;
                }
                QTabBar::tab:selected {
                    background-color: #454545;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
                QLabel {
                    color: white;
                }
                QStatusBar {
                    color: white;
                }
            """)
    
    def _toggle_theme(self):
        """切换主题"""
        current = self.engine.config.get("theme", "light")
        self.engine.config["theme"] = "dark" if current == "light" else "light"
        self._apply_theme()
        
        # 保存配置
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.engine.config, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _update_device_info(self):
        """更新设备信息"""
        self.device_tree.clear()
        
        for device in self.engine.devices:
            item = QTreeWidgetItem(self.device_tree)
            item.setText(0, f"{device.type}: {device.name[:30]}")
            specs = f"内存: {device.memory}MB | 核心: {device.compute_units}"
            if device.type == "GPU":
                specs += f" | FP64: {'✓' if device.supports_fp64 else '✗'}"
            item.setText(1, specs)
    
    def _update_stats(self):
        """更新统计信息"""
        stats = self.engine.get_stats()
        self.stats_labels["待处理任务"].setText(f"待处理任务: {stats['tasks_pending']}")
        self.stats_labels["已完成任务"].setText(f"已完成任务: {stats['tasks_completed']}")
        self.stats_labels["缓存数量"].setText(f"缓存数量: {stats['cache_size']}")
        self.stats_labels["缓存命中"].setText(f"缓存命中: {stats['cache_hits']}")
        
        # 更新监控信息
        monitor_text = f"""系统状态
==========
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
OpenCL: {'可用' if self.engine.cl_available else '不可用'}
CPU核心: {self.engine.config['max_cpu_threads']}
缓存目录: {os.path.abspath(CACHE_DIR)}

任务队列状态
==========
高优先级: {len(self.engine.task_queue[TaskPriority.HIGH])}
普通优先级: {len(self.engine.task_queue[TaskPriority.NORMAL])}
低优先级: {len(self.engine.task_queue[TaskPriority.LOW])}

最近任务
==========
"""
        for task in list(self.engine.task_history)[-5:]:
            monitor_text += f"- {task['type']}: {task['status']} ({task['elapsed']:.2f}s)\n"
        
        self.monitor_text.setText(monitor_text)
    
    def _show_matrix_dialog(self):
        """显示矩阵计算对话框"""
        dialog = MatrixDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            
            try:
                # 生成随机矩阵
                a = np.random.rand(params["rows"], params["a_cols"])
                b = np.random.rand(params["a_cols"], params["b_cols"])
                
                # 提交任务
                task_id = self.engine.submit_task({
                    "type": "matrix_mult",
                    "a": a,
                    "b": b,
                    "backend": params["backend"],
                    "cacheable": True
                })
                
                self._monitor_task(task_id, "矩阵计算")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"参数错误: {e}")
    
    def _show_image_dialog(self):
        """显示图像处理对话框"""
        dialog = ImageDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            
            if not params["image_path"]:
                QMessageBox.warning(self, "警告", "请选择图片")
                return
            
            if not params["operations"]:
                QMessageBox.warning(self, "警告", "请至少选择一个操作")
                return
            
            # 提交任务
            task_id = self.engine.submit_task({
                "type": "image_process",
                "image": params["image_path"],
                "ops": params["operations"],
                "cacheable": True
            })
            
            self._monitor_task(task_id, "图像处理")
    
    def _monitor_task(self, task_id: str, task_name: str):
        """监控任务进度"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"{task_name} 进行中...")
        
        def check_result():
            result = self.engine.get_task_result(task_id, timeout=0.1)
            if result:
                self.progress_bar.setVisible(False)
                if result.success:
                    self._show_result(result)
                else:
                    QMessageBox.critical(self, "错误", f"任务失败: {result.message}")
                self.status_bar.showMessage("就绪")
                return
            
            # 模拟进度
            current = self.progress_bar.value()
            if current < 90:
                self.progress_bar.setValue(current + 5)
            
            QTimer.singleShot(100, check_result)
        
        QTimer.singleShot(100, check_result)
    
    def _show_result(self, result: TaskResult):
        """显示结果"""
        self.tab_widget.setCurrentIndex(1)  # 切换到结果标签页
        self.result_text.clear()
        
        # 显示任务信息
        info = f"""任务结果
==========
任务ID: {result.task_id}
后端: {result.backend}
耗时: {result.elapsed:.3f} 秒
状态: {'成功' if result.success else '失败'}
消息: {result.message}
时间: {datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')}

"""
        self.result_text.append(info)
        
        # 显示具体结果
        if isinstance(result.data, np.ndarray):
            self.result_text.append(f"结果矩阵形状: {result.data.shape}")
            self.result_text.append(f"数据统计:")
            self.result_text.append(f"  最小值: {result.data.min():.4f}")
            self.result_text.append(f"  最大值: {result.data.max():.4f}")
            self.result_text.append(f"  平均值: {result.data.mean():.4f}")
            self.result_text.append(f"  标准差: {result.data.std():.4f}")
            
            # 显示部分数据
            if result.data.size < 100:
                self.result_text.append(f"\n矩阵内容:\n{result.data}")
            else:
                self.result_text.append(f"\n前5x5子矩阵:\n{result.data[:5, :5]}")
            
            self.result_image_label.clear()
            
        elif isinstance(result.data, Image.Image):
            self.result_text.append(f"图像处理完成")
            self.result_text.append(f"尺寸: {result.data.size}")
            self.result_text.append(f"模式: {result.data.mode}")
            self.result_text.append(f"格式: {result.data.format}")
            
            # 显示图像
            data = result.data.convert("RGB")
            data = data.resize((400, 300), Image.Resampling.LANCZOS)
            
            # 转换为QPixmap
            if data.mode == "RGB":
                r, g, b = data.split()
                data = Image.merge("RGB", (b, g, r))
            
            img_data = data.tobytes("raw", data.mode)
            qimage = QImage(img_data, data.size[0], data.size[1], 
                           QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            self.result_image_label.setPixmap(
                pixmap.scaled(400, 300, Qt.KeepAspectRatio, 
                             Qt.SmoothTransformation)
            )
        
        # 添加到历史
        item = QTreeWidgetItem(self.history_tree)
        item.setText(0, result.task_id[:12])
        item.setText(1, "矩阵计算" if isinstance(result.data, np.ndarray) else "图像处理")
        item.setText(2, datetime.now().strftime("%H:%M:%S"))
        item.setText(3, f"{result.elapsed:.2f}s")
        item.setText(4, "✅ 成功" if result.success else "❌ 失败")
        
        # 自动滚动到底部
        self.history_tree.scrollToBottom()
    
    def _clear_cache(self):
        """清理缓存"""
        reply = QMessageBox.question(
            self, "确认", "确定要清理所有缓存吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for key in list(self.engine.cache.keys()):
                self.engine._remove_cache(key)
            QMessageBox.information(self, "完成", "缓存已清理")
    
    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, "退出", "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.update_timer.stop()
            self.engine.shutdown()
            event.accept()
        else:
            event.ignore()

# === 主程序 ===
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("GPU加速专家")
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()