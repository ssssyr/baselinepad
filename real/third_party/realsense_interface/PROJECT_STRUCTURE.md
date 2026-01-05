# 项目结构说明

```
realsense_interface/
├── README.md                    # 项目说明文档
├── LICENSE                      # 许可证文件
├── PROJECT_STRUCTURE.md         # 项目结构说明 (本文件)
├── requirements.txt             # Python依赖列表
├── setup.py                     # 安装配置
├── install.sh                   # 安装脚本
├── __init__.py                  # 包初始化文件
│
├── configs/                     # 配置文件目录
│   ├── 415_high_accuracy_mode.json  # D415高精度模式配置
│   └── 435_high_accuracy_mode.json  # D435高精度模式配置
│
├── shared_memory/               # 共享内存模块
│   ├── __init__.py
│   ├── shared_ndarray.py        # 共享NumPy数组
│   ├── shared_memory_util.py    # 共享内存工具
│   ├── shared_memory_queue.py   # 共享内存队列
│   └── shared_memory_ring_buffer.py  # 共享内存环形缓冲区
│
├── common/                      # 通用工具模块
│   ├── __init__.py
│   └── timestamp_accumulator.py # 时间戳累加器
│
├── examples/                    # 示例代码
│   ├── __init__.py
│   ├── single_camera_example.py # 单相机示例
│   ├── multi_camera_example.py  # 多相机示例
│   └── test_camera.py           # 相机测试工具
│
├── tests/                       # 测试代码
│   ├── __init__.py
│   └── test_shared_memory.py    # 共享内存测试
│
├── single_realsense.py          # 单相机接口
├── multi_realsense.py           # 多相机接口
└── video_recorder.py            # 视频录制模块
```

## 核心模块说明

### 1. 相机接口模块
- `single_realsense.py`: 单个RealSense相机的完整封装，支持多进程、共享内存、视频录制
- `multi_realsense.py`: 多相机管理器，可同时控制多个RealSense相机

### 2. 共享内存模块 (`shared_memory/`)
高性能的进程间通信组件:
- `SharedNDArray`: 共享NumPy数组，支持零拷贝数据传输
- `SharedMemoryQueue`: 无锁FIFO队列，用于命令传递
- `SharedMemoryRingBuffer`: 无锁环形缓冲区，用于高频数据流
- `SharedAtomicCounter`: 原子计数器，支持线程安全的计数操作

### 3. 视频录制模块
- `video_recorder.py`: H.264视频录制，支持实时编码和时间戳同步

### 4. 配置文件 (`configs/`)
预设的相机配置，优化了不同型号RealSense相机的性能:
- D415高精度模式: 适用于精密测量应用
- D435高精度模式: 适用于机器人视觉应用

### 5. 示例代码 (`examples/`)
- 单相机使用示例: 展示基本的相机操作
- 多相机使用示例: 展示多相机同步捕获
- 相机测试工具: 用于验证相机功能和性能

## 设计特点

### 高性能架构
1. **多进程设计**: 相机捕获在独立进程中运行，避免GIL限制
2. **共享内存**: 使用共享内存传输图像数据，减少拷贝开销
3. **无锁数据结构**: 环形缓冲区和队列使用无锁设计，提高并发性能

### 易用性
1. **上下文管理器**: 支持`with`语句，自动管理资源
2. **统一接口**: 单相机和多相机使用相似的API
3. **配置管理**: 预设配置文件，开箱即用

### 扩展性
1. **数据变换**: 支持自定义数据处理管道
2. **视频录制**: 内置视频录制功能，支持多种格式
3. **参数控制**: 实时调整相机参数，如曝光、增益、白平衡

## 依赖关系

### 核心依赖
- `pyrealsense2`: Intel RealSense SDK Python绑定
- `numpy`: 数值计算库
- `opencv-python`: 计算机视觉库
- `av`: 视频编解码库

### 可选依赖
- `atomics`: 原子操作库 (提供更好的性能)
- `threadpoolctl`: 线程池控制

### 系统依赖
- Intel RealSense SDK 2.0
- librealsense2 (Linux系统库)

## 使用场景

1. **机器人视觉**: 实时图像处理和深度感知
2. **增强现实**: 高精度相机标定和跟踪
3. **工业检测**: 多相机同步测量
4. **科研应用**: 高性能数据采集和分析