# RealSense Camera Interface

这是一个从diffusion_policy项目中提取的RealSense相机接口模块，提供了对Intel RealSense相机的高级封装和多相机支持。

## 功能特性

- **单相机支持**: `SingleRealsense` 类提供单个RealSense相机的完整控制
- **多相机支持**: `MultiRealsense` 类支持同时管理多个RealSense相机
- **高性能**: 使用多进程和共享内存实现高效的数据传输
- **视频录制**: 内置H.264视频录制功能
- **实时控制**: 支持曝光、增益、白平衡等参数的实时调整
- **配置管理**: 支持高精度模式等预设配置

## 系统要求

- Python 3.7+
- Intel RealSense SDK 2.0
- pyrealsense2
- OpenCV
- NumPy
- PyAV (用于视频录制)

## 安装依赖

```bash
# 安装RealSense SDK (Ubuntu/Debian)
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev

# 安装Python依赖
pip install -r requirements.txt
```

## 快速开始

### 单相机使用

```python
from realsense_interface import SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import time

# 获取连接的相机序列号
serials = SingleRealsense.get_connected_devices_serial()
print(f"发现相机: {serials}")

with SharedMemoryManager() as shm_manager:
    with SingleRealsense(
        shm_manager=shm_manager,
        serial_number=serials[0],
        resolution=(1280, 720),
        capture_fps=30
    ) as camera:
        # 设置相机参数
        camera.set_exposure(exposure=120, gain=0)
        camera.set_white_balance(white_balance=5900)
        
        # 获取相机内参
        intrinsics = camera.get_intrinsics()
        print(f"相机内参: {intrinsics}")
        
        # 开始录制视频
        camera.start_recording("output.mp4", start_time=time.time() + 1)
        
        # 获取图像数据
        for i in range(100):
            data = camera.get()
            print(f"帧 {i}: {data['color'].shape}")
            time.sleep(1/30)
        
        # 停止录制
        camera.stop_recording()
```

### 多相机使用

```python
from realsense_interface import MultiRealsense
import time

with MultiRealsense(
    resolution=(1280, 720),
    capture_fps=30,
    enable_color=True,
    enable_depth=False
) as cameras:
    # 设置所有相机参数
    cameras.set_exposure(exposure=120, gain=0)
    cameras.set_white_balance(white_balance=5900)
    
    # 获取所有相机内参
    intrinsics = cameras.get_intrinsics()
    print(f"相机数量: {cameras.n_cameras}")
    
    # 开始录制 (每个相机一个文件)
    cameras.start_recording("multi_camera_output", start_time=time.time() + 1)
    
    # 获取所有相机数据
    for i in range(100):
        data = cameras.get()
        for cam_idx, cam_data in data.items():
            print(f"相机 {cam_idx} 帧 {i}: {cam_data['color'].shape}")
        time.sleep(1/30)
    
    # 停止录制
    cameras.stop_recording()
```

## 高级配置

### 使用预设配置文件

```python
import json
from realsense_interface import SingleRealsense

# 加载高精度模式配置
with open('configs/415_high_accuracy_mode.json', 'r') as f:
    config = json.load(f)

with SingleRealsense(
    shm_manager=shm_manager,
    serial_number=serial,
    advanced_mode_config=config
) as camera:
    # 相机将使用高精度模式配置
    pass
```

### 自定义数据变换

```python
def custom_transform(data):
    # 对图像数据进行自定义处理
    data['color'] = cv2.resize(data['color'], (640, 480))
    return data

with SingleRealsense(
    shm_manager=shm_manager,
    serial_number=serial,
    transform=custom_transform
) as camera:
    # 获取的数据将经过自定义变换
    data = camera.get()
```

## API 参考

### SingleRealsense

主要参数:
- `serial_number`: 相机序列号
- `resolution`: 图像分辨率 (width, height)
- `capture_fps`: 捕获帧率
- `enable_color`: 启用彩色图像
- `enable_depth`: 启用深度图像
- `enable_infrared`: 启用红外图像
- `advanced_mode_config`: 高级模式配置字典

主要方法:
- `get()`: 获取最新图像数据
- `set_exposure(exposure, gain)`: 设置曝光和增益
- `set_white_balance(white_balance)`: 设置白平衡
- `get_intrinsics()`: 获取相机内参矩阵
- `start_recording(path, start_time)`: 开始录制视频
- `stop_recording()`: 停止录制视频

### MultiRealsense

继承SingleRealsense的所有功能，额外提供:
- 自动发现和管理多个相机
- 同步数据获取
- 批量参数设置

## 配置文件

项目包含两个预设配置文件:
- `configs/415_high_accuracy_mode.json`: RealSense D415高精度模式
- `configs/435_high_accuracy_mode.json`: RealSense D435高精度模式

这些配置文件优化了深度精度和图像质量。

## 性能优化

1. **多进程架构**: 相机捕获在独立进程中运行，避免GIL限制
2. **共享内存**: 使用共享内存传输图像数据，减少拷贝开销
3. **环形缓冲区**: 实现高效的数据缓存和访问
4. **线程控制**: 优化OpenCV和视频编码线程使用

## 故障排除

### 常见问题

1. **相机无法连接**
   - 检查USB连接和权限
   - 确认RealSense SDK正确安装
   - 运行 `rs-enumerate-devices` 检查设备

2. **性能问题**
   - 降低分辨率或帧率
   - 检查USB带宽限制
   - 使用USB 3.0接口

3. **录制问题**
   - 确认有足够磁盘空间
   - 检查视频编码器可用性

## 许可证

本项目基于原diffusion_policy项目的许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。