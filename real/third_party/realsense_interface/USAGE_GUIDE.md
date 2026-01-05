# RealSense Interface 使用指南

## 快速开始

### 1. 安装

```bash
# 克隆或下载项目
cd realsense_interface

# 运行安装脚本 (推荐)
./install.sh

# 或手动安装
pip install -r requirements.txt
pip install -e .
```

### 2. 验证安装

```bash
# 检测相机
realsense-test --list-only

# 测试相机功能
realsense-test --serial YOUR_CAMERA_SERIAL
```

## 基本使用

### 单相机使用

```python
from realsense_interface import SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import time

# 获取相机序列号
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
        
        # 获取图像
        for i in range(100):
            data = camera.get()
            print(f"帧 {i}: {data['color'].shape}")
            time.sleep(1/30)
```

### 多相机使用

```python
from realsense_interface import MultiRealsense

with MultiRealsense(
    resolution=(1280, 720),
    capture_fps=30
) as cameras:
    print(f"相机数量: {cameras.n_cameras}")
    
    # 设置所有相机参数
    cameras.set_exposure(exposure=120, gain=0)
    
    # 获取所有相机数据
    data = cameras.get()
    for cam_idx, cam_data in data.items():
        print(f"相机 {cam_idx}: {cam_data['color'].shape}")
```

## 高级功能

### 1. 视频录制

```python
# 单相机录制
camera.start_recording("output.mp4", start_time=time.time() + 1)
# ... 获取数据 ...
camera.stop_recording()

# 多相机录制
cameras.start_recording("multi_output", start_time=time.time() + 1)
# 会创建 multi_output/0.mp4, multi_output/1.mp4 等文件
cameras.stop_recording()
```

### 2. 使用配置文件

```python
import json

# 加载高精度配置
with open('configs/415_high_accuracy_mode.json', 'r') as f:
    config = json.load(f)

with SingleRealsense(
    shm_manager=shm_manager,
    serial_number=serial,
    advanced_mode_config=config
) as camera:
    # 相机将使用高精度模式
    pass
```

### 3. 自定义数据变换

```python
def resize_transform(data):
    """将图像缩放到640x480"""
    import cv2
    data['color'] = cv2.resize(data['color'], (640, 480))
    return data

with SingleRealsense(
    shm_manager=shm_manager,
    serial_number=serial,
    transform=resize_transform
) as camera:
    # 获取的数据将自动缩放
    data = camera.get()
```

### 4. 深度图像

```python
with SingleRealsense(
    shm_manager=shm_manager,
    serial_number=serial,
    enable_color=True,
    enable_depth=True
) as camera:
    data = camera.get()
    color_img = data['color']  # (H, W, 3) uint8
    depth_img = data['depth']  # (H, W) uint16
    
    # 获取深度比例因子
    depth_scale = camera.get_depth_scale()
    depth_meters = depth_img * depth_scale
```

## 性能优化

### 1. 调整缓冲区大小

```python
with SingleRealsense(
    shm_manager=shm_manager,
    serial_number=serial,
    get_max_k=50,  # 增加缓冲区大小
    put_fps=60     # 提高数据传输频率
) as camera:
    pass
```

### 2. 多相机USB带宽管理

```python
# 降低分辨率以支持更多相机
with MultiRealsense(
    resolution=(640, 480),  # 较低分辨率
    capture_fps=15          # 较低帧率
) as cameras:
    pass
```

### 3. 线程优化

```python
import cv2
cv2.setNumThreads(1)  # 限制OpenCV线程数
```

## 常见问题

### 1. 相机无法连接

```bash
# 检查USB连接
lsusb | grep Intel

# 检查权限
sudo usermod -a -G plugdev $USER
# 重新登录

# 检查RealSense服务
rs-enumerate-devices
```

### 2. 性能问题

- 降低分辨率或帧率
- 使用USB 3.0接口
- 检查CPU使用率
- 调整缓冲区参数

### 3. 内存问题

```python
# 手动管理共享内存
with SharedMemoryManager() as shm_manager:
    # 使用完毕后自动清理
    pass
```

## API 参考

### SingleRealsense 参数

- `serial_number`: 相机序列号
- `resolution`: 图像分辨率 (width, height)
- `capture_fps`: 捕获帧率
- `enable_color`: 启用彩色图像
- `enable_depth`: 启用深度图像
- `enable_infrared`: 启用红外图像
- `advanced_mode_config`: 高级模式配置
- `transform`: 数据变换函数
- `video_recorder`: 自定义视频录制器

### 主要方法

- `get()`: 获取最新数据
- `get(k=N)`: 获取最近N帧数据
- `set_exposure(exposure, gain)`: 设置曝光参数
- `set_white_balance(wb)`: 设置白平衡
- `get_intrinsics()`: 获取相机内参
- `start_recording(path, start_time)`: 开始录制
- `stop_recording()`: 停止录制

## 示例项目

查看 `examples/` 目录中的完整示例:

- `single_camera_example.py`: 单相机基本使用
- `multi_camera_example.py`: 多相机同步使用
- `test_camera.py`: 相机测试和诊断工具