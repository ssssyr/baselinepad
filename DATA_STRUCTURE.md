# Demo Gripper 数据保存结构分析 (Enhanced Version)

## 概述
`demo_gripper.py` 脚本使用了增强版的 `RealEnvGripper` 类，在原有数据基础上新增了夹爪相关数据的收集和保存功能。

## 保存的数据类型

### 1. 主数据文件
- **文件位置**: `<output_dir>/replay_buffer.zarr`
- **格式**: Zarr 格式（高效的多维数组存储）
- **压缩**: 使用 'disk' 压缩器

### 2. 视频文件
- **文件位置**: `<output_dir>/videos/<episode_id>/<camera_id>.mp4`
- **格式**: H.264 编码的 MP4 文件
- **分辨率**: 默认 1280x720 (可配置)
- **帧率**: 默认 30 FPS (可配置)
- **编码参数**: CRF=21 (可配置)

## 详细数据内容

### 每个 Episode 包含的数据字段：

#### 1. 时间戳数据
- **`timestamp`**: 观测时间戳数组
  - 类型: float64 数组
  - 单位: Unix 时间戳（秒）
  - 用途: 同步所有数据流

#### 2. 机器人状态数据
- **`robot_eef_pose`**: 机器人末端执行器位姿
  - 类型: float64 数组，形状 [n_steps, 6]
  - 内容: [x, y, z, rx, ry, rz] (位置 + 旋转)
  - 单位: 米 + 弧度

- **`robot_eef_pose_vel`**: 末端执行器速度
  - 类型: float64 数组，形状 [n_steps, 6]
  - 内容: [vx, vy, vz, wx, wy, wz] (线速度 + 角速度)
  - 单位: 米/秒 + 弧度/秒

- **`robot_joint`**: 机器人关节位置
  - 类型: float64 数组，形状 [n_steps, 6]
  - 内容: 6个关节的角度值
  - 单位: 弧度

- **`robot_joint_vel`**: 机器人关节速度
  - 类型: float64 数组，形状 [n_steps, 6]
  - 内容: 6个关节的角速度值
  - 单位: 弧度/秒

#### 3. 动作数据
- **`action`**: 执行的动作命令
  - 类型: float64 数组，形状 [n_steps, 6]
  - 内容: 目标末端执行器位姿 [x, y, z, rx, ry, rz]
  - 单位: 米 + 弧度

- **`stage`**: 动作阶段标识
  - 类型: int64 数组，形状 [n_steps]
  - 内容: 阶段编号（通常为0）

#### 4. 🆕 夹爪状态数据 
- **`gripper_closed`**: 夹爪抓取状态
  - 类型: int32 数组，形状 [n_steps]
  - 内容: 0=未抓到东西, 1=抓到东西
  - 数据来源: RG2-FT夹爪的gripDetected传感器

- **`gripper_target`**: 夹爪目标状态
  - 类型: int32 数组，形状 [n_steps]
  - 内容: 0=目标打开, 1=目标闭合
  - 判断标准: 目标位置 < 0.5时为目标闭合

- **`gripper_width`**: 夹爪真实张开宽度
  - 类型: float32 数组，形状 [n_steps]
  - 内容: 夹爪当前宽度值
  - 单位: 毫米 (mm)

- **`gripper_force`**: 夹爪六轴力信号
  - 类型: float32 数组，形状 [n_steps, 6]
  - 内容: [fx, fy, fz, tx, ty, tz] (合并左右传感器)
  - 单位: 力 (N) + 力矩 (Nm)

#### 5. 相机数据
- **`camera_<id>`**: 相机图像数据
  - 类型: uint8 数组，形状 [n_steps, height, width, 3]
  - 内容: RGB 图像数据
  - 分辨率: 默认 1280x720 (可配置)
  - 颜色空间: RGB

## 数据同步机制

### 时间对齐
- 相机数据: 30 Hz 采集
- 机器人数据: 125 Hz 采集
- 控制频率: 10 Hz (可配置)
- 所有数据通过时间戳对齐到控制频率

### 数据插值
- 使用最近邻插值方法
- 确保所有数据流在相同时间点有对应值

## 夹爪数据收集机制 (🆕 新增功能)

### 数据来源
- **RG2-FT夹爪**: 通过Modbus协议实时读取
- **左右传感器**: 分别获取6轴力/力矩数据后合并
- **状态判断**: 基于宽度阈值和目标位置自动判断

### 数据更新频率
- **夹爪状态读取**: 每100ms读取一次（可配置）
- **数据保存频率**: 与控制频率同步（默认10Hz）
- **力传感器**: 实时读取并合并左右传感器数据

### 夹爪状态判断逻辑
```python
# 抓取状态判断 (直接使用RG2-FT传感器)
gripper_closed = modbus_gripper.gripDetected  # 1=抓到东西, 0=未抓到

# 目标状态判断  
gripper_target = 1 if target_position < 0.5 else 0
```

## 数据同步机制

### 时间对齐
- 相机数据: 30 Hz 采集
- 机器人数据: 125 Hz 采集
- 夹爪数据: 10 Hz 采集（与控制频率同步）
- 控制频率: 10 Hz (可配置)
- 所有数据通过时间戳对齐到控制频率

### 数据插值
- 使用最近邻插值方法
- 确保所有数据流在相同时间点有对应值
- 夹爪数据在每个时间步重复当前值

## 数据使用建议

### 训练数据预处理
1. **图像预处理**: 
   - 归一化到 [0,1] 范围
   - 可能需要调整分辨率

2. **机器人状态归一化**:
   - 位置数据可能需要相对于工作空间归一化
   - 关节角度已经在合理范围内

3. **时间序列处理**:
   - 可以使用滑动窗口创建历史观测
   - 注意处理不同长度的 episode

### 数据质量检查
1. **时间戳连续性**: 检查时间戳是否单调递增
2. **数据完整性**: 确保所有字段都有数据
3. **异常值检测**: 检查机器人状态是否在合理范围内

## 存储空间估算 (更新)

### 单个 Episode (假设 10 秒，10 Hz)
- 时间戳: 100 × 8 bytes = 800 bytes
- 机器人状态: 100 × 6 × 8 × 4 = 19.2 KB
- 动作数据: 100 × 6 × 8 = 4.8 KB
- **🆕 夹爪数据**: 
  - gripper_closed: 100 × 4 = 400 bytes (抓取状态)
  - gripper_target: 100 × 4 = 400 bytes  
  - gripper_width: 100 × 4 = 400 bytes
  - gripper_force: 100 × 6 × 4 = 2.4 KB
- 图像数据: 100 × 1280 × 720 × 3 = 276 MB
- 视频文件: ~20-50 MB (取决于内容复杂度)

### 总计每个 Episode: ~300-350 MB (夹爪数据增加约3.6KB)

## 配置参数影响

### 可调整的参数及其对数据的影响：
- `frequency`: 控制采样率，影响数据点数量
- `obs_image_resolution`: 影响图像数据大小
- `video_capture_resolution`: 影响视频文件大小
- `video_crf`: 影响视频压缩质量和文件大小
- `n_obs_steps`: 影响观测历史长度

## 数据访问示例 (更新)

```python
import zarr
import numpy as np

# 打开数据文件
store = zarr.open('output_dir/replay_buffer.zarr', mode='r')

# 访问特定 episode
episode_0 = store['data']['0']

# 获取机器人位姿数据
robot_poses = episode_0['robot_eef_pose'][:]

# 获取图像数据
camera_images = episode_0['camera_0'][:]

# 获取时间戳
timestamps = episode_0['timestamp'][:]

# 🆕 获取夹爪数据
gripper_closed = episode_0['gripper_closed'][:]      # 夹爪抓取状态 (0/1)
gripper_target = episode_0['gripper_target'][:]      # 夹爪目标状态 (0/1)
gripper_width = episode_0['gripper_width'][:]        # 夹爪宽度 (mm)
gripper_force = episode_0['gripper_force'][:]        # 夹爪6轴力 [n_steps, 6]

# 分析夹爪数据
print(f"Episode duration: {len(timestamps)} steps")
print(f"Gripper grasped ratio: {np.mean(gripper_closed):.2%}")
print(f"Average gripper width: {np.mean(gripper_width):.1f} mm")
print(f"Max force magnitude: {np.max(np.linalg.norm(gripper_force[:, :3], axis=1)):.2f} N")
```

## 🆕 新增功能使用指南

### 在demo_gripper.py中的使用
```python
# 使用增强版环境
from diffusion_policy.real_world.real_env_gripper import RealEnvGripper

# 创建环境时传递夹爪对象
env = RealEnvGripper(
    output_dir=output,
    robot_ip=robot_ip,
    modbus_gripper=modbus_gripper,  # 传递夹爪对象
    gripper_max_width=gripper_max_width
)

# 执行动作时包含夹爪动作
env.exec_actions_with_gripper(
    actions=[target_pose],
    timestamps=[timestamp],
    gripper_actions=[gripper_position],  # 0.0-1.0
    stages=[0]
)

# 获取夹爪状态用于显示
gripper_state = env.get_gripper_state()
print(f"Gripper grasped: {gripper_state['gripper_closed']}")
print(f"Gripper width: {gripper_state['gripper_width']:.1f} mm")
```