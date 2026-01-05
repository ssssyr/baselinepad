# UR10 机器人落地代码运行指南

> 完整的部署运行步骤和故障排查

---

## 目录

1. [运行前检查清单](#运行前检查清单)
2. [硬件连接与测试](#硬件连接与测试)
3. [软件环境配置](#软件环境配置)
4. [运行程序](#运行程序)
5. [故障排查](#故障排查)
6. [调试技巧](#调试技巧)

---

## 运行前检查清单

### 硬件清单

- [ ] **UR10 机器人** 已开机并连接到网络
- [ ] **RealSense 相机** 已连接到电脑
- [ ] **RG2-FT 夹爪**（可选）已连接到网络
- [ ] **网络连接** 电脑、机器人、夹爪在同一局域网

### 软件清单

- [ ] Python 3.8+
- [ ] PyTorch (GPU 版本推荐)
- [ ] UR RTDE 库
- [ ] RealSense SDK
- [ ] 所有模型文件已下载

### 文件清单

- [ ] 模型文件: `/home/syr/code/checkpoints/checkpoint0103mf/0030000.pt`
- [ ] VAE 模型: `/home/syr/code/models/sd-vae-ft-mse/`
- [ ] CLIP 模型: `/home/syr/code/models/clip-vit-base-patch32/`

---

## 硬件连接与测试

### Step 1: 检查 UR10 机器人连接

```bash
# 测试网络连通性
ping 192.168.1.50

# 如果不通，检查：
# 1. 机器人是否开机
# 2. 网线是否连接
# 3. IP 地址是否正确（在机器人示教器上查看）
```

### Step 2: 检查 RealSense 相机

```bash
# 列出所有连接的 RealSense 相机
rs-enumerate-devices

# 或使用 Python 测试
python3 -c "import pyrealsense2 as rs; ctx = rs.context(); print([d.get_info(rs.camera_info.serial_number) for d in ctx.devices])"

# 记录下你的相机序列号，例如: 1234567890
```

如果相机未检测到：
```bash
# 重新插拔 USB 线
# 或安装 RealSense SDK
sudo apt-get install librealsense2-utils
```

### Step 3: 检查夹爪（如果有）

```bash
# 测试夹爪网络连接
ping <夹爪IP>

# 默认夹爪 IP 通常在 192.168.1.x 网段
```

---

## 软件环境配置

### Step 1: 安装 Python 依赖

```bash
cd /home/syr/code/baselinepad

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install opencv-python numpy
pip install transformers diffusers
pip install pyyaml click
pip install pymodbus
pip install pygame  # 手柄控制需要

# 安装 UR10 通信库
pip install ur_rtde

# 安装 RealSense 接口
pip install realsense2
```

### Step 2: 验证安装

```bash
# 验证 PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 验证其他库
python3 -c "import cv2, rs, rtde, transformers; print('All libraries OK')"
```

---

## 运行程序

### 方式 1: 模拟模式测试（无硬件）

如果你想先测试代码逻辑，无需实际硬件：

```bash
cd /home/syr/code/baselinepad/real

# 创建一个测试脚本
cat > test_simulation.py << 'EOF'
import sys
sys.path.append("..")

from configs.ur10_config import CONFIG

print("=== 测试配置加载 ===")
print(f"模型路径: {CONFIG['model']['ckpt_path']}")
print(f"VAE 路径: {CONFIG['model']['vae_path']}")
print(f"CLIP 路径: {CONFIG['model']['clip_path']}")
print(f"机器人 IP: {CONFIG['robot']['ip']}")
print(f"任务指令: {CONFIG['task']['task_instruction']}")

# 测试模型加载（不连接硬件）
print("\n=== 测试模型加载 ===")
try:
    from evaluation.agent import DiffusionAgent
    agent = DiffusionAgent(
        ckpt_path=CONFIG['model']['ckpt_path'],
        vae_path=CONFIG['model']['vae_path'],
        clip_path=CONFIG['model']['clip_path'],
        denoise_steps=CONFIG['model']['denoise_steps'],
        device_id=CONFIG['model']['gpu_id'],
    )
    print("✓ 模型加载成功！")
    print(f"  action_scale: {agent.args.action_scale}")
    print(f"  predict_horizon: {agent.args.predict_horizon}")
    print(f"  use_force: {agent.args.use_force}")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
EOF

python3 test_simulation.py
```

### 方式 2: 完整运行（有硬件）

#### 2.1 更新配置文件

编辑 `/home/syr/code/baselinepad/real/configs/ur10_config.py`：

```python
CONFIG = {
    "model": {
        "ckpt_path": "/home/syr/code/checkpoints/checkpoint0103mf/0030000.pt",
        "vae_path": "/home/syr/code/models/sd-vae-ft-mse",
        "clip_path": "/home/syr/code/models/clip-vit-base-patch32",
        "denoise_steps": 5,
        "gpu_id": 0,
    },
    "camera": {
        "serial_number": "",  # 留空自动检测
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "robot": {
        "ip": "192.168.1.102",  # 确认这是你的 UR10 IP
        "control_freq": 10,
    },
    "task": {
        "max_steps": 50,
        "task_instruction": "pick up the red block",  # 修改为你的任务
    },
}
```

#### 2.2 运行主程序

```bash
cd /home/syr/code/baselinepad/real

# 激活虚拟环境（如果使用）
source venv/bin/activate

# 运行
python main.py
```

### 方式 3: 手柄控制模式

如果你想用手柄远程控制机器人：

```bash
cd /home/syr/code/baselinepad/real/scripts

# 运行手柄控制
python gamepad_control.py --robot_ip 192.168.1.102

# 如果有夹爪
python gamepad_control.py --robot_ip 192.168.1.102 --gripper_ip <夹爪IP>
```

---

## 运行输出解读

### 正常启动的输出

```
--- UR10 Real-World Deployment Script ---

1. Initializing Hardware...
No serial number provided, will auto-detect camera...
Starting camera <serial>...
Camera started successfully.
机械臂连接成功: 192.168.1.102
当前位姿: [-0.074, 0.661, 0.002, -2.211, -2.170, -0.014]

2. Loading AI Agent...
🔄 Loading model from: /home/syr/code/checkpoints/checkpoint0103mf/0030000.pt
📁 Model file size: 4.73 GB
📋 Model keys: ['model', 'args', 'epoch', 'global_step']
🎯 Action scale: 1
📊 Image size: 256
load dit
load diffusion
load vae and clip
Agent built successfully.

Initialization complete. Starting main control loop.

--- Step 1/50 ---
y shape: torch.Size([1, 512])
x_cond shape: torch.Size([1, 4, 32, 32])
depth_cond shape: None
🎲 Initial noise z mean: 0.0012, std: 1.0234
🎯 Input state: [-0.074  0.661  0.002  0.5]
🎯 Action scale: 1
🎯 Action prediction shape: torch.Size([1, 3, 4])
🎯 Action prediction sample values: [[ 0.123 -0.456  0.789  0.5]]
Moving to pose: [ 0.123 -0.456  0.789 -2.211 -2.170 -0.014]
Movement complete.
```

### 关键输出含义

| 输出 | 含义 |
|------|------|
| `Camera started successfully` | 相机初始化成功 |
| `机械臂连接成功` | UR10 连接成功 |
| `Agent built successfully` | 模型加载成功 |
| `Action scale: 1` | 动作缩放因子为 1（来自 checkpoint） |
| `Action prediction shape: torch.Size([1, 3, 4])` | 预测 3 帧动作，每帧 4 个值 (x,y,z,gripper) |
| `Moving to pose: [...]` | 机器人正在移动到目标位置 |

---

## 故障排查

### 问题 1: 模型加载失败

```
Error: Failed to import DiffusionAgent
```

**解决方案：**
```bash
# 检查项目路径
cd /home/syr/code/baselinepad
python3 -c "import sys; sys.path.append('.'); from evaluation.agent import DiffusionAgent; print('OK')"

# 检查模型文件是否存在
ls -lh /home/syr/code/checkpoints/checkpoint0103mf/0030000.pt

# 检查 VAE/CLIP 路径
ls -la /home/syr/code/models/sd-vae-ft-mse/
ls -la /home/syr/code/models/clip-vit-base-patch32/
```

---

### 问题 2: UR10 连接失败

```
机械臂连接失败: [Errno 111] Connection refused
警告：机器人处于模拟模式
```

**解决方案：**
```bash
# 1. 检查网络连通性
ping 192.168.1.102

# 2. 检查 RTDE 连接
python3 -c "import rtde_control; r = rtde_control.RTDEControlInterface('192.168.1.102'); print('OK')"

# 3. 在 UR10 示教器上检查：
#    - 设置 → 网络 → 查看 IP 地址
#    - 确保 UR10 的 "Remote Control" 已开启
```

---

### 问题 3: 相机初始化失败

```
Failed to start camera: No device connected
```

**解决方案：**
```bash
# 1. 列出所有 USB 设备
lsusb | grep Intel

# 2. 测试 RealSense
rs-enumerate-devices

# 3. 重新插拔相机，或安装驱动
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
```

---

### 问题 4: CUDA/GPU 错误

```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 1. 检查 GPU 状态
nvidia-smi

# 2. 如果显存不足，使用 CPU
# 编辑 ur10_config.py: "gpu_id": -1  # 使用 CPU

# 3. 减少 batch size 或模型大小
```

---

### 问题 5: 动作预测异常

```
Warning: Target pose out of workspace
```

**解决方案：**
```python
# 检查工作空间限制是否合理
# 编辑 main.py 中的 is_pose_safe() 函数：

default_limits = {
    'x_min': -0.8, 'x_max': 0.8,  # 根据你的 UR10 调整
    'y_min': -0.8, 'y_max': 0.8,
    'z_min': 0.0,  'z_max': 1.2
}
```

---

## 调试技巧

### 技巧 1: 分步测试

创建测试脚本逐步验证每个模块：

```python
# test_hardware.py
from hardware.ur10_manager import UR10Manager
from hardware.camera_manager import CameraManager

# 测试机器人
robot = UR10Manager(robot_ip="192.168.1.102")
pose, gripper = robot.get_tcp_pose()
print(f"机器人位姿: {pose}")

# 测试相机
camera = CameraManager(serial_number=None)
frame = camera.get_latest_frame()
print(f"相机图像: {frame.shape}")
```

### 技巧 2: 可视化调试

```python
# 在 main.py 中添加可视化
import cv2

# 在获取图像后
cv2.imwrite(f"debug_frame_{step}.jpg", rgb_image)

# 可视化预测的动作
print(f"预测位置: x={target_xyz[0]:.3f}, y={target_xyz[1]:.3f}, z={target_xyz[2]:.3f}")
```

### 技巧 3: 降低控制频率

```python
# 在 ur10_config.py 中
"robot": {
    "control_freq": 5,  # 从 10 降到 5，给机器人更多时间
}
```

### 技巧 4: 记录日志

```python
import logging

logging.basicConfig(
    filename='deployment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 在关键位置添加日志
logging.info(f"Step {step}: Target pose = {target_pose}")
```

---

## 快速参考

### 常用命令

```bash
# 查看相机列表
rs-enumerate-devices

# 测试机器人连接
ping 192.168.1.102

# 运行主程序
cd /home/syr/code/baselinepad/real && python main.py

# 运行手柄控制
cd /home/syr/code/baselinepad/real/scripts
python gamepad_control.py --robot_ip 192.168.1.102

# 查看 GPU 状态
nvidia-smi

# 查看 Python 路径
python3 -c "import sys; print('\n'.join(sys.path))"
```

### 配置文件位置

```
主配置: /home/syr/code/baselinepad/real/configs/ur10_config.py
训练配置: /home/syr/code/baselinepad/configs/metaworld_4d.yaml
模型文件: /home/syr/code/checkpoints/checkpoint0103mf/0030000.pt
```

---

## 安全注意事项

⚠️ **重要安全提示**

1. **急停开关**: 机器人旁必须有人看管，随时准备按急停
2. **低速测试**: 首次运行时使用较低的速度
3. **工作空间**: 确保机器人周围无障碍物
4. **模拟测试**: 先在模拟模式或低速下测试
5. **力传感器**: 注意力反馈，异常时立即停止

---

## 下一步

运行成功后，你可以：

1. **调整任务指令**: 修改 `task_instruction` 尝试不同任务
2. **调优参数**: 调整 `control_freq`、`denoise_steps` 等
3. **添加新功能**: 扩展代码支持更多传感器或任务
4. **性能优化**: 提高推理速度和精度

---

*文档更新时间：2026-01-03*
