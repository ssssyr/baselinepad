# DeMUSE 架构附录：仿真增强与实物流水线

## 概述

本文档详细描述了 DeMUSE 架构中的两个核心组件：(1) 对 MetaWorld 仿真环境的全模态增强，包括力觉感知的坐标系变换、Savitzky-Golay 滤波和深度图优化；(2) 真实世界机器人数据采集流水线的技术细节。

---

## 核心参数总结

| 参数 | 数值 | 说明 |
|------|------|------|
| **力传感器** | OnRobot RG2-FT | 6 轴力/力矩 |
| **力裁剪范围** | ±100 N (真实) / ±20 N (仿真) | 去除离群值 |
| **力矩裁剪范围** | ±10 Nm (真实) / ±2 Nm (仿真) | 去除离群值 |
| **SG 滤波窗口** | 5 | Savitzky-Golay 窗口大小 |
| **SG 多项式阶数** | 2 | Savitzky-Golay 多项式阶数 |
| **深度滤波** | MedianBlur, kernel=15 | 去除椒盐噪声 |
| **深度裁剪范围** | [1000, 5000] | 有效量程 (mm) |
| **相机频率** | 30 Hz | RealSense D435i |
| **力传感器频率** | 100 Hz | 高频采集 |
| **对齐阈值** | 50 ms | 异步同步最大误差 |
| **Ring Buffer** | 2 秒 | 滑动窗口大小 |

---

## 增强型 MetaWorld MT50 平台

为了模拟复杂的物理交互，我们对 MetaWorld 仿真环境进行了全模态增强，主要包括力觉感知（含坐标系变换与信号滤波）和深度图优化两个方面。

### 力觉感知与预处理

DeMUSE 引入 6 轴力/力矩信号以增强机器人对接触状态的理解。力觉数据 $\boldsymbol{f}_t = [f_x, f_y, f_z, \tau_x, \tau_y, \tau_z]^\top \in \mathbb{R}^6$ 包含三轴力和三轴力矩，直接反映末端执行器（End-Effector, EE）与环境的交互状态。

#### 坐标系变换：从世界坐标系到 EE 坐标系

MetaWorld 仿真环境中的力传感器默认输出**世界坐标系**下的力和力矩数据。为了使力信号与末端执行器的姿态对齐，我们需要将其变换到 EE 坐标系中。

**变换数学推导**

设 $\boldsymbol{f}_{\text{world}} \in \mathbb{R}^3$ 和 $\boldsymbol{\tau}_{\text{world}} \in \mathbb{R}^3$ 分别为世界坐标系下的力和力矩，$\boldsymbol{q} = [q_w, q_x, q_y, q_z]^\top$ 为 EE 的姿态四元数（MuJoCo 格式，scalar-first）。

坐标系变换步骤如下：

1. **获取旋转矩阵**：将四元数转换为旋转矩阵 $\boldsymbol{R}_{\text{ee}}^{\text{world}}$（表示从 EE 坐标系到世界坐标系的旋转）
   $$
   \boldsymbol{R}_{\text{ee}}^{\text{world}} = \text{Rotation}(\boldsymbol{q}).\text{as_matrix}()
   $$

2. **求逆变换**：EE 坐标系到世界坐标系的变换矩阵为转置关系
   $$
   \boldsymbol{R}_{\text{world}}^{\text{ee}} = \left(\boldsymbol{R}_{\text{ee}}^{\text{world}}\right)^\top
   $$

3. **力和力矩变换**：通过旋转矩阵将世界坐标系的量变换到 EE 坐标系
   $$
   \boldsymbol{f}_{\text{ee}} = \boldsymbol{R}_{\text{world}}^{\text{ee}} \cdot \boldsymbol{f}_{\text{world}}
   $$
   $$
   \boldsymbol{\tau}_{\text{ee}} = \boldsymbol{R}_{\text{world}}^{\text{ee}} \cdot \boldsymbol{\tau}_{\text{world}}
   $$

**代码实现** (datasets/collect_metaworld_data_raw.py:215-250):
```python
def get_ee_force_torque(env) -> np.ndarray:
    # 获取世界坐标系下的力和力矩
    force_world = env.sim.data.sensordata[force_adr:force_adr+3].copy()
    torque_world = env.sim.data.sensordata[torque_adr:torque_adr+3].copy()

    # 获取EE的姿态四元数 [w, x, y, z]
    body_id = env.model.body_name2id("hand")
    quat = env.sim.data.body_xquat[body_id].copy()  # [w, x, y, z]

    # 转换为旋转矩阵（scipy需要 [x, y, z, w] 格式）
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    R_world_to_ee = rotation.as_matrix().T  # 转置得到世界到EE的变换

    # 坐标系变换
    force_ee = R_world_to_ee @ force_world
    torque_ee = R_world_to_ee @ torque_world

    return np.concatenate([force_ee, torque_ee])
```

#### Savitzky-Golay 滤波器

MetaWorld 仿真环境中的接触力计算会产生瞬间巨大的接触力峰值（100N+），这与真实物理接触的连续性不符。为平滑这些异常峰值，我们采用 **Savitzky-Golay (SG) 滤波器**对力信号进行时域平滑。

**滤波器配置**

| 参数 | 数值 | 说明 |
|------|------|------|
| `window_size` | 5 | SG 滤波窗口大小（奇数） |
| `polyorder` | 2 | 多项式拟合阶数 |
| `clip_force` | 20.0 N | 力截断阈值 |
| `clip_torque` | 2.0 Nm | 力矩截断阈值 |

**滤波流程**

1. **预截断**：首先对原始信号进行硬截断，防止异常值污染滤波器
   $$
   \tilde{f}_i = \text{clip}(f_i, -F_{\text{max}}, F_{\text{max}}), \quad i \in \{x, y, z\}
   $$
   $$
   \tilde{\tau}_j = \text{clip}(\tau_j, -T_{\text{max}}, T_{\text{max}}), \quad j \in \{x, y, z\}
   $$
   其中 $F_{\text{max}} = 20.0$ N, $T_{\text{max}} = 2.0$ Nm。

2. **SG 滤波**：对每个维度分别应用 Savitzky-Golay 数字滤波器
   $$
   \hat{f}_i^{(t)} = \text{SGFilter}\left([f_i^{(t-w+1)}, \ldots, f_i^{(t)}]\right)
   $$
   其中 $w = 5$ 为窗口大小，使用 2 阶多项式拟合。

3. **历史维护**：维护一个固定长度的历史窗口，仅更新最新滤波值，保持历史一致性。

**代码实现** (datasets/collect_metaworld_data_raw.py:255-315):
```python
class ForceFilter:
    def __init__(self, window_size=5, polyorder=2, clip_force=20.0, clip_torque=2.0):
        self.window_size = window_size
        self.polyorder = polyorder
        self.clip_force = clip_force
        self.clip_torque = clip_torque
        self.history = []

    def filter(self, force: np.ndarray) -> np.ndarray:
        force = np.array(force, dtype=np.float32)

        # 1. 先截断极端值（防止 SG 滤波器被异常值污染）
        force[:3] = np.clip(force[:3], -self.clip_force, self.clip_force)
        force[3:] = np.clip(force[3:], -self.clip_torque, self.clip_torque)

        # 2. 添加到历史
        self.history.append(force.copy())

        # 3. 使用 Savitzky-Golay 滤波器平滑
        if len(self.history) >= self.window_size:
            recent = np.array(self.history[-self.window_size:])
            for i in range(6):
                filtered = savgol_filter(recent[:, i], self.window_size, self.polyorder)
                self.history[-1][i] = filtered[-1]

        return self.history[-1]
```

#### 真实机器人力信号处理

对于真实机器人系统（UR10 + OnRobot RG2-FT），力信号通过 Modbus TCP 协议从夹爪的左右两个传感器读取，然后取平均值作为最终输出。

**传感器配置**
- 左传感器寄存器：259-264
- 右传感器寄存器：268-273
- 力转换系数：÷ 10.0 → N
- 力矩转换系数：÷ 100.0 → Nm

**裁剪与归一化**

真实世界的力信号采用更宽松的裁剪阈值（传感器量程更大）：

$$
f_i^{\text{clipped}} = \text{clip}(f_i, -100, 100), \quad i \in \{x, y, z\}
$$
$$
\tau_j^{\text{clipped}} = \text{clip}(\tau_j, -10, 10), \quad j \in \{x, y, z\}
$$

然后进行全局均值-方差归一化：

$$
\hat{f}_i = \frac{f_i^{\text{clipped}} - \mu_i}{\sigma_i}, \quad
\hat{\tau}_j = \frac{\tau_j^{\text{clipped}} - \mu_{j+3}}{\sigma_{j+3}}
$$

其中 $\boldsymbol{\mu}, \boldsymbol{\sigma}$ 从数据集的 `force_stats.json` 中加载（默认 $\boldsymbol{\mu} = \boldsymbol{0}$, $\boldsymbol{\sigma} = \boldsymbol{1}$）。

**代码实现** (datasets/dataset.py:49-75):
```python
def normalize_force(force, mean, std):
    force[:3] = np.clip(force[:3], -100, 100)   # fx, fy, fz (牛顿)
    force[3:] = np.clip(force[3:], -10, 10)     # tx, ty, tz (牛·米)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (force - mean) / std
    return normalized
```

### 深度感知优化

深度图作为视觉模态的重要补充，提供场景的几何结构信息。DeMUSE 对深度图进行了专门的滤波和归一化处理，以匹配 VAE 潜在空间的分布特性。

**深度图滤波**

我们提供两种深度图预处理方案：

1. **基础滤波 (filter)**：简单下采样
   ```python
   cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST)
   ```

2. **增强滤波 (filter2)**：包含去噪和归一化的完整流程
   - **深度裁剪**：限制有效量程 $[1000, 5000]$ mm，去除超出范围的噪声
   - **归一化**：$\hat{d} = \text{clip}(d, 1000, 5000) / 5000 \in [0.2, 1]$
   - **中值滤波**：`cv2.medianBlur(depth, 15)`，窗口大小 15×15 像素
   - **下采样**：调整至 $32 \times 32$ 分辨率以匹配 VAE 潜在空间

**代码实现** (datasets/dataset.py:454-459):
```python
def filter2(depth):
    depth = np.clip(depth, 1000, 5000) / 5000
    depth = np.array(depth * 256, dtype=np.uint8)
    depth = cv2.medianBlur(depth, 15)
    return cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST) / 256
```

**深度图在模型中的嵌入**

深度图通过专用的 Patch 嵌入层映射到 $D = 1152$ 维潜在空间：
- 输入分辨率：$32 \times 32$
- Patch 大小：$8 \times 8$（与 RGB 的 $2 \times 2$ 不同，适配深度图的空间特性）
- Patch 数量：$(32/8)^2 = 16$ tokens
- 位置编码：从 RGB 位置编码下采样获得，保持空间对齐

---

## 真实世界数据采集流水线

我们构建了一套针对接触敏感型操作的自动化数据采集系统，集成了 UR10 机械臂、OnRobot RG2-FT 夹爪（带 6 轴力传感器）和 RealSense D435i 相机。系统采用异步架构实现异构传感器的精确时间同步。

### 异构传感器同步

**硬件配置**

| 组件 | 型号/规格 | 关键参数 |
|------|-----------|----------|
| **机械臂** | UR10 | 控制频率 10 Hz，IP: 192.168.1.50 |
| **夹爪/力传感器** | OnRobot RG2-FT | Modbus TCP, IP: 192.168.1.1, 100 Hz |
| **相机** | Intel RealSense D435i | 1280×720@30Hz |

**异步架构设计**

系统采用事件驱动的异步架构，包含三个并发线程：

1. **相机采集线程**：事件驱动，仅在有新帧时处理
   - 维护一个 2 秒滑动窗口的 Ring Buffer
   - 保存多时间戳：`cam_ts_hw`（硬件时间）、`cam_ts_mono`（单调时间）、`cam_ts_recv`（接收时间）
   - 三层去重：帧 ID > 硬件时间戳 > 图像哈希

2. **机器人采集线程**：高频 100 Hz 循环
   - 采集 TCP 位姿、夹爪状态、6 轴力/力矩
   - 使用 `time.monotonic()` 作为时间戳基准
   - 数据存入 2 秒 Ring Buffer

3. **主控线程**：10 Hz 低频循环
   - 以相机帧为基准（驱动整个系统的同步）
   - 从 Ring Buffer 中查找时间最近匹配的机器人状态

**双时间戳对齐机制**

时间对齐的核心是使用 **双时间戳系统**：

| 时间戳 | 来源 | 用途 |
|--------|------|------|
| `cam_ts_hw` | 相机硬件时钟 | 去重检测（5ms 阈值） |
| `cam_ts_mono` | `time.monotonic()` | 与机器人时间对齐 |
| `robot_ts` | `time.monotonic()` | 机器人数据时间戳 |

对齐算法 (data_collector.py:398-400):
```python
ALIGNMENT_THRESHOLD_S = 0.05  # 50ms 对齐阈值
# 从相机缓冲区获取最新帧
cam_entry = self.camera_buffer.get_nearest(newest_cam_ts)
# 从机器人缓冲区查找时间最近的匹配（最大误差 50ms）
robot_entry = self.robot_buffer.get_nearest(cam_ts_mono, max_dt=ALIGNMENT_THRESHOLD_S)
```

**因果顺序保证**

系统严格遵守观测-动作的因果顺序：
$$
o_t \rightarrow a_t \rightarrow o_{t+1}
$$
每个时间步的数据包含：
- 观测 $o_t$：RGB 图像、深度图、力/力矩、TCP 位姿
- 动作 $a_t$：速度命令 $[v_x, v_y, v_z, \omega_x, \omega_y, \omega_z, g]$
- 时间戳：`robot_ts`（观测时间）、`action_ts`（动作执行时间）

### 任务与数据集分布

真实世界数据采集覆盖四个核心灵巧操作任务：

| 任务 | 轨迹数量 | 主要挑战 |
|------|----------|----------|
| **Sweeping** | ~100 | 扫除轨迹跟踪、接触力控制 |
| **Dispense Sanitizer** | ~100 | 挤压瓶的力-位混合控制 |
| **Drawer Organization** | ~100 | 抽屉开闭的精密操作 |
| **Fill Cup** | ~100 | 倾倒动作的动力学建模 |

数据采集采用遥操作方式，通过 Xbox 手柄控制机器人执行任务。每个 episode 包含：
- RGB 图像序列 (1280×720，中心裁剪为正方形)
- 深度图序列（经滤波处理）
- 6 轴力/力矩序列 (100 Hz)
- TCP 位姿和夹爪状态 (10 Hz)
- 速度命令序列 (10 Hz)

**数据格式** (data_collector.py:183-190):
```python
data_dict = {
    'image': [],           # RGB 图像
    'cam_ts_hw': [],       # 相机硬件时间戳
    'cam_ts_mono': [],     # 相机单调时间戳
    'robot_ts': [],        # 机器人时间戳
    'action_ts': [],       # 动作时间戳
    'action': [],          # 速度命令 [vx,vy,vz,wx,wy,wz,gripper]
    'robot_pose': [],      # TCP 位姿 [x,y,z,rx,ry,rz]
    'gripper_state': [],   # 夹爪状态 {0,1}
    'force_torque': [],    # 6 轴力/力矩 [fx,fy,fz,tx,ty,tz]
    'color_space': [],     # 颜色空间 'RGB'
    'frame_id': [],        # 帧序号
}
```

每个 episode 保存为两个文件：
- `episode_XXXX.npz`：压缩的数据数组
- `episode_XXXX_metadata.json`：元数据（时长、相机配置、任务描述）
- `episode_XXXX_vis/`：可视化图像序列

---

## 技术架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    异步数据采集架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  相机线程    │    │  机器人线程  │    │  主控线程    │         │
│  │  (事件驱动)  │    │  (100 Hz)   │    │  (10 Hz)    │         │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤         │
│  │ RealSense   │    │ UR10 + FT   │    │ 数据对齐    │         │
│  │ D435i @30Hz │    │ Sensor @100 │    │ Ring Buffer │         │
│  │             │    │             │    │             │         │
│  │ cam_ts_hw   │    │ robot_ts    │◄───│ cam_ts_mono │         │
│  │ cam_ts_mono │    │ force_torque│    │             │         │
│  │ image       │    │ tcp_pose    │    │ 对齐阈值    │         │
│  │             │    │ gripper     │    │ 50ms        │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
│                    ┌───────▼────────┐                            │
│                    │  Ring Buffer   │                            │
│                    │  (2秒窗口)     │                            │
│                    └───────┬────────┘                            │
│                            │                                    │
│                    ┌───────▼────────┐                            │
│                    │  Episode       │                            │
│                    │  .npz + .json  │                            │
│                    └────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策总结

| 设计决策 | 理由 | 实现位置 |
|----------|------|----------|
| **坐标系变换** | EE 坐标系力信号与机器人姿态对齐 | collect_metaworld_data_raw.py:215-250 |
| **SG 滤波器** | 平滑仿真中的接触力峰值 | collect_metaworld_data_raw.py:255-315 |
| **力滤波窗口=5** | 平衡平滑效果与响应速度 | collect_metaworld_data_raw.py:148 |
| **仿真力裁剪 ±20N** | 去除仿真碰撞异常峰值 | collect_metaworld_data_raw.py:150 |
| **真实力裁剪 ±100N** | 传感器量程限制 | dataset.py:68 |
| **中值滤波 kernel=15** | 去除深度图椒盐噪声 | dataset.py:458 |
| **50ms 对齐阈值** | 平衡同步精度与数据利用率 | data_collector.py:53 |
| **cam_ts_mono 基准** | 与机器人使用同一时钟域 | data_collector.py:158 |

---

## 附录：坐标系变换数学推导

设世界坐标系为 $\mathcal{W}$，EE 坐标系为 $\mathcal{E}$。EE 在世界坐标系中的姿态由旋转矩阵 $\boldsymbol{R}_{\mathcal{E}}^{\mathcal{W}}$ 表示。

对于力向量 $\boldsymbol{f}_{\mathcal{W}} \in \mathbb{R}^3$（在世界坐标系中表示），其在 EE 坐标系中的表示为：

$$
\boldsymbol{f}_{\mathcal{E}} = \boldsymbol{R}_{\mathcal{W}}^{\mathcal{E}} \cdot \boldsymbol{f}_{\mathcal{W}}
$$

由于旋转矩阵的正交性 $\boldsymbol{R}_{\mathcal{W}}^{\mathcal{E}} = (\boldsymbol{R}_{\mathcal{E}}^{\mathcal{W}})^{-1} = (\boldsymbol{R}_{\mathcal{E}}^{\mathcal{W}})^\top$，我们有：

$$
\boldsymbol{f}_{\mathcal{E}} = (\boldsymbol{R}_{\mathcal{E}}^{\mathcal{W}})^\top \cdot \boldsymbol{f}_{\mathcal{W}}
$$

力矩遵循相同的变换规则（因为力矩也是伪向量，在纯旋转变换下与向量一致）：

$$
\boldsymbol{\tau}_{\mathcal{E}} = (\boldsymbol{R}_{\mathcal{E}}^{\mathcal{W}})^\top \cdot \boldsymbol{\tau}_{\mathcal{W}}
$$

---

## 参考文献

1. OnRobot. "RG2-FT Technical Documentation." 2023.
2. Intel. "RealSense D435i Product Specification." 2022.
3. Universal Robots. "UR10 Technical Documentation." 2021.
4. Savitzky, A., & Golay, M.J.E. "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." Analytical Chemistry, 1964.
