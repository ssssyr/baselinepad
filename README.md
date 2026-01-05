# Prediction with Action (PAD)

> 基于 Diffusion Transformer 的视觉策略学习，支持多模态输入（RGB图像、动作、深度、文本、力/力矩）与 Mixture of Experts 架构

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-red)](https://neurips.cc/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5.1-orange)](https://pytorch.org/)

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目架构](#项目架构)
- [模型架构详解](#模型架构详解)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [常见问题](#常见问题)

---

## 项目概述

**Prediction with Action (PAD)** 是一个端到端的视觉运动策略学习框架，通过联合去噪过程同时预测未来视觉状态和相应的机器人动作。该方法基于扩散模型和 Diffusion Transformer (DiT)，支持多模态条件输入，并在 MetaWorld 和 BridgeData 等机器人学习基准上取得了优秀性能。

### 核心思想

传统方法将视觉预测和动作预测分离处理，本项目的创新点在于：

1. **联合预测**: 在同一个扩散过程中同时预测未来帧和动作序列
2. **多模态融合**: 支持 RGB、深度、力/力矩、文本指令等多种条件输入
3. **Mixture of Experts**: 使用稀疏专家混合提高模型容量和效率
4. **时序感知**: Horizon-aware 权重自适应，支持可变长度的未来预测

---

## 核心特性

| 特性 | 描述 |
|------|------|
| **多模态输入** | RGB图像、动作、深度图、文本指令(CLIP)、6D力/力矩 |
| **动作预测** | 同时预测未来视觉帧和对应动作序列 |
| **MoE 架构** | 稀疏专家混合，支持模态感知路由 |
| **力传感器融合** | 支持接触力/力矩条件输入（6D: fx,fy,fz,tx,ty,tz） |
| **多环境支持** | MetaWorld、BridgeData、真实机器人数据 |
| **分布式训练** | 支持多 GPU 加速，云端 A100 训练脚本 |
| **WandB 集成** | 实验追踪和可视化 |

---

## 项目架构

```
prediction_with_action/
|
+-- configs/                      # 配置文件目录
│   +-- metaworld_4d.yaml         # MetaWorld 4-DOF 配置
│   +-- bridge_vision.yaml        # BridgeData 配置
│   +-- metaworld_4d_cotrain.yaml # 联合训练配置
│
+-- datasets/                     # 数据处理模块
│   +-- dataset.py                # RobotDataset 主数据集类
│   +-- collect_metaworld_data_raw.py    # MetaWorld 数据采集
│   +-- convert_real_robot_data.py      # 真实机器人数据转换
│   +-- extract_features_complete.py     # 特征提取
│
+-- diffusion/                    # 扩散模型实现
│   +-- gaussian_diffusion.py     # 高斯扩散过程
│   +-- resample.py               # 重采样工具
│
+-- evaluation/                   # 评估模块
│   +-- agent.py                  # Diffusion Agent
│   +-- run_cfg.py                # 评估配置
│
+-- metaworld/                    # MetaWorld 环境集成
│
+-- mujoco/                       # MuJoCo 仿真器
│
+-- models.py                     # DiT 核心模型
+-- moe_blocks.py                 # Mixture of Experts 实现
+-- train_robot.py                # 主训练脚本
+-- run_metaworld.py              # MetaWorld 评估脚本
+-- config_loader.py              # 配置加载器
```

---

## 模型架构详解

### 整体架构图

```
                    输入层 (多模态融合)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    RGB 图像          动作序列          力/力矩
  (VAE latent)      (action_embed)     (force_embed)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    +--------------+
                    │  Patch Embed │
                    +--------------+
                           │
                    +--------------+
                    │ Positional   │
                    │   Embedding  │
                    +--------------+
                           │
        ┌──────────────────┼──────────────────┐
        │          Timestep Embedding         │
        │           Text/Label Embed          │
        └──────────────────┼──────────────────┘
                           │
              ┌────────────┴────────────┐
              │   DiT Transformer Blocks  │
              │   (with optional MoE)     │
              │  - Self-Attention         │
              │  - adaLN-Zero Modulation  │
              │  - Sparse MoE FFN         │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │      Final Layer        │
              │  - RGB Prediction Head  │
              │  - Action Prediction    │
              │  - Depth Prediction     │
              └────────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    未来帧预测          动作预测           深度预测
  (Future Frames)    (Actions)          (Depth)
```

### 1. 多模态输入嵌入层

模型支持多种输入模态，每种模态通过独立的嵌入层转换为统一的 token 表示：

#### 1.1 RGB 图像嵌入
```python
# 输入: VAE 潜在特征 (B, 4, 32, 32)
x = self.x_embedder(x) + self.pos_embed  # → (B, num_patches, hidden_size)
```
- 使用 `PatchEmbed` 将图像分割为 patches
- 添加固定的 2D sinusoidal 位置编码

#### 1.2 动作嵌入
```python
# 输入: (B, action_steps, action_dim)
a = self.a_embedder(noised_action) + self.a_pos_embed
```
- 线性投影到 hidden_size
- 支持 sin-cos 位置编码或可学习位置编码

#### 1.3 力/力矩嵌入
```python
# 输入: (B, 6) → [fx, fy, fz, tx, ty, tz]
f = self.force_embedder(force_cond) + self.f_pos_embed
```
- 6D 力/力矩数据通过线性层投影

#### 1.4 深度嵌入
```python
# 输入: (B, 1, 32, 32)
d = self.d_embedder(depth) + self.d_pos_embed
```
- 使用独立的 PatchEmbed 处理深度图

#### 1.5 文本/标签嵌入
```python
# 输入: CLIP 嵌入 (B, 512) 或 类别标签 (B,)
y = self.y_embedder(y)
```

### 2. 模态 Token 序列组织

所有模态的 token 被组织成一个统一的序列：

```
Token 序列结构:
[RGB Patches][Action Tokens][Force Tokens][Depth Patches]
    ↑              ↑                ↑              ↑
   num_patches   action_steps        1        d_num_patches

例如: [1024 RGB tokens][3 action tokens][1 force token][64 depth tokens]
```

每个 token 都有对应的 `modality_id`，用于 MoE 模态感知路由：
- `modality_id = 0`: RGB tokens
- `modality_id = 1`: Action tokens
- `modality_id = 2`: Force tokens (如启用)
- `modality_id = 3`: Depth tokens (如启用)

### 3. DiT Block (Transformer 层)

每个 DiT Block 包含以下组件：

#### 3.1 AdaLN-Zero 调制
```python
def forward(self, x, c, modality_ids=None):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=1)

    # 调制后的注意力
    x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

    # 调制后的 MLP/MoE
    x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
```
- 使用 timestep 和条件信息 `c` 生成 6 组调制参数
- 分别调制 attention 和 MLP 层的输入和输出

#### 3.2 多头自注意力
```python
class Attention(nn.Module):
    # 标准的 Multi-Head Self-Attention
    # 支持 fused attention 实现 (A100 加速)
    # 可选的 attention mask 控制模态间交互
```

#### 3.3 MoE 前馈网络（可选）
当 `use_moe=True` 时，替换标准 MLP 为 Sparse MoE：

```python
class DiTBlock:
    if use_moe:
        self.mlp = SparseMoeBlock(
            embed_dim=hidden_size,
            num_experts=num_experts,      # 专家数量 (默认 4)
            num_experts_per_tok=top_k,     # 每个 token 选择的专家数 (默认 2)
            n_shared_experts=shared_experts, # 共享专家 (默认 4)
            ...
        )
```

### 4. Sparse Mixture of Experts (MoE) 架构

#### 4.1 MoE Gate (路由网络)
```python
class MoEGate(nn.Module):
    def forward(self, hidden_states, modality_ids=None):
        # 计算 logits
        logits = F.linear(hidden_states, self.weight)

        # 模态感知偏置 (可选)
        if use_modality_bias:
            logits += modality_bias[modality_ids]

        # Top-K 选择
        topk_weight, topk_idx = torch.topk(logits.softmax(dim=-1), k=top_k)

        # 辅助损失 (负载均衡)
        aux_loss = (pi * fi).sum() * alpha
```

**模态感知路由**: 不同模态的 token 可以有不同的专家偏好，通过 `modality_bias` 实现。

#### 4.2 SparseMoeBlock
```python
class SparseMoeBlock(nn.Module):
    def forward(self, hidden_states, modality_ids=None):
        # 1. Gate 选择专家
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, modality_ids)

        # 2. 分发给选中的专家
        for expert_idx, expert in enumerate(self.experts):
            mask = (topk_idx == expert_idx)
            routed[mask] = expert(hidden_states[mask])

        # 3. 加权求和
        output = (routed * topk_weight).sum(dim=1)

        # 4. 添加共享专家输出
        output = output + self.shared_experts(hidden_states)

        return output
```

**关键特性**:
- **稀疏激活**: 每个 token 只使用 top-k 个专家
- **共享专家**: 所有 token 都经过共享专家，保持稳定性
- **辅助损失**: 鼓励专家负载均衡

#### 4.3 模态感知路由统计
```python
# 记录每个模态的路由统计 (用于监控)
stats = {
    "action_hit_rate": action 命中 expert0 的比例,
    "action_coverage": action 使用的专家比例,
    "rgb_coverage": RGB tokens 使用的专家比例,
    "depth_coverage": Depth tokens 使用的专家比例
}
```

### 5. 最终输出层

```python
class FinalLayer(nn.Module):
    def forward(self, x, c):
        # RGB 预测
        rgb = modulate(self.norm_final(rgb_tokens), shift, scale)
        rgb = self.linear(rgb)  # → (B, num_patches, patch_size^2 * out_channels)

        # 动作预测
        if use_action:
            a = modulate(self.a_norm_final(action_tokens), shift, scale)
            a = self.a_linear(a)  # → (B, action_steps, action_dim * 2)

        # 深度预测
        if use_depth:
            d = modulate(self.d_norm_final(depth_tokens), shift, scale)
            d = self.d_linear(d)  # → (B, d_patches, d_patch_size^2 * horizon)
```

### 6. 前向传播流程

```python
def forward(self, x, t, y, x_cond=None, action_cond=None,
            noised_action=None, force_cond=None, depth_cond=None, noised_depth=None):

    # 1. 拼接条件图像（如果有）
    if x_cond is not None:
        x = torch.cat([x, x_cond], dim=1)

    # 2. 各模态嵌入
    x = self.x_embedder(x) + self.pos_embed           # RGB
    a = self.a_embedder(noised_action) + self.a_pos_embed  # Action
    f = self.force_embedder(force_cond) + self.f_pos_embed  # Force
    d = self.d_embedder(noised_depth) + self.d_pos_embed    # Depth

    # 3. 拼接所有 token
    x = torch.cat([x, a, f, d], dim=1)

    # 4. 生成 modality_ids (用于 MoE)
    modality_ids = ...  # [0,...,0, 1,...,1, 2,...,2, 3,...,3]

    # 5. Timestep 和 条件嵌入
    c = self.t_embedder(t) + self.y_embedder(y)

    # 6. 通过 DiT Blocks
    for block in self.blocks:
        x = block(x, c, modality_ids)

    # 7. 解码输出
    rgb, action, depth = self.final_layer(x, c)

    return rgb, action, depth
```

### 7. 模型变体

| 模型 | Hidden Size | Depth | Heads | Params |
|------|-------------|-------|-------|--------|
| DiT-S | 384 | 12 | 6 | ~33M |
| DiT-B | 768 | 12 | 12 | ~131M |
| DiT-L | 1152 | 24 | 16 | ~458M |
| DiT-XL | 1152 | 28 | 16 | ~675M |

### 8. MoE 配置示例

```yaml
moe:
  use_moe: true              # 启用 MoE
  num_experts: 4             # 专家数量
  moe_top_k: 2               # 每个 token 选择 2 个专家
  aux_loss_weight: 0.01      # 辅助损失权重
  moe_start_layer: 14        # 从第 14 层开始使用 MoE
  moe_shared_experts: 4      # 共享专家数量
  use_modality_bias: false   # 模态感知路由
```

---

## 安装说明

### 环境要求

- Python 3.9 - 3.12
- CUDA 12.1+
- 16GB+ GPU 显存（推荐 A100 40GB）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd prediction_with_action
```

2. **创建 Conda 环境**
```bash
conda create -n pad python=3.10
conda activate pad
```

3. **安装依赖**
```bash
pip install -r requirements-cloud.txt
```

### 依赖版本

| 依赖 | 版本 |
|------|------|
| PyTorch | 2.5.1+cu121 |
| Diffusers | 0.25.0 |
| Transformers | 4.36.2 |
| Timm | 0.9.12 |
| MuJoCo-py | 2.1.2.14 |
| WandB | 0.15.12 |

---

## 快速开始

### 1. 数据准备

**MetaWorld 数据**：
```bash
cd datasets
python collect_metaworld_data_raw.py
```

**真实机器人数据**：
```bash
python convert_real_robot_data.py \
    --input /path/to/newdata \
    --output /path/to/converted \
    --instruction "夹起魔方放到盘子里"
```

### 2. 训练模型

```bash
python train_robot.py --config configs/metaworld_4d.yaml
```

### 3. 评估模型

```bash
python run_metaworld.py
```

---

## 配置说明

配置文件采用 YAML 格式，主要包含以下部分：

### training - 训练设置

```yaml
training:
  feature_path: "/path/to/dataset"          # 数据集路径
  results_dir: "results_metaworld_4d"       # 输出目录
  model: "DiT-XL/2"                         # 模型架构
  image_size: 256                           # 图像分辨率
  predict_horizon: 3                        # 预测未来帧数
  global_batch_size: 64                     # 全局批次大小
  learning_rate: 5e-5                       # 学习率
```

### components - 组件设置

```yaml
components:
  # VAE 和文本
  vae_path: "/path/to/sd-vae-ft-mse"
  clip_path: "/path/to/clip-vit-base-patch32"

  # 多模态开关
  text_cond: true           # 文本条件
  use_depth: false          # 深度条件
  use_force: true           # 力/力矩条件

  # 动作设置
  action_steps: 3
  action_dim: 4             # MetaWorld: 4-DOF
  action_condition: true
```

### moe - MoE 设置

```yaml
moe:
  use_moe: true
  num_experts: 4
  moe_top_k: 2
  aux_loss_weight: 0.01
```

---

## 数据准备

### 数据格式

```
dataset_path/
├── dataset_info.json           # 数据集元数据
├── force_stats.json            # 力统计信息
├── episode0000000/
│   ├── color_wrist_1_0000.npy  # VAE 潜在特征
│   ├── color_wrist_1_0001.npy
│   └── text_clip.npy           # CLIP 文本嵌入
└── episode0000001/
```

### MetaWorld 数据采集

```bash
python datasets/collect_metaworld_data_raw.py

# 输出：50 个任务 × 50 条轨迹
# 指令示例: "press the button", "open the door"
```

---

## 训练

### 基础训练

```bash
python train_robot.py --config configs/metaworld_4d.yaml
```

### 多 GPU 训练

```bash
bash start_train_cloud_a100.sh
```

---

## 评估

### MetaWorld 评估

```bash
python run_metaworld.py --task button-press-v2
```

### 评估指标

- Success Rate（任务成功率）
- Action Loss（动作预测损失）
- Visual FID（视觉预测质量）

---

## 常见问题

### Q1: 中文文本指令支持吗？

CLIP 对中文有一定支持，但主要在英文上训练。建议训练和推理使用相同语言。

### Q2: 如何添加新的输入模态？

在 `models.py` 中添加新的嵌入层，并在 `dataset.py` 中加载对应数据。

### Q3: 力信号如何归一化？

使用 `force_stats.json` 中的均值和标准差：
```python
force_normalized = (force - mean) / (std + 1e-8)
```

### Q4: 内存不足怎么办？

- 减小 `global_batch_size`
- 减小 `predict_horizon`
- 使用梯度累积

---

## 引用

如果本项目对你有帮助，请引用：

```bibtex
@inproceedings{pad2024,
  title={Prediction with Action: A Unified Approach to Visual Policy Learning},
  booktitle={NeurIPS},
  year={2024}
}
```
