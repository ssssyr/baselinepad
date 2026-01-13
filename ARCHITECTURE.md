# Uni-Embodied: 项目架构说明

> 本文档提供完整的项目架构概览，用于绘制系统架构图

---

## 1. 系统总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Uni-Embodied Framework                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐             │
│  │   Data       │      │    Model     │      │  Deployment  │             │
│  │  Pipeline    │ ───▶ │  Architecture │ ───▶ │    Agent     │             │
│  └──────────────┘      └──────────────┘      └──────────────┘             │
│         │                     │                      │                       │
│         ▼                     ▼                      ▼                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐             │
│  │  Collection │      │   Training    │      │  Evaluation  │             │
│  │             │      │     Loop      │      │              │             │
│  └──────────────┘      └──────────────┘      └──────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据流架构 (Data Pipeline)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Multi-Modal Input                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │   RGB    │    │  Depth   │    │  Force   │    │  Action  │           │
│  │  256×256  │    │  (opt)   │    │   6-DOF  │    │   4-DOF  │           │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘           │
│       │                │                │                │                   │
│       ▼                ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                    Tokenization Layer                            │     │
│  ├─────────────────────────────────────────────────────────────────┤     │
│  │  RGB: PatchEmbed (16×16 patches) → 256 tokens                     │     │
│  │  Action: Linear Embedding → 3 tokens                               │     │
│  │  Force: Linear Embedding → 1 token                                 │     │
│  │  Depth: PatchEmbed (opt) → N tokens                                 │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                      DiT Backbone                                │     │
│  │  ┌─────────────────────────────────────────────────────────┐   │     │
│  │  │              DiTBlock × 28 (XL/2)                        │   │     │
│  │  │  ┌─────────────────────────────────────────────────────┐ │   │     │
│  │  │  │  1. Expert AdaLN (per-modality LayerNorm)          │ │   │     │
│  │  │  │     ├─ norm1_experts[RGB/Action/Force/Depth]       │ │   │     │
│  │  │  │     └─ norm2_experts[RGB/Action/Force/Depth]       │ │   │     │
│  │  │  │  2. adaLN Modulation (shared)                       │ │   │     │
│  │  │  │     └─ 6×hidden (shift/scale/gate × 2)             │ │   │     │
│  │  │  │  3. Multi-Head Self-Attention                      │ │   │     │
│  │  │  │  4. MoE MLP (layers 14-27)                          │ │   │     │
│  │  │  │     ├─ 4 Routed Experts (top-2)                    │ │   │     │
│  │  │  │     └─ 4 Shared Experts                            │ │   │     │
│  │  │  └─────────────────────────────────────────────────────┘ │   │     │
│  │  └─────────────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                      FinalLayer                                  │     │
│  │  ├─ RGB Head: norm_final → adaLN → linear                     │     │
│  │  ├─ Action Head: a_norm_final → a_adaLN → a_linear             │     │
│  │  └─ Depth Head: d_norm_final → d_adaLN → d_linear (opt)         │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                   Multi-Modal Output                             │     │
│  │  • RGB: (B, 4*H, H', W') - predicted frames                     │     │
│  │  • Action: (B, steps, action_dim*2) - mean + std               │     │
│  │  • Depth: (B, H', W') - predicted depth (opt)                   │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 Expert AdaLN (关键创新)

```python
# models.py - 核心机制
def apply_expert_ln(x, modality_ids, experts):
    """
    按模态应用专家 LayerNorm

    输入:
        x: (B, N, D) - tokens
        modality_ids: (B, N) - [0=RGB, 1=Action, 2=Force, 3=Depth]
        experts: ModuleList of M LayerNorms

    输出:
        (B, N, D) - 每个模态使用其专属的 LayerNorm
    """
    for m in range(num_modalities):
        mask = (modality_ids == m)
        if mask.any():
            output[mask] = experts[m](x[mask])  # 模态专属归一化
    return output
```

**架构图示意**：
```
┌─────────────────────────────────────────────────────────────────┐
│                     DiTBlock (单层)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Tokens (B, N, D)                                         │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │        Expert AdaLN (norm1)                         │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│       │
│  │  │ RGB LN  │  │Act LN   │  │Force LN │  │Depth LN ││       │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│       │
│  │       │            │            │            │        │       │
│  │       └────────────┴────────────┴────────────┘        │       │
│  │                    ▼                                 │       │
│  │            Modality-Specific Norm                   │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │        adaLN Modulation (共享)                      │       │
│  │  shift_msa, scale_msa, gate_msa,                    │       │
│  │  shift_mlp, scale_mlp, gate_mlp                     │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           Multi-Head Self-Attention                 │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │        Expert AdaLN (norm2)                         │       │
│  │  (same structure as norm1)                           │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │                  MoE MLP (可选)                       │       │
│  │  ┌─────────────────────────────────────────┐        │       │
│  │  │  MoE Router → Top-2 Expert Selection    │        │       │
│  │  │  ├─ Expert 0  ├─ Expert 1               │        │       │
│  │  │  ├─ Expert 2  ├─ Expert 3               │        │       │
│  │  │  └─ 4 Shared Experts                   │        │       │
│  │  └─────────────────────────────────────────┘        │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│                    Output Tokens (B, N, D)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 MoE 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    SparseMoeBlock                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: (B, N, D) + modality_ids                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              MoE Router (MoEGate)                    │       │
│  │  • Logits = Linear(hidden → num_experts)            │       │
│  │  • + modality_bias[modality_id] (if enabled)        │       │
│  │  • Top-k selection (k=2)                             │       │
│  │  • Output: top_k_idx, top_k_weight, aux_loss        │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Expert Execution                         │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│       │
│  │  │Expert 0 │  │Expert 1 │  │Expert 2 │  │Expert 3 ││       │
│  │  │  GELU   │  │  GELU   │  │  GELU   │  │  GELU   ││       │
│  │  │FC1+FC2 │  │FC1+FC2 │  │FC1+FC2 │  │FC1+FC2 ││       │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│       │
│  │       │            │            │            │        │       │
│  │       └────────────┴────────────┴────────────┘        │       │
│  │                        │                               │       │
│  │                        ▼                               │       │
│  │              Weighted Sum                             │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │            Shared Experts (Dense Path)              │       │
│  │            GELU MLP (4× width for capacity)         │       │
│  └────────────────────────────┬────────────────────────┘       │
│                             │                                  │
│                             ▼                                  │
│                    Output: (B, N, D)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Token 布局 (Token Layout)

``┌─────────────────────────────────────────────────────────────────┐
│                   Token Sequence Layout                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Position:  [0:256)  [256:259)  [259:260)                      │
│             RGB       Action      Force                         │
│             │         │           │                             │
│             ▼         ▼           ▼                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  modality_ids = [0,0,...,0, 1,1,1, 2]                   │    │
│  │  共 260 tokens:                                           │    │
│  │    • 256 RGB tokens (16×16 patches)                       │    │
│  │    • 3 Action tokens (temporal steps)                     │    │
│  │    • 1 Force token                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  start_idx = [0,   256,     259,     260]                     │
│  end_idx   = [256, 259,     260,     260]                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop (train_robot.py)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 初始化                                                        │
│     ├─ 加载配置 (YAML + CLI)                                     │
│     ├─ 创建模型 (DiT-XL/2)                                       │
│     ├─ 加载预训练权重 (rgb_init)                                  │
│     ├─ 创建数据集 (RobotDataset)                                  │
│     └─ 初始化优化器 (AdamW)                                       │
│                                                                 │
│  2. 训练循环 (for epoch in epochs)                               │
│     ├─ 数据加载 (DataLoader)                                      │
│     │   ├─ x_cond: 当前帧                                        │
│     │   ├─ x: 未来帧 (H=3)                                       │
│     │   ├─ action: 动作序列                                       │
│     │   ├─ force: 力/力矩                                         │
│     │   └─ y: 任务指令                                           │
│     │                                                             │
│     ├─ VAE 编码 (x → latent)                                      │
│     │                                                             │
│     ├─ 扩散前向                                                    │
│     │   ├─ 采样 timesteps                                        │
│     │   ├─ 添加噪声 ε ~ N(0, I)                                   │
│     │   └─ 模型预测 ε_θ(x_t, t, y)                                │
│     │                                                             │
│     ├─ 损失计算                                                    │
│     │   ├─ loss_image = MSE(ε, ε_θ)                               │
│     │   ├─ loss_action = MSE(action, action_pred)                 │
│     │   └─ loss_total = loss_image + λ·loss_action + aux_loss    │
│     │                                                             │
│     ├─ 反向传播与优化                                               │
│     │   ├─ 梯度累积 (记录各模态梯度范数)                            │
│     │   ├─ optimizer.step()                                       │
│     │   └─ update_ema()                                           │
│     │                                                             │
│     └─ 日志记录 (每 log_every 步)                                  │
│         ├─ 损失值                                                │
│         ├─ 梯度范数 (按模态)                                        │
│         ├─ MoE 路由统计                                           │
│         └─ WandB 上传                                              │
│                                                                 │
│  3. 评估 (每 eval_every 步)                                       │
│     ├─ DDPM 采样 (250 steps)                                      │
│     ├─ 计算成功率/MSE                                             │
│     └─ 保存最佳模型                                               │
│                                                                 │
│  4. 检查点保存 (每 ckpt_every 步)                                  │
│     ├─ 模型权重                                                   │
│     ├─ 优化器状态                                                 │
│     └─ 学习率调度器状态                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 配置系统

### 5.1 配置层级结构

```
configs/
├── metaworld_4d.yaml          # MetaWorld 4-DOF 配置
│   ├─ training:              # 训练设置
│   │   ├─ learning_rate: 5e-5
│   │   ├─ batch_size: 64
│   │   └─ epochs: 1500
│   │
│   ├─ components:            # 组件设置
│   │   ├─ model: DiT-XL/2
│   │   ├─ predict_horizon: 3
│   │   ├─ action_steps: 3
│   │   ├─ use_force: true
│   │   └─ use_depth: false
│   │
│   └─ moe:                   # MoE 设置
│       ├─ use_moe: true
│       ├─ num_experts: 4
│       ├─ moe_top_k: 2
│       ├─ moe_start_layer: 14
│       └─ aux_loss_weight: 0.01
│
├── bridge_vision.yaml        # Bridge 环境
└── action_prediction.yaml    # 动作预测设置
```

### 5.2 配置加载流程

```
CLI Arguments
       │
       ▼
┌──────────────────┐
│  config_loader   │
│  .load_config()  │
└────────┬─────────┘
         │
         ├─ 加载 YAML
         │
         ├─ 合并 CLI 参数
         │
         ├─ 参数验证
         │
         └─ 迭平嵌套结构
         │
         ▼
┌──────────────────┐
│     args         │
│  (Namespace)     │
└──────────────────┘
```

---

## 6. 文件依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                      核心文件依赖图                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  train_robot.py                                                 │
│       │                                                         │
│       ├──────────────┬──────────────┬──────────────┐            │
│       ▼              ▼              ▼              ▼            │
│  models.py    config_loader.py  diffusion/  datasets/           │
│       │              │              │            │             │
│       │              │              │            ▼             │
│       │              │              │      dataset.py           │
│       │              │              │            │             │
│       ▼              │              │            ├─────────────┐│
│  moe_blocks.py       │              │            │collect_*.py ││
│       │              │              │            └─────────────┘│
│       └──────────────┴──────────────┴──────────────────────────┘│
│                                                                 │
│  evaluation/                                                     │
│       │                                                         │
│       ├─ agent.py (使用 models.py)                             │
│       └─ run_cfg.py                                             │
│                                                                 │
│  real/                                                          │
│       │                                                         │
│       ├─ main.py (部署入口)                                     │
│       ├─ agent.py (DiffusionAgent)                              │
│       ├─ hardware/ (硬件抽象)                                   │
│       │   ├─ camera_manager.py                                  │
│       │   ├─ ur10_manager.py                                    │
│       │   └─ gripper_controller.py                              │
│       └─ scripts/                                               │
│           └─ data_collector.py                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键参数说明

### 7.1 模型规模

| 模型 | Depth | Hidden | Heads | 参数量 |
|------|-------|--------|-------|--------|
| DiT-S/8 | 12 | 384 | 6 | ~30M |
| DiT-B/4 | 12 | 768 | 12 | ~130M |
| DiT-L/2 | 24 | 1024 | 16 | ~458M |
| **DiT-XL/2** | **28** | **1152** | **16** | **~680M** |

### 7.2 AdaLN 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adamn` | False | 启用 per-modality adaptive LayerNorm |
| `num_modalities` | 2-4 | 模态数量 (RGB+Action+Force+Depth) |
| `hidden_size` | 1152 | 隐藏层维度 |
| `eps` | 1e-6 | LayerNorm 稳定性参数 |

### 7.3 MoE 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_moe` | True | 启用 MoE |
| `num_experts` | 4 | 路由专家数量 |
| `moe_top_k` | 2 | 每个 token 选择的专家数 |
| `moe_start_layer` | 14 | 开始使用 MoE 的层 |
| `shared_experts` | 4 | 共享专家数量 |
| `aux_loss_weight` | 0.01 | 负载均衡损失权重 |

---

## 8. 绘制架构图建议

### 8.1 总体架构图

推荐工具：
- **draw.io**: 免费，支持多种导出格式
- **Figma**: 在线协作，美观
- **TikZ**: LaTeX 原生，适合论文

关键要素：
1. **多模态输入** (RGB, Force, Action, Depth)
2. **Tokenization** (Patch Embedding, Linear Embedding)
3. **DiT Backbone** (28层 DiTBlock)
4. **Expert AdaLN** (每个 Block 内)
5. **MoE** (后14层)
6. **多头输出** (RGB, Action, Depth)

### 8.2 细节模块图

建议绘制：
1. **DiTBlock 内部结构** (参考本文档第3.1节)
2. **Expert AdaLN 机制** (模态专属 LayerNorm)
3. **MoE 路由架构** (Top-K 选择)
4. **训练流程图** (数据流 + 梯度流)

---

## 9. 绘图元素颜色建议

| 模态 | 推荐颜色 | RGB值 |
|------|----------|-------|
| RGB | 蓝色 | #3498db |
| Action | 绿色 | #2ecc71 |
| Force | 橙色 | #e67e22 |
| Depth | 紫色 | #9b59b6 |
| 共享模块 | 灰色 | #95a5a6 |

---

## 10. LaTeX TikZ 模板

```latex
\tikzset{
    modality/.style={rectangle, draw, rounded corners, minimum height=0.8cm},
    block/.style={rectangle, draw, minimum width=2cm, minimum height=1cm},
    arrow/.style={->, thick}
}

\begin{tikzpicture}[node distance=1.5cm]
    % Inputs
    \node[modality, fill=blue!20] (rgb) {RGB};
    \node[modality, fill=green!20, right=of rgb] (action) {Action};
    \node[modality, fill=orange!20, right=of action] (force) {Force};

    % Tokenization
    \node[block, below=1cm of rgb] (tok) {Tokenization};

    % DiT
    \node[block, below=1cm of tok] (dit) {DiT-XL/2};

    % Output
    \node[modality, fill=blue!20, below=1cm of dit] (out) {Multi-Modal Output};

    % Arrows
    \draw[arrow] (rgb) -- (tok);
    \draw[arrow] (action) -- (tok);
    \draw[arrow] (force) -- (tok);
    \draw[arrow] (tok) -- (dit);
    \draw[arrow] (dit) -- (out);
\end{tikzpicture}
```

---

## 附录: 文件清单

### 核心文件
- `models.py` (779行) - DiT模型定义
- `moe_blocks.py` (322行) - MoE实现
- `train_robot.py` (948行) - 训练脚本
- `diffusion/` - 扩散模型组件

### 数据处理
- `datasets/dataset.py` - 数据集类
- `datasets/collect_metaworld_data_raw.py` - MetaWorld数据收集
- `datasets/extract_features_complete.py` - 特征提取

### 部署
- `real/main.py` - 实机部署入口
- `real/agent.py` - DiffusionAgent
- `evaluation/agent.py` - 评估脚本

### 配置
- `config_loader.py` - 配置加载器
- `configs/metaworld_4d.yaml` - MetaWorld配置
