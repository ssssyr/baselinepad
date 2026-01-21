# Uni-Embodied DiT 模型架构详解

## 目录
- [1. 论文描述与代码实现对比](#1-论文描述与代码实现对比)
- [2. 整体架构详解](#2-整体架构详解)
- [3. 模块化组件分析](#3-模块化组件分析)
- [4. 训练流程详解](#4-训练流程详解)
- [5. 问题与修正建议](#5-问题与修正建议)
- [6. 代码引用索引](#6-代码引用索引)

---

## 1. 论文描述与代码实现对比

### 1.1 论文原文分析

#### 原文段落 1: System Overview

> "Uni-Embodied DiT comprises three primary components:
> - **Unified Tokenizer**: projects heterogeneous observations into a homogeneous token sequence. The input stream encompasses the noisy joint variables $\mathbf{w}_k$ (comprising the noisy states of future visual sequences $\mathbf{z}_{t+1:t+H}$ and action chunks $\mathbf{a}_{t:t+K-1}$) along with the omni-modal conditional context $\mathbf{c}_t$ (including current latent variables $z_t$, proprioceptive states $s_t$, depth maps $\mathcal{D}_t$, and 6-axis force feedback $f_t$).
>
> - **Multimodal Backbone**: Consisting of $L$ stacked Transformer Blocks
>
> - **Joint Prediction Head**: predicts the joint noise estimate $\hat{\boldsymbol{\epsilon}}$"

#### ❌ 问题分析

| 论文描述 | 实际代码 | 问题 |
|---------|---------|------|
| "Unified Tokenizer" 作为一个组件 | **不存在独立模块** | tokenization是分散在多个embedder中 |
| "noisy joint variables $\mathbf{w}_k$" | **没有这个变量** | 实际是 `z`, `z_a`, `z_d` 三个独立噪声 |
| "current latent variables $z_t$" | `x_cond` | 符号不一致 |
| "proprioceptive states $s_t$" | `action_cond` 或 `state` | 符号不一致 |
| "Joint Prediction Head" | `FinalLayer` | 名称不一致 |
| 单一输出 $\hat{\boldsymbol{\epsilon}}$ | **三个独立输出** | 缺少多输出说明 |

---

#### 原文段落 2: Block Design

> "Each block follows a hierarchical logic of **'Modulate--Synthesize--Scale'**:
> 1. **Modulate**: AdaMN assigns dedicated modulation trajectories to tokens of different modalities
> 2. **Synthesize**: All-to-All self-attention layer
> 3. **Scale**: Sparse MoE expansion layer"

#### ✅ 基本正确，但细节不够

论文描述的三阶段流程与代码基本一致，但缺少关键细节：

```python
# models.py:283-314 - DiTBlock.forward()

def forward(self, x, c, modality_ids=None):
    # ========== Stage 1: Modulate ==========
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=1)

    # AdaMN: 按模态应用不同的LayerNorm
    if self.use_adamn:
        normed_x = apply_expert_ln(x, modality_ids, self.norm1_experts)
    else:
        normed_x = self.norm1(x)

    # ========== Stage 2: Synthesize (Attention) ==========
    x = x + gate_msa * self.attn(modulate(normed_x, shift_msa, scale_msa))

    # AdaMN for MLP input
    if self.use_adamn:
        normed_x = apply_expert_ln(x, modality_ids, self.norm2_experts)
    else:
        normed_x = self.norm2(x)

    # ========== Stage 3: Scale (MoE or Dense MLP) ==========
    mlp_input = modulate(normed_x, shift_mlp, scale_mlp)
    if self.use_moe:
        x = x + gate_mlp * self.mlp(mlp_input, modality_ids=modality_ids)
    else:
        x = x + gate_mlp * self.mlp(mlp_input)

    return x
```

#### 🔴 缺失的关键细节

1. **AdaMN的exact机制**：论文说"dedicated modulation trajectories"，实际是per-modality LayerNorm + shared modulation
2. **两个gate参数**：`gate_msa` 和 `gate_mlp` 分别控制attention和MLP的残差连接强度
3. **MoE只在后半部分使用**：`moe_start_layer=14` 表示前14层用dense，后14层用MoE

---

#### 原文段落 3: Training Protocol

> "Two-stage paradigm: **Large-scale Pre-training → Domain-specific Fine-tuning**
> - Pre-training on Open X-Embodiment (OXE) dataset
> - Fine-tuning on domain-specific data"

#### ⚠️ 需要验证

从代码 `train_robot.py:461-481` 可以看到预训练权重加载逻辑：

```python
if args.rgb_init is not None:
    checkpoint = torch.load(args.rgb_init, map_location='cpu')
    state_dict = checkpoint['model']
    # ... adaptation logic ...
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Successfully loaded & adapted pretrained weights from {args.rgb_init}")
```

**配置文件显示** (`configs/metaworld_4d.yaml:59-60`):
```yaml
dit_init: null
rgb_init: "/path/to/bridge_pre.pt"  # Bridge数据集预训练权重
```

#### 🔴 问题

1. **没有提到Bridge数据集**：论文说OXE，但实际用的是Bridge预训练权重
2. **"Fine-tuning"不准确**：代码中是`load_state_dict(..., strict=False)`，允许部分权重缺失，更像"transfer learning"而非标准fine-tuning

---

## 2. 整体架构详解

### 2.1 架构概览图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Uni-Embodied DiT                             │
│                    (Diffusion Transformer XL/2)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  1. Modality Embeddings (分散的"Unified Tokenizer")           │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │ │
│  │  │ RGB Embedder│  │Action Embed.│  │Force Embed. │  ...      │ │
│  │  │ PatchEmbed  │  │ Linear      │  │ Linear      │          │ │
│  │  │ +PosEmbed   │  │ +PosEmbed   │  │ +PosEmbed   │          │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │ │
│  │         │                │                │                   │ │
│  │         └────────────────┴────────────────┘                   │ │
│  │                           ▼                                   │ │
│  │                  Concatenate Tokens                           │ │
│  │                  [B, 1060, 1152]                             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  2. Conditional Modulation (AdaLN)                            │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  t_emb = TimestepEmbedder(t)      → [B, 1152]                │ │
│  │  y_emb = LanguageEmbedder(y)      → [B, 1152]                │ │
│  │  c = t_emb + y_emb                → [B, 1152]                │ │
│  │                                                               │ │
│  │  shift_msa, scale_msa, gate_msa,                              │ │
│  │  shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(c)        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  3. Transformer Backbone (L=28 blocks)                       │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  for block in self.blocks:  # 前14层dense, 后14层MoE          │ │
│  │      x = block(x, c, modality_ids)                           │ │
│  │                                                               │ │
│  │  每个block内部 (Modulate--Synthesize--Scale):                 │ │
│  │  ┌─────────────────────────────────────────────────────┐     │ │
│  │  │ 1. AdaMN: apply_expert_ln(x, modality_ids)          │     │ │
│  │  │ 2. Modulate: x * (1 + scale) + shift                │     │ │
│  │  │ 3. Synthesize: Attention(x)                         │     │ │
│  │  │ 4. Residual: x = x + gate * attn_out                │     │ │
│  │  │ 5. AdaMN + Modulate (for MLP)                        │     │ │
│  │  │ 6. Scale: MoE(x) or Dense MLP(x)                    │     │ │
│  │  │ 7. Residual: x = x + gate * mlp_out                 │     │ │
│  │  └─────────────────────────────────────────────────────┘     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  4. Joint Prediction Head (FinalLayer)                       │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  按modality_ids分离tokens:                                    │ │
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │ │
│  │  │ RGB tokens  │Action tokens│Force tokens │Depth tokens │  │ │
│  │  │[0:1024]     │[1024:1027]  │[1027:1028]  │[1028:1060]  │  │ │
│  │  └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘  │ │
│  │         │             │             │             │          │ │
│  │         ▼             ▼             ▼             ▼          │ │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │ │
│  │  │norm_final│   │a_norm   │   │(skip)   │   │d_norm   │     │ │
│  │  └────┬────┘   └────┬────┘   └─────────┘   └────┬────┘     │ │
│  │       │              │                             │          │ │
│  │       ▼              ▼                             ▼          │ │
│  │  ┌─────────┐   ┌─────────┐                   ┌─────────┐    │ │
│  │  │AdaLN    │   │a_AdaLN  │                   │d_AdaLN  │    │ │
│  │  └────┬────┘   └────┬────┘                   └────┬────┘    │ │
│  │       │              │                             │          │ │
│  │       ▼              ▼                             ▼          │ │
│  │  ┌─────────┐   ┌─────────┐                   ┌─────────┐    │ │
│  │  │Linear   │   │a_linear │                   │d_linear │    │ │
│  │  │→rgb_out │   │→act_out │                   │→depth_out│    │ │
│  │  └─────────┘   └─────────┘                   └─────────┘    │ │
│  │                                                               │ │
│  │  return (rgb_out, act_out, depth_out)                        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键参数配置 (DiT-XL/2)

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 1152 | Transformer隐藏维度 |
| `depth` | 28 | Transformer block数量 |
| `num_heads` | 16 | 注意力头数量 |
| `mlp_ratio` | 4.0 | MLP扩展比例 |
| `num_experts` | 4 | MoE专家数量 |
| `moe_top_k` | 1 | 每个token选择的专家数 |
| `moe_start_layer` | 14 | 从第14层开始使用MoE |
| `shared_experts` | 4 | 每个MoE层的共享专家数 |
| `patch_size` | 2 | 图像patch大小 |
| `input_size` | 32 | 潜在空间大小 (256/8) |

---

## 3. 模块化组件分析

### 3.1 分散的"Unified Tokenizer"

论文中描述的"Unified Tokenizer"在代码中**不是单一模块**，而是由多个embedder组成：

#### RGB Tokenizer (`models.py:446-479`)

```python
# 输入: [B, 2*C, H, W] 其中 C=4*predict_horizon
#   - 前C通道: 噪声 (需要去噪)
#   - 后C通道: 条件 (当前观测)

self.x_embedder = PatchEmbed(
    input_size=32,      # 潜在空间大小
    patch_size=2,       # 每个patch 2×2
    in_channels=2*C,    # 2*C = 2*4*3 = 24 (predict_horizon=3)
    hidden_size=1152
)

# 内部: Conv2d(24, 1152, kernel_size=2, stride=2)
# 输出: [B, 1024, 1152]  (32/2)² = 256 patches → 1024 tokens

self.pos_embed = nn.Parameter(torch.zeros(1, 1024, 1152))
# 固定的sin-cos位置编码
```

#### Action Tokenizer (`models.py:449-456`)

```python
# 输入: [B, action_steps, action_dim] 或 [B, 1, action_dim*(action_steps+1)]
action_input_shape = action_dim * (action_steps + 1) if action_condition else action_dim

self.a_embedder = nn.Linear(action_input_shape, 1152)

# 位置编码:
if learnable_action_pos:
    self.a_pos_embed = nn.Parameter(torch.zeros(1, action_steps, 1152))
else:
    self.a_pos_embed = nn.Parameter(torch.zeros(1, action_steps, 1152))
    # 使用固定的sin-cos编码

# 输出: [B, 3, 1152]  (action_steps=3)
```

#### Force Tokenizer (`models.py:457-459`)

```python
# 输入: [B, 1, force_dim] 其中 force_dim=6

self.force_embedder = nn.Linear(6, 1152)
self.f_pos_embed = nn.Parameter(torch.zeros(1, 1, 1152))

# 输出: [B, 1, 1152]
```

#### Depth Tokenizer (`models.py:460-470`)

```python
# 输入: [B, 2, 32, 32]
#   - 通道0: 噪声深度
#   - 通道1: 条件深度

self.d_embedder = PatchEmbed(
    d_input_size=32,
    d_patch_size=8,         # 8×8 patches
    d_embedder_channels=2,
    hidden_size=1152
)

# 内部: Conv2d(2, 1152, kernel_size=8, stride=8)
# 输出: [B, 16, 1152]  (32/8)² = 16 patches

self.d_num_patches = 16
```

---

### 3.2 Adaptive Modality-specific Normalization (AdaMN)

#### 概念

AdaMN是**Per-Modality LayerNorm**，每个模态有独立的归一化参数：

```python
# models.py:242-251

if use_adamn:
    # 为每个模态创建独立的LayerNorm
    self.norm1_experts = nn.ModuleList([
        nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        for _ in range(num_modalities)  # 2-4个模态
    ])
    self.norm2_experts = nn.ModuleList([
        nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        for _ in range(num_modalities)
    ])
else:
    # 共享LayerNorm (原始DiT)
    self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
    self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
```

#### 应用

```python
# models.py:25-59 - apply_expert_ln()

def apply_expert_ln(x, modality_ids, experts):
    """
    x: [B, N, D] 输入tokens
    modality_ids: [B, N] 每个token的模态ID
    experts: ModuleList of M LayerNorm modules

    为每个模态应用独立的LayerNorm
    """
    B, N, D = x.shape
    output = torch.zeros_like(x)

    for m in range(num_modalities):
        mask = (modality_ids == m)  # 找到属于该模态的tokens
        if mask.any():
            tokens = x[mask]        # 选择这些tokens
            normalized = experts[m](tokens)  # 应用该模态的LayerNorm
            output[mask] = normalized  # 放回原位

    return output
```

#### 效果对比

| 场景 | 共享LayerNorm | AdaMN |
|------|--------------|-------|
| 参数量 | 0 (无affine) 或 2×D | 2×M×D (M=模态数) |
| 行为 | 所有模态共享归一化统计 | 每个模态独立归一化 |
| 优势 | 参数少，计算快 | 缓解模态不平衡 |
| 劣势 | 视觉模态主导 | 参数增加 |

---

### 3.3 Sparse Mixture-of-Experts (MoE)

#### 架构 (`moe_blocks.py`)

```python
class SparseMoeBlock(nn.Module):
    def __init__(self, embed_dim=1152, mlp_ratio=4.0,
                 num_experts=4, num_experts_per_tok=1,
                 n_shared_experts=4, ...):
        # ========== Gate ==========
        self.gate = MoEGate(
            embed_dim=1152,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            use_modality_bias=True,  # 关键: 模态感知路由
            num_modalities=3,  # RGB + Action + Depth
            modality_bias_init=bias  # 初始化偏置
        )

        # ========== Shared Experts ==========
        self.shared_experts = nn.Sequential(
            nn.Linear(1152, 1152 * mlp_ratio),
            nn.GELU(),
            nn.Linear(1152 * mlp_ratio, 1152)
        )

        # ========== Routed Experts ==========
        self.experts = nn.ModuleList([
            MLP(1152, 1152 * mlp_ratio)
            for _ in range(num_experts)
        ])
```

#### 模态感知路由 (`moe_blocks.py:66-99`)

```python
def forward(self, hidden_states, modality_ids=None):
    # 1. 计算原始logits
    logits = F.linear(hidden_states, self.weight)  # [B*N, n_experts]

    # 2. 添加模态偏置 (关键!)
    if use_modality_bias and modality_ids is not None:
        flat_modality = modality_ids.reshape(-1)
        bias = self.modality_bias[flat_modality]  # [B*N, n_experts]
        logits = logits + bias  # 每个模态的专家偏好不同

    # 3. Top-K选择
    topk_weight, topk_idx = torch.topk(logits, k=self.top_k)

    # 4. 计算辅助损失 (load balancing)
    # 注意: 排除action tokens (modality_id=1) 不参与辅助损失
    if modality_ids is not None:
        keep_mask = (flat_modality != 1)
        aux_loss = compute_aux_loss(scores[keep_mask])

    return topk_idx, topk_weight, aux_loss
```

#### 初始化偏置 (`configs/metaworld_4d.yaml:110-112`)

```yaml
modality_bias_strength_action: 0.5  # 动作模态偏向expert 0
modality_bias_strength_depth: 0.0   # 深度模态无偏置
```

```python
# models.py:434-443
if use_modality_bias:
    bias = torch.zeros(num_modalities, num_experts)
    if modality_bias_strength_action != 0.0:
        bias[1, 0] = 0.5  # Action (id=1) → Expert 0
    if modality_bias_strength_depth != 0.0:
        depth_id = 2 + int(use_force)  # 2或3
        bias[depth_id, 1] = strength
```

---

### 3.4 Joint Prediction Head (FinalLayer)

#### 多头输出结构 (`models.py:358-382`)

```python
class FinalLayer(nn.Module):
    def forward(self, x, c):
        B = x.shape[0]

        # ========== RGB Head ==========
        start, end = args.start_idx[0], args.end_idx[0]  # 0, 1024
        rgb = x[:, start:end]  # [B, 1024, 1152]

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        rgb = modulate(self.norm_final(rgb), shift, scale)
        rgb = self.linear(rgb)  # [B, 1024, 4*p²*predict_horizon]
        # = [B, 1024, 4*4*3] = [B, 1024, 48]

        # ========== Action Head ==========
        if self.use_action:
            start, end = args.start_idx[1], args.end_idx[1]  # 1024, 1027
            a = x[:, start:end]  # [B, 3, 1152]

            shift, scale = self.a_adaLN_modulation(c).chunk(2, dim=1)
            a = modulate(self.a_norm_final(a), shift, scale)
            a = self.a_linear(a)  # [B, 3, action_dim*2]
            # = [B, 3, 8]  (mean + variance)

        # ========== Depth Head ==========
        if self.use_depth:
            start, end = args.start_idx[3], args.end_idx[3]  # 1028, 1060
            d = x[:, start:end]  # [B, 32, 1152]

            d_shift, d_scale = self.d_adaLN_modulation(c).chunk(2, dim=1)
            d = modulate(self.d_norm_final(d), d_shift, d_scale)
            d = self.d_linear(d)  # [B, 32, d_patch_size²*predict_horizon*2]
            # = [B, 32, 64*3*2] = [B, 32, 384]

        return (rgb, a, d)
```

#### 输出维度总结

| 模态 | 输入shape | 输出shape | 解码后 |
|------|----------|-----------|--------|
| RGB | [B, 1024, 1152] | [B, 1024, 48] | [B, 12, 32, 32] → VAE → [B, 3, 256, 256] |
| Action | [B, 3, 1152] | [B, 3, 8] | [B, 3, 4] (xyzg) |
| Depth | [B, 32, 1152] | [B, 32, 384] | [B, 3, 32, 32] |

---

## 4. 训练流程详解

### 4.1 训练数据流 (`train_robot.py:657-721`)

```python
# ========== 输入数据 ==========
for x_cond, x, depth_cond, depth, action_cond, action, force_cond, y in loader:
    # x_cond: [B, 1, 4, 32, 32] → [B, 4, 32, 32]  (当前图像latent)
    # x:      [B, 1, 12, 32, 32] → [B, 12, 32, 32] (未来图像latent, 3帧×4通道)
    # depth_cond: [B, 1, 32, 32]  (当前深度)
    # depth:      [B, 3, 32, 32]   (未来深度, 3帧)
    # action_cond: [B, 4]         (当前状态)
    # action:      [B, 3, 4]       (未来动作, 3步×4维)
    # force_cond:  [B, 6]          (当前力)
    # y:           [B]             (文本指令索引)

    # ========== 噪声注入 ==========
    t = torch.randint(0, num_timesteps, (B,))

    # ========== 前向传播 ==========
    model_kwargs = {
        'y': y,
        'x_cond': x_cond,
        'depth_cond': depth_cond,
        'depth': depth,
        'action': action,
        'action_cond': action_cond,
        'force_cond': force_cond
    }

    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

    # ========== 损失计算 ==========
    loss = loss_dict["loss"].mean()  # RGB损失

    if args.action_steps > 0 and "loss_a" in loss_dict:
        a_coeff = 1.0 if train_steps > args.action_loss_start else 0.0
        loss = loss + loss_dict["loss_a"].mean() * args.action_loss_lambda * a_coeff

    if args.use_depth and "loss_depth" in loss_dict:
        loss = loss + loss_dict["loss_depth"].mean()

    # ========== MoE辅助损失 ==========
    if args.use_moe:
        aux_loss = model.get_last_aux_loss()
        loss = loss + aux_loss * args.aux_loss_weight

    # ========== 反向传播 ==========
    loss.backward()
    optimizer.step()
```

### 4.2 损失函数组成

```python
Total Loss = Loss_RGB + λ_action * Loss_action + Loss_depth + λ_moe * Loss_moe

其中:
- Loss_RGB: MSE(ε, ε_θ(x, x_cond, ...))  # 图像噪声预测
- Loss_action: MSE(ε_a, ε_θ_a(...))       # 动作噪声预测
- Loss_depth: MSE(ε_d, ε_θ_d(...))        # 深度噪声预测
- Loss_moe: Load balancing loss          # MoE负载均衡
```

### 4.3 预训练权重加载 (`train_robot.py:461-481`)

```python
if args.rgb_init is not None:
    # 从Bridge预训练权重加载
    checkpoint = torch.load(args.rgb_init, map_location='cpu')
    state_dict = checkpoint['model']

    # ===== 关键适配逻辑 =====
    # 1. x_embedder.proj.weight: [C_in, 3, ...] → [C_in, 4*predict_horizon, ...]
    if "x_embedder.proj.weight" in state_dict:
        old_weight = state_dict["x_embedder.proj.weight"]  # [C, 3, p, p]
        new_C_in = 4 * predict_horizon  # 12
        new_weight = torch.zeros(new_C_in, 2, patch_size, patch_size)

        # 复制通道
        new_weight[:3] = old_weight[:3]  # RGB通道
        new_weight[3:4] = old_weight.mean(0, keepdim=True)  # 用均值初始化第4通道

        # 为predict_horizon扩展
        for h in range(1, predict_horizon):
            new_weight[h*4:(h+1)*4] = new_weight[:4]

        state_dict["x_embedder.proj.weight"] = new_weight

    # 2. 其他类似适配...

    # 3. 加载权重 (允许部分缺失)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
```

### 4.4 学习率调度 (`configs/metaworld_4d.yaml:34-37`)

```yaml
use_lr_scheduler: true
scheduler_type: "cosine"
warmup_steps: 8000      # 前8000步线性warmup
min_lr_ratio: 0.01      # 衰减到初始LR的1%
```

实际LR曲线:
```
LR(t) = {
    min_lr + (max_lr - min_lr) * t / warmup_steps,           if t < warmup_steps
    min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress)), otherwise
}

其中 progress = (t - warmup_steps) / (total_steps - warmup_steps)
```

---

## 5. 问题与修正建议

### 5.1 问题汇总表

| 章节 | 问题 | 严重程度 | 建议 |
|------|------|---------|------|
| System Overview | "Unified Tokenizer" 不存在 | 🔴 高 | 删除或重命名为 "Modality Embeddings" |
| System Overview | `$\mathbf{w}_k$` 符号未定义 | 🔴 高 | 改为 `$\{\mathbf{z}_k, \mathbf{a}_k, \mathbf{d}_k\}$` |
| System Overview | 单一输出 `$$\hat{\boldsymbol{\epsilon}}$$` | 🔴 高 | 改为多输出 |
| Block Design | 缺少两个gate参数 | 🟡 中 | 补充 `gate_msa`, `gate_mlp` |
| Block Design | "Modulate--Synthesize--Scale" 不够精确 | 🟡 中 | 补充 "Normalize" 阶段 |
| Training | "OXE dataset" 不准确 | 🟡 中 | 改为 "Bridge dataset" |
| Training | "Fine-tuning" 术语不准确 | 🟢 低 | 改为 "Transfer Learning" |

### 5.2 建议的修正版本

#### System Overview 修正

```latex
\textbf{System Overview.} As illustrated in Figure 1, Uni-Embodied DiT
comprises three primary components:

\begin{itemize}
    \item \textbf{Modality Embeddings:} A collection of modality-specific
    encoders that project heterogeneous observations into a homogeneous
    token sequence. For each spatial modality, we adopt a \textbf{[noisy target + clean condition]}
    pairing scheme:
    \begin{itemize}
        \item RGB: PatchEmbed$([\mathbf{z}_k, \mathbf{z}_t])$ → 1024 tokens
        \item Action: Linear$([\mathbf{a}_k, \mathbf{a}_t])$ → 3 tokens
        \item Force: Linear$(\mathbf{f}_t)$ → 1 token
        \item Depth: PatchEmbed$([\mathbf{d}_k, \mathbf{d}_t])$ → 32 tokens
    \end{itemize}
    where $\mathbf{z}_k, \mathbf{a}_k, \mathbf{d}_k$ are noisy targets to be denoised,
    and $\mathbf{z}_t, \mathbf{a}_t, \mathbf{d}_t, \mathbf{f}_t$ are clean observations.

    \item \textbf{Multimodal Backbone:} Consisting of $L=28$ stacked Transformer Blocks,
    with sparse Mixture-of-Experts (MoE) enabled from layer 14 onwards.
    Each block follows a \textit{Normalize--Modulate--Synthesize--Scale} paradigm
    detailed in Section X.

    \item \textbf{Joint Prediction Head:} A multi-head output layer that predicts
    modality-specific noise estimates $\{\hat{\boldsymbol{\epsilon}}_z, \hat{\boldsymbol{\epsilon}}_a, \hat{\boldsymbol{\epsilon}}_d\}$
    under the shared diffusion step $k$, driving the reverse diffusion transition
    for all modalities simultaneously.
\end{itemize}
```

#### Block Design 修正

```latex
\textbf{Block Design Paradigm: Normalize--Modulate--Synthesize--Scale.}

Each Transformer block follows a four-stage paradigm:

\begin{enumerate}
    \item \textbf{Normalize (AdaMN):} Apply Adaptive Modality-specific Normalization,
    where each modality (RGB, action, force, depth) has independent LayerNorm
    parameters with learnable affine transformations.

    \item \textbf{Modulate (AdaLN):} Compute modulation parameters from the
    diffusion timestep $k$ and instruction $y$:
    \begin{equation}
    \text{shift}, \text{scale}, \text{gate} = \text{AdaLN}(t_k, y)
    \end{equation}
    Separate modulation is applied for attention (shift\textsubscript{MSA}, scale\textsubscript{MSA}, gate\textsubscript{MSA})
    and MLP (shift\textsubscript{MLP}, scale\textsubscript{MLP}, gate\textsubscript{MLP}).

    \item \textbf{Synthesize (Attention):} Apply multi-head self-attention with
    AdaLN-modulated inputs:
    \begin{equation}
    \mathbf{x}' = \mathbf{x} + \text{gate}_{\text{MSA}} \cdot \text{Attention}(\text{AdaMN}(\mathbf{x}) \odot (1 + \text{scale}_{\text{MSA}}) + \text{shift}_{\text{MSA}})
    \end{equation}

    \item \textbf{Scale (MoE):} Apply sparse mixture-of-experts expansion:
    \begin{equation}
    \mathbf{x}'' = \mathbf{x}' + \text{gate}_{\text{MLP}} \cdot \text{MoE}(\text{AdaMN}(\mathbf{x}') \odot (1 + \text{scale}_{\text{MLP}}) + \text{shift}_{\text{MLP}})
    \end{equation}
    where MoE uses modality-aware gating to route tokens to specialized experts.
\end{enumerate}
```

#### Training Protocol 修正

```latex
\textbf{Training Protocol.}

We adopt a transfer learning paradigm rather than end-to-end pre-training:

\begin{itemize}
    \item \textbf{Source Model:} Pre-trained on the Bridge dataset [citation],
    a large-scale robotic manipulation dataset with visual observations and actions.
    The source model predicts future RGB frames from current images and text instructions.

    \item \textbf{Weight Adaptation:} When loading the pre-trained weights, we adapt
    the image embedder to accommodate the \textit{[noisy + condition]} pairing scheme
    and extend the channel dimension to support multi-frame prediction ($H=3$).

    \item \textbf{Target Domain Training:} On target datasets (e.g., MetaWorld),
    we train the model with the joint denoising objective:
    \begin{equation}
    \mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_a \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{depth}} + \lambda_{\text{MoE}} \mathcal{L}_{\text{aux}}
    \end{equation}
    where the action loss is gradually introduced after 1000 steps to stabilize
    early training.
\end{itemize}
```

---

## 6. 代码引用索引

### 6.1 关键文件速查

| 功能 | 文件 | 关键行号 |
|------|------|---------|
| **模型定义** | | |
| DiT主类 | `models.py` | 385-530 |
| DiTBlock | `models.py` | 218-314 |
| FinalLayer | `models.py` | 317-382 |
| AdaMN应用 | `models.py` | 25-59, 287-290 |
| **MoE组件** | | |
| MoEGate | `moe_blocks.py` | 25-136 |
| SparseMoeBlock | `moe_blocks.py` | 169-241 |
| 模态感知路由 | `moe_blocks.py` | 73-78 |
| **训练** | | |
| 主训练循环 | `train_robot.py` | 652-950 |
| 损失计算 | `train_robot.py` | 714-720 |
| 预训练加载 | `train_robot.py` | 461-481 |
| **推理** | | |
| Agent.action | `evaluation/agent.py` | 204-280 |
| 扩散采样 | `diffusion/gaussian_diffusion.py` | 402-560 |

### 6.2 重要数据形状速查

| 变量 | Shape | 说明 |
|------|-------|------|
| `x_cond` | `[B, 4, 32, 32]` | 当前图像latent |
| `x` | `[B, 12, 32, 32]` | 未来3帧图像latent |
| `z` | `[B, 12, 32, 32]` | 噪声 (推理时初始化) |
| `action_cond` | `[B, 4]` | 当前状态xyzg |
| `action` | `[B, 3, 4]` | 未来3步动作 |
| `z_a` | `[B, 3, 4]` | 动作噪声 |
| `force_cond` | `[B, 6]` | 当前六维力 |
| `depth_cond` | `[B, 1, 32, 32]` | 当前深度 |
| `depth` | `[B, 3, 32, 32]` | 未来3帧深度 |
| `z_d` | `[B, 3, 32, 32]` | 深度噪声 |
| `modality_ids` | `[B, 1060]` | 每个token的模态ID |
| `c` | `[B, 1152]` | 条件调制 (t+y) |
| `x` (in Transformer) | `[B, 1060, 1152]` | 统一token序列 |

---

## 附录A: 完整前向传播伪代码

```python
def uni_embodied_dit_forward(
    # Noisy targets (需要去噪)
    z: Tensor[B, 12, 32, 32],          # RGB噪声
    z_a: Tensor[B, 3, 4],              # 动作噪声
    z_d: Tensor[B, 3, 32, 32],         # 深度噪声

    # Clean conditions (观测值)
    x_cond: Tensor[B, 4, 32, 32],      # 当前图像
    action_cond: Tensor[B, 4],         # 当前状态
    force_cond: Tensor[B, 6],          # 当前力
    depth_cond: Tensor[B, 1, 32, 32],  # 当前深度

    # Diffusion conditions
    t: Tensor[B],                      # 时间步
    y: Tensor[B],                      # 文本指令索引
):
    # ========== Stage 1: Modality Embeddings ==========

    # RGB: [噪声, 条件] → PatchEmbed
    x_rgb = torch.cat([z, x_cond], dim=1)  # [B, 16, 32, 32]
    x_rgb = x_embedder(x_rgb) + pos_embed   # [B, 1024, 1152]

    # Action: [噪声, 条件] → Linear
    a_input = torch.cat([z_a, action_cond], dim=-1)  # [B, 3, 8]
    a = a_embedder(a_input) + a_pos_embed            # [B, 3, 1152]

    # Force: 条件 → Linear
    f = force_embedder(force_cond) + f_pos_embed     # [B, 1, 1152]

    # Depth: [噪声, 条件] → PatchEmbed
    d_input = torch.cat([z_d, depth_cond], dim=1)    # [B, 4, 32, 32]
    d = d_embedder(d_input) + d_pos_embed            # [B, 32, 1152]

    # Concatenate all tokens
    x = torch.cat([x_rgb, a, f, d], dim=1)           # [B, 1060, 1152]

    # Assign modality IDs
    modality_ids = torch.zeros(B, 1060, dtype=torch.long)
    modality_ids[:, 1024:1027] = 1   # Action
    modality_ids[:, 1027:1028] = 2   # Force
    modality_ids[:, 1028:1060] = 3   # Depth

    # ========== Stage 2: Conditional Modulation ==========

    t_emb = t_embedder(t)            # [B, 1152]
    y_emb = y_embedder(y)            # [B, 1152]
    c = t_emb + y_emb                # [B, 1152]

    # ========== Stage 3: Transformer Backbone ==========

    for i, block in enumerate(blocks):
        # 前14层: dense, 后14层: MoE
        use_moe = (i >= moe_start_layer)
        x = block(x, c, modality_ids, use_moe)

    # ========== Stage 4: Joint Prediction Head ==========

    # Split by modality
    rgb_tokens = x[:, 0:1024, :]      # [B, 1024, 1152]
    action_tokens = x[:, 1024:1027, :] # [B, 3, 1152]
    depth_tokens = x[:, 1028:1060, :]  # [B, 32, 1152]

    # RGB head
    rgb = norm_final(rgb_tokens)
    shift, scale = adaLN_modulation(c).chunk(2, dim=1)
    rgb = rgb * (1 + scale) + shift
    rgb_out = linear(rgb)             # [B, 1024, 48]

    # Action head
    a = a_norm_final(action_tokens)
    shift_a, scale_a = a_adaLN_modulation(c).chunk(2, dim=1)
    a = a * (1 + scale_a) + shift_a
    action_out = a_linear(a)          # [B, 3, 8]

    # Depth head
    d = d_norm_final(depth_tokens)
    shift_d, scale_d = d_adaLN_modulation(c).chunk(2, dim=1)
    d = d * (1 + scale_d) + shift_d
    depth_out = d_linear(d)           # [B, 32, 384]

    return rgb_out, action_out, depth_out
```

---

## 附录B: 术语对照表

| 论文术语 | 代码变量 | 类型 | 说明 |
|----------|----------|------|------|
| `$\mathbf{w}_k$` | - | ❌ 不存在 | 应使用 `$\{\mathbf{z}_k, \mathbf{a}_k, \mathbf{d}_k\}$` |
| `$\mathbf{z}_k$` | `z` | 噪声目标 | 未来RGB latents |
| `$\mathbf{z}_t$` | `x_cond` | 条件 | 当前图像latent |
| `$\mathbf{a}_k$` | `z_a` | 噪声目标 | 未来动作噪声 |
| `$\mathbf{a}_t$` | `action_cond` | 条件 | 当前机器人状态 |
| `$\mathbf{d}_k$` | `z_d` | 噪声目标 | 未来深度噪声 |
| `$\mathbf{d}_t$` | `depth_cond` | 条件 | 当前深度 |
| `$\mathbf{f}_t$` | `force_cond` | 条件 | 当前六维力 |
| `$\hat{\boldsymbol{\epsilon}}$` | - | ❌ 不明确 | 应使用多输出符号 |
| `$\hat{\boldsymbol{\epsilon}}_z$` | `model_output` | RGB噪声预测 | |
| `$\hat{\boldsymbol{\epsilon}}_a$` | `extra` | 动作噪声预测 | |
| `$\hat{\boldsymbol{\epsilon}}_d$` | `extra2` | 深度噪声预测 | |
| `$k$` | `t` | 时间步 | 扩散步数 |
| `$y$` | `y` | 文本条件 | CLIP编码的指令 |
| `Uni-Embodied DiT` | `DiT` | 模型类 | |
| `Unified Tokenizer` | - | ❌ 不存在 | 分散的embedders |
| `AdaMN` | `apply_expert_ln` | 函数 | Per-modality LayerNorm |
| `Joint Prediction Head` | `FinalLayer` | 类 | 多头输出层 |
