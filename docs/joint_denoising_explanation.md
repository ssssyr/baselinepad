# 联合反向去噪过程：代码实现与论文描述对照

## 目录
- [1. 论文初稿与代码实现的对比](#1-论文初稿与代码实现的对比)
- [2. 核心问题详细分析](#2-核心问题详细分析)
- [3. 正确的模型架构理解](#3-正确的模型架构理解)
- [4. 建议的修正版本](#4-建议的修正版本)
- [5. 代码引用索引](#5-代码引用索引)

---

## 1. 论文初稿与代码实现的对比

### 1.1 原论文初稿

> **Joint Reverse Denoising.** The core of our approach is to learn a conditional denoiser $\boldsymbol{\epsilon}_\theta$ that recovers the clean objective $\mathbf{w}_0$ from $\mathbf{w}_k$. Crucially, the model operates on a unified sequence $\mathbf{X} = \{z_t, d_t, f_t, \mathbf{w}_k\}$, where the current observations $\{z_t, d_t, f_t\}$ serve as _un-noised context tokens_ and $\mathbf{w}_k$ acts as the _noisy target tokens_.

### 1.2 实际代码实现

#### 输入准备 (`evaluation/agent.py:204-255`)

```python
# ========== 噪声目标 (需要被去噪) ==========
z = torch.randn(1, in_channels*predict_horizon, latent_size, latent_size)
# → 未来RGB latents的纯噪声

z_a = torch.randn(1, action_steps, action_dim)
# → 未来动作的纯噪声

z_d = torch.randn(1, predict_horizon, d_hidden_size, d_hidden_size)
# → 未来深度的纯噪声

# ========== 无噪声条件 (作为context) ==========
x_cond = self.encode_image(rgb)
# → 当前观测图像的VAE编码，无噪声

depth_cond = self.filter(depth)
# → 当前观测深度图，无噪声

force_cond = normalize(force)
# → 当前观测的六维力，无噪声

action_cond = state * action_scale
# → 当前机器人状态，无噪声
```

#### 模型内部处理 (`models.py:655-677`)

```python
def forward(self, x, t, y, x_cond=None, action_cond=None,
            noised_action=None, force_cond=None, depth_cond=None, noised_depth=None):

    # ========== 1. 图像模态：[噪声 + 条件] 配对 ==========
    if x_cond is not None:
        x = torch.cat([x, x_cond], dim=1)
        # x: [B, C, H, W] (噪声)
        # x_cond: [B, C, H, W] (条件)
        # 结果: [B, 2*C, H, W]

    x = self.x_embedder(x) + self.pos_embed
    # 输出: [B, 1024, 1152] → 1024个图像token

    # ========== 2. 动作模态：[噪声 + 条件] 配对 ==========
    if self.args.action_steps > 0:
        if self.args.action_condition:
            noised_action = torch.cat([noised_action, action_cond], dim=-1)

        a = self.a_embedder(noised_action) + self.a_pos_embed
        # 输出: [B, 3, 1152] → 3个动作token

        x = torch.cat([x, a], dim=1)

    # ========== 3. 力模态：仅条件（无噪声） ==========
    if self.use_force:
        f = self.force_embedder(force_cond) + self.f_pos_embed
        # 输出: [B, 1, 1152] → 1个力token

        x = torch.cat([x, f], dim=1)

    # ========== 4. 深度模态：[噪声 + 条件] 配对 ==========
    if self.args.use_depth:
        noised_depth = torch.cat([noised_depth, depth_cond], dim=1)
        # [B, 2, 32, 32] → 噪声帧 + 条件帧

        d = self.d_embedder(noised_depth) + self.d_pos_embed
        # 输出: [B, 32, 1152] → 32个深度token

        x = torch.cat([x, d], dim=1)

    # ========== 5. 统一的token序列 ==========
    # x: [B, 1024+3+1+32, 1152] = [B, 1060, 1152]
    #
    # 索引分布:
    # [0:1024]     → RGB tokens (模态ID=0)
    # [1024:1027]  → Action tokens (模态ID=1)
    # [1027:1028]  → Force tokens (模态ID=2)
    # [1028:1060]  → Depth tokens (模态ID=3)

    # ========== 6. 模态ID标记 ==========
    modality_ids = torch.zeros((B, 1060), dtype=torch.long)
    modality_ids[:, 1024:1027] = 1   # Action
    modality_ids[:, 1027:1028] = 2   # Force
    modality_ids[:, 1028:1060] = 3   # Depth
```

---

## 2. 核心问题详细分析

### 问题 1: 输入序列构成描述错误

#### ❌ 论文原表述
> "unified sequence $\mathbf{X} = \{z_t, d_t, f_t, \mathbf{w}_k\}$, where the current observations $\{z_t, d_t, f_t\}$ serve as _un-noised context tokens_"

#### ✅ 实际情况

从代码 `models.py:655-677` 可以看出，每个模态的实际构成是：

| 模态 | 论文中的符号 | 实际构成 | 是否有噪声 |
|------|-------------|---------|-----------|
| **RGB** | `$z_t$` (声称un-noised) | `[z (噪声), x_cond (条件)]` | **有噪声部分** |
| **Depth** | `$d_t$` (声称un-noised) | `[z_d (噪声), depth_cond (条件)]` | **有噪声部分** |
| **Force** | `$f_t$` | `force_cond (条件)` | 无噪声 ✓ |
| **Action** | 未明确提及 | `[z_a (噪声), action_cond (条件)]` | **有噪声部分** |

#### 🔴 关键错误

**论文声称** `$z_t, d_t$` 是 "un-noised context tokens"

**实际代码** 显示：
- `z` 是 `torch.randn(...)` 产生的**纯噪声**
- `d` (即 `z_d`) 是 `torch.randn(...)` 产生的**纯噪声**
- 它们都**需要被去噪**，不是 "un-noised context"

真正的 "un-noised context" 是：
- `x_cond` (当前观测图像)
- `depth_cond` (当前观测深度)
- `force_cond` (当前观测力)

---

### 问题 2: `$\mathbf{w}_k$` 定义不明确

#### ❌ 论文原表述
> "$\mathbf{w}_k$ acts as the _noisy target tokens_"

#### ✅ 实际情况

代码中**没有** `$\mathbf{w}_k$` 这个符号。实际有**三个独立的噪声目标**：

```python
# 从 p_sample_loop 返回三个独立的去噪结果:
samples, samples_a, samples_d = self.diffusion.p_sample_loop(...)

# 其中：
# samples    → 去噪后的RGB latents (来自z)
# samples_a  → 去噪后的actions (来自z_a)
# samples_d  → 去噪后的depth (来自z_d)
```

#### 🔴 关键错误

1. **符号混淆**：`$\mathbf{w}_k$` 在代码中不存在
2. **数量错误**：不是单一目标，而是三个独立目标同时被去噪
3. **模态缺失**：论文没有明确提到动作和深度也是被预测的目标

---

### 问题 3: 缺少 [噪声+条件] 配对机制

#### ❌ 论文描述
论文中没有明确描述每个模态内部的 [噪声 + 条件] 配对结构。

#### ✅ 实际代码设计

这是一个**核心设计**：每个空间模态都采用 [噪声 + 条件] 配对：

```python
# 图像模态
x = torch.cat([z, x_cond], dim=1)
# [噪声的未来帧, 当前观测帧] → 2通道 → PatchEmbed → 1024 tokens

# 深度模态
noised_depth = torch.cat([z_d, depth_cond], dim=1)
# [噪声的未来帧, 当前观测帧] → 2通道 → PatchEmbed → 32 tokens

# 动作模态
noised_action = torch.cat([z_a, action_cond], dim=-1)
# [噪声的未来动作, 当前状态] → 线性层 → 3 tokens
```

这种设计的**优势**：
1. **时序连贯性**：条件帧提供起点，噪声帧预测未来
2. **多尺度交互**：条件帧的语义通过attention传播到噪声帧
3. **训练稳定性**：[噪声+条件] 对在训练时提供强监督信号

---

### 问题 4: 输出描述不完整

#### ❌ 论文原表述
> "recovers the clean objective $\mathbf{w}_0$ from $\mathbf{w}_k$"

#### ✅ 实际情况

模型输出**三个独立的预测**：

```python
# FinalLayer (models.py:358-382)
def forward(self, x, c):
    # 分离不同模态的tokens
    rgb = x[:, start_idx[0]:end_idx[0]]   # [B, 1024, 1152]
    a = x[:, start_idx[1]:end_idx[1]]     # [B, 3, 1152]
    d = x[:, start_idx[3]:end_idx[3]]     # [B, 32, 1152]

    # 各自的输出头
    rgb = self.linear(rgb)                # → [B, 1024, 4*patch_size²*predict_horizon]
    a = self.a_linear(a)                  # → [B, 3, action_dim*2] (mean+var)
    d = self.d_linear(d)                  # → [B, 32, d_patch_size²*predict_horizon*2]

    return (rgb, a, d)
```

#### 🔴 关键错误

论文暗示输出是单一的 `$\mathbf{w}_0$`，实际是：
- RGB预测：未来 `predict_horizon` 帧的图像
- Action预测：未来 `action_steps` 步的动作
- Depth预测：未来 `predict_horizon` 帧的深度图

---

## 3. 正确的模型架构理解

### 3.1 完整的数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        输入准备阶段                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │ 当前图像     │  │ 当前深度     │  │ 当前力       │  │ 当前状态     ││
│  │ (256×256×3) │  │ (480×640)   │  │ (6维向量)    │  │ (4维向量)    ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘│
│         │                │                │                │        │
│         ▼                ▼                ▼                ▼        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │ VAE编码     │  │ 滤波+resize  │  │ 标准化       │  │ 缩放        ││
│  │ x_cond      │  │ depth_cond  │  │ force_cond  │  │ action_cond ││
│  │ (32×32×4)   │  │ (32×32)     │  │ (6维)       │  │ (4维)       ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ 随机噪声z   │  │ 随机噪声z_a │  │ 随机噪声z_d │                  │
│  │ (32×32×12)  │  │ (3×4)       │  │ (3×32×32)   │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     模态嵌入与拼接阶段                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RGB:    [z, x_cond] → PatchEmbed → 1024 tokens                     │
│  Action: [z_a, action_cond] → Linear → 3 tokens                     │
│  Force:  force_cond → Linear → 1 token                              │
│  Depth:  [z_d, depth_cond] → PatchEmbed → 32 tokens                 │
│                                                                     │
│  拼接: X = [1024 RGB, 3 Action, 1 Force, 32 Depth] = 1060 tokens    │
│                                                                     │
│  模态ID: [0×1024, 1×3, 2×1, 3×32]                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Transformer处理阶段                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  c = t_embedder(t) + y_embedder(y)  # 时间步+文本的条件调制          │
│                                                                     │
│  for each DiTBlock:                                                │
│      x = block(x, c, modality_ids)  # AdaMN: 按模态归一化           │
│      # 所有模态通过self-attention交互                               │
│                                                                     │
│  输出: x [B, 1060, 1152]                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      输出分离与预测阶段                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FinalLayer根据modality_ids分离输出:                                │
│                                                                     │
│  ┌─────────────┐                                                   │
│  │ RGB tokens  │ → unpatchify → [B, 12, 32, 32] → VAE解码 → 图像    │
│  │ [0:1024]    │   (3帧×4通道)                                      │
│  └─────────────┘                                                   │
│                                                                     │
│  ┌─────────────┐                                                   │
│  │ Action token│ → reshape → [B, 3, 4] → 3步动作序列               │
│  │ [1024:1027] │   (xyzg)                                          │
│  └─────────────┘                                                   │
│                                                                     │
│  ┌─────────────┐                                                   │
│  │ Depth tokens│ → unpatchify → [B, 3, 32, 32] → 深度图            │
│  │ [1028:1060] │   (3帧×1通道)                                      │
│  └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计原则

1. **[噪声 + 条件] 配对**
   - 空间模态（RGB、深度）都采用 2通道输入
   - 通道0：噪声（需要去噪）
   - 通道1：条件（观测值，提供起点）

2. **统一的token空间**
   - 所有模态映射到相同的维度 (1152)
   - 通过共享的Transformer处理
   - 允许跨模态的attention交互

3. **模态特定的调制**
   - AdaMN: 按模态ID应用不同的LayerNorm
   - AdaLN: 时间步和文本的全局调制
   - 实现模态间的解耦与交互平衡

---

## 4. 建议的修正版本

### 4.1 完整的修正文本

```latex
\textbf{Joint Reverse Denoising.} The core of our approach is to learn a
conditional denoiser $\boldsymbol{\epsilon}_\theta$ that \textit{simultaneously}
recovers multiple clean objectives from their noisy counterparts.

Crucially, each spatial modality is represented as a \textbf{[noisy target + clean condition]}
pair that is embedded and concatenated into a unified sequence:

\begin{equation}
\mathbf{X} = \text{Concat}\Big(
    \underbrace{[\mathbf{z}_k, \mathbf{z}_{t}]}_{\substack{\text{RGB} \\ \text{1024 tokens}}},
    \underbrace{[\mathbf{a}_k, \mathbf{a}_t]}_{\substack{\text{Action} \\ \text{3 tokens}}},
    \underbrace{\mathbf{f}_t}_{\substack{\text{Force} \\ \text{1 token}}},
    \underbrace{[\mathbf{d}_k, \mathbf{d}_t]}_{\substack{\text{Depth} \\ \text{32 tokens}}}
\Big) \in \mathbb{R}^{1060 \times D}
\label{eq:unified_sequence}
\end{equation}

where:
\begin{itemize}
    \item $\mathbf{z}_k, \mathbf{a}_k, \mathbf{d}_k$ are the \textit{noisy targets} (future RGB latents, actions, and depth maps to be denoised)
    \item $\mathbf{z}_{t}, \mathbf{a}_t, \mathbf{d}_t, \mathbf{f}_t$ are the \textit{clean conditions} (current observations)
\end{itemize}

The denoising process is modulated by semantic instructions $y$ and diffusion
steps $k$ through an Adaptive Modality-specific Normalization (AdaMN) mechanism:

\begin{equation}
\{\hat{\boldsymbol{\epsilon}}_z, \hat{\boldsymbol{\epsilon}}_a, \hat{\boldsymbol{\epsilon}}_d\} =
\boldsymbol{\epsilon}_\theta(\mathbf{X} \mid y, k)
\label{eq:joint_denoiser}
\end{equation}

where $\hat{\boldsymbol{\epsilon}}_z, \hat{\boldsymbol{\epsilon}}_a, \hat{\boldsymbol{\epsilon}}_d$
are the predicted noise for RGB, action, and depth respectively.

During the iterative refinement (Figure X), all modalities interact through
self-attention while being processed by modality-specific normalization.
This enforces physical consistency: the predicted actions must be executable
given the inferred geometry ($\mathbf{d}_t$) and contact semantics ($\mathbf{f}_t$),
while the imagined future visual trajectory ($\mathbf{z}_0$) must align with
the planned motion ($\mathbf{a}_0$).
```

### 4.2 主要修改点

| 原文 | 修改 | 原因 |
|------|------|------|
| `recovers the clean objective $\mathbf{w}_0$ from $\mathbf{w}_k$` | `simultaneously recovers multiple clean objectives from their noisy counterparts` | 明确多目标预测 |
| `unified sequence $\mathbf{X} = \{z_t, d_t, f_t, \mathbf{w}_k\}$` | Eq.\ref{eq:unified_sequence} with explicit [noise+condition] pairs | 准确反映代码结构 |
| `$\{z_t, d_t, f_t\}$ serve as un-noised context tokens` | `$\mathbf{z}_k, \mathbf{a}_k, \mathbf{d}_k$ are noisy targets; $\mathbf{z}_t, \mathbf{a}_t, \mathbf{d}_t, \mathbf{f}_t$ are clean conditions` | 修正噪声/条件的分类 |
| `$\mathbf{w}_k$ acts as the noisy target tokens` | `三个独立的噪声目标 $\mathbf{z}_k, \mathbf{a}_k, \mathbf{d}_k$` | 明确数量和含义 |
| `$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{X} \mid y, k)$` | `$\{\hat{\boldsymbol{\epsilon}}_z, \hat{\boldsymbol{\epsilon}}_a, \hat{\boldsymbol{\epsilon}}_d\} = \boldsymbol{\epsilon}_\theta(\mathbf{X} \mid y, k)$` | 明确多输出 |

---

## 5. 代码引用索引

### 5.1 关键文件和行号

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 输入准备 | `evaluation/agent.py` | 204-255 | 噪声初始化和条件编码 |
| 图像嵌入 | `models.py` | 655-658 | x与x_cond的concat和PatchEmbed |
| 动作嵌入 | `models.py` | 659-664 | noised_action与action_cond的处理 |
| 力嵌入 | `models.py` | 665-673 | force_cond的Linear映射 |
| 深度嵌入 | `models.py` | 674-677 | noised_depth与depth_cond的PatchEmbed |
| 模态ID标记 | `models.py` | 679-686 | modality_ids的赋值 |
| FinalLayer | `models.py` | 358-382 | 多模态输出的分离 |
| 去噪采样 | `diffusion/gaussian_diffusion.py` | 402-460 | p_sample的实现 |
| 联合去噪 | `diffusion/gaussian_diffusion.py` | 254-358 | p_mean_variance处理多模态 |

### 5.2 关键数据结构

```python
# ========== Token序列结构 ==========
总token数: 1060 = 1024(RGB) + 3(Action) + 1(Force) + 32(Depth)
每个token维度: 1152 (DiT-XL/2)

# ========== 模态ID映射 ==========
0: RGB tokens      [0:1024]
1: Action tokens   [1024:1027]
2: Force tokens     [1027:1028]
3: Depth tokens    [1028:1060]

# ========== 输入形状 ==========
z:         [B, 4*predict_horizon, 32, 32]     predict_horizon=3 → [B, 12, 32, 32]
x_cond:    [B, 4, 32, 32]
z_a:       [B, action_steps, action_dim]     action_steps=3, action_dim=4 → [B, 3, 4]
z_d:       [B, predict_horizon, 32, 32]      [B, 3, 32, 32]
depth_cond:[B, 1, 32, 32]
force_cond:[B, 1, 6]

# ========== 输出形状 ==========
rgb:       [B, 4*predict_horizon, 32, 32]    → [B, 12, 32, 32]
action:    [B, action_steps, action_dim*2]   → [B, 3, 8] (mean+var)
depth:     [B, predict_horizon, 32, 32]      → [B, 3, 32, 32]
```

---

## 附录：符号对照表

| 论文符号 | 代码变量 | 类型 | 说明 |
|----------|----------|------|------|
| `$\mathbf{z}_k$` | `z` | 噪声目标 | 未来RGB latents的噪声 |
| `$\mathbf{z}_t$` | `x_cond` | 条件 | 当前观测图像的VAE编码 |
| `$\mathbf{a}_k$` | `z_a` | 噪声目标 | 未来动作的噪声 |
| `$\mathbf{a}_t$` | `action_cond` | 条件 | 当前机器人状态 |
| `$\mathbf{d}_k$` | `z_d` | 噪声目标 | 未来深度的噪声 |
| `$\mathbf{d}_t$` | `depth_cond` | 条件 | 当前观测深度 |
| `$\mathbf{f}_t$` | `force_cond` | 条件 | 当前六维力 |
| `$\mathbf{X}$` | `x` (在forward中) | 统一序列 | 所有模态token的拼接 |
| `$\boldsymbol{\epsilon}_\theta$` | `model.forward()` | 去噪器 | DiT模型 |
| `$\hat{\boldsymbol{\epsilon}}_z$` | `model_output[:,:,:4*H]` | RGB噪声预测 | |
| `$\hat{\boldsymbol{\epsilon}}_a$` | `extra` | 动作噪声预测 | |
| `$\hat{\boldsymbol{\epsilon}}_d$` | `extra2` | 深度噪声预测 | |
| `$y$` | `y` | 文本条件 | CLIP编码的指令 |
| `$k$` | `t` | 时间步 | 扩散时间步 |
