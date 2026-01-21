# DeMUSE 架构附录：异构特征嵌入层

## 概述

本文档详细描述了 DeMUSE (Deep Multimodal Unified Sparse Experts) 架构中异构特征嵌入层的技术实现细节。

---

## 核心参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| `D` (hidden_size) | 1152 | Transformer 全局隐藏维度 |
| `depth` | 28 | DiT-XL Transformer 层数 |
| `num_heads` | 16 | 多头注意力头数 |
| `patch_size` (RGB) | 2 | RGB 图像 Patch 大小 |
| `d_patch_size` (Depth) | 8 | 深度图 Patch 大小 |
| `latent_dim` (VAE) | 4 | SD VAE 潜在通道数 |
| `input_size` | 32×32 | VAE 压缩后的特征图尺寸 |
| `force_dim` | 6 | 6轴力/力矩信号维数 |

---

## 异构特征嵌入层 (Heterogeneous Embedding Layer)

在 DeMUSE 框架中，来自不同模态的观测值被映射到一个统一的 $D$ 维潜空间中 [cite: 350]。设 $D = 1152$ 为 Transformer 的全局隐藏维度，各模态的嵌入过程如下：

### RGB 图像 ($I_t$)

RGB 图像通过预训练的 Stable Diffusion VAE (`sd-vae-ft-mse`) 编码器进行压缩：

- **输入尺寸**: $H \times W \times 3 = 256 \times 256 \times 3$
- **VAE 压缩**: 8倍下采样，得到 $32 \times 32 \times 4$ 的潜在表示
- **VAE latent_dim**: $C_{\text{VAE}} = 4$ 通道

随后，VAE 潜在特征通过 patch-projection 层进行嵌入：

```python
self.x_embedder = PatchEmbed(
    input_size=32,      # VAE 输出的空间分辨率
    patch_size=2,       # Patch 大小
    in_channels=4,      # VAE 潜在通道数
    hidden_size=1152    # 输出维度 D
)
```

- **Patch 分割**: $32 \times 32$ 特征图被划分为 $(32/2)^2 = 16^2 = 256$ 个非重叠 patches
- **投影层**: `Conv2d(4, 1152, kernel_size=2, stride=2, bias=True)`
- **输出形状**: $[B, 256, 1152]$

最后添加可学习的位置编码 `pos_embed`，维度为 $[1, 256, 1152]$。

### 深度图 ($\mathcal{D}_t$)

深度图通过专用的 Patch 嵌入层进行投影，其配置与 RGB 独立：

```python
self.d_embedder = PatchEmbed(
    input_size=32,      # 深度图输入分辨率 (与 VAE latent 一致)
    patch_size=8,       # 较大的 patch size (RGB 的 4 倍)
    in_channels=1,      # 单通道深度图
    hidden_size=1152    # 输出维度 D
)
```

- **输入尺寸**: $32 \times 32 \times 1$（深度图在 VAE 空间分辨率下处理）
- **Patch 分割**: $(32/8)^2 = 4^2 = 16$ 个 patches
- **投影层**: `Conv2d(1, 1152, kernel_size=8, stride=8, bias=True)`
- **输出形状**: $[B, 16, 1152]$

深度图的位置编码 `d_pos_embed` 通过对 RGB 位置编码进行空间下采样获得（下采样比例为 `d_patch_size // patch_size = 4`），确保空间对齐性。

### 6轴力矩 ($f_t$)

6轴力/力矩信号经过单层线性投影处理：

```python
self.force_embedder = nn.Linear(
    in_features=6,     # [fx, fy, fz, tx, ty, tz]
    out_features=1152  # 统一映射到 D 维
)
```

- **输入维数**: 6 (三轴力 + 三轴力矩)
- **输入形状**: $[B, 1, 6]$
- **投影层**: `Linear(6, 1152, bias=True)`
- **输出形状**: $[B, 1, 1152]$
- **位置编码**: 固定的正弦-余弦位置编码 `f_pos_embed`

力矩信号在去噪过程中以单一 Token 形式参与注意力计算，提供触觉反馈信息。

### 动作 ($A_t$)

动作信号根据 `action_condition` 标志采用不同的 Token 化策略：

当 `action_condition=True` 时（条件生成模式）：
```python
action_input_shape = action_dim * (action_steps + 1)
self.a_embedder = nn.Linear(action_input_shape, hidden_size)
```
- **输入维数**: `action_dim * (action_steps + 1) = 4 × 4 = 16`（MetaWorld 4-DOF 配置）
- **输入形状**: $[B, 1, 16]$（将历史动作与当前动作拼接）
- **输出形状**: $[B, 1, 1152]$（单个条件 Token）

当 `action_condition=False` 时（序列预测模式）：
- **输入形状**: $[B, action_steps, action_dim] = [B, 3, 4]$
- **输出形状**: $[B, 3, 1152]$（多个预测 Token）

动作位置编码 `a_pos_embed` 可配置为：
- **可学习模式**: `learnable_action_pos=True` → 随机初始化并端到端优化
- **固定模式**: `learnable_action_pos=False` → 使用 1D 正弦-余弦位置编码

---

## 多模态 Token 拼接与模态标识

在 forward 过程中，各模态 Token 按固定顺序拼接：

```python
# 模态顺序：RGB → Action → Force → Depth
x = torch.cat([rgb_tokens, action_tokens, force_tokens, depth_tokens], dim=1)
```

为支持 AdaMN (Adaptive Modality-Normalization) 和 MoE 模态感知路由，系统为每个 Token 分配模态标识 `modality_ids`：

| 模态 | modality_id | Token 数量 |
|------|-------------|------------|
| RGB | 0 | 256 |
| Action | 1 | 1 / action_steps |
| Force | 2 | 1 |
| Depth | 3 (或 2) | 16 |

```python
modality_ids = torch.zeros((B, N), device=x.device, dtype=torch.long)
modality_ids[:, start_idx[1]:end_idx[1]] = 1  # Action
modality_ids[:, start_idx[2]:end_idx[2]] = 2  # Force
modality_ids[:, start_idx[3]:end_idx[3]] = 3  # Depth
```

---

## 维度变换流向图

```
输入模态                      嵌入层                    统一潜空间
─────────────────────────────────────────────────────────────────────────
RGB:  [B, 3, 256, 256]        VAE: 8× down              [B, 256, 1152]
                              → [B, 4, 32, 32]
                              PatchEmbed(2×2)
                              → 256 patches

Depth: [B, 1, 32, 32]         PatchEmbed(8×8)           [B, 16, 1152]
                              → 16 patches

Force: [B, 1, 6]              Linear(6→1152)            [B, 1, 1152]

Action: [B, 1, 16]            Linear(16→1152)           [B, 1, 1152]
      (条件模式)
─────────────────────────────────────────────────────────────────────────
拼接后: [B, 274, 1152]        (256 + 16 + 1 + 1)
```

---

## 参考文献

[cite: 350] Peebles & Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
