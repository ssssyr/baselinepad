# DeMUSE 架构完整超参数清单

> 本文档为 ICML 论文附录准备，详尽列出 DeMUSE (Deep Multimodal Unified Sparse Experts) 架构的所有超参数及其代码位置。

---

## 目录

1. [模型架构 (Model Architecture)](#1-模型架构-model-architecture)
2. [优化与训练 (Optimization & Training)](#2-优化与训练-optimization--training)
3. [扩散过程 (Diffusion Specifics)](#3-扩散过程-diffusion-specifics)
4. [数据与环境 (Data & Environment)](#4-数据与环境-data--environment)
5. [硬件与软件 (Hardware & Software)](#5-硬件与软件-hardware--software)

---

## 1. 模型架构 (Model Architecture)

### 1.1 Transformer 全局架构

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `hidden_size` ($D$) | 1152 | int | `models.py:394`, `configs/metaworld_4d.yaml:14` |
| `depth` | 28 | int | `models.py:395`, DiT-XL 配置 |
| `num_heads` | 16 | int | `models.py:396` |
| `mlp_ratio` | 4.0 | float | `models.py:397`, `models.py:215` |
| `dropout` | 0 | float | 未使用 (无显式 dropout) |
| `activation_function` | GELU | str | `models.py:275` (`approx_gelu()`) |
| `attention_bias` | True | bool | `models.py:257` (`qkv_bias=True`) |

**激活函数详情**：
```python
# models.py:275
approx_gelu = lambda: nn.GELU(approximate="tanh")
```

### 1.2 MoE (混合专家) 配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `use_moe` | True | bool | `configs/metaworld_4d.yaml:103` |
| `num_experts` | 4 | int | `configs/metaworld_4d.yaml:104` |
| `moe_top_k` | 1 | int | `configs/metaworld_4d.yaml:105` |
| `n_shared_experts` | 4 | int | `configs/metaworld_4d.yaml:109` |
| `aux_loss_alpha` | 0.01 | float | `configs/metaworld_4d.yaml:106` |
| `router_z_loss_weight` | 0.001 | float | `configs/metaworld_4d.yaml:107` |
| `moe_start_layer` | 14 | int | `configs/metaworld_4d.yaml:108` |
| `use_modality_bias` | True | bool | `configs/metaworld_4d.yaml:110` |
| `modality_bias_strength_action` | 0.5 | float | `configs/metaworld_4d.yaml:111` |
| `collect_stats` | True | bool | `configs/metaworld_4d.yaml:112` |

**MoE 中间层维度**：
```python
# moe_blocks.py:215
intermediate_size = int(mlp_ratio * embed_dim)  # 4 * 1152 = 4608
```

**共享专家中间层维度**：
```python
# moe_blocks.py:237
shared_intermediate = embed_dim * n_shared_experts  # 1152 * 4 = 4608
```

### 1.3 嵌入层配置

#### RGB 图像嵌入
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `image_size` | 256 | int | `configs/metaworld_4d.yaml:15` |
| `latent_size` | 32 | int | `train_robot.py:452` (256÷8) |
| `patch_size` (RGB) | 2 | int | `models.py:392` |
| `latent_dim` (VAE) | 4 | int | `train_robot.py:167` (SD VAE) |
| `x_embedder.in_channels` | 4 | int | `models.py:444` |
| `x_embedder.out_channels` | 1152 | int | `models.py:446` |
| `num_patches` (RGB) | 256 | int | (32÷2)² |

#### 深度图嵌入
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `use_depth` | False | bool | `configs/metaworld_4d.yaml:74` |
| `d_hidden_size` | 32 | int | `configs/metaworld_4d.yaml:75` |
| `d_patch_size` | 8 | int | `configs/metaworld_4d.yaml:76` |
| `d_embedder_channels` | 1 | int | `models.py:461` |
| `d_num_patches` | 16 | int | `models.py:467` (32÷8)² |

#### 力矩嵌入
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `use_force` | False | bool | `configs/metaworld_4d.yaml:88` |
| `force_dim` | 6 | int | `configs/metaworld_4d.yaml:89` |
| `force_embedder.in` | 6 | int | `models.py:458` |
| `force_embedder.out` | 1152 | int | `models.py:458` |

#### 动作嵌入
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `action_steps` | 3 | int | `configs/metaworld_4d.yaml:80` |
| `action_dim` | 4 | int | `configs/metaworld_4d.yaml:81` |
| `action_scale` | 1.0 | float | `configs/metaworld_4d.yaml:82` |
| `absolute_action` | True | bool | `configs/metaworld_4d.yaml:83` |
| `action_condition` | True | bool | `configs/metaworld_4d.yaml:84` |
| `learnable_action_pos` | False | bool | `configs/metaworld_4d.yaml:85` |
| `a_embedder.in` | 16 | int | `models.py:450` (4×4) |
| `a_embedder.out` | 1152 | int | `models.py:451` |

### 1.4 AdaMN (自适应模态规范化) 配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `use_adamn` | True | bool | `configs/metaworld_4d.yaml:93` |
| `elementwise_affine` | True | bool | `models.py:245` |
| `eps` | 1e-6 | float | `models.py:245` |
| `num_modalities` | 2 | int | `models.py:431-432` (RGB + Action) |

**AdaMN LayerNorm 结构**：
```python
# models.py:244-250
self.norm1_experts = nn.ModuleList([
    nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
    for _ in range(num_modalities)
])
self.norm2_experts = nn.ModuleList([
    nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
    for _ in range(num_modalities)
])
```

### 1.5 位置编码配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `pos_embed.shape` | [1, 256, 1152] | tensor | `models.py:479` |
| `a_pos_embed.shape` | [1, 1, 1152] | tensor | `models.py:454` |
| `f_pos_embed.shape` | [1, 1, 1152] | tensor | `models.py:459` |
| `d_pos_embed.shape` | [1, 16, 1152] | tensor | `models.py:470` |

### 1.6 注意力掩码配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `attn_mask` | True | bool | `configs/metaworld_4d.yaml:63` |
| RGB 掩码 | 自注意力 | str | `models.py:494` (不能看后续 token) |
| Force 掩码 | 单向 | str | `models.py:495-504` (仅看自己) |

### 1.7 最终层配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `final_layer.linear.in` | 1152 | int | `models.py:324` |
| `final_layer.linear.out` | 32 | int | `models.py:324` (2²×4×2) |
| `a_linear.in` | 1152 | int | `models.py:342` |
| `a_linear.out` | 8 | int | `models.py:341` (4×2) |
| `d_linear.in` | 1152 | int | `models.py:352` |
| `d_linear.out` | 128 | int | `models.py:352` (8²×3×2) |

---

## 2. 优化与训练 (Optimization & Training)

### 2.1 优化器配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `optimizer_type` | AdamW | str | `train_robot.py:548` |
| `learning_rate` | 1e-4 | float | `configs/metaworld_4d.yaml:28` |
| `weight_decay` | 0.0 | float | `configs/metaworld_4d.yaml:29` |
| `adam_beta1` | 0.9 | float | `configs/metaworld_4d.yaml:30` |
| `adam_beta2` | 0.999 | float | `configs/metaworld_4d.yaml:31` |
| `fused` | True | bool | `train_robot.py:547` (A100 优化) |

**优化器初始化**：
```python
# train_robot.py:541-548
adamw_kwargs = dict(lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999))
if hasattr(torch.optim.AdamW, "fused"):
    adamw_kwargs["fused"] = True
opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
```

### 2.2 学习率调度器配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `use_lr_scheduler` | True | bool | `configs/metaworld_4d.yaml:34` |
| `scheduler_type` | cosine | str | `configs/metaworld_4d.yaml:35` |
| `warmup_steps` | 8000 | int | `configs/metaworld_4d.yaml:36` |
| `min_lr_ratio` | 0.01 | float | `configs/metaworld_4d.yaml:37` |
| `total_steps` | ~300,000 | int | `train_robot.py:576` (500 epochs × 19万样本 ÷ 64) |

**Cosine 调度器公式**：
```python
# train_robot.py:580-587
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return 1.0  # 恒定学习率
    else:
        progress = (current_step - warmup_steps) / cosine_steps
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(π * progress))
```

### 2.3 训练循环配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `epochs` | 500 | int | `configs/metaworld_4d.yaml:21` |
| `global_batch_size` | 64 | int | `configs/metaworld_4d.yaml:22` |
| `per_device_batch_size` | 16 | int | `train_robot.py:554` (64÷4 GPUs) |
| `num_workers` | 16 | int | `configs/metaworld_4d.yaml:24` |
| `prefetch_factor` | 4 | int | `train_robot.py:560` |
| `persistent_workers` | True | bool | `train_robot.py:559` |
| `drop_last` | True | bool | `train_robot.py:558` |
| `pin_memory` | True | bool | `train_robot.py:557` |

### 2.4 正则化配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `ema_decay` | 0.9999 | float | `train_robot.py:69` |
| `grad_clip_norm` | 未使用 | - | - |
| `label_smoothing` | 未使用 | - | - |
| `action_loss_lambda` | 2.0 | float | `configs/metaworld_4d.yaml:96` |
| `action_loss_start` | 0 | int | `configs/metaworld_4d.yaml:97` |

**EMA 更新**：
```python
# train_robot.py:69-75
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
```

### 2.5 日志与检查点配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `log_every` | 100 | int | `configs/metaworld_4d.yaml:40` |
| `eval_every` | 50000 | int | `configs/metaworld_4d.yaml:41` |
| `ckpt_every` | 10000 | int | `configs/metaworld_4d.yaml:42` |
| `ckpt_wrapper` | False | bool | `configs/metaworld_4d.yaml:43` |
| `save_model_only` | True | bool | `configs/metaworld_4d.yaml:44` |
| `without_ema` | False | bool | `configs/metaworld_4d.yaml:25` |

### 2.6 梯度追踪配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `track_expert_gradients` | False | bool | `train_robot.py:608` |
| `gradient_track_interval` | 1000 | int | `train_robot.py:755` |
| `collect_stats` | True | bool | `configs/metaworld_4d.yaml:112` |

---

## 3. 扩散过程 (Diffusion Specifics)

### 3.1 时间步配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `num_diffusion_steps` (训练) | 1000 | int | `diffusion/__init__.py:18`, `train_robot.py:524` |
| `sampling_steps` (评估) | 250 | int | `train_robot.py:525` |
| `timestep_respacing` | "" | str | `train_robot.py:524` (训练时全步数) |

### 3.2 噪声调度配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `noise_schedule` | linear | str | `diffusion/__init__.py:12`, `train_robot.py:524` |
| `beta_start` | 0.0001 | float | `diffusion/gaussian_diffusion.py:112` |
| `beta_end` | 0.02 | float | `diffusion/gaussian_diffusion.py:113` |

**Linear Beta 调度**：
```python
# diffusion/gaussian_diffusion.py:106-114
if schedule_name == "linear":
    scale = 1000 / num_diffusion_timesteps
    return get_beta_schedule(
        "linear",
        beta_start=scale * 0.0001,
        beta_end=scale * 0.02,
        num_diffusion_timesteps=num_diffusion_timesteps,
    )
```

### 3.3 预测类型配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `model_mean_type` | EPSILON | enum | `diffusion/__init__.py:33` |
| `model_var_type` | LEARNED_RANGE | enum | `diffusion/__init__.py:36-42` |
| `loss_type` | MSE | enum | `diffusion/__init__.py:26` |
| `learn_sigma` | True | bool | `diffusion/__init__.py:16` |

### 3.4 联合去噪配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `predict_horizon` ($H$) | 3 | int | `configs/metaworld_4d.yaml:17` |
| `action_steps` ($K$) | 3 | int | `configs/metaworld_4d.yaml:80` |
| `skip_step` | 4 | int | `configs/metaworld_4d.yaml:18` |
| `dynamics` | True | bool | `configs/metaworld_4d.yaml:66` |

### 3.5 采样配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `clip_denoised` | False | bool | `train_robot.py:1044` |
| `progress` | True | bool | `train_robot.py:1045` |
| `eta` (DDIM) | 0.0 | float | `diffusion/gaussian_diffusion.py:574` (确定性采样) |

---

## 4. 数据与环境 (Data & Environment)

### 4.1 预处理配置

#### RGB 图像
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `image_resolution` | 256×256 | int | `configs/metaworld_4d.yaml:15` |
| `VAE_scale_factor` | 0.18215 | float | `train_robot.py:1099` (解码) |
| `VAE_compression` | 8× | int | 256 → 32 |
| `latent_channels` | 4 | int | SD VAE |

#### 深度图
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `depth_resolution` | 32×32 | int | `datasets/dataset.py:452` |
| `depth_filter` | False | bool | `configs/metaworld_4d.yaml:77` |
| `depth_clip_range` | [1000, 5000] | int | `datasets/dataset.py:456` (filter2) |
| `medianBlur_kernel` | 15 | int | `datasets/dataset.py:458` |

**深度图滤波 (filter2)**：
```python
# datasets/dataset.py:455-459
def filter2(depth):
    depth = np.clip(depth, 1000, 5000) / 5000
    depth = np.array(depth * 256, dtype=np.uint8)
    depth = cv2.medianBlur(depth, 15)
    return cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST) / 256
```

#### 力矩数据
| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `force_dim` | 6 | int | `[fx, fy, fz, tx, ty, tz]` |
| `force_clip_force` | ±100 N | float | `datasets/dataset.py:68` |
| `force_clip_torque` | ±10 Nm | float | `datasets/dataset.py:69` |
| `force_default_mean` | [0,0,0,0,0,0] | float | `datasets/dataset.py:46` |
| `force_default_std` | [1,1,1,1,1,1] | float | `datasets/dataset.py:46` |

**力矩归一化**：
```python
# datasets/dataset.py:49-75
def normalize_force(force, mean, std):
    force[:3] = np.clip(force[:3], -100, 100)   # fx, fy, fz (牛顿)
    force[3:] = np.clip(force[3:], -10, 10)     # tx, ty, tz (牛·米)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (force - mean) / std
    return normalized
```

### 4.2 采样率配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `control_frequency` | 20 Hz | float | 推断值 (skip_step=4, 50fps视频) |
| `skip_step` | 4 | int | `configs/metaworld_4d.yaml:18` |
| `sensor_sync_rate` | 同步 | - | 深度/RGB 同步采集 |

### 4.3 数据增强配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `random_crop` | 未使用 | - | - |
| `horizontal_flip` | 未使用 | - | - |
| `color_jitter` | 未使用 | - | - |

### 4.4 文本嵌入配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `text_cond` | True | bool | `configs/metaworld_4d.yaml:69` |
| `clip_path` | CLIP ViT-B/32 | str | `configs/metaworld_4d.yaml:70` |
| `text_emb_size` | 512 | int | `configs/metaworld_4d.yaml:71` |
| `y_embedder.in` | 512 | int | `models.py:204` |
| `y_embedder.out` | 1152 | int | `models.py:204` |

---

## 5. 硬件与软件 (Hardware & Software)

### 5.1 设备配置

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `GPU_type` | A100 (推测) | str | `train_robot.py:31-32` (TF32 优化) |
| `num_devices` | 4 | int | `global_batch_size=64`, 推测 4 GPUs |
| `precision` | FP32/TF32 | str | `train_robot.py:31-32` |
| `mixed_precision` | False | bool | 未使用 torch.cuda.amp |

**TF32 优化**：
```python
# train_robot.py:31-32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 5.2 库版本配置

| 库 | 版本要求 | 代码位置 |
|------|----------|----------|
| PyTorch | >=1.12 | TF32 支持 |
| diffusers | - | `diffusers.models.AutoencoderKL` |
| accelerate | - | `train_robot.py:26` |
| PIL (Pillow) | - | `train_robot.py:21` |
| numpy | - | `train_robot.py:20` |
| opencv-python | - | `datasets/dataset.py:17` |

### 5.3 数据加载优化

| 参数 | 数值 | 类型 | 代码位置 |
|------|------|------|----------|
| `pin_memory` | True | bool | `train_robot.py:557` |
| `persistent_workers` | True | bool | `train_robot.py:559` |
| `prefetch_factor` | 4 | int | `train_robot.py:560` |
| `drop_last` | True | bool | `train_robot.py:558` |

---

## 附录 A: LaTeX 兼容的参数名称

为便于论文撰写，以下参数名称已做 LaTeX 转义处理：

```latex
% 模型架构
\texttt{hidden\_size} = 1152
\texttt{depth} = 28
\texttt{num\_heads} = 16
\texttt{mlp\_ratio} = 4.0
\texttt{patch\_size} = 2
\texttt{d\_patch\_size} = 8

% MoE 配置
\texttt{num\_experts} = 4
\texttt{moe\_top\_k} = 1
\texttt{n\_shared\_experts} = 4
\texttt{aux\_loss\_alpha} = 0.01

% 训练配置
\texttt{learning\_rate} = 10^{-4}
\texttt{weight\_decay} = 0.0
\texttt{adam\_beta1} = 0.9
\texttt{adam\_beta2} = 0.999
\texttt{ema\_decay} = 0.9999
\texttt{warmup\_steps} = 8000

% 扩散配置
\texttt{num\_diffusion\_steps} = 1000
\texttt{sampling\_steps} = 250
\texttt{predict\_horizon} = 3
\texttt{action\_steps} = 3
\texttt{skip\_step} = 4

% 数据配置
\texttt{image\_size} = 256
\texttt{latent\_size} = 32
\texttt{action\_dim} = 4
\texttt{force\_dim} = 6
```

---

## 附录 B: 模型参数量统计

| 组件 | 参数量 | 计算方式 |
|------|--------|----------|
| Transformer Block | ~300M | 28 × (2 × 1152² × 4 + 4 × 1152²) |
| x_embedder | ~3.6M | 4 × 1152 × 2² |
| 共享专家 (4个) | ~85M | 28 × 4 × (1152×4608 + 4608×1152) |
| 路由专家 (4个) | ~85M | 28 × 4 × (1152×4608 + 4608×1152) |
| 总计 | ~600M | (DiT-XL/2 基座) |

---

## 附录 C: Token 序列结构

| 模态 | Token 数量 | 位置索引 |
|------|-----------|----------|
| RGB | 256 | [0, 256) |
| Action | 1 | [256, 257) |
| Depth | 16 | [257, 273) |
| **总计** | **273** | - |

---

## 参考文献

1. Peebles & Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
2. Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." 2017.
3. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022.
