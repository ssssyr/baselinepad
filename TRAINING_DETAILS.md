# 训练细节文档 (Training Details)

> 本文档提供 Uni-Embodied 模型的完整训练设置，用于论文撰写

---

## 1. 总体训练设置

### 1.1 基础训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练轮数 (Epochs) | 1500 | ~300,000 步 |
| 全局批大小 (Global Batch Size) | 64 | 4 GPU × 16 local batch |
| 学习率 (Learning Rate) | 5×10⁻⁵ | AdamW 初始学习率 |
| 优化器 (Optimizer) | AdamW | β₁=0.9, β₂=0.999 |
| 权重衰减 (Weight Decay) | 0.0 | L2 正则化 |
| 梯度累积 (Gradient Accumulation) | 1 | 无累积 |
| 混合精度 (Mixed Precision) | FP16 | 由 Accelerate 自动启用 |
| 分布式训练 (Distributed) | DDP | 4 GPU |

### 1.2 学习率调度

```python
# 配置: scheduler_type = "constant"
# 学习率保持常数，不进行衰减

lr(t) = 5×10⁻⁵  # 恒定

# 可选: cosine annealing (未使用)
# warmup_steps = 8000
# min_lr_ratio = 1.0  # 不衰减
```

**特点**：
- 无学习率衰减
- 无 warmup（配置中 `min_lr_ratio=1.0`）
- 稳定训练，适合大规模预训练

---

## 2. 损失函数 (Loss Function)

### 2.1 总体损失公式

$$
\mathcal{L} = \mathcal{L}_{\text{image}} + \lambda_a \cdot \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{depth}} + \alpha \cdot \mathcal{L}_{\text{aux}}
$$

其中：
- $\mathcal{L}_{\text{image}}$: RGB 图像重建损失
- $\mathcal{L}_{\text{action}}$: 动作预测损失（$\lambda_a=2.0$）
- $\mathcal{L}_{\text{depth}}$: 深度预测损失（可选）
- $\mathcal{L}_{\text{aux}}$: MoE 辅助损失（$\alpha=0.01$）

---

### 2.2 RGB 图像损失

#### 2.2.1 扩散过程

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中：
- $x_0$: 原始图像（VAE latent）
- $x_t$: $t$ 时刻的噪声图像
- $\bar{\alpha}_t$: 累积噪声调度参数

#### 2.2.2 模型预测

模型输出 $(\mu_\theta, \sigma_\theta^2)$，预测噪声 $\epsilon$：

$$
\mathcal{L}_{\text{mse}} = \mathbb{E}_{t, \epsilon, x_0} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|_2^2 \right]
$$

#### 2.2.3 变分下界 (VB Loss)

$$
\mathcal{L}_{\text{vb}} = \text{KL}\left( q_\phi(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t) \right)
$$

实现中，方差参数使用冻结梯度（`detach()`）不影响均值预测：

```python
model_output, model_var_values = th.split(model_output, C, dim=1)
frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
vb_loss = self._vb_terms_bpd(model=lambda *args, r=frozen_out: r, ...)
```

#### 2.2.4 总图像损失

$$
\mathcal{L}_{\text{image}} = \mathcal{L}_{\text{mse}} + \mathcal{L}_{\text{vb}}
$$

---

### 2.3 动作损失 (Action Loss)

#### 2.3.1 动作扩散

动作与 RGB 共享同一扩散时间步 $t$：

$$
a_t = \sqrt{\bar{\alpha}_t} \cdot a_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_a, \quad \epsilon_a \sim \mathcal{N}(0, I)
$$

#### 2.3.2 动作预测

```python
# 动作输出形状: (B, C, a_dim * 2)
action_output = model(x_t, t, noised_action=a_t, ...)
action_pred, action_var = th.split(action_output, a_dim, dim=2)
```

#### 2.3.3 动作 MSE 损失

$$
\mathcal{L}_{\text{action\_mse}} = \mathbb{E}_{t, \epsilon_a, a_0} \left[ \| \epsilon_a - \epsilon_\theta^a(x_t, t, y) \|_2^2 \right]
$$

#### 2.3.4 动作 VB 损失

$$
\mathcal{L}_{\text{action\_vb}} = \text{KL}\left( q_\phi(a_{t-1}|a_t, a_0) \,\|\, p_\theta(a_{t-1}|a_t) \right)
$$

#### 2.3.5 总动作损失

$$
\mathcal{L}_{\text{action}} = \mathcal{L}_{\text{action\_mse}} + \mathcal{L}_{\text{action\_vb}}
$$

#### 2.3.6 渐进式训练

```python
# train_robot.py:596-597
a_coeff = 1.0 if train_steps > args.action_loss_start else 0.0
loss = loss + loss_a * args.action_loss_lambda * a_coeff
```

- `action_loss_start = 0`: 从第 0 步开始训练动作
- `action_loss_lambda = 2.0`: 动作损失权重

---

### 2.4 深度损失 (Depth Loss, 可选)

深度损失与动作损失结构相同：

$$
\mathcal{L}_{\text{depth}} = \mathcal{L}_{\text{depth\_mse}} + \mathcal{L}_{\text{depth\_vb}}
$$

当前配置中 `use_depth = False`，未启用。

---

## 3. MoE 损失与参数

### 3.1 MoE 架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_experts` | 4 | 路由专家数量 |
| `moe_top_k` | 2 | 每个 token 选择的专家数 |
| `moe_start_layer` | 14 | 从第 14 层开始使用 MoE |
| `shared_experts` | 4 | 共享专家数量 |

### 3.2 MoE 辅助损失 (Auxiliary Loss)

#### 3.2.1 负载均衡损失

```python
# moe_blocks.py:95-113
# 计算专家利用率 (ce)
mask_ce = F.one_hot(topk_idx, num_classes=num_experts)
ce = mask_ce.float().mean(0)  # (num_experts,)

# 计算路由概率 (pi)
pi = scores.softmax(dim=-1).mean(0)  # (num_experts,)

# 负载均衡损失
fi = ce * num_experts  # 目标: 均匀分布
aux_loss = (pi * fi).sum() * alpha
```

#### 3.2.2 数学形式

$$
\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{N} f_i \cdot p_i
$$

其中：
- $N=4$: 专家数量
- $f_i = N \cdot c_i$: 专家 $i$ 的利用率（归一化）
- $p_i$: 路由到专家 $i$ 的平均概率
- $\alpha = 0.01$: 辅助损失权重

**目标**：鼓励均匀利用所有专家，避免专家塌陷。

#### 3.2.3 模态感知路由 (可选)

```python
# moe_blocks.py:62-65
if use_modality_bias and modality_ids is not None:
    logits = logits + modality_bias[modality_ids]
```

当前配置中 `use_modality_bias = False`，未启用。

---

### 3.3 Expert 初始化

从预训练 dense 模型初始化 MoE 专家：

```python
# train_robot.py:255-284
# 共享专家: 直接复制 dense FFN 权重
shared.fc1.weight = dense.fc1.weight
shared.fc2.weight = dense.fc2.weight

# 路由专家: 添加小噪声打破对称性
expert.fc1.weight = dense.fc1.weight + 𝒩(0, 10⁻³)
expert.fc2.weight = dense.fc2.weight + 𝒩(0, 10⁻³)
```

---

## 4. 数据加载与预处理

### 4.1 数据集配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集大小 | ~190,000 轨迹 | MetaWorld MT50 |
| 每条轨迹长度 | ~100 步 | 可变长度 |
| 图像分辨率 | 256×256 | 训练时使用 |
| VAE Latent | 32×32×4 | 8× 下采样 |

### 4.2 DataLoader 设置

```python
DataLoader(
    dataset,
    batch_size=16,  # 每个GPU
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

### 4.3 力信号归一化

```python
# datasets/dataset.py
force_normalized = (force - force_mean) / force_std
```

统计量从 `force_stats.json` 加载（Welford 在线算法计算）。

---

## 5. EMA (Exponential Moving Average)

### 5.1 EMA 更新

```python
# train_robot.py:489, 652-653
ema = deepcopy(model)
update_ema(ema, model, decay=0.95)  # 每步更新

# 首次同步
update_ema(ema, model, decay=0)
```

### 5.2 EMA 衰减率

$$
\theta_{\text{EMA}}^{(t)} = \rho \cdot \theta_{\text{EMA}}^{(t-1)} + (1-\rho) \cdot \theta^{(t)}
$$

其中 $\rho = 0.95$。

---

## 6. 训练稳定性技巧

### 6.1 NaN 处理

```python
# diffusion/gaussian_diffusion.py:881-884
if th.isnan(terms["loss"]).any():
    terms["loss"] = th.where(
        th.isnan(terms["loss"]),
        th.zeros_like(terms["loss"]),
        terms["loss"]
    )
```

### 6.2 动作 Token 排除

```python
# moe_blocks.py:87-93
# 排除 action tokens (modality_id == 1) 的负载均衡损失
keep_mask = flat_modality != 1
scores_for_aux = scores_for_aux[keep_mask]
```

**原因**：动作 embeddings 不应被路由正则化过度驱动。

### 6.3 梯度检查点 (Gradient Checkpointing)

```python
# train_robot.py:671-673
if args.ckpt_wrapper:
    x = torch.utils.checkpoint.checkpoint(
        self.ckpt_wrapper(block), x, c, modality_ids,
        use_reentrant=False
    )
```

当前配置中 `ckpt_wrapper = False`。

---

## 7. 评估与采样

### 7.1 采样配置

```python
# train_robot.py:413
eval_diffusion = create_diffusion(str(250))  # 250 步采样
```

### 7.2 评估频率

| 事件 | 频率 |
|------|------|
| 日志记录 (Logging) | 每 100 步 |
| 模型评估 (Evaluation) | 每 50,000 步 |
| 检查点保存 (Checkpoint) | 每 10,000 步 |

### 7.3 评估指标

```python
# train_robot.py:800-802
action_mse_error = F.mse_loss(pred_action, gt_action)
img_mse_value = F.mse_loss(pred_img, gt_img)
```

---

## 8. 关键实现细节

### 8.1 Multi-Modal Output Splitting

```python
# diffusion/gaussian_diffusion.py:801-806
model_output = model(x_t, t, **model_kwargs)
if isinstance(model_output, tuple):
    model_output, action_output, depth_output = model_output
else:
    action_output = None
    depth_output = None
```

### 8.2 Variance 学习策略

```python
# diffusion/gaussian_diffusion.py:819
# 冻结 variance 梯度，只优化均值预测
frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
```

这是标准的 IDDPM (Improved DDPM) 实现技巧。

---

## 9. 超参数敏感性分析

### 9.1 动作损失权重

| `action_loss_lambda` | 效果 |
|---------------------|------|
| 0.0 | 无动作学习 |
| 1.0 | 标准（baseline） |
| **2.0** | **当前配置** - 强化动作学习 |
| >3.0 | 可能过度优化动作，影响图像质量 |

### 9.2 MoE 辅助损失权重

| `aux_loss_weight` | 效果 |
|-------------------|------|
| 0.0 | 无负载均衡，专家塌陷风险高 |
| **0.01** | **当前配置** - 平衡负载与质量 |
| 0.1 | 强制均匀分布，可能降低性能 |

### 9.3 MoE Top-K

| `moe_top_k` | 效果 |
|------------|------|
| 1 | 纯稀疏，每个 token 只用 1 个专家 |
| **2** | **当前配置** - 平衡稀疏性与表达能力 |
| 4 | 接近 dense，参数利用率高 |

---

## 10. 论文写作建议

### 10.1 训练设置描述模板

> We train our model for 300,000 gradient steps with a global batch size of 64 across 4 NVIDIA A100 GPUs. We use the AdamW optimizer [1] with a fixed learning rate of 5×10⁻⁵ (β₁=0.9, β₂=0.999) and no weight decay. We maintain an exponential moving average (EMA) of the model parameters with decay 0.95.

### 10.2 损失函数描述模板

> Our model is trained to denoise multi-modal latents via a unified diffusion objective. For RGB images, we optimize the standard diffusion loss [2] augmented with a learned variance term [3]. Actions share the same diffusion timesteps and are trained with a weighted MSE loss (λ=2.0). For MoE layers, we add a load-balancing auxiliary loss (α=0.01) to encourage uniform expert utilization.

### 10.3 MoE 描述模板

> We employ Sparse Mixture of Experts (MoE) [4] in the final 14 transformer layers (layers 14-27). Each MoE layer contains 4 routed experts with top-2 routing and 4 shared experts to maintain model capacity. Experts are initialized from pretrained dense FFN weights with small Gaussian noise (σ=10⁻³) to break symmetry.

---

## 参考文献

[1] Loshchilov & B. "Decoupled Weight Decay Regularization." NeurIPS 2019.

[2] Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.

[3] Nichol & Dhariwal. "Improved Denoising Diffusion Probabilistic Models." ICLR 2021.

[4] Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.

---

## 附录：配置文件摘要

```yaml
# configs/metaworld_4d.yaml 关键参数
training:
  epochs: 1500
  global_batch_size: 64
  learning_rate: 5e-5
  weight_decay: 0.0

  # 调度器
  use_lr_scheduler: true
  scheduler_type: "constant"
  warmup_steps: 8000
  min_lr_ratio: 1.0

components:
  # 模型
  model: "DiT-XL/2"
  predict_horizon: 3
  action_steps: 3
  action_dim: 4

  # 模态
  use_depth: false
  use_force: true
  force_dim: 6
  text_cond: true

  # 损失权重
  action_loss_lambda: 2.0
  action_loss_start: 0

moe:
  use_moe: true
  num_experts: 4
  moe_top_k: 2
  aux_loss_weight: 0.01
  router_z_loss_weight: 0.001
  moe_start_layer: 14
  moe_shared_experts: 4
  use_modality_bias: false
```