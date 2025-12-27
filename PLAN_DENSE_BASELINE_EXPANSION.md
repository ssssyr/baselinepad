# Dense Baseline 参数量扩展计划

## 目标
扩展当前 Dense DiT-XL/2 模型的参数量，使其与 MoE 版本 (~1363.63M) 接近，作为对比实验的 baseline。

---

## 用户选择的最终配置

```python
hidden_size = 1536
depth = 32
mlp_ratio = 4.0
num_heads = 24  # head_dim = 1536 / 24 = 64
patch_size = 2
```

### 参数量分析

| 指标 | 数值 |
|------|------|
| **Dense Extended 参数量** | **1372.92M** |
| MoE 版本参数量 | 1363.63M |
| **差异** | **+9.29M (+0.68%)** |
| 当前 Dense (XL/2) | 677.05M |
| **扩展倍数** | **2.03x** |

### 显存估算 (训练时)

| 组件 | 显存占用 |
|------|----------|
| 模型参数 | 5.36 GB |
| 梯度 | 5.36 GB |
| 优化器状态 (AdamW) | 10.73 GB |
| EMA 模型 | 5.36 GB |
| 批次数据 (batch=64) | 0.06 GB |
| **总训练内存** | **~26.88 GB** |

**GPU 建议**: A100 (40GB) 或 H100 (80GB)

---

## 实现步骤

### Step 1: 在 models.py 中添加扩展模型定义

在 `models.py` 文件的 DiT 配置部分添加：

```python
def DiT_XL_2_Extended(**kwargs):
    """
    Extended Dense Baseline for MoE Comparison: ~1373M parameters

    Configuration:
        hidden_size: 1536
        depth: 32
        mlp_ratio: 4.0
        num_heads: 24 (head_dim = 64)
        patch_size: 2

    This configuration matches the MoE version (1363.63M) within 0.68%,
    providing a fair comparison baseline.

    Compared to original DiT-XL/2 (677M):
        - 2.03x parameters
        - Deeper network (32 vs 28 layers)
        - Wider hidden dimension (1536 vs 1152)
    """
    return DiT(depth=32, hidden_size=1536, patch_size=2, num_heads=24, mlp_ratio=4.0, **kwargs)
```

### Step 2: 更新 DiT_models 字典

```python
DiT_models = {
    # === Original DiT configs ===
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,

    # === Extended Dense Baseline (for MoE comparison) ===
    'DiT-XL/2-Extended': DiT_XL_2_Extended,  # ~1373M
}
```

### Step 3: 创建配置文件

创建 `configs/metaworld_4d_dense_extended.yaml`:

```yaml
# Configuration for Dense Extended Baseline (for fair MoE comparison)

# ======================================================
#               General Training Settings
# ======================================================
training:
  feature_path: "/home/ct_24210860031/812datasets/SYR/metaworld_corner3_all_with_force_features"
  video_path: null
  results_dir: "results_metaworld_4d_dense_extended"

  # Model - Extended Dense Baseline
  model: "DiT-XL/2-Extended"  # ~1373M parameters
  image_size: 256
  num_classes: 1000
  predict_horizon: 3
  skip_step: 4

  # Training Loop
  epochs: 1500
  global_batch_size: 64
  global_seed: 42
  num_workers: 8
  without_ema: false

  # Optimizer Settings
  learning_rate: 5e-5
  weight_decay: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.999

  # Learning Rate Scheduler
  use_lr_scheduler: true
  scheduler_type: "constant"
  warmup_steps: 8000
  min_lr_ratio: 1.0

  # Logging
  log_every: 100
  eval_every: 50000
  ckpt_every: 10000
  ckpt_wrapper: false
  save_model_only: true

  # Resume
  resume: null
  auto_resume: true

# ======================================================
#                 Component Settings
# ======================================================
components:
  vae: "ema"
  vae_path: "/home/ct_24210860031/812code/SYR/models/sd-vae-ft-mse"

  dit_init: null
  rgb_init: "/home/ct_24210860031/812code/SYR/models/pad_bridge_pre/best_action_loss.pt"

  attn_mask: false
  dynamics: true

  text_cond: true
  clip_path: "/home/ct_24210860031/812code/SYR/models/clip-vit-base-patch32"
  text_emb_size: 512

  use_depth: false
  d_hidden_size: 32
  d_patch_size: 8
  depth_filter: false

  action_steps: 3
  action_dim: 4
  action_scale: 1
  absolute_action: true
  action_condition: true
  learnable_action_pos: false

  use_force: true
  force_dim: 6
  force_stats_path: null

  action_loss_lambda: 2.0
  action_loss_start: 0

# ======================================================
#                  MoE Settings (DISABLED)
# ======================================================
moe:
  use_moe: false  # DENSE baseline, no MoE
  num_experts: 4
  moe_top_k: 2
  aux_loss_weight: 0.01
  router_z_loss_weight: 0.001

# ======================================================
#                  WandB Logging
# ======================================================
wandb:
  use_wandb: true
  wandb_project: "metaworld-dense-baseline-comparison"
  wandb_run_name: "metaworld_4d_dense_extended"

# ======================================================
#               MetaWorld Environment Settings
# ======================================================
metaworld:
  task_names: ["button-press-v2"]
  data_path: ""
  image_size: 256
  use_depth: false
  camera_name: "corner3"
```

---

## 训练命令

```bash
# 使用配置文件训练
python train_robot.py --config configs/metaworld_4d_dense_extended.yaml
```

或通过命令行参数覆盖：

```bash
python train_robot.py \
    --config configs/metaworld_4d.yaml \
    training.model "DiT-XL/2-Extended" \
    training.results_dir "results_metaworld_4d_dense_extended" \
    wandb.wandb_run_name "dense_extended_baseline"
```

---

## 实验对比表格

| 模型 | hidden_size | depth | mlp_ratio | num_heads | 参数量 (M) |
|------|-------------|-------|-----------|-----------|------------|
| **MoE 版本** | 1152 | 28 | 4.0 | 16 | **1363.63** |
| **Dense Extended (新)** | **1536** | **32** | **4.0** | **24** | **1372.92** |
| Dense Baseline (原始) | 1152 | 28 | 4.0 | 16 | 677.05 |

### 关键对比指标

| 指标 | Dense Extended | MoE 版本 |
|------|----------------|----------|
| 总参数量 | 1372.92M | 1363.63M |
| 实际计算量 | 100% (dense) | ~50% (sparse, top-k=2/4) |
| 内存占用 (训练) | ~26.88 GB | ~25.78 GB |
| 收敛速度 | ? | ? |
| 最终性能 | ? | ? |

---

## 参数量详细分解

### Dense Extended (hidden_size=1536, depth=32)

| 组件 | 参数量 |
|------|--------|
| Patch 嵌入 | 26,112 |
| 时间嵌入 | 2,360,448 |
| 文本嵌入 | 787,200 |
| 动作嵌入 | 26,112 |
| **DiT Blocks (32层)** | **1,360,630,272** |
| - Attention (每层) | 10,621,824 |
| - MLP (每层) | 37,748,736 |
| - AdaLN (每层) | 14,155,712 |
| RGB 输出层 | 3,679,872 |
| 动作预测头 | 3,568,200 |
| **总计** | **1,372,915,416 (~1372.92M)** |

---

## 注意事项

1. **权重初始化**
   - 由于 hidden_size 和 depth 与预训练模型不同，无法直接加载权重
   - 可以考虑部分加载（如从预训练模型加载部分层的权重）
   - 或者从头训练

2. **训练稳定性**
   - 更深的网络可能需要更长的 warmup
   - 考虑使用 gradient clipping
   - 监控 loss 曲线，必要时调整学习率

3. **评估指标**
   - Action prediction loss
   - Image reconstruction quality (FID, etc.)
   - Sample efficiency (达到相同性能所需步数)

4. **公平性保证**
   - 使用相同的训练数据
   - 相同的 batch size 和学习率
   - 相同的评估流程

---

## 实验检查清单

- [ ] 在 `models.py` 中添加 `DiT_XL_2_Extended` 函数
- [ ] 更新 `DiT_models` 字典
- [ ] 创建配置文件 `configs/metaworld_4d_dense_extended.yaml`
- [ ] 运行验证脚本确认参数量
- [ ] 开始训练
- [ ] 记录训练曲线和最终性能
- [ ] 与 MoE 版本对比分析
