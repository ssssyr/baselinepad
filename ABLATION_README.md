# Ablation Study Documentation

## 概述

本文档描述了如何进行消融实验，研究视频（RGB）扩散预测对模型性能的影响。消融实验的目标是对比"视频+动作联合扩散"与"仅动作扩散"的性能差异。

---

## 实验设计

### 正常模式（Baseline）
- **RGB**: 参与扩散预测（加噪 → 去噪 → 计算损失）
- **动作**: 参与扩散预测（加噪 → 去噪 → 计算损失）
- **深度**: 参与扩散预测（如果启用）

### 消融模式（Ablation）
- **RGB**: 仅作为条件（不参与加噪、不预测、不计算损失）
- **动作**: 参与扩散预测（加噪 → 去噪 → 计算损失）
- **深度**: 参与扩散预测（如果启用）

---

## 实现原理

### 核心思路

通过**零填充**的方式保持模型结构不变，同时让 RGB 不参与扩散过程：

1. **输入层**: 用零填充替代加噪的未来帧
2. **损失层**: 跳过 RGB 损失计算
3. **模型结构**: 完全不变，权重兼容

### 为什么用零填充？

```
正常模式输入: x(4T) + x_cond(4) = 4T+4 通道
消融模式输入: zeros(4T) + x_cond(4) = 4T+4 通道
```

这样保持了输入通道数不变，无需修改 `x_embedder` 的卷积权重。

---

## 代码修改详解

### 1. models.py - 消融模式 Forward 逻辑

**位置**: `DiT.forward()` 函数（第 645-667 行）

```python
def forward(self, x, t, y, x_cond=None, ...):
    # 获取消融标志
    ablation_no_rgb = getattr(self.args, 'ablation_no_rgb_diffusion', False)

    if x_cond is not None:
        if ablation_no_rgb:
            # 消融模式：用零填充替代加噪的未来帧
            B = x_cond.shape[0]
            zeros = torch.zeros(B, 4 * self.args.predict_horizon, *x_cond.shape[2:],
                               device=x_cond.device, dtype=x_cond.dtype)
            x = torch.cat([zeros, x_cond], dim=1)  # (B, 4T+4, H, W)
        else:
            # 正常模式：加噪的未来帧 + 当前帧
            x = torch.cat([x, x_cond], dim=1)

    x = self.x_embedder(x) + self.pos_embed
    # ... 后续代码不变
```

**效果**:
- 正常模式: `x` 包含加噪的未来帧
- 消融模式: `x` 的未来帧部分全为零，模型只看 `x_cond`

---

### 2. train_robot.py - 参数和损失控制

#### 2.1 添加命令行参数

**位置**: 参数解析部分（第 1228-1230 行）

```python
# Ablation study
parser.add_argument("--ablation-no-rgb-diffusion", action="store_true",
                    help="Ablation: disable RGB diffusion (RGB as condition only)")
```

#### 2.2 传递消融标志

**位置**: 训练循环（第 688-691 行）

```python
model_kwargs = dict(
    y=y,
    x_cond=x_cond,
    ablation_no_rgb_diffusion=getattr(args, 'ablation_no_rgb_diffusion', False)
)
```

#### 2.3 修改损失计算

**位置**: 训练循环（第 714-733 行）

```python
loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
loss = loss_dict["loss"].mean()

# 获取消融标志
ablation_no_rgb = getattr(args, 'ablation_no_rgb_diffusion', False)

if args.action_steps > 0 and "loss_a" in loss_dict:
    a_coffi = 1.0 if train_steps > args.action_loss_start else 0.0
    if not ablation_no_rgb:
        loss = loss + loss_dict["loss_a"].mean() * args.action_loss_lambda * a_coffi
    else:
        # 消融模式：只计算动作损失，跳过 RGB 损失
        loss = loss_dict["loss_a"].mean() * args.action_loss_lambda * a_coffi

if args.use_depth and "loss_depth" in loss_dict:
    loss = loss + loss_dict["loss_depth"].mean()
```

**效果**:
- 正常模式: `loss = loss_rgb + loss_action + loss_depth`
- 消融模式: `loss = loss_action + loss_depth`

---

### 3. diffusion/gaussian_diffusion.py - 消融支持

**位置**: `training_losses()` 函数（第 767-794 行）

```python
def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
    if model_kwargs is None:
        model_kwargs = {}

    # 接收并移除消融标志（避免传递到模型）
    ablation_no_rgb = model_kwargs.pop('ablation_no_rgb_diffusion', False)

    if noise is None:
        noise = th.randn_like(x_start)

    # 对视频加噪（正常和消融模式都会执行，但消融模式在模型中会用零填充替代）
    x_t = self.q_sample(x_start, t, noise=noise)

    # ... 后续代码不变
```

**说明**: 这里的 `ablation_no_rgb` 标志主要用于未来扩展（如跳过加噪计算以节省算力），当前实现仍在 `models.py` 中通过零填充实现消融。

---

## 使用方法

### 方法 1: 命令行参数

```bash
# 正常训练（RGB 参与扩散）
python train_robot.py --config metaworld_4d.yaml

# 消融实验（RGB 不参与扩散）
python train_robot.py --config metaworld_4d.yaml --ablation-no-rgb-diffusion
```

### 方法 2: 配置文件

在 `configs/metaworld_4d.yaml` 中添加：

```yaml
# Ablation Study - Disable RGB diffusion
ablation_no_rgb_diffusion: true  # RGB as condition only, no prediction
```

然后直接运行：

```bash
python train_robot.py --config metaworld_4d.yaml
```

---

## 实验对比

| 项目 | 正常模式 | 消融模式 |
|------|----------|----------|
| **RGB 加噪** | ✅ | ✅ (但被零填充替代) |
| **RGB 预测** | ✅ | ❌ |
| **RGB 损失** | ✅ | ❌ |
| **动作加噪** | ✅ | ✅ |
| **动作预测** | ✅ | ✅ |
| **动作损失** | ✅ | ✅ |
| **深度加噪** | ✅ | ✅ |
| **深度预测** | ✅ | ✅ |
| **深度损失** | ✅ | ✅ |
| **模型结构** | 完全相同 | 完全相同 |
| **权重兼容** | - | ✅ |

---

## 预期结果分析

### 性能指标对比

| 指标 | 正常模式 | 消融模式 | 差异 |
|------|----------|----------|------|
| **RGB MSE** | 低 | N/A | - |
| **动作 MSE** | 低 | 高 | 联合扩散的收益 |
| **深度 MSE** | 低 | 高 | 联合扩散的收益 |
| **训练速度** | 基准 | 略快 | 跳过 RGB 损失计算 |

### 消融实验结论

如果消融模式下动作 MSE 显著高于正常模式，说明：
- **RGB 扩散预测对动作预测有帮助**
- 联合扩散架构是有效的

如果差异不大，说明：
- **RGB 扩散预测对动作预测帮助有限**
- 可以考虑简化模型架构

---

## 分支管理

### 当前分支结构

```
ablation-unit (消融实验分支)
├── models.py (消融逻辑)
├── train_robot.py (参数和损失控制)
├── diffusion/gaussian_diffusion.py (消融支持)
└── configs/metaworld_4d.yaml (消融配置)

expert-adaln (原分支，未受影响)
└── ...
```

### 云服务器部署

在云服务器上拉取消融分支：

```bash
git fetch origin
git checkout ablation-unit
git pull
```

---

## 常见问题

### Q1: 为什么不直接移除 RGB 预测头？

**A**: 为了保持模型结构完全一致，便于：
1. 加载预训练权重
2. 对比实验的公平性
3. 代码的可维护性

### Q2: 消融模式下 RGB 的零填充会影响性能吗？

**A**: 不会，因为：
1. 零填充部分没有梯度回传
2. 模型学习到忽略这部分输入
3. 计算开销可忽略

### Q3: 如何验证消融模式是否生效？

**A**: 查看训练日志：
- 正常模式: `Train Loss image: 0.xxxx`
- 消融模式: `Train Loss image: 0.000000` (或接近零)

### Q4: 能否同时对多个模态进行消融？

**A**: 可以扩展，例如：
- `--ablation-no-action-diffusion`: 动作不参与扩散
- `--ablation-no-depth-diffusion`: 深度不参与扩散

---

## 修改文件清单

| 文件 | 修改内容 | 行数变化 |
|------|----------|----------|
| `models.py` | 添加消融模式 forward 逻辑 | +12 -1 |
| `train_robot.py` | 添加参数、损失控制、评估支持 | +23 -4 |
| `diffusion/gaussian_diffusion.py` | 添加消融支持 | +12 -1 |
| `configs/metaworld_4d.yaml` | 添加消融配置 | +3 |

---

## 提交历史

```
d835c60 - Enable ablation mode in metaworld_4d.yaml config
81f53d0 - lujing
2c8b996 - Add ablation study support: disable RGB diffusion
```

---

## 联系方式

如有问题，请联系项目维护者。
