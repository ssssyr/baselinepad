# 训练参数详细说明

本文档详细解释训练日志中记录的各项梯度参数的含义和作用。

## 目录

1. [梯度范数（Grad Norm）基础概念](#梯度范数基础概念)
2. [各项参数详解](#各项参数详解)
3. [证明模态偏置路由有效的证据](#证明模态偏置路由有效的证据)
4. [模型架构与参数对应关系](#模型架构与参数对应关系)
5. [训练监控与调试指南](#训练监控与调试指南)

---

## 梯度范数基础概念

### 什么是梯度范数？

**梯度范数（Gradient Norm）**是模型参数梯度向量的L2范数（欧几里得范数），用于衡量梯度的幅度大小。

计算公式：
```
||∇L||₂ = √(Σ(∂L/∂wᵢ)²)
```

其中：
- `L` 是损失函数
- `wᵢ` 是模型参数
- `∂L/∂wᵢ` 是参数对应的梯度

### 为什么监控梯度范数？

1. **检测训练稳定性**：
   - 梯度过大 → 可能导致梯度爆炸，参数更新幅度过大
   - 梯度过小 → 可能导致梯度消失，学习停滞

2. **诊断学习问题**：
   - 不同组件梯度差异过大 → 可能存在学习不平衡
   - 梯度突然变化 → 可能是数据问题或学习率设置不当

3. **优化超参数**：
   - 调整学习率
   - 调整损失权重
   - 配置梯度裁剪阈值

### ⚠️ 重要：梯度范数没有绝对健康的范围

**关键认知**：梯度范数的大小**高度依赖于**以下因素，因此不存在一个适用于所有配置的"健康范围"：

| 影响因素 | 影响 | 你的配置 |
|----------|------|----------|
| **学习率** | 学习率×2 ≈ 梯度范数×2 | `5e-5` (较小) |
| **Batch size** | 更大batch → 梯度更稳定但范数可能更小 | `64` |
| **模型深度** | 更深模型 → 梯度可能更小 | DiT-XL (深) |
| **归一化层** | LayerNorm会稳定梯度，使范数更小 | 有使用 |
| **优化器** | AdamW自适应调整，有效梯度可能不同于原始梯度 | AdamW |
| **Token数量** | 更多token → 梯度累积更多，但会被取平均 | RGB: 256, Action: 3 |

**正确的判断标准**：

| 判断依据 | 健康状态 | 不健康状态 |
|----------|----------|------------|
| **损失在稳定下降** | ✅ 训练正常 | ❌ 损失不变或上升 |
| **梯度范数量级稳定** | ✅ 没有突然100x变化 | ❌ 突然变成0或NaN |
| **ratio_head稳定** | ✅ 多任务平衡 | ❌ 比值突变 |

**你观察到的情况（rgb=0.001, action=0.005）是完全正常的！**
- 学习率5e-5较小 → 梯度范数自然就小
- 损失在下降 → 说明训练正常
- ratio ≈ 5 → 说明action任务学习难度更高，需要更多梯度

---

## 各项参数详解

### 1. `grad/grad_norm/shared_experts`

**完整路径**: `grad/grad_norm/shared_experts`

**对应代码**: `moe_blocks.py:223-231`, `train_robot.py:721-722`

#### 功能描述

`shared_experts`（共享专家）是MoE（Mixture of Experts）架构中的一个关键组件，提供所有模态共享的前馈神经网络路径。

#### 架构细节

```python
# moe_blocks.py:223-231
self.shared_experts = DenseGeluMLP(
    embed_dim,
    intermediate_size=embed_dim * n_shared_experts,
    bias=config.use_bias,
    drop=config.use_dropout
)
```

#### 工作原理

1. **输入处理**：所有模态的token都会经过共享专家层
2. **固定路径**：与路由专家不同，共享专家总是被激活
3. **宽度计算**：`embed_dim * n_shared_experts`（通常为4）

#### 设计目的

| 目的 | 说明 |
|------|------|
| 知识共享 | 让所有模态共享一部分基础知识表示 |
| 迁移兼容 | 从dense模型迁移时保持性能 |
| 稳定性 | 提供稳定的学习路径，不依赖路由决策 |

---

### 2. `grad/grad_norm/router`

**完整路径**: `grad/grad_norm/router`

**对应代码**: `moe_blocks.py:14-125`, `train_robot.py:707-708`

#### 功能描述

`router`（路由器）是MoE架构中的门控网络（Gating Network），负责为每个token选择最合适的专家。

#### 架构细节

```python
# moe_blocks.py 核心逻辑
class MoEGate(nn.Module):
    def forward(self, hidden_states):
        # 1. 计算每个专家的得分
        router_logits = self.gate(hidden_states)  # [batch, seq_len, n_experts]

        # 2. 应用模态偏置（可选）
        if self.use_modality_bias:
            router_logits = router_logits + self.modality_bias

        # 3. Top-K专家选择
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k)
```

#### 工作原理

```
输入Token
    ↓
Router (线性层 → Softmax)
    ↓
专家得分向量 [e₁, e₂, e₃, e₄]
    ↓
Top-K选择 (通常K=1或2)
    ↓
选中的专家处理输入
```

#### 模态偏置（Modality Bias）

当启用时，路由器会为不同模态添加偏置：

```python
# configs/metaworld_4d.yaml
use_modality_bias: true
modality_bias_strength_action: 0.5  # action tokens偏向expert 0
```

效果：
- **RGB tokens**: 自由选择专家（无偏置）
- **Action tokens**: 倾向选择expert 0（偏置+0.5）

#### 设计目的

| 目的 | 说明 |
|------|------|
| 稀疏激活 | 每个token只使用部分专家，提高效率 |
| 专家专业化 | 让不同专家学习不同的模态或任务 |
| 模态分离 | 通过偏置让不同模态使用不同专家 |

---

### 3. `grad/grad_norm/rgb_head`

**完整路径**: `grad/grad_norm/rgb_head`

**对应代码**: `models.py:315-380`, `train_robot.py:702-704`

#### 功能描述

`rgb_head` 是RGB图像预测的输出头，负责生成预测的RGB图像均值和方差。

#### 输出格式

- **均值**: 预测的去噪后RGB图像
- **方差**: 预测的不确定性（用于训练时计算损失）
- **形状**: `[batch, height, width, 3]`

---

### 4. `grad/grad_norm/action_head`

**完整路径**: `grad/grad_norm/action_head`

**对应代码**: `models.py:331-345`, `train_robot.py:700-701`

#### 功能描述

`action_head` 是动作序列预测的输出头，负责生成预测的动作（如关节角度、末端执行器位置等）。

#### 输出格式

- **均值**: 预测的动作序列 `[batch, horizon, action_dim]`
- **方差**: 预测的不确定性
- **horizon**: 预测的未来时间步数（如3步）

#### 典型动作维度

| 机器人类型 | action_dim | 说明 |
|-----------|------------|------|
| MetaWorld 4-DOF | 4 | [x, y, z, gripper] |
| 机械臂 | 7~14 | 关节角度 + 末端执行器 |
| 移动机器人 | 3~6 | 线速度 + 角速度 |

---

### 5. `grad/grad_norm/ratio_head`

**完整路径**: `grad/grad_norm/ratio_head`

**对应代码**: `train_robot.py:724-727`

#### 功能描述

`ratio_head` 是计算得到的指标，表示动作头梯度与RGB头梯度的比率。

#### 计算公式

```python
ratio_head = grad_norm_action_head / grad_norm_rgb_head
```

#### 你的情况分析

```
ratio_head = 0.005 / 0.001 = 5
```

**这说明**：
- Action任务的学习难度更高，需要更大的梯度
- 这与`action_loss_lambda: 2.0`的设置是一致的
- 两个任务都在学习（损失都在下降）

---

## 证明模态偏置路由有效的证据

### 关键问题：输出头梯度范数和模态偏置有什么关系？

**答案**：输出头梯度范数（rgb_head, action_head）和模态偏置路由是**不同层面的东西**，但可以通过**其他指标**证明模态偏置有效。

### 证据层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① 输入 → Token嵌入                                          │
│      RGB: 256 tokens, Action: 3 tokens                       │
│                                                             │
│  ② MoE Router (模态偏置在这里!) ★★★                          │
│      └─ 决定: RGB token去哪个专家?                           │
│             Action token去哪个专家?                          │
│                                                             │
│  ③ 专家处理 (不同专家学习不同模态的特征)                       │
│      └─ grad_norm/adaln_norm1_rgb  ← 专家AdaLN梯度          │
│         grad_norm/adaln_norm1_action ← 专家AdaLN梯度         │
│                                                             │
│  ④ 输出头预测                                                 │
│      └─ grad_norm/rgb_head    ← 输出头梯度                 │
│         grad_norm/action_head ← 输出头梯度                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 直接证据：路由统计信息

你的代码已经收集了证明模态偏置有效的**直接证据**！查看这些WandB日志：

#### 1. `routing/action/top1_hist` - Action token的专家分配直方图

**有模态偏置时的预期表现**：
```
Expert 0: ████████████████████ 80%  ← action tokens主要选expert 0
Expert 1: ██ 10%
Expert 2: ██ 7%
Expert 3: █ 3%
```

**无模态偏置时的预期表现**：
```
Expert 0: ████████ 25%  ← 均匀分布
Expert 1: ████████ 25%
Expert 2: ████████ 25%
Expert 3: ████████ 25%
```

#### 2. `routing/rgb/top1_hist` - RGB token的专家分配直方图

**预期表现**（无偏置，更均匀）：
```
Expert 0: ██████████ 30%
Expert 1: ████████████ 40%
Expert 2: ████████ 20%
Expert 3: ████ 10%
```

#### 3. `routing/action/entropy` - Action路由熵

| 熵值 | 含义 | 证明 |
|------|------|------|
| 0.2 ~ 0.4 | 低熵，选择很确定 | ✅ 模态偏置起作用 |
| 0.8 ~ 1.0 | 高熵，选择随机 | ❌ 模态偏置无效 |

**计算公式**：
```
entropy = -Σ(pᵢ * log(pᵢ)) / log(num_experts)
```

#### 4. `moe/action_hit_rate` - Action tokens选择expert 0的比例

| hit_rate | 含义 | 证明 |
|----------|------|------|
| > 0.7 | 大部分action tokens选了expert 0 | ✅ 模态偏置起作用 |
| ~0.25 | 均匀分布（4个专家） | ❌ 模态偏置无效 |

#### 5. 对比实验设计（用于论文）

| 实验组 | use_modality_bias | 预期结果 |
|--------|-------------------|----------|
| Baseline | false | action/entropy ≈ 0.9, hit_rate ≈ 0.25 |
| Ours | true | action/entropy ≈ 0.3, hit_rate > 0.7 |

### 间接证据：梯度差异

虽然输出头梯度范数不能直接证明模态偏置，但以下梯度指标可以作为**间接证据**：

#### 1. `grad_norm/adaln_norm1_rgb` vs `grad_norm/adaln_norm1_action`

如果模态偏置有效，不同专家的AdaLN梯度应该有差异：
- **专门处理RGB的专家** → rgb模态的AdaLN梯度更大
- **专门处理Action的专家** → action模态的AdaLN梯度更大

#### 2. `grad_norm/router`

如果模态偏置有效，router的梯度应该：
- 前期：学习模态偏置模式
- 后期：梯度变小（偏置已经稳定）

### 论文中如何展示

建议在论文中展示以下内容证明模态偏置有效：

```
Figure X: Modality-aware Routing Effectiveness
┌─────────────────────────────────────────────────────────┐
│ (a) Expert Assignment Heatmap                           │
│       Expert 0  Expert 1  Expert 2  Expert 3           │
│ RGB     ████      ██████      ████      ██             │
│ Action  ████████  ██          ██        █              │
│                                                         │
│ (b) Routing Entropy (lower is more deterministic)      │
│       │                                                  │
│ 0.9  ┤ ──── Baseline (no bias)                         │
│ 0.7  ┤                                                  │
│ 0.5  ┤                                                  │
│ 0.3  ┤ ──── Ours (with modality bias)                   │
│ 0.1  ┤                                                  │
│       └──────────────────────────────────→ Training     │
│                                                         │
│ (c) Action Token Hit Rate to Expert 0                  │
│       │                                                  │
│ 1.0  ┤ ──── Ours: 85%                                   │
│ 0.8  ┤                                                  │
│ 0.6  ┤                                                  │
│ 0.4  ┤                                                  │
│ 0.2  ┤                                                  │
│ 0.25 ┤ ──── Baseline: 25% (random)                      │
│       └──────────────────────────────────→ Training     │
└─────────────────────────────────────────────────────────┘
```

### 总结：如何用你的日志证明模态偏置有效

| WandB路径 | 什么指标 | 预期值（有偏置） | 说明 |
|-----------|----------|------------------|------|
| `routing/action/top1_hist/e0` | action选expert 0的比例 | > 70% | 主要证据 |
| `routing/action/entropy` | action路由熵 | < 0.4 | 越低越确定 |
| `moe/action_hit_rate` | action hit rate | > 0.7 | 越高越好 |
| `routing/rgb/top1_hist` | rgb专家分配 | 更均匀 | 对比组 |

**对比实验**：
- 训练一个`use_modality_bias: false`的模型作为baseline
- 对比上述指标，差异越大说明模态偏置越有效

---

## 模型架构与参数对应关系

### 整体架构图

```
                                    输入数据
                              ┌─────────────┐
                              │  RGB图像序列 │ 256 tokens
                              │  动作序列    │ 3 tokens
                              │  力传感器等  │ 1 token (optional)
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  Patch化    │
                              │  Token嵌入  │
                              └──────┬──────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────┐
│                      DiT Backbone (14层)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 层1-14: MoE Transformer Blocks                       │  │
│  │                                                        │  │
│  │  ┌────────────┐    ┌─────────────────────────┐      │  │
│  │  │  Self-Attn │    │      MoE Block          │      │  │
│  │  └────────────┘    │  ┌─────────────────┐    │      │  │
│  │                    │  │ Router (模态偏置)│◄───┼──────┼──│── grad_norm/router
│  │                    │  │ + modality_bias │    │      │  │
│  │                    │  └────────┬────────┘    │      │  │
│  │                    │           │             │      │  │
│  │                    │      Top-K选择          │      │  │
│  │                    │  ┌────────┴────────┐    │      │  │
│  │                    │  │ Routed Experts  │    │      │  │
│  │                    │  │ (4个专家)        │    │      │  │
│  │                    │  │ AdaLN per mod   │    │      │  │
│  │                    │  └─────────────────┘    │      │  │
│  │                    │                         │      │  │
│  │                    │  ┌─────────────────┐    │      │  │
│  │                    │  │ Shared Experts  │◄───┼──────┼──│── grad_norm/shared_experts
│  │                    │  │ (Dense MLP)     │    │      │  │
│  │                    │  └─────────────────┘    │      │  │
│  │                    └─────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  Final Layer│
                              └──────┬──────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
         ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
         │  RGB Head   │      │Action Head  │      │  Depth Head │
         │(grad_norm)  │      │(grad_norm)  │      │(grad_norm)  │
         │ ~0.001      │      │ ~0.005      │      │  (optional) │
         └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
                │                    │                    │
                ▼                    ▼                    ▼
         ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
         │ RGB预测输出  │      │动作预测输出  │      │深度预测输出  │
         │(均值+方差)   │      │(均值+方差)   │      │(均值+方差)   │
         └─────────────┘      └─────────────┘      └─────────────┘

                                   │
                                   ▼
                            ratio_head = 5
                        (action_grad / rgb_grad)
```

### MoE组件详细结构（含模态偏置）

```
                    MoE Block (modality_ids作为输入)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
   ┌─────────┐                    ┌─────────┐
   │ Router  │                    │Shared   │
   │ + Mod.  │                    │Experts  │
   │ Bias    │                    │(Dense)  │
   └────┬────┘                    └────┬────┘
        │                              │
        │ modality_bias[action] += 0.5  │
        │                              │
        │ Top-K (K=1)                   │
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │Expert 0 │◄────┐                   │
   │(Action  │     │                   │
   │ prefer) │     │                   │
   └─────────┘     │                   │
   ┌─────────┐     │                   │
   │Expert 1 │◄────┤                   │
   └─────────┘     │Combine            │
   ┌─────────┐     │(加权求和)          │
   │Expert 2 │◄────┤                   │
   └─────────┘     │                   │
   ┌─────────┐     │                   │
   │Expert 3 │◄────┘                   │
   └─────────┘                         │
        │                              │
        └──────────────┬───────────────┘
                       ▼
                    输出融合
           (routed_output + shared_output)
```

### 数据流：从模态偏置到梯度

```
输入: RGB tokens (256个), Action tokens (3个)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Router + Modality Bias                               │
│                                                       │
│  RGB tokens:    logits + [0, 0, 0, 0]                │
│  Action tokens: logits + [0.5, 0, 0, 0]  ← 偏向expert 0│
│                                                       │
│  结果:                                                │
│    - RGB tokens → 专家分布较均匀                       │
│    - Action tokens → 主要选expert 0 (80%)             │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Expert Processing (AdaLN per modality)               │
│                                                       │
│  Expert 0: 主要处理action tokens                       │
│    → adaLN_norm1_action梯度大                         │
│                                                       │
│  Expert 1-3: 主要处理rgb tokens                        │
│    → adaLN_norm1_rgb梯度大                            │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Output Heads                                         │
│                                                       │
│  rgb_head: 256个token的聚合 → grad_norm ≈ 0.001      │
│  action_head: 3个token的聚合 → grad_norm ≈ 0.005     │
│              (学习难度更高)                             │
└───────────────────────────────────────────────────────┘
```

---

## 训练监控与调试指南

### 日志输出格式

训练过程中，梯度范数会以以下格式输出到WandB：

```
train/loss: 2.345
train/loss_action: 0.123
grad/grad_norm/rgb_head: 0.001
grad/grad_norm/action_head: 0.005
grad/grad_norm/ratio_head: 5.0
grad/grad_norm/shared_experts: 0.002
grad/grad_norm/router: 0.0003
routing/action/top1_hist/e0: 0.85  ← 85% action tokens选expert 0
routing/action/entropy: 0.35       ← 低熵，选择确定
moe/action_hit_rate: 0.85          ← 模态偏置起作用
```

### 判断训练是否健康的标准

| 检查项 | 健康状态 | 不健康状态 |
|--------|----------|------------|
| **损失趋势** | 稳定下降 | 不变、波动或上升 |
| **梯度范数量级** | 稳定在某个范围 | 突然变为0或NaN |
| **梯度范数变化** | 平滑变化 | 突然100x跳跃 |
| **ratio_head** | 稳定在某个范围 | 剧烈波动 |

### 常见问题与解决方案

#### 问题1: 损失不下降但梯度范数正常

**症状**：梯度范数正常（如0.001~0.005），但损失不变

**可能原因**：
- 学习率过小
- 数据问题
- 模型架构问题

**解决方案**：
```yaml
training:
  learning_rate: 1e-4  # 增加学习率
```

#### 问题2: 模态偏置似乎不起作用

**症状**：`routing/action/top1_hist/e0` < 0.3，action tokens没有主要选expert 0

**可能原因**：
- `modality_bias_strength`太小
- 模态偏置被学习到的特征覆盖

**解决方案**：
```yaml
moe:
  modality_bias_strength_action: 1.0  # 增大偏置强度
```

#### 问题3: 专家负载不均衡

**症状**：某些专家几乎不被使用

**解决方案**：
```yaml
moe:
  aux_loss_weight: 0.01  # 增加负载均衡损失
```

### 可视化建议

使用WandB创建以下面板：

**面板1: 路由行为（证明模态偏置）**
- `routing/action/top1_hist` (stacked bar chart)
- `routing/rgb/top1_hist` (stacked bar chart)
- `routing/action/entropy` (line chart)
- `moe/action_hit_rate` (line chart)

**面板2: 梯度监控**
- `grad/grad_norm/rgb_head` (line chart)
- `grad/grad_norm/action_head` (line chart)
- `grad/grad_norm/ratio_head` (line chart)
- `grad/grad_norm/router` (line chart)

**面板3: 训练健康度**
- `train/loss` (line chart)
- `train/learning_rate` (line chart)
- `train/moe_aux_loss` (line chart)

---

## 总结

### 参数含义总结

| 参数 | 组件 | 监控目的 |
|------|------|----------|
| `shared_experts` | 共享专家MLP | 确保共享知识正常学习 |
| `router` | MoE路由器+模态偏置 | 监控专家选择机制 |
| `rgb_head` | RGB预测头 | 监控图像预测学习 |
| `action_head` | 动作预测头 | 监控动作序列学习 |
| `ratio_head` | 计算指标 | 平衡多任务学习 |

### 证明模态偏置有效的关键指标

| WandB路径 | 预期值 | 说明 |
|-----------|--------|------|
| `routing/action/top1_hist/e0` | > 0.7 | 主要证据：action主要选expert 0 |
| `routing/action/entropy` | < 0.4 | 低熵 = 选择确定 |
| `moe/action_hit_rate` | > 0.7 | hit rate越高越好 |
| `routing/rgb/top1_hist` | 更均匀 | 对比组：rgb分布更平均 |

### 常见误区

| 误区 | 纠正 |
|------|------|
| "梯度范数必须在0.5~5.0才健康" | ❌ 取决于学习率、batch size等 |
| "输出头梯度能直接证明模态偏置" | ❌ 需要看路由统计信息 |
| "ratio_head必须接近1.0" | ❌ 取决于任务难度，你的ratio=5是正常的 |
| "梯度范数小就是学习停滞" | ❌ 损失下降才是关键 |

---

**文档版本**: 2.0
**最后更新**: 2026-01-09
**对应代码分支**: expert-adaln
