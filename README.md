# PAD-MoE 扩展迁移路线图（实践版）

## 🍱 与现有 `PAD-MoE_Integration_Plan.md` 的对比

| 评价点 | 现有文档表现 | 我的建议 |
| --- | --- | --- |
| **阶段拆分** | 阶段 1~4 目标分明，循序渐进 ✅ | 保留整体结构，并在每阶段开始前设置“进入条件”，避免未完成就堆叠新复杂度。 |
| **监控指标** | 提供了丰富的 WandB 指标与诊断脚本 ✅ | 保留这些监控想法，但先挑选与当前阶段直接相关的指标，防止监控面板噪声过大。 |
| **技术深度** | 模态专家/对数域融合/自门控覆盖很全 ✅ | 逐步引入，避免“尚未验证基础MoE就进入高级路由”——本文按照“先复刻 DiT-MoE，再按需增强”的顺序排布。 |
| **执行粒度** | 部分步骤偏概念化（如多模态专家池实现仍待拆解）⚠️ | 为每一步补充“操作+目的+完成标准”，确保落地时可直接勾选。 |
| **风险控制** | 有通用风险表，但缺少阶段退出机制 ⚠️ | 每阶段附带“进入下一阶段前必须满足的验收标准”和“回退策略”。 |

结论：原文档的宏观思路值得延续（阶段划分、监控意识、自门控方案），本指南做的改动主要是**压实操作步骤、加入阶段网关与回退策略**，以便你可以按 checklist 推进。

---

## 🗺️ 总体路线图

| 阶段 | 目标 | 核心交付 | 进入条件 | 退出标准 |
| --- | --- | --- | --- | --- |
| 0. 基线复盘 | 梳理代码分支/配置/数据 | `moe` 工作分支、基准指标表 | 无 | 形成 baseline 报告 |
| 1. 稀疏 MoE 引入 | 复刻 DiT-MoE 的 SparseMoEBlock | `models/moe_blocks.py`、DiTBlock 切换 | 完成阶段 0 | MoE 版本在验证集上不低于基准 -5% |
| 2. 模态感知路由 | 为不同 token 模态添加偏置/统计 | 模态 mask + 路由日志 | 阶段 1 稳定训练 2 次 | 覆盖率按模态单独设阈值（RGB 高、动作低） |
| 3. 自门控与对数域融合 | 实现 token 自门控、logit 融合 | 自门控模块 / beta 调度 | 阶段 2 指标达标 | 自门控接受率 70%±15%，loss 稳定 |
| 4. 训练流程固化 | 配置、脚本、监控、风险预案 | YAML + 脚本 + 监控模版 | 阶段 3 通过 | 可一键切换 dense/MoE 训练 |

---

## 🧩 阶段 0：基线与准备

| 操作 | 目的 |
| --- | --- |
| 建立 `moe` 或 `feature/moe` 分支，记录当前 commit id | 保证可随时回退到无 MoE 的稳定版本 |
| 运行一次现有 PAD 训练/评估（小数据即可），记录训练 loss / eval 成绩 / 内存占用 | 形成对比基准，后续每阶段都用同一张表更新 |
| 整理配置差异表：列出 `args` 中与 MoE 相关的潜在开关（action_steps、use_depth 等） | 后续在配置里加入 `use_moe`、`moe_num_experts` 等参数时可对齐命名 |
| 复制 `PAD-MoE_Integration_Plan.md` 里已有的监控想法，形成最小化监控清单（例如“专家使用率/aux loss”） | 避免阶段 1 就把仪表盘堆满，提高可读性 |

**进入下一阶段条件：** 有 baseline 报告；明确回退点；决定首个实验配置 (如 `DiT_L_4`, batch=32)。

---

## 🚀 阶段 1：引入稀疏 MoE（复刻 DiT-MoE）

| 操作 | 目的 |
| --- | --- |
| 新建 `models/moe_blocks.py`，直接移植 `DiT-MoE/models.py:205-299` 的 `MoEGate + MoeMLP + SparseMoeBlock + AddAuxiliaryLoss` | 以经验证的实现开局，减少自研 Bug 面 |
| 在 `models.py` 中添加 `class DiTBlockMoE(DiTBlock)`，将 `self.mlp` 替换为 `SparseMoeBlock`，并保留 adaLN/gate 逻辑 | 保持接口兼容，便于在配置里选择 dense 或 MoE block |
| 调整 `DiT` 构造函数：根据 `args.use_moe` 选择 block 类型，增加 `args.moe_num_experts/args.moe_topk` 等参数 | 让 MoE 可配置可回退 |
| 在训练/推理脚本中增加 aux loss 汇总（`aux_loss_weight * aux_loss`），并在 log 中记录 `aux_loss` | 控制专家负载、监控训练稳定性 |
| 编写 `tests/test_moe_block.py`（或最少一个脚本）验证前向/反向、top-k 行为 | 及早发现形状/设备问题 |

**验证指标：**
- 验证集性能 ≥ baseline -5%
- aux loss 收敛且非零
- 专家使用熵 > 1.0（避免全部落到单专家）

**回退策略：** `args.use_moe=False`；删除 `moe_blocks.py` 引用即可恢复。

---

## 🎯 阶段 2：模态感知路由

| 操作 | 目的 |
| --- | --- |
| 基于 `args.start_idx/end_idx` 生成 `modality_ids`（RGB=0, Action=1, Depth=2）；在前向中缓存 | 让每个 token 知道自己属于哪一模态 |
| 在 `MoEGate` 中添加 `self.modality_bias = nn.Parameter(num_modalities, num_experts)`；路由前 `logits += modality_bias[mod_id]` | 轻量地引导专家专注不同模态 |
| 为每个 batch 统计各模态→专家的分配矩阵，记录 `modality_coverage = (#使用专家)/(num_experts)` | 监控模态专业化程度 |
| 若 action/depth token 数远少于 RGB，可在 gate 前对这些 token 的 logits 加权（参考 DAE-MoE 的 priority bias 思路） | 防止模态数据量不均导致路由长期忽视某类专家 |
| WandB 新增图表：`moe/modal_rgb_entropy`、`moe/modal_action_entropy` 等 | 观察模态级别的多样性趋势 |

**验证指标：**
- RGB 覆盖率 >0.4；动作至少命中 1 个专家（约等于 1/num_experts≈0.125，top-k=2 时可看作 ≥0.25）；Depth 按实际 token 数设中间值（例如 0.2）
- 模态偏置加入后，Loss 无明显震荡
- 如果出现某模态完全不用任何专家，及时调高偏置或增加该模态 token 数

**回退策略：** 通过配置关闭 `use_modality_bias`；保留阶段 1 的 MoE 结构继续训练。

---

## 🧠 阶段 3：自门控 + 对数域融合

| 操作 | 目的 |
| --- | --- |
| 在每个专家 MLP 上添加 `SelfGate`（参考 `DAE-Moe/src/models/moe_modules.py:121-185`），输出 `suitability` ∈ (0,1) | 让专家拥有“拒绝”不擅长 token 的权利 |
| 在 `MoEGate` 里新增 `beta` 超参及 warmup 逻辑；融合方式 `gate_logits = router_logits + beta * logit(suitability)` | 引入你验证过的对数域加法，避免概率乘法的梯度退化 |
| 记录 `self_gate_accept_rate`, `expert_rejection_rate`, `beta_schedule`，并监控是否出现全体拒绝或全体接受 | 衡量自门控是否在健康区间 |
| 若路由后出现 token 无专家接收，给出 fallback（直接复用 `SparseMoeBlock` 中 shared_expert 或 dense MLP） | 确保模型对所有 token 都有输出 |
| 训练脚本中加入 `beta` warmup（例如 0→0.5，5k steps），防止开局剧烈震荡 | 平滑引入自门控影响 |

**验证指标：**
- 自门控接受率保持在 60%~85%
- `beta` 达到上限时，loss 曲线仍稳定
- 路由熵较阶段 2 略有下降，但无专家坍塌

**回退策略：** 将 `beta=0`，禁用自门控，恢复阶段 2 行为。

---

## 🛠️ 阶段 4：训练流程与监控固化

| 操作 | 目的 |
| --- | --- |
| 在 `configs/*.yaml` 中补充 `moe` 配置段（示例：`use_moe`, `num_experts`, `top_k`, `modality_bias`, `beta_schedule`） | 让实验切换可溯源、可复现 |
| 训练脚本加入 CLI 参数和 config merge 逻辑，支持 `--use-moe --moe-config configs/moe/base.yaml` | 简化实验指令 |
| 建立 WandB 模版面板（训练 loss、aux loss、路由熵、模态覆盖、自门控接受率、GPU 内存） | 复用监控，减少每次人工配置 |
| 编写 `docs/MoE_troubleshooting.md`：记录常见异常（专家坍塌、OOM、梯度爆炸）及快速检查步骤 | 缩短排障时间 |
| 每阶段结束打 tag，如 `moe-stage1-ok`；在 README/计划中更新实验状态 | 保持阶段性可追踪 |

**完成标准：**
- 任意配置可通过 flag 切换 Dense/MoE
- 监控仪表盘可一键复用
- 有最新的 troubleshooting 文档与基准表

---

## 🧯 风险与回退一览

| 风险 | 触发信号 | 快速处理 |
| --- | --- | --- |
| 专家坍塌 | 某专家使用率 >80% | 提高 aux loss、调大 top-k、或减小 lr |
| 内存飙升 | 显存 > 目标 +2GB | 降低 experts_per_tok、启用 gradient checkpoint |
| 自门控拒绝全部 token | `accept_rate < 0.2` | 下调 `beta` 或放宽自门控初始化 bias |
| 模态严重失衡 | 模态覆盖 < 阈值 | 调整 priority bias 或增加该模态 tokens |

---

## ✅ 使用建议

1. **严格按照阶段顺序推进**：每阶段都有进入条件与回退策略，确保问题可定位。
2. **记录实验表**：将“配置→指标→备注”集中在一个表格，便于比较 dense/MoE 的收益。
3. **监控先行**：阶段 1 即启用最小监控集，后续只在必要时添加新图表。
4. **善用对比实验**：同一 batch/seed 下比较 `use_moe=False` 与 `True`，观察差异。
5. **保持沟通日志**：每完成一个阶段，把结论写入 README 或 docs，方便他人接手。

---

**祝实施顺利！** 这个指南可作为 checklist：完成一个操作就勾选并记录结果。如果需要进一步细化代码层面的 TODO，可以在每个阶段下再拆分 issue/任务。搬运 DiT-MoE 的成熟模块、借鉴 DAE-MoE 的自门控设计、再结合 PAD 的多模态特性，就能稳步完成 PAD→PAD-MoE 的演进。***
