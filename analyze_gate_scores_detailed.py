#!/usr/bin/env python3
"""详细分析路由得分并生成可视化数据"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 读取JSON文件
with open('/home/syr/code/prediction_with_action/docs/gate_scores_step_10000.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("路由得分详细分析 - Step 10000")
print("=" * 80)

# 汇总统计
all_block_stats = []

modality_names = {0: "RGB", 1: "Action", 2: "Depth", 3: "Force"}

for block in data['blocks']:
    block_idx = block['block_idx']
    logits = np.array(block['logits'])
    modality_ids = block['modality_ids']

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 分析Action tokens
    action_mask = [i for i, m in enumerate(modality_ids) if m == 1]
    rgb_mask = [i for i, m in enumerate(modality_ids) if m == 0]

    stats = {
        'block_idx': block_idx,
        'action_count': len(action_mask),
        'rgb_count': len(rgb_mask),
    }

    # Action tokens统计
    if len(action_mask) > 0:
        action_probs = probs[action_mask]
        action_top1 = np.argmax(action_probs, axis=-1)
        action_hist = np.bincount(action_top1, minlength=4)
        stats['action_expert0_pct'] = action_hist[0] / len(action_mask) * 100
        stats['action_top1_hist'] = action_hist

        # 计算熵
        entropies = []
        for p in action_probs:
            ent = -np.sum(p * np.log(p + 1e-10)) / np.log(4)
            entropies.append(ent)
        stats['action_entropy'] = np.mean(entropies)

        # Top-1概率
        stats['action_top1_prob'] = np.mean(np.max(action_probs, axis=-1))

    # RGB tokens统计
    if len(rgb_mask) > 0:
        rgb_probs = probs[rgb_mask]
        rgb_top1 = np.argmax(rgb_probs, axis=-1)
        rgb_hist = np.bincount(rgb_top1, minlength=4)
        stats['rgb_expert0_pct'] = rgb_hist[0] / len(rgb_mask) * 100
        stats['rgb_top1_hist'] = rgb_hist

        entropies = []
        for p in rgb_probs:
            ent = -np.sum(p * np.log(p + 1e-10)) / np.log(4)
            entropies.append(ent)
        stats['rgb_entropy'] = np.mean(entropies)

        stats['rgb_top1_prob'] = np.mean(np.max(rgb_probs, axis=-1))

    all_block_stats.append(stats)

print("\n【关键指标汇总】\n")

# 计算平均统计
avg_action_expert0 = np.mean([s['action_expert0_pct'] for s in all_block_stats])
avg_action_entropy = np.mean([s['action_entropy'] for s in all_block_stats])
avg_rgb_expert0 = np.mean([s['rgb_expert0_pct'] for s in all_block_stats])
avg_rgb_entropy = np.mean([s['rgb_entropy'] for s in all_block_stats])

print(f"Action Token平均选Expert 0比例: {avg_action_expert0:.1f}%")
print(f"Action Token平均熵: {avg_action_entropy:.4f}")
print(f"RGB Token平均选Expert 0比例: {avg_rgb_expert0:.1f}%")
print(f"RGB Token平均熵: {avg_rgb_entropy:.4f}")

print("\n【逐层详情】\n")
print(f"{'Block':<6} {'Action%':<10} {'RGB%':<10} {'Action熵':<10} {'RGB熵':<10} {'状态'}")
print("-" * 80)

for stats in all_block_stats:
    action_pct = stats['action_expert0_pct']
    rgb_pct = stats['rgb_expert0_pct']
    action_ent = stats['action_entropy']
    rgb_ent = stats['rgb_entropy']

    # 判断状态
    if action_pct > 70 and action_ent < 0.5:
        status = "✅ 优秀"
    elif action_pct > 60:
        status = "⚠️ 中等"
    else:
        status = "❌ 较弱"

    print(f"{stats['block_idx']:<6} {action_pct:<10.1f} {rgb_pct:<10.1f} {action_ent:<10.4f} {rgb_ent:<10.4f} {status}")

print("\n【分析结论】\n")

# 分析结论
conclusions = []

# 1. 模态偏置效果
strong_blocks = sum(1 for s in all_block_stats if s['action_expert0_pct'] > 70)
weak_blocks = sum(1 for s in all_block_stats if s['action_expert0_pct'] < 50)

conclusions.append(f"1. 模态偏置效果:")
conclusions.append(f"   - {strong_blocks}/14 个block的Action Token选Expert 0比例 > 70% ✅")
conclusions.append(f"   - 平均{avg_action_expert0:.1f}%的Action Token选Expert 0（预期100%）")
conclusions.append(f"   - RGB Token均匀分布（{avg_rgb_expert0:.1f}%选Expert 0，接近25%）✅")

# 2. 熵分析
if avg_action_entropy < 0.4:
    conclusions.append(f"2. 路由确定性: 高（平均熵{avg_action_entropy:.4f} < 0.4）✅")
elif avg_action_entropy < 0.7:
    conclusions.append(f"2. 路由确定性: 中等（平均熵{avg_action_entropy:.4f}）⚠️")
else:
    conclusions.append(f"2. 路由确定性: 低（平均熵{avg_action_entropy:.4f}，接近随机）❌")

# 3. 层间差异
action_pcts = [s['action_expert0_pct'] for s in all_block_stats]
pct_std = np.std(action_pcts)
if pct_std > 20:
    conclusions.append(f"3. 层间稳定性: 差（标准差{pct_std:.1f}%）❌")
elif pct_std > 10:
    conclusions.append(f"3. 层间稳定性: 中等（标准差{pct_std:.1f}%）⚠️")
else:
    conclusions.append(f"3. 层间稳定性: 好（标准差{pct_std:.1f}%）✅")

# 4. 建议
conclusions.append(f"\n【改进建议】\n")

if avg_action_entropy > 0.5:
    conclusions.append(f"- 当前熵较高（{avg_action_entropy:.4f}），建议:")
    conclusions.append(f"  1. 增加模态偏置强度: 0.5 → 1.0")
    conclusions.append(f"  2. 继续训练更多步数")

if pct_std > 15:
    conclusions.append(f"- 层间差异大（标准差{pct_std:.1f}%），建议:")
    conclusions.append(f"  1. 检查浅层和深层的学习率")
    conclusions.append(f"  2. 可能需要per-layer的modality bias")

strong_blocks_count = sum(1 for s in all_block_stats if s['action_expert0_pct'] > 70)
if strong_blocks_count < 10:
    conclusions.append(f"- 只有{strong_blocks_count}/14个block效果明显，建议:")
    conclusions.append(f"  1. 增加modality_bias_strength_action")
    conclusions.append(f"  2. 或使用更强的专家分配约束")

for line in conclusions:
    print(line)

# 生成表格数据（用于论文）
print(f"\n【论文用表格数据】\n")
print("Block | Action% Expert0 | RGB% Expert0 | Action Entropy | Status")
print("-" * 80)
for stats in all_block_stats:
    action_pct = stats['action_expert0_pct']
    rgb_pct = stats['rgb_expert0_pct']
    action_ent = stats['action_entropy']
    if action_pct > 70:
        status = "✓"
    elif action_pct > 50:
        status = "~"
    else:
        status = "✗"
    print(f"{stats['block_idx']:>5} | {action_pct:>14.1f}% | {rgb_pct:>12.1f}% | {action_ent:>14.4f} | {status}")
