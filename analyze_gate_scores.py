#!/usr/bin/env python3
"""分析路由得分的JSON文件"""

import json
import numpy as np
from collections import defaultdict

# 读取JSON文件
with open('/home/syr/code/prediction_with_action/docs/gate_scores_step_10000.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("路由得分分析报告")
print("=" * 80)

# 打印元数据
print("\n【元数据】")
print(f"  训练步数: {data['metadata']['step']}")
print(f"  MoE块数量: {data['metadata']['num_blocks']}")
print(f"  模态偏置强度: {data['metadata']['modality_bias_strength']}")

# 分析每个block
modality_names = {0: "RGB", 1: "Action", 2: "Depth", 3: "Force"}

for block in data['blocks']:
    block_idx = block['block_idx']
    logits = np.array(block['logits'])  # [num_tokens, num_experts]
    modality_ids = block['modality_ids']

    num_tokens, num_experts = logits.shape

    # 计算概率（softmax）
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 按模态分组统计
    modality_stats = defaultdict(lambda: {
        'count': 0,
        'top1_expert': [],
        'probs': [],
        'entropy': []
    })

    for token_idx, mod_id in enumerate(modality_ids):
        mod_name = modality_names.get(mod_id, f"Mod_{mod_id}")
        modality_stats[mod_name]['count'] += 1

        # Top-1 expert
        top1_expert = np.argmax(probs[token_idx])
        modality_stats[mod_name]['top1_expert'].append(top1_expert)

        # 所有概率
        modality_stats[mod_name]['probs'].append(probs[token_idx])

        # 熵
        token_probs = probs[token_idx]
        entropy = -np.sum(token_probs * np.log(token_probs + 1e-10))
        normalized_entropy = entropy / np.log(num_experts)
        modality_stats[mod_name]['entropy'].append(normalized_entropy)

    print(f"\n{'=' * 80}")
    print(f"Block {block_idx} 分析")
    print(f"{'=' * 80}")
    print(f"  总Token数: {num_tokens}")

    for mod_name in sorted(modality_stats.keys()):
        stats = modality_stats[mod_name]
        count = stats['count']
        if count == 0:
            continue

        print(f"\n  【{mod_name}】 (Token数量: {count})")

        # Top-1专家分布
        top1_experts = np.array(stats['top1_expert'])
        top1_hist = np.bincount(top1_experts, minlength=num_experts)
        top1_pct = top1_hist / count * 100

        print(f"    Top-1专家分布:")
        for exp_idx in range(num_experts):
            bar = "█" * int(top1_pct[exp_idx] / 2)
            print(f"      Expert {exp_idx}: {top1_pct[exp_idx]:5.1f}% {bar}")

        # 概率统计
        all_probs = np.array(stats['probs'])
        mean_probs = np.mean(all_probs, axis=0)

        print(f"    平均专家选择概率:")
        for exp_idx in range(num_experts):
            print(f"      Expert {exp_idx}: {mean_probs[exp_idx]:.4f}")

        # 熵统计
        entropies = np.array(stats['entropy'])
        print(f"    平均路由熵: {np.mean(entropies):.4f} (越低越确定)")

        # Top-1概率（置信度）
        top1_probs = np.max(all_probs, axis=-1)
        print(f"    平均Top-1概率: {np.mean(top1_probs):.4f}")

# 汇总分析所有blocks的action token
print(f"\n{'=' * 80}")
print("【汇总分析：Action Token的路由行为】")
print(f"{'=' * 80}")

for block in data['blocks']:
    block_idx = block['block_idx']
    logits = np.array(block['logits'])
    modality_ids = block['modality_ids']

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 找action tokens (modality_id = 1)
    action_mask = [i for i, m in enumerate(modality_ids) if m == 1]

    if len(action_mask) > 0:
        action_probs = probs[action_mask]
        top1_experts = np.argmax(action_probs, axis=-1)
        top1_hist = np.bincount(top1_experts, minlength=4)
        top1_pct = top1_hist / len(action_mask) * 100

        print(f"\nBlock {block_idx}:")
        print(f"  Action Token数量: {len(action_mask)}")
        print(f"  Expert 0选择率: {top1_pct[0]:.1f}% {'✅' if top1_pct[0] > 70 else '⚠️'}")

        action_entropies = []
        for p in action_probs:
            ent = -np.sum(p * np.log(p + 1e-10)) / np.log(4)
            action_entropies.append(ent)
        print(f"  平均熵: {np.mean(action_entropies):.4f} {'✅' if np.mean(action_entropies) < 0.4 else '⚠️'}")
