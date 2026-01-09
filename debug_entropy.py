#!/usr/bin/env python3
"""展示为什么Expert 0选择率71%但熵还是0.98"""

import json
import numpy as np

# 读取JSON
with open('/home/syr/code/prediction_with_action/docs/gate_scores_step_10000.json', 'r') as f:
    data = json.load(f)

# 分析Block 2（Expert 0选择率93.8%，但熵0.99）
block = data['blocks'][2]
logits = np.array(block['logits'])
modality_ids = block['modality_ids']

# Softmax
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# 找Action tokens
action_indices = [i for i, m in enumerate(modality_ids) if m == 1]

print("=" * 80)
print("Block 2 的 Action Tokens 详细分析")
print("=" * 80)
print(f"Action Token数量: {len(action_indices)}")
print(f"Expert 0 Top-1选择率: 93.8%")
print(f"平均熵: 0.9913")
print("\n" + "=" * 80)
print("每个Action Token的详细概率分布")
print("=" * 80)

top1_count = 0
total_entropy = 0

for idx in action_indices:
    token_probs = probs[idx]
    top1_expert = np.argmax(token_probs)
    top1_prob = token_probs[top1_expert]

    # 计算熵
    entropy = -np.sum(token_probs * np.log(token_probs + 1e-10)) / np.log(4)

    if top1_expert == 0:
        top1_count += 1
    total_entropy += entropy

    prob_str = ", ".join([f"E{i}:{p:.3f}" for i, p in enumerate(token_probs)])
    print(f"Token {idx:3d}: [{prob_str}] → Top1: E{top1_expert} ({top1_prob:.1%}), 熵: {entropy:.4f}")

print("\n" + "=" * 80)
print("关键发现")
print("=" * 80)
print(f"Top-1选Expert 0的比例: {top1_count}/{len(action_indices)} = {top1_count/len(action_indices)*100:.1f}%")
print(f"平均熵: {total_entropy/len(action_indices):.4f}")

print("\n" + "=" * 80)
print("为什么熵高？")
print("=" * 80)
avg_probs = np.mean(probs[action_indices], axis=0)
print(f"\n平均概率分布: {avg_probs}")
prob_str = ", ".join([f"E{i}:{p:.3f}" for i, p in enumerate(avg_probs)])
print(f"即: [{prob_str}]")
print(f"\n这个分布非常平坦！虽然Expert 0略高，但差距很小。")
print(f"Expert 0 ({avg_probs[0]:.3f}) vs Expert 1 ({avg_probs[1]:.3f}) 只差 {avg_probs[0]-avg_probs[1]:.3f}")

print("\n" + "=" * 80)
print("对比：理想的低熵情况")
print("=" * 80)
ideal_probs = np.array([0.9, 0.05, 0.03, 0.02])
ideal_entropy = -np.sum(ideal_probs * np.log(ideal_probs + 1e-10)) / np.log(4)
print(f"理想分布: [0.900, 0.050, 0.030, 0.020]")
print(f"理想熵: {ideal_entropy:.4f}  ← 这才是低熵！")
print(f"\n你的平均分布: [{', '.join([f'{p:.3f}' for p in avg_probs])}]")
print(f"你的熵: {total_entropy/len(action_indices):.4f}  ← 接近1.0，几乎随机！")
