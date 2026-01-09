#!/usr/bin/env python3
"""检查模型参数命名规则"""
import sys
sys.path.insert(0, '.')

# 模拟加载模型
from models import DiT_models
from config_loader import load_config
import torch

config = load_config('/home/syr/code/prediction_with_action/configs/metaworld_4d.yaml')
model = DiT_models['DiT-XL/2'](config)

print("=== 检查参数命名规则 ===\n")

# 找到专家相关的参数
expert_params = []
for name, param in model.named_parameters():
    if 'experts' in name:
        expert_params.append(name)
        if len(expert_params) <= 20:
            print(name)

print(f"\n总共找到 {len(expert_params)} 个专家相关参数")

# 分析命名模式
print("\n=== 专家参数命名模式分析 ===")
patterns = {}
for name in expert_params:
    # 提取模式，如 blocks.N.mlp.experts.M.fc1.weight
    parts = name.split('.')
    if 'experts' in parts:
        expert_idx = parts[parts.index('experts') + 1]
        if expert_idx.isdigit():
            # 提取expert编号前面的部分
            prefix = '.'.join(parts[:parts.index('experts') + 2])
            if prefix not in patterns:
                patterns[prefix] = []
            patterns[prefix].append(name)

for prefix in sorted(patterns.keys())[:5]:
    print(f"\n{prefix}:")
    for name in patterns[prefix][:3]:
        print(f"  {name}")
